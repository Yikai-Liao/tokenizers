//! Read disjoint posting slices, route births by their own block, then write.
//! Task order is source block, rule rank, offset. A birth key has one producer,
//! so reverse chains concatenate into ordered local postings without sorting.
use super::*;
use std::sync::Arc;

struct Task<O: Offset, const INLINE: usize> {
    block: usize,
    rank: usize,
    posting: Arc<PackedPosting<O, INLINE>>,
    begin: usize,
    end: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_trace<C: Slot, O: Offset, const INLINE: usize>(
        trainer: &BpeTrainer,
        words: &AHashMap<CompactString, u64>,
        bits: u8,
        workers: usize,
    ) -> IndexedTraining {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        let session = posting_arena::Session::new(&pool, None);
        let mut ids = AHashMap::new();
        let mut strings = Vec::new();
        trainer.add_special_tokens(&mut ids, &mut strings);
        trainer.compute_alphabet(words, &mut ids, &mut strings);
        let mut got = pool
            .install(|| {
                train_in_pool::<C, O, INLINE>(
                    trainer,
                    words,
                    IndexedParallelConfig {
                        workers,
                        posting_block_bits: bits,
                        atomic_corpus: C::SHARED,
                        narrow_corpus: C::NARROW,
                        ..Default::default()
                    },
                    ids,
                    strings,
                    Instant::now(),
                    0.0,
                    0,
                    &pool,
                    None,
                    posting_arena::Policy::Auto,
                    SelectionMode::Bulk(4),
                    corpus::Order::WeightSorted,
                )
            })
            .unwrap();
        got.stats.posting_allocations = session.finish();
        assert_eq!(
            got.stats.posting_allocations.heap_requested_bytes,
            got.stats.posting_allocations.heap_freed_bytes
        );
        assert_eq!(
            got.stats.posting_allocations.arena_requested_bytes,
            got.stats.posting_allocations.arena_retired_bytes
        );
        got
    }

    #[test]
    fn block_fused_full_traces_cover_adjacent_rules_aa_weights_reserved_and_limits() {
        let words: AHashMap<CompactString, u64> = [
            ("xabcdabefghi中".repeat(80), 7),
            ("cdabghabef文".repeat(70), 3),
            ("aaaaabbbbcdd".repeat(60), 1),
            ("zeroab中".repeat(80), 0),
        ]
        .into_iter()
        .map(|(w, n)| (w.into(), n))
        .collect();
        for limit in [None, Some(3), Some(17)] {
            let trainer = BpeTrainer::builder()
                .vocab_size(70)
                .min_frequency(2)
                .show_progress(false)
                .max_token_length(limit)
                .special_tokens(vec![AddedToken::from("abcd", true)])
                .build();
            let mut trace = Vec::new();
            let (vocab, merges, _) = trainer
                .do_train_observed(&words, |p, n, id| trace.push((p, n, id)))
                .unwrap();
            for workers in [1, 4] {
                for bits in [4, 16] {
                    let a = full_trace::<AtomicU16, u16, 4>(&trainer, &words, bits, workers);
                    let b = full_trace::<AtomicU32, u32, 2>(&trainer, &words, bits, workers);
                    let old = full_trace::<u32, u32, 2>(&trainer, &words, bits, workers);
                    for got in [&a, &b, &old] {
                        assert_eq!(got.trace, trace);
                        assert_eq!(got.vocab, vocab);
                        assert_eq!(got.merges, merges);
                    }
                    assert!(a.stats.fused_block_batches > 0);
                    assert!(b.stats.fused_block_batches > 0);
                    assert_eq!(old.stats.fused_block_batches, 0);
                }
            }
        }
    }

    #[test]
    fn word_aligned_u16_blocks_preserve_full_training_trace() {
        let words: AHashMap<CompactString, u64> = [
            ("a".repeat(60_000).into(), 3),
            ("ab".repeat(5_000).into(), 2),
            ("b".repeat(1_000).into(), 1),
        ]
        .into_iter()
        .collect();
        let trainer = BpeTrainer::builder()
            .vocab_size(8)
            .min_frequency(1)
            .show_progress(false)
            .build();
        let mut reference_trace = Vec::new();
        let (vocab, merges, _) = trainer
            .do_train_observed(&words, |pair, frequency, id| {
                reference_trace.push((pair, frequency, id));
            })
            .unwrap();
        for workers in [1, 4] {
            let flat = full_trace::<u32, u32, 2>(&trainer, &words, 32, workers);
            let local = full_trace::<AtomicU16, u16, 4>(&trainer, &words, 16, workers);
            assert_eq!(local.trace, reference_trace);
            assert_eq!(local.vocab, vocab);
            assert_eq!(local.merges, merges);
            assert_eq!(local.trace, flat.trace);
            assert_eq!(local.vocab, flat.vocab);
            assert_eq!(local.merges, flat.merges);
            assert!(local.stats.corpus_padding_slots > 0);
            assert!(local.stats.fused_block_batches > 0);
        }
    }

    fn fixture<C: Slot, O: Offset, const INLINE: usize>(wide: bool) {
        let bits = 3;
        let (l, a, b, r) = if wide {
            (65_536, 65_537, 65_538, 65_539)
        } else {
            (0, 1, 2, 3)
        };
        let x = r + 1;
        let y = x + 1;
        let mut lengths = vec![0; y as usize + 1];
        lengths[l as usize] = 25;
        for id in [a, b, r] {
            lengths[id as usize] = 1;
        }
        lengths[x as usize] = 2;
        lengths[y as usize] = 2;
        // First L spans blocks 0..3; AB is in block 3. Its left birth
        // belongs to block 0, not source_block-1. Second AB crosses block 4,
        // and its neighboring CD rule starts in block 5.
        let mut raw = vec![NONE; 51];
        raw[1] = l;
        raw[25] = l;
        raw[26] = a;
        raw[27] = b;
        raw[28] = r;
        raw[39] = a;
        raw[40] = b;
        raw[41] = l;
        raw[42] = r;
        // In the second case L is a short neighbor; use a separate ID.
        let c = y + 1;
        lengths.resize(c as usize + 1, 1);
        raw[41] = c;
        let rules = [
            Rule {
                edge: (a, b),
                replacement: x,
                left_len: 1,
                right_len: 1,
            },
            Rule {
                edge: (c, r),
                replacement: y,
                left_len: 1,
                right_len: 1,
            },
        ];
        let corpus: Vec<C> = raw.into_iter().map(C::encode).collect();
        let mut blocks: Vec<Block<O, INLINE>> = (0..7)
            .map(|i| Block::new(i << bits, if i < 4 { 0 } else { 7 }))
            .collect();
        for (pos, k) in [(26, key(a, b)), (39, key(a, b)), (41, key(c, r))] {
            blocks[pos >> bits]
                .postings
                .entry(k)
                .or_default()
                .push(O::encode(pos & 7))
                .unwrap();
        }
        let block_rules = [(3, vec![0]), (4, vec![0]), (5, vec![1])]
            .into_iter()
            .collect();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let session = posting_arena::Session::new(&pool, None);
        let mut prepared = pool
            .install(|| {
                prepare(
                    &corpus,
                    &rules,
                    &mut blocks,
                    &block_rules,
                    &lengths,
                    None,
                    usize::MAX,
                    bits,
                    4,
                )
            })
            .unwrap();
        assert!(
            prepared
                .jobs
                .iter()
                .flat_map(|j| &j.fragments)
                .any(|f| f.block == 0 && f.key == key(l, x) && f.weight == 0)
        );
        assert!(
            prepared
                .jobs
                .iter()
                .flat_map(|j| &j.fragments)
                .any(|f| f.block == 4 && f.key == key(x, y))
        );
        pool.install(|| prepared.apply(&corpus, &rules, bits));
        assert_eq!(corpus[26].token(), x);
        assert_eq!(corpus[39].token(), x);
        assert_eq!(corpus[41].token(), y);
        let directory = pool
            .install(|| prepared.install(&mut blocks, |_| true, &AHashMap::new(), 4))
            .unwrap();
        assert!(
            directory[0]
                .iter()
                .flatten()
                .any(|&(k, b)| k == key(l, x) && b == 0)
        );
        assert_eq!(
            blocks[0].postings[&key(l, x)]
                .as_slice()
                .iter()
                .map(|p| p.index())
                .collect::<Vec<_>>(),
            [1]
        );
        assert_eq!(
            blocks[4].postings[&key(x, y)]
                .as_slice()
                .iter()
                .map(|p| p.index())
                .collect::<Vec<_>>(),
            [7]
        );
        // Posting owners allocated during this session must die before finish.
        prepared.outputs.clear();
        drop(prepared);
        drop(blocks);
        let counters = session.finish();
        assert_eq!(counters.heap_requested_bytes, counters.heap_freed_bytes);
    }

    #[test]
    fn block_fragments_route_long_left_tokens_adjacent_crossings_and_zero_weights() {
        fixture::<AtomicU16, u16, 4>(false);
        fixture::<AtomicU32, u32, 2>(true);
    }
}
struct Valid {
    block: usize,
    rank: usize,
    begin: usize,
    end: usize,
}
struct Fragment {
    block: usize,
    key: u64,
    head: u32,
    count: u32,
    weight: u64,
}
struct Job {
    valid: Vec<Valid>,
    offsets: Vec<u32>,
    nodes: Vec<Node>,
    fragments: Vec<Fragment>,
}
struct Neighbor {
    id: u32,
    removed: u64,
    born: u64,
    last: u32,
}
struct Scratch {
    indices: Vec<u32>,
    sparse: AHashMap<u32, u32>,
    groups: Vec<Neighbor>,
}
impl Scratch {
    fn new(identities: usize) -> Self {
        Self {
            indices: if identities <= 65_536 {
                vec![NONE; identities]
            } else {
                Vec::new()
            },
            sparse: AHashMap::new(),
            groups: Vec::new(),
        }
    }
    fn group(&mut self, id: u32) -> &mut Neighbor {
        let index = if self.indices.is_empty() {
            self.sparse.entry(id).or_insert(NONE)
        } else {
            &mut self.indices[id as usize]
        };
        if *index == NONE {
            *index = self.groups.len() as u32;
            self.groups.push(Neighbor {
                id,
                removed: 0,
                born: 0,
                last: NONE,
            });
        }
        &mut self.groups[*index as usize]
    }
    fn remove(&mut self, id: u32, weight: u64) {
        self.group(id).removed += weight;
    }
    fn birth(
        &mut self,
        id: u32,
        k: u64,
        p: usize,
        weight: u64,
        bits: u8,
        job: &mut Job,
    ) -> Result<()> {
        let target = p >> bits;
        let offset =
            u32::try_from(p - (target << bits)).map_err(|_| "block birth offset exceeds u32")?;
        let head = u32::try_from(job.nodes.len()).map_err(|_| "block birth chain exceeds u32")?;
        if head == NONE {
            return Err("block birth chain sentinel collision".into());
        }
        let group = self.group(id);
        if group.last == NONE || job.fragments[group.last as usize].block != target {
            let fragment = u32::try_from(job.fragments.len())
                .map_err(|_| "block fragment count exceeds u32")?;
            if fragment == NONE {
                return Err("block fragment sentinel collision".into());
            }
            job.fragments.push(Fragment {
                block: target,
                key: k,
                head: NONE,
                count: 0,
                weight: 0,
            });
            group.last = fragment;
        }
        let fragment = &mut job.fragments[group.last as usize];
        debug_assert_eq!(fragment.key, k);
        job.nodes.push(Node {
            position: offset,
            next: fragment.head,
        });
        fragment.head = head;
        fragment.count = fragment
            .count
            .checked_add(1)
            .ok_or("block fragment posting exceeds u32")?;
        fragment.weight += weight;
        group.born += weight;
        Ok(())
    }
    fn flush<O: Offset, const INLINE: usize>(
        &mut self,
        output: &mut Output<O, INLINE>,
        rule: &Rule,
        left: bool,
    ) {
        for group in self.groups.drain(..) {
            if !self.indices.is_empty() {
                self.indices[group.id as usize] = NONE;
            }
            if group.removed != 0 {
                output.remove(
                    if left {
                        key(group.id, rule.edge.0)
                    } else {
                        key(rule.edge.1, group.id)
                    },
                    group.removed,
                );
            }
            if group.last != NONE {
                let k = if left {
                    key(group.id, rule.replacement)
                } else {
                    key(rule.replacement, group.id)
                };
                let o = owner(k, output.born.len());
                // Include zero-weight births: another fragment can make this
                // key eligible, and every physical start must then be installed.
                *output.born[o].entry(k).or_default() += group.born;
            }
        }
        self.sparse.clear();
    }
    fn bytes(&self) -> usize {
        self.indices.capacity() * 4
            + self.groups.capacity() * std::mem::size_of::<Neighbor>()
            + table_bytes(self.sparse.capacity(), 8)
    }
}

pub(in super::super) struct Prepared<O: Offset, const INLINE: usize> {
    jobs: Vec<Job>,
    pub(in super::super) outputs: Vec<Output<O, INLINE>>,
    pub(in super::super) valid_bytes: usize,
    pub(in super::super) node_bytes: usize,
    pub(in super::super) fragment_bytes: usize,
    pub(in super::super) task_bytes: usize,
    pub(in super::super) selected_bytes: usize,
    pub(in super::super) scratch_bound_bytes: usize,
}

pub(in super::super) fn prepare<C: Slot, O: Offset, const INLINE: usize>(
    corpus: &[C],
    rules: &[Rule],
    blocks: &mut [Block<O, INLINE>],
    block_rules: &AHashMap<usize, Vec<usize>>,
    lengths: &[usize],
    uniform: Option<u64>,
    max_length: usize,
    bits: u8,
    workers: usize,
) -> Result<Prepared<O, INLINE>> {
    debug_assert!(C::SHARED && rules.iter().all(|r| r.edge.0 != r.edge.1));
    let selected = Selected::new(rules, lengths.len());
    let source: Vec<Vec<_>> = blocks
        .par_iter_mut()
        .enumerate()
        .map(|(b, block)| {
            block_rules
                .get(&b)
                .into_iter()
                .flatten()
                .map(|&rank| {
                    let r = &rules[rank];
                    (
                        b,
                        rank,
                        Arc::new(block.postings.remove(&key(r.edge.0, r.edge.1)).unwrap()),
                    )
                })
                .collect()
        })
        .collect();
    let total: usize = source.iter().flatten().map(|(_, _, p)| p.len()).sum();
    // Bound each job's u32 chain indices, including both birth directions.
    // Scratch directories exist only in executing jobs, not in collected results.
    let chunk = total.div_ceil(workers).clamp(4096, 1 << 20);
    let mut tasks = Vec::<Vec<Task<O, INLINE>>>::new();
    let mut visited = 0;
    for (block, rank, posting) in source.into_iter().flatten() {
        let mut begin = 0;
        while begin < posting.len() {
            let job = visited / chunk;
            if tasks.len() == job {
                tasks.push(Vec::new());
            }
            let take = (chunk - visited % chunk).min(posting.len() - begin);
            tasks[job].push(Task {
                block,
                rank,
                posting: Arc::clone(&posting),
                begin,
                end: begin + take,
            });
            visited += take;
            begin += take;
        }
        // Tasks are the only remaining shared posting owners. Each consumed
        // task drops its Arc, freeing the source after its last slice finishes.
    }
    let task_bytes = tasks
        .iter()
        .map(|j| j.capacity() * std::mem::size_of::<Task<O, INLINE>>())
        .sum();
    let results: Vec<_> = tasks
        .into_par_iter()
        .map(|tasks| -> Result<_> {
            let mut job = Job {
                valid: Vec::with_capacity(tasks.len()),
                offsets: Vec::new(),
                nodes: Vec::new(),
                fragments: Vec::new(),
            };
            let mut output = Output::new(workers, false);
            let mut left = Scratch::new(lengths.len());
            let mut right_cache = Scratch::new(lengths.len());
            for task in tasks {
                let rule = &rules[task.rank];
                let block = &blocks[task.block];
                let mut valid = Valid {
                    block: task.block,
                    rank: task.rank,
                    begin: job.offsets.len(),
                    end: job.offsets.len(),
                };
                let mut cursor = 0;
                for &offset in &task.posting.as_slice()[task.begin..task.end] {
                    let p = block.base + offset.index();
                    let right = p + rule.left_len;
                    if corpus[p].token() != rule.edge.0
                        || right >= corpus.len()
                        || corpus[right].token() != rule.edge.1
                    {
                        continue;
                    }
                    job.offsets.push(
                        u32::try_from(offset.index()).map_err(|_| "valid offset exceeds u32")?,
                    );
                    let after = right + rule.right_len;
                    let weight = block.weight_forward(p, uniform, &mut cursor);
                    let prior = corpus[p - 1].token();
                    if prior != NONE {
                        let before = p - lengths[prior as usize];
                        if !selected.left_selected(corpus, before, prior) {
                            left.remove(prior, weight);
                            if lengths[prior as usize] + rule.length() < max_length {
                                left.birth(
                                    prior,
                                    key(prior, rule.replacement),
                                    before,
                                    weight,
                                    bits,
                                    &mut job,
                                )?;
                            }
                        }
                    }
                    let next = corpus[after].token();
                    if next != NONE {
                        right_cache.remove(next, weight);
                        let final_next = selected.final_next(corpus, after, next, lengths);
                        if rule.length() + lengths[final_next as usize] < max_length {
                            right_cache.birth(
                                final_next,
                                key(rule.replacement, final_next),
                                p,
                                weight,
                                bits,
                                &mut job,
                            )?;
                        }
                    }
                }
                left.flush(&mut output, rule, true);
                right_cache.flush(&mut output, rule, false);
                valid.end = job.offsets.len();
                job.valid.push(valid);
            }
            Ok((job, output, left.bytes() + right_cache.bytes()))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut scratch: Vec<_> = results.iter().map(|(_, _, bytes)| *bytes).collect();
    scratch.sort_unstable_by(|a, b| b.cmp(a));
    // This sum is a bound on simultaneously executing scratch, not a sampled peak.
    let scratch_bound_bytes = scratch.into_iter().take(workers).sum();
    let mut prepared = Prepared {
        jobs: Vec::with_capacity(results.len()),
        outputs: Vec::with_capacity(results.len()),
        valid_bytes: 0,
        node_bytes: 0,
        fragment_bytes: 0,
        task_bytes,
        selected_bytes: selected.bytes(),
        scratch_bound_bytes,
    };
    for (job, output, _) in results {
        prepared.valid_bytes += job.offsets.capacity() * 4;
        prepared.node_bytes += job.nodes.capacity() * std::mem::size_of::<Node>();
        prepared.fragment_bytes += job.fragments.capacity() * std::mem::size_of::<Fragment>();
        prepared.jobs.push(job);
        prepared.outputs.push(output);
    }
    Ok(prepared)
}

impl<O: Offset, const INLINE: usize> Prepared<O, INLINE> {
    pub(in super::super) fn apply<C: Slot>(&self, corpus: &[C], rules: &[Rule], bits: u8) {
        debug_assert!(C::SHARED);
        self.jobs.par_iter().for_each(|job| {
            for valid in &job.valid {
                let rule = &rules[valid.rank];
                for &offset in &job.offsets[valid.begin..valid.end] {
                    let p = (valid.block << bits) + offset as usize;
                    let right = p + rule.left_len;
                    corpus[p].set_shared(rule.replacement);
                    if rule.right_len == 1 {
                        corpus[right].set_shared(rule.replacement);
                    } else {
                        corpus[right].set_shared(NONE);
                        corpus[right + rule.right_len - 1].set_shared(rule.replacement);
                    }
                }
            }
        });
    }
    pub(in super::super) fn install<F: Fn(u64) -> bool + Sync>(
        &self,
        blocks: &mut [Block<O, INLINE>],
        accepted: F,
        retired: &AHashMap<usize, Vec<u64>>,
        workers: usize,
    ) -> Result<Vec<Vec<Vec<(u64, u32)>>>> {
        let mut routed = vec![Vec::new(); blocks.len()];
        for (job, data) in self.jobs.iter().enumerate() {
            for (fragment, f) in data.fragments.iter().enumerate() {
                if accepted(f.key) {
                    routed[f.block].push((job, fragment));
                }
            }
        }
        blocks
            .par_iter_mut()
            .enumerate()
            .map(|(b, block)| -> Result<_> {
                if let Some(keys) = retired.get(&b) {
                    for k in keys {
                        block.postings.remove(k);
                    }
                }
                let mut groups = AHashMap::<u64, u32>::new();
                for &(j, f) in &routed[b] {
                    let fragment = &self.jobs[j].fragments[f];
                    let entry = groups.entry(fragment.key).or_default();
                    *entry = entry
                        .checked_add(fragment.count)
                        .ok_or("block birth posting exceeds u32")?;
                }
                let mut directory: Vec<Vec<(u64, u32)>> =
                    (0..workers).map(|_| Vec::new()).collect();
                for (k, count) in groups {
                    debug_assert!(!block.postings.contains_key(&k));
                    block
                        .postings
                        .insert(k, PackedPosting::<O, INLINE>::with_capacity(count)?);
                    directory[owner(k, workers)].push((
                        k,
                        u32::try_from(b).map_err(|_| "posting directory exceeds u32")?,
                    ));
                }
                for &(j, f) in &routed[b] {
                    let job = &self.jobs[j];
                    let fragment = &job.fragments[f];
                    let posting = block.postings.get_mut(&fragment.key).unwrap();
                    let mut head = fragment.head;
                    posting.append_reversed_reserved(fragment.count, || {
                        let node = &job.nodes[head as usize];
                        head = node.next;
                        O::encode(node.position as usize)
                    })?;
                    debug_assert_eq!(head, NONE);
                }
                debug_assert!(directory.iter().flatten().all(|(k, _)| {
                    block.postings[k]
                        .as_slice()
                        .windows(2)
                        .all(|p| p[0].index() < p[1].index())
                }));
                Ok(directory)
            })
            .collect()
    }
}
