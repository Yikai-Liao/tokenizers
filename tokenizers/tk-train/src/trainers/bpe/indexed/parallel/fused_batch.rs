//! Flat, non-AA batches: read and route once, join, then write disjoint spans.
//! Jobs cover consecutive sections of (rule rank, ordered posting) so each
//! birth key's unique producer also supplies ordered positions across jobs.
use super::weight_lookup::WeightLookup;

mod aggregate;
pub(super) mod block;
use super::*;

// Compile-time ablation switch for benchmarks; normal builds prefetch. There
// is no environment lookup in the posting loop.
const PREFETCH: bool = match option_env!("TK_POSTING_PREFETCH") {
    Some(s) => s.as_bytes().len() == 1 && s.as_bytes()[0] == b'1',
    None => true,
};
const FUSED_DECODE: bool = match option_env!("TK_POSTING_FUSED_DECODE") {
    Some(s) => s.as_bytes().len() == 1 && s.as_bytes()[0] == b'1',
    None => true,
};
const DECODE_BATCH: usize = 128;
const PREFETCH_DISTANCE: usize = 16;
#[inline]
fn prefetch_position<C>(corpus: &[C], position: usize) {
    #[cfg(target_arch = "x86_64")]
    if position < corpus.len() {
        // SAFETY: the checked position is inside the live corpus allocation.
        // Prefetch does not read a token or affect the atomic rewrite protocol.
        unsafe {
            std::arch::x86_64::_mm_prefetch(
                corpus.as_ptr().add(position).cast::<i8>(),
                std::arch::x86_64::_MM_HINT_T0,
            );
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = (corpus, position);
}

const EMPTY: u64 = u64::MAX;
const MULTIPLE: u64 = u64::MAX - 1;
struct Selected {
    heads: Vec<u64>,
    tails: Vec<u64>,
    multiple: AHashMap<u64, u32>,
}
impl Selected {
    fn new(rules: &[Rule], identities: usize) -> Self {
        let mut heads = vec![EMPTY; identities];
        let mut tails = vec![EMPTY; identities];
        let mut multiple = AHashMap::new();
        for r in rules {
            let head = &mut heads[r.edge.0 as usize];
            *head = if *head == EMPTY {
                key(r.edge.1, r.replacement)
            } else {
                MULTIPLE
            };
            let tail = &mut tails[r.edge.1 as usize];
            *tail = if *tail == EMPTY {
                key(r.edge.0, r.replacement)
            } else {
                MULTIPLE
            };
        }
        for r in rules {
            if heads[r.edge.0 as usize] == MULTIPLE || tails[r.edge.1 as usize] == MULTIPLE {
                multiple.insert(key(r.edge.0, r.edge.1), r.replacement);
            }
        }
        Self {
            heads,
            tails,
            multiple,
        }
    }
    fn bytes(&self) -> usize {
        (self.heads.capacity() + self.tails.capacity()) * 8
            + table_bytes(self.multiple.capacity(), 16)
    }
    fn left_selected<C: Slot>(&self, corpus: &[C], before: usize, prior: u32) -> bool {
        let tail = self.tails[prior as usize];
        if tail == EMPTY {
            return false;
        }
        let previous = corpus[before - 1].token();
        if tail == MULTIPLE {
            self.multiple.contains_key(&key(previous, prior))
        } else {
            (tail >> 32) as u32 == previous
        }
    }
    fn final_next<C: Slot>(&self, corpus: &[C], after: usize, next: u32, lengths: &[usize]) -> u32 {
        let head = self.heads[next as usize];
        if head == EMPTY {
            return next;
        }
        let following = corpus[after + lengths[next as usize]].token();
        if head == MULTIPLE {
            self.multiple
                .get(&key(next, following))
                .copied()
                .unwrap_or(next)
        } else if (head >> 32) as u32 == following {
            head as u32
        } else {
            next
        }
    }
}

struct Task<'a> {
    rank: usize,
    posting: &'a BlockPosting,
    begin: usize,
    end: usize,
}
struct Valid {
    rank: usize,
    positions: address_scratch::Positions,
}
pub(super) struct Prepared<O: Offset, const INLINE: usize> {
    valid: Vec<Vec<Valid>>,
    pub(super) outputs: Vec<Output<O, INLINE>>,
    pub(super) valid_bytes: usize,
    pub(super) selected_bytes: usize,
    pub(super) aggregate_bytes: usize,
}

pub(super) fn prepare_with_grouping<C: Slot, O: Offset, const INLINE: usize>(
    corpus: &[C],
    rules: &[Rule],
    postings: &[BlockPosting],
    blocks: &[Block<O, INLINE>],
    lengths: &[usize],
    uniform: Option<u64>,
    weight_lookup: Option<&weight_lookup::WeightLookups>,
    max_length: usize,
    workers: usize,
    grouped: bool,
    bits: u8,
) -> Result<Prepared<O, INLINE>> {
    // Cap per-job directory storage; this does not narrow canonical IDs.
    if grouped && lengths.len() <= 65_536 {
        prepare_with_mode::<C, O, INLINE, true>(
            corpus,
            rules,
            postings,
            blocks,
            lengths,
            uniform,
            weight_lookup,
            max_length,
            workers,
            bits,
        )
    } else {
        prepare_with_mode::<C, O, INLINE, false>(
            corpus,
            rules,
            postings,
            blocks,
            lengths,
            uniform,
            weight_lookup,
            max_length,
            workers,
            bits,
        )
    }
}

fn prepare_with_mode<C: Slot, O: Offset, const INLINE: usize, const GROUPED: bool>(
    corpus: &[C],
    rules: &[Rule],
    postings: &[BlockPosting],
    blocks: &[Block<O, INLINE>],
    lengths: &[usize],
    uniform: Option<u64>,
    weight_lookup: Option<&weight_lookup::WeightLookups>,
    max_length: usize,
    workers: usize,
    bits: u8,
) -> Result<Prepared<O, INLINE>> {
    debug_assert!(C::SHARED);
    debug_assert!(rules.iter().all(|r| r.edge.0 != r.edge.1));
    let selected = Selected::new(rules, lengths.len());
    let total: usize = postings.iter().map(BlockPosting::len).sum();
    let chunk = total.div_ceil(workers).clamp(1, 1 << 26);
    // Each job owns one route set, even when a batch contains many tiny rules.
    let mut jobs: Vec<Vec<Task<'_>>> = Vec::new();
    let mut visited = 0;
    for (rank, posting) in postings.iter().enumerate() {
        let mut begin = 0;
        while begin < posting.len() {
            let job = visited / chunk;
            if jobs.len() == job {
                jobs.push(Vec::new());
            }
            let take = (chunk - visited % chunk).min(posting.len() - begin);
            jobs[job].push(Task {
                rank,
                posting,
                begin,
                end: begin + take,
            });
            visited += take;
            begin += take;
        }
    }
    let prepared: Vec<_> = jobs
        .par_iter()
        .map(|tasks| -> Result<_> {
            let mut output = Output::new(workers, true);
            if GROUPED {
                output.enable_dense_births(rules.len());
            }
            let mut left_cache = aggregate::Scratch::new(if GROUPED { lengths.len() } else { 0 });
            let mut right_cache = aggregate::Scratch::new(if GROUPED { lengths.len() } else { 0 });
            let mut valid = Vec::with_capacity(tasks.len());
            let mut decoded = [0usize; DECODE_BATCH];
            for task in tasks {
                let rule = &rules[task.rank];
                let mut positions = address_scratch::Positions::default();
                let mut weight_cursor = weight_lookup::Cursor::default();
                macro_rules! consume {
                    ($position:expr) => {{
                        let p = $position;
                        let right = p + rule.left_len;
                        if corpus[p].token() != rule.edge.0
                            || right >= corpus.len()
                            || corpus[right].token() != rule.edge.1
                        {
                            continue;
                        }
                        positions.push(p);
                        let after = right + rule.right_len;
                        let weight = if let Some(weight) = uniform {
                            weight
                        } else if let Some(lookup) = weight_lookup {
                            lookup.weight(p, blocks, bits, &mut weight_cursor)
                        } else {
                            weight_cursor.weight(p, blocks, bits)
                        };
                        let prior = corpus[p - 1].token();
                        if prior != NONE {
                            let before = p - lengths[prior as usize];
                            // If the left neighbor is itself selected, its right delta
                            // accounts for this boundary using the final two outputs.
                            let left_selected = selected.left_selected(corpus, before, prior);
                            if !left_selected {
                                if GROUPED {
                                    left_cache.remove(prior, weight);
                                } else {
                                    output.remove(key(prior, rule.edge.0), weight);
                                }
                                if lengths[prior as usize] + rule.length() < max_length {
                                    if GROUPED {
                                        left_cache.birth(
                                            &mut output,
                                            prior,
                                            key(prior, rule.replacement),
                                            before,
                                            weight,
                                        )?;
                                    } else {
                                        output.birth(
                                            key(prior, rule.replacement),
                                            before,
                                            weight,
                                            32,
                                        )?;
                                    }
                                }
                            }
                        }
                        let next = corpus[after].token();
                        if next != NONE {
                            if GROUPED {
                                right_cache.remove(next, weight);
                            } else {
                                output.remove(key(rule.edge.1, next), weight);
                            }
                            let final_next = selected.final_next(corpus, after, next, lengths);
                            if rule.length() + lengths[final_next as usize] < max_length {
                                if GROUPED {
                                    right_cache.birth(
                                        &mut output,
                                        final_next,
                                        key(rule.replacement, final_next),
                                        p,
                                        weight,
                                    )?;
                                } else {
                                    output.birth(
                                        key(rule.replacement, final_next),
                                        p,
                                        weight,
                                        32,
                                    )?;
                                }
                            }
                        }
                    }};
                }
                let mut decoder = task.posting.decoder(bits, task.begin, task.end);
                if FUSED_DECODE {
                    // Interleave decoding with consumption; retain the same
                    // 16-position prefetch distance as the batched variant.
                    let mut ring = [0usize; PREFETCH_DISTANCE];
                    let mut active = decoder.fill(&mut ring);
                    if PREFETCH {
                        for &p in &ring[..active] {
                            prefetch_position(corpus, p);
                        }
                    }
                    let mut index = 0;
                    while active != 0 {
                        let p = ring[index];
                        if let Some(next) = decoder.next() {
                            ring[index] = next;
                            if PREFETCH {
                                prefetch_position(corpus, next);
                            }
                        } else {
                            active -= 1;
                        }
                        index = (index + 1) % PREFETCH_DISTANCE;
                        consume!(p);
                    }
                } else {
                    loop {
                        let count = decoder.fill(&mut decoded);
                        if count == 0 {
                            break;
                        }
                        if PREFETCH {
                            for &p in &decoded[..count.min(PREFETCH_DISTANCE)] {
                                prefetch_position(corpus, p);
                            }
                        }
                        for (i, &p) in decoded[..count].iter().enumerate() {
                            if PREFETCH && i + PREFETCH_DISTANCE < count {
                                prefetch_position(corpus, decoded[i + PREFETCH_DISTANCE]);
                            }
                            consume!(p);
                        }
                    }
                }
                if GROUPED {
                    left_cache.flush_dense(&mut output, rule, true, task.rank)?;
                    right_cache.flush_dense(&mut output, rule, false, task.rank)?;
                }
                valid.push(Valid {
                    rank: task.rank,
                    positions,
                });
            }
            Ok((valid, output, left_cache.bytes() + right_cache.bytes()))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut valid = Vec::with_capacity(prepared.len());
    let mut outputs = Vec::with_capacity(prepared.len());
    let mut valid_bytes = 0;
    let mut aggregate_bytes = 0;
    for (v, output, bytes) in prepared {
        aggregate_bytes += bytes;
        valid_bytes += v.iter().map(|v| v.positions.bytes()).sum::<usize>();
        valid.push(v);
        outputs.push(output);
    }
    Ok(Prepared {
        valid,
        outputs,
        valid_bytes,
        selected_bytes: selected.bytes(),
        aggregate_bytes,
    })
}

impl<O: Offset, const INLINE: usize> Prepared<O, INLINE> {
    pub(super) fn apply<C: Slot>(&self, corpus: &[C], rules: &[Rule]) {
        // All prepare reads have joined. Non-AA certificate gives disjoint
        // merged spans. Atomic stores permit mutation through shared references;
        // this par_iter joins before commit or the next batch can read corpus.
        debug_assert!(C::SHARED);
        self.valid.par_iter().for_each(|job| {
            for valid in job {
                let rule = &rules[valid.rank];
                valid.positions.for_each(|p| {
                    let right = p + rule.left_len;
                    corpus[p].set_shared(rule.replacement);
                    if rule.right_len == 1 {
                        corpus[right].set_shared(rule.replacement);
                    } else {
                        corpus[right].set_shared(NONE);
                        corpus[right + rule.right_len - 1].set_shared(rule.replacement);
                    }
                });
            }
        });
    }
}
