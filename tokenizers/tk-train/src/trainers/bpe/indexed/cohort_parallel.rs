//! General HF cohorts partitioned at immutable word boundaries. Each job owns
//! its corpus range and performs the original left-to-right rewrites locally;
//! aliases and occurrence-specific spans do not cross word separators.
use super::*;
use rayon::prelude::*;

#[derive(Clone, Copy)]
pub(super) struct Options {
    pub sort_weights: bool,
    pub narrow_corpus: bool,
    pub weight_lookup: bool,
    pub initial_grouped: bool,
    pub parallel_apply: bool,
    pub grouped_tail: bool,
    pub scratch_cache: bool,
    pub character_cache: bool,
    pub packed_queue: bool,
    pub parallel_measure: bool,
    pub guarded_fast: bool,
    pub parallel_alphabet: bool,
    pub arena_allocator: bool,
    pub batch_execution: bool,
    pub queue_prefetch: bool,
    pub fused_batch: bool,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            sort_weights: true,
            narrow_corpus: true,
            // Ordered posting walks can use the block's forward weight cursor.
            // Keep the lookup available for explicit mixed-order controls.
            weight_lookup: false,
            initial_grouped: true,
            parallel_apply: true,
            grouped_tail: true,
            scratch_cache: true,
            character_cache: true,
            packed_queue: true,
            parallel_measure: true,
            guarded_fast: true,
            parallel_alphabet: true,
            arena_allocator: true,
            batch_execution: true,
            queue_prefetch: true,
            fused_batch: true,
        }
    }
}
#[cfg(test)]
impl Options {
    fn all_enabled() -> Self {
        Self {
            weight_lookup: true,
            ..Self::default()
        }
    }
    fn without(index: usize) -> Self {
        let mut options = Self::all_enabled();
        let flags = [
            &mut options.sort_weights,
            &mut options.narrow_corpus,
            &mut options.weight_lookup,
            &mut options.initial_grouped,
            &mut options.parallel_apply,
            &mut options.grouped_tail,
            &mut options.scratch_cache,
            &mut options.character_cache,
            &mut options.packed_queue,
            &mut options.parallel_measure,
            &mut options.guarded_fast,
            &mut options.parallel_alphabet,
            &mut options.arena_allocator,
            &mut options.batch_execution,
            &mut options.queue_prefetch,
            &mut options.fused_batch,
        ];
        for (i, flag) in flags.into_iter().enumerate() {
            if index == i || index == 16 {
                *flag = false;
            }
        }
        options
    }
}
pub(super) struct Execution<'a> {
    pub(super) pool: Option<&'a rayon::ThreadPool>,
    pub(super) initialization_pool: Option<&'a rayon::ThreadPool>,
    pub(super) options: Options,
    pub(super) policy: posting_arena::Policy,
}
impl Execution<'_> {
    pub(super) fn configure(&self, edges: usize) {
        let cutoff = self.policy.cutoff(edges);
        if let Some(pool) = self.pool {
            posting_arena::configure(pool, self.initialization_pool, cutoff);
        } else {
            posting_arena::configure_local(cutoff);
        }
    }
    pub(super) fn workers(&self) -> usize {
        self.pool.map_or(1, rayon::ThreadPool::current_num_threads)
    }
    pub(super) fn initialization_pool(&self) -> Option<&rayon::ThreadPool> {
        self.initialization_pool.or(self.pool)
    }
}

struct Node {
    position: u32,
    next: u32,
}
struct Group {
    neighbor: u32,
    removed: i64,
    born: i64,
    head: u32,
    count: u32,
}
struct Scratch {
    indices: Vec<u32>,
    sparse: AHashMap<u32, u32>,
    groups: Vec<Group>,
    nodes: Vec<Node>,
}
impl Scratch {
    fn reset(&mut self, identities: usize) {
        if !self.indices.is_empty() {
            for group in &self.groups {
                self.indices[group.neighbor as usize] = NONE;
            }
        }
        self.sparse.clear();
        self.groups.clear();
        self.nodes.clear();
        if identities <= 65_536 {
            self.indices.resize(identities, NONE);
        } else {
            self.indices = Vec::new();
        }
    }
    fn new(identities: usize) -> Self {
        Self {
            indices: if identities <= 65_536 {
                vec![NONE; identities]
            } else {
                Vec::new()
            },
            sparse: AHashMap::new(),
            groups: Vec::new(),
            nodes: Vec::new(),
        }
    }
    fn group(&mut self, neighbor: u32) -> &mut Group {
        let index = if self.indices.is_empty() {
            self.sparse.entry(neighbor).or_insert(NONE)
        } else {
            &mut self.indices[neighbor as usize]
        };
        if *index == NONE {
            *index = self.groups.len() as u32;
            self.groups.push(Group {
                neighbor,
                removed: 0,
                born: 0,
                head: NONE,
                count: 0,
            });
        }
        &mut self.groups[*index as usize]
    }
    fn remove(&mut self, neighbor: u32, weight: i64) -> Result<()> {
        let group = self.group(neighbor);
        group.removed = group
            .removed
            .checked_add(weight)
            .ok_or("cohort removal exceeds i64")?;
        Ok(())
    }
    fn birth(&mut self, neighbor: u32, position: usize, weight: i64) -> Result<()> {
        let head = u32::try_from(self.nodes.len()).map_err(|_| "cohort chain exceeds u32")?;
        if head == NONE {
            return Err("cohort chain sentinel collision".into());
        }
        let group = self.group(neighbor);
        let next = group.head;
        group.born = group
            .born
            .checked_add(weight)
            .ok_or("cohort birth weight exceeds i64")?;
        group.count = group
            .count
            .checked_add(1)
            .ok_or("cohort birth count exceeds u32")?;
        group.head = head;
        self.nodes.push(Node {
            position: u32::try_from(position).map_err(|_| "cohort position exceeds u32")?,
            next,
        });
        Ok(())
    }
    fn get(&self, neighbor: u32) -> Option<&Group> {
        let index = if self.indices.is_empty() {
            self.sparse.get(&neighbor).copied().unwrap_or(NONE)
        } else {
            self.indices[neighbor as usize]
        };
        (index != NONE).then(|| &self.groups[index as usize])
    }
    fn append(&self, group: &Group, positions: &mut SmallPosting) -> Result<()> {
        let mut head = group.head;
        positions.append_reversed_reserved(group.count, || {
            let node = &self.nodes[head as usize];
            head = node.next;
            node.position
        })?;
        debug_assert_eq!(head, NONE);
        Ok(())
    }
    fn bytes(&self) -> usize {
        self.indices.capacity() * 4
            + self.groups.capacity() * std::mem::size_of::<Group>()
            + self.nodes.capacity() * 8
            + parallel::table_bytes(self.sparse.capacity(), 8)
    }
}
pub(super) struct Workspace {
    left: Scratch,
    right: Scratch,
}
impl Workspace {
    fn new(identities: usize) -> Self {
        Self {
            left: Scratch::new(identities),
            right: Scratch::new(identities),
        }
    }
    fn reset(&mut self, identities: usize) {
        self.left.reset(identities);
        self.right.reset(identities);
    }
    fn bytes(&self) -> usize {
        self.left.bytes() + self.right.bytes()
    }
}
enum Starts<'a> {
    Positions(&'a [u32]),
    Words(&'a [u32]),
}
struct Job<'a, C: parallel::Slot> {
    base: usize,
    corpus: &'a mut [C],
    spans: Option<&'a mut [u32]>,
    starts: Starts<'a>,
}
struct Worker<'a, C: parallel::Slot> {
    base: usize,
    corpus: &'a mut [C],
    spans: Option<&'a mut [u32]>,
    lengths: &'a [u32],
    left: Scratch,
    right: Scratch,
    stale: usize,
    scanned: usize,
}
impl<C: parallel::Slot> Worker<'_, C> {
    fn span(&self, pos: usize) -> u32 {
        self.spans.as_ref().map_or_else(
            || self.lengths[self.corpus[pos].token() as usize],
            |spans| spans[pos],
        )
    }
    fn inspect(&self, pos: usize, pair: Pair) -> Option<Context> {
        if self.corpus[pos].token() != pair.0 {
            return None;
        }
        let left = self.span(pos);
        let right = pos + left as usize;
        if left == 0 || right >= self.corpus.len() || self.corpus[right].token() != pair.1 {
            return None;
        }
        let right_len = self.span(right);
        let after = right + right_len as usize;
        if right_len == 0 || after >= self.corpus.len() {
            return None;
        }
        // A job starts at a word pivot. Its omitted previous slot is a separator.
        let before = (pos != 0 && self.corpus[pos - 1].token() != NONE)
            .then(|| pos - self.span(pos - 1) as usize);
        Some(Context {
            pos,
            right,
            after,
            before,
            len: left + right_len,
        })
    }
    fn merge_at(
        &mut self,
        context: Context,
        replacement: u32,
        limit: usize,
        weight: i64,
    ) -> Result<usize> {
        let Context {
            pos,
            right,
            after,
            before,
            len,
        } = context;
        if let Some(before) = before {
            let prior = self.corpus[before].token();
            self.left.remove(prior, weight)?;
            if self.span(before) as usize + (len as usize) < limit {
                self.left.birth(prior, self.base + before, weight)?;
            }
        }
        let next = self.corpus[after].token();
        if next != NONE {
            self.right.remove(next, weight)?;
            if len as usize + (self.span(after) as usize) < limit {
                self.right.birth(next, self.base + pos, weight)?;
            }
        }
        self.corpus[pos].set(replacement);
        self.corpus[right].set(if right + 1 == after {
            replacement
        } else {
            NONE
        });
        self.corpus[after - 1].set(replacement);
        if let Some(spans) = &mut self.spans {
            if right + 1 != after {
                spans[right] = 0;
            }
            spans[pos] = len;
            spans[after - 1] = len;
        }
        Ok(after)
    }
}

pub(super) fn apply<C: parallel::Slot>(
    index: &mut Index<C>,
    top: &Candidate,
    replacement: u32,
    scan_words: bool,
    limit: usize,
    pool: Option<&rayon::ThreadPool>,
    stats: &mut IndexedTrainingStats,
) -> Result<()> {
    let workers = pool.map_or(1, rayon::ThreadPool::current_num_threads);
    let postings = top.positions.as_slice();
    let mut words = Vec::new();
    let mut cuts = vec![(0_usize, 0_usize)];
    if scan_words {
        let collect = || {
            postings
                .par_iter()
                .map(|&p| index.word_at(p) as u32)
                .collect()
        };
        words = if let Some(pool) = pool {
            pool.install(collect)
        } else {
            postings.iter().map(|&p| index.word_at(p) as u32).collect()
        };
        if let Some(pool) = pool {
            pool.install(|| words.par_sort_unstable());
        } else {
            words.sort_unstable();
        }
        words.dedup();
        let chunk = words.len().div_ceil(workers).max(1);
        for start in (chunk..words.len()).step_by(chunk) {
            cuts.push((start, index.pivots[words[start] as usize] as usize));
        }
        stats.cohort_scan_activations += 1;
        stats.cohort_words_scanned += words.len();
        stats.cohort_scan_max_words = stats.cohort_scan_max_words.max(words.len());
    } else {
        // Before the first active identity reuse, a birth has a unique side
        // producer. Initial postings and job concatenation are spatially ordered.
        debug_assert!(postings.windows(2).all(|w| w[0] < w[1]));
        let chunk = postings.len().div_ceil(workers).max(1);
        for desired in (chunk..postings.len()).step_by(chunk) {
            let word = index.word_at(postings[desired]);
            let pivot = index.pivots[word];
            let start = postings.partition_point(|&p| p < pivot);
            if start > cuts.last().unwrap().0 {
                cuts.push((start, pivot as usize));
            }
        }
    }
    let total = if scan_words {
        words.len()
    } else {
        postings.len()
    };
    cuts.push((total, index.corpus.len()));
    let pivots = &index.pivots;
    let weight_pivots = index.weight_pivots.as_deref().unwrap_or(&index.pivots);
    let weights = &index.weights;
    let lookup = index.weight_lookup.as_ref();
    let weight_at = |p: u32| {
        lookup.map_or_else(
            || {
                let i = weight_pivots.partition_point(|&q| q <= p);
                if i == 0 { 0 } else { weights[i - 1] }
            },
            |l| l.weight_parts(weight_pivots, weights, 0, p),
        )
    };
    let lengths = &index.lengths;
    let mut rest = index.corpus.as_mut_slice();
    let mut span_rest = index.spans.as_deref_mut();
    let mut jobs = Vec::with_capacity(cuts.len() - 1);
    for cut in cuts.windows(2) {
        let (start, base) = cut[0];
        let (end, after) = cut[1];
        let (corpus, next) = rest.split_at_mut(after - base);
        rest = next;
        let spans = if let Some(all) = span_rest.take() {
            let (part, next) = all.split_at_mut(after - base);
            span_rest = Some(next);
            Some(part)
        } else {
            None
        };
        jobs.push(Job {
            base,
            corpus,
            spans,
            starts: if scan_words {
                Starts::Words(&words[start..end])
            } else {
                Starts::Positions(&postings[start..end])
            },
        });
    }
    let begin = Instant::now();
    let mut workspace = if pool.is_none() {
        let mut workspace = index
            .cohort_workspace
            .take()
            .unwrap_or_else(|| Workspace::new(lengths.len()));
        workspace.reset(lengths.len());
        Some(workspace)
    } else {
        None
    };
    let execute = |job: Job<'_, C>, workspace: Workspace| -> Result<_> {
        let mut worker = Worker {
            base: job.base,
            corpus: job.corpus,
            spans: job.spans,
            lengths,
            left: workspace.left,
            right: workspace.right,
            stale: 0,
            scanned: 0,
        };
        match job.starts {
            Starts::Words(words) => {
                for &word in words {
                    let mut pos = pivots[word as usize] as usize - worker.base;
                    let weight = weight_at(pivots[word as usize]) as i64;
                    while worker.corpus[pos].token() != NONE {
                        worker.scanned += 1;
                        pos = if let Some(context) = worker.inspect(pos, top.pair) {
                            worker.merge_at(context, replacement, limit, weight)?
                        } else {
                            pos + worker.span(pos) as usize
                        };
                    }
                }
            }
            Starts::Positions(positions) => {
                for &position in positions {
                    let pos = position as usize - worker.base;
                    if let Some(context) = worker.inspect(pos, top.pair) {
                        worker.merge_at(context, replacement, limit, weight_at(position) as i64)?;
                    } else {
                        worker.stale += 1;
                    }
                }
            }
        }
        Ok((worker.left, worker.right, worker.stale, worker.scanned))
    };
    let mut results: Vec<_> = if let Some(pool) = pool {
        pool.install(|| {
            jobs.into_par_iter()
                .map(|job| execute(job, Workspace::new(lengths.len())))
                .collect::<Result<Vec<_>>>()
        })?
    } else {
        vec![execute(jobs.pop().unwrap(), workspace.take().unwrap())?]
    };
    let rewrite_ms = begin.elapsed().as_secs_f64() * 1000.0;
    if pool.is_some() {
        stats.cohort_parallel_rounds += 1;
        stats.cohort_parallel_jobs += results.len();
        stats.fused_prepare_ms += rewrite_ms;
    } else {
        stats.cohort_serial_rounds += 1;
        stats.cohort_serial_postings += postings.len();
        stats.cohort_serial_ms += rewrite_ms;
    }
    let begin = Instant::now();
    let mut born = AHashMap::<Pair, usize>::new();
    stats.peak_prepare_aggregate_bytes = stats.peak_prepare_aggregate_bytes.max(
        results
            .iter()
            .map(|(left, right, _, _)| left.bytes() + right.bytes())
            .sum::<usize>()
            + index.cohort_workspace.as_ref().map_or(0, Workspace::bytes),
    );
    for (left, right, stale, scanned) in &results {
        stats.stale_posting_visits += stale;
        stats.word_scan_steps += scanned;
        if pool.is_some() {
            stats.cohort_parallel_delta_groups += left.groups.len() + right.groups.len();
        } else {
            stats.cohort_serial_delta_groups += left.groups.len() + right.groups.len();
        }
        for (scratch, is_left) in [(left, true), (right, false)] {
            for group in &scratch.groups {
                let old = if is_left {
                    (group.neighbor, top.pair.0)
                } else {
                    (top.pair.1, group.neighbor)
                };
                use std::collections::hash_map::Entry;
                let old_count = match index.counts.entry(old) {
                    Entry::Occupied(entry) => Some(entry.into_mut()),
                    Entry::Vacant(_) if index.monotone_floor.is_some() => None,
                    Entry::Vacant(entry) => Some(entry.insert(0)),
                };
                if let Some(count) = old_count {
                    *count = count
                        .checked_sub(group.removed)
                        .ok_or("indexed BPE pair ledger exceeds i64 range")?;
                }
                if group.count == 0 {
                    continue;
                }
                let pair = if is_left {
                    (group.neighbor, replacement)
                } else {
                    (replacement, group.neighbor)
                };
                let count = index.counts.entry(pair).or_default();
                *count = count
                    .checked_add(group.born)
                    .ok_or("indexed BPE pair ledger exceeds i64 range")?;
                *born.entry(pair).or_default() += group.count as usize;
            }
        }
    }
    for (pair, count) in born {
        // The general HF queue adds a newborn cohort only at positive ledger
        // frequency. A fully cancelled intermediate key needs no final buffer.
        if index.counts[&pair] <= 0 {
            continue;
        }
        let mut positions = SmallPosting::with_capacity(
            u32::try_from(count).map_err(|_| "cohort birth posting exceeds u32")?,
        )?;
        for (left, right, _, _) in &results {
            if pair.0 == replacement {
                if let Some(group) = right.get(pair.1) {
                    right.append(group, &mut positions)?;
                }
            }
            if pair.1 == replacement {
                if let Some(group) = left.get(pair.0) {
                    left.append(group, &mut positions)?;
                }
            }
        }
        // All intermediate births remain, including retired starts. If an active
        // ID aliases the replacement, both directions may feed this same key;
        // scan_words preserves its historical word set independent of node order.
        index.births.insert(pair, positions);
    }
    stats.commit_ms += begin.elapsed().as_secs_f64() * 1000.0;
    if pool.is_none() {
        let (left, right, _, _) = results.pop().unwrap();
        let workspace = Workspace { left, right };
        // Cache only bounded tail scratch; a large serial cohort must not pin
        // its node capacity for the rest of training.
        if index.options.scratch_cache && workspace.bytes() <= 4 * 1024 * 1024 {
            index.cohort_workspace = Some(workspace);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_released(counters: &posting_arena::Counters) {
        assert_eq!(counters.heap_requested_bytes, counters.heap_freed_bytes);
        assert_eq!(counters.arena_requested_bytes, counters.arena_retired_bytes);
    }

    fn guarded_oracle(
        trainer: &BpeTrainer,
        words: &AHashMap<CompactString, u64>,
        config: IndexedParallelConfig,
        options: Options,
    ) -> IndexedTraining {
        let mut trace = Vec::new();
        let (vocab, merges, _) = trainer
            .do_train_observed(words, |pair, count, id| trace.push((pair, count, id)))
            .unwrap();
        let got = trainer
            .train_affixed_options(words, config, options)
            .unwrap();
        assert_eq!(got.trace, trace);
        assert_eq!(got.vocab, vocab);
        assert_eq!(got.merges, merges);
        assert_released(&got.stats.posting_allocations);
        assert_released(&got.stats.speculative_posting_allocations);
        got
    }

    #[test]
    fn guarded_all_migrations_preserve_hf_with_and_without_active_aliases() {
        let words = (0..72)
            .map(|i| {
                (
                    format!("{i}{}baaba中文", "aaaabcd".repeat(6)).into(),
                    [0, 1, 7][i % 3],
                )
            })
            .collect();
        for (prefix, suffix, limit) in [
            (Some("##"), None, None),
            (None, Some("</w>"), Some(9)),
            (None, Some("a"), None),
            (Some("ab"), None, Some(9)),
        ] {
            let mut trainer = BpeTrainer::builder()
                .vocab_size(96)
                .min_frequency(2)
                .show_progress(false)
                .max_token_length(limit)
                .build();
            trainer.continuing_subword_prefix = prefix.map(str::to_owned);
            trainer.end_of_word_suffix = suffix.map(str::to_owned);
            for disabled in 0..=18 {
                let options = if disabled == 18 {
                    Options::default()
                } else if disabled == 17 {
                    Options::all_enabled()
                } else {
                    Options::without(disabled)
                };
                let got = guarded_oracle(
                    &trainer,
                    &words,
                    IndexedParallelConfig {
                        workers: 4,
                        initialization_workers: Some(2),
                        narrow_corpus: true,
                        ..Default::default()
                    },
                    options,
                );
                if options.guarded_fast && !got.stats.alias_fallback {
                    assert!(got.stats.alias_guarded);
                    if !options.batch_execution {
                        assert!(got.stats.max_batch_rules <= 1);
                    }
                    if !options.fused_batch {
                        assert_eq!(got.stats.fused_batches, 0);
                    }
                    if !options.queue_prefetch {
                        assert_eq!(got.stats.queue_prefetched, 0);
                    }
                }
                if !options.arena_allocator {
                    assert_eq!(got.stats.posting_allocations.arena_buffers, 0);
                    assert_eq!(got.stats.speculative_posting_allocations.arena_buffers, 0);
                }
            }
        }
    }

    #[test]
    fn guarded_alias_reconstruction_covers_first_and_late_collision() {
        let trainer = BpeTrainer::builder()
            .vocab_size(48)
            .min_frequency(1)
            .show_progress(false)
            .end_of_word_suffix("a".into())
            .build();
        for late in [false, true] {
            let mut words: AHashMap<CompactString, u64> =
                [("baaba".into(), 1)].into_iter().collect();
            if late {
                words.insert("xyxyxy".into(), 100);
            }
            for workers in [1, 4] {
                let got = guarded_oracle(
                    &trainer,
                    &words,
                    IndexedParallelConfig {
                        workers,
                        narrow_corpus: true,
                        ..Default::default()
                    },
                    Options::default(),
                );
                assert!(got.stats.alias_guarded && got.stats.alias_fallback);
                if late {
                    assert!(got.stats.speculative_applied_merges > 0);
                } else {
                    assert_eq!(got.stats.speculative_applied_merges, 0);
                }
                assert!(got.stats.reused_ids > 0);
                assert!(!got.stats.monotone_pairs);
            }
        }
    }

    #[test]
    fn guarded_decorated_slots_cover_widths_layouts_reserved_and_length_gates() {
        let words: AHashMap<CompactString, u64> = [
            ("xabcdab中abab".repeat(2100), 7),
            ("abababaaaa中文".repeat(80), 11),
            ("zeroaaaa🙂".repeat(80), 0),
            (String::new(), 1),
        ]
        .into_iter()
        .map(|(w, n)| (w.into(), n))
        .collect();
        let trainer = BpeTrainer::builder()
            .vocab_size(80)
            .min_frequency(2)
            .show_progress(false)
            .max_token_length(Some(7))
            .continuing_subword_prefix("##".into())
            .end_of_word_suffix("</w>".into())
            .special_tokens(vec![AddedToken::from("##ab", true)])
            .build();
        for workers in [1, 4] {
            for bits in [16, 32] {
                for narrow in [false, true] {
                    for atomic in [false, true] {
                        let got = guarded_oracle(
                            &trainer,
                            &words,
                            IndexedParallelConfig {
                                workers,
                                posting_block_bits: bits,
                                narrow_corpus: narrow,
                                atomic_corpus: atomic,
                                ..Default::default()
                            },
                            Options::default(),
                        );
                        assert!(!got.stats.alias_fallback);
                        assert!(got.stats.reused_ids > 0);
                        assert_eq!(got.stats.corpus_slot_bytes, if narrow { 2 } else { 4 });
                    }
                }
            }
        }
    }

    #[test]
    fn guarded_wide_frequencies_match_the_full_i64_cohort_ledger() {
        // HF's original oracle uses i32 counts. Exercise the extended checked
        // range against its cohort adaptation instead of wrapped HF arithmetic.
        let words = [
            ("ababaa中".into(), u32::MAX as u64 + 1),
            ("ab中aba".into(), 3),
        ]
        .into_iter()
        .collect();
        let trainer = BpeTrainer::builder()
            .vocab_size(40)
            .min_frequency(2)
            .show_progress(false)
            .end_of_word_suffix("</w>".into())
            .build();
        let config = IndexedParallelConfig {
            workers: 4,
            narrow_corpus: true,
            ..Default::default()
        };
        let expected = trainer.train_cohorts_parallel(&words, config).unwrap();
        let got = trainer
            .train_affixed_options(&words, config, Options::default())
            .unwrap();
        assert_eq!(got.trace, expected.trace);
        assert_eq!(got.vocab, expected.vocab);
        assert_eq!(got.merges, expected.merges);
        assert!(!got.stats.alias_fallback);
        assert_released(&got.stats.posting_allocations);
    }

    #[test]
    fn migration_switches_preserve_full_hf_trace_and_model() {
        let words: AHashMap<CompactString, u64> = (0..96)
            .map(|i| {
                (
                    format!("{i}{}baaba中文", "aaaaabcd".repeat(4)).into(),
                    [0, 1, 7][i % 3],
                )
            })
            .collect();
        for (prefix, suffix, limit) in [(None, Some("a"), None), (Some("ab"), None, Some(9))] {
            let mut trainer = BpeTrainer::builder()
                .vocab_size(90)
                .min_frequency(2)
                .show_progress(false)
                .max_token_length(limit)
                .build();
            trainer.continuing_subword_prefix = prefix.map(str::to_owned);
            trainer.end_of_word_suffix = suffix.map(str::to_owned);
            let mut trace = Vec::new();
            let (vocab, merges, _) = trainer
                .do_train_observed(&words, |pair, count, id| trace.push((pair, count, id)))
                .unwrap();
            for disabled in 0..=16 {
                let options = Options::without(disabled);
                let got = trainer
                    .train_cohorts_parallel_options(
                        &words,
                        IndexedParallelConfig {
                            workers: 4,
                            ..Default::default()
                        },
                        options,
                    )
                    .unwrap();
                assert_eq!(got.trace, trace, "disabled {disabled}");
                assert_eq!(got.vocab, vocab, "disabled {disabled}");
                assert_eq!(got.merges, merges, "disabled {disabled}");
                assert_eq!(
                    got.stats.corpus_slot_bytes,
                    if options.narrow_corpus { 2 } else { 4 }
                );
                assert_eq!(
                    got.stats.posting_allocations.heap_requested_bytes,
                    got.stats.posting_allocations.heap_freed_bytes
                );
                if options.narrow_corpus {
                    assert_eq!(got.stats.initial_corpus_bytes, got.stats.initial_slots * 2);
                }
                if options.sort_weights {
                    assert!(got.stats.weight_interval_count <= 3);
                }
            }
        }
    }
    #[test]
    fn narrow_corpus_reserves_its_sentinel_and_wide_counts_keep_wide_priorities() {
        let words: AHashMap<CompactString, u64> = [("abab".into(), 1)].into_iter().collect();
        for (vocab_size, slot_bytes) in [(65_535, 2), (65_536, 4)] {
            let trainer = BpeTrainer::builder()
                .vocab_size(vocab_size)
                .min_frequency(1)
                .show_progress(false)
                .end_of_word_suffix("</w>".into())
                .build();
            let got = trainer.do_train_indexed(&words).unwrap();
            assert_eq!(got.stats.corpus_slot_bytes, slot_bytes);
        }
        let words = [("abab".into(), u32::MAX as u64 + 1)].into_iter().collect();
        let trainer = BpeTrainer::builder()
            .vocab_size(20)
            .min_frequency(1)
            .show_progress(false)
            .end_of_word_suffix("</w>".into())
            .build();
        let got = trainer.do_train_indexed(&words).unwrap();
        assert_eq!(got.stats.corpus_slot_bytes, 2);
        assert_eq!(got.stats.queue_selection_mode, "cohort_wide");
    }

    #[test]
    fn cached_groups_reset_across_dense_and_sparse_identity_domains() {
        let mut scratch = Scratch::new(16);
        scratch.remove(3, 9).unwrap();
        scratch.birth(3, 40, 0).unwrap();
        scratch.birth(3, 50, 7).unwrap();
        let mut positions = SmallPosting::with_capacity(2).unwrap();
        scratch
            .append(scratch.get(3).unwrap(), &mut positions)
            .unwrap();
        assert_eq!(positions.as_slice(), &[40, 50]);
        scratch.reset(65_536);
        assert!(scratch.get(3).is_none());
        scratch.birth(65_535, 60, 0).unwrap();
        scratch.reset(65_537);
        assert!(scratch.indices.is_empty());
        assert!(scratch.get(65_535).is_none());
        scratch.birth(65_536, 70, 11).unwrap();
        scratch.reset(65_538);
        assert!(scratch.get(65_536).is_none());
        scratch.remove(65_537, 13).unwrap();
        assert_eq!(scratch.get(65_537).unwrap().removed, 13);
    }

    #[test]
    fn serial_grouped_tail_preserves_hf_cohorts_and_releases_postings() {
        let words: AHashMap<CompactString, u64> = (0..192)
            .map(|i| {
                (
                    format!("{i}{}baaba", "aaaaabcd".repeat(8)).into(),
                    [0, 1, 7][i % 3],
                )
            })
            .collect();
        for (prefix, suffix, limit) in [(None, Some("a"), None), (Some("ab"), None, Some(9))] {
            let mut trainer = BpeTrainer::builder()
                .vocab_size(100)
                .min_frequency(2)
                .show_progress(false)
                .max_token_length(limit)
                .build();
            trainer.continuing_subword_prefix = prefix.map(str::to_owned);
            trainer.end_of_word_suffix = suffix.map(str::to_owned);
            let mut trace = Vec::new();
            let (vocab, merges, _) = trainer
                .do_train_observed(&words, |pair, count, id| trace.push((pair, count, id)))
                .unwrap();
            let got = trainer.do_train_indexed(&words).unwrap();
            assert_eq!(got.trace, trace);
            assert_eq!(got.vocab, vocab);
            assert_eq!(got.merges, merges);
            assert!(got.stats.cohort_serial_rounds > 0);
            assert_eq!(
                got.stats.posting_allocations.heap_requested_bytes,
                got.stats.posting_allocations.heap_freed_bytes
            );
            if prefix.is_none() {
                assert!(got.stats.reused_ids > 0);
                assert!(got.stats.cohort_scan_activations > 0);
            }
        }
    }

    #[test]
    fn word_partitioning_preserves_hf_aliases_lengths_aa_and_zero_weights() {
        let words: AHashMap<CompactString, u64> = (0..768)
            .map(|i| {
                (
                    format!("{i}{}baaba", "aaaaaaaaabcd".repeat(16)).into(),
                    [0, 1, 7, 13][i % 4],
                )
            })
            .collect();
        for (prefix, suffix, limit) in [
            (None, Some("a"), None),
            (Some("ab"), None, Some(9)),
            (Some("##"), Some("</w>"), Some(3)),
        ] {
            let mut trainer = BpeTrainer::builder()
                .vocab_size(80)
                .min_frequency(2)
                .show_progress(false)
                .max_token_length(limit)
                .build();
            trainer.continuing_subword_prefix = prefix.map(str::to_owned);
            trainer.end_of_word_suffix = suffix.map(str::to_owned);
            let mut trace = Vec::new();
            let (vocab, merges, _) = trainer
                .do_train_observed(&words, |pair, count, id| trace.push((pair, count, id)))
                .unwrap();
            for initialization_workers in [Some(1), Some(4)] {
                let got = trainer
                    .train_cohorts_parallel(
                        &words,
                        IndexedParallelConfig {
                            workers: 4,
                            initialization_workers,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                assert_eq!(got.trace, trace);
                assert_eq!(got.vocab, vocab);
                assert_eq!(got.merges, merges);
                assert!(got.stats.cohort_parallel_rounds > 0);
                assert_eq!(
                    got.stats.posting_allocations.heap_requested_bytes,
                    got.stats.posting_allocations.heap_freed_bytes
                );
                if prefix.is_none() {
                    assert!(got.stats.reused_ids > 0);
                    assert!(got.stats.cohort_scan_activations > 0);
                }
            }
        }
    }

    #[test]
    fn wide_initial_ids_keep_general_hash_counts_and_parallel_cohorts() {
        let words: AHashMap<CompactString, u64> = (0..128)
            .map(|i| (format!("{i}{}", "abcdbaaba".repeat(80)).into(), 1))
            .collect();
        let special = (0..65_536)
            .map(|i| AddedToken::from(format!("reserved{i}"), true))
            .collect();
        let trainer = BpeTrainer::builder()
            .vocab_size(65_620)
            .min_frequency(2)
            .show_progress(false)
            .end_of_word_suffix("a".into())
            .special_tokens(special)
            .build();
        let expected = trainer.do_train_indexed(&words).unwrap();
        let got = trainer
            .train_cohorts_parallel(&words, IndexedParallelConfig::default())
            .unwrap();
        assert_eq!(got.trace, expected.trace);
        assert_eq!(got.vocab, expected.vocab);
        assert_eq!(got.merges, expected.merges);
        assert_eq!(got.stats.initial_count_backend, "cohort_grouped_hash");
        assert_eq!(got.stats.initialization_workers, 1);
        assert!(got.stats.cohort_parallel_rounds > 0);
    }

    #[test]
    fn parallel_cohorts_keep_full_u64_weights_beyond_the_hf_reference_i32_domain() {
        let words: AHashMap<CompactString, u64> = (0..128)
            .map(|i| {
                (
                    format!("{i}{}", "baabaabcd".repeat(80)).into(),
                    [0, 1, u32::MAX as u64 + 3][i % 3],
                )
            })
            .collect();
        let trainer = BpeTrainer::builder()
            .vocab_size(80)
            .min_frequency(2)
            .show_progress(false)
            .continuing_subword_prefix("##".into())
            .build();
        // The legacy HF oracle casts word weights to i32. The existing general
        // index is the lossless wide-weight oracle outside that legacy domain.
        let expected = trainer.do_train_indexed(&words).unwrap();
        let got = trainer
            .train_cohorts_parallel(&words, IndexedParallelConfig::default())
            .unwrap();
        assert_eq!(got.trace, expected.trace);
        assert_eq!(got.vocab, expected.vocab);
        assert_eq!(got.merges, expected.merges);
        assert!(got.stats.cohort_parallel_rounds > 0);
    }
}
