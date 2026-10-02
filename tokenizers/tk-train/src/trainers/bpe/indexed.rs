//! Sequential HF adaptation of Efficient BPE's occurrence index and local rewrites.
//!
//! HF heap entries own historical cohorts of words. Its counts deliberately omit
//! the selected edge's removal. Affix aliases can make the ledger and cohorts
//! observable, so the general path retains them and scans cohort words as needed.
//! Ordinary character BPE has a stronger invariant: every replacement identity
//! appears in the corpus for the first time, even when its string already has a
//! reserved vocabulary ID. Old pairs then only lose edges. The guarded path uses
//! that proof to retire low-frequency pairs, including finite length limits;
//! nonempty affixes use it until a runtime check finds active ID reuse, then
//! reconstruct the general ledger from the original input. See
//! benchmarks/hf-bpe/PAIR_MONOTONICITY.md for the proof and its scope.
use super::*;
use small_posting::SmallPosting;
use std::time::Instant;

mod aa_parity;
mod cohort_parallel;
mod cohort_queue;
mod compact;
mod parallel;
mod posting_arena;
mod small_posting;

const NONE: u32 = u32::MAX;

/// Measurements for the experimental sequential occurrence-index trainer.
#[derive(Debug, Default, Clone, Serialize)]
pub(super) struct IndexedTrainingStats {
    pub alias_guarded: bool,
    pub alias_fallback: bool,
    pub corpus_slot_bytes: usize,
    pub speculative_selected_merges: usize,
    pub speculative_applied_merges: usize,
    pub speculative_initialize_ms: f64,
    pub speculative_merge_ms: f64,
    pub speculative_total_ms: f64,
    pub speculative_posting_allocations: posting_arena::Counters,
    pub posting_allocation_policy: &'static str,
    pub posting_arena_cutoff_bytes: usize,
    pub posting_allocations: posting_arena::Counters,
    pub initialize_ms: f64,
    pub merge_ms: f64,
    pub initial_symbols: usize,
    pub posting_visits: usize,
    pub stale_posting_visits: usize,
    pub word_scan_steps: usize,
    pub span_table_bytes: usize,
    pub cohort_scan_activations: usize,
    pub cohort_words_scanned: usize,
    pub cohort_scan_max_words: usize,
    pub cohort_parallel_rounds: usize,
    pub cohort_parallel_jobs: usize,
    pub cohort_parallel_delta_groups: usize,
    pub cohort_serial_rounds: usize,
    pub cohort_serial_postings: usize,
    pub cohort_serial_delta_groups: usize,
    pub cohort_serial_ms: f64,
    pub reused_ids: usize,
    pub monotone_pairs: bool,
    pub pruned_pairs: usize,
    pub layout: &'static str,
    pub corpus_bytes: usize,
    pub corpus_padding_slots: usize,
    pub posting_bytes: usize,
    pub peak_birth_bytes: usize,
    pub workers: usize,
    pub atomic_corpus: bool,
    pub initialization_workers: usize,
    pub batch_rounds: usize,
    pub max_batch_rules: usize,
    pub initial_corpus_bytes: usize,
    pub initial_posting_bytes: usize,
    pub initial_pair_table_bytes: usize,
    pub initial_block_table_bytes: usize,
    pub initial_directory_bytes: usize,
    pub initial_heap_bytes: usize,
    pub initial_slots: usize,
    pub initial_edges: usize,
    pub initial_pairs: usize,
    pub initial_blocks: usize,
    pub initial_block_pairs: usize,
    pub initial_slot_bytes: usize,
    pub initial_length_bytes: usize,
    pub initial_weight_bytes: usize,
    pub tokenize_ms: f64,
    pub alphabet_ms: f64,
    pub alphabet_scratch_bytes: usize,
    pub character_table_bytes: usize,
    pub corpus_measure_ms: f64,
    pub corpus_relayout_ms: f64,
    /// Sorting is included in corpus_measure_ms and initialize_ms.
    pub corpus_sort_ms: f64,
    pub corpus_sort_buffer_bytes: usize,
    pub corpus_stable_weight_sort: bool,
    pub corpus_weight_order: &'static str,
    pub corpus_word_reference_bytes: usize,
    pub corpus_temporary_weight_bytes: usize,
    pub weight_interval_count: usize,
    pub corpus_allocate_ms: f64,
    pub corpus_fill_ms: f64,
    pub initial_route_ms: f64,
    pub initial_count_ms: f64,
    pub initial_bounded_tiles: usize,
    pub initial_bounded_groups: usize,
    pub initial_bounded_hash_edges: usize,
    /// Sum of per-block sort maxima in the largest initialization wave.
    pub initial_bounded_sort_buffer_bound_bytes: usize,
    pub initial_summary_waves: usize,
    /// Sum of allocated summary Vec capacities across consumed waves.
    pub initial_summary_buffer_bytes: usize,
    /// Maximum simultaneous summary Vec backing (including nested Vec metadata).
    pub peak_initial_summary_buffer_bytes: usize,
    pub initial_count_backend: &'static str,
    pub initial_weight_lookup_ms: f64,
    pub initial_weight_lookup_bytes: usize,
    pub initial_route_compact_ms: f64,
    pub initial_radix_sort_ms: f64,
    pub initial_group_count_ms: f64,
    pub initial_posting_install_ms: f64,
    pub initial_route_buffer_bytes: usize,
    pub peak_initial_route_buffer_bytes: usize,
    pub initial_radix_scratch_bytes: usize,
    pub initial_group_buffer_bytes: usize,
    pub initial_heap_ms: f64,
    pub select_ms: f64,
    pub queue_selection_mode: &'static str,
    pub queue_owner_probes: usize,
    pub queue_leader_updates: usize,
    pub queue_truth_checks: usize,
    pub queue_stale_corrections: usize,
    pub queue_prefetched: usize,
    pub queue_unused_restored: usize,
    pub queue_serial_refills: usize,
    /// Sum of worker prefetch time, nested inside initialization/commit.
    pub queue_worker_prefetch_ms: f64,
    pub plan_ms: f64,
    pub fused_prepare_ms: f64,
    pub fused_batches: usize,
    pub fused_block_batches: usize,
    pub peak_block_birth_node_bytes: usize,
    pub peak_block_birth_fragment_bytes: usize,
    pub peak_block_task_bytes: usize,
    /// Sum of the largest concurrently executing job scratch capacities.
    pub peak_block_scratch_bound_bytes: usize,
    pub peak_valid_start_bytes: usize,
    pub weight_lookup_build_ms: f64,
    pub weight_lookup_bytes: usize,
    pub weight_one_bucket_count: usize,
    pub weight_bucket_count: usize,
    pub weight_one_bucket_bytes: usize,
    pub peak_selected_lookup_bytes: usize,
    pub peak_prepare_aggregate_bytes: usize,
    pub delta_ms: f64,
    pub rewrite_ms: f64,
    pub commit_ms: f64,
    pub route_ms: f64,
}

/// Explicit controls for the experimental parallel occurrence trainer.
#[derive(Debug, Clone, Copy)]
pub(super) struct IndexedParallelConfig {
    pub workers: usize,
    /// Optional initialization pool for controlled comparisons; merges keep workers.
    pub initialization_workers: Option<usize>,
    /// 16 uses u16 local posting offsets; 32 uses u32 offsets.
    pub posting_block_bits: u8,
    /// Use u16 corpus IDs when the complete possible ID domain fits.
    pub narrow_corpus: bool,
    /// Experimental Relaxed atomic slot accesses with otherwise identical work.
    pub atomic_corpus: bool,
    /// Maximum certified rules per batch, at most 256.
    pub batch_size: usize,
}

impl Default for IndexedParallelConfig {
    fn default() -> Self {
        Self {
            workers: 4,
            initialization_workers: None,
            posting_block_bits: 32,
            narrow_corpus: true,
            atomic_corpus: false,
            batch_size: 256,
        }
    }
}

/// Raw HF model parts and measurements; build a `PipelineBPE` using the trainer's affixes.
pub(super) struct IndexedTraining {
    pub vocab: Vocab,
    pub merges: Merges,
    pub special_tokens: Vec<AddedToken>,
    pub stats: IndexedTrainingStats,
    #[cfg(test)]
    trace: Vec<(Pair, u64, u32)>,
}

// Like Efficient BPE, live tokens occupy their original start/end positions in
// one u32 corpus. NONE separates words (HF's token ID zero is a valid token).
// With uniquely represented raw spans, a per-ID length table finds neighbours.
// Affix identity collisions need a position-wide span table: the same ID can have
// different physical lengths. Historical positions retain their word identity
// through the immutable pivots, without a word ID on every symbol.
struct Context {
    pos: usize,
    right: usize,
    after: usize,
    before: Option<usize>,
    len: u32,
}

struct Candidate {
    pair: Pair,
    count: u64,
    positions: SmallPosting,
}
impl Eq for Candidate {}
impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.pair == other.pair && self.count == other.count
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count
            .cmp(&other.count)
            .then_with(|| other.pair.cmp(&self.pair))
    }
}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct Index<C: parallel::Slot = u32> {
    corpus: Vec<C>,
    lengths: Vec<u32>,
    spans: Option<Vec<u32>>,
    pivots: Vec<u32>,
    weights: Vec<u64>,
    counts: AHashMap<Pair, i64>,
    queue: cohort_queue::Queue,
    births: AHashMap<Pair, SmallPosting>,
    cohort_workspace: Option<cohort_parallel::Workspace>,
    options: cohort_parallel::Options,
    weight_pivots: Option<Vec<u32>>,
    weight_lookup: Option<parallel::weight_lookup::WeightLookup>,
    corpus_sort_ms: f64,
    corpus_relayout_ms: f64,
    // Some(floor) certifies ordinary global character BPE, not fresh numeric IDs.
    monotone_floor: Option<u64>,
    pruned_pairs: usize,
    initial_edges: usize,
    tokenize_ms: f64,
    initial_count_ms: f64,
    character_table_bytes: usize,
}

struct PreparedCorpus<C: parallel::Slot = u32> {
    corpus: Vec<C>,
    lengths: Vec<u32>,
    pivots: Vec<u32>,
    weights: Vec<u64>,
    monotone_floor: Option<u64>,
    edges: usize,
    character_table_bytes: usize,
}

/// Initial decorations depend only on the retained character and its original
/// first/last flags. Cache resolved IDs, while still allocating unseen IDs in
/// the original word traversal order. Payload aliases remain handled by Index.
struct DecoratedCharacters {
    // A compact rank separates the variant cache from reserved vocabulary IDs.
    characters: Vec<(u32, u32)>,
    variants: Vec<[u32; 3]>,
}
impl DecoratedCharacters {
    fn new(ids: &AHashMap<CompactString, u32>) -> Self {
        let mut characters = vec![(NONE, NONE); 0x110000];
        let mut variants = Vec::new();
        for (token, &id) in ids {
            let mut chars = token.chars();
            if let Some(c) = chars.next()
                && chars.next().is_none()
            {
                characters[c as usize] = (id, variants.len() as u32);
                variants.push([NONE; 3]);
            }
        }
        Self {
            characters,
            variants,
        }
    }
    fn bytes(&self) -> usize {
        self.characters.capacity() * std::mem::size_of::<(u32, u32)>()
            + self.variants.capacity() * std::mem::size_of::<[u32; 3]>()
    }
}

pub(super) fn supports_compact(trainer: &BpeTrainer) -> bool {
    trainer
        .continuing_subword_prefix
        .as_deref()
        .is_none_or(str::is_empty)
        && trainer
            .end_of_word_suffix
            .as_deref()
            .is_none_or(str::is_empty)
}

impl PreparedCorpus {
    fn tokenize(
        trainer: &BpeTrainer,
        wc: &AHashMap<CompactString, u64>,
        w2id: &mut AHashMap<CompactString, u32>,
        id2w: &mut Vec<CompactString>,
        progress: &Option<ProgressBar>,
    ) -> Result<Self> {
        Self::tokenize_with_cache(trainer, wc, w2id, id2w, progress, true, None)
    }
    fn tokenize_with_cache(
        trainer: &BpeTrainer,
        wc: &AHashMap<CompactString, u64>,
        w2id: &mut AHashMap<CompactString, u32>,
        id2w: &mut Vec<CompactString>,
        progress: &Option<ProgressBar>,
        cache: bool,
        measure_pool: Option<&rayon::ThreadPool>,
    ) -> Result<Self> {
        // Count Unicode scalars, rather than reserving the UTF-8 byte bound,
        // so Chinese input does not reserve three times its corpus payload.
        let capacity = if let Some(pool) = measure_pool.filter(|p| p.current_num_threads() > 1) {
            use rayon::prelude::*;
            pool.install(|| {
                wc.par_iter()
                    .try_fold(
                        || 0_usize,
                        |size, (word, _)| size.checked_add(word.chars().count())?.checked_add(1),
                    )
                    .try_reduce(|| 0_usize, |a, b| a.checked_add(b))
            })
            .and_then(|size| size.checked_add(1))
        } else {
            wc.keys().try_fold(1_usize, |size, word| {
                size.checked_add(word.chars().count())?.checked_add(1)
            })
        }
        .ok_or("indexed BPE corpus size exceeds usize")?;
        if capacity >= NONE as usize {
            return Err("indexed BPE corpus including separators must fit u32 positions".into());
        }
        let mut corpus = Vec::with_capacity(capacity);
        corpus.push(NONE);
        let mut pivots = Vec::with_capacity(wc.len());
        let mut weights = Vec::with_capacity(wc.len());
        let mut edges = 0;
        // Zero means this identity has never been live, including long strings
        // reserved by special_tokens. A vocabulary entry alone is not activation.
        let mut lengths = vec![0; id2w.len()];
        let mut decorated = String::new();
        let mut character_cache =
            (cache && !supports_compact(trainer)).then(|| DecoratedCharacters::new(w2id));
        let character_table_bytes = character_cache
            .as_ref()
            .map_or(0, DecoratedCharacters::bytes);
        for (word, &weight) in wc {
            let start = corpus.len();
            pivots.push(corpus.len() as u32);
            weights.push(weight);
            for (byte, c) in word.char_indices() {
                let is_first = byte == 0;
                let is_last = byte + c.len_utf8() == word.len();
                let mut utf8 = [0_u8; 4];
                let plain: &str = c.encode_utf8(&mut utf8);
                // Use the original character's position for affixes, even if
                // alphabet filtering removed the first or last character.
                let (plain_id, rank) = if let Some(cache) = &character_cache {
                    let (id, rank) = cache.characters[c as usize];
                    if id == NONE {
                        continue;
                    }
                    (id, rank)
                } else if let Some(&id) = w2id.get(plain) {
                    (id, NONE)
                } else {
                    continue;
                };
                let prefix = (!is_first)
                    .then_some(trainer.continuing_subword_prefix.as_deref())
                    .flatten();
                let suffix = is_last
                    .then_some(trainer.end_of_word_suffix.as_deref())
                    .flatten();
                let flags = usize::from(prefix.is_some_and(|p| !p.is_empty()))
                    | (usize::from(suffix.is_some_and(|s| !s.is_empty())) << 1);
                let cached = character_cache.as_ref().and_then(|cache| {
                    (flags != 0).then(|| cache.variants[rank as usize][flags - 1])
                });
                if let Some(id) = cached.filter(|&id| id != NONE) {
                    lengths[id as usize] = 1;
                    corpus.push(id);
                    continue;
                }
                let key = if flags != 0 {
                    decorated.clear();
                    decorated.push_str(prefix.unwrap_or(""));
                    decorated.push_str(plain);
                    decorated.push_str(suffix.unwrap_or(""));
                    decorated.as_str()
                } else {
                    plain
                };
                let id = if prefix.is_none_or(str::is_empty) && suffix.is_none_or(str::is_empty) {
                    plain_id
                } else if let Some(&id) = w2id.get(key) {
                    id
                } else {
                    let id = u32::try_from(id2w.len()).map_err(|_| "BPE vocabulary exceeds u32")?;
                    if id == NONE {
                        return Err("indexed BPE reserves u32::MAX for word separators".into());
                    }
                    let token = CompactString::from(key);
                    id2w.push(token.clone());
                    w2id.insert(token, id);
                    lengths.push(0);
                    id
                };
                if let Some(cache) = &mut character_cache
                    && flags != 0
                {
                    cache.variants[rank as usize][flags - 1] = id;
                }
                lengths[id as usize] = 1;
                corpus.push(id);
            }
            edges += (corpus.len() - start).saturating_sub(1);
            corpus.push(NONE);
            if let Some(p) = progress {
                p.inc(1);
            }
        }
        trainer.finalize_progress(progress, wc.len(), "Tokenize words");
        trainer.update_progress(progress, wc.len(), "Count pairs");
        // Nonempty affixes can alias distinct raw spans. Without affixes, the
        // length predicate is constant per pair identity, so the proof still
        // applies to finite HF limits (including the initial-pair exception).
        let plain = supports_compact(trainer);
        Ok(Self {
            corpus,
            pivots,
            weights,
            lengths,
            monotone_floor: plain.then_some(trainer.min_frequency.max(1)),
            edges,
            character_table_bytes,
        })
    }
}

impl PreparedCorpus {
    fn relayout<C: parallel::Slot>(self, sort: bool) -> (PreparedCorpus<C>, f64, f64) {
        let begin = Instant::now();
        if !sort {
            return (
                PreparedCorpus {
                    corpus: C::from_u32(self.corpus),
                    pivots: self.pivots,
                    weights: self.weights,
                    lengths: self.lengths,
                    monotone_floor: self.monotone_floor,
                    edges: self.edges,
                    character_table_bytes: self.character_table_bytes,
                },
                begin.elapsed().as_secs_f64() * 1000.0,
                0.0,
            );
        }
        let mut order: Vec<usize> = (0..self.pivots.len()).collect();
        let sort_begin = Instant::now();
        order.sort_unstable_by_key(|&i| std::cmp::Reverse(self.weights[i]));
        let sort_ms = sort_begin.elapsed().as_secs_f64() * 1000.0;
        let mut corpus = Vec::with_capacity(self.corpus.len());
        let mut pivots = Vec::with_capacity(order.len());
        let mut weights = Vec::with_capacity(order.len());
        corpus.push(C::encode(NONE));
        for word in order {
            let start = self.pivots[word] as usize;
            let end = self
                .pivots
                .get(word + 1)
                .map_or(self.corpus.len(), |&p| p as usize);
            pivots.push(corpus.len() as u32);
            weights.push(self.weights[word]);
            corpus.extend(self.corpus[start..end].iter().map(|&id| C::encode(id)));
        }
        (
            PreparedCorpus {
                corpus,
                pivots,
                weights,
                lengths: self.lengths,
                monotone_floor: self.monotone_floor,
                edges: self.edges,
                character_table_bytes: self.character_table_bytes,
            },
            begin.elapsed().as_secs_f64() * 1000.0,
            sort_ms,
        )
    }
}

impl<C: parallel::Slot> Index<C> {
    fn from_prepared(
        input: PreparedCorpus<C>,
        tokenize_ms: f64,
        relayout_ms: f64,
        sort_ms: f64,
        execution: &cohort_parallel::Execution<'_>,
    ) -> Result<Self> {
        // Fresh pair frequencies are bounded by the initial weighted edge mass.
        // Compact priority encoding is disabled for the entire wide ID domain.
        let packed_range = C::NARROW
            && execution.options.packed_queue
            && input
                .pivots
                .iter()
                .zip(&input.weights)
                .enumerate()
                .try_fold(0_u64, |mass, (i, (&start, &weight))| {
                    let end = input
                        .pivots
                        .get(i + 1)
                        .map_or(input.corpus.len(), |&p| p as usize);
                    mass.checked_add(
                        weight.checked_mul(end.saturating_sub(start as usize + 2) as u64)?,
                    )
                })
                .is_some_and(|mass| mass <= u32::MAX as u64);
        let character_table_bytes = input.character_table_bytes;
        execution.configure(input.edges);
        let initial_pool = execution.initialization_pool();
        let parallel_initial = initial_pool
            .filter(|pool| execution.options.initial_grouped && pool.current_num_threads() > 1)
            .filter(|_| input.lengths.len() <= u16::MAX as usize + 1);
        let mut index = if let Some(pool) = parallel_initial {
            Self::new_grouped(input, pool, execution.options.sort_weights)?
        } else {
            Self::new(
                input.corpus,
                input.pivots,
                input.weights,
                input.lengths,
                input.monotone_floor,
            )?
        };
        if packed_range {
            index.queue.pack();
        }
        index.options = execution.options;
        index.corpus_sort_ms = sort_ms;
        index.corpus_relayout_ms = relayout_ms;
        if execution.options.sort_weights {
            let mut pivots = Vec::new();
            let mut weights = Vec::new();
            for (&pivot, &weight) in index.pivots.iter().zip(&index.weights) {
                if weights.last() != Some(&weight) {
                    pivots.push(pivot);
                    weights.push(weight);
                }
            }
            index.weight_pivots = Some(pivots);
            index.weights = weights;
        }
        if execution.options.weight_lookup {
            index.weight_lookup = Some(parallel::weight_lookup::WeightLookup::from_parts(
                index.weight_pivots.as_deref().unwrap_or(&index.pivots),
                &index.weights,
                0,
                execution.options.sort_weights,
                index.corpus.len(),
            ));
        }
        index.tokenize_ms = tokenize_ms;
        index.character_table_bytes = character_table_bytes;
        Ok(index)
    }
    fn new(
        corpus: Vec<C>,
        pivots: Vec<u32>,
        weights: Vec<u64>,
        lengths: Vec<u32>,
        monotone_floor: Option<u64>,
    ) -> Result<Self> {
        let begin = Instant::now();
        // One probe per physical edge updates both its ledger and posting.
        // Every key, including zero-weight keys, remains in the general ledger.
        struct Initial {
            count: i64,
            positions: SmallPosting,
        }
        let mut initial = AHashMap::<Pair, Initial>::new();
        let mut total = 0_i64;
        let mut initial_edges = 0;
        for (&pivot, &weight) in pivots.iter().zip(&weights) {
            let weight =
                i64::try_from(weight).map_err(|_| "indexed BPE weight exceeds i64::MAX")?;
            let mut pos = pivot as usize;
            while corpus[pos].token() != NONE {
                if corpus[pos + 1].token() != NONE {
                    initial_edges += 1;
                    total = total
                        .checked_add(weight)
                        .ok_or("indexed BPE weighted pair counts exceed i64::MAX")?;
                    let pair = (corpus[pos].token(), corpus[pos + 1].token());
                    let entry = initial.entry(pair).or_insert_with(|| Initial {
                        count: 0,
                        positions: SmallPosting::default(),
                    });
                    entry.count += weight;
                    entry.positions.push(pos as u32)?;
                }
                pos += 1;
            }
        }
        let mut counts: AHashMap<Pair, i64> = initial
            .iter()
            .map(|(&pair, entry)| (pair, entry.count))
            .collect();
        let queue: OctonaryHeap<Candidate> = initial
            .into_iter()
            .filter_map(|(pair, entry)| {
                let count = entry.count;
                (count > 0 && count as u64 >= monotone_floor.unwrap_or(1)).then_some(Candidate {
                    pair,
                    count: count as u64,
                    positions: entry.positions,
                })
            })
            .collect();
        let mut pruned_pairs = 0;
        if let Some(floor) = monotone_floor {
            let before = counts.len();
            counts.retain(|_, count| *count > 0 && *count as u64 >= floor);
            pruned_pairs = before - counts.len();
        }
        Ok(Self {
            corpus,
            lengths,
            spans: None,
            pivots,
            weights,
            counts,
            queue: queue.into(),
            births: AHashMap::new(),
            cohort_workspace: None,
            options: cohort_parallel::Options::default(),
            weight_pivots: None,
            weight_lookup: None,
            corpus_sort_ms: 0.0,
            corpus_relayout_ms: 0.0,
            monotone_floor,
            pruned_pairs,
            initial_edges,
            tokenize_ms: 0.0,
            initial_count_ms: begin.elapsed().as_secs_f64() * 1000.0,
            character_table_bytes: 0,
        })
    }

    fn new_grouped(
        input: PreparedCorpus<C>,
        pool: &rayon::ThreadPool,
        sorted_weights: bool,
    ) -> Result<Self> {
        let begin = Instant::now();
        let mut mass = 0_i64;
        for (word, &weight) in input.weights.iter().enumerate() {
            let weight =
                i64::try_from(weight).map_err(|_| "indexed BPE weight exceeds i64::MAX")?;
            let start = input.pivots[word] as usize;
            let end = input
                .pivots
                .get(word + 1)
                .map_or(input.corpus.len(), |&p| p as usize);
            let edges = end.saturating_sub(start + 2);
            mass = mass
                .checked_add(
                    weight
                        .checked_mul(edges as i64)
                        .ok_or("indexed BPE weighted pair counts exceed i64::MAX")?,
                )
                .ok_or("indexed BPE weighted pair counts exceed i64::MAX")?;
        }
        let (counts, queue) = pool.install(|| {
            parallel::initial_cohorts(
                &input.corpus,
                &input.pivots,
                &input.weights,
                pool.current_num_threads(),
                sorted_weights,
            )
        })?;
        Ok(Self {
            corpus: input.corpus,
            lengths: input.lengths,
            spans: None,
            pivots: input.pivots,
            weights: input.weights,
            counts,
            queue: queue.into(),
            births: AHashMap::new(),
            cohort_workspace: None,
            options: cohort_parallel::Options::default(),
            weight_pivots: None,
            weight_lookup: None,
            corpus_sort_ms: 0.0,
            corpus_relayout_ms: 0.0,
            monotone_floor: input.monotone_floor,
            pruned_pairs: 0,
            initial_edges: input.edges,
            tokenize_ms: 0.0,
            initial_count_ms: begin.elapsed().as_secs_f64() * 1000.0,
            character_table_bytes: 0,
        })
    }

    #[inline(always)]
    fn weight_at(&self, pos: u32) -> u64 {
        let pivots = self.weight_pivots.as_deref().unwrap_or(&self.pivots);
        self.weight_lookup.as_ref().map_or_else(
            || {
                let i = pivots.partition_point(|&q| q <= pos);
                if i == 0 { 0 } else { self.weights[i - 1] }
            },
            |l| l.weight_parts(pivots, &self.weights, 0, pos),
        )
    }
    fn word_at(&self, pos: u32) -> usize {
        self.pivots.partition_point(|&pivot| pivot <= pos) - 1
    }

    fn span(&self, pos: usize) -> u32 {
        match &self.spans {
            Some(spans) => spans[pos],
            None => self.lengths[self.corpus[pos].token() as usize],
        }
    }

    fn inspect(&self, pos: usize, pair: Pair) -> Option<Context> {
        if self.corpus[pos].token() != pair.0 {
            return None;
        }
        let left_len = self.span(pos);
        let right = pos + left_len as usize;
        if left_len == 0 || right >= self.corpus.len() || self.corpus[right].token() != pair.1 {
            return None;
        }
        let right_len = self.span(right);
        let after = right + right_len as usize;
        if right_len == 0 || after >= self.corpus.len() {
            return None;
        }
        let before = if self.corpus[pos - 1].token() == NONE {
            None
        } else {
            Some(pos - self.span(pos - 1) as usize)
        };
        Some(Context {
            pos,
            right,
            after,
            before,
            len: left_len + right_len,
        })
    }

    fn prepare_replacement(&mut self, pair: Pair, replacement: u32) {
        if self.monotone_floor.is_some() {
            let length = self.lengths[pair.0 as usize] + self.lengths[pair.1 as usize];
            if replacement as usize == self.lengths.len() {
                self.lengths.push(length);
            } else {
                // The isolation lemma rules out a second activation of a raw
                // string. Reusing a reserved ID is its first live occurrence.
                debug_assert_eq!(self.lengths[replacement as usize], 0);
                self.lengths[replacement as usize] = length;
            }
            return;
        }
        if replacement as usize == self.lengths.len() {
            let length = if self.spans.is_none() {
                self.lengths[pair.0 as usize] + self.lengths[pair.1 as usize]
            } else {
                0 // Occurrence spans supply lengths after the first ID reuse.
            };
            self.lengths.push(length);
        } else if self.spans.is_none() {
            let length = self.lengths[pair.0 as usize] + self.lengths[pair.1 as usize];
            let previous = self.lengths[replacement as usize];
            // A reserved, never-live ID has no existing physical span. Reusing
            // an active ID with the same span also preserves the length table.
            // This check is valid for arbitrary affixes; it makes no claim
            // about pair monotonicity or whether historical cohorts can revive.
            if previous == 0 || previous == length {
                self.lengths[replacement as usize] = length;
                return;
            }
            // Materialize only live endpoints before changing an existing ID's
            // physical span. No corpus-sized compatibility array in the fresh path.
            let mut spans = vec![0; self.corpus.len()];
            for &pivot in &self.pivots {
                let mut pos = pivot as usize;
                while self.corpus[pos].token() != NONE {
                    let len = self.lengths[self.corpus[pos].token() as usize];
                    spans[pos] = len;
                    spans[pos + len as usize - 1] = len;
                    pos += len as usize;
                }
            }
            self.spans = Some(spans);
        }
    }

    fn change(&mut self, pair: Pair, change: i64, pos: u32, weight: i64) -> Result<()> {
        let count = match self.counts.entry(pair) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(_)
                if change < 0 && self.monotone_floor.is_some() =>
            {
                // A retired old pair can only lose occurrences. Its forgotten
                // count cannot affect eligibility or create another candidate.
                return Ok(());
            }
            std::collections::hash_map::Entry::Vacant(entry) => entry.insert(0),
        };
        *count = count
            .checked_add(change * weight)
            .ok_or("indexed BPE pair ledger exceeds i64 range")?;
        if change > 0 {
            self.births.entry(pair).or_default().push(pos)?;
        }
        Ok(())
    }

    fn merge_at(
        &mut self,
        context: Context,
        replacement: u32,
        max_length: usize,
        weight: i64,
    ) -> Result<usize> {
        let Context {
            pos,
            right,
            after,
            before,
            len,
        } = context;
        let left_id = self.corpus[pos].token();
        let right_id = self.corpus[right].token();
        // Match Word::merge: remove both old neighbours unconditionally, and
        // introduce a new neighbour only for a STRICTLY smaller combined span.
        // The chosen pair's own frequency is not subtracted in the HF ledger.
        if let Some(before) = before {
            let prior_id = self.corpus[before].token();
            self.change((prior_id, left_id), -1, before as u32, weight)?;
            if (self.span(before) as usize) + (len as usize) < max_length {
                self.change((prior_id, replacement), 1, before as u32, weight)?;
            }
        }
        if self.corpus[after].token() != NONE {
            let after_id = self.corpus[after].token();
            self.change((right_id, after_id), -1, right as u32, weight)?;
            if (len as usize) + (self.span(after) as usize) < max_length {
                self.change((replacement, after_id), 1, pos as u32, weight)?;
            }
        }
        // Same endpoint rewrite as the prototype: retain the new ID at the
        // start/end, and clear a retired right start when it is not also the end.
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

    fn apply(
        &mut self,
        mut top: Candidate,
        replacement: u32,
        scan_words: bool,
        max_length: usize,
        stats: &mut IndexedTrainingStats,
        pool: Option<&rayon::ThreadPool>,
    ) -> Result<()> {
        self.prepare_replacement(top.pair, replacement);
        stats.posting_visits += top.positions.len();
        let parallel_pool = pool.filter(|p| {
            self.options.parallel_apply
                && p.current_num_threads() > 1
                && top.positions.len() >= 4096
        });
        if parallel_pool.is_some() || (self.options.grouped_tail && top.positions.len() >= 64) {
            if !scan_words && top.pair.0 == top.pair.1 {
                top.positions.as_mut_slice().sort_unstable();
            }
            cohort_parallel::apply(
                self,
                &top,
                replacement,
                scan_words,
                max_length,
                parallel_pool,
                stats,
            )?;
        } else if scan_words {
            stats.cohort_scan_activations += 1;
            // An indexed edge stands for its entire word cohort in HF. Even a
            // retired or length-excluded occurrence can identify that word.
            let mut words: Vec<u32> = top
                .positions
                .as_slice()
                .iter()
                .map(|&p| self.word_at(p) as u32)
                .collect();
            words.sort_unstable();
            words.dedup();
            stats.cohort_words_scanned += words.len();
            stats.cohort_scan_max_words = stats.cohort_scan_max_words.max(words.len());
            for word in words {
                let mut pos = self.pivots[word as usize] as usize;
                let weight = self.weight_at(self.pivots[word as usize]) as i64;
                while self.corpus[pos].token() != NONE {
                    stats.word_scan_steps += 1;
                    pos = if let Some(context) = self.inspect(pos, top.pair) {
                        self.merge_at(context, replacement, max_length, weight)?
                    } else {
                        pos + self.span(pos) as usize
                    };
                }
            }
        } else {
            // AB occurrences cannot overlap. AA must choose the leftmost edge
            // in each run; births are not necessarily globally ordered.
            if top.pair.0 == top.pair.1 {
                top.positions.as_mut_slice().sort_unstable();
            }
            for &pos in top.positions.as_slice() {
                if let Some(context) = self.inspect(pos as usize, top.pair) {
                    let weight = self.weight_at(pos) as i64;
                    self.merge_at(context, replacement, max_length, weight)?;
                } else {
                    stats.stale_posting_visits += 1;
                }
            }
        }
        let queue_begin = Instant::now();
        for (pair, positions) in self.births.drain() {
            let count = self.counts[&pair];
            // Aggregate the entire round first: multiple births of a NEW pair
            // can increase its count before it becomes a completed old key.
            if count > 0 && count as u64 >= self.monotone_floor.unwrap_or(1) {
                self.queue.push(Candidate {
                    pair,
                    count: count as u64,
                    positions,
                });
            } else if self.monotone_floor.is_some() {
                self.counts.remove(&pair);
                self.pruned_pairs += 1;
            }
        }
        stats.commit_ms += queue_begin.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }
}

impl BpeTrainer {
    pub(super) fn train_vocab_indexed_parallel(
        &self,
        config: IndexedParallelConfig,
    ) -> Result<IndexedTraining> {
        self.do_train_indexed_parallel(&self.words, config)
    }

    /// Affixes use guarded fresh-ID batches and reconstruct HF cohorts on reuse.
    /// Workers are explicit and independent of the global Rayon pool.
    pub(super) fn do_train_indexed_parallel(
        &self,
        word_counts: &AHashMap<CompactString, u64>,
        config: IndexedParallelConfig,
    ) -> Result<IndexedTraining> {
        if config.workers == 0
            || config.initialization_workers == Some(0)
            || !matches!(config.posting_block_bits, 16 | 32)
            || config.batch_size == 0
            || config.batch_size > 256
        {
            return Err("invalid indexed parallel configuration".into());
        }
        if supports_compact(self) {
            parallel::train(self, word_counts, config)
        } else {
            let options = cohort_parallel::Options::default();
            self.train_affixed_options(word_counts, config, options)
        }
    }
    fn train_affixed_options(
        &self,
        word_counts: &AHashMap<CompactString, u64>,
        config: IndexedParallelConfig,
        options: cohort_parallel::Options,
    ) -> Result<IndexedTraining> {
        if !options.guarded_fast {
            return self.train_cohorts_parallel_options(word_counts, config, options);
        }
        let begin = Instant::now();
        match parallel::train_affixed(self, word_counts, config, options) {
            Ok(result) => Ok(result),
            Err(error) => match error.downcast::<parallel::AliasCollision>() {
                Ok(collision) => {
                    let spent = begin.elapsed().as_secs_f64() * 1000.0;
                    // The original weighted words are unchanged. Reconstruct
                    // every HF intermediate cohort, including ones fast pruning
                    // omitted, before accepting any reused live identity.
                    let mut result =
                        self.train_cohorts_parallel_options(word_counts, config, options)?;
                    result.stats.alias_guarded = true;
                    result.stats.alias_fallback = true;
                    result.stats.speculative_initialize_ms = collision.stats.initialize_ms;
                    result.stats.speculative_merge_ms = collision.stats.merge_ms;
                    result.stats.speculative_total_ms = spent;
                    result.stats.speculative_applied_merges =
                        collision.stats.speculative_applied_merges;
                    result.stats.speculative_selected_merges =
                        collision.stats.speculative_selected_merges;
                    result.stats.speculative_posting_allocations =
                        collision.stats.posting_allocations;
                    Ok(result)
                }
                Err(error) => Err(error),
            },
        }
    }
    /// Experimental sequential position-index trainer. The ordinary `train_vocab`
    /// remains the reference backend. Initialization preserves HF's token identities
    /// while writing directly into Efficient BPE's flat endpoint layout.
    pub(super) fn train_vocab_indexed(&self) -> Result<IndexedTraining> {
        self.do_train_indexed(&self.words)
    }

    /// Combine stable endpoint indexing for long pieces with the PR's compact
    /// word arena and forward rewrite for short pieces. Affixes keep HF handling.
    pub(super) fn train_vocab_fused(&self) -> Result<IndexedTraining> {
        self.do_train_fused(&self.words)
    }

    pub(super) fn do_train_fused(
        &self,
        word_counts: &AHashMap<CompactString, u64>,
    ) -> Result<IndexedTraining> {
        if supports_compact(self) {
            compact::train(self, word_counts, true)
        } else {
            self.do_train_indexed(word_counts)
        }
    }

    /// Train weighted pretokenized words with the experimental occurrence index.
    /// Nonempty affixes retain HF word-cohort handling. Ordinary character BPE
    /// uses the proven monotone-pair path, including reserved IDs and length gates.
    pub(super) fn do_train_indexed(
        &self,
        word_counts: &AHashMap<CompactString, u64>,
    ) -> Result<IndexedTraining> {
        if supports_compact(self) {
            return compact::train(self, word_counts, false);
        }
        self.train_cohorts_with_policy(word_counts, posting_arena::Policy::System)
    }

    fn train_cohorts_with_policy(
        &self,
        word_counts: &AHashMap<CompactString, u64>,
        policy: posting_arena::Policy,
    ) -> Result<IndexedTraining> {
        self.train_cohorts_local_options(word_counts, policy, cohort_parallel::Options::default())
    }
    fn train_cohorts_local_options(
        &self,
        word_counts: &AHashMap<CompactString, u64>,
        policy: posting_arena::Policy,
        options: cohort_parallel::Options,
    ) -> Result<IndexedTraining> {
        let session = posting_arena::LocalSession::new();
        // Every posting owner is local to train_cohorts. They are destroyed
        // before session.finish, including when training returns an error.
        let execution = cohort_parallel::Execution {
            pool: None,
            initialization_pool: None,
            policy,
            options,
        };
        let mut result = self.train_cohorts(word_counts, &execution);
        let allocations = session.finish();
        if let Ok(trained) = &mut result {
            trained.stats.posting_allocation_policy = policy.label();
            trained.stats.posting_arena_cutoff_bytes = policy.cutoff(trained.stats.initial_edges);
            trained.stats.posting_allocations = allocations;
        }
        result
    }

    fn train_cohorts_parallel(
        &self,
        word_counts: &AHashMap<CompactString, u64>,
        config: IndexedParallelConfig,
    ) -> Result<IndexedTraining> {
        let mut options = cohort_parallel::Options::default();
        options.narrow_corpus &= config.narrow_corpus;
        self.train_cohorts_parallel_options(word_counts, config, options)
    }
    fn train_cohorts_parallel_options(
        &self,
        word_counts: &AHashMap<CompactString, u64>,
        config: IndexedParallelConfig,
        mut options: cohort_parallel::Options,
    ) -> Result<IndexedTraining> {
        options.narrow_corpus &= config.narrow_corpus;
        let policy = if options.arena_allocator {
            posting_arena::Policy::Auto
        } else {
            posting_arena::Policy::System
        };
        if config.workers == 1 && config.initialization_workers.unwrap_or(1) == 1 {
            return self.train_cohorts_local_options(word_counts, policy, options);
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.workers)
            .build()?;
        let initialization_pool = config
            .initialization_workers
            .filter(|&n| n != config.workers)
            .map(|n| rayon::ThreadPoolBuilder::new().num_threads(n).build())
            .transpose()?;
        let session = posting_arena::Session::new(&pool, initialization_pool.as_ref());
        let execution = cohort_parallel::Execution {
            pool: Some(&pool),
            initialization_pool: initialization_pool.as_ref(),
            policy,
            options,
        };
        // The coordinator also runs in this dedicated pool. No posting owner
        // escapes the call, so allocation and retirement counters share a session.
        let mut result = pool.install(|| self.train_cohorts(word_counts, &execution));
        let allocations = session.finish();
        if let Ok(trained) = &mut result {
            trained.stats.posting_allocation_policy = policy.label();
            trained.stats.posting_arena_cutoff_bytes = policy.cutoff(trained.stats.initial_edges);
            trained.stats.posting_allocations = allocations;
        }
        result
    }

    fn train_cohorts(
        &self,
        word_counts: &AHashMap<CompactString, u64>,
        execution: &cohort_parallel::Execution<'_>,
    ) -> Result<IndexedTraining> {
        let begin = Instant::now();
        let mut word_to_id = AHashMap::with_capacity(self.vocab_size);
        let mut id_to_word = Vec::with_capacity(self.vocab_size);
        let progress = self.setup_progress();
        self.add_special_tokens(&mut word_to_id, &mut id_to_word);
        let alphabet_begin = Instant::now();
        let alphabet_scratch_bytes = if execution.options.parallel_alphabet {
            if let Some(pool) = execution.initialization_pool() {
                pool.install(|| {
                    parallel::initialize_alphabet(
                        self,
                        word_counts,
                        &mut word_to_id,
                        &mut id_to_word,
                        pool.current_num_threads(),
                    )
                })
            } else {
                self.compute_alphabet(word_counts, &mut word_to_id, &mut id_to_word);
                0
            }
        } else {
            self.compute_alphabet(word_counts, &mut word_to_id, &mut id_to_word);
            0
        };
        let alphabet_ms = alphabet_begin.elapsed().as_secs_f64() * 1000.0;
        self.update_progress(&progress, word_counts.len(), "Tokenize words");
        let tokenize_begin = Instant::now();
        let input = PreparedCorpus::tokenize_with_cache(
            self,
            word_counts,
            &mut word_to_id,
            &mut id_to_word,
            &progress,
            execution.options.character_cache,
            execution
                .options
                .parallel_measure
                .then(|| execution.initialization_pool())
                .flatten(),
        )?;
        let tokenize_ms = tokenize_begin.elapsed().as_secs_f64() * 1000.0;
        if execution.options.narrow_corpus
            && self.vocab_size.max(id_to_word.len()) <= u16::MAX as usize
        {
            let (input, relayout_ms, sort_ms) =
                input.relayout::<u16>(execution.options.sort_weights);
            self.train_prepared(
                input,
                tokenize_ms,
                relayout_ms,
                sort_ms,
                word_to_id,
                id_to_word,
                progress,
                alphabet_ms,
                alphabet_scratch_bytes,
                begin,
                execution,
            )
        } else {
            let (input, relayout_ms, sort_ms) =
                input.relayout::<u32>(execution.options.sort_weights);
            self.train_prepared(
                input,
                tokenize_ms,
                relayout_ms,
                sort_ms,
                word_to_id,
                id_to_word,
                progress,
                alphabet_ms,
                alphabet_scratch_bytes,
                begin,
                execution,
            )
        }
    }
    fn train_prepared<C: parallel::Slot>(
        &self,
        input: PreparedCorpus<C>,
        tokenize_ms: f64,
        relayout_ms: f64,
        sort_ms: f64,
        mut word_to_id: AHashMap<CompactString, u32>,
        mut id_to_word: Vec<CompactString>,
        progress: Option<ProgressBar>,
        alphabet_ms: f64,
        alphabet_scratch_bytes: usize,
        begin: Instant,
        execution: &cohort_parallel::Execution<'_>,
    ) -> Result<IndexedTraining> {
        let mut index = Index::from_prepared(input, tokenize_ms, relayout_ms, sort_ms, execution)?;
        self.finalize_progress(&progress, index.pivots.len(), "Count pairs");
        let mut stats = IndexedTrainingStats {
            initialize_ms: begin.elapsed().as_secs_f64() * 1000.0,
            alphabet_ms,
            alphabet_scratch_bytes,
            tokenize_ms: index.tokenize_ms,
            initial_count_ms: index.initial_count_ms,
            character_table_bytes: index.character_table_bytes,
            initial_symbols: index.corpus.len() - index.pivots.len() - 1,
            initial_slots: index.corpus.len(),
            initial_edges: index.initial_edges,
            initial_pairs: index.counts.len(),
            initial_length_bytes: index.lengths.capacity() * std::mem::size_of::<u32>(),
            workers: execution.workers(),
            initialization_workers: if execution.options.initial_grouped
                && index.lengths.len() <= u16::MAX as usize + 1
            {
                execution
                    .initialization_pool()
                    .map_or(1, rayon::ThreadPool::current_num_threads)
            } else {
                1
            },
            initial_count_backend: if execution
                .initialization_pool()
                .is_some_and(|p| p.current_num_threads() > 1)
                && execution.options.initial_grouped
                && index.lengths.len() <= u16::MAX as usize + 1
            {
                "cohort_stable_radix16"
            } else {
                "cohort_grouped_hash"
            },
            monotone_pairs: index.monotone_floor.is_some(),
            layout: "hf_cohorts",
            corpus_bytes: index.corpus.capacity() * std::mem::size_of::<C>(),
            initial_corpus_bytes: index.corpus.capacity() * std::mem::size_of::<C>(),
            initial_slot_bytes: index.corpus.capacity() * std::mem::size_of::<C>(),
            corpus_slot_bytes: std::mem::size_of::<C>(),
            initial_heap_bytes: index.queue.bytes(),
            queue_selection_mode: if index.queue.packed() {
                "cohort_packed"
            } else {
                "cohort_wide"
            },
            corpus_sort_ms: index.corpus_sort_ms,
            corpus_relayout_ms: index.corpus_relayout_ms,
            corpus_weight_order: if execution.options.sort_weights {
                "descending_weight"
            } else {
                "input"
            },
            weight_interval_count: index.weights.len(),
            initial_weight_bytes: index.weights.capacity() * 8
                + index.weight_pivots.as_ref().map_or(0, |p| p.capacity() * 4),
            weight_lookup_bytes: index.weight_lookup.as_ref().map_or(0, |l| l.bytes()),
            ..Default::default()
        };
        let begin = Instant::now();
        let mut scan_words = self.max_token_length.is_some();
        let max_length = self.max_token_length.unwrap_or(usize::MAX);
        let mut merges = Vec::new();
        #[cfg(test)]
        let mut trace = Vec::new();
        self.update_progress(&progress, self.vocab_size, "Compute merges");
        let mut select_begin = Instant::now();
        while word_to_id.len() < self.vocab_size {
            let Some(mut top) = index.queue.pop() else {
                break;
            };
            let Some(&count) = index.counts.get(&top.pair) else {
                debug_assert!(index.monotone_floor.is_some());
                continue;
            };
            if let Some(floor) = index.monotone_floor
                && (count <= 0 || (count as u64) < floor)
            {
                index.counts.remove(&top.pair);
                index.pruned_pairs += 1;
                continue;
            }
            let count = count as u64;
            if top.count != count {
                top.count = count;
                index.queue.push(top);
                continue;
            }
            if count == 0 || count < self.min_frequency {
                break;
            }
            let a = &id_to_word[top.pair.0 as usize];
            let b = id_to_word[top.pair.1 as usize].as_str();
            let b = self
                .continuing_subword_prefix
                .as_deref()
                .and_then(|p| b.strip_prefix(p))
                .unwrap_or(b);
            let token = CompactString::from(format!("{a}{b}"));
            let id = if let Some(&id) = word_to_id.get(&token) {
                stats.reused_ids += 1;
                // Promote before aliases can create multiple equal-key cohorts.
                index.queue.promote();
                // First activation of a reserved ID cannot revive historical
                // cohorts containing it. Once an already-live ID is reused,
                // retain word scanning for the remainder of the general run.
                scan_words |= index.monotone_floor.is_none()
                    && (index.spans.is_some() || index.lengths[id as usize] != 0);
                id
            } else {
                let id = u32::try_from(id_to_word.len())
                    .map_err(|_| "BPE vocabulary exceeds u32::MAX")?;
                if id == NONE {
                    return Err("indexed BPE reserves u32::MAX for word separators".into());
                }
                id_to_word.push(token.clone());
                word_to_id.insert(token, id);
                id
            };
            #[cfg(test)]
            trace.push((top.pair, top.count, id));
            merges.push(top.pair);
            stats.select_ms += select_begin.elapsed().as_secs_f64() * 1000.0;
            index.apply(top, id, scan_words, max_length, &mut stats, execution.pool)?;
            if let Some(p) = &progress {
                p.inc(1)
            }
            self.emit_json_progress("Compute merges", merges.len(), self.vocab_size);
            select_begin = Instant::now();
        }
        stats.select_ms += select_begin.elapsed().as_secs_f64() * 1000.0;
        self.finalize_progress(&progress, merges.len(), "Compute merges");
        stats.merge_ms = begin.elapsed().as_secs_f64() * 1000.0;
        stats.span_table_bytes = index.spans.as_ref().map_or(0, |s| s.capacity() * 4);
        stats.pruned_pairs = index.pruned_pairs;
        let vocab = word_to_id
            .into_iter()
            .map(|(token, id)| (token.to_string(), id))
            .collect();
        let merges = merges
            .into_iter()
            .map(|(a, b)| {
                (
                    id_to_word[a as usize].to_string(),
                    id_to_word[b as usize].to_string(),
                )
            })
            .collect();
        Ok(IndexedTraining {
            vocab,
            merges,
            special_tokens: self.special_tokens.clone(),
            stats,
            #[cfg(test)]
            trace,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tk_encode::pipeline::Model;

    #[test]
    fn general_reserved_first_activation_keeps_exact_cohorts_without_alias_scan() {
        let trainer = BpeTrainer::builder()
            .show_progress(false)
            .vocab_size(64)
            .min_frequency(1)
            .end_of_word_suffix("</w>".into())
            .special_tokens(vec![AddedToken::from("ab", true)])
            .build();
        let words = counts(&[("abx", 3), ("aby", 2), ("qqabz", 1)]);
        let trained = check(&trainer, &words);
        assert!(trained.stats.reused_ids > 0);
        assert_eq!(trained.stats.word_scan_steps, 0);
        assert_eq!(trained.stats.span_table_bytes, 0);
        assert_eq!(
            trained.stats.posting_allocations.arena_requested_bytes,
            trained.stats.posting_allocations.arena_retired_bytes
        );
    }

    fn counts(items: &[(&str, u64)]) -> AHashMap<CompactString, u64> {
        items.iter().map(|&(s, n)| (s.into(), n)).collect()
    }

    fn check(trainer: &BpeTrainer, wc: &AHashMap<CompactString, u64>) -> IndexedTraining {
        let mut trace = Vec::new();
        let (vocab, merges, special) = trainer
            .do_train_observed(wc, |p, c, id| trace.push((p, c, id)))
            .unwrap();
        let indexed = trainer.do_train_indexed(wc).unwrap();
        assert_eq!(
            indexed.trace, trace,
            "trace mismatch; words={wc:?}; trainer={trainer:?}"
        );
        assert_eq!(indexed.vocab, vocab);
        assert_eq!(indexed.merges, merges);
        assert_eq!(indexed.special_tokens, special);
        let fused = trainer.do_train_fused(wc).unwrap();
        assert_eq!(
            fused.trace, trace,
            "fused trace mismatch; words={wc:?}; trainer={trainer:?}"
        );
        assert_eq!(fused.vocab, vocab);
        assert_eq!(fused.merges, merges);
        assert_eq!(fused.special_tokens, special);
        let parallel = trainer
            .do_train_indexed_parallel(
                wc,
                IndexedParallelConfig {
                    workers: 2,
                    initialization_workers: None,
                    posting_block_bits: 16,
                    narrow_corpus: true,
                    atomic_corpus: false,
                    batch_size: 256,
                },
            )
            .unwrap();
        assert_eq!(
            parallel.trace, trace,
            "parallel trace mismatch; words={wc:?}; trainer={trainer:?}"
        );
        assert_eq!(parallel.vocab, vocab);
        assert_eq!(parallel.merges, merges);
        assert_eq!(parallel.special_tokens, special);
        let fused_parallel = trainer
            .do_train_indexed_parallel(
                wc,
                IndexedParallelConfig {
                    workers: 4,
                    initialization_workers: None,
                    posting_block_bits: 32,
                    narrow_corpus: false,
                    atomic_corpus: true,
                    batch_size: 256,
                },
            )
            .unwrap();
        assert_eq!(
            fused_parallel.trace, trace,
            "fused parallel trace mismatch; words={wc:?}; trainer={trainer:?}"
        );
        assert_eq!(fused_parallel.vocab, vocab);
        assert_eq!(fused_parallel.merges, merges);
        assert_eq!(fused_parallel.special_tokens, special);
        indexed
    }

    #[test]
    fn weighted_ties_overlap_empty_unicode_and_long_spans() {
        let trainer = BpeTrainer::builder()
            .vocab_size(100)
            .show_progress(false)
            .build();
        check(
            &trainer,
            &counts(&[
                ("", 1),
                ("a", 10),
                ("aaaaaaa", 3),
                ("abababab", 4),
                ("abcabc", 4),
                ("测试测试", 7),
                ("ééé", 2),
            ]),
        );
        let wc = counts(&[("a".repeat(1024).as_str(), 3)]);
        let result = check(&trainer, &wc);
        assert!(result.vocab.contains_key(&"a".repeat(1024)));
        assert_eq!(result.stats.word_scan_steps, 0);
    }

    #[test]
    fn special_token_identity_reuse_and_duplicate_specials() {
        let trainer = BpeTrainer::builder()
            .vocab_size(30)
            .show_progress(false)
            .special_tokens(
                ["ab", "aba", "ab", "aa", "aaaa"]
                    .map(|s| AddedToken::from(s, true))
                    .to_vec(),
            )
            .build();
        let result = check(
            &trainer,
            &counts(&[("abababa", 5), ("aaaaaaa", 7), ("baab", 2)]),
        );
        assert!(result.stats.reused_ids >= 2);
        assert!(result.stats.monotone_pairs);
        assert_eq!(result.stats.word_scan_steps, 0);
        assert!(result.merges.len() > result.vocab.len() - 7);
    }

    #[test]
    fn monotone_pruning_waits_for_global_birth_count() {
        let trainer = BpeTrainer::builder()
            .vocab_size(30)
            .min_frequency(2)
            .show_progress(false)
            .build();
        let result = check(&trainer, &counts(&[("xabp", 1), ("xabq", 1)]));
        assert!(result.stats.monotone_pairs);
        assert!(result.stats.pruned_pairs > 0);
        // Each word contributes only one new (x,ab) edge, but their global
        // birth count is two. Per-word filtering would lose the second rule.
        assert_eq!(
            result.merges,
            vec![("a".into(), "b".into()), ("x".into(), "ab".into())]
        );
        let mut empty_affixes = trainer.clone();
        empty_affixes.continuing_subword_prefix = Some(String::new());
        empty_affixes.end_of_word_suffix = Some(String::new());
        assert!(
            check(&empty_affixes, &counts(&[("xabp", 1), ("xabq", 1)]))
                .stats
                .monotone_pairs
        );
    }

    #[test]
    fn suffix_alias_really_increases_an_old_pair() {
        let trainer = BpeTrainer::builder()
            .vocab_size(10)
            .end_of_word_suffix("a".into())
            .show_progress(false)
            .build();
        let result = check(&trainer, &counts(&[("baaba", 1)]));
        assert!(!result.stats.monotone_pairs);
        assert_eq!(result.stats.pruned_pairs, 0);
        // a=0,b=1,aa=2; the terminal a already has the string identity aa.
        // [b,a,a,b,aa] -> [b,aa,b,aa] makes old (b,aa) grow from one to two.
        assert_eq!(result.trace[0], ((0, 0), 1, 2));
        assert_eq!(result.trace[1], ((1, 2), 2, 3));
        assert!(result.stats.reused_ids > 0);
        assert!(result.stats.word_scan_steps > 0);
    }

    #[test]
    fn affixes_alphabet_and_length_boundaries() {
        let wc = counts(&[
            ("aaaaaaa", 11),
            ("abab", 7),
            ("caba", 4),
            ("测试测试", 3),
            ("ccc", 1),
        ]);
        for prefix in [None, Some("##"), Some("a"), Some("")] {
            for suffix in [None, Some("</w>"), Some("a"), Some("")] {
                for limit in [None, Some(0), Some(1), Some(2), Some(3), Some(5), Some(16)] {
                    let mut trainer = BpeTrainer::builder()
                        .vocab_size(70)
                        .show_progress(false)
                        .special_tokens(vec![AddedToken::from("aa", true)])
                        .build();
                    trainer.continuing_subword_prefix = prefix.map(str::to_owned);
                    trainer.end_of_word_suffix = suffix.map(str::to_owned);
                    trainer.max_token_length = limit;
                    check(&trainer, &wc);
                }
            }
        }
        let trainer = BpeTrainer::builder()
            .vocab_size(30)
            .show_progress(false)
            .limit_alphabet(3)
            .initial_alphabet(['a', '测'].into())
            .build();
        check(&trainer, &wc);
    }

    fn next(rng: &mut u64) -> u64 {
        *rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *rng >> 32
    }

    #[test]
    fn randomized_round_by_round_hf_differential() {
        let mut rng = 2348;
        for case in 0..1500 {
            let mut wc = AHashMap::<CompactString, u64>::new();
            for _ in 0..(1 + next(&mut rng) % 12) {
                let word: String = (0..next(&mut rng) % 24)
                    .map(|_| ['a', 'b', 'c', '测'][next(&mut rng) as usize % 4])
                    .collect();
                *wc.entry(word.into()).or_default() += 1 + next(&mut rng) % 9;
            }
            let mut trainer = BpeTrainer::builder()
                .show_progress(false)
                .vocab_size((4 + next(&mut rng) % 50) as usize)
                .min_frequency(next(&mut rng) % 6)
                .build();
            if case % 3 == 0 {
                trainer.special_tokens = ["aa", "ab", "aba", "abc", "ba", "测测"]
                    .map(|s| AddedToken::from(s, true))
                    .to_vec();
            }
            if case % 4 == 0 {
                trainer.continuing_subword_prefix =
                    Some(["##", "a", "", "测"][(case / 4) % 4].into());
            }
            if case % 5 == 0 {
                trainer.end_of_word_suffix = Some(["</w>", "a", ""][case % 3].into());
            }
            if case % 2 == 0 {
                trainer.max_token_length = Some((next(&mut rng) % 10) as usize);
            }
            check(&trainer, &wc);
        }
    }

    #[test]
    fn encode_and_json_reload_match_reference() {
        let wc = counts(&[("aaaaa", 3), ("abab", 9), ("测试测试", 5), ("cababa", 2)]);
        for affix in [false, true] {
            let mut trainer = BpeTrainer::builder()
                .vocab_size(50)
                .show_progress(false)
                .special_tokens(vec![AddedToken::from("ab", true)])
                .build();
            if affix {
                trainer.continuing_subword_prefix = Some("##".into());
                trainer.end_of_word_suffix = Some("</w>".into());
            }
            let got = check(&trainer, &wc);
            // Persist the exact raw parts the prototype produces, including rank order.
            let tmp = tempfile::NamedTempFile::new().unwrap();
            serde_json::to_writer(tmp.as_file(), &(&got.vocab, &got.merges)).unwrap();
            let (vocab, merges): (Vocab, Merges) =
                serde_json::from_reader(tmp.reopen().unwrap()).unwrap();
            let reloaded = PipelineBPE::from_config(BpeConfig {
                vocab,
                merges,
                ..trainer.model_options()
            })
            .unwrap();
            let (vocab, merges, _) = trainer.do_train(&wc).unwrap();
            let reference = PipelineBPE::from_config(BpeConfig {
                vocab,
                merges,
                ..trainer.model_options()
            })
            .unwrap();
            for input in [
                "",
                "aaaaa",
                "abab",
                "cababa",
                "测试测试",
                "aaab测c",
                "unknown",
            ] {
                let encode = |model: &PipelineBPE| {
                    let mut scratch = model.init_scratch();
                    let mut output = Vec::new();
                    model
                        .tokenize_pipeline(input, &mut scratch, &mut output)
                        .unwrap();
                    output.iter().map(|t| t.id()).collect::<Vec<_>>()
                };
                assert_eq!(encode(&reloaded), encode(&reference), "{input:?}");
            }
        }
    }

    #[test]
    fn public_affix_training_matches_original_hf_model() {
        let wc = counts(&[("aaaaabcdbaaba", 7), ("baaba中文abab", 3), ("cababa", 0)]);
        for (prefix, suffix, limit) in [(None, Some("a"), None), (Some("ab"), None, Some(9))] {
            let mut trainer = BpeTrainer::builder()
                .vocab_size(80)
                .min_frequency(2)
                .show_progress(false)
                .max_token_length(limit)
                .build();
            trainer.continuing_subword_prefix = prefix.map(str::to_owned);
            trainer.end_of_word_suffix = suffix.map(str::to_owned);
            let expected = trainer.do_train_observed(&wc, |_, _, _| {}).unwrap();
            let got = trainer.do_train(&wc).unwrap();
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn fed_input_and_thread_counts_match() {
        let wc = counts(&[
            ("aaaaaaa", 3),
            ("abababab", 7),
            ("cababa", 2),
            ("测试测试", 4),
        ]);
        let trainer = BpeTrainer::builder()
            .vocab_size(50)
            .show_progress(false)
            .build();
        for threads in [1, 2, 4] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    check(&trainer, &wc);
                });
        }
        let mut trainer = trainer;
        trainer
            .feed(["aa aa", "ab aa"].into_iter(), |s| {
                Ok(s.split_whitespace().map(str::to_owned).collect())
            })
            .unwrap();
        let (vocab, merges, special) = trainer.train_vocab().unwrap();
        let got = trainer.train_vocab_indexed().unwrap();
        assert_eq!(got.vocab, vocab);
        assert_eq!(got.merges, merges);
        assert_eq!(got.special_tokens, special);
    }

    #[test]
    fn wide_weights_and_overflow_guard() {
        let trainer = BpeTrainer::builder()
            .vocab_size(10)
            .show_progress(false)
            .build();
        let got = trainer
            .do_train_indexed(&counts(&[("ab", u32::MAX as u64)]))
            .unwrap();
        assert_eq!(got.trace, vec![((0, 1), u32::MAX as u64, 2)]);
        assert!(
            trainer
                .do_train_indexed(&counts(&[("ab", u64::MAX)]))
                .is_err()
        );
    }
}

#[cfg(test)]
mod greedy_tests {
    use super::*;

    // Independent oracle: recompute every live edge and rewrite whole vectors.
    // It intentionally has no heap, historical posting, or delta maintenance.
    fn greedy(trainer: &BpeTrainer, wc: &AHashMap<CompactString, u64>) -> Vec<(Pair, u64, u32)> {
        let mut ids = AHashMap::new();
        let mut strings = Vec::new();
        trainer.add_special_tokens(&mut ids, &mut strings);
        trainer.compute_alphabet(wc, &mut ids, &mut strings);
        let (words, weights) = trainer.tokenize_words(wc, &mut ids, &mut strings, &None);
        let mut words: Vec<Vec<u32>> = words.iter().map(Word::get_chars).collect();
        let mut trace = Vec::new();
        while ids.len() < trainer.vocab_size {
            let mut counts = AHashMap::<Pair, u64>::new();
            for (word, &weight) in words.iter().zip(&weights) {
                for edge in word.windows(2) {
                    *counts.entry((edge[0], edge[1])).or_default() += weight;
                }
            }
            let Some((pair, count)) = counts
                .into_iter()
                .max_by(|(a, ac), (b, bc)| ac.cmp(bc).then_with(|| b.cmp(a)))
            else {
                break;
            };
            if count == 0 || count < trainer.min_frequency {
                break;
            }
            let token = CompactString::from(format!(
                "{}{}",
                strings[pair.0 as usize], strings[pair.1 as usize]
            ));
            let id = if let Some(&id) = ids.get(&token) {
                id
            } else {
                let id = strings.len() as u32;
                strings.push(token.clone());
                ids.insert(token, id);
                id
            };
            trace.push((pair, count, id));
            for word in &mut words {
                let mut output = Vec::new();
                let mut i = 0;
                while i < word.len() {
                    if i + 1 < word.len() && (word[i], word[i + 1]) == pair {
                        output.push(id);
                        i += 2;
                    } else {
                        output.push(word[i]);
                        i += 1;
                    }
                }
                *word = output;
            }
        }
        trace
    }

    #[test]
    fn recomputing_greedy_oracle_without_affixes_or_length_filter() {
        let mut rng = 1400_u64;
        for _ in 0..250 {
            let mut wc = AHashMap::<CompactString, u64>::new();
            for _ in 0..12 {
                let word: String = (0..30)
                    .map(|_| {
                        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                        ['a', 'b', 'c', '测'][(rng >> 32) as usize % 4]
                    })
                    .collect();
                *wc.entry(word.into()).or_default() += 1 + (rng >> 32) % 9;
            }
            let trainer = BpeTrainer::builder()
                .vocab_size(100)
                .show_progress(false)
                .build();
            let out = trainer.do_train_indexed(&wc).unwrap();
            assert_eq!(out.trace, greedy(&trainer, &wc), "words={wc:?}");
        }
    }
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;

    #[test]
    fn reused_id_resurrects_low_frequency_pair_and_keeps_hf_cohorts() {
        // A synthetic index state, NOT a reachable ordinary-character BPE
        // counterexample (see PAIR_MONOTONICITY.md). It checks the general
        // ledger's handling of aliases, which affixes can actually produce.
        // One word still contains IDs (0,1,2), another
        // already contains the identity 3 that (0,1) will reuse. The old (3,2)
        // frequency is below min_frequency=2 and must remain indexed.
        let mut index = Index::new(
            vec![NONE, 0, 1, 2, NONE, 3, 2, NONE],
            vec![1, 5],
            vec![3, 1],
            vec![1; 4],
            None,
        )
        .unwrap();
        let select = |index: &mut Index| -> Candidate {
            loop {
                let mut top = index.queue.pop().unwrap();
                let count = index.counts[&top.pair] as u64;
                if top.count == count {
                    return top;
                }
                top.count = count;
                index.queue.push(top);
            }
        };
        let mut stats = IndexedTrainingStats::default();
        let first = select(&mut index);
        assert_eq!(first.pair, (0, 1));
        assert!(index.spans.is_none());
        index
            .apply(first, 3, true, usize::MAX, &mut stats, None)
            .unwrap();
        // Same ID, distinct physical spans: the merged occurrence is two
        // characters, while the other word's preexisting occurrence is one.
        assert_eq!(index.span(1), 2);
        assert_eq!(index.span(5), 1);
        assert_eq!(index.inspect(1, (3, 2)).unwrap().after, 4);
        assert_eq!(index.inspect(5, (3, 2)).unwrap().after, 7);
        assert_eq!(index.counts[&(3, 2)], 4);
        let birth_cohort = select(&mut index);
        assert_eq!((birth_cohort.pair, birth_cohort.count), ((3, 2), 4));
        index
            .apply(birth_cohort, 4, true, usize::MAX, &mut stats, None)
            .unwrap();
        assert_eq!(index.corpus[index.pivots[1] as usize], 3);
        // HF's older candidate owns the other word, and keeps its global ledger
        // snapshot even after the first cohort was consumed. It is still used.
        let old_cohort = select(&mut index);
        assert_eq!((old_cohort.pair, old_cohort.count), ((3, 2), 4));
        index
            .apply(old_cohort, 4, true, usize::MAX, &mut stats, None)
            .unwrap();
        assert_eq!(index.corpus[index.pivots[1] as usize], 4);
    }
}
