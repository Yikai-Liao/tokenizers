//! Global pair owners with segmented local-u32 posting streams.
//! Address blocks share high bits; jobs split by occurrence count.
//! Planning reads the old corpus before disjoint writes, including cross-block AA.
use super::small_posting::{PackedPosting, SmallPosting};
use super::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU16, AtomicU32, Ordering as AtomicOrdering};

mod alphabet;
pub(super) fn initialize_alphabet(
    trainer: &BpeTrainer,
    wc: &AHashMap<CompactString, u64>,
    ids: &mut AHashMap<CompactString, u32>,
    strings: &mut Vec<CompactString>,
    workers: usize,
) -> usize {
    alphabet::initialize(trainer, wc, ids, strings, workers)
}
mod candidate_heap;
mod validation_window;
use validation_window::{Frontier, SelectionMode, Window};
mod address_scratch;
mod block_posting;
mod bounded_initial;
mod corpus;
mod flat_commit;
use block_posting::BlockPosting;
mod fused_batch;
mod radix_count;
use candidate_heap::CandidateHeap;
pub(super) mod weight_lookup;

/// Fast execution can discard low-frequency history while replacements are
/// newly activated identities. Stop before the first active collision, so the
/// caller can reconstruct the full HF cohorts from the unchanged input words.
#[derive(Debug)]
pub(super) struct AliasCollision {
    pub(super) stats: Box<IndexedTrainingStats>,
}
impl std::fmt::Display for AliasCollision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("active affix identity requires HF cohort reconstruction")
    }
}
impl std::error::Error for AliasCollision {}

pub(super) fn train_affixed(
    trainer: &BpeTrainer,
    wc: &AHashMap<CompactString, u64>,
    mut config: IndexedParallelConfig,
    options: cohort_parallel::Options,
) -> Result<IndexedTraining> {
    config.narrow_corpus &= options.narrow_corpus;
    if !options.parallel_apply {
        config.initialization_workers =
            Some(config.initialization_workers.unwrap_or(config.workers));
        config.workers = 1;
    }
    if !options.batch_execution {
        config.batch_size = 1;
    }
    let policy = if options.arena_allocator {
        posting_arena::Policy::Auto
    } else {
        posting_arena::Policy::System
    };
    let selection = if options.queue_prefetch {
        SelectionMode::Bulk(4)
    } else {
        SelectionMode::Serial
    };
    let order = if options.sort_weights {
        corpus::Order::WeightSorted
    } else {
        corpus::Order::Original
    };
    train_with_corpus_options(trainer, wc, config, policy, selection, order, Some(options))
}

// General cohorts reuse the same stable pair grouping without its monotone
// ledger pruning. The resulting queue still owns every historical cohort.
pub(super) fn initial_cohorts<C: Slot>(
    corpus: &[C],
    pivots: &[u32],
    weights: &[u64],
    workers: usize,
    sorted_weights: bool,
) -> Result<(AHashMap<Pair, i64>, OctonaryHeap<super::Candidate>)> {
    radix_count::cohorts(corpus, pivots, weights, workers, sorted_weights)
}

pub(super) trait Slot: Default + Send + Sync {
    const NARROW: bool;
    const SHARED: bool = false;
    fn set_shared(&self, _id: u32) {
        panic!("shared writes require atomic slots");
    }
    fn encode(id: u32) -> Self;
    fn from_u32(corpus: Vec<u32>) -> Vec<Self>
    where
        Self: Sized,
    {
        corpus.into_iter().map(Self::encode).collect()
    }
    fn token(&self) -> u32;
    fn set(&mut self, id: u32);
}
impl Slot for u32 {
    fn from_u32(corpus: Vec<u32>) -> Vec<Self> {
        corpus
    }
    const NARROW: bool = false;
    fn encode(id: u32) -> Self {
        id
    }
    fn token(&self) -> u32 {
        *self
    }
    fn set(&mut self, id: u32) {
        *self = id;
    }
}
impl Slot for u16 {
    fn from_u32(corpus: Vec<u32>) -> Vec<Self> {
        // Vec's in-place collect may retain the original wide allocation.
        // Allocate the exact narrow payload so the unsorted path also releases it.
        let mut narrow = Vec::with_capacity(corpus.len());
        narrow.extend(corpus.iter().map(|&id| <Self as Slot>::encode(id)));
        narrow
    }
    const NARROW: bool = true;
    fn encode(id: u32) -> Self {
        debug_assert!(id == NONE || id < u16::MAX as u32);
        if id == NONE { u16::MAX } else { id as u16 }
    }
    fn token(&self) -> u32 {
        if *self == u16::MAX {
            NONE
        } else {
            *self as u32
        }
    }
    fn set(&mut self, id: u32) {
        *self = <u16 as Slot>::encode(id);
    }
}
impl Slot for AtomicU32 {
    const SHARED: bool = true;
    fn set_shared(&self, id: u32) {
        self.store(id, AtomicOrdering::Relaxed);
    }
    const NARROW: bool = false;
    fn encode(id: u32) -> Self {
        Self::new(id)
    }
    fn token(&self) -> u32 {
        self.load(AtomicOrdering::Relaxed)
    }
    fn set(&mut self, id: u32) {
        self.store(id, AtomicOrdering::Relaxed);
    }
}
impl Slot for AtomicU16 {
    const SHARED: bool = true;
    fn set_shared(&self, id: u32) {
        self.store(<u16 as Slot>::encode(id), AtomicOrdering::Relaxed);
    }
    const NARROW: bool = true;
    fn encode(id: u32) -> Self {
        Self::new(<u16 as Slot>::encode(id))
    }
    fn token(&self) -> u32 {
        <u16 as Slot>::token(&self.load(AtomicOrdering::Relaxed))
    }
    fn set(&mut self, id: u32) {
        self.store(<u16 as Slot>::encode(id), AtomicOrdering::Relaxed);
    }
}

trait Offset: Copy + Default + Send + Sync {
    fn encode(offset: usize) -> Self;
    fn index(self) -> usize;
}
impl Offset for u32 {
    fn encode(offset: usize) -> Self {
        u32::try_from(offset).expect("u32 address block")
    }
    fn index(self) -> usize {
        self as usize
    }
}
impl Offset for u16 {
    fn encode(offset: usize) -> Self {
        u16::try_from(offset).expect("u16 address block")
    }
    fn index(self) -> usize {
        self as usize
    }
}

fn key(a: u32, b: u32) -> u64 {
    (u64::from(a) << 32) | u64::from(b)
}
fn pair(k: u64) -> Pair {
    ((k >> 32) as u32, k as u32)
}
#[inline]
fn owner(k: u64, workers: usize) -> usize {
    // Mix both IDs; assignment affects ownership only, never candidate ordering.
    let mixed = ((k ^ (k >> 32)).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 32) as usize;
    if workers.is_power_of_two() {
        mixed & (workers - 1)
    } else {
        mixed % workers
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct Candidate {
    frequency: u64,
    key: u64,
}
impl Ord for Candidate {
    fn cmp(&self, rhs: &Self) -> Ordering {
        self.frequency
            .cmp(&rhs.frequency)
            .then_with(|| rhs.key.cmp(&self.key))
    }
}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        Some(self.cmp(rhs))
    }
}
struct Entry {
    frequency: u64,
    blocks: BlockPosting,
}
#[derive(Default)]
struct Owner {
    entries: AHashMap<u64, Entry>,
    heap: CandidateHeap,
    window: Window,
    truth_checks: usize,
    stale_corrections: usize,
}
impl Owner {
    fn peek_current(&mut self) -> Option<Candidate> {
        loop {
            let top = self.heap.peek()?;
            self.truth_checks += 1;
            match self.entries.get(&top.key) {
                None => {
                    self.stale_corrections += 1;
                    self.heap.pop();
                }
                Some(entry) if entry.frequency != top.frequency => {
                    self.stale_corrections += 1;
                    self.heap.pop();
                    self.heap.push(Candidate {
                        frequency: entry.frequency,
                        key: top.key,
                    });
                }
                Some(_) => return Some(top),
            }
        }
    }
}

struct Block<O: Offset, const INLINE: usize> {
    base: usize,
    postings: AHashMap<u64, PackedPosting<O, INLINE>>,
    // Weight boundaries are local addresses; equal-weight words can share a run.
    pivots: Vec<u32>,
    weights: Vec<u64>,
    previous_weight: u64,
    next_weight_change: usize,
    weight_intervals: bool,
}
impl<O: Offset, const INLINE: usize> Block<O, INLINE> {
    fn new(base: usize, previous_weight: u64) -> Self {
        Self {
            base,
            postings: AHashMap::new(),
            pivots: Vec::new(),
            weights: Vec::new(),
            previous_weight,
            next_weight_change: 0,
            weight_intervals: false,
        }
    }
    fn weight(&self, position: usize, uniform: Option<u64>) -> u64 {
        if let Some(weight) = uniform {
            return weight;
        }
        let local = (position - self.base) as u32;
        let end = self.pivots.partition_point(|&p| p <= local);
        if end == 0 {
            self.previous_weight
        } else {
            self.weights[end - 1]
        }
    }

    // Plans arrive in spatial order. Reuse the current word inside a run,
    // check a short nearby gap, and binary-skip distant words in sparse batches.
    fn weight_forward(&self, position: usize, uniform: Option<u64>, cursor: &mut usize) -> u64 {
        if let Some(weight) = uniform {
            return weight;
        }
        let local = (position - self.base) as u32;
        let near = cursor.saturating_add(8).min(self.pivots.len());
        while *cursor < near && self.pivots[*cursor] <= local {
            *cursor += 1;
        }
        if *cursor < self.pivots.len() && self.pivots[*cursor] <= local {
            *cursor += self.pivots[*cursor..].partition_point(|&p| p <= local);
        }
        if *cursor == 0 {
            self.previous_weight
        } else {
            self.weights[*cursor - 1]
        }
    }
}

struct Rule {
    edge: Pair,
    replacement: u32,
    left_len: usize,
    right_len: usize,
}
impl Rule {
    fn length(&self) -> usize {
        self.left_len + self.right_len
    }
}
#[derive(Clone, Copy)]
struct Plan {
    position: usize,
    rank: usize,
}
impl Plan {
    fn after(self, rules: &[Rule]) -> usize {
        self.position + rules[self.rank].length()
    }
}

struct Output<O: Offset, const INLINE: usize> {
    removed: Vec<AHashMap<u64, u64>>,
    born: Vec<AHashMap<u64, u64>>,
    blocks: AHashMap<usize, AHashMap<u64, PackedPosting<O, INLINE>>>,
    flat_routes: Vec<Route>,
    // [owner][rule * 2 + direction]. A birth's new identity determines its
    // unique bucket; only its neighbor needs to be reduced inside that bucket.
    flat_births: Vec<Vec<Vec<(u32, Group)>>>,
}
struct Group {
    weight: u64,
    head: u32,
    occurrences: u32,
}
impl Default for Group {
    fn default() -> Self {
        Self {
            weight: 0,
            head: NONE,
            occurrences: 0,
        }
    }
}
struct Node {
    position: u32,
    next: u32,
}
#[derive(Default)]
struct Route {
    delta: AHashMap<u64, Group>,
    nodes: Vec<Node>,
    high: address_scratch::HighParts,
}
impl<O: Offset, const INLINE: usize> Output<O, INLINE> {
    fn new(workers: usize, flat: bool) -> Self {
        Self {
            removed: (0..if flat { 0 } else { workers })
                .map(|_| AHashMap::new())
                .collect(),
            born: (0..if flat { 0 } else { workers })
                .map(|_| AHashMap::new())
                .collect(),
            blocks: AHashMap::new(),
            flat_routes: (0..if flat { workers } else { 0 })
                .map(|_| Route::default())
                .collect(),
            flat_births: Vec::new(),
        }
    }
    fn enable_dense_births(&mut self, rules: usize) {
        self.flat_births = (0..self.flat_routes.len())
            .map(|_| (0..rules * 2).map(|_| Vec::new()).collect())
            .collect();
    }
    fn remove(&mut self, k: u64, weight: u64) {
        if !self.flat_routes.is_empty() {
            let o = owner(k, self.flat_routes.len());
            self.flat_routes[o].delta.entry(k).or_default().weight += weight;
            return;
        }
        let workers = self.removed.len();
        *self.removed[owner(k, workers)].entry(k).or_default() += weight;
    }
    fn birth(&mut self, k: u64, position: usize, weight: u64, bits: u8) -> Result<()> {
        if !self.flat_routes.is_empty() {
            let o = owner(k, self.flat_routes.len());
            let route = &mut self.flat_routes[o];
            let head = u32::try_from(route.nodes.len()).map_err(|_| "birth chain exceeds u32")?;
            if head == NONE {
                return Err("birth chain sentinel collision".into());
            }
            let group = route.delta.entry(k).or_default();
            group.weight += weight;
            group.occurrences += 1;
            let position = route.high.push(route.nodes.len(), position);
            route.nodes.push(Node {
                position,
                next: group.head,
            });
            group.head = head;
            return Ok(());
        }
        let workers = self.born.len();
        *self.born[owner(k, workers)].entry(k).or_default() += weight;
        let block = position >> bits;
        self.blocks
            .entry(block)
            .or_default()
            .entry(k)
            .or_default()
            .push(O::encode(position - (block << bits)))
    }
}

// Each split is the next plan's start, after every write in the left branch.
// Rayon joins complete before any subsequent corpus read. No unsafe aliasing,
// corpus atomics, or locks are needed.
fn write_plans<C: Slot>(
    corpus: &mut [C],
    base: usize,
    plans: &[Plan],
    rules: &[Rule],
    jobs: usize,
) {
    if jobs > 1 && plans.len() >= 2048 {
        let mid = plans.len() / 2;
        let cut = plans[mid].position;
        let (left, right) = corpus.split_at_mut(cut - base);
        rayon::join(
            || write_plans(left, base, &plans[..mid], rules, jobs / 2),
            || write_plans(right, cut, &plans[mid..], rules, jobs - jobs / 2),
        );
    } else {
        for &plan in plans {
            let rule = &rules[plan.rank];
            let start = plan.position - base;
            let right = start + rule.left_len;
            let after = right + rule.right_len;
            corpus[start].set(rule.replacement);
            if rule.right_len == 1 {
                corpus[right].set(rule.replacement);
            } else {
                corpus[right].set(NONE);
                corpus[after - 1].set(rule.replacement);
            }
        }
    }
}

// HashMap::capacity is its usable slot count, not the raw bucket count.
pub(super) fn table_bytes(capacity: usize, bucket_bytes: usize) -> usize {
    if capacity == 0 {
        return 0;
    }
    let buckets = (capacity.saturating_mul(8).div_ceil(7)).next_power_of_two();
    buckets * (bucket_bytes + 1) + 16
}

pub(super) fn train(
    trainer: &BpeTrainer,
    wc: &AHashMap<CompactString, u64>,
    config: IndexedParallelConfig,
) -> Result<IndexedTraining> {
    train_with_policy(trainer, wc, config, super::posting_arena::Policy::Auto)
}

pub(super) fn train_with_policy(
    trainer: &BpeTrainer,
    wc: &AHashMap<CompactString, u64>,
    config: IndexedParallelConfig,
    policy: super::posting_arena::Policy,
) -> Result<IndexedTraining> {
    train_with_selection(trainer, wc, config, policy, SelectionMode::Bulk(4))
}

fn train_with_selection(
    trainer: &BpeTrainer,
    wc: &AHashMap<CompactString, u64>,
    config: IndexedParallelConfig,
    policy: super::posting_arena::Policy,
    selection: SelectionMode,
) -> Result<IndexedTraining> {
    train_with_corpus_order(
        trainer,
        wc,
        config,
        policy,
        selection,
        corpus::Order::WeightSorted,
    )
}

fn train_with_corpus_order(
    trainer: &BpeTrainer,
    wc: &AHashMap<CompactString, u64>,
    config: IndexedParallelConfig,
    policy: super::posting_arena::Policy,
    selection: SelectionMode,
    order: corpus::Order,
) -> Result<IndexedTraining> {
    train_with_corpus_options(trainer, wc, config, policy, selection, order, None)
}
fn train_with_corpus_options(
    trainer: &BpeTrainer,
    wc: &AHashMap<CompactString, u64>,
    config: IndexedParallelConfig,
    policy: posting_arena::Policy,
    selection: SelectionMode,
    order: corpus::Order,
    options: Option<cohort_parallel::Options>,
) -> Result<IndexedTraining> {
    let begin = Instant::now();
    let mut ids = AHashMap::with_capacity(trainer.vocab_size);
    let mut strings = Vec::with_capacity(trainer.vocab_size);
    trainer.add_special_tokens(&mut ids, &mut strings);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.workers)
        .build()?;
    let initialization_pool = config
        .initialization_workers
        .filter(|&n| n != config.workers)
        .map(|n| rayon::ThreadPoolBuilder::new().num_threads(n).build())
        .transpose()?;
    let alphabet_begin = Instant::now();
    let alphabet_scratch_bytes = if options.is_some_and(|o| !o.parallel_alphabet) {
        trainer.compute_alphabet(wc, &mut ids, &mut strings);
        0
    } else {
        initialization_pool.as_ref().unwrap_or(&pool).install(|| {
            alphabet::initialize(
                trainer,
                wc,
                &mut ids,
                &mut strings,
                config.initialization_workers.unwrap_or(config.workers),
            )
        })
    };
    let alphabet_ms = alphabet_begin.elapsed().as_secs_f64() * 1000.0;
    let decorated = if let Some(options) = options {
        Some(PreparedCorpus::tokenize_with_cache(
            trainer,
            wc,
            &mut ids,
            &mut strings,
            &None,
            options.character_cache,
            options
                .parallel_measure
                .then_some(initialization_pool.as_ref().unwrap_or(&pool)),
        )?)
    } else {
        None
    };
    // Select the final slot type before emitting any corpus. Reserved IDs and
    // forced alphabet count towards this bound, not just the requested vocabulary.
    let narrow = config.narrow_corpus && strings.len().max(trainer.vocab_size) <= u16::MAX as usize;
    let session = super::posting_arena::Session::new(&pool, initialization_pool.as_ref());
    let mut result = match (config.atomic_corpus, narrow, config.posting_block_bits) {
        (false, true, 16) => train_typed::<u16, u16, 4>(
            trainer,
            wc,
            config,
            ids,
            strings,
            begin,
            alphabet_ms,
            alphabet_scratch_bytes,
            &pool,
            initialization_pool.as_ref(),
            policy,
            selection,
            order,
            decorated,
            options,
        ),
        (true, true, 16) => train_typed::<AtomicU16, u16, 4>(
            trainer,
            wc,
            config,
            ids,
            strings,
            begin,
            alphabet_ms,
            alphabet_scratch_bytes,
            &pool,
            initialization_pool.as_ref(),
            policy,
            selection,
            order,
            decorated,
            options,
        ),
        (false, true, 32) => train_typed::<u16, u32, 2>(
            trainer,
            wc,
            config,
            ids,
            strings,
            begin,
            alphabet_ms,
            alphabet_scratch_bytes,
            &pool,
            initialization_pool.as_ref(),
            policy,
            selection,
            order,
            decorated,
            options,
        ),
        (true, true, 32) => train_typed::<AtomicU16, u32, 2>(
            trainer,
            wc,
            config,
            ids,
            strings,
            begin,
            alphabet_ms,
            alphabet_scratch_bytes,
            &pool,
            initialization_pool.as_ref(),
            policy,
            selection,
            order,
            decorated,
            options,
        ),
        (false, false, 16) => train_typed::<u32, u16, 4>(
            trainer,
            wc,
            config,
            ids,
            strings,
            begin,
            alphabet_ms,
            alphabet_scratch_bytes,
            &pool,
            initialization_pool.as_ref(),
            policy,
            selection,
            order,
            decorated,
            options,
        ),
        (true, false, 16) => train_typed::<AtomicU32, u16, 4>(
            trainer,
            wc,
            config,
            ids,
            strings,
            begin,
            alphabet_ms,
            alphabet_scratch_bytes,
            &pool,
            initialization_pool.as_ref(),
            policy,
            selection,
            order,
            decorated,
            options,
        ),
        (false, false, 32) => train_typed::<u32, u32, 2>(
            trainer,
            wc,
            config,
            ids,
            strings,
            begin,
            alphabet_ms,
            alphabet_scratch_bytes,
            &pool,
            initialization_pool.as_ref(),
            policy,
            selection,
            order,
            decorated,
            options,
        ),
        (true, false, 32) => train_typed::<AtomicU32, u32, 2>(
            trainer,
            wc,
            config,
            ids,
            strings,
            begin,
            alphabet_ms,
            alphabet_scratch_bytes,
            &pool,
            initialization_pool.as_ref(),
            policy,
            selection,
            order,
            decorated,
            options,
        ),
        _ => unreachable!("validated configuration"),
    };
    let allocations = session.finish();
    if let Ok(training) = &mut result {
        training.stats.posting_allocation_policy = policy.label();
        training.stats.posting_arena_cutoff_bytes = policy.cutoff(training.stats.initial_edges);
        training.stats.posting_allocations = allocations;
    } else if let Err(error) = &mut result {
        if let Some(collision) = error.downcast_mut::<AliasCollision>() {
            collision.stats.posting_allocation_policy = policy.label();
            collision.stats.posting_arena_cutoff_bytes =
                policy.cutoff(collision.stats.initial_edges);
            collision.stats.posting_allocations = allocations;
        }
    }
    result
}

fn train_typed<C: Slot, O: Offset, const INLINE: usize>(
    trainer: &BpeTrainer,
    wc: &AHashMap<CompactString, u64>,
    config: IndexedParallelConfig,
    ids: AHashMap<CompactString, u32>,
    strings: Vec<CompactString>,
    begin: Instant,
    alphabet_ms: f64,
    alphabet_scratch_bytes: usize,
    pool: &rayon::ThreadPool,
    initialization_pool: Option<&rayon::ThreadPool>,
    policy: super::posting_arena::Policy,
    selection: SelectionMode,
    order: corpus::Order,
    decorated: Option<PreparedCorpus>,
    options: Option<cohort_parallel::Options>,
) -> Result<IndexedTraining> {
    // Keep the coordinator inside this pool too: small serial rounds then do
    // not pay a caller->worker handoff at every stage.
    pool.install(|| {
        train_in_pool_options::<C, O, INLINE>(
            trainer,
            wc,
            config,
            ids,
            strings,
            begin,
            alphabet_ms,
            alphabet_scratch_bytes,
            pool,
            initialization_pool,
            policy,
            selection,
            order,
            decorated,
            options,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn train_in_pool<C: Slot, O: Offset, const INLINE: usize>(
    trainer: &BpeTrainer,
    wc: &AHashMap<CompactString, u64>,
    config: IndexedParallelConfig,
    ids: AHashMap<CompactString, u32>,
    strings: Vec<CompactString>,
    begin: Instant,
    alphabet_ms: f64,
    alphabet_scratch_bytes: usize,
    pool: &rayon::ThreadPool,
    initialization_pool: Option<&rayon::ThreadPool>,
    policy: super::posting_arena::Policy,
    selection: SelectionMode,
    order: corpus::Order,
) -> Result<IndexedTraining> {
    train_in_pool_options::<C, O, INLINE>(
        trainer,
        wc,
        config,
        ids,
        strings,
        begin,
        alphabet_ms,
        alphabet_scratch_bytes,
        pool,
        initialization_pool,
        policy,
        selection,
        order,
        None,
        None,
    )
}
#[allow(clippy::too_many_arguments)]
fn train_in_pool_options<C: Slot, O: Offset, const INLINE: usize>(
    trainer: &BpeTrainer,
    wc: &AHashMap<CompactString, u64>,
    config: IndexedParallelConfig,
    mut ids: AHashMap<CompactString, u32>,
    mut strings: Vec<CompactString>,
    begin: Instant,
    alphabet_ms: f64,
    alphabet_scratch_bytes: usize,
    pool: &rayon::ThreadPool,
    initialization_pool: Option<&rayon::ThreadPool>,
    policy: super::posting_arena::Policy,
    selection: SelectionMode,
    order: corpus::Order,
    decorated: Option<PreparedCorpus>,
    options: Option<cohort_parallel::Options>,
) -> Result<IndexedTraining> {
    let bits = config.posting_block_bits;
    let block_size = 1_usize
        .checked_shl(bits as u32)
        .ok_or("posting blocks require 64-bit usize")?;
    let constructing_pool = initialization_pool.unwrap_or(pool);
    let guarded = decorated.is_some();
    let prepared = if let Some(input) = decorated {
        constructing_pool.install(|| corpus::from_decorated::<C, O, INLINE>(input, bits, order))?
    } else {
        constructing_pool.install(|| {
            corpus::build::<C, O, INLINE>(
                wc,
                &ids,
                strings.len(),
                trainer.limit_alphabet.is_none(),
                bits,
                config.initialization_workers.unwrap_or(config.workers),
                order,
            )
        })?
    };
    let corpus::Prepared {
        slots: mut corpus,
        mut lengths,
        mut blocks,
        uniform,
        symbols: initial_symbols,
        edges: initial_edges,
        weighted_edges,
        timings,
        character_table_bytes,
        word_reference_bytes,
        temporary_weight_bytes,
        padding_slots: prepared_padding_slots,
        oversized_words: prepared_oversized_words,
        padding_plan_bytes: prepared_padding_plan_bytes,
    } = prepared;
    weight_lookup::link_intervals(&mut blocks);
    super::posting_arena::configure(pool, initialization_pool, policy.cutoff(initial_edges));
    // Fresh activation holds throughout a guarded epoch: an active canonical
    // collision returns before consuming the candidate or writing its batch.
    // Each merge deletes boundaries, so no historical pair frequency exceeds
    // the checked initial weighted edge mass. Reserved and forced alphabet IDs
    // are already included in strings; future IDs stop at the vocab target.
    let packed_heap = (supports_compact(trainer) || guarded)
        && options.is_none_or(|o| o.packed_queue)
        && weighted_edges <= u32::MAX as u64
        && strings.len().max(trainer.vocab_size) <= u16::MAX as usize + 1;
    let mut owners: Vec<Owner> = (0..config.workers).map(|_| Owner::default()).collect();
    let tokenize_ms = begin.elapsed().as_secs_f64() * 1000.0;
    let floor = trainer.min_frequency.max(1);
    let radix_eligible =
        options.is_none_or(|o| o.initial_grouped) && lengths.len() <= u16::MAX as usize + 1;
    let mut initial_weight_lookup: Option<weight_lookup::WeightLookups> = None;
    let mut initialize = || -> Result<IndexedTrainingStats> {
        let initial_route_ms;
        let initial_count_ms;
        let mut radix_metrics = radix_count::Metrics::default();
        let mut initial_weight_lookup_ms = 0.0;
        let initial_summary_waves = 0;
        let initial_summary_buffer_bytes = 0;
        let peak_initial_summary_buffer_bytes = 0;
        let initial_bounded_tiles = 0;
        let initial_bounded_groups = 0;
        let initial_bounded_hash_edges = 0;
        let initial_bounded_sort_buffer_bound_bytes = 0;
        if radix_eligible {
            let lookup_begin = Instant::now();
            initial_weight_lookup = (uniform.is_none() && options.is_none_or(|o| o.weight_lookup))
                .then(|| weight_lookup::WeightLookups::new(&blocks, corpus.len(), bits));
            initial_weight_lookup_ms = lookup_begin.elapsed().as_secs_f64() * 1000.0;
            radix_metrics = radix_count::initialize_segmented(
                &corpus,
                &blocks,
                uniform,
                initial_weight_lookup.as_ref(),
                config.workers,
                floor,
                &mut owners,
                bits,
            )?;
            initial_route_ms = radix_metrics.route_ms;
            initial_count_ms = radix_metrics.count_ms;
        } else {
            let route_begin = Instant::now();
            // Bounded spatial waves avoid a whole-corpus u64 occurrence buffer.
            for (wave, slots) in corpus.chunks(1 << 20).enumerate() {
                let base = wave * (1 << 20);
                let mut routes = (0..config.workers).map(|_| Vec::new()).collect::<Vec<_>>();
                for (i, slot) in slots.iter().enumerate() {
                    let p = base + i;
                    if p + 1 == corpus.len() {
                        break;
                    }
                    let a = slot.token();
                    let b = corpus[p + 1].token();
                    if a != NONE && b != NONE {
                        routes[owner(key(a, b), config.workers)].push((key(a, b), p));
                    }
                }
                owners
                    .par_iter_mut()
                    .zip(routes.into_par_iter())
                    .map(|(ledger, records)| -> Result<()> {
                        let mut grouped = AHashMap::<u64, (u64, Vec<usize>)>::new();
                        for (k, p) in records {
                            let group = grouped.entry(k).or_default();
                            group.0 += blocks[p >> bits].weight(p, uniform);
                            group.1.push(p);
                        }
                        for (k, (frequency, positions)) in grouped {
                            let positions = positions.as_slice();
                            let mut cursor = positions.len();
                            let posting = BlockPosting::from_reversed(cursor, bits, move || {
                                cursor -= 1;
                                positions[cursor]
                            })?;
                            match ledger.entries.entry(k) {
                                std::collections::hash_map::Entry::Vacant(e) => {
                                    e.insert(Entry {
                                        frequency,
                                        blocks: posting,
                                    });
                                }
                                std::collections::hash_map::Entry::Occupied(mut e) => {
                                    e.get_mut().frequency += frequency;
                                    e.get_mut().blocks.append(posting)?;
                                }
                            }
                        }
                        Ok(())
                    })
                    .collect::<Result<Vec<_>>>()?;
            }
            initial_route_ms = route_begin.elapsed().as_secs_f64() * 1000.0;
            initial_count_ms = 0.0;
        }
        let mut stats = IndexedTrainingStats {
            initial_symbols,
            tokenize_ms,
            alphabet_ms,
            alphabet_scratch_bytes,
            character_table_bytes,
            corpus_measure_ms: timings.measure_ms,
            corpus_sort_ms: timings.sort_ms,
            corpus_sort_buffer_bytes: timings.sort_buffer_bytes,
            corpus_stable_weight_sort: order == corpus::Order::WeightStable,
            corpus_weight_order: if order == corpus::Order::Original {
                "original"
            } else {
                "weight_sorted"
            },
            corpus_padding_slots: prepared_padding_slots,
            corpus_oversized_words: prepared_oversized_words,
            corpus_padding_plan_bytes: prepared_padding_plan_bytes,
            corpus_word_reference_bytes: word_reference_bytes,
            corpus_temporary_weight_bytes: temporary_weight_bytes,
            weight_interval_count: blocks.iter().map(|b| b.pivots.len()).sum(),
            corpus_allocate_ms: timings.allocate_ms,
            corpus_fill_ms: timings.fill_ms,
            initial_route_ms,
            initial_count_ms,
            initial_bounded_tiles,
            initial_bounded_groups,
            initial_bounded_hash_edges,
            initial_bounded_sort_buffer_bound_bytes,
            initial_summary_waves,
            initial_summary_buffer_bytes,
            peak_initial_summary_buffer_bytes,
            initial_count_backend: if radix_eligible {
                "stable_radix16"
            } else {
                "spatial_owner_hash"
            },
            initial_weight_lookup_ms,
            initial_weight_lookup_bytes: initial_weight_lookup
                .as_ref()
                .map_or(0, |l| l.iter().map(|v| v.bytes()).sum()),
            initial_route_compact_ms: radix_metrics.compact_ms,
            initial_radix_sort_ms: radix_metrics.sort_ms,
            initial_group_count_ms: radix_metrics.group_ms,
            initial_posting_install_ms: radix_metrics.install_ms,
            initial_route_buffer_bytes: radix_metrics.route_bytes,
            peak_initial_route_buffer_bytes: radix_metrics.peak_route_bytes,
            initial_radix_scratch_bytes: radix_metrics.scratch_bytes,
            initial_group_buffer_bytes: radix_metrics.group_bytes,
            pruned_pairs: radix_metrics.pruned,

            monotone_pairs: true,
            alias_guarded: guarded,
            corpus_slot_bytes: std::mem::size_of::<C>(),
            workers: config.workers,
            initialization_workers: config.initialization_workers.unwrap_or(config.workers),
            atomic_corpus: config.atomic_corpus,
            layout: if bits == 32 {
                "parallel_u32_planes32"
            } else {
                "parallel_u32_planes16"
            },
            ..Default::default()
        };
        let heap_begin = Instant::now();
        stats.pruned_pairs += owners
            .par_iter_mut()
            .map(|ledger| {
                let old = ledger.entries.len();
                ledger.entries.retain(|_, entry| entry.frequency >= floor);
                ledger.heap = CandidateHeap::new(
                    ledger.entries.iter().map(|(&key, e)| Candidate {
                        key,
                        frequency: e.frequency,
                    }),
                    packed_heap,
                );
                ledger.prepare_window(selection);
                old - ledger.entries.len()
            })
            .sum::<usize>();
        stats.initial_heap_ms = heap_begin.elapsed().as_secs_f64() * 1000.0;
        stats.initial_corpus_bytes = corpus.capacity() * std::mem::size_of::<C>()
            + lengths.capacity() * std::mem::size_of::<usize>()
            + blocks
                .iter()
                .map(|b| b.pivots.capacity() * 4 + b.weights.capacity() * 8)
                .sum::<usize>();
        stats.initial_slots = corpus.len();
        stats.initial_edges = initial_edges;
        stats.initial_pairs = owners.iter().map(|o| o.entries.len()).sum();
        stats.initial_blocks = blocks.len();
        stats.initial_block_pairs = owners
            .iter()
            .flat_map(|o| o.entries.values())
            .map(|e| e.blocks.run_count())
            .sum();
        stats.initial_slot_bytes = corpus.capacity() * std::mem::size_of::<C>();
        stats.initial_length_bytes = lengths.capacity() * std::mem::size_of::<usize>();
        stats.initial_weight_bytes =
            stats.initial_corpus_bytes - stats.initial_slot_bytes - stats.initial_length_bytes;
        stats.initial_posting_bytes = owners
            .iter()
            .flat_map(|o| o.entries.values())
            .map(|e| e.blocks.payload_bytes())
            .sum();
        stats.initial_pair_table_bytes = owners
            .iter()
            .map(|o| table_bytes(o.entries.capacity(), 8 + std::mem::size_of::<Entry>()))
            .sum();
        stats.initial_block_table_bytes =
            blocks.capacity() * std::mem::size_of::<Block<O, INLINE>>();
        stats.initial_directory_bytes = owners
            .iter()
            .flat_map(|o| o.entries.values())
            .map(|e| e.blocks.directory_bytes())
            .sum();
        stats.initial_heap_bytes = owners.iter().map(|o| o.heap.capacity_bytes()).sum();
        stats.corpus_bytes = stats.initial_corpus_bytes;
        stats.posting_bytes = stats.initial_posting_bytes;
        stats.initialize_ms = begin.elapsed().as_secs_f64() * 1000.0;
        Ok(stats)
    };
    let mut stats = if let Some(pool) = initialization_pool {
        pool.install(initialize)?
    } else {
        initialize()?
    };
    let begin = Instant::now();
    let max_length = trainer.max_token_length.unwrap_or(usize::MAX);
    let mut merges = Vec::new();
    #[cfg(test)]
    let mut trace = Vec::new();
    let lookup_begin = Instant::now();
    let weight_lookup = initial_weight_lookup;
    stats.weight_lookup_build_ms = lookup_begin.elapsed().as_secs_f64() * 1000.0;
    stats.weight_lookup_bytes = weight_lookup
        .as_ref()
        .map_or(0, |l| l.iter().map(|v| v.bytes()).sum());
    stats.weight_one_bucket_count = weight_lookup
        .as_ref()
        .map_or(0, |l| l.iter().map(|v| v.one_bucket_count()).sum());
    stats.weight_bucket_count = weight_lookup
        .as_ref()
        .map_or(0, |l| l.iter().map(|v| v.bucket_count()).sum());
    stats.weight_one_bucket_bytes = weight_lookup
        .as_ref()
        .map_or(0, |l| l.iter().map(|v| v.one_bucket_bytes()).sum());
    let mut frontier = Frontier::default();
    let mut committed_merges = 0;
    while ids.len() < trainer.vocab_size {
        let stage = Instant::now();
        let cap = config.batch_size.min(trainer.vocab_size - ids.len());
        let mut rules = Vec::<Rule>::new();
        let mut heads = AHashSet::new();
        let mut tails = AHashSet::new();
        let mut selected_postings = Vec::new();
        frontier.begin_epoch(&mut owners, selection);
        while rules.len() < cap {
            let best = frontier.best(&mut owners, selection);
            let Some((o, top)) = best else {
                break;
            };
            let edge = pair(top.key);
            if !rules.is_empty()
                && (edge.0 == edge.1 || tails.contains(&edge.0) || heads.contains(&edge.1))
            {
                break;
            }
            let mut token = CompactString::with_capacity(
                strings[edge.0 as usize].len() + strings[edge.1 as usize].len(),
            );
            token.push_str(&strings[edge.0 as usize]);
            let right = strings[edge.1 as usize].as_str();
            let right = if guarded {
                trainer
                    .continuing_subword_prefix
                    .as_deref()
                    .and_then(|prefix| right.strip_prefix(prefix))
                    .unwrap_or(right)
            } else {
                right
            };
            token.push_str(right);
            let reserved = ids.get(&token).copied();
            if guarded && reserved.is_some_and(|id| lengths[id as usize] != 0) {
                // No rewrite involving this reused live ID has occurred. Every
                // completed fast batch used newly activated identities only.
                stats.merge_ms = begin.elapsed().as_secs_f64() * 1000.0;
                stats.speculative_selected_merges = merges.len();
                stats.speculative_applied_merges = committed_merges;
                return Err(AliasCollision {
                    stats: Box::new(stats),
                }
                .into());
            }
            // A reserved canonical ID can win an equal-frequency birth tie
            // before its old witness. Use one rule for that activation.
            if reserved.is_some() && !rules.is_empty() {
                break;
            }
            frontier.consume(&mut owners, o, selection);
            let entry = owners[o].entries.remove(&top.key).unwrap();
            let length = lengths[edge.0 as usize]
                .checked_add(lengths[edge.1 as usize])
                .ok_or("token span exceeds usize")?;
            let replacement = if let Some(id) = reserved {
                debug_assert_eq!(lengths[id as usize], 0);
                lengths[id as usize] = length;
                stats.reused_ids += 1;
                id
            } else {
                let id = u32::try_from(strings.len()).map_err(|_| "vocabulary exceeds u32")?;
                if id == NONE {
                    return Err("token ID collides with separator".into());
                }
                ids.insert(token.clone(), id);
                strings.push(token);
                lengths.push(length);
                id
            };
            // Directory ownership ends here. Selected local postings are freed
            // during planning, before allocating the next generation's postings.
            stats.posting_visits += entry.blocks.len();
            selected_postings.push(entry.blocks);
            rules.push(Rule {
                edge,
                replacement,
                left_len: lengths[edge.0 as usize],
                right_len: lengths[edge.1 as usize],
            });
            heads.insert(edge.0);
            tails.insert(edge.1);
            merges.push(edge);
            #[cfg(test)]
            trace.push((edge, top.frequency, replacement));
            if edge.0 == edge.1 || reserved.is_some() {
                break;
            }
        }
        if !matches!(selection, SelectionMode::Serial) {
            for ledger in &mut owners {
                ledger.end_selection();
            }
        }
        if rules.is_empty() {
            break;
        }
        stats.select_ms += stage.elapsed().as_secs_f64() * 1000.0;
        let stage = Instant::now();
        stats.batch_rounds += 1;
        stats.max_batch_rules = stats.max_batch_rules.max(rules.len());
        let outputs: Vec<Output<O, INLINE>> = if options.is_none_or(|o| o.fused_batch)
            && C::SHARED
            && rules[0].edge.0 != rules[0].edge.1
        {
            let prepared = pool.install(|| {
                fused_batch::prepare_with_grouping(
                    &corpus,
                    &rules,
                    &selected_postings,
                    &blocks,
                    &lengths,
                    uniform,
                    weight_lookup.as_ref(),
                    max_length,
                    config.workers,
                    options.is_none_or(|o| o.grouped_tail),
                    bits,
                )
            })?;
            stats.fused_batches += 1;
            stats.peak_selected_lookup_bytes = stats
                .peak_selected_lookup_bytes
                .max(prepared.selected_bytes);
            stats.peak_valid_start_bytes = stats.peak_valid_start_bytes.max(prepared.valid_bytes);
            stats.peak_prepare_aggregate_bytes = stats
                .peak_prepare_aggregate_bytes
                .max(prepared.aggregate_bytes);
            let elapsed = stage.elapsed().as_secs_f64() * 1000.0;
            // Filter and neighbor deltas share this phase in the fused path.
            stats.fused_prepare_ms += elapsed;
            stats.delta_ms += elapsed;
            drop(selected_postings);
            let stage = Instant::now();
            pool.install(|| prepared.apply(&corpus, &rules));
            stats.rewrite_ms += stage.elapsed().as_secs_f64() * 1000.0;
            prepared.outputs
        } else {
            let chunks: Vec<Vec<Plan>> = pool.install(|| {
                let corpus = &corpus;
                selected_postings
                    .par_iter()
                    .enumerate()
                    .map(|(rank, posting)| {
                        let rule = &rules[rank];
                        posting
                            .iter(bits)
                            .filter_map(|p| {
                                let right = p + rule.left_len;
                                (corpus[p].token() == rule.edge.0
                                    && right < corpus.len()
                                    && corpus[right].token() == rule.edge.1)
                                    .then_some(Plan { position: p, rank })
                            })
                            .collect()
                    })
                    .collect()
            });
            drop(selected_postings);
            let count: usize = chunks.iter().map(Vec::len).sum();
            let mut plans = Vec::with_capacity(count);
            for mut chunk in chunks {
                plans.append(&mut chunk);
            }
            // Posting producers append ordered final boundary positions. A single
            // rule keeps that order, including AA; only mixed rules need sorting.
            if rules.len() > 1 {
                pool.install(|| plans.par_sort_unstable_by_key(|p| p.position));
            }
            // AA is the only self-overlapping rule, and always forms a single-rule
            // batch. Only O(chunks) parity propagation is serial; summaries use logarithmic boundary checks and local selection runs
            // in parallel, even when a long run spans address blocks or empty chunks.
            if rules[0].edge.0 == rules[0].edge.1 {
                let length = rules[0].left_len;
                let summaries: Vec<_> = pool.install(|| {
                    plans
                        .par_chunks(4096)
                        .map(|chunk| {
                            super::aa_parity::summarize_by(
                                chunk.len(),
                                |i| chunk[i].position,
                                length,
                            )
                        })
                        .collect()
                });
                let incoming = super::aa_parity::incoming_parities(&summaries, length);
                let selected: Vec<Vec<Plan>> = pool.install(|| {
                    plans
                        .par_chunks(4096)
                        .zip(incoming.par_iter())
                        .map(|(chunk, &odd)| {
                            let mut valid = Vec::with_capacity(chunk.len().div_ceil(2));
                            super::aa_parity::for_each_selected(
                                chunk.iter().map(|p| p.position),
                                length,
                                odd,
                                |position| valid.push(Plan { position, rank: 0 }),
                            );
                            valid
                        })
                        .collect()
                });
                plans.clear();
                for mut chunk in selected {
                    plans.append(&mut chunk);
                }
            }
            debug_assert!(
                plans
                    .windows(2)
                    .all(|w| w[0].after(&rules) <= w[1].position)
            );
            stats.plan_ms += stage.elapsed().as_secs_f64() * 1000.0;
            let stage = Instant::now();
            // AA's overlapping occurrences are visited, but are not stale records.
            // One route buffer per worker, as in the prototype. A posting task is
            // not a route owner: returning a new map for every 4096 occurrences
            // duplicates groups and allocations throughout a large batch.
            // At most two birth nodes per plan; local u32 links stay bounded.
            let output_chunk = plans.len().div_ceil(config.workers).clamp(4096, 1 << 26);
            let outputs: Vec<Output<O, INLINE>> = pool.install(|| {
                plans
                    .par_chunks(output_chunk)
                    .enumerate()
                    .map(|(chunk, local)| -> Result<_> {
                        let mut output = Output::new(config.workers, true);
                        let mut weight_block = usize::MAX;
                        let mut weight_cursor = 0;
                        for (j, &plan) in local.iter().enumerate() {
                            let i = chunk * output_chunk + j;
                            let rule = &rules[plan.rank];
                            let p = plan.position;
                            let after = plan.after(&rules);
                            let b = p >> bits;
                            if b != weight_block {
                                weight_block = b;
                                weight_cursor = 0;
                            }
                            let weight = blocks[b].weight_forward(p, uniform, &mut weight_cursor);
                            let prior = corpus[p - 1].token();
                            let left_selected = i > 0 && plans[i - 1].after(&rules) == p;
                            if prior != NONE && !left_selected {
                                let before = p - lengths[prior as usize];
                                output.remove(key(prior, rule.edge.0), weight);
                                if lengths[prior as usize] + rule.length() < max_length {
                                    output.birth(
                                        key(prior, rule.replacement),
                                        before,
                                        weight,
                                        bits,
                                    )?;
                                }
                            }
                            let next = corpus[after].token();
                            if next != NONE {
                                output.remove(key(rule.edge.1, next), weight);
                                let final_next =
                                    if plans.get(i + 1).is_some_and(|q| q.position == after) {
                                        rules[plans[i + 1].rank].replacement
                                    } else {
                                        next
                                    };
                                if rule.length() + lengths[final_next as usize] < max_length {
                                    output.birth(
                                        key(rule.replacement, final_next),
                                        p,
                                        weight,
                                        bits,
                                    )?;
                                }
                            }
                        }
                        Ok(output)
                    })
                    .collect::<Result<Vec<_>>>()
            })?;
            stats.delta_ms += stage.elapsed().as_secs_f64() * 1000.0;
            let stage = Instant::now();
            pool.install(|| write_plans(&mut corpus, 0, &plans, &rules, config.workers));
            stats.rewrite_ms += stage.elapsed().as_secs_f64() * 1000.0;
            outputs
        };
        stats.peak_birth_bytes = stats.peak_birth_bytes.max(
            outputs
                .iter()
                .flat_map(|o| &o.flat_routes)
                .map(|r| r.nodes.capacity() * std::mem::size_of::<Node>() + r.high.bytes())
                .sum(),
        );
        let stage = Instant::now();
        let commits: Vec<_> = pool.install(|| {
            owners
                .par_iter_mut()
                .enumerate()
                .map(|(o, ledger)| -> Result<_> {
                    let mut retired = Vec::new();
                    if !outputs[0].flat_births.is_empty() {
                        let result = flat_commit::dense(
                            &outputs,
                            o,
                            ledger,
                            &rules,
                            lengths.len(),
                            floor,
                            bits,
                        )?;
                        ledger.prepare_window(selection);
                        return Ok(result);
                    }
                    {
                        let mut born = AHashMap::<u64, (u64, usize)>::new();
                        for output in &outputs {
                            for (&k, group) in &output.flat_routes[o].delta {
                                if group.occurrences != 0 {
                                    let entry = born.entry(k).or_default();
                                    entry.0 += group.weight;
                                    entry.1 = entry
                                        .1
                                        .checked_add(group.occurrences as usize)
                                        .ok_or("birth posting count exceeds usize")?;
                                } else if let Some(entry) = ledger.entries.get_mut(&k) {
                                    entry.frequency = entry
                                        .frequency
                                        .checked_sub(group.weight)
                                        .ok_or("old pair frequency underflow")?;
                                    if entry.frequency < floor {
                                        ledger.entries.remove(&k);
                                        retired.push((k, SmallPosting::default()));
                                    }
                                }
                            }
                        }
                        let mut dropped = 0;
                        for (k, (frequency, count)) in born {
                            if frequency < floor {
                                dropped += 1;
                                continue;
                            }
                            debug_assert!(!ledger.entries.contains_key(&k));
                            ledger.entries.insert(
                                k,
                                Entry {
                                    frequency,
                                    blocks: BlockPosting::with_capacity(count)?,
                                },
                            );
                            ledger.heap.push(Candidate { key: k, frequency });
                        }
                        for output in &outputs {
                            let route = &output.flat_routes[o];
                            for (&k, group) in &route.delta {
                                if group.head == NONE {
                                    continue;
                                }
                                let Some(entry) = ledger.entries.get_mut(&k) else {
                                    continue;
                                };
                                let start = entry.blocks.len();
                                let mut head = group.head;
                                entry.blocks.append_reversed_reserved_at(
                                    group.occurrences as usize,
                                    bits,
                                    move || {
                                        let index = head as usize;
                                        let node = &route.nodes[index];
                                        head = node.next;
                                        route.high.address(index, node.position)
                                    },
                                )?;
                                debug_assert!(
                                    entry
                                        .blocks
                                        .iter(bits)
                                        .skip(start.saturating_sub(1))
                                        .collect::<Vec<_>>()
                                        .windows(2)
                                        .all(|w| w[0] < w[1])
                                );
                            }
                        }
                        ledger.prepare_window(selection);
                        Ok((retired, AHashSet::new(), dropped))
                    }
                })
                .collect::<Result<Vec<_>>>()
        })?;
        stats.commit_ms += stage.elapsed().as_secs_f64() * 1000.0;
        let stage = Instant::now();
        for (retired, _, dropped) in &commits {
            stats.pruned_pairs += retired.len() + dropped;
        }
        stats.route_ms += stage.elapsed().as_secs_f64() * 1000.0;
        committed_merges = merges.len();
    }
    stats.queue_selection_mode = selection.label();
    stats.queue_owner_probes = frontier.owner_probes;
    stats.queue_leader_updates = frontier.leader_updates;
    for ledger in &owners {
        stats.queue_truth_checks += ledger.truth_checks;
        stats.queue_stale_corrections += ledger.stale_corrections;
        stats.queue_prefetched += ledger.window.prefetched;
        stats.queue_unused_restored += ledger.window.restored;
        stats.queue_serial_refills += ledger.window.serial_refills;
        stats.queue_worker_prefetch_ms += ledger.window.worker_ms;
    }
    stats.merge_ms = begin.elapsed().as_secs_f64() * 1000.0;
    Ok(IndexedTraining {
        vocab: ids.into_iter().map(|(s, id)| (s.to_string(), id)).collect(),
        merges: merges
            .into_iter()
            .map(|(a, b)| {
                (
                    strings[a as usize].to_string(),
                    strings[b as usize].to_string(),
                )
            })
            .collect(),
        special_tokens: trainer.special_tokens.clone(),
        stats,
        #[cfg(test)]
        trace,
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn selection_modes_keep_weighted_reserved_and_length_limited_full_traces() {
        let words = [
            ("ababcd中aa".repeat(300), 3),
            ("cdab文abab".repeat(250), u32::MAX as u64 + 1),
            ("aaaaabbbb中".repeat(200), 0),
            ("abcdabcdef".repeat(220), 1),
        ]
        .into_iter()
        .map(|(w, n)| (w.into(), n))
        .collect();
        let trainer = BpeTrainer::builder()
            .vocab_size(100)
            .min_frequency(2)
            .max_token_length(Some(8))
            .show_progress(false)
            .special_tokens(vec![AddedToken::from("ab", true)])
            .build();
        for bits in [16, 32] {
            for workers in [1, 4, 32] {
                let config = IndexedParallelConfig {
                    workers,
                    initialization_workers: Some(1),
                    posting_block_bits: bits,
                    atomic_corpus: true,
                    narrow_corpus: false,
                    ..Default::default()
                };
                let expected = train_with_corpus_order(
                    &trainer,
                    &words,
                    config,
                    super::super::posting_arena::Policy::Auto,
                    SelectionMode::Serial,
                    corpus::Order::Original,
                )
                .unwrap();
                let stable = train_with_corpus_order(
                    &trainer,
                    &words,
                    config,
                    super::super::posting_arena::Policy::Auto,
                    SelectionMode::Bulk(4),
                    corpus::Order::WeightStable,
                )
                .unwrap();
                assert_eq!(stable.trace, expected.trace);
                assert_eq!(stable.vocab, expected.vocab);
                assert_eq!(stable.merges, expected.merges);
                for mode in [
                    SelectionMode::Cached,
                    SelectionMode::Leader,
                    SelectionMode::Bulk(4),
                    SelectionMode::Bulk(16),
                ] {
                    let got = train_with_selection(
                        &trainer,
                        &words,
                        config,
                        super::super::posting_arena::Policy::Auto,
                        mode,
                    )
                    .unwrap();
                    assert_eq!(got.trace, expected.trace);
                    assert_eq!(got.vocab, expected.vocab);
                    assert_eq!(got.merges, expected.merges);
                    assert_eq!(got.stats.batch_rounds, expected.stats.batch_rounds);
                    assert_eq!(got.stats.posting_visits, expected.stats.posting_visits);
                }
            }
        }
    }
    #[test]
    fn simultaneous_arena_policies_preserve_training_results() {
        use super::super::posting_arena::Policy;
        let words: AHashMap<CompactString, u64> = [
            ("xab中abq".repeat(10000), 3),
            ("中文aaabc".repeat(9000), u32::MAX as u64 + 1),
            ("zeroabab".repeat(8000), 0),
        ]
        .into_iter()
        .map(|(text, weight)| (text.into(), weight))
        .collect();
        let trainer = BpeTrainer::builder()
            .vocab_size(96)
            .min_frequency(2)
            .show_progress(false)
            .build();
        let expected = trainer.do_train_indexed(&words).unwrap();
        std::thread::scope(|scope| {
            let jobs: Vec<_> = [Policy::System, Policy::Fixed(32), Policy::Auto]
                .into_iter()
                .map(|policy| {
                    let trainer = &trainer;
                    let words = &words;
                    scope.spawn(move || {
                        let mut results = Vec::new();
                        for bits in [16, 32] {
                            let got = train_with_policy(
                                trainer,
                                words,
                                IndexedParallelConfig {
                                    workers: 2,
                                    initialization_workers: Some(1),
                                    posting_block_bits: bits,
                                    atomic_corpus: true,
                                    narrow_corpus: false,
                                    ..Default::default()
                                },
                                policy,
                            )
                            .unwrap();
                            assert_eq!(
                                got.stats.posting_arena_cutoff_bytes,
                                policy.cutoff(got.stats.initial_edges)
                            );
                            assert_eq!(
                                got.stats.posting_allocations.arena_requested_bytes,
                                got.stats.posting_allocations.arena_retired_bytes
                            );
                            if matches!(policy, Policy::System) {
                                assert_eq!(got.stats.posting_allocations.arena_buffers, 0);
                            }
                            results.push(got);
                        }
                        results
                    })
                })
                .collect();
            for job in jobs {
                for got in job.join().unwrap() {
                    assert_eq!(got.trace, expected.trace);
                    assert_eq!(got.vocab, expected.vocab);
                    assert_eq!(got.merges, expected.merges);
                }
            }
        });
    }
    use super::*;

    fn verify(
        trainer: &BpeTrainer,
        words: &AHashMap<CompactString, u64>,
        config: IndexedParallelConfig,
    ) -> IndexedTraining {
        let mut expected = Vec::new();
        let (vocab, merges, _) = trainer
            .do_train_observed(words, |p, f, id| expected.push((p, f, id)))
            .unwrap();
        let got = trainer.do_train_indexed_parallel(words, config).unwrap();
        assert_eq!(got.trace, expected);
        assert_eq!(got.vocab, vocab);
        assert_eq!(got.merges, merges);
        got
    }

    #[test]
    fn fused_flat_batches_keep_adjacent_and_shared_head_tail_births_ordered() {
        let words = [
            ("abcdxabcyabdz".repeat(3000), 2),
            ("abefcdabghcd".repeat(2000), 5),
            ("aaaabcdd".repeat(1000), 3),
            ("中ab文cd中ef文".repeat(1000), 1),
        ]
        .into_iter()
        .map(|(w, n)| (w.into(), n))
        .collect();
        for limit in [None, Some(3), Some(7)] {
            let trainer = BpeTrainer::builder()
                .vocab_size(120)
                .show_progress(false)
                .max_token_length(limit)
                .special_tokens(vec![AddedToken::from("abcd", true)])
                .build();
            let expected = trainer.do_train_indexed(&words).unwrap();
            for workers in [1, 4] {
                let got = trainer
                    .do_train_indexed_parallel(
                        &words,
                        IndexedParallelConfig {
                            workers,
                            initialization_workers: None,
                            posting_block_bits: 32,
                            narrow_corpus: false,
                            atomic_corpus: true,
                            batch_size: 256,
                        },
                    )
                    .unwrap();
                assert_eq!(got.trace, expected.trace);
                assert_eq!(got.vocab, expected.vocab);
                assert_eq!(got.merges, expected.merges);
                assert!(got.stats.fused_batches > 0);
                assert!(got.stats.peak_valid_start_bytes > 0);
            }
        }
    }

    #[test]
    fn disjoint_batches_and_reserved_identity_ties() {
        let trainer = BpeTrainer::builder()
            .vocab_size(100)
            .show_progress(false)
            .build();
        let words = [("ab", 10), ("cd", 9), ("ef", 8), ("gh", 7), ("ij", 6)]
            .into_iter()
            .map(|(w, n)| (w.into(), n))
            .collect();
        let got = verify(&trainer, &words, IndexedParallelConfig::default());
        assert_eq!(got.stats.max_batch_rules, 5);
        let trainer = BpeTrainer::builder()
            .vocab_size(30)
            .show_progress(false)
            .special_tokens(vec![AddedToken::from("ab", true)])
            .build();
        let words = [("abc", 1), ("ab", 1), ("ad", 1)]
            .into_iter()
            .map(|(w, n)| (w.into(), n))
            .collect();
        let got = verify(&trainer, &words, IndexedParallelConfig::default());
        assert_eq!(got.trace[0].2, 0);
        assert_eq!(
            got.trace[1].0.0, 0,
            "reserved output wins the newborn frequency tie"
        );
    }

    #[test]
    fn cross_block_aa_and_weighted_words_in_all_storage_widths() {
        let words = [
            ("a".repeat(70009), 3),
            ("b".repeat(11003), 7),
            ("abcd".repeat(19000), 2),
        ]
        .into_iter()
        .map(|(w, n)| (w.into(), n))
        .collect();
        for max_length in [None, Some(3), Some(19)] {
            let trainer = BpeTrainer::builder()
                .vocab_size(100)
                .show_progress(false)
                .max_token_length(max_length)
                .build();
            // The serial index is already checked against HF round by round.
            // Reuse it here to exercise block addresses without repeating HF's
            // quadratic Vec shifts on the same 70k-character AA word.
            let expected = trainer.do_train_indexed(&words).unwrap();
            for (workers, bits, narrow, atomic) in [
                (1, 32, false, false),
                (4, 32, true, false),
                (4, 16, false, false),
                (4, 16, true, false),
                (1, 32, false, true),
                (4, 32, true, true),
                (4, 16, false, true),
                (4, 16, true, true),
            ] {
                let got = trainer
                    .do_train_indexed_parallel(
                        &words,
                        IndexedParallelConfig {
                            workers,
                            initialization_workers: None,
                            posting_block_bits: bits,
                            narrow_corpus: narrow,
                            atomic_corpus: atomic,
                            batch_size: 256,
                        },
                    )
                    .unwrap();
                assert_eq!(got.trace, expected.trace);
                assert_eq!(got.vocab, expected.vocab);
                assert_eq!(got.merges, expected.merges);
                assert_eq!(got.stats.corpus_slot_bytes == 2, narrow);
            }
        }
    }

    #[test]
    fn birth_floor_is_applied_after_all_worker_counts() {
        let words = [(CompactString::from("xabq".repeat(20000)), 1)]
            .into_iter()
            .collect();
        let trainer = BpeTrainer::builder()
            .vocab_size(40)
            .min_frequency(15000)
            .show_progress(false)
            .build();
        let expected = trainer.do_train_indexed(&words).unwrap();
        for bits in [16, 32] {
            for narrow in [false, true] {
                for atomic in [false, true] {
                    let got = trainer
                        .do_train_indexed_parallel(
                            &words,
                            IndexedParallelConfig {
                                workers: 4,
                                initialization_workers: None,
                                posting_block_bits: bits,
                                narrow_corpus: narrow,
                                atomic_corpus: atomic,
                                batch_size: 256,
                            },
                        )
                        .unwrap();
                    assert_eq!(got.trace, expected.trace);
                    assert_eq!(got.vocab, expected.vocab);
                    assert_eq!(got.merges, expected.merges);
                }
            }
        }
    }

    #[test]
    fn target_and_reserved_alphabet_guard_narrow_corpus() {
        let words = [("ab", 2)]
            .into_iter()
            .map(|(w, n)| (w.into(), n))
            .collect();
        for (size, layout) in [
            (65535, "parallel_u32_planes32"),
            (65536, "parallel_u32_planes32"),
        ] {
            let trainer = BpeTrainer::builder()
                .vocab_size(size)
                .show_progress(false)
                .build();
            assert_eq!(
                verify(&trainer, &words, IndexedParallelConfig::default())
                    .stats
                    .layout,
                layout
            );
        }
        let trainer = BpeTrainer::builder()
            .vocab_size(8)
            .show_progress(false)
            .special_tokens(
                (0..65535)
                    .map(|i| AddedToken::from(format!("reserved{i}"), true))
                    .collect(),
            )
            .build();
        let got = trainer
            .do_train_indexed_parallel(&words, IndexedParallelConfig::default())
            .unwrap();
        assert_eq!(got.stats.layout, "parallel_u32_planes32");
        assert!(got.vocab["b"] >= 65535);
    }

    #[test]
    fn exclusive_writes_use_full_global_addresses() {
        let base = (u32::MAX as usize) + 70001;
        let mut slots = [1_u16, 2, 3, 4, 65535];
        let rules = [
            Rule {
                edge: (1, 2),
                replacement: 7,
                left_len: 1,
                right_len: 1,
            },
            Rule {
                edge: (3, 4),
                replacement: 8,
                left_len: 1,
                right_len: 1,
            },
        ];
        let plans = [
            Plan {
                position: base,
                rank: 0,
            },
            Plan {
                position: base + 2,
                rank: 1,
            },
        ];
        write_plans(&mut slots, base, &plans, &rules, 4);
        assert_eq!(slots, [7, 7, 8, 8, 65535]);
        // Exercise recursive split_at_mut with global addresses above u32,
        // without allocating the preceding address space.
        let many: Vec<_> = (0..4096)
            .map(|i| Plan {
                position: base + 2 * i,
                rank: i % 2,
            })
            .collect();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let mut ordinary = vec![0_u16; 8193];
        ordinary[8192] = u16::MAX;
        pool.install(|| write_plans(&mut ordinary, base, &many, &rules, 4));
        let mut atomic: Vec<_> = (0..8193).map(|_| AtomicU16::new(0)).collect();
        atomic[8192].store(u16::MAX, AtomicOrdering::Relaxed);
        pool.install(|| write_plans(&mut atomic, base, &many, &rules, 4));
        for (i, (&plain, atom)) in ordinary.iter().zip(&atomic).enumerate() {
            let expected = if i == 8192 {
                u16::MAX
            } else {
                7 + ((i / 2) % 2) as u16
            };
            assert_eq!(plain, expected);
            assert_eq!(atom.load(AtomicOrdering::Relaxed), expected);
        }
        let block = Block::<u16, 4>::new(base, 13);
        assert_eq!(block.weight(base + 65535, None), 13);
        assert_eq!(<u16 as Offset>::encode(65535_usize), 65535);
        assert_eq!(std::mem::size_of::<Entry>(), 24);
    }

    #[test]
    fn forward_weight_cursor_handles_near_words_and_sparse_gaps() {
        let base = u32::MAX as usize + 1;
        let mut block = Block::<u32, 2>::new(base, 13);
        block.pivots = (0..1000).map(|i| 5 + i * 37).collect();
        block.weights = (0..1000).map(|i| 101 + i).collect();
        let mut cursor = 0;
        for offset in [0, 1, 4, 5, 5, 6, 41, 42, 500, 501, 36890, 65535] {
            assert_eq!(
                block.weight_forward(base + offset, None, &mut cursor),
                block.weight(base + offset, None)
            );
        }
        assert_eq!(
            block.weight_forward(base + 65535, Some(99), &mut cursor),
            99
        );
        let empty = Block::<u16, 4>::new(base, 27);
        assert_eq!(empty.weight_forward(base + 65535, None, &mut 0), 27);
    }

    #[test]
    fn candidate_heap_range_fallback_keeps_weighted_greedy_semantics() {
        let run = |weight, vocab| {
            let words = [(CompactString::from("aaaababa"), weight)]
                .into_iter()
                .collect();
            let trainer = BpeTrainer::builder()
                .vocab_size(vocab)
                .min_frequency(2)
                .show_progress(false)
                .build();
            // The legacy HF oracle casts weights to i32; the wide indexed
            // serial oracle keeps u64 frequencies for this range test.
            let expected = trainer.do_train_indexed(&words).unwrap();
            let got = trainer
                .do_train_indexed_parallel(&words, IndexedParallelConfig::default())
                .unwrap();
            assert_eq!(got.trace, expected.trace);
            assert_eq!(got.vocab, expected.vocab);
            assert_eq!(got.merges, expected.merges);
            got
        };
        let packed = run(3, 32);
        let large_weight = run(u32::MAX as u64 + 1, 32);
        let large_id_domain = run(3, 70000);
        assert_eq!(packed.stats.initial_pairs, large_weight.stats.initial_pairs);
        assert_eq!(
            packed.stats.initial_pairs,
            large_id_domain.stats.initial_pairs
        );
        assert_eq!(
            large_weight.stats.initial_heap_bytes,
            packed.stats.initial_heap_bytes * 2
        );
        assert_eq!(
            large_id_domain.stats.initial_heap_bytes,
            packed.stats.initial_heap_bytes * 2
        );
    }

    #[test]
    fn nonempty_alias_affixes_keep_exact_cohorts() {
        for (prefix, suffix, text) in [
            (None, Some("a"), "baaba"),
            (Some("ab"), None, "abaa"),
            (Some("##"), Some("</w>"), "aaaababa"),
        ] {
            let mut trainer = BpeTrainer::builder()
                .vocab_size(24)
                .show_progress(false)
                .build();
            trainer.continuing_subword_prefix = prefix.map(str::to_owned);
            trainer.end_of_word_suffix = suffix.map(str::to_owned);
            let words = [(CompactString::from(text), 1)].into_iter().collect();
            let expected = trainer.do_train_indexed(&words).unwrap();
            let got = trainer
                .do_train_indexed_parallel(&words, IndexedParallelConfig::default())
                .unwrap();
            assert_eq!(got.trace, expected.trace);
            assert_eq!(got.vocab, expected.vocab);
            assert_eq!(got.merges, expected.merges);
            assert_eq!(got.stats.workers, 4);
            assert_eq!(got.stats.initialization_workers, 4);
            if suffix == Some("a") {
                assert_eq!(got.trace[1], ((1, 2), 2, 3));
                assert!(!got.stats.monotone_pairs);
            }
        }
    }

    #[test]
    fn block_counts_keep_uniform_zero_and_sparse_nonunit_weights_exact() {
        let texts = [
            "xabq".repeat(18000),
            "ab中中".repeat(19000),
            "中文aab".repeat(15000),
        ];
        for weights in [
            [1, 1, 1],
            [2, 2, 2],
            [0, 0, 0],
            [0, 1, 7],
            [3, u32::MAX as u64 + 1, 1],
        ] {
            let words = texts
                .iter()
                .zip(weights)
                .map(|(text, weight)| (CompactString::from(text.as_str()), weight))
                .collect();
            let trainer = BpeTrainer::builder()
                .vocab_size(96)
                .min_frequency(2)
                .show_progress(false)
                .build();
            let expected = trainer.do_train_indexed(&words).unwrap();
            for atomic in [false, true] {
                let got = trainer
                    .do_train_indexed_parallel(
                        &words,
                        IndexedParallelConfig {
                            posting_block_bits: 16,
                            atomic_corpus: atomic,
                            narrow_corpus: false,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                assert!(got.stats.initial_blocks > 1);
                assert_eq!(got.trace, expected.trace);
                assert_eq!(got.vocab, expected.vocab);
                assert_eq!(got.merges, expected.merges);
            }
        }
    }

    #[test]
    fn block_summary_waves_keep_global_floor_and_cross_wave_aa_order() {
        let words = [
            ("aaab".repeat(25000), 0),
            ("aaab中".repeat(21000), 3),
            ("中aaab".repeat(24000), 1),
        ]
        .into_iter()
        .map(|(text, weight)| (CompactString::from(text), weight))
        .collect();
        let trainer = BpeTrainer::builder()
            .vocab_size(48)
            .min_frequency(2)
            .show_progress(false)
            .build();
        let expected = trainer.do_train_indexed(&words).unwrap();
        for workers in [1, 2, 4] {
            for atomic in [false, true] {
                let got = trainer
                    .do_train_indexed_parallel(
                        &words,
                        IndexedParallelConfig {
                            workers,
                            initialization_workers: Some(2),
                            posting_block_bits: 16,
                            narrow_corpus: false,
                            atomic_corpus: atomic,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                assert!(got.stats.initial_blocks > 4);
                assert_eq!(got.stats.initial_summary_waves, 0);
                assert!(got.stats.initial_route_buffer_bytes > 0);
                assert_eq!(got.stats.initial_count_backend, "stable_radix16");
                assert_eq!(got.trace, expected.trace);
                assert_eq!(got.vocab, expected.vocab);
                assert_eq!(got.merges, expected.merges);
            }
        }
    }

    #[test]
    fn serial_initialization_keeps_parallel_greedy_order() {
        let words = [("xabq".repeat(2100), 3), ("中文中文abcdef".repeat(1300), 7)]
            .into_iter()
            .map(|(w, n)| (w.into(), n))
            .collect();
        let trainer = BpeTrainer::builder()
            .vocab_size(100)
            .show_progress(false)
            .build();
        let expected = trainer.do_train_indexed(&words).unwrap();
        let got = trainer
            .do_train_indexed_parallel(
                &words,
                IndexedParallelConfig {
                    initialization_workers: Some(1),
                    narrow_corpus: false,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(got.stats.initialization_workers, 1);
        assert_eq!(got.stats.workers, 4);
        assert_eq!(got.trace, expected.trace);
        assert_eq!(got.vocab, expected.vocab);
        assert_eq!(got.merges, expected.merges);
    }
}
