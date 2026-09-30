//! Certified batches with spatial posting dictionaries and exclusive corpus writes.
//! Planning only borrows the old corpus. AA selection continues across dictionary
//! boundaries. Sorted, disjoint plans then define safe split_at_mut boundaries.
use super::small_posting::{PackedPosting, SmallPosting};
use super::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU16, AtomicU32, Ordering as AtomicOrdering};

mod alphabet;
mod corpus;

trait Slot: Default + Send + Sync {
    const NARROW: bool;
    fn encode(id: u32) -> Self;
    fn token(&self) -> u32;
    fn set(&mut self, id: u32);
}
impl Slot for u32 {
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
    blocks: SmallPosting,
}
#[derive(Default)]
struct Owner {
    entries: AHashMap<u64, Entry>,
    heap: OctonaryHeap<Candidate>,
}
impl Owner {
    fn peek_current(&mut self) -> Option<Candidate> {
        loop {
            let top = *self.heap.peek()?;
            match self.entries.get(&top.key) {
                None => {
                    self.heap.pop();
                }
                Some(entry) if entry.frequency != top.frequency => {
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
    // Word starts are also local addresses; a long word can span several blocks.
    pivots: Vec<u32>,
    weights: Vec<u64>,
    previous_weight: u64,
}
impl<O: Offset, const INLINE: usize> Block<O, INLINE> {
    fn new(base: usize, previous_weight: u64) -> Self {
        Self {
            base,
            postings: AHashMap::new(),
            pivots: Vec::new(),
            weights: Vec::new(),
            previous_weight,
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
        }
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
            route.nodes.push(Node {
                position: u32::try_from(position).map_err(|_| "flat birth position exceeds u32")?,
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
    let alphabet_scratch_bytes = initialization_pool.as_ref().unwrap_or(&pool).install(|| {
        alphabet::initialize(
            trainer,
            wc,
            &mut ids,
            &mut strings,
            config.initialization_workers.unwrap_or(config.workers),
        )
    });
    let alphabet_ms = alphabet_begin.elapsed().as_secs_f64() * 1000.0;
    // Select the final slot type before emitting any corpus. Reserved IDs and
    // forced alphabet count towards this bound, not just the requested vocabulary.
    let narrow = config.narrow_corpus && strings.len().max(trainer.vocab_size) <= u16::MAX as usize;
    match (config.atomic_corpus, narrow, config.posting_block_bits) {
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
        ),
        _ => unreachable!("validated configuration"),
    }
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
) -> Result<IndexedTraining> {
    // Keep the coordinator inside this pool too: small serial rounds then do
    // not pay a caller->worker handoff at every stage.
    pool.install(|| {
        train_in_pool::<C, O, INLINE>(
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
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn train_in_pool<C: Slot, O: Offset, const INLINE: usize>(
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
) -> Result<IndexedTraining> {
    let bits = config.posting_block_bits;
    let block_size = 1_usize
        .checked_shl(bits as u32)
        .ok_or("posting blocks require 64-bit usize")?;
    let constructing_pool = initialization_pool.unwrap_or(pool);
    let prepared = constructing_pool.install(|| {
        corpus::build::<C, O, INLINE>(
            wc,
            &ids,
            strings.len(),
            trainer.limit_alphabet.is_none(),
            bits,
            config.initialization_workers.unwrap_or(config.workers),
        )
    })?;
    let corpus::Prepared {
        slots: mut corpus,
        mut lengths,
        mut blocks,
        uniform,
        symbols: initial_symbols,
        edges: initial_edges,
        timings,
        character_table_bytes,
    } = prepared;
    let flat = bits == 32 && corpus.len() <= u32::MAX as usize + 1;
    let mut owners: Vec<Owner> = (0..config.workers).map(|_| Owner::default()).collect();
    let tokenize_ms = begin.elapsed().as_secs_f64() * 1000.0;
    let floor = trainer.min_frequency.max(1);
    let mut initialize = || -> Result<IndexedTrainingStats> {
        let initial_begin = Instant::now();
        let initial_route_ms;
        let initial_count_ms;
        if flat {
            // Prototype's two-stage initialization: route compact positions, then
            // let each owner count and build its own lists. Unicode lookup and the
            // much larger pair hash tables no longer interleave on every character.
            // One reusable route buffer per worker. Contiguous ranges keep each
            // owner's incoming positions ordered, allowing a linear weight cursor.
            let route_chunk = corpus.len().div_ceil(config.workers).max(1);
            let routes: Vec<Vec<Vec<u32>>> = corpus
                .par_chunks(route_chunk)
                .enumerate()
                .map(|(chunk, slots)| {
                    let base = chunk * route_chunk;
                    let mut routed: Vec<Vec<u32>> =
                        (0..config.workers).map(|_| Vec::new()).collect();
                    for (i, slot) in slots.iter().enumerate() {
                        let p = base + i;
                        if p + 1 == corpus.len() {
                            break;
                        }
                        let a = slot.token();
                        let b = corpus[p + 1].token();
                        if a != NONE && b != NONE {
                            routed[owner(key(a, b), config.workers)].push(p as u32);
                        }
                    }
                    routed
                })
                .collect();
            initial_route_ms = initial_begin.elapsed().as_secs_f64() * 1000.0;
            let count_begin = Instant::now();
            owners
                .par_iter_mut()
                .enumerate()
                .map(|(o, ledger)| -> Result<()> {
                    let block = &blocks[0];
                    let mut word = 0;
                    for routed in &routes {
                        for &position in &routed[o] {
                            let p = position as usize;
                            let weight = if let Some(weight) = uniform {
                                weight
                            } else {
                                while word + 1 < block.pivots.len()
                                    && block.pivots[word + 1] <= position
                                {
                                    word += 1;
                                }
                                block.weights[word]
                            };
                            let k = key(corpus[p].token(), corpus[p + 1].token());
                            let entry = ledger.entries.entry(k).or_insert_with(|| Entry {
                                frequency: 0,
                                blocks: SmallPosting::default(),
                            });
                            entry.frequency += weight; // checked weighted total bounds every key
                            entry.blocks.push(position)?;
                        }
                    }
                    Ok(())
                })
                .collect::<Result<Vec<_>>>()?;
            initial_count_ms = count_begin.elapsed().as_secs_f64() * 1000.0;
        } else {
            // Local dictionaries are built by physical block owners. Only key-level
            // frequencies and block directories are reduced globally, not positions.
            let frequencies: Vec<Vec<Vec<(u64, u64)>>> = blocks
                .par_iter_mut()
                .map(|block| -> Result<_> {
                    let mut counts = AHashMap::new();
                    let end = corpus.len().saturating_sub(1).min(block.base + block_size);
                    let mut word = 0;
                    for p in block.base..end {
                        let a = corpus[p].token();
                        let b = corpus[p + 1].token();
                        if a == NONE || b == NONE {
                            continue;
                        }
                        let local = (p - block.base) as u32;
                        let weight = if let Some(weight) = uniform {
                            weight
                        } else {
                            while word < block.pivots.len() && block.pivots[word] <= local {
                                word += 1;
                            }
                            if word == 0 {
                                block.previous_weight
                            } else {
                                block.weights[word - 1]
                            }
                        };
                        let k = key(a, b);
                        *counts.entry(k).or_default() += weight;
                        block
                            .postings
                            .entry(k)
                            .or_default()
                            .push(O::encode(p - block.base))?;
                    }
                    let mut routed: Vec<Vec<(u64, u64)>> =
                        (0..config.workers).map(|_| Vec::new()).collect();
                    for (k, weight) in counts {
                        routed[owner(k, config.workers)].push((k, weight));
                    }
                    Ok(routed)
                })
                .collect::<Result<Vec<_>>>()?;
            initial_route_ms = initial_begin.elapsed().as_secs_f64() * 1000.0;
            let count_begin = Instant::now();
            owners
                .par_iter_mut()
                .enumerate()
                .map(|(o, ledger)| -> Result<()> {
                    for (b, counts) in frequencies.iter().enumerate() {
                        for &(k, weight) in &counts[o] {
                            let entry = ledger.entries.entry(k).or_insert_with(|| Entry {
                                frequency: 0,
                                blocks: SmallPosting::default(),
                            });
                            entry.frequency += weight;
                            entry.blocks.push(b as u32)?;
                        }
                    }
                    Ok(())
                })
                .collect::<Result<Vec<_>>>()?;
            initial_count_ms = count_begin.elapsed().as_secs_f64() * 1000.0;
        }
        let mut stats = IndexedTrainingStats {
            initial_symbols,
            tokenize_ms,
            alphabet_ms,
            alphabet_scratch_bytes,
            character_table_bytes,
            corpus_measure_ms: timings.measure_ms,
            corpus_allocate_ms: timings.allocate_ms,
            corpus_fill_ms: timings.fill_ms,
            initial_route_ms,
            initial_count_ms,
            monotone_pairs: true,
            workers: config.workers,
            initialization_workers: config.initialization_workers.unwrap_or(config.workers),
            atomic_corpus: config.atomic_corpus,
            layout: match (C::NARROW, bits, flat) {
                (true, 16, _) => "parallel_u16_dict16",
                (true, 32, true) => "parallel_u16_flat32",
                (true, 32, false) => "parallel_u16_dict32",
                (false, 16, _) => "parallel_u32_dict16",
                (false, 32, true) => "parallel_u32_flat32",
                _ => "parallel_u32_dict32",
            },
            ..Default::default()
        };
        let heap_begin = Instant::now();
        stats.pruned_pairs += owners
            .par_iter_mut()
            .map(|ledger| {
                let old = ledger.entries.len();
                ledger.entries.retain(|_, entry| entry.frequency >= floor);
                ledger.heap = ledger
                    .entries
                    .iter()
                    .map(|(&key, e)| Candidate {
                        key,
                        frequency: e.frequency,
                    })
                    .collect();
                old - ledger.entries.len()
            })
            .sum::<usize>();
        if !flat {
            blocks.par_iter_mut().for_each(|block| {
                block
                    .postings
                    .retain(|k, _| owners[owner(*k, config.workers)].entries.contains_key(k));
            });
        }
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
        stats.initial_block_pairs = blocks.iter().map(|b| b.postings.len()).sum();
        stats.initial_slot_bytes = corpus.capacity() * std::mem::size_of::<C>();
        stats.initial_length_bytes = lengths.capacity() * std::mem::size_of::<usize>();
        stats.initial_weight_bytes =
            stats.initial_corpus_bytes - stats.initial_slot_bytes - stats.initial_length_bytes;
        stats.initial_posting_bytes = if flat {
            owners
                .iter()
                .flat_map(|o| o.entries.values())
                .map(|e| e.blocks.allocated_capacity() * 4)
                .sum()
        } else {
            blocks
                .iter()
                .flat_map(|b| b.postings.values())
                .map(|p| p.allocated_capacity() * std::mem::size_of::<O>())
                .sum()
        };
        stats.initial_pair_table_bytes = owners
            .iter()
            .map(|o| table_bytes(o.entries.capacity(), 32))
            .sum();
        stats.initial_block_table_bytes = blocks
            .iter()
            .map(|b| table_bytes(b.postings.capacity(), 24))
            .sum::<usize>()
            + blocks.capacity() * std::mem::size_of::<Block<O, INLINE>>();
        stats.initial_directory_bytes = if flat {
            0
        } else {
            owners
                .iter()
                .flat_map(|o| o.entries.values())
                .map(|e| e.blocks.allocated_capacity() * 4)
                .sum()
        };
        stats.initial_heap_bytes = owners.iter().map(|o| o.heap.capacity() * 16).sum();
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
    while ids.len() < trainer.vocab_size {
        let stage = Instant::now();
        let cap = config.batch_size.min(trainer.vocab_size - ids.len());
        let mut rules = Vec::<Rule>::new();
        let mut heads = AHashSet::new();
        let mut tails = AHashSet::new();
        let mut block_rules = AHashMap::<usize, Vec<usize>>::new();
        let mut flat_postings = Vec::new();
        while rules.len() < cap {
            let best = owners
                .iter_mut()
                .enumerate()
                .filter_map(|(i, o)| o.peek_current().map(|c| (i, c)))
                .max_by_key(|(_, c)| *c);
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
            token.push_str(&strings[edge.1 as usize]);
            let reserved = ids.get(&token).copied();
            // A reserved canonical ID can win an equal-frequency birth tie
            // before its old witness. Use one rule for that activation.
            if reserved.is_some() && !rules.is_empty() {
                break;
            }
            owners[o].heap.pop();
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
            let rank = rules.len();
            // Directory ownership ends here. Selected local postings are freed
            // during planning, before allocating the next generation's postings.
            if flat {
                stats.posting_visits += entry.blocks.len();
                flat_postings.push(entry.blocks);
            } else {
                for &b in entry.blocks.as_slice() {
                    block_rules.entry(b as usize).or_default().push(rank);
                }
                stats.posting_visits += entry
                    .blocks
                    .as_slice()
                    .iter()
                    .map(|&b| blocks[b as usize].postings[&top.key].len())
                    .sum::<usize>();
            }
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
        if rules.is_empty() {
            break;
        }
        stats.select_ms += stage.elapsed().as_secs_f64() * 1000.0;
        let stage = Instant::now();
        stats.batch_rounds += 1;
        stats.max_batch_rules = stats.max_batch_rules.max(rules.len());
        let chunks: Vec<Vec<Plan>> = if flat {
            pool.install(|| {
                flat_postings
                    .par_iter()
                    .enumerate()
                    .map(|(rank, posting)| {
                        let rule = &rules[rank];
                        posting
                            .as_slice()
                            .par_iter()
                            .filter_map(|&position| {
                                let p = position as usize;
                                let right = p + rule.left_len;
                                (corpus[p].token() == rule.edge.0
                                    && right < corpus.len()
                                    && corpus[right].token() == rule.edge.1)
                                    .then_some(Plan { position: p, rank })
                            })
                            .collect()
                    })
                    .collect()
            })
        } else {
            pool.install(|| {
                blocks
                    .par_iter_mut()
                    .enumerate()
                    .map(|(b, block)| -> Vec<Plan> {
                        let mut plans = Vec::new();
                        if let Some(ranks) = block_rules.get(&b) {
                            for &rank in ranks {
                                let rule = &rules[rank];
                                let posting = block
                                    .postings
                                    .remove(&key(rule.edge.0, rule.edge.1))
                                    .unwrap();
                                let valid: Vec<_> = posting
                                    .as_slice()
                                    .par_iter()
                                    .filter_map(|&offset| {
                                        let p = block.base + offset.index();
                                        let right = p + rule.left_len;
                                        if corpus[p].token() == rule.edge.0
                                            && right < corpus.len()
                                            && corpus[right].token() == rule.edge.1
                                        {
                                            Some(Plan { position: p, rank })
                                        } else {
                                            None
                                        }
                                    })
                                    .collect();
                                plans.extend(valid);
                            }
                        }
                        if block_rules.get(&b).is_some_and(|r| r.len() > 1) {
                            plans.par_sort_unstable_by_key(|p| p.position);
                        }
                        plans
                    })
                    .collect()
            })
        };
        drop(flat_postings);
        let count: usize = chunks.iter().map(Vec::len).sum();
        let mut plans = Vec::with_capacity(count);
        for mut chunk in chunks {
            plans.append(&mut chunk);
        }
        // Posting producers append ordered final boundary positions. A single
        // rule keeps that order, including AA; only mixed rules need sorting.
        if flat && rules.len() > 1 {
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
                        super::aa_parity::summarize_by(chunk.len(), |i| chunk[i].position, length)
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
        let output_chunk = plans.len().div_ceil(config.workers).max(4096);
        let outputs: Vec<Output<O, INLINE>> = pool.install(|| {
            plans
                .par_chunks(output_chunk)
                .enumerate()
                .map(|(chunk, local)| -> Result<_> {
                    let mut output = Output::new(config.workers, flat);
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
                                output.birth(key(prior, rule.replacement), before, weight, bits)?;
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
                                output.birth(key(rule.replacement, final_next), p, weight, bits)?;
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
        let stage = Instant::now();
        let commits: Vec<_> = pool.install(|| {
            owners
                .par_iter_mut()
                .enumerate()
                .map(|(o, ledger)| -> Result<_> {
                    let mut retired = Vec::new();
                    if flat {
                        let mut born = AHashMap::<u64, (u64, u32)>::new();
                        for output in &outputs {
                            for (&k, group) in &output.flat_routes[o].delta {
                                if group.occurrences != 0 {
                                    let entry = born.entry(k).or_default();
                                    entry.0 += group.weight;
                                    entry.1 = entry
                                        .1
                                        .checked_add(group.occurrences)
                                        .ok_or("birth posting count exceeds u32")?;
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
                                    blocks: SmallPosting::with_capacity(count)?,
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
                                while head != NONE {
                                    let node = &route.nodes[head as usize];
                                    entry.blocks.push(node.position)?;
                                    head = node.next;
                                }
                                // Chunks are ordered spatially; each chain is
                                // reversed locally so AA keeps an ordered list.
                                entry.blocks.as_mut_slice()[start..].reverse();
                            }
                        }
                        return Ok((retired, AHashSet::new(), dropped));
                    }
                    let mut sums = AHashMap::<u64, u64>::new();
                    for output in &outputs {
                        for (&k, &weight) in &output.removed[o] {
                            if let Some(entry) = ledger.entries.get_mut(&k) {
                                entry.frequency = entry
                                    .frequency
                                    .checked_sub(weight)
                                    .ok_or("old pair frequency underflow")?;
                                if entry.frequency < floor {
                                    let entry = ledger.entries.remove(&k).unwrap();
                                    if flat {
                                        drop(entry);
                                        retired.push((k, SmallPosting::default()));
                                    } else {
                                        retired.push((k, entry.blocks));
                                    }
                                }
                            }
                        }
                        for (&k, &weight) in &output.born[o] {
                            *sums.entry(k).or_default() += weight;
                        }
                    }
                    let mut accepted = AHashSet::new();
                    let mut dropped = 0;
                    for (k, frequency) in sums {
                        if frequency < floor {
                            dropped += 1;
                            continue;
                        }
                        debug_assert!(
                            !ledger.entries.contains_key(&k),
                            "a birth contains a newly activated identity"
                        );
                        ledger.entries.insert(
                            k,
                            Entry {
                                frequency,
                                blocks: SmallPosting::default(),
                            },
                        );
                        ledger.heap.push(Candidate { key: k, frequency });
                        accepted.insert(k);
                    }
                    Ok((retired, accepted, dropped))
                })
                .collect::<Result<Vec<_>>>()
        })?;
        stats.commit_ms += stage.elapsed().as_secs_f64() * 1000.0;
        let stage = Instant::now();
        let mut retired_blocks = AHashMap::<usize, Vec<u64>>::new();
        for (retired, _, dropped) in &commits {
            stats.pruned_pairs += retired.len() + dropped;
            for (k, directory) in retired {
                for &b in directory.as_slice() {
                    retired_blocks.entry(b as usize).or_default().push(*k);
                }
            }
        }
        if !flat {
            // Dictionaries install their own births. Count once and reserve the
            // final list, rather than copying via a coordinator's routed map.
            let directories: Vec<Vec<Vec<(u64, u32)>>> = blocks
                .par_iter_mut()
                .enumerate()
                .map(|(b, block)| -> Result<_> {
                    if let Some(keys) = retired_blocks.get(&b) {
                        for k in keys {
                            block.postings.remove(k);
                        }
                    }
                    let mut counts = AHashMap::<u64, usize>::new();
                    for output in &outputs {
                        if let Some(births) = output.blocks.get(&b) {
                            for (&k, positions) in births {
                                if commits[owner(k, config.workers)].1.contains(&k) {
                                    *counts.entry(k).or_default() += positions.len();
                                }
                            }
                        }
                    }
                    let mut directory: Vec<Vec<(u64, u32)>> =
                        (0..config.workers).map(|_| Vec::new()).collect();
                    for (k, count) in counts {
                        let mut positions = PackedPosting::<O, INLINE>::with_capacity(
                            u32::try_from(count).map_err(|_| "block posting count exceeds u32")?,
                        )?;
                        for output in &outputs {
                            if let Some(source) =
                                output.blocks.get(&b).and_then(|births| births.get(&k))
                            {
                                for &offset in source.as_slice() {
                                    positions.push(offset)?;
                                }
                            }
                        }
                        debug_assert!(!block.postings.contains_key(&k));
                        block.postings.insert(k, positions);
                        directory[owner(k, config.workers)].push((k, b as u32));
                    }
                    Ok(directory)
                })
                .collect::<Result<Vec<_>>>()?;
            owners
                .par_iter_mut()
                .enumerate()
                .map(|(o, ledger)| -> Result<()> {
                    for directory in &directories {
                        for &(k, b) in &directory[o] {
                            ledger.entries.get_mut(&k).unwrap().blocks.push(b)?;
                        }
                    }
                    Ok(())
                })
                .collect::<Result<Vec<_>>>()?;
        }
        stats.route_ms += stage.elapsed().as_secs_f64() * 1000.0;
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
                assert_eq!(got.stats.layout.starts_with("parallel_u16"), narrow);
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
            (65535, "parallel_u16_flat32"),
            (65536, "parallel_u32_flat32"),
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
        assert_eq!(got.stats.layout, "parallel_u32_flat32");
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
