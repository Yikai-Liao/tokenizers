//! Local preaggregation into a shared table; feed returns unordered counts.
use super::word_counts::WordCounts;
use ahash::{AHashMap, RandomState};
use compact_str::CompactString;
use std::collections::hash_map::Entry;
use tk_encode::{Result, parallelism::*};

type CountMap = AHashMap<CompactString, u64>;
type SharedCounts = scc::HashMap<CompactString, u64, RandomState>;

// Cap distinct local keys; strings and callback output are not byte-bounded.
const LOCAL_KEY_LIMIT: usize = 2048;
// Amortize the bridge's serialized source `next()` lock across several inputs.
const FEED_BATCH_SIZE: usize = 32;

fn batches<I, S>(iterator: I, mut startup_singles: usize) -> impl Iterator<Item = Vec<S>> + Send
where
    I: Iterator<Item = S> + Send,
    S: Send,
{
    // Keep one exhaustion state across batches, including a partial final batch.
    let mut iterator = iterator.fuse();
    std::iter::from_fn(move || {
        if startup_singles > 0 {
            let item = iterator.next()?;
            startup_singles -= 1;
            return Some(vec![item]);
        }
        let batch: Vec<_> = iterator.by_ref().take(FEED_BATCH_SIZE).collect();
        (!batch.is_empty()).then_some(batch)
    })
}

/// One fold's bounded distinct-word cache and its shared weighted-count destination.
/// Full caches flush during collection; finish flushes remaining entries only
/// after every fold succeeds. Dropping failed work never publishes its remainder.
struct LocalCounts<'shared> {
    counts: CountMap,
    shared: &'shared SharedCounts,
}

impl<'shared> LocalCounts<'shared> {
    fn new(shared: &'shared SharedCounts, hash: RandomState) -> Self {
        Self {
            counts: CountMap::with_hasher(hash),
            shared,
        }
    }

    fn add(&mut self, word: CompactString) {
        match self.counts.entry(word) {
            Entry::Occupied(mut entry) => *entry.get_mut() += 1,
            Entry::Vacant(entry) => {
                entry.insert(1);
                // Only a new key can fill the cache; repeated words need no size check.
                if self.counts.len() == LOCAL_KEY_LIMIT {
                    self.flush();
                }
            }
        }
    }

    fn flush(&mut self) {
        for (word, count) in self.counts.drain() {
            self.shared
                .entry_sync(word)
                .and_modify(|total| *total += count)
                .or_insert(count);
        }
    }

    fn finish(mut self) {
        self.flush();
    }
}

fn accumulate<S, F, C, A>(counts: Result<C>, sequence: S, process: &F, mut add: A) -> Result<C>
where
    S: AsRef<str>,
    F: Fn(&str) -> Result<Vec<String>>,
    A: FnMut(&mut C, CompactString),
{
    // Run the callback even after a normal error; retain this fold's first error.
    let words = process(sequence.as_ref());
    let mut counts = counts?;
    for word in words? {
        add(&mut counts, CompactString::from(word));
    }
    Ok(counts)
}

pub(super) fn count<I, S, F>(iterator: I, process: &F) -> Result<WordCounts>
where
    I: Iterator<Item = S> + Send,
    S: AsRef<str> + Send,
    F: Fn(&str) -> Result<Vec<String>> + Sync,
{
    let hash = RandomState::default();
    let new_counts = || Ok(CountMap::with_hasher(hash.clone()));
    let parallel_workers = if get_parallelism() {
        current_num_threads()
    } else {
        1
    };
    if parallel_workers == 1 {
        // A single worker needs neither shared updates nor a final copy. Keep
        // its map, using the same callback/error handling as the parallel path.
        return iterator
            .fold(new_counts(), |counts, sequence| {
                accumulate(counts, sequence, process, |counts, word| {
                    *counts.entry(word).or_default() += 1;
                })
            })
            .map(WordCounts::from_map);
    }
    let shared = SharedCounts::with_hasher(hash.clone());
    // Keep preprocessing behind the upstream bridge: mapping it on the input
    // iterator would execute callbacks under the bridge's serial next() lock.
    // Local caches absorb repeats; flushes update weighted counts directly,
    // avoiding a separate partition-and-reduce pass over all local entries.
    // Seed the ambient pool with independent inputs before amortizing the
    // bridge lock. A short stream of expensive documents would otherwise form
    // one batch whose callbacks all run serially on the same worker. Source
    // exhaustion naturally caps these singletons at the available input count.
    let results: Vec<Result<LocalCounts<'_>>> = batches(iterator, parallel_workers)
        .maybe_par_bridge()
        .flat_map_iter(std::iter::IntoIterator::into_iter)
        .fold(
            || Ok(LocalCounts::new(&shared, hash.clone())),
            |counts, sequence| accumulate(counts, sequence, process, LocalCounts::add),
        )
        .collect();
    let locals: Vec<_> = results.into_iter().collect::<Result<Vec<_>>>()?;
    locals.into_maybe_par_iter().for_each(LocalCounts::finish);
    // Training only traverses counts, so consume keys into an unordered vector
    // instead of paying to allocate and rehash another dictionary. Training
    // sorts borrowed entries; feed retains no frequency ordering or index.
    let mut words = Vec::with_capacity(shared.len());
    // Keep table capacity stable while consumption removes its entries.
    let _reservation = shared.reserve(shared.capacity());
    shared.iter_mut_sync(|entry| {
        words.push(entry.consume());
        true
    });
    Ok(WordCounts::from_entries(words))
}
