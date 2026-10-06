//! Local preaggregation into a shared table; feed returns unordered counts.
use super::word_counts::WordCounts;
use ahash::{AHashMap, RandomState};
use compact_str::CompactString;
use std::collections::hash_map::Entry;
use tk_encode::{Result, parallelism::*};

type CountMap = AHashMap<CompactString, u64>;
type Shared = scc::HashMap<CompactString, u64, RandomState>;
// Bound per-worker memory while amortizing shared-table writes across repeats.
const LOCAL_KEYS: usize = 2048;

fn flush(local: &mut CountMap, shared: &Shared) {
    for (word, count) in local.drain() {
        shared
            .entry_sync(word)
            .and_modify(|total| *total += count)
            .or_insert(count);
    }
}

fn add_local(local: &mut CountMap, word: CompactString, shared: &Shared) {
    match local.entry(word) {
        Entry::Occupied(mut entry) => *entry.get_mut() += 1,
        Entry::Vacant(entry) => {
            entry.insert(1);
            // Only a new key can fill the cache; repeated words need no size check.
            if local.len() == LOCAL_KEYS {
                flush(local, shared);
            }
        }
    }
}

fn accumulate<S, F, A>(
    counts: Result<CountMap>,
    sequence: S,
    process: &F,
    mut add: A,
) -> Result<CountMap>
where
    S: AsRef<str>,
    F: Fn(&str) -> Result<Vec<String>>,
    A: FnMut(&mut CountMap, CompactString),
{
    // Callbacks still run after a normal error; the first error is retained.
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
    if !get_parallelism() || current_num_threads() == 1 {
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
    let shared = Shared::with_hasher(hash.clone());
    // Keep preprocessing behind the upstream bridge: mapping it on the input
    // iterator would execute callbacks under the bridge's serial next() lock.
    // Local caches absorb repeats; flushes update weighted counts directly,
    // avoiding a separate partition-and-reduce pass over all local entries.
    let results: Vec<Result<CountMap>> = iterator
        .maybe_par_bridge()
        .fold(new_counts, |counts, sequence| {
            accumulate(counts, sequence, process, |counts, word| {
                add_local(counts, word, &shared)
            })
        })
        .collect();
    let locals: Vec<_> = results.into_iter().collect::<Result<Vec<_>>>()?;
    locals
        .into_maybe_par_iter()
        .for_each(|mut local| flush(&mut local, &shared));
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
