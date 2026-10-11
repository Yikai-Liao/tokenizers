//! Parallel updates into a shared table; feed returns unordered counts.
use super::word_counts::WordCounts;
use ahash::{AHashMap, RandomState};
use compact_str::CompactString;
use tk_encode::{Result, parallelism::*};

type CountMap = AHashMap<CompactString, u64>;
type SharedCounts = scc::HashMap<CompactString, u64, RandomState>;

const FEED_BATCH_SIZE: usize = 32;

pub(super) fn count<I, S, F>(iterator: I, process: &F) -> Result<WordCounts>
where
    I: Iterator<Item = S> + Send,
    S: AsRef<str> + Send,
    F: Fn(&str) -> Result<Vec<String>> + Sync,
{
    let hash = RandomState::default();
    let workers = if get_parallelism() {
        current_num_threads()
    } else {
        1
    };
    if workers == 1 {
        return iterator
            .fold(
                Ok(CountMap::with_hasher(hash.clone())),
                |counts, sequence: S| {
                    let words = process(sequence.as_ref());
                    let mut counts = counts?;
                    for word in words? {
                        *counts.entry(CompactString::from(word)).or_default() += 1;
                    }
                    Ok(counts)
                },
            )
            .map(WordCounts::from_map);
    }
    let shared = SharedCounts::with_hasher(hash.clone());
    let accumulate = |state: Result<()>, sequence: S| -> Result<()> {
        // Call every input after ordinary errors; retain this fold's first error.
        let words = process(sequence.as_ref());
        state?;
        for word in words? {
            shared
                .entry_sync(CompactString::from(word))
                .and_modify(|count| *count += 1)
                .or_insert(1);
        }
        Ok(())
    };
    // Seed with single inputs, then amortize the bridge lock across batches.
    let mut iterator = iterator.fuse();
    let mut singles = workers;
    let batches = std::iter::from_fn(move || {
        let size = if singles > 0 { 1 } else { FEED_BATCH_SIZE };
        singles = singles.saturating_sub(1);
        let batch: Vec<_> = iterator.by_ref().take(size).collect();
        (!batch.is_empty()).then_some(batch)
    });
    batches
        .maybe_par_bridge()
        .flat_map_iter(std::iter::IntoIterator::into_iter)
        .fold(|| Ok(()), accumulate)
        .reduce(|| Ok(()), |left, right| left.and(right))?;

    let mut words = Vec::with_capacity(shared.len());
    // Keep table capacity stable while consumption removes its entries.
    let _reservation = shared.reserve(shared.capacity());
    shared.iter_mut_sync(|entry| {
        words.push(entry.consume());
        true
    });
    Ok(WordCounts::from_entries(words))
}
