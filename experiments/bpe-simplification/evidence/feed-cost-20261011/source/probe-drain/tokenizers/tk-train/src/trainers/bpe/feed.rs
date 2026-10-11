//! Local preaggregation into a shared table; feed returns unordered counts.
use super::word_counts::WordCounts;
use ahash::{AHashMap, RandomState};
use compact_str::CompactString;
use std::collections::hash_map::Entry;
use tk_encode::{Result, parallelism::*};

type CountMap = AHashMap<CompactString, u64>;
type SharedCounts = scc::HashMap<CompactString, u64, RandomState>;

// Bound distinct local keys in parallel, not bytes or the serial map.
const LOCAL_KEY_LIMIT: usize = 2048;
const FEED_BATCH_SIZE: usize = 32;

pub(super) fn count<I, S, F>(iterator: I, process: &F) -> Result<WordCounts>
where
    I: Iterator<Item = S> + Send,
    S: AsRef<str> + Send,
    F: Fn(&str) -> Result<Vec<String>> + Sync,
{
    let hash = RandomState::default();
    let trace_start = std::time::Instant::now();
    let workers = if get_parallelism() {
        current_num_threads()
    } else {
        1
    };
    let parallel = workers > 1;
    let shared = parallel.then(|| SharedCounts::with_hasher(hash.clone()));
    let flush = |counts: &mut CountMap| {
        if let Some(shared) = &shared {
            for (word, count) in counts.drain() {
                shared
                    .entry_sync(word)
                    .and_modify(|total| *total += count)
                    .or_insert(count);
            }
        }
    };
    let accumulate = |counts: Result<CountMap>, sequence: S| -> Result<CountMap> {
        // Call every input after ordinary errors; retain this fold's first error.
        let words = process(sequence.as_ref());
        let mut counts = counts?;
        for word in words? {
            match counts.entry(CompactString::from(word)) {
                Entry::Occupied(mut entry) => *entry.get_mut() += 1,
                Entry::Vacant(entry) => {
                    entry.insert(1);
                    // Repeated words need no size check.
                    if counts.len() == LOCAL_KEY_LIMIT {
                        flush(&mut counts);
                    }
                }
            }
        }
        Ok(counts)
    };
    if !parallel {
        return iterator
            .fold(Ok(CountMap::with_hasher(hash.clone())), accumulate)
            .map(WordCounts::from_map);
    }

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
        .fold(|| Ok(CountMap::with_hasher(hash.clone())), accumulate)
        .map(|counts| -> Result<()> {
            let mut counts = counts?;
            flush(&mut counts);
            Ok(())
        })
        // Ordinary reduction completes all callbacks despite ordinary errors.
        .reduce(|| Ok(()), |left, right| left.and(right))?;

    let shared = shared.unwrap();
    trace_table(&shared, "after-aggregation", trace_start);
    let mut words = Vec::with_capacity(shared.len());
    trace_table(&shared, "before-consume", trace_start);
    shared.iter_mut_sync(|entry| {
        words.push(entry.consume());
        true
    });
    trace_table(&shared, "after-consume", trace_start);
    Ok(WordCounts::from_entries(words))
}

fn trace_table(shared: &SharedCounts, phase: &str, start: std::time::Instant) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let value = |name: &str| {
        status
            .lines()
            .find(|line| line.starts_with(name))
            .and_then(|line| line.split_whitespace().nth(1))
            .unwrap()
            .parse::<u64>()
            .unwrap()
    };
    eprintln!(
        "BPE_FEED_TABLE {}",
        serde_json::json!({
            "phase":phase, "elapsed":start.elapsed().as_secs_f64(),
            "capacity":shared.capacity(), "rss_kib":value("VmRSS:"), "hwm_kib":value("VmHWM:")
        })
    );
}
