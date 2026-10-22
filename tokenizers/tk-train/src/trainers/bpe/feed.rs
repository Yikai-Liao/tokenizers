//! Local preaggregation into a shared table; feed returns unordered counts.
use ahash::{AHashMap, RandomState};
use compact_str::CompactString;
use std::collections::hash_map::Entry;
use tk_encode::Result;
use tk_encode::parallelism::MaybeParallelBridge;

type CountMap = AHashMap<CompactString, u64>;
type SharedCounts = scc::HashMap<CompactString, u64, RandomState>;

// Bound distinct local keys in parallel, not bytes or the serial map.
const LOCAL_KEY_LIMIT: usize = 2048;
const FEED_BATCH_SIZE: usize = 32;

pub(super) fn count<I, S, F>(iterator: I, process: &F) -> Result<CountMap>
where
    I: Iterator<Item = S> + Send,
    S: AsRef<str> + Send,
    F: Fn(&str) -> Result<Vec<String>> + Sync,
{
    let hash = RandomState::default();
    let workers = super::num_workers();
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
        return iterator.fold(Ok(CountMap::with_hasher(hash.clone())), accumulate);
    }

    // Seed with single inputs, then amortize the bridge lock across batches.
    let mut iterator = iterator.fuse();
    let mut singles = workers;
    let batches = std::iter::from_fn(move || {
        let size = if singles > 0 { 1 } else { FEED_BATCH_SIZE };
        // Exhaust the one-input seed budget without wrapping: once it reaches
        // zero, every subsequent bridge item keeps the regular batch size.
        singles = singles.saturating_sub(1);
        let batch: Vec<_> = iterator.by_ref().take(size).collect();
        (!batch.is_empty()).then_some(batch)
    });
    batches
        .maybe_par_bridge_cond(parallel)
        .flat_map_iter(std::iter::IntoIterator::into_iter)
        .fold(|| Ok(CountMap::with_hasher(hash.clone())), accumulate)
        .map(|counts| -> Result<()> {
            let mut counts = counts?;
            flush(&mut counts);
            Ok(())
        })
        // Both fold results already exist: left.and(right) keeps the left error
        // when present, otherwise the right result. Reduction does not skip callbacks.
        .reduce(|| Ok(()), |left, right| left.and(right))?;

    let shared = shared.expect("parallel feed initializes its shared count table");
    let mut words = CountMap::with_capacity(shared.len());
    // No writers remain; consume moves each visited entry out of the table.
    shared.iter_mut_sync(|entry| {
        let (word, count) = entry.consume();
        words.insert(word, count);
        true
    });
    Ok(words)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BpeTrainer, Trainer};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn counts_survive_flush_boundaries() {
        for unique in [0, 2047, 2048, 2049, 4097] {
            let mut input: Vec<_> = (0..unique).map(|i| format!("词{i}🙂")).collect();
            input.extend(["repeat".into(), "repeat".into()]);
            let actual = count(input.iter(), &|text| Ok(vec![text.into()])).unwrap();
            assert_eq!(actual.len(), unique + 1);
            assert_eq!(actual["repeat"], 2);
            assert!(
                actual
                    .iter()
                    .all(|(key, &value)| key == "repeat" || value == 1)
            );
        }
    }

    #[test]
    fn errors_preserve_stored_counts_and_visit_remaining_inputs() {
        let mut trainer = BpeTrainer::default();
        trainer
            .feed(["saved"].into_iter(), |s| Ok(vec![s.into()]))
            .unwrap();
        let previous = trainer.clone();
        let called = AtomicUsize::new(0);
        let result = trainer.feed(["ok", "error", "tail"].into_iter(), |text| {
            called.fetch_add(1, Ordering::Relaxed);
            if text == "error" {
                Err("expected".into())
            } else {
                Ok(vec![text.into()])
            }
        });
        assert!(result.is_err());
        assert_eq!(called.load(Ordering::Relaxed), 3);
        assert_eq!(trainer, previous);
        trainer
            .feed(std::iter::empty::<&str>(), |_| unreachable!())
            .unwrap();
        assert_eq!(trainer.get_word_count(), 0);
    }
}
