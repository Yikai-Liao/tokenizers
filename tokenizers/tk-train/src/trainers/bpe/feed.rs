//! YTTM-style static input shards, one word-count table per worker, then one
//! global union. Tokenizers' callback owns normalization and word boundaries.
use ahash::AHashMap;
use compact_str::CompactString;
use tk_encode::Result;

pub(super) fn count<I, S, F>(
    iterator: I,
    process: F,
    workers: usize,
) -> Result<AHashMap<CompactString, u64>>
where
    I: Iterator<Item = S> + Send,
    S: AsRef<str> + Send,
    F: Fn(&str) -> Result<Vec<String>> + Sync,
{
    // YTTM retains its input while statically partitioning it. Moving each
    // sequence into one shard also avoids requiring Sync on the public S type.
    let sequences: Vec<_> = iterator.collect();
    let workers = workers.min(sequences.len().max(1)).max(1);
    let chunk = sequences.len().div_ceil(workers).max(1);
    let mut shards: Vec<Vec<S>> = (0..workers).map(|_| Vec::new()).collect();
    for (i, sequence) in sequences.into_iter().enumerate() {
        shards[(i / chunk).min(workers - 1)].push(sequence);
    }
    let maps = std::thread::scope(|scope| {
        let handles: Vec<_> = shards
            .into_iter()
            .map(|shard| {
                let process = &process;
                scope.spawn(move || -> Result<AHashMap<CompactString, u64>> {
                    let mut words: AHashMap<CompactString, u64> = AHashMap::new();
                    for sequence in shard {
                        for word in process(sequence.as_ref())? {
                            let count = words.entry(CompactString::from(word)).or_default();
                            *count = count
                                .checked_add(1)
                                .ok_or("BPE word frequency exceeds u64")?;
                        }
                    }
                    Ok(words)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().map_err(|_| "YTTM feed worker panicked")?)
            .collect::<Result<Vec<_>>>()
    })?;
    let mut maps = maps.into_iter();
    let mut words = maps.next().unwrap_or_default();
    for map in maps {
        for (word, count) in map {
            let total = words.entry(word).or_default();
            *total = total
                .checked_add(count)
                .ok_or("BPE word frequency exceeds u64")?;
        }
    }
    Ok(words)
}
