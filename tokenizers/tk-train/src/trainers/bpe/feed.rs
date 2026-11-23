//! Count a feed with bounded parallel scratch space. Return fresh counts only
//! after every callback has run successfully.

use ahash::{AHashMap, RandomState};
use compact_str::CompactString;
use std::collections::hash_map::Entry;
use std::sync::{Mutex, RwLock};
use tk_encode::Result;
use tk_encode::parallelism::MaybeParallelBridge;

type WordCounts = AHashMap<CompactString, u64>;

const SHARD_COUNT: usize = 16;
const LOCAL_KEY_LIMIT: usize = 2048;
const SHARD_BATCH_SIZE: usize = 32;

type Batch = Vec<(CompactString, u64)>;

pub(super) fn count<I, S, F>(iterator: I, process: &F) -> Result<WordCounts>
where
    I: Iterator<Item = S> + Send,
    S: AsRef<str> + Send,
    F: Fn(&str) -> Result<Vec<String>> + Sync,
{
    let words = iterator
        .maybe_par_bridge()
        .map(|sequence| process(sequence.as_ref()));
    // Keep one map and avoid any shard locks when the bridge is serial.
    if words.is_serial() {
        return words
            .fold(
                || Ok(WordCounts::new()),
                |counts, words| {
                    let mut counts = counts?;
                    for word in words? {
                        *counts.entry(CompactString::from(word)).or_default() += 1;
                    }
                    Ok(counts)
                },
            )
            .reduce(|| Ok(WordCounts::new()), Result::and);
    }

    let shared = ShardedCounts::default();
    words
        .fold(
            || -> Result<_> { Ok(LocalCounts::default()) },
            |counts, words| {
                let mut counts = counts?;
                counts.add(words.inspect_err(|_| shared.discard())?, &shared);
                Ok(counts)
            },
        )
        .map(|counts| {
            shared.flush(&mut counts?);
            Ok(())
        })
        .reduce(|| Ok(()), Result::and)?;
    Ok(shared.into_counts())
}

#[derive(Default)]
struct LocalCounts {
    words: WordCounts,
    batches: [Batch; SHARD_COUNT],
}

impl LocalCounts {
    fn add(&mut self, words: Vec<String>, shared: &ShardedCounts) {
        for word in words {
            match self.words.entry(CompactString::from(word)) {
                Entry::Occupied(mut entry) => *entry.get_mut() += 1,
                Entry::Vacant(entry) => {
                    entry.insert(1);
                    if self.words.len() == LOCAL_KEY_LIMIT {
                        shared.flush(self);
                    }
                }
            }
        }
    }
}

type Shards = [Mutex<WordCounts>; SHARD_COUNT];

struct ShardedCounts {
    routing_hash: RandomState,
    shards: RwLock<Option<Shards>>,
}

impl Default for ShardedCounts {
    fn default() -> Self {
        Self {
            routing_hash: RandomState::new(),
            shards: RwLock::new(Some(Default::default())),
        }
    }
}

impl ShardedCounts {
    fn discard(&self) {
        let shards = self.shards.write().unwrap().take();
        drop(shards);
    }

    fn flush(&self, local: &mut LocalCounts) {
        // Discard takes the write lock, so no flush can outlive the tables.
        let shards = self.shards.read().unwrap();
        let Some(shards) = shards.as_ref() else {
            local.words.clear();
            return;
        };
        for (word, count) in local.words.drain() {
            let shard = self.routing_hash.hash_one(&word) as usize % SHARD_COUNT;
            let batch = &mut local.batches[shard];
            batch.push((word, count));
            if batch.len() == SHARD_BATCH_SIZE {
                Self::merge(&shards[shard], batch);
            }
        }
        for (shard, batch) in shards.iter().zip(&mut local.batches) {
            if !batch.is_empty() {
                Self::merge(shard, batch);
            }
        }
    }

    fn merge(shard: &Mutex<WordCounts>, batch: &mut Batch) {
        let mut counts = shard.lock().unwrap();
        for (word, count) in batch.drain(..) {
            *counts.entry(word).or_default() += count;
        }
    }

    fn into_counts(self) -> WordCounts {
        let mut shards = self
            .shards
            .into_inner()
            .unwrap()
            .expect("successful feed retains its shards")
            .map(|shard| shard.into_inner().unwrap());
        // Shard keys are disjoint, so the largest map can become the output.
        let largest = shards.iter_mut().max_by_key(|shard| shard.len()).unwrap();
        let mut counts = std::mem::take(largest);
        // Grow as needed so consumed shard allocations are released along the way.
        for shard in shards {
            counts.extend(shard);
        }
        counts
    }
}

#[cfg(test)]
mod tests;
