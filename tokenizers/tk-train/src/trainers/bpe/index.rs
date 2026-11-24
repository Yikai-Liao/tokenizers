//! Weighted pair counts and lazy merge priority.

use super::corpus::{Corpus, WordChange};
use super::positions::Positions;
use crate::progress::ProgressBar;
use ahash::AHashMap;
use dary_heap::OctonaryHeap;
use std::cmp::Ordering;
use tk_encode::Result;
use tk_encode::models::bpe::Pair;

pub(super) struct Candidate {
    pub(super) pair: Pair,
    count: u64,
    pub(super) positions: Positions,
}

impl Eq for Candidate {}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.pair == other.pair
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count
            .cmp(&other.count)
            .then_with(|| other.pair.cmp(&self.pair))
    }
}

pub(super) struct PairIndex {
    // Whole-word application can remove a neighbor excluded by the strict birth limit.
    counts: AHashMap<Pair, i128>,
    queue: OctonaryHeap<Candidate>,
    minimum: u64,
}

impl PairIndex {
    pub(super) fn build(
        corpus: &Corpus,
        minimum: u64,
        progress: &Option<ProgressBar>,
    ) -> Result<Self> {
        let pairs = corpus.count_pairs(progress)?;
        let mut counts = AHashMap::with_capacity(pairs.len());
        let mut queue = OctonaryHeap::with_capacity(pairs.len());

        for (pair, (count, positions)) in pairs {
            counts.insert(pair, i128::from(count));
            if count > 0 {
                let positions = Positions::from_unsorted(positions)?;
                queue.push(Candidate {
                    pair,
                    count,
                    positions,
                });
            }
        }

        Ok(Self {
            counts,
            queue,
            minimum,
        })
    }

    pub(super) fn take_best(&mut self) -> Option<Candidate> {
        loop {
            let mut candidate = self.queue.pop()?;
            let current = *self.counts.get(&candidate.pair).unwrap_or(&0);
            // Preserve the upstream i32-to-u64 priority for a negative legacy ledger entry.
            let current_priority = current as u64;
            if candidate.count != current_priority {
                candidate.count = current_priority;
                self.queue.push(candidate);
                continue;
            }

            if candidate.count < self.minimum.max(1) {
                return None;
            }
            return Some(candidate);
        }
    }

    pub(super) fn commit(&mut self, changes: Vec<WordChange>) -> Result<()> {
        let mut births: AHashMap<Pair, Vec<u64>> = AHashMap::new();
        for change in changes {
            let delta = i128::from(change.weight);
            let delta = if change.added { delta } else { -delta };
            let count = self.counts.entry(change.pair).or_default();
            *count = count
                .checked_add(delta)
                .ok_or("BPE pair count exceeds the checked ledger range")?;
            if *count > i128::from(u64::MAX) {
                return Err("BPE pair count exceeds u64".into());
            }
            if change.added {
                births
                    .entry(change.pair)
                    .or_default()
                    .push(change.coordinate);
            }
        }

        for (pair, positions) in births {
            let count = *self.counts.get(&pair).unwrap_or(&0);
            if count > 0 {
                self.queue.push(Candidate {
                    pair,
                    count: u64::try_from(count).map_err(|_| "BPE pair count exceeds u64")?,
                    positions: Positions::from_unsorted(positions)?,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{PairIndex, WordChange};
    use ahash::AHashMap;
    use dary_heap::OctonaryHeap;

    #[test]
    fn ledger_accepts_u64_max_and_rejects_the_next_increment() {
        let pair = (1, 2);
        let mut index = PairIndex {
            counts: AHashMap::from_iter([(pair, i128::from(u64::MAX) - 1)]),
            queue: OctonaryHeap::new(),
            minimum: 1,
        };
        index
            .commit(vec![WordChange {
                pair,
                added: true,
                coordinate: 4,
                weight: 1,
            }])
            .unwrap();
        assert_eq!(index.counts[&pair], i128::from(u64::MAX));

        let error = index
            .commit(vec![WordChange {
                pair,
                added: true,
                coordinate: 8,
                weight: 1,
            }])
            .unwrap_err();
        assert_eq!(error.to_string(), "BPE pair count exceeds u64");
    }

    #[test]
    fn signed_ledger_keeps_legacy_unindexed_neighbor_removals() {
        let pair = (1, 2);
        let mut index = PairIndex {
            counts: AHashMap::from_iter([(pair, 0)]),
            queue: OctonaryHeap::new(),
            minimum: 1,
        };
        index
            .commit(vec![WordChange {
                pair,
                added: false,
                coordinate: 4,
                weight: u64::MAX,
            }])
            .unwrap();
        assert_eq!(index.counts[&pair], -i128::from(u64::MAX));
    }
}
