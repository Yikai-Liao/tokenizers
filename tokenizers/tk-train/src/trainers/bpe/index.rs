//! Authoritative weighted counts and owning historical word-domain candidates.
use super::corpus::{Corpus, WordChange};
use super::positions::Positions;
use crate::progress::ProgressBar;
use ahash::AHashMap;
use dary_heap::OctonaryHeap;
use std::cmp::Ordering;
use tk_encode::models::bpe::Pair;

#[derive(Debug, Eq)]
pub(super) struct Candidate {
    pub(super) pair: Pair,
    count: u64,
    pub(super) positions: Positions,
}

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
        if self.count != other.count {
            self.count.cmp(&other.count)
        } else {
            // Here we want ascending order
            other.pair.cmp(&self.pair)
        }
    }
}

pub(super) struct PairIndex {
    counts: AHashMap<Pair, i32>,
    queue: OctonaryHeap<Candidate>,
    minimum: u64,
}

impl PairIndex {
    pub(super) fn build(corpus: &Corpus, minimum: u64, progress: &Option<ProgressBar>) -> Self {
        let (counts, mut births) = corpus.count_pairs(progress);
        let mut queue = OctonaryHeap::with_capacity(counts.len());
        for (pair, pos) in births.drain() {
            let count = counts[&pair];
            if count > 0 {
                queue.push(Candidate::new(pair, count as u64, pos));
            }
        }
        Self {
            counts,
            queue,
            minimum,
        }
    }

    pub(super) fn take(&mut self) -> Option<Candidate> {
        loop {
            let mut top = self.queue.pop()?;
            if top.count != self.counts[&top.pair] as u64 {
                top.count = self.counts[&top.pair] as u64;
                self.queue.push(top);
                continue;
            }
            return (top.count >= self.minimum.max(1)).then_some(top);
        }
    }

    pub(super) fn commit(&mut self, changes: Vec<WordChange>) {
        let mut births: AHashMap<Pair, Vec<u64>> = AHashMap::new();
        for change in changes {
            let count = change.delta * change.weight as i32;
            *self.counts.entry(change.pair).or_default() += count;
            if change.delta > 0 {
                births
                    .entry(change.pair)
                    .or_default()
                    .push(change.coordinate);
            }
        }
        for (pair, pos) in births.drain() {
            let count = self.counts[&pair];
            if count > 0 {
                self.queue.push(Candidate::new(pair, count as u64, pos));
            }
        }
    }
}

impl Candidate {
    fn new(pair: Pair, count: u64, positions: Vec<u64>) -> Self {
        // Keep the cohort boundary; only its own coordinate order is normalized.
        Self {
            pair,
            count,
            positions: Positions::from_unsorted(positions),
        }
    }
}
