//! Authoritative weighted counts and owning historical word-domain candidates.
use super::TrainingMode;
use super::corpus::{CorpusPlan, InitialChunk, InitialEdge, ProducerChanges};
use super::positions::Positions;
use crate::progress::ProgressBar;
use ahash::AHashMap;
use dary_heap::OctonaryHeap;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::cmp::{Ordering, Reverse};
use tk_encode::Result;
use tk_encode::models::bpe::Pair;
use tk_encode::parallelism;
use tk_encode::parallelism::MaybeParallelIterator;

/// Queue key: highest count first, then smallest pair.
/// Field order defines heap priority; occurrence payloads never break ties.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(super) struct Priority {
    count: u64,
    pair: Reverse<Pair>,
}

impl Priority {
    #[cfg(test)]
    pub(super) fn count(self) -> u64 {
        self.count
    }

    pub(super) fn pair(self) -> Pair {
        self.pair.0
    }
}

/// Owning queue entry: a possibly stale count snapshot and its occurrence list.
/// PairIndex corrects the priority against the count shard before selection;
/// positions remain a historical cohort when identities can be reused.
pub(super) struct Candidate<P = Positions> {
    pub(super) priority: Priority,
    pub(super) positions: P,
}

impl Candidate {
    fn new(pair: Pair, count: u64, positions: Positions) -> Self {
        Self {
            priority: Priority {
                count,
                pair: Reverse(pair),
            },
            positions,
        }
    }
}

impl<P> Eq for Candidate<P> {}

impl<P> PartialEq for Candidate<P> {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl<P> Ord for Candidate<P> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.cmp(&other.priority)
    }
}

impl<P> PartialOrd for Candidate<P> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Move priority/index entries through the heap while payloads stay in one owner.
/// Vacant slots are reused; each live entry exclusively owns its indexed list.
#[derive(Default)]
struct CandidateQueue {
    heap: OctonaryHeap<Candidate<usize>>,
    positions: Vec<Positions>,
    vacant: Vec<usize>,
}

impl FromIterator<Candidate> for CandidateQueue {
    fn from_iter<T: IntoIterator<Item = Candidate>>(candidates: T) -> Self {
        let candidates = candidates.into_iter();
        let capacity = candidates.size_hint().0;
        let mut heap = OctonaryHeap::with_capacity(capacity);
        let mut positions = Vec::with_capacity(capacity);
        for candidate in candidates {
            let slot = positions.len();
            positions.push(candidate.positions);
            heap.push(Candidate {
                priority: candidate.priority,
                positions: slot,
            });
        }
        Self {
            heap,
            positions,
            vacant: Vec::new(),
        }
    }
}

impl CandidateQueue {
    fn peek(&self) -> Option<Priority> {
        self.heap.peek().map(|candidate| candidate.priority)
    }

    fn push(&mut self, candidate: Candidate) {
        let Candidate {
            priority,
            positions,
        } = candidate;
        let slot = if let Some(slot) = self.vacant.pop() {
            self.positions[slot] = positions;
            slot
        } else {
            self.positions.push(positions);
            self.positions.len() - 1
        };
        self.heap.push(Candidate {
            priority,
            positions: slot,
        });
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<Candidate> {
        let candidate = self.heap.pop()?;
        let positions = std::mem::take(&mut self.positions[candidate.positions]);
        self.vacant.push(candidate.positions);
        Some(Candidate {
            priority: candidate.priority,
            positions,
        })
    }

    fn rekey(&mut self, count: u64) {
        // Pop/update/push retains the historical cohort tie-selection order.
        let mut candidate = self.heap.pop().expect("observed candidate exists");
        candidate.priority.count = count;
        self.heap.push(candidate);
    }

    fn retire(self) {
        let Self {
            heap,
            positions,
            vacant,
        } = self;
        drop(heap);
        drop(vacant);
        if parallelism::get_parallelism() {
            positions.into_par_iter().with_min_len(4096).for_each(drop);
        } else {
            drop(positions);
        }
    }
}

pub(super) struct PairIndex {
    shard: Shard,
    queue: CandidateQueue,
    minimum: u64,
    mode: TrainingMode,
}

impl PairIndex {
    pub(super) fn build(
        corpus: &CorpusPlan<'_>,
        minimum: u64,
        progress: &Option<ProgressBar>,
    ) -> Result<Self> {
        let slots = if cfg!(test) {
            16
        } else {
            corpus
                .slot_count()
                .div_ceil(super::num_workers())
                .clamp(1, 1 << 24)
        };
        let pieces = corpus
            .chunks(slots)
            .into_maybe_par_iter()
            .map(|chunk| count_chunk(corpus.token_count(), chunk, progress))
            .collect::<Result<Vec<_>>>()?;
        let mut states = AHashMap::<Pair, PairAggregate<SmallVec<[Positions; 1]>>>::new();
        for piece in pieces {
            for (pair, state) in piece {
                let total = states.entry(pair).or_default();
                super::add(&mut total.count, state.count)?;
                total.positions.push(state.positions);
            }
        }
        let mut counts = AHashMap::new();
        let mut candidates = Vec::new();
        for (pair, state) in states {
            if corpus.mode() == TrainingMode::Fresh && state.count < minimum.max(1) {
                continue;
            }
            counts.insert(pair, state.count);
            if state.count != 0 {
                let positions = Positions::concat(state.positions)?;
                candidates.push(Candidate::new(pair, state.count, positions));
            }
        }
        Ok(Self {
            shard: Shard {
                counts,
                ..Shard::default()
            },
            queue: candidates.into_iter().collect(),
            minimum,
            mode: corpus.mode(),
        })
    }

    pub(super) fn take(&mut self) -> Option<Candidate> {
        loop {
            let top = self.queue.peek()?;
            let Some(&count) = self.shard.counts.get(&top.pair()) else {
                self.queue.pop();
                continue;
            };
            if top.count != count {
                self.queue.rekey(count);
                continue;
            }
            if count < self.minimum.max(1) {
                return None;
            }
            let candidate = self.queue.pop()?;
            if self.mode == TrainingMode::Fresh {
                self.shard.counts.remove(&candidate.priority.pair());
            }
            return Some(candidate);
        }
    }

    pub(super) fn commit(&mut self, changes: Vec<ProducerChanges>) -> Result<()> {
        for producer in changes {
            for change in producer.changes {
                if producer.whole_rule
                    && change.born.0 != change.born.1
                    && !change.positions.is_empty()
                {
                    if change.born_weight >= self.minimum.max(1) {
                        self.shard.completed.push(Candidate::new(
                            change.born,
                            change.born_weight,
                            change.positions,
                        ));
                    }
                    if change.removed_weight != 0 {
                        self.shard.actions.push(PendingAction::Remove {
                            pair: change.removed,
                            weight: change.removed_weight,
                        });
                    }
                    continue;
                }
                if change.removed_weight != 0 {
                    self.shard.actions.push(PendingAction::Remove {
                        pair: change.removed,
                        weight: change.removed_weight,
                    });
                }
                if !change.positions.is_empty() {
                    self.shard.actions.push(PendingAction::Birth {
                        pair: change.born,
                        weight: change.born_weight,
                        birth_group: change.birth_group,
                    });
                    self.shard.positions.push(change.positions);
                }
            }
        }
        self.shard
            .commit(self.mode == TrainingMode::Reuse, self.minimum.max(1))?;
        for candidate in self.shard.completed.drain(..) {
            self.queue.push(candidate);
        }
        Ok(())
    }
}

fn adjust_signed(count: &mut u64, amount: u64, remove: bool) -> Result<()> {
    // Reuse keeps a signed ledger in u64 bits to preserve the queue's unsigned
    // ordering. Check each action, not its net delta: intermediate overflow matters.
    let amount = i64::try_from(amount).map_err(|_| "BPE identity-reuse adjustment exceeds i64")?;
    let delta = if remove { -amount } else { amount };
    *count = (*count as i64)
        .checked_add(delta)
        .ok_or("BPE identity-reuse count adjustment exceeds i64")? as u64;
    Ok(())
}

impl PairIndex {
    pub(super) fn retire(self) {
        let Self { shard, queue, .. } = self;
        drop(shard);
        queue.retire();
    }
}
#[cfg(test)]
mod queue_tests {
    use super::*;
    #[test]
    fn equal_priority_cohorts_keep_pop_update_push_order() {
        let make = |p| Candidate::new((0, 1), 9, Positions::from_unsorted(vec![p]).unwrap());
        let mut reference = OctonaryHeap::new();
        for candidate in [make(1), make(2), make(3)] {
            reference.push(candidate);
        }
        let mut queue: CandidateQueue = [make(1), make(2), make(3)].into_iter().collect();
        let mut top = reference.pop().unwrap();
        top.priority.count = 7;
        reference.push(top);
        queue.rekey(7);
        for _ in 0..3 {
            let old = reference.pop().unwrap();
            let new = queue.pop().unwrap();
            assert_eq!(old.priority.count(), new.priority.count());
            assert_eq!(old.priority.pair(), new.priority.pair());
            assert_eq!(
                old.positions.iter().collect::<Vec<_>>(),
                new.positions.iter().collect::<Vec<_>>()
            );
        }
        assert!(queue.pop().is_none());
    }
}

#[derive(Default)]
struct PairAggregate<P> {
    count: u64,
    positions: P,
}
impl PairAggregate<Positions> {
    #[inline(always)]
    fn record(&mut self, edge: InitialEdge) -> Result<()> {
        debug_assert!(edge.pair.0 != u32::MAX && edge.pair.1 != u32::MAX);
        super::add(&mut self.count, edge.weight)?;
        // Keep checked count addition before appending this ordered coordinate.
        self.positions.push(edge.coordinate)
    }
}

type InitialPiece = Vec<(Pair, PairAggregate<Positions>)>;
enum InitialPairCounts {
    Dense {
        domain: usize,
        states: Vec<PairAggregate<Positions>>,
    },
    Sparse(AHashMap<Pair, PairAggregate<Positions>>),
}

impl InitialPairCounts {
    fn new(domain: usize) -> Self {
        // The dense plane has one cell for every ordered pair: domain squared.
        // A 256-ID cutoff limits it to 65,536 cells; larger alphabets stay sparse.
        if domain <= 256 {
            Self::Dense {
                domain,
                states: (0..domain * domain)
                    .map(|_| PairAggregate::default())
                    .collect(),
            }
        } else {
            Self::Sparse(AHashMap::new())
        }
    }

    fn scan(&mut self, chunk: InitialChunk<'_, '_>) -> Result<usize> {
        // Pick the representation once; each visitor borrows only concrete storage.
        match self {
            Self::Dense { domain, states } => {
                let domain = *domain;
                let states = states.as_mut_slice();
                chunk.scan(
                    #[inline(always)]
                    |edge| {
                        // The IDs choose one row and column in this fixed dense plane.
                        states[edge.pair.0 as usize * domain + edge.pair.1 as usize].record(edge)
                    },
                )
            }
            Self::Sparse(states) => chunk.scan(
                #[inline(always)]
                |edge| states.entry(edge.pair).or_default().record(edge),
            ),
        }
    }

    fn into_entries(self) -> Vec<(Pair, PairAggregate<Positions>)> {
        // Invert the dense row-major key: quotient is the left ID, remainder
        // is the right ID. Sparse keys already carry their ordered pair.
        match self {
            Self::Dense { domain, states } => states
                .into_iter()
                .enumerate()
                .filter(|(_, state)| !state.positions.is_empty())
                .map(|(key, state)| (((key / domain) as u32, (key % domain) as u32), state))
                .collect(),
            Self::Sparse(states) => states.into_iter().collect(),
        }
    }
}
fn count_chunk(
    domain: usize,
    chunk: InitialChunk<'_, '_>,
    progress: &Option<ProgressBar>,
) -> Result<InitialPiece> {
    let mut counts = InitialPairCounts::new(domain);
    let words = counts.scan(chunk)?;
    if let Some(p) = progress {
        p.inc(words as u64);
    }
    Ok(counts.into_entries())
}

#[derive(Default)]
struct Shard {
    counts: AHashMap<Pair, u64>,
    actions: Vec<PendingAction>,
    // One stream per birth action, in the same order.
    positions: Vec<Positions>,
    completed: Vec<Candidate>,
}
enum PendingAction {
    Remove {
        pair: Pair,
        weight: u64,
    },
    Birth {
        pair: Pair,
        weight: u64,
        birth_group: usize,
    },
}
struct BirthFragment {
    birth_group: usize,
    pair: Pair,
    weight: u64,
    positions: Positions,
}

impl Shard {
    fn commit(&mut self, reuse: bool, floor: u64) -> Result<()> {
        let mut fragments = Vec::with_capacity(self.positions.len());
        // Drains empty pending actions and streams on error; unpublished births
        // stay with this shard until the failed attempt is discarded.
        let mut positions = self.positions.drain(..);
        for action in self.actions.drain(..) {
            // A boundary removal precedes its replacement birth. The signed reuse
            // ledger checks each action in order, including intermediate overflow.
            match action {
                PendingAction::Remove { pair, weight } => {
                    if reuse {
                        let count = self.counts.entry(pair).or_default();
                        adjust_signed(count, weight, true)?;
                    } else if let Some(count) = self.counts.get_mut(&pair) {
                        *count = count
                            .checked_sub(weight)
                            .ok_or("BPE fresh removal exceeds the current count")?;
                        if *count < floor {
                            self.counts.remove(&pair);
                        }
                    }
                }
                PendingAction::Birth {
                    pair,
                    weight,
                    birth_group,
                } => {
                    if reuse {
                        let count = self.counts.entry(pair).or_default();
                        adjust_signed(count, weight, false)?;
                    }
                    fragments.push(BirthFragment {
                        birth_group,
                        pair,
                        weight,
                        positions: positions.next().expect("each birth owns one stream"),
                    });
                }
            }
        }

        for candidate in &self.completed {
            debug_assert!(!reuse);
            let pair = candidate.priority.pair();
            debug_assert!(!self.counts.contains_key(&pair));
            self.counts.insert(pair, candidate.priority.count);
        }
        // Stable key order keeps each cohort's spatial fragment order while
        // publishing in birth_group/pair order. Reuse streams may interleave.
        fragments.sort_by_key(|fragment| (fragment.birth_group, fragment.pair));
        let mut fragments = fragments.into_iter();
        while let Some(fragment) = fragments.as_slice().first() {
            let key = (fragment.birth_group, fragment.pair);
            let mut group_len = 0;
            let mut weight = 0;
            for part in fragments
                .as_slice()
                .iter()
                .take_while(|part| (part.birth_group, part.pair) == key)
            {
                super::add(&mut weight, part.weight)?;
                group_len += 1;
            }
            let pair = key.1;
            let count = if reuse { self.counts[&pair] } else { weight };
            let streams = fragments
                .by_ref()
                .take(group_len)
                .map(|fragment| fragment.positions);
            // A positive reuse ledger publishes a historical cohort even below
            // the floor; selection applies the floor after correcting its head.
            // Interpret reuse counts as signed bits, rejecting zero/negative
            // ledgers. Fresh counts are unsigned and must meet the floor now.
            if (reuse && (count as i64) <= 0) || (!reuse && count < floor) {
                streams.for_each(drop);
                continue;
            }
            let positions = if reuse {
                Positions::merge(streams)?
            } else {
                Positions::concat(streams)?
            };
            if !reuse {
                debug_assert!(!self.counts.contains_key(&pair));
                self.counts.insert(pair, count);
            }
            self.completed.push(Candidate::new(pair, count, positions));
        }
        Ok(())
    }
}
