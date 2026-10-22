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
use tk_encode::parallelism::{MaybeParallelIterator, MaybeParallelRefMutIterator};

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
        let mut queue = Self {
            heap: OctonaryHeap::with_capacity(capacity),
            positions: Vec::with_capacity(capacity),
            vacant: Vec::new(),
        };
        for candidate in candidates {
            queue.push(candidate);
        }
        queue
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
    shards: Box<[Shard]>,
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
        let mode = corpus.mode();
        let reuse = mode == TrainingMode::Reuse;
        let shard_count = super::num_workers();
        let domain = corpus.token_count();
        // 1. Count and freeze each whole-word range independently.
        // Bound each range to 2^24 resident slots; divide smaller inputs among
        // workers. Tests use 16 slots to force boundaries on small fixtures.
        let pieces = corpus
            .chunks(if cfg!(test) {
                16
            } else {
                corpus.slot_count().div_ceil(shard_count).clamp(1, 1 << 24)
            })
            .into_maybe_par_iter()
            .map(|chunk| count_chunk(domain, chunk, progress))
            .collect::<Result<Vec<_>>>()?;

        // 2. Route ordered fragments to the sole count owner of each pair.
        let mut routed: Vec<Vec<_>> = (0..shard_count).map(|_| Vec::new()).collect();
        for piece in pieces {
            for (pair, state) in piece {
                routed[owner(pair, shard_count)].push((pair, state));
            }
        }

        // 3. Aggregate counts, admit pairs, and concatenate their sorted fragments.
        let owners = routed
            .into_maybe_par_iter()
            .map(|pieces| build_owner(pieces, minimum.max(1), reuse))
            .collect::<Result<Vec<_>>>()?;
        let (shards, candidates): (Vec<_>, Vec<_>) = owners.into_iter().unzip();
        Ok(Self {
            shards: shards.into_boxed_slice(),
            queue: candidates.into_iter().flatten().collect(),
            minimum: minimum.max(1),
            mode,
        })
    }

    pub(super) fn take(&mut self) -> Candidate {
        let candidate = self.queue.pop().expect("certified candidate exists");
        if self.mode == TrainingMode::Fresh {
            let pair = candidate.priority.pair();
            let shard_count = self.shards.len();
            self.shards[owner(pair, shard_count)]
                .counts
                .remove(&pair)
                .expect("certified pair exists");
        }
        candidate
    }

    pub(super) fn commit(&mut self, changes: Vec<ProducerChanges>) -> Result<()> {
        // 1. Route each removal and birth to its count owner.
        self.route(changes);

        // 2. Apply owner-local actions and encode their birth cohorts in parallel.
        let reuse = self.mode == TrainingMode::Reuse;
        let floor = self.minimum.max(1);
        self.shards
            .maybe_par_iter_mut()
            .filter(|shard| !shard.actions.is_empty() || !shard.completed.is_empty())
            .map(|shard| shard.commit(reuse, floor))
            .collect::<Result<()>>()?;

        // 3. Publish only after all owners join, in the same owner traversal order.
        for shard in self.shards.iter_mut() {
            for candidate in shard.completed.drain(..) {
                self.queue.push(candidate);
            }
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
        let Self { shards, queue, .. } = self;
        drop(shards);
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
        let mut fragments = self.apply_actions(reuse, floor)?;
        for candidate in &self.completed {
            debug_assert!(!reuse);
            let pair = candidate.priority.pair();
            debug_assert!(!self.counts.contains_key(&pair));
            self.counts.insert(pair, candidate.priority.count);
        }
        // Stable key order preserves spatial fragment order within each cohort.
        fragments.sort_by_key(|fragment| (fragment.birth_group, fragment.pair));
        let mut fragments = fragments.into_iter();
        while !fragments.as_slice().is_empty() {
            let (pair, group_len, weight) = BirthFragment::group_summary(fragments.as_slice())?;
            let streams = fragments
                .by_ref()
                .take(group_len)
                .map(|part| part.positions);
            self.publish_birth(pair, weight, streams, reuse, floor)?;
        }
        Ok(())
    }

    fn apply_actions(&mut self, reuse: bool, floor: u64) -> Result<Vec<BirthFragment>> {
        let mut fragments = Vec::with_capacity(self.positions.len());
        // Drains empty pending actions and streams on error; unpublished births
        // stay with this shard until the failed attempt is discarded.
        let mut positions = self.positions.drain(..);
        for action in self.actions.drain(..) {
            // A boundary removal precedes its replacement birth. The signed reuse
            // ledger checks each action in order, including intermediate overflow.
            match action {
                PendingAction::Remove { pair, weight } => {
                    remove_count(&mut self.counts, pair, weight, reuse, floor)?;
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

        Ok(fragments)
    }

    fn publish_birth(
        &mut self,
        pair: Pair,
        weight: u64,
        streams: impl Iterator<Item = Positions>,
        reuse: bool,
        floor: u64,
    ) -> Result<()> {
        // A positive reuse ledger publishes a historical cohort even below
        // the floor; selection applies the floor after correcting its head.
        // Interpret reuse counts as signed bits, rejecting zero/negative
        // ledgers. Fresh counts are unsigned and must meet the floor now.
        let Some(count) = self.birth_count(pair, weight, reuse, floor) else {
            streams.for_each(drop);
            return Ok(());
        };
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
        Ok(())
    }

    fn birth_count(&self, pair: Pair, weight: u64, reuse: bool, floor: u64) -> Option<u64> {
        if reuse {
            let count = self.counts[&pair];
            ((count as i64) > 0).then_some(count)
        } else {
            (weight >= floor).then_some(weight)
        }
    }
}

// Apply each removal before its replacement birth: a net delta would hide
// intermediate signed overflow and change fresh below-floor retirement.
fn remove_count(
    counts: &mut AHashMap<Pair, u64>,
    pair: Pair,
    weight: u64,
    reuse: bool,
    floor: u64,
) -> Result<()> {
    if reuse {
        return adjust_signed(counts.entry(pair).or_default(), weight, true);
    }
    if let Some(count) = counts.get_mut(&pair) {
        *count = count
            .checked_sub(weight)
            .ok_or("BPE fresh removal exceeds the current count")?;
        if *count < floor {
            counts.remove(&pair);
        }
    }
    Ok(())
}

impl BirthFragment {
    fn group_summary(fragments: &[Self]) -> Result<(Pair, usize, u64)> {
        let first = &fragments[0];
        let key = (first.birth_group, first.pair);
        let mut len = 0;
        let mut weight = 0;
        for part in fragments
            .iter()
            .take_while(|part| (part.birth_group, part.pair) == key)
        {
            super::add(&mut weight, part.weight)?;
            len += 1;
        }
        Ok((first.pair, len, weight))
    }
}

fn owner(pair: Pair, shard_count: usize) -> usize {
    // Zero seeds keep routing repeatable within this build, without per-call
    // randomness. Hash mixing is provided by the existing ahash dependency.
    const HASHER: ahash::RandomState = ahash::RandomState::with_seeds(0, 0, 0, 0);
    // Both branches retain the original hash modulo this nonzero owner count.
    let hash = HASHER.hash_one(pair);
    if shard_count.is_power_of_two() {
        hash as usize & (shard_count - 1)
    } else {
        (hash % shard_count as u64) as usize
    }
}
fn build_owner(pieces: InitialPiece, floor: u64, reuse: bool) -> Result<(Shard, Vec<Candidate>)> {
    let mut states = AHashMap::<Pair, PairAggregate<SmallVec<[Positions; 1]>>>::new();
    for (pair, state) in pieces {
        let total = states.entry(pair).or_default();
        super::add(&mut total.count, state.count)?;
        total.positions.push(state.positions);
    }
    let mut shard = Shard::default();
    let mut candidates = Vec::new();
    for (pair, state) in states {
        // Reuse retains even below-floor ledger entries for later signed
        // adjustments. Fresh mode can retire those pairs immediately.
        if reuse || state.count >= floor {
            let positions = Positions::concat(state.positions)?;
            shard.counts.insert(pair, state.count);
            if state.count != 0 {
                candidates.push(Candidate::new(pair, state.count, positions));
            }
        }
    }
    Ok((shard, candidates))
}

impl PairIndex {
    fn route(&mut self, changes: Vec<ProducerChanges>) {
        let shard_count = self.shards.len();
        // Joined commits drain each shard's pending work; any error aborts this attempt.
        for producer in changes {
            for change in producer.changes {
                if change.removed_weight != 0 {
                    let shard = owner(change.removed, shard_count);
                    self.shards[shard].actions.push(PendingAction::Remove {
                        pair: change.removed,
                        weight: change.removed_weight,
                    });
                }
                self.route_birth(change, producer.whole_rule);
            }
        }
    }

    fn route_birth(&mut self, change: super::corpus::PairChange<Positions>, whole_rule: bool) {
        if change.positions.is_empty() {
            return;
        }
        let shard_count = self.shards.len();
        let shard = &mut self.shards[owner(change.born, shard_count)];
        // Only non-AA complete births have one producer. Both directions can
        // name (id,id), so those still require owner aggregation. Zero-weight
        // births still own positions, including reuse cohorts.
        if whole_rule && self.mode == TrainingMode::Fresh && change.born.0 != change.born.1 {
            if change.born_weight >= self.minimum.max(1) {
                shard.completed.push(Candidate::new(
                    change.born,
                    change.born_weight,
                    change.positions,
                ));
            }
        } else {
            shard.actions.push(PendingAction::Birth {
                pair: change.born,
                weight: change.born_weight,
                birth_group: change.birth_group,
            });
            shard.positions.push(change.positions);
        }
    }
}

impl PairIndex {
    pub(super) fn peek(&mut self) -> Option<Priority> {
        loop {
            let top = self.queue.peek()?;
            let Some(&count) = self.shards[owner(top.pair(), self.shards.len())]
                .counts
                .get(&top.pair())
            else {
                // Fresh counts retire below the floor. Their stale payloads are
                // reclaimed when they reach the head of this owning queue.
                self.queue.pop();
                continue;
            };
            if top.count == count {
                return (count >= self.minimum.max(1)).then_some(top);
            }
            // Equal pair/priority entries can own different occurrence cohorts.
            // Keep pop/update/push: peek_mut can change their tie selection order.
            self.queue.rekey(count);
        }
    }
    pub(super) fn mode(&self) -> TrainingMode {
        self.mode
    }
}

#[cfg(test)]
mod count_contract_tests {
    use super::*;

    #[test]
    fn signed_adjustments_preserve_negative_bits_and_reject_overflow() {
        let mut count = 0;
        adjust_signed(&mut count, 1, true).unwrap();
        assert_eq!(count, u64::MAX);
        adjust_signed(&mut count, 2, false).unwrap();
        assert_eq!(count, 1);
        let mut high = i64::MAX as u64;
        assert!(adjust_signed(&mut high, 1, false).is_err());
        assert_eq!(high, i64::MAX as u64);
        let mut low = i64::MIN as u64;
        assert!(adjust_signed(&mut low, 1, true).is_err());
        assert_eq!(low, i64::MIN as u64);
        assert!(adjust_signed(&mut count, i64::MAX as u64 + 1, false).is_err());
        assert_eq!(count, 1);
    }

    #[test]
    fn reuse_checks_each_action_before_publishing_a_birth() {
        let pair = (0, 1);
        let mut shard = Shard::default();
        shard.counts.insert(pair, i64::MAX as u64);
        shard.actions = vec![
            PendingAction::Birth {
                pair,
                weight: 1,
                birth_group: 0,
            },
            PendingAction::Remove { pair, weight: 1 },
        ];
        shard
            .positions
            .push(Positions::from_unsorted(vec![7]).unwrap());
        assert!(shard.commit(true, 1).is_err());
        assert!(shard.completed.is_empty());
        assert!(shard.actions.is_empty());
        assert!(shard.positions.is_empty());
    }

    #[test]
    fn positive_reuse_cohort_survives_below_selection_floor() {
        let pair = (0, 1);
        let mut shard = Shard::default();
        shard.counts.insert(pair, 3);
        shard.actions = vec![
            PendingAction::Remove { pair, weight: 4 },
            PendingAction::Birth {
                pair,
                weight: 2,
                birth_group: 0,
            },
            PendingAction::Birth {
                pair,
                weight: 0,
                birth_group: 0,
            },
        ];
        shard.positions = vec![
            Positions::from_unsorted(vec![9]).unwrap(),
            Positions::from_unsorted(vec![3]).unwrap(),
        ];
        shard.commit(true, 5).unwrap();
        assert_eq!(shard.counts[&pair], 1);
        assert_eq!(shard.completed.len(), 1);
        assert_eq!(shard.completed[0].priority.count(), 1);
        assert_eq!(
            shard.completed[0].positions.iter().collect::<Vec<_>>(),
            [3, 9]
        );
    }
}
