//! Count owners publish complete lists; the queue repairs stale snapshots lazily.
use super::corpus::PairChange;
use super::{
    TrainingMode, WORD_SEPARATOR_ID, add,
    corpus::{CorpusPlan, InitialEdge},
    positions::Positions,
};
use ahash::AHashMap;
use dary_heap::OctonaryHeap;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::cmp::{Ordering, Reverse};
use std::collections::hash_map::Entry;
use tk_encode::{Result, models::bpe::Pair, parallelism::*, utils::progress::ProgressBar};

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
pub(super) struct Candidate {
    pub(super) priority: Priority,
    pub(super) positions: Positions,
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

impl Eq for Candidate {}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.cmp(&other.priority)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Weighted count and occurrence storage, mutable during aggregation and frozen
/// for initial routing or candidate publication.
#[derive(Default)]
struct PairAggregate<P> {
    count: u64,
    positions: P,
}

// Metadata stays borrowed during commit; each position stream has one owner.
type Event = PairChange<()>;

/// One pair-count owner and its reusable pending work, in producer order.
/// Actions index shared event metadata; the separate birth array owns one stream
/// per birth action and avoids reserving a payload for removal-only actions.
#[derive(Default)]
struct Shard {
    counts: AHashMap<Pair, u64>,
    actions: Vec<PendingAction>,
    // One stream per birth action, in the same order.
    positions: Vec<Positions>,
    diagnostic_group_indices: Vec<u32>,
}

struct PendingAction {
    event_index: usize,
    remove_count: bool,
    has_birth: bool,
}

struct BirthFragment {
    birth_group: usize,
    pair: Pair,
    weight: u64,
    positions: Positions,
}

/// Authoritative pair-count shards plus an owning queue of occurrence candidates.
/// Each pair has one count owner; queued priorities are repaired lazily at the head.
/// Fresh counts retire below the floor; reuse retains the per-action signed ledger.
pub(super) struct PairIndex {
    shards: Box<[Shard]>,
    queue: OctonaryHeap<Candidate>,
    floor: u64,
    mode: TrainingMode,
}

// Map an ordered pair to its sole count owner, with a nonzero shard count fixed for
// the index. Initial counting, queue lookup and commit must use this same mapping.
fn owner(pair: Pair, shard_count: usize) -> usize {
    // Zero seeds keep routing repeatable within this build, without per-call
    // randomness. Hash mixing is provided by the existing ahash dependency.
    const HASHER: ahash::RandomState = ahash::RandomState::with_seeds(0, 0, 0, 0);
    // Modulo selects exactly one owner in 0..shard_count; the fixed nonzero
    // shard count also bounds the result before conversion to resident indexing.
    (HASHER.hash_one(pair) % shard_count as u64) as usize
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
        // The production budget is 2^24 resident slots per range; tests use
        // 16 slots to force range boundaries on small fixtures.
        let pieces = corpus
            .chunks(if cfg!(test) { 16 } else { 1 << 24 })
            .into_maybe_par_iter()
            .map(|(words, edges)| count_chunk(domain, words, edges, progress))
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
            floor: minimum.max(1),
            mode,
        })
    }

    pub(super) fn mode(&self) -> TrainingMode {
        self.mode
    }

    /// Consume remaining candidates at the successful training boundary.
    /// Ordinary retirement and errors keep their normal, synchronous drops.
    pub(super) fn retire(self) {
        let Self { shards, queue, .. } = self;
        drop(shards);
        if get_parallelism() {
            queue
                .into_vec()
                .into_par_iter()
                .with_min_len(4096)
                .for_each(drop);
        } else {
            drop(queue);
        }
    }

    pub(super) fn peek(&mut self) -> Option<Priority> {
        loop {
            let top = self.queue.peek()?.priority;
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
                return (count >= self.floor).then_some(top);
            }
            // Equal pair/priority entries can own different occurrence cohorts.
            // Keep pop/update/push: peek_mut can change their tie selection order.
            let mut candidate = self.queue.pop().expect("observed candidate exists");
            candidate.priority.count = count;
            self.queue.push(candidate);
        }
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

    pub(super) fn commit(&mut self, changes: Vec<Vec<PairChange<Positions>>>) -> Result<()> {
        // 1. Route each removal and birth to its count owner.
        let events = self.route(changes);

        // 2. Apply owner-local actions and encode their birth cohorts in parallel.
        let reuse = self.mode == TrainingMode::Reuse;
        let floor = self.floor;
        let births = self
            .shards
            .maybe_par_iter_mut()
            .filter(|shard| !shard.actions.is_empty())
            .map(|shard| shard.commit(&events, reuse, floor))
            .collect::<Result<Vec<_>>>()?;

        // 3. Publish only after all owners join, in the same owner traversal order.
        for candidate in births.into_iter().flatten() {
            self.queue.push(candidate);
        }
        Ok(())
    }

    fn route(&mut self, changes: Vec<Vec<PairChange<Positions>>>) -> Vec<Event> {
        let shard_count = self.shards.len();
        // Joined commits drain each shard's pending work; any error aborts this attempt.
        let mut events = Vec::with_capacity(changes.iter().map(Vec::len).sum());
        for change in changes.into_iter().flatten() {
            let removed = (change.removed_weight != 0).then(|| owner(change.removed, shard_count));
            // Zero-weight births still own positions, including reuse cohorts.
            let born = (!change.positions.is_empty()).then(|| owner(change.born, shard_count));
            let index = events.len();
            events.push(Event {
                removed: change.removed,
                born: change.born,
                removed_weight: change.removed_weight,
                born_weight: change.born_weight,
                birth_group: change.birth_group,
                positions: (),
            });
            match (removed, born) {
                (Some(removed), Some(born)) if removed == born => {
                    self.shards[removed].push(index, true, Some(change.positions));
                }
                (removed, born) => {
                    if let Some(removed) = removed {
                        self.shards[removed].push(index, true, None);
                    }
                    if let Some(born) = born {
                        self.shards[born].push(index, false, Some(change.positions));
                    }
                }
            }
        }
        events
    }
}

// Initial fragments own their storage: an owner may discard or concatenate them
// after range counting joins, without retaining every worker's temporary builder.
type InitialPiece = Vec<(Pair, PairAggregate<Positions>)>;

/// Range-local initial counts, choosing dense pair keys only for a small ID domain.
/// The domain travels with its array; sparse counting owns no unused dense plane.
/// Recording preserves each pair's ordered positions before storage is frozen.
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

    fn record(&mut self, pair: Pair, position: u64, weight: u64) -> Result<()> {
        let state = match self {
            Self::Dense { domain, states } => {
                // Row-major pair key: left ID chooses a domain-wide row and
                // right ID chooses its column. Both IDs are below domain.
                &mut states[pair.0 as usize * *domain + pair.1 as usize]
            }
            Self::Sparse(states) => states.entry(pair).or_default(),
        };
        add(&mut state.count, weight)?;
        // Initial edges arrive in corpus order; construction checks the full list.
        state.positions.push(position)?;
        Ok(())
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
    words: usize,
    edges: impl Iterator<Item = InitialEdge>,
    progress: &Option<ProgressBar>,
) -> Result<InitialPiece> {
    let mut counts = InitialPairCounts::new(domain);
    for edge in edges {
        debug_assert!(edge.pair.0 != WORD_SEPARATOR_ID && edge.pair.1 != WORD_SEPARATOR_ID);
        counts.record(edge.pair, edge.coordinate, edge.weight)?;
    }
    if let Some(p) = progress {
        p.inc(words as u64);
    }
    Ok(counts.into_entries())
}

fn build_owner(pieces: InitialPiece, floor: u64, reuse: bool) -> Result<(Shard, Vec<Candidate>)> {
    let mut states = AHashMap::<Pair, PairAggregate<SmallVec<[Positions; 1]>>>::new();
    for (pair, state) in pieces {
        let total = states.entry(pair).or_default();
        add(&mut total.count, state.count)?;
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

impl Shard {
    fn push(&mut self, index: usize, remove: bool, birth: Option<Positions>) {
        self.actions.push(PendingAction {
            event_index: index,
            remove_count: remove,
            has_birth: birth.is_some(),
        });
        self.positions.extend(birth);
    }

    fn commit(&mut self, events: &[Event], reuse: bool, floor: u64) -> Result<Vec<Candidate>> {
        let mut fragments = Vec::with_capacity(self.positions.len());
        let mut candidates = Vec::new();
        // Both drains own their remaining items: errors drop unpublished births
        // and empty the pending vectors while retaining their capacity.
        let mut positions = self.positions.drain(..);
        for action in self.actions.drain(..) {
            let change = &events[action.event_index];
            // A boundary removal precedes its replacement birth. The signed reuse
            // ledger checks each action in order, including intermediate overflow.
            if action.remove_count {
                if reuse {
                    let count = self.counts.entry(change.removed).or_default();
                    adjust_signed(count, change.removed_weight, true)?;
                } else if let Entry::Occupied(mut entry) = self.counts.entry(change.removed) {
                    let count = entry.get_mut();
                    *count = count
                        .checked_sub(change.removed_weight)
                        .ok_or("BPE fresh removal exceeds the current count")?;
                    if *count < floor {
                        entry.remove();
                    }
                }
            }
            if !action.has_birth {
                continue;
            }
            if reuse {
                let count = self.counts.entry(change.born).or_default();
                adjust_signed(count, change.born_weight, false)?;
            }
            let fragment = positions.next().expect("each birth owns one stream");
            fragments.push(BirthFragment {
                birth_group: change.birth_group,
                pair: change.born,
                weight: change.born_weight,
                positions: fragment,
            });
        }

        drop(positions);
        if !reuse {
            return self.diagnostic_fresh_groups(fragments, floor);
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
                add(&mut weight, part.weight)?;
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
            candidates.push(Candidate::new(pair, count, positions));
        }
        Ok(candidates)
    }
}

impl Shard {
    // Diagnostic only: group fresh births by replacement/direction, then neighbor.
    // Count actions have already run in their original order. Reuse keeps its path.
    fn diagnostic_fresh_groups(
        &mut self,
        fragments: Vec<BirthFragment>,
        floor: u64,
    ) -> Result<Vec<Candidate>> {
        if fragments.is_empty() {
            return Ok(Vec::new());
        }
        let replacement = |part: &BirthFragment| {
            if part.birth_group == 0 {
                part.pair.1
            } else {
                part.pair.0
            }
        };
        let min = fragments.iter().map(&replacement).min().unwrap();
        let max = fragments.iter().map(&replacement).max().unwrap();
        // Fresh compatible batches append at most 256 IDs; activated old IDs run alone.
        assert!(max - min < 256);
        let mut buckets: Vec<Vec<BirthFragment>> = (0..((max - min + 1) as usize) * 2)
            .map(|_| Vec::new())
            .collect();
        for part in fragments {
            let bucket = (replacement(&part) - min) as usize * 2 + part.birth_group;
            buckets[bucket].push(part);
        }
        let mut candidates = Vec::new();
        for (bucket, parts) in buckets.into_iter().enumerate() {
            let left = bucket % 2 == 0;
            let id = min + (bucket / 2) as u32;
            let mut groups: Vec<(u32, PairAggregate<SmallVec<[Positions; 1]>>)> = Vec::new();
            for part in parts {
                let neighbor = if left { part.pair.0 } else { part.pair.1 };
                if self.diagnostic_group_indices.len() <= neighbor as usize {
                    self.diagnostic_group_indices
                        .resize(neighbor as usize + 1, u32::MAX);
                }
                let slot = &mut self.diagnostic_group_indices[neighbor as usize];
                if groups
                    .get(*slot as usize)
                    .is_none_or(|group| group.0 != neighbor)
                {
                    *slot = groups.len() as u32;
                    groups.push((neighbor, PairAggregate::default()));
                }
                let group = &mut groups[*slot as usize].1;
                add(&mut group.count, part.weight)?;
                group.positions.push(part.positions);
            }
            for (neighbor, group) in groups {
                if group.count < floor {
                    continue;
                }
                let pair = if left { (neighbor, id) } else { (id, neighbor) };
                let positions = Positions::concat(group.positions)?;
                debug_assert!(!self.counts.contains_key(&pair));
                self.counts.insert(pair, group.count);
                candidates.push(Candidate::new(pair, group.count, positions));
            }
        }
        Ok(candidates)
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::tokens;
    use super::*;
    use crate::trainers::bpe::{
        tests::{initialize, trainer},
        word_counts::WordCountsView,
    };

    fn event(pair: Pair, add: u64, remove: u64, birth_group: usize) -> Event {
        Event {
            removed: pair,
            born: pair,
            removed_weight: remove,
            born_weight: add,
            birth_group,
            positions: (),
        }
    }

    #[test]
    fn intermediate_overflow() {
        let pair = (0, 1);
        for reuse in [false, true] {
            let mut shard = Shard::default();
            if reuse {
                shard.counts.insert(pair, i64::MAX as u64);
            }
            let events = [
                event(pair, if reuse { 1 } else { u64::MAX }, 0, 0),
                event(pair, if reuse { 0 } else { 1 }, u64::from(reuse), 0),
            ];
            shard.push(0, false, Some(Positions::from_sorted(&[0]).unwrap()));
            shard.push(1, reuse, Some(Positions::from_sorted(&[1]).unwrap()));
            let error = shard
                .commit(&events, reuse, 1)
                .err()
                .expect("overflow must fail");
            assert!(
                error
                    .to_string()
                    .contains(if reuse { "exceeds i64" } else { "exceeds u64" }),
                "{error}"
            );
            assert!(shard.actions.is_empty() && shard.positions.is_empty());
        }
    }

    #[test]
    fn failed_rounds() {
        for workers in [1, 6] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap()
                .install(|| {
                    for preparation in [true, false] {
                        let mut trainer = trainer();
                        let words: AHashMap<compact_str::CompactString, u64> = if preparation {
                            [
                                ("ab".into(), u64::MAX),
                                ("cd".into(), u64::MAX),
                                ("xef".into(), u64::MAX),
                            ]
                            .into_iter()
                            .collect()
                        } else {
                            [("abcd".into(), 1)].into_iter().collect()
                        };
                        let (mut vocabulary, mut corpus, mut index) =
                            initialize(&trainer, WordCountsView::from(&words), !preparation);
                        // First commit AB alone. Its history must survive a later failed round.
                        trainer.vocab_size = vocabulary.len() + 1;
                        assert!(matches!(
                            trainer
                                .merge_next(&mut vocabulary, &mut corpus, &mut index)
                                .unwrap(),
                            super::super::MergeOutcome::Committed { rule_count: 1 }
                        ));
                        let committed = vocabulary.merges().to_vec();
                        let trace = vocabulary.trace.clone();
                        assert_eq!(committed.len(), 1);
                        trainer.vocab_size = 64;
                        if preparation {
                            // CD prepares successfully; duplicate EF overflows its left neighbor count.
                            let mut candidates = std::mem::take(&mut index.queue).into_vec();
                            let candidate = candidates
                                .iter_mut()
                                .find(|c| c.priority.pair() == (4, 5))
                                .unwrap();
                            let start = candidate.positions.iter().next().unwrap();
                            candidate.positions = Positions::from_sorted(&[start, start]).unwrap();
                            index.queue = candidates.into_iter().collect();
                        } else {
                            // CD writes ID 5; its left birth (AB, CD) then overflows the ledger.
                            let shard = owner((4, 5), index.shards.len());
                            index.shards[shard].counts.insert((4, 5), i64::MAX as u64);
                        }
                        let before = tokens(&corpus);
                        let queued = index.queue.len();
                        let error = trainer
                            .merge_next(&mut vocabulary, &mut corpus, &mut index)
                            .unwrap_err();
                        assert!(
                            error.to_string().contains(if preparation {
                                "exceeds u64"
                            } else {
                                "exceeds i64"
                            }),
                            "{error}"
                        );
                        let after = tokens(&corpus);
                        assert_eq!(vocabulary.merges(), committed);
                        assert_eq!(vocabulary.trace, trace);
                        if preparation {
                            assert_eq!(after, before);
                        } else {
                            assert_ne!(after, before, "commit errors do not roll back writes");
                            assert!(after.contains(&5));
                            assert_eq!(
                                index.queue.len(),
                                queued - 1,
                                "failed owners publish no births"
                            );
                        }
                    }
                });
        }
    }
}
