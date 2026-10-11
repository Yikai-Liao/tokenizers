//! Count owners publish complete lists; the queue repairs stale snapshots lazily.
use super::merge::Change;
use super::{WORD_SEPARATOR_ID, add, corpus::CorpusPlan, positions::Positions};
use ahash::AHashMap;
use dary_heap::OctonaryHeap;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::cmp::{Ordering, Reverse};
use itertools::Itertools;
use tk_encode::{Result, models::bpe::Pair, utils::progress::ProgressBar};

/// Queue key: highest count first, then smallest pair.
/// Field order defines heap priority; occurrence payloads never break ties.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(super) struct Priority {
    pub(super) count: u64,
    pair: Reverse<Pair>,
}

impl Priority {
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
struct State<P> {
    count: u64,
    positions: P,
}

// Metadata stays borrowed during commit; each position stream has one owner.
type Event = Change<()>;

/// One pair-count owner and its reusable pending work, in producer order.
/// Actions index shared event metadata; the separate birth array owns one stream
/// per birth action and avoids reserving a payload for removal-only actions.
#[derive(Default)]
struct Shard {
    counts: AHashMap<Pair, u64>,
    actions: Vec<(usize, bool, bool)>,
    // One stream per birth action, in the same order.
    positions: Vec<Positions>,
}

/// Authoritative pair-count shards plus an owning queue of occurrence candidates.
/// Each pair has one count owner; queued priorities are repaired lazily at the head.
/// Fresh counts retire below the floor; reuse retains the per-action signed ledger.
pub(super) struct PairIndex {
    shards: Box<[Shard]>,
    queue: OctonaryHeap<Candidate>,
    floor: u64,
    reuse: bool,
}

// Map an ordered pair to its sole count owner, with a nonzero shard count fixed for
// the index. Initial counting, queue lookup and commit must use this same mapping.
fn owner(pair: Pair, shard_count: usize) -> usize {
    // Zero seeds keep routing repeatable within this build, without per-call
    // randomness. Hash mixing is provided by the existing ahash dependency.
    const HASHER: ahash::RandomState = ahash::RandomState::with_seeds(0, 0, 0, 0);
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
        shard_count: usize,
        reuse: bool,
        progress: &Option<ProgressBar>,
    ) -> Result<Self> {
        // 1. Count and freeze each whole-word range independently.
        let pieces = corpus
            .initial_ranges(if cfg!(test) { 16 } else { 1 << 24 })
            .into_par_iter()
            .map(|range| count_range(corpus, range, progress))
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
            .into_par_iter()
            .map(|pieces| build_owner(pieces, minimum.max(1), reuse))
            .collect::<Result<Vec<_>>>()?;
        let (shards, candidates): (Vec<_>, Vec<_>) = owners.into_iter().unzip();
        Ok(Self {
            shards: shards.into_boxed_slice(),
            queue: candidates.into_iter().flatten().collect(),
            floor: minimum.max(1),
            reuse,
        })
    }

    pub(super) fn reuse(&self) -> bool {
        self.reuse
    }

    /// Consume remaining candidates at the successful training boundary.
    /// Ordinary retirement and errors keep their normal, synchronous drops.
    pub(super) fn retire(self) {
        let Self { shards, queue, .. } = self;
        let phase = std::time::Instant::now();
        drop(shards);
        eprintln!("BPE_CLEANUP_SHARDS_SECONDS {:.9}", phase.elapsed().as_secs_f64());
        let phase = std::time::Instant::now();
        if std::env::var("BPE_DIAGNOSTIC_PARALLEL_DROP").as_deref() == Ok("0") {
            drop(queue);
        } else {
            queue.into_vec().into_par_iter().with_min_len(4096).for_each(drop);
        }
        eprintln!("BPE_CLEANUP_CANDIDATES_SECONDS {:.9}", phase.elapsed().as_secs_f64());
    }

    pub(super) fn best(&mut self) -> Option<Priority> {
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

    pub(super) fn take(&mut self, priority: Priority) -> Candidate {
        let candidate = self.queue.pop().expect("certified candidate exists");
        debug_assert_eq!(candidate.priority, priority);
        if !self.reuse {
            let shard_count = self.shards.len();
            self.shards[owner(priority.pair(), shard_count)]
                .counts
                .remove(&priority.pair())
                .expect("certified pair exists");
        }
        candidate
    }

    pub(super) fn commit(&mut self, changes: Vec<Vec<Change<Positions>>>) -> Result<()> {
        // 1. Route each removal and birth to its count owner.
        let events = self.route_changes(changes);

        // 2. Apply owner-local actions and encode their birth cohorts in parallel.
        let reuse = self.reuse;
        let floor = self.floor;
        let births = self
            .shards
            .par_iter_mut()
            .filter(|shard| !shard.actions.is_empty())
            .map(|shard| shard.commit(&events, reuse, floor))
            .collect::<Result<Vec<_>>>()?;

        // 3. Publish only after all owners join, in the same owner traversal order.
        for candidate in births.into_iter().flatten() {
            self.queue.push(candidate);
        }
        Ok(())
    }

    fn route_changes(&mut self, changes: Vec<Vec<Change<Positions>>>) -> Vec<Event> {
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
                bucket: change.bucket,
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
type InitialPiece = Vec<(Pair, State<Positions>)>;

/// Range-local initial counts, choosing dense pair keys only for a small ID domain.
/// The domain travels with its array; sparse counting owns no unused dense plane.
/// Recording preserves each pair's ordered positions before storage is frozen.
enum InitialPairCounts {
    Dense {
        domain: usize,
        states: Vec<State<Positions>>,
    },
    Sparse(AHashMap<Pair, State<Positions>>),
}

impl InitialPairCounts {
    fn new(domain: Option<usize>) -> Self {
        match domain {
            Some(domain) => Self::Dense {
                domain,
                states: (0..domain * domain).map(|_| State::default()).collect(),
            },
            None => Self::Sparse(AHashMap::new()),
        }
    }

    fn record(&mut self, pair: Pair, position: u64, weight: u64) -> Result<()> {
        let state = match self {
            Self::Dense { domain, states } => {
                &mut states[pair.0 as usize * *domain + pair.1 as usize]
            }
            Self::Sparse(states) => states.entry(pair).or_default(),
        };
        add(&mut state.count, weight)?;
        // Initial edges arrive in corpus order; construction checks the full list.
        state.positions.push(position)?;
        Ok(())
    }

    fn into_states(self) -> Vec<(Pair, State<Positions>)> {
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

fn count_range(
    corpus: &CorpusPlan<'_>,
    range: std::ops::Range<usize>,
    progress: &Option<ProgressBar>,
) -> Result<InitialPiece> {
    let mut counts = InitialPairCounts::new(corpus.small_pair_domain());
    corpus.initial_edges(range.clone(), |pair, p, weight| {
        debug_assert!(pair.0 != WORD_SEPARATOR_ID && pair.1 != WORD_SEPARATOR_ID);
        counts.record(pair, p, weight)
    })?;
    if let Some(p) = progress {
        p.inc(range.len() as u64);
    }
    Ok(counts.into_states())
}

fn build_owner(pieces: InitialPiece, floor: u64, reuse: bool) -> Result<(Shard, Vec<Candidate>)> {
    let mut states = AHashMap::<Pair, State<SmallVec<[Positions; 1]>>>::new();
    for (pair, state) in pieces {
        let total = states.entry(pair).or_default();
        add(&mut total.count, state.count)?;
        total.positions.push(state.positions);
    }
    let mut shard = Shard::default();
    let mut candidates = Vec::new();
    for (pair, state) in states {
        if reuse || state.count >= floor {
            let positions = Positions::concat(state.positions)?;
            shard.counts.insert(pair, state.count);
            if state.count != 0 {
                candidates.push(Candidate {
                    priority: Priority {
                        pair: Reverse(pair),
                        count: state.count,
                    },
                    positions,
                });
            }
        }
    }
    Ok((shard, candidates))
}

impl Shard {
    fn push(&mut self, index: usize, remove: bool, birth: Option<Positions>) {
        self.actions.push((index, remove, birth.is_some()));
        self.positions.extend(birth);
    }

    fn commit(&mut self, events: &[Event], reuse: bool, floor: u64) -> Result<Vec<Candidate>> {
        let mut groups = AHashMap::<(usize, Pair), State<SmallVec<[Positions; 1]>>>::new();
        let mut candidates = Vec::new();
        // Both drains own their remaining items: errors drop unpublished births
        // and empty the pending vectors while retaining their capacity.
        let mut positions = self.positions.drain(..);
        for (index, remove, birth) in self.actions.drain(..) {
            let change = &events[index];
            // A boundary removal precedes its replacement birth. The signed reuse
            // ledger checks each action in order, including intermediate overflow.
            if remove {
                if reuse {
                    let count = self.counts.entry(change.removed).or_default();
                    adjust_signed(count, change.removed_weight, true)?;
                } else if let Some(count) = self.counts.get_mut(&change.removed) {
                    *count = count
                        .checked_sub(change.removed_weight)
                        .ok_or("BPE fresh removal exceeds the current count")?;
                    if *count < floor {
                        self.counts.remove(&change.removed);
                    }
                }
            }
            if !birth {
                continue;
            }
            if reuse {
                let count = self.counts.entry(change.born).or_default();
                adjust_signed(count, change.born_weight, false)?;
            }
            let fragment = positions.next().expect("each birth owns one stream");
            let group = groups.entry((change.bucket, change.born)).or_default();
            add(&mut group.count, change.born_weight)?;
            // Compatible fresh rules uniquely produce each born key, and task
            // results retain spatial order. Reuse streams may interleave.
            group.positions.push(fragment);
        }

        // Uniform groups publish in bucket/pair order, retaining historical cohorts.
        for ((_, pair), state) in groups.into_iter().sorted_unstable_by_key(|(key, _)| *key) {
            let count = if reuse {
                self.counts[&pair]
            } else {
                state.count
            };
            // A positive reuse ledger publishes a historical cohort even below
            // the floor; selection applies the floor after correcting its head.
            if (reuse && (count as i64) <= 0) || (!reuse && count < floor) {
                continue;
            }
            let positions = if reuse {
                merge_positions(state.positions)?
            } else {
                Positions::concat(state.positions)?
            };
            if !reuse {
                debug_assert!(!self.counts.contains_key(&pair));
                self.counts.insert(pair, count);
            }
            candidates.push(Candidate {
                priority: Priority {
                    count,
                    pair: Reverse(pair),
                },
                positions,
            });
        }
        Ok(candidates)
    }
}

/// Interleaved reuse cohorts retain duplicates, but publish one ordered leaf.
/// Only the iterator frontier is buffered; coordinates append directly.
fn merge_positions(fragments: SmallVec<[Positions; 1]>) -> Result<Positions> {
    if fragments.len() == 1 {
        return Ok(fragments.into_iter().next().expect("one fragment"));
    }
    use itertools::Itertools;
    let mut positions = Positions::default();
    for position in fragments.iter().map(Positions::iter).kmerge() {
        positions.push(position)?;
    }
    Ok(positions)
}

#[cfg(test)]
mod tests {
    use super::super::tests::tokens;
    use super::*;
    use crate::trainers::bpe::{
        tests::{initialize, trainer},
        word_counts::WordCountsView,
    };

    fn event(pair: Pair, add: u64, remove: u64, bucket: usize) -> Event {
        Event {
            removed: pair,
            born: pair,
            removed_weight: remove,
            born_weight: add,
            bucket,
            positions: (),
        }
    }

    #[test]
    fn unified_aggregation() {
        let pair = (0, 1);
        for reuse in [false, true] {
            let mut shard = Shard::default();
            let events = [event(pair, 1, 0, 0), event(pair, 1, 0, 0)];
            let streams: [&[u64]; 2] = if reuse {
                [&[1, 4, 4], &[0, 4, 5]]
            } else {
                [&[0, 1, 4], &[4, 4, 5]]
            };
            for (index, stream) in streams.into_iter().enumerate() {
                shard.push(index, false, Some(Positions::from_sorted(stream).unwrap()));
            }
            let candidates = shard.commit(&events, reuse, 2).unwrap();
            assert_eq!(candidates.len(), 1);
            assert_eq!(candidates[0].priority.count, 2);
            assert_eq!(
                candidates[0].positions.iter().collect::<Vec<_>>(),
                [0, 1, 4, 4, 4, 5]
            );
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
    fn failures_before_and_after_writes() {
        let trainer = trainer();
        for preparation in [true, false] {
            let words = if preparation {
                [("ab".into(), u64::MAX), ("xcd".into(), u64::MAX)]
                    .into_iter()
                    .collect()
            } else {
                [("abc".into(), 1)].into_iter().collect()
            };
            let (mut vocabulary, mut corpus, mut index) =
                initialize(&trainer, WordCountsView::from_map(&words), !preparation);
            if preparation {
                // AB is valid; duplicate CD overflows its neighbor count before writes.
                let mut candidates = std::mem::take(&mut index.queue).into_vec();
                let candidate = candidates
                    .iter_mut()
                    .find(|c| c.priority.pair() == (2, 3))
                    .unwrap();
                let start = candidate.positions.iter().next().unwrap();
                candidate.positions = Positions::from_sorted(&[start, start]).unwrap();
                index.queue = candidates.into_iter().collect();
            } else {
                // AB writes ID 3 before its right birth overflows the signed owner count.
                index.shards[0].counts.insert((3, 2), i64::MAX as u64);
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
            if preparation {
                assert_eq!(after, before);
            } else {
                assert_ne!(after, before, "commit errors do not roll back writes");
                assert!(after.contains(&3));
                assert_eq!(
                    index.queue.len(),
                    queued - 1,
                    "failed owners publish no births"
                );
            }
        }
    }

    #[test]
    fn historical_cohorts() {
        let pair = (7, 8);
        let mut shard = Shard::default();
        // Separate commits retain two cohorts; the second changes the count 10→5.
        let mut candidates = Vec::new();
        for (add, remove, position) in [(10, 0, 20), (3, 8, 21)] {
            shard.push(
                0,
                remove != 0,
                Some(Positions::from_sorted(&[position]).unwrap()),
            );
            candidates.extend(
                shard
                    .commit(&[event(pair, add, remove, 0)], true, 1)
                    .unwrap(),
            );
        }
        assert_eq!(candidates.len(), 2);
        let mut index = PairIndex {
            shards: vec![shard].into_boxed_slice(),
            queue: candidates.into_iter().collect(),
            floor: 1,
            reuse: true,
        };
        // Repair pop/update/push preserves which equal-priority cohort comes first.
        let priority = index.best().unwrap();
        assert_eq!(priority.count, 5);
        assert_eq!(
            index.take(priority).positions.iter().collect::<Vec<_>>(),
            [21]
        );
        let priority = index.best().unwrap();
        assert_eq!(
            index.take(priority).positions.iter().collect::<Vec<_>>(),
            [20]
        );
        assert!(index.best().is_none());
    }
}
