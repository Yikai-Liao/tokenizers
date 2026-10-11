//! Count owners publish complete lists; the queue repairs stale snapshots lazily.
use super::merge::{Birth, Change};
use super::{WORD_SEPARATOR_ID, add, corpus::CorpusPlan, positions::Positions};
use ahash::AHashMap;
use dary_heap::OctonaryHeap;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::cmp::{Ordering, Reverse};
use std::collections::BTreeMap;
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
    positions: Vec<Birth>,
}

/// Authoritative pair-count shards plus an owning queue of occurrence candidates.
/// Each pair has one count owner; queued priorities are repaired lazily at the head.
/// Fresh counts retire below the floor; reuse retains the per-action signed ledger.
pub(super) struct PairIndex {
    shards: Vec<Shard>,
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
            shards,
            queue: candidates.into_iter().flatten().collect(),
            floor: minimum.max(1),
            reuse,
        })
    }

    pub(super) fn reuse(&self) -> bool {
        self.reuse
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

    pub(super) fn commit(&mut self, changes: Vec<Vec<Change<Birth>>>) -> Result<()> {
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

    fn route_changes(&mut self, changes: Vec<Vec<Change<Birth>>>) -> Vec<Event> {
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
        states: Vec<State<SmallVec<[u64; 2]>>>,
    },
    Sparse(AHashMap<Pair, State<SmallVec<[u64; 2]>>>),
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
        state.positions.push(position);
        Ok(())
    }

    fn into_states(self) -> Vec<(Pair, State<SmallVec<[u64; 2]>>)> {
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
    let states = counts.into_states();

    states
        .into_iter()
        .map(|(pair, state)| {
            let positions = Positions::from_sorted(&state.positions)?;
            Ok((
                pair,
                State {
                    count: state.count,
                    positions,
                },
            ))
        })
        .collect::<Result<Vec<_>>>()
}

fn build_owner(pieces: InitialPiece, floor: u64, reuse: bool) -> Result<(Shard, Vec<Candidate>)> {
    let mut states = AHashMap::<Pair, State<Vec<Positions>>>::new();
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
    fn push(&mut self, index: usize, remove: bool, birth: Option<Birth>) {
        self.actions.push((index, remove, birth.is_some()));
        self.positions.extend(birth);
    }

    fn commit(&mut self, events: &[Event], reuse: bool, floor: u64) -> Result<Vec<Candidate>> {
        let mut groups = BTreeMap::<(usize, Pair), State<SmallVec<[u64; 2]>>>::new();
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
            let values = match positions.next().expect("each birth owns one stream") {
                Birth::Complete(positions) => {
                    // Fresh IDs and compatible rules give each complete birth one producer.
                    debug_assert!(!reuse && change.born_weight >= floor);
                    debug_assert!(!self.counts.contains_key(&change.born));
                    self.counts.insert(change.born, change.born_weight);
                    candidates.push(Candidate {
                        priority: Priority {
                            count: change.born_weight,
                            pair: Reverse(change.born),
                        },
                        positions,
                    });
                    continue;
                }
                Birth::Partial(values) => values,
            };
            let group = groups.entry((change.bucket, change.born)).or_default();
            add(&mut group.count, change.born_weight)?;
            // Fresh jobs follow rule rank and spatial ranges. Each born key has
            // one producer, so fragments concatenate sorted. Reuse sorts below.
            // Fresh fragments transfer their first allocation. Reuse copies into
            // owner-local aggregation storage before sorting interleaved fragments.
            if !reuse && group.positions.is_empty() {
                group.positions = values;
            } else {
                group.positions.extend_from_slice(&values);
            }
        }

        // Complete producers precede partial groups, which follow bucket/pair order.
        for ((_, pair), mut state) in groups {
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
            if reuse {
                // Historical cohorts retain duplicate positions.
                state.positions.sort_unstable();
            } else {
                debug_assert!(!self.counts.contains_key(&pair));
                self.counts.insert(pair, count);
            }
            candidates.push(Candidate {
                priority: Priority {
                    count,
                    pair: Reverse(pair),
                },
                positions: Positions::from_sorted(&state.positions)?,
            });
        }
        Ok(candidates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainers::bpe::{
        BpeTrainer, Trace, merge::MergeOutcome, vocabulary::Vocabulary, word_counts::WordCountsView,
    };
    use tk_encode::{
        models::bpe::{Merges, Vocab},
        vocab::bucket_added_vocabulary::AddedToken,
    };

    #[test]
    fn reuse_groups_sort_positions_keep_duplicates_and_publish_in_bucket_order() {
        let pair = (7, 8);
        let mut shard = Shard::default();
        let mut events = Vec::new();
        for (bucket, weight, values) in [(1, 1, &[8][..]), (0, 0, &[9][..]), (0, 2, &[2, 2][..])] {
            shard.push(
                events.len(),
                false,
                Some(Birth::Partial(SmallVec::from_slice(values))),
            );
            events.push(Event {
                removed: pair,
                born: pair,
                removed_weight: 0,
                born_weight: weight,
                bucket,
                positions: (),
            });
        }
        let candidates = shard.commit(&events, true, 10).unwrap();
        assert_eq!(shard.counts[&pair], 3);
        assert_eq!(
            candidates.len(),
            2,
            "positive cohorts survive below the floor"
        );
        assert_eq!(
            candidates[0].positions.iter().collect::<Vec<_>>(),
            [2, 2, 9]
        );
        assert_eq!(candidates[1].positions.iter().collect::<Vec<_>>(), [8]);
        assert!(
            candidates
                .iter()
                .all(|candidate| candidate.priority.count == 3)
        );
        assert!(shard.actions.is_empty() && shard.positions.is_empty());
    }

    #[test]
    fn fresh_commit_retires_removals_and_publishes_complete_then_partial_births() {
        let removed = (0, 1);
        let partial = (2, 3);
        let complete = (4, 5);
        let pruned = (6, 7);
        let mut shard = Shard::default();
        shard.counts.insert(removed, 5);
        let events = [
            Event {
                removed,
                born: partial,
                removed_weight: 3,
                born_weight: 1,
                bucket: 0,
                positions: (),
            },
            Event {
                removed,
                born: partial,
                removed_weight: 0,
                born_weight: 2,
                bucket: 0,
                positions: (),
            },
            Event {
                removed,
                born: complete,
                removed_weight: 0,
                born_weight: 4,
                bucket: 1,
                positions: (),
            },
            Event {
                removed,
                born: pruned,
                removed_weight: 0,
                born_weight: 1,
                bucket: 2,
                positions: (),
            },
        ];
        shard.push(0, true, Some(Birth::Partial(SmallVec::from_slice(&[20]))));
        shard.push(
            1,
            false,
            Some(Birth::Partial(SmallVec::from_slice(&[21, 22]))),
        );
        shard.push(
            2,
            false,
            Some(Birth::Complete(Positions::from_sorted(&[11]).unwrap())),
        );
        shard.push(3, false, Some(Birth::Partial(SmallVec::from_slice(&[30]))));
        let candidates = shard.commit(&events, false, 3).unwrap();
        assert!(!shard.counts.contains_key(&removed));
        assert!(!shard.counts.contains_key(&pruned));
        assert_eq!(shard.counts[&partial], 3);
        assert_eq!(shard.counts[&complete], 4);
        assert_eq!(
            candidates
                .iter()
                .map(|c| c.priority.pair())
                .collect::<Vec<_>>(),
            [complete, partial]
        );
        assert_eq!(
            candidates[1].positions.iter().collect::<Vec<_>>(),
            [20, 21, 22]
        );
    }

    #[test]
    fn failed_commit_drains_pending_streams_and_checks_intermediate_overflow() {
        let pair = (0, 1);
        for reuse in [false, true] {
            let mut shard = Shard::default();
            if reuse {
                shard.counts.insert(pair, i64::MAX as u64);
            }
            let events = [
                Event {
                    removed: pair,
                    born: pair,
                    removed_weight: 0,
                    born_weight: if reuse { 1 } else { u64::MAX },
                    bucket: 0,
                    positions: (),
                },
                Event {
                    removed: pair,
                    born: pair,
                    removed_weight: if reuse { 1 } else { 0 },
                    born_weight: if reuse { 0 } else { 1 },
                    bucket: 0,
                    positions: (),
                },
            ];
            shard.push(0, false, Some(Birth::Partial(SmallVec::from_slice(&[0]))));
            shard.push(1, reuse, Some(Birth::Partial(SmallVec::from_slice(&[1]))));
            let capacities = (shard.actions.capacity(), shard.positions.capacity());
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
            assert_eq!(
                (shard.actions.capacity(), shard.positions.capacity()),
                capacities
            );
        }
    }

    #[test]
    fn preparation_failure_leaves_all_endpoints_unchanged() {
        for workers in [1, 4, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            pool.install(|| {
                let trainer = BpeTrainer::builder()
                    .vocab_size(16)
                    .min_frequency(1)
                    .show_progress(false)
                    .build();
                let words = [("ab".into(), u64::MAX), ("xcd".into(), u64::MAX)]
                    .into_iter()
                    .collect();
                let view = WordCountsView::from_map(&words);
                let progress = trainer.setup_progress();
                let (mut vocabulary, ids) =
                    Vocabulary::initialize(&trainer, view, workers, &mut None, &progress).unwrap();
                let plan = CorpusPlan::build(view, ids, &trainer, false).unwrap();
                let mut index = PairIndex::build(&plan, 1, workers, false, &progress).unwrap();
                let mut corpus = plan.materialize();
                let mut candidates = std::mem::take(&mut index.queue).into_vec();
                let candidate = candidates
                    .iter_mut()
                    .find(|c| c.priority.pair() == (2, 3))
                    .unwrap();
                // AB has valid deferred writes. Duplicate CD's start to overflow
                // its left-neighbor count during preparation of the same round.
                let start = candidate.positions.iter().next().unwrap();
                candidate.positions = Positions::from_sorted(&[start, start]).unwrap();
                index.queue = candidates.into_iter().collect();
                let before: Vec<_> = (0..corpus.len()).map(|p| corpus.token(p)).collect();
                let error = trainer
                    .merge_next(&mut vocabulary, &mut corpus, &mut index)
                    .unwrap_err();
                assert!(error.to_string().contains("exceeds u64"), "{error}");
                let after: Vec<_> = (0..corpus.len()).map(|p| corpus.token(p)).collect();
                assert_eq!(after, before, "workers={workers}");
            });
        }
    }

    type Outcome = (Trace, (Vocab, Merges));

    #[test]
    fn owner_failure_returns_no_records_after_endpoint_writes() {
        for workers in [1, 4, 8] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap()
                .install(|| {
                    let trainer = BpeTrainer::builder()
                        .vocab_size(16)
                        .min_frequency(1)
                        .show_progress(false)
                        .build();
                    let words = [("abc".into(), 1)].into_iter().collect();
                    let view = WordCountsView::from_map(&words);
                    let progress = trainer.setup_progress();
                    let (mut vocabulary, ids) =
                        Vocabulary::initialize(&trainer, view, workers, &mut None, &progress)
                            .unwrap();
                    let plan = CorpusPlan::build(view, ids, &trainer, true).unwrap();
                    let mut index = PairIndex::build(&plan, 1, workers, true, &progress).unwrap();
                    let mut corpus = plan.materialize();
                    // AB writes ID 3 before its right birth (3, C) reaches its
                    // owner. Force that owner's signed addition to overflow.
                    let born = (3, 2);
                    index.shards[owner(born, workers)]
                        .counts
                        .insert(born, i64::MAX as u64);
                    let before: Vec<_> = (0..corpus.len()).map(|p| corpus.token(p)).collect();
                    let queued = index.queue.len();
                    let error = trainer
                        .merge_next(&mut vocabulary, &mut corpus, &mut index)
                        .unwrap_err();
                    assert!(error.to_string().contains("exceeds i64"), "{error}");
                    let after: Vec<_> = (0..corpus.len()).map(|p| corpus.token(p)).collect();
                    assert_ne!(after, before, "commit errors do not roll back writes");
                    assert!(after.contains(&3));
                    assert_eq!(
                        index.queue.len(),
                        queued - 1,
                        "failed owners publish no births"
                    );
                });
        }
    }

    fn train_with_equal_cohorts() -> Outcome {
        let mut trainer = BpeTrainer::builder()
            .vocab_size(64)
            .min_frequency(1)
            .show_progress(false)
            .build();
        trainer.continuing_subword_prefix = Some("ab".into());
        trainer.end_of_word_suffix = Some("c".into());
        trainer.special_tokens = vec![AddedToken::from("abac", true)];
        let mut words = AHashMap::with_hasher(ahash::RandomState::with_seeds(11, 13, 17, 19));
        words.extend([("pacx".into(), 4), ("yacx".into(), 1), ("pa".into(), 4)]);
        let view = WordCountsView::from_map(&words);
        let progress = trainer.setup_progress();
        let (mut vocabulary, ids) =
            Vocabulary::initialize(&trainer, view, 1, &mut None, &progress).unwrap();
        let pair = (
            ids.id('a', false, false).unwrap(),
            ids.id('c', false, false).unwrap(),
        );
        let plan = CorpusPlan::build(view, ids, &trainer, true).unwrap();
        // Either cohort reuses the activated "abac" identity. The heavier cohort
        // raises its left neighbor's existing count above the other cohort.
        let mut cohorts = [Vec::new(), Vec::new()];
        plan.initial_edges(0..plan.word_count(), |edge, position, weight| {
            if edge == pair {
                cohorts[usize::from(weight == 1)].push(position);
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(cohorts.each_ref().map(|values| values.len()), [1, 1]);
        let mut index = PairIndex::build(&plan, 1, 1, true, &progress).unwrap();
        assert_eq!(index.shards[0].counts[&pair], 5);
        let mut candidates = index.queue.into_vec();
        candidates.retain(|candidate| candidate.priority.pair() != pair);
        for (positions, count) in cohorts.into_iter().zip([10, 5]) {
            candidates.push(Candidate {
                priority: Priority {
                    count,
                    pair: Reverse(pair),
                },
                positions: Positions::from_sorted(&positions).unwrap(),
            });
        }
        index.queue = candidates.into_iter().collect();
        let mut corpus = plan.materialize();
        let mut trace = Vec::new();
        let mut merges = Vec::new();
        while vocabulary.len() < trainer.vocab_size {
            let records = match trainer
                .merge_next(&mut vocabulary, &mut corpus, &mut index)
                .unwrap()
            {
                MergeOutcome::Committed(records) => records,
                MergeOutcome::Finished => break,
                MergeOutcome::RestartForReuse => panic!("reuse cannot request another attempt"),
            };
            for record in records {
                merges.push(record.pair);
                trace.push((record.pair, record.count, record.replacement));
            }
        }
        (trace, vocabulary.into_model_parts(merges))
    }

    #[test]
    fn equal_pair_cohorts_preserve_each_merge_and_model_after_count_repair() {
        let actual = train_with_equal_cohorts();
        assert_eq!(
            actual.0,
            vec![
                ((6, 7), 5, 0),
                ((6, 7), 5, 0),
                ((3, 0), 8, 9),
                ((3, 0), 8, 9),
                ((9, 8), 4, 10),
                ((0, 8), 1, 11),
                ((0, 8), 1, 11),
                ((5, 11), 1, 12),
            ],
            "count repair must preserve the historical cohort merge order"
        );
        let tokens = [
            "abac", "a", "c", "p", "x", "y", "aba", "abc", "abxc", "pac", "pacxc", "abacxc",
            "yacxc",
        ];
        let vocab = tokens
            .iter()
            .enumerate()
            .map(|(id, text)| ((*text).to_owned(), id as u32))
            .collect();
        let merges = actual
            .0
            .iter()
            .map(|&(pair, _, _)| {
                (
                    tokens[pair.0 as usize].to_owned(),
                    tokens[pair.1 as usize].to_owned(),
                )
            })
            .collect();
        assert_eq!(
            actual.1,
            (vocab, merges),
            "cohort repair must preserve the complete model"
        );
    }
}
