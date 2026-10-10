//! Count owners publish complete lists; the queue repairs stale snapshots lazily.
use super::merge::{Birth, Change};
use super::{WORD_SEPARATOR_ID, add, corpus::CorpusPlan, positions::Positions};
use ahash::AHashMap;
use dary_heap::OctonaryHeap;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::cmp::{Ordering, Reverse};
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

/// Initial pair count and positions, mutable while counting and frozen for routing.
/// Mutable SmallVec storage becomes Positions without duplicating the weighted count.
#[derive(Default)]
struct State<P> {
    count: u64,
    positions: P,
}

/// Owner-local partial births for one (producer bucket, pair), awaiting publication.
/// Fresh fragments concatenate in order; reuse positions accumulate unordered
/// and are sorted, retaining duplicates, before their historical cohort is frozen.
struct Group {
    count: u64,
    positions: GroupPositions,
}

enum GroupPositions {
    Ordered(SmallVec<[u64; 4]>),
    Unordered(Vec<u64>),
}

impl Group {
    fn new(reuse: bool) -> Self {
        Self {
            count: 0,
            positions: if reuse {
                GroupPositions::Unordered(Vec::new())
            } else {
                GroupPositions::Ordered(SmallVec::new())
            },
        }
    }
}

// Metadata stays borrowed during commit; each position stream has one owner.
type Event = Change<()>;

/// Reusable actions destined for one pair-count owner, in producer order.
/// Actions index shared event metadata; the separate birth array owns one stream
/// per birth action and avoids reserving a payload for removal-only actions.
#[derive(Default)]
struct Route {
    actions: Vec<(usize, bool, bool)>,
    // One stream per birth action, in the same order.
    positions: Vec<Birth>,
}

impl Route {
    fn push(&mut self, index: usize, remove: bool, birth: Option<Birth>) {
        self.actions.push((index, remove, birth.is_some()));
        self.positions.extend(birth);
    }

    fn drain(&mut self) -> impl Iterator<Item = (usize, bool, Option<Birth>)> + '_ {
        // The two arrays save a large optional payload on removal-only actions.
        // Both drains own their remaining items, so an error drops unpublished births
        // and restores empty reusable routes before the failed attempt is discarded.
        let mut positions = self.positions.drain(..);
        self.actions.drain(..).map(move |(index, remove, birth)| {
            (
                index,
                remove,
                birth.then(|| positions.next().expect("each birth owns one stream")),
            )
        })
    }
}

/// Authoritative pair-count shards plus an owning queue of occurrence candidates.
/// Each pair has one count owner; queued priorities are repaired lazily at the head.
/// Fresh counts retire below the floor; reuse retains the per-action signed ledger.
pub(super) struct PairIndex {
    shards: Vec<AHashMap<Pair, u64>>,
    routes: Vec<Route>,
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
            routes: Vec::new(),
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
            let Some(&count) = self.shards[owner(top.pair(), self.shards.len())].get(&top.pair())
            else {
                // Fresh counts retire below the floor. Their stale payloads are
                // reclaimed when they reach the head of this owning queue.
                self.queue.pop();
                continue;
            };
            if top.count == count {
                return (count >= self.floor).then_some(top);
            }
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
            .zip(self.routes.par_iter_mut())
            .filter(|(_, route)| !route.actions.is_empty())
            .map(|(shard, route)| -> Result<_> {
                OwnerCommit::new(shard, reuse, floor).commit(route, &events)
            })
            .collect::<Result<Vec<_>>>()?;

        // 3. Publish only after all owners join, in the same owner traversal order.
        for candidate in births.into_iter().flatten() {
            self.queue.push(candidate);
        }
        Ok(())
    }

    fn route_changes(&mut self, changes: Vec<Vec<Change<Birth>>>) -> Vec<Event> {
        let shard_count = self.shards.len();
        self.routes.resize_with(shard_count, Route::default);
        // Joined commits drain every route; any error aborts this attempt.
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
                    self.routes[removed].push(index, true, Some(change.positions));
                }
                (removed, born) => {
                    if let Some(removed) = removed {
                        self.routes[removed].push(index, true, None);
                    }
                    if let Some(born) = born {
                        self.routes[born].push(index, false, Some(change.positions));
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
        states: Vec<State<SmallVec<[u64; 4]>>>,
    },
    Sparse(AHashMap<Pair, State<SmallVec<[u64; 4]>>>),
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

    fn into_states(self) -> Vec<(Pair, State<SmallVec<[u64; 4]>>)> {
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

fn build_owner(
    pieces: InitialPiece,
    floor: u64,
    reuse: bool,
) -> Result<(AHashMap<Pair, u64>, Vec<Candidate>)> {
    let mut states = AHashMap::<Pair, State<Vec<Positions>>>::new();
    for (pair, state) in pieces {
        let total = states.entry(pair).or_default();
        add(&mut total.count, state.count)?;
        total.positions.push(state.positions);
    }
    let mut shard = AHashMap::new();
    let mut candidates = Vec::new();
    for (pair, state) in states {
        if reuse || state.count >= floor {
            let positions = Positions::concat(state.positions)?;
            shard.insert(pair, state.count);
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

/// One owner's sequential count update and birth publication within a joined commit.
/// Borrows its count shard and owns pending groups and candidates.
/// Failed work drops unpublished lists; the attempt is discarded.
struct OwnerCommit<'index> {
    shard: &'index mut AHashMap<Pair, u64>,
    reuse: bool,
    floor: u64,
    groups: AHashMap<(usize, Pair), Group>,
    candidates: Vec<Candidate>,
}

impl<'index> OwnerCommit<'index> {
    fn new(shard: &'index mut AHashMap<Pair, u64>, reuse: bool, floor: u64) -> Self {
        Self {
            shard,
            reuse,
            floor,
            groups: AHashMap::new(),
            candidates: Vec::new(),
        }
    }

    // A boundary removal precedes its replacement birth. The signed reuse ledger
    // checks each action in that order, including intermediate overflow.
    fn commit(mut self, route: &mut Route, events: &[Event]) -> Result<Vec<Candidate>> {
        for (index, remove, birth) in route.drain() {
            let change = &events[index];
            if remove {
                self.remove_weight(change.removed, change.removed_weight)?;
            }
            if let Some(positions) = birth {
                self.record_birth(change, positions)?;
            }
        }
        self.publish_groups()
    }

    fn remove_weight(&mut self, pair: Pair, weight: u64) -> Result<()> {
        if self.reuse {
            let count = self.shard.entry(pair).or_default();
            adjust_signed(count, weight, true)?;
        } else if let Some(state) = self.shard.get_mut(&pair) {
            *state = state
                .checked_sub(weight)
                .ok_or("BPE fresh removal exceeds the current count")?;
            if *state < self.floor {
                self.shard.remove(&pair);
            }
        }
        Ok(())
    }

    fn record_birth(&mut self, change: &Event, positions: Birth) -> Result<()> {
        if self.reuse {
            let count = self.shard.entry(change.born).or_default();
            adjust_signed(count, change.born_weight, false)?;
        }
        let positions = match positions {
            Birth::Complete(positions) => {
                // Fresh IDs and compatible rules give each complete birth one producer.
                debug_assert!(!self.reuse && change.born_weight >= self.floor);
                debug_assert!(!self.shard.contains_key(&change.born));
                self.shard.insert(change.born, change.born_weight);
                self.candidates.push(Candidate {
                    priority: Priority {
                        count: change.born_weight,
                        pair: Reverse(change.born),
                    },
                    positions,
                });
                return Ok(());
            }
            Birth::Partial(positions) => positions,
        };
        let group = self
            .groups
            .entry((change.bucket, change.born))
            .or_insert_with(|| Group::new(self.reuse));
        add(&mut group.count, change.born_weight)?;
        match &mut group.positions {
            GroupPositions::Unordered(values) => values.extend(positions.iter().copied()),
            GroupPositions::Ordered(values) => {
                // Fresh jobs follow rule rank and spatial ranges. Each
                // born key has one producer, so lists concatenate sorted.
                if values.is_empty() {
                    *values = positions;
                } else {
                    values.extend(positions);
                }
            }
        }
        Ok(())
    }

    fn publish_groups(mut self) -> Result<Vec<Candidate>> {
        // Partial birth groups follow bucket/pair order within this owner. Complete
        // producers have already published directly into its candidate list.
        let mut groups: Vec<_> = self.groups.into_iter().collect();
        groups.sort_unstable_by_key(|((bucket, pair), _)| (*bucket, *pair));
        for ((_, pair), state) in groups {
            let count = if self.reuse {
                self.shard[&pair]
            } else {
                state.count
            };
            // A positive reuse ledger publishes a historical birth cohort even
            // below the floor; selection applies the floor after correcting its head.
            if (self.reuse && (count as i64) <= 0) || (!self.reuse && count < self.floor) {
                continue;
            }
            let priority = Priority {
                count,
                pair: Reverse(pair),
            };
            let positions = match state.positions {
                GroupPositions::Unordered(mut values) => {
                    values.sort_unstable();
                    Positions::from_sorted(&values)?
                }
                GroupPositions::Ordered(values) => {
                    debug_assert!(!self.shard.contains_key(&pair));
                    self.shard.insert(pair, count);
                    Positions::from_sorted(&values)?
                }
            };
            self.candidates.push(Candidate {
                priority,
                positions,
            });
        }

        Ok(self.candidates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainers::bpe::{
        BpeTrainer, Trace,
        merge::{Batch, Selection},
        vocabulary::Vocabulary,
        word_counts::WordCountsView,
    };
    use tk_encode::{
        models::bpe::{Merges, Vocab},
        vocab::bucket_added_vocabulary::AddedToken,
    };

    type Outcome = (Trace, (Vocab, Merges));

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
        let mut vocabulary = Vocabulary::initialize(&trainer, view, 1, &mut None).unwrap();
        let ids = vocabulary.initial_ids(view, &progress).unwrap();
        let pair = (
            ids.id('a', false, false).unwrap(),
            ids.id('c', false, false).unwrap(),
        );
        let plan = CorpusPlan::build(view, &mut vocabulary, &trainer, true, &progress).unwrap();
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
        assert_eq!(index.shards[0][&pair], 5);
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
            let batch =
                match Batch::select(&trainer, &mut vocabulary, &mut corpus, &mut index).unwrap() {
                    Selection::Ready(batch) => batch,
                    Selection::Finished => break,
                    Selection::Restart => panic!("reuse cannot request another attempt"),
                };
            trace.extend(batch.trace());
            merges.extend(batch.pairs());
            let prepared = batch.prepare(&corpus, usize::MAX).unwrap();
            index.commit(prepared.apply(&corpus)).unwrap();
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
