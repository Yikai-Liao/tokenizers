//! Count owners publish complete lists; the queue repairs stale snapshots lazily.
use super::merge::{Birth, Change};
use super::{
    CorpusPlan, WORD_SEPARATOR_ID, add,
    positions::{Arena, Builder, Input, Positions},
};
use crate::progress::TrainingProgress;
use ahash::AHashMap;
use dary_heap::OctonaryHeap;
use rayon::prelude::*;
use std::cmp::{Ordering, Reverse};
use tk_encode::{Result, models::bpe::Pair};

// Highest count first, then smallest pair. Field order defines heap priority.
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
pub(super) struct Candidate<'arena> {
    pub(super) priority: Priority,
    pub(super) positions: Positions<'arena>,
}
impl Eq for Candidate<'_> {}
impl PartialEq for Candidate<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}
impl Ord for Candidate<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.cmp(&other.priority)
    }
}
impl PartialOrd for Candidate<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
#[derive(Default)]
struct State<P> {
    count: u64,
    positions: P,
}
#[derive(Default)]
struct Group {
    count: u64,
    positions: Builder,
    unordered: Vec<u64>,
}
// Metadata stays borrowed during commit; each position stream has one owner.
type Event = Change<()>;
#[derive(Default)]
struct Route<'arena> {
    actions: Vec<(usize, bool, bool)>,
    // One stream per birth action, in the same order.
    positions: Vec<Birth<'arena>>,
}
impl<'arena> Route<'arena> {
    fn push(&mut self, index: usize, remove: bool, birth: Option<Birth<'arena>>) {
        self.actions.push((index, remove, birth.is_some()));
        self.positions.extend(birth);
    }
    fn drain(&mut self) -> impl Iterator<Item = (usize, bool, Option<Birth<'arena>>)> + '_ {
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
pub(super) struct PairIndex<'arena> {
    arena: &'arena Arena,
    shards: Vec<AHashMap<Pair, u64>>,
    routes: Vec<Route<'arena>>,
    queue: OctonaryHeap<Candidate<'arena>>,
    floor: u64,
    reuse: bool,
}
fn owner(pair: Pair, workers: usize) -> usize {
    let key = (u64::from(pair.0) << 32) | u64::from(pair.1);
    (key.wrapping_mul(0x9e3779b97f4a7c15).rotate_left(23) % workers as u64) as usize
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
impl<'arena> PairIndex<'arena> {
    pub(super) fn build(
        arena: &'arena Arena,
        corpus: &CorpusPlan<'_>,
        minimum: u64,
        workers: usize,
        reuse: bool,
        progress: &TrainingProgress,
    ) -> Result<Self> {
        let work = progress.stage("Count initial pairs", corpus.word_count());
        let pieces = corpus
            .initial_ranges(if cfg!(test) { 16 } else { 1 << 24 })
            .into_par_iter()
            .map(|range| -> Result<_> {
                let domain = corpus.small_pair_domain();
                let mut dense: Vec<State<Builder>> = (0..domain.map_or(0, |n| n * n))
                    .map(|_| State::default())
                    .collect();
                let mut counts = AHashMap::<Pair, State<Builder>>::new();
                corpus.initial_edges(range.clone(), |pair, p, weight| {
                    debug_assert!(pair.0 != WORD_SEPARATOR_ID && pair.1 != WORD_SEPARATOR_ID);
                    let state = match domain {
                        Some(n) => &mut dense[pair.0 as usize * n + pair.1 as usize],
                        None => counts.entry(pair).or_default(),
                    };
                    add(&mut state.count, weight)?;
                    state.positions.push(p)
                })?;
                work.complete(range.len());
                let states = match domain {
                    Some(n) => dense
                        .into_iter()
                        .enumerate()
                        .filter(|(_, s)| !s.positions.is_empty())
                        .map(|(key, state)| (((key / n) as u32, (key % n) as u32), state))
                        .collect::<Vec<_>>(),
                    None => counts.into_iter().collect(),
                };
                let mut lease = arena.lease();
                states
                    .into_iter()
                    .map(|(pair, state)| {
                        let positions = Positions::from_sorted_owned(
                            Input::Builder(&state.positions),
                            &mut lease,
                        )?;
                        Ok((
                            pair,
                            State {
                                count: state.count,
                                positions,
                            },
                        ))
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let mut routed: Vec<Vec<_>> = (0..workers).map(|_| Vec::new()).collect();
        for piece in pieces {
            for (pair, state) in piece {
                routed[owner(pair, workers)].push((pair, state));
            }
        }
        let owners = routed
            .into_par_iter()
            .map(|pieces| -> Result<_> {
                let mut states = AHashMap::<Pair, State<Vec<Positions>>>::new();
                for (pair, state) in pieces {
                    let total = states.entry(pair).or_default();
                    add(&mut total.count, state.count)?;
                    total.positions.push(state.positions);
                }
                let mut lease = arena.lease();
                let mut shard = AHashMap::new();
                let mut candidates = Vec::new();
                for (pair, mut state) in states {
                    if reuse || state.count >= minimum.max(1) {
                        let positions = if state.positions.len() == 1 {
                            state.positions.pop().unwrap()
                        } else {
                            Positions::from_sorted(Input::Fragments(&state.positions), &mut lease)?
                        };
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
            })
            .collect::<Result<Vec<_>>>()?;
        let (shards, candidates): (Vec<_>, Vec<_>) = owners.into_iter().unzip();
        Ok(Self {
            arena,
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
    pub(super) fn take(&mut self, priority: Priority) -> Candidate<'arena> {
        let candidate = self.queue.pop().expect("certified candidate exists");
        debug_assert_eq!(candidate.priority, priority);
        if !self.reuse {
            let workers = self.shards.len();
            self.shards[owner(priority.pair(), workers)]
                .remove(&priority.pair())
                .expect("certified pair exists");
        }
        candidate
    }
    pub(super) fn commit(&mut self, changes: Vec<Vec<Change<Birth<'arena>>>>) -> Result<()> {
        let workers = self.shards.len();
        self.routes.resize_with(workers, Route::default);
        // Joined commits drain every route; any error aborts this attempt.
        let mut events = Vec::with_capacity(changes.iter().map(Vec::len).sum());
        for change in changes.into_iter().flatten() {
            let removed = (change.removed_weight != 0).then(|| owner(change.removed, workers));
            // Zero-weight births still own positions, including reuse cohorts.
            let born = (!change.positions.is_empty()).then(|| owner(change.born, workers));
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
        let reuse = self.reuse;
        let floor = self.floor;
        let arena = self.arena;
        let births = self
            .shards
            .par_iter_mut()
            .zip(self.routes.par_iter_mut())
            .filter(|(_, route)| !route.actions.is_empty())
            .map(|(shard, route)| -> Result<_> {
                let mut lease = arena.lease();
                let mut groups = AHashMap::<(usize, Pair), Group>::new();
                let mut candidates = Vec::new();
                for (index, remove, birth) in route.drain() {
                    let change = &events[index];
                    // A boundary removal precedes its replacement birth. Reordering
                    // these actions changes signed alias counts and their error boundary.
                    if remove {
                        if reuse {
                            let count = shard.entry(change.removed).or_default();
                            adjust_signed(count, change.removed_weight, true)?;
                        } else if let Some(state) = shard.get_mut(&change.removed) {
                            *state = state
                                .checked_sub(change.removed_weight)
                                .ok_or("BPE fresh removal exceeds the current count")?;
                            if *state < floor {
                                shard.remove(&change.removed);
                            }
                        }
                    }
                    if let Some(positions) = birth {
                        if reuse {
                            let count = shard.entry(change.born).or_default();
                            adjust_signed(count, change.born_weight, false)?;
                        }
                        let positions = match positions {
                            Birth::Complete(positions) => {
                                // Fresh IDs and compatible rules give each complete birth one producer.
                                debug_assert!(!reuse && change.born_weight >= floor);
                                debug_assert!(!shard.contains_key(&change.born));
                                shard.insert(change.born, change.born_weight);
                                candidates.push(Candidate {
                                    priority: Priority {
                                        count: change.born_weight,
                                        pair: Reverse(change.born),
                                    },
                                    positions,
                                });
                                continue;
                            }
                            Birth::Partial(positions) => positions,
                        };
                        let group = groups.entry((change.bucket, change.born)).or_default();
                        add(&mut group.count, change.born_weight)?;
                        if reuse {
                            group.unordered.extend(positions.iter());
                        } else {
                            // Fresh jobs follow rule rank and spatial ranges. Each
                            // born key has one producer, so lists concatenate sorted.
                            group.positions.append(positions)?;
                        }
                    }
                }
                let mut groups: Vec<_> = groups.into_iter().collect();
                groups.sort_unstable_by_key(|((bucket, pair), _)| (*bucket, *pair));
                for ((_, pair), mut state) in groups {
                    let count = if reuse { shard[&pair] } else { state.count };
                    // A positive reuse ledger publishes a historical birth cohort even
                    // below the floor; selection applies the floor after correcting its head.
                    if (reuse && (count as i64) <= 0) || (!reuse && count < floor) {
                        continue;
                    }
                    let priority = Priority {
                        count,
                        pair: Reverse(pair),
                    };
                    let positions = if reuse {
                        state.unordered.sort_unstable();
                        Positions::from_sorted(Input::Slice(&state.unordered), &mut lease)?
                    } else {
                        debug_assert!(!shard.contains_key(&pair));
                        shard.insert(pair, count);
                        Positions::from_sorted(Input::Builder(&state.positions), &mut lease)?
                    };
                    candidates.push(Candidate {
                        priority,
                        positions,
                    });
                }
                Ok(candidates)
            })
            .collect::<Result<Vec<_>>>()?;
        for candidate in births.into_iter().flatten() {
            self.queue.push(candidate);
        }
        Ok(())
    }
}
