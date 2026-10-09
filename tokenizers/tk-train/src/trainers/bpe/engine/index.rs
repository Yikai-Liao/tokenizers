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
#[derive(Default)]
struct Shard<'arena> {
    states: AHashMap<Pair, State<Positions<'arena>>>,
    queue: OctonaryHeap<Priority>,
}
impl<'arena> Shard<'arena> {
    fn publish(&mut self, pair: Pair, count: u64, positions: Positions<'arena>) {
        debug_assert!(!self.states.contains_key(&pair));
        self.states.insert(pair, State { count, positions });
        self.queue.push(Priority {
            count,
            pair: Reverse(pair),
        });
    }
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
    shards: Vec<Shard<'arena>>,
    routes: Vec<Route<'arena>>,
    cohorts: OctonaryHeap<Candidate<'arena>>,
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
        let chunk = corpus
            .word_count()
            .div_ceil(workers.saturating_mul(4))
            .max(1);
        let pieces = (0..corpus.word_count())
            .step_by(chunk)
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|begin| -> Result<_> {
                let domain = corpus.small_pair_domain();
                let mut dense: Vec<State<Builder>> = (0..domain.map_or(0, |n| n * n))
                    .map(|_| State::default())
                    .collect();
                let mut counts = AHashMap::<Pair, State<Builder>>::new();
                corpus.initial_edges(
                    begin..(begin + chunk).min(corpus.word_count()),
                    |pair, p, weight| {
                        debug_assert!(pair.0 != WORD_SEPARATOR_ID && pair.1 != WORD_SEPARATOR_ID);
                        let state = match domain {
                            Some(n) => &mut dense[pair.0 as usize * n + pair.1 as usize],
                            None => counts.entry(pair).or_default(),
                        };
                        add(&mut state.count, weight)?;
                        state.positions.push(p)
                    },
                )?;
                work.complete(chunk.min(corpus.word_count() - begin));
                Ok(match domain {
                    Some(n) => dense
                        .into_iter()
                        .enumerate()
                        .filter(|(_, s)| !s.positions.is_empty())
                        .map(|(key, state)| (((key / n) as u32, (key % n) as u32), state))
                        .collect::<Vec<_>>(),
                    None => counts.into_iter().collect(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut routed: Vec<Vec<_>> = (0..workers).map(|_| Vec::new()).collect();
        for piece in pieces {
            for (pair, state) in piece {
                routed[owner(pair, workers)].push((pair, state));
            }
        }
        let shards = routed
            .into_par_iter()
            .map(|pieces| -> Result<_> {
                let mut states = AHashMap::<Pair, State<Builder>>::new();
                for (pair, state) in pieces {
                    let total = states.entry(pair).or_default();
                    add(&mut total.count, state.count)?;
                    total.positions.append(state.positions)?;
                }
                let mut lease = arena.lease();
                let mut shard = Shard::default();
                for (pair, state) in states {
                    if reuse || state.count >= minimum.max(1) {
                        let positions =
                            Positions::from_sorted(Input::Builder(&state.positions), &mut lease)?;
                        shard.states.insert(
                            pair,
                            State {
                                count: state.count,
                                positions,
                            },
                        );
                    }
                }
                if !reuse {
                    shard
                        .queue
                        .extend(shard.states.iter().map(|(&pair, state)| Priority {
                            pair: Reverse(pair),
                            count: state.count,
                        }));
                }
                Ok(shard)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut index = Self {
            arena,
            shards,
            routes: Vec::new(),
            cohorts: OctonaryHeap::new(),
            floor: minimum.max(1),
            reuse,
        };
        if reuse {
            for shard in &mut index.shards {
                for (&pair, state) in &mut shard.states {
                    if state.count == 0 {
                        continue;
                    }
                    index.cohorts.push(Candidate {
                        priority: Priority {
                            pair: Reverse(pair),
                            count: state.count,
                        },
                        positions: std::mem::take(&mut state.positions),
                    });
                }
            }
        }
        Ok(index)
    }
    pub(super) fn reuse(&self) -> bool {
        self.reuse
    }
    pub(super) fn best(&mut self) -> Option<Priority> {
        if self.reuse {
            loop {
                let top = self.cohorts.peek()?.priority;
                let count =
                    self.shards[owner(top.pair(), self.shards.len())].states[&top.pair()].count;
                if top.count == count {
                    return (count >= self.floor).then_some(top);
                }
                let mut candidate = self.cohorts.pop().expect("observed candidate exists");
                candidate.priority.count = count;
                self.cohorts.push(candidate);
            }
        }
        let mut best = None;
        for shard in &mut self.shards {
            while let Some(top) = shard.queue.peek().copied() {
                let Some(state) = shard.states.get(&top.pair()) else {
                    shard.queue.pop();
                    continue;
                };
                if top.count == state.count {
                    if best.is_none_or(|previous| top > previous) {
                        best = Some(top);
                    }
                    break;
                }
                let count = state.count;
                shard.queue.pop();
                shard.queue.push(Priority { count, ..top });
            }
        }
        best
    }
    pub(super) fn take(&mut self, priority: Priority) -> Candidate<'arena> {
        if self.reuse {
            return self.cohorts.pop().expect("certified cohort exists");
        }
        let workers = self.shards.len();
        let shard = &mut self.shards[owner(priority.pair(), workers)];
        shard.queue.pop();
        let state = shard
            .states
            .remove(&priority.pair())
            .expect("certified pair exists");
        Candidate {
            priority,
            positions: state.positions,
        }
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
                for (index, remove, birth) in route.drain() {
                    let change = &events[index];
                    // A boundary removal precedes its replacement birth. Reordering
                    // these actions changes signed alias counts and their error boundary.
                    if remove {
                        if reuse {
                            let count = &mut shard.states.entry(change.removed).or_default().count;
                            adjust_signed(count, change.removed_weight, true)?;
                        } else if let Some(state) = shard.states.get_mut(&change.removed) {
                            state.count = state
                                .count
                                .checked_sub(change.removed_weight)
                                .ok_or("BPE fresh removal exceeds the current count")?;
                            if state.count < floor {
                                shard.states.remove(&change.removed);
                            }
                        }
                    }
                    if let Some(positions) = birth {
                        if reuse {
                            let count = &mut shard.states.entry(change.born).or_default().count;
                            adjust_signed(count, change.born_weight, false)?;
                        }
                        let positions = match positions {
                            Birth::Complete(positions) => {
                                // Fresh IDs and compatible rules give each complete birth one producer.
                                debug_assert!(!reuse && change.born_weight >= floor);
                                shard.publish(change.born, change.born_weight, positions);
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
                let mut candidates = Vec::new();
                let mut groups: Vec<_> = groups.into_iter().collect();
                groups.sort_unstable_by_key(|((bucket, pair), _)| (*bucket, *pair));
                for ((_, pair), mut state) in groups {
                    let count = if reuse {
                        shard.states[&pair].count
                    } else {
                        state.count
                    };
                    // A positive reuse ledger publishes a historical birth cohort even
                    // below the floor; selection applies the floor after correcting its head.
                    if (reuse && (count as i64) <= 0) || (!reuse && count < floor) {
                        continue;
                    }
                    let priority = Priority {
                        count,
                        pair: Reverse(pair),
                    };
                    if reuse {
                        state.unordered.sort_unstable();
                        let positions =
                            Positions::from_sorted(Input::Slice(&state.unordered), &mut lease)?;
                        candidates.push(Candidate {
                            priority,
                            positions,
                        });
                    } else {
                        let positions =
                            Positions::from_sorted(Input::Builder(&state.positions), &mut lease)?;
                        shard.publish(pair, count, positions);
                    }
                }
                Ok(candidates)
            })
            .collect::<Result<Vec<_>>>()?;
        for candidate in births.into_iter().flatten() {
            self.cohorts.push(candidate);
        }
        Ok(())
    }
}
