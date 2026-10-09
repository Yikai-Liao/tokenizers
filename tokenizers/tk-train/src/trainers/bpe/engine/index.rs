//! Count owners publish complete lists; the queue repairs stale snapshots lazily.
use super::merge::Change;
use super::{CorpusPlan, WORD_SEPARATOR_ID, positions::Positions};
use crate::progress::TrainingProgress;
use ahash::AHashMap;
use dary_heap::OctonaryHeap;
use rayon::prelude::*;
use std::cmp::Ordering;
use tk_encode::{Result, models::bpe::Pair};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Priority {
    pub(super) pair: Pair,
    pub(super) count: u64,
}
impl Ord for Priority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count
            .cmp(&other.count)
            .then_with(|| other.pair.cmp(&self.pair))
    }
}
impl PartialOrd for Priority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
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
#[derive(Default)]
struct State {
    count: u64,
    positions: Positions,
}
#[derive(Default)]
struct Group {
    count: u64,
    positions: Positions,
    unordered: Vec<u64>,
}
#[derive(Default)]
struct Shard {
    states: AHashMap<Pair, State>,
    queue: OctonaryHeap<Priority>,
}
impl Shard {
    fn publish(&mut self, pair: Pair, count: u64, positions: Positions) {
        debug_assert!(!self.states.contains_key(&pair));
        self.states.insert(pair, State { count, positions });
        self.queue.push(Priority { pair, count });
    }
}
pub(super) struct PairIndex {
    shards: Vec<Shard>,
    cohorts: OctonaryHeap<Candidate>,
    floor: u64,
    reuse: bool,
}
fn owner(pair: Pair, workers: usize) -> usize {
    let key = (u64::from(pair.0) << 32) | u64::from(pair.1);
    (key.wrapping_mul(0x9e3779b97f4a7c15).rotate_left(23) % workers as u64) as usize
}
fn add(count: &mut u64, amount: u64) -> Result<()> {
    *count = count
        .checked_add(amount)
        .ok_or("BPE pair frequency exceeds u64")?;
    Ok(())
}
impl PairIndex {
    pub(super) fn build(
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
                let mut dense: Vec<State> = (0..domain.map_or(0, |n| n * n))
                    .map(|_| State::default())
                    .collect();
                let mut counts = AHashMap::<Pair, State>::new();
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
                let mut shard = Shard::default();
                for (pair, state) in pieces {
                    let total = shard.states.entry(pair).or_default();
                    add(&mut total.count, state.count)?;
                    total.positions.append(state.positions)?;
                }
                shard
                    .states
                    .retain(|_, state| reuse || state.count >= minimum.max(1));
                if !reuse {
                    shard
                        .queue
                        .extend(shard.states.iter().map(|(&pair, state)| Priority {
                            pair,
                            count: state.count,
                        }));
                }
                Ok(shard)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut index = Self {
            shards,
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
                            pair,
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
                let count = self.shards[owner(top.pair, self.shards.len())].states[&top.pair].count;
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
                let Some(state) = shard.states.get(&top.pair) else {
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
    pub(super) fn take(&mut self, priority: Priority) -> Candidate {
        if self.reuse {
            return self.cohorts.pop().expect("certified cohort exists");
        }
        let workers = self.shards.len();
        let shard = &mut self.shards[owner(priority.pair, workers)];
        shard.queue.pop();
        let state = shard
            .states
            .remove(&priority.pair)
            .expect("certified pair exists");
        Candidate {
            priority,
            positions: state.positions,
        }
    }
    pub(super) fn commit(&mut self, changes: Vec<Change>) -> Result<()> {
        let workers = self.shards.len();
        let mut routes: Vec<Vec<(Change, bool, bool)>> = (0..workers).map(|_| Vec::new()).collect();
        for change in changes {
            let removed = owner(change.removed, workers);
            let born = owner(change.born, workers);
            if removed == born {
                routes[removed].push((change, true, true));
            } else {
                let removal = Change {
                    positions: Positions::default(),
                    born_weight: 0,
                    ..change
                };
                routes[removed].push((removal, true, false));
                routes[born].push((change, false, true));
            }
        }
        let reuse = self.reuse;
        let floor = self.floor;
        let births = self
            .shards
            .par_iter_mut()
            .zip(routes)
            .map(|(shard, route)| -> Result<_> {
                let mut groups = AHashMap::<(usize, Pair), Group>::new();
                for (change, remove, birth) in route {
                    if remove {
                        if reuse {
                            let count = &mut shard.states.entry(change.removed).or_default().count;
                            let amount = i64::try_from(change.removed_weight)
                                .map_err(|_| "BPE identity-reuse removal exceeds i64")?;
                            *count = (*count as i64)
                                .checked_sub(amount)
                                .ok_or("BPE identity-reuse count subtraction exceeds i64")?
                                as u64;
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
                    if birth {
                        if reuse {
                            let count = &mut shard.states.entry(change.born).or_default().count;
                            let amount = i64::try_from(change.born_weight)
                                .map_err(|_| "BPE identity-reuse birth exceeds i64")?;
                            *count = (*count as i64)
                                .checked_add(amount)
                                .ok_or("BPE identity-reuse count addition exceeds i64")?
                                as u64;
                        }
                        if !change.positions.is_empty() {
                            if change.complete {
                                // The complete producer already reduced and pruned.
                                // Fresh IDs and compatible rules give each birth one producer.
                                debug_assert!(!reuse && change.born_weight >= floor);
                                shard.publish(change.born, change.born_weight, change.positions);
                                continue;
                            }
                            let group = groups.entry((change.bucket, change.born)).or_default();
                            add(&mut group.count, change.born_weight)?;
                            if reuse {
                                group.unordered.extend(change.positions.iter());
                            } else {
                                // Fresh jobs follow rule rank and spatial ranges. Each
                                // born key has one producer, so lists concatenate sorted.
                                group.positions.append(change.positions)?;
                            }
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
                    if if reuse {
                        (count as i64) <= 0
                    } else {
                        count < floor
                    } {
                        continue;
                    }
                    let priority = Priority { pair, count };
                    if reuse {
                        state.unordered.sort_unstable();
                        candidates.push(Candidate {
                            priority,
                            positions: Positions::from_sorted(&state.unordered)?,
                        });
                    } else {
                        shard.publish(pair, count, state.positions);
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
