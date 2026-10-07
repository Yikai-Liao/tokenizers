//! Stable full-pair grouping decouples checked count work and birth encoding
//! from persistent hash owners. Reuse retains its original ordered commit.
use super::super::{
    execution::Execution,
    merge::{CompletedBirth, MergeEvents},
    storage::{AllocationArena, radix},
};
use super::*;

struct BirthJob<'records> {
    key: u64,
    weight: u64,
    count: usize,
    references: &'records [Reference],
}
#[derive(Clone, Copy, Default)]
struct Reference {
    key: u64,
    chunk: usize,
    index: usize,
}
impl radix::RadixRecord for Reference {
    fn key(self) -> u64 {
        self.key
    }
}

impl<'arena> PairIndex<'arena> {
    pub(super) fn commit_data_births(
        &mut self,
        events: &MergeEvents,
        _identities: usize,
        execution: &Execution,
        arena: &'arena AllocationArena,
        floor: u64,
    ) -> Result<()> {
        let router = ShardRouter::new(self.shards.len());
        let removals = sorted_references(events, false);
        let updates = key_groups(&removals)
            .into_par_iter()
            .map(|group| -> Result<_> {
                let key = group[0].key;
                let mut count = self.shards[router.owner(key)]
                    .states
                    .get(&key)
                    .map(|state| state.ledger_count_bits);
                // Retirement is a state transition, not a net subtraction. Once
                // retired, later actions for this key are ignored exactly as before.
                for reference in group {
                    if let Some(value) = count {
                        let amount =
                            events.chunks[reference.chunk].changes[reference.index].removed_weight;
                        let next = value
                            .checked_sub(amount)
                            .ok_or("BPE fresh removal exceeds the current count")?;
                        count = (next >= floor).then_some(next);
                    }
                }
                Ok((key, count))
            })
            .collect::<Result<Vec<_>>>()?;
        drop(removals);
        let mut owner_updates: Vec<Vec<_>> = (0..self.shards.len()).map(|_| Vec::new()).collect();
        for update in updates {
            owner_updates[router.owner(update.0)].push(update);
        }
        let changed: Vec<_> = owner_updates
            .iter()
            .map(|updates| !updates.is_empty())
            .collect();
        self.shards
            .par_iter_mut()
            .zip(owner_updates)
            .for_each(|(shard, updates)| {
                for (key, count) in updates {
                    if let Some(count) = count {
                        shard
                            .states
                            .get_mut(&key)
                            .expect("snapshot state exists")
                            .ledger_count_bits = count;
                    } else {
                        shard.states.remove(&key);
                    }
                }
            });
        let references = sorted_references(events, true);
        let groups = key_groups(&references);
        let mut jobs = groups
            .into_par_iter()
            .map(|group| -> Result<Option<BirthJob<'_>>> {
                let mut weight = 0_u64;
                let mut count = 0_usize;
                for reference in group {
                    let chunk = &events.chunks[reference.chunk];
                    let change = &chunk.changes[reference.index];
                    weight = weight
                        .checked_add(change.born_weight)
                        .ok_or("BPE birth frequency exceeds u64")?;
                    count = count
                        .checked_add(change.positions.len())
                        .ok_or("BPE birth position count exceeds resident bounds")?;
                }
                if weight < floor {
                    return Ok(None);
                }
                Ok(Some(BirthJob {
                    key: group[0].key,
                    weight,
                    count,
                    references: group,
                }))
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        jobs.sort_unstable_by_key(|job| std::cmp::Reverse(job.count));
        let completed = jobs
            .into_par_iter()
            .map(|job| -> Result<CompletedBirth<'arena>> {
                let sources = job.references.iter().rev().map(|reference| {
                    let chunk = &events.chunks[reference.chunk];
                    (&chunk.chains, chunk.changes[reference.index].positions)
                });
                let positions = if job.count >= 16_384 {
                    SortedPositions::from_cooperative_chains(job.count, sources, arena)?
                } else {
                    let worker = execution.current_worker();
                    let lease = arena.lease(worker);
                    let mut scratch = execution.encoding(worker);
                    SortedPositions::from_reversed_iter(
                        job.count,
                        sources.flat_map(|(owner, chain)| owner.reversed(chain)),
                        &mut scratch,
                        &lease,
                    )?
                };
                Ok(CompletedBirth {
                    key: job.key,
                    weight: job.weight,
                    positions,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        for birth in completed {
            self.prepared_births[router.owner(birth.key)].push(birth);
        }
        self.shards
            .par_iter_mut()
            .zip(self.prepared_births.par_iter_mut())
            .zip(changed)
            .for_each(|((shard, births), changed)| {
                if !births.is_empty() || changed {
                    shard.publish_completed_births(births, IdentityPolicy::FirstActivationOnly);
                    shard.prepare_prefix(floor);
                }
            });
        Ok(())
    }
}

fn sorted_references(events: &MergeEvents, births: bool) -> Vec<Reference> {
    let mut output = Vec::new();
    for (chunk_index, chunk) in events.chunks.iter().enumerate() {
        for (index, change) in chunk.changes.iter().enumerate() {
            if if births {
                !change.positions.is_empty()
            } else {
                change.removed_weight != 0
            } {
                output.push(Reference {
                    key: if births {
                        change.born_key
                    } else {
                        change.removed_key
                    },
                    chunk: chunk_index,
                    index,
                });
            }
        }
    }
    if output.len() <= radix::MAX_RECORDS {
        radix::sort_by_key(&mut output);
    } else {
        output.sort_by_key(|reference| reference.key);
    }
    output
}
fn key_groups(references: &[Reference]) -> Vec<&[Reference]> {
    let mut groups = Vec::new();
    let mut remaining = references;
    while let Some(first) = remaining.first() {
        let end = remaining.partition_point(|reference| reference.key == first.key);
        let (group, next) = remaining.split_at(end);
        groups.push(group);
        remaining = next;
    }
    groups.sort_unstable_by_key(|group| std::cmp::Reverse(group.len()));
    groups
}
