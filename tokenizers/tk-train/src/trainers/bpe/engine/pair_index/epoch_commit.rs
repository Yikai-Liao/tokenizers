//! Checked key-level replacements are computed against one immutable epoch.
//! Bulk publication copies tree paths, never compressed position payloads.
use super::super::{
    execution::Execution,
    merge::{CompletedBirth, MergeEvents},
    storage::{AllocationArena, radix},
};
use super::epoch::{Update, Value};
use super::*;
use std::sync::Arc;

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
struct BirthJob<'records> {
    key: u64,
    weight: u64,
    count: usize,
    references: &'records [Reference],
}
impl<'arena> PairIndex<'arena> {
    pub(super) fn commit_epoch(
        &mut self,
        events: &MergeEvents,
        execution: &Execution,
        arena: &'arena AllocationArena,
        mut prepared: Vec<CompletedBirth<'arena>>,
    ) -> Result<()> {
        let floor = self.minimum_frequency.max(1);
        let epoch = self.epoch.as_ref().expect("fresh immutable epoch");
        let removals = references(events, false);
        let updates = groups(&removals)
            .into_par_iter()
            .map(|group| -> Result<Option<Update<'arena>>> {
                let key = group[0].key;
                let Some(old) = epoch.get(key) else {
                    return Ok(None);
                };
                let mut count = Some(old.count);
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
                Ok(Some(Update {
                    key,
                    value: count.map(|count| Value {
                        count,
                        positions: Arc::clone(&old.positions),
                    }),
                }))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut updates: Vec<_> = updates.into_iter().flatten().collect();
        drop(removals);
        let births = references(events, true);
        let mut jobs: Vec<_> = groups(&births)
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
            .collect();
        jobs.sort_unstable_by_key(|job| std::cmp::Reverse(job.count));
        let complete = jobs
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
        drop(births);
        prepared.extend(complete);
        updates.extend(prepared.into_iter().map(|birth| Update {
            key: birth.key,
            value: Some(Value {
                count: birth.weight,
                positions: Arc::new(birth.positions),
            }),
        }));
        self.epoch
            .as_mut()
            .expect("fresh immutable epoch")
            .publish(&mut updates);
        Ok(())
    }
}
fn references(events: &MergeEvents, births: bool) -> Vec<Reference> {
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
fn groups(references: &[Reference]) -> Vec<&[Reference]> {
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
