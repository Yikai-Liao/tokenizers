//! Residual fresh data tasks borrow event payloads; only joined, complete lists
//! reach the unique state publisher. Reuse retains the original commit engine.
use super::super::{
    execution::Execution,
    merge::{CompletedBirth, MergeEvents},
    storage::{AllocationArena, PositionChain, PositionChains},
};
use super::*;

struct BirthJob<'events> {
    key: u64,
    weight: u64,
    count: usize,
    sources: Vec<(&'events PositionChains, PositionChain)>,
}
#[derive(Default)]
struct Group<'events> {
    weight: u64,
    count: usize,
    sources: Vec<(&'events PositionChains, PositionChain)>,
}

impl<'arena> PairIndex<'arena> {
    pub(super) fn commit_data_births(
        &mut self,
        events: &MergeEvents,
        identities: usize,
        execution: &Execution,
        arena: &'arena AllocationArena,
        floor: u64,
    ) -> Result<()> {
        // Preserve checked, original-order removals and failure cleanup. No
        // worker-local resource remains leased after this phase joins.
        self.shards
            .par_iter_mut()
            .zip(self.routes.par_iter())
            .try_for_each(|(shard, route)| {
                shard.apply_first_activation_counts(route, events, floor)
            })?;
        let references = grouped_buckets(events);
        let mut buckets = Vec::new();
        let mut start = 0;
        while start < references.len() {
            let (chunk, index) = references[start];
            let bucket = events.chunks[chunk].changes[index].bucket;
            let end = start
                + references[start..].partition_point(|&(chunk, index)| {
                    events.chunks[chunk].changes[index].bucket == bucket
                });
            buckets.push(&references[start..end]);
            start = end;
        }
        // Scheduling order is independent of each bucket's preserved producer
        // order. A bucket's neighbor directory is borrowed only sequentially.
        buckets.sort_unstable_by_key(|bucket| std::cmp::Reverse(bucket.len()));
        let groups = buckets
            .into_par_iter()
            .map(|bucket| {
                execution.with_accumulator::<Group<'_>, _>(identities, 0, |neighbors| {
                    let (chunk, index) = bucket[0];
                    let first = &events.chunks[chunk].changes[index];
                    let left = first.bucket & 1 == 0;
                    let pair = key_pair(first.born_key);
                    let replacement = if left { pair.1 } else { pair.0 };
                    for &(chunk, index) in bucket {
                        let chunk = &events.chunks[chunk];
                        let change = &chunk.changes[index];
                        let pair = key_pair(change.born_key);
                        let neighbor = if left { pair.0 } else { pair.1 };
                        let group = neighbors.touch(neighbor);
                        group.weight = group
                            .weight
                            .checked_add(change.born_weight)
                            .ok_or("BPE birth frequency exceeds u64")?;
                        group.count = group
                            .count
                            .checked_add(change.positions.len())
                            .ok_or("BPE birth position count exceeds resident bounds")?;
                        group.sources.push((&chunk.chains, change.positions));
                    }
                    let mut output = Vec::new();
                    for (neighbor, mut group) in neighbors.drain() {
                        if group.weight < floor {
                            continue;
                        }
                        group.sources.reverse();
                        output.push(BirthJob {
                            key: pair_key(if left {
                                (neighbor, replacement)
                            } else {
                                (replacement, neighbor)
                            }),
                            weight: group.weight,
                            count: group.count,
                            sources: group.sources,
                        });
                    }
                    Ok(output)
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut jobs: Vec<_> = groups.into_iter().flatten().collect();
        jobs.sort_unstable_by_key(|job| std::cmp::Reverse(job.count));
        let completed = jobs
            .into_par_iter()
            .map(|job| -> Result<CompletedBirth<'arena>> {
                let positions = if job.count >= 16_384 {
                    SortedPositions::from_cooperative_chains(job.count, &job.sources, arena)?
                } else {
                    let worker = execution.current_worker();
                    let lease = arena.lease(worker);
                    let mut scratch = execution.encoding(worker);
                    SortedPositions::from_reversed_iter(
                        job.count,
                        job.sources
                            .iter()
                            .flat_map(|(owner, chain)| owner.reversed(*chain)),
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
        let router = ShardRouter::new(self.shards.len());
        for birth in completed {
            self.prepared_births[router.owner(birth.key)].push(birth);
        }
        self.shards
            .par_iter_mut()
            .zip(self.prepared_births.par_iter_mut())
            .zip(self.routes.par_iter())
            .for_each(|((shard, births), route)| {
                if !births.is_empty() || !route.changes.is_empty() {
                    shard.publish_completed_births(births, IdentityPolicy::FirstActivationOnly);
                    shard.prepare_prefix(floor);
                }
            });
        Ok(())
    }
}

/// Stable counting scatter over the existing canonical rule/direction domain.
/// O(M+B) metadata only; completed producers have no birth chain in this stream.
fn grouped_buckets(events: &MergeEvents) -> Vec<(usize, usize)> {
    let mut offsets = vec![0_usize; events.buckets];
    for chunk in &events.chunks {
        for change in &chunk.changes {
            if !change.positions.is_empty() {
                offsets[change.bucket as usize] += 1;
            }
        }
    }
    let mut total = 0;
    for offset in &mut offsets {
        let count = *offset;
        *offset = total;
        total += count;
    }
    let mut output = vec![(0, 0); total];
    for (chunk_index, chunk) in events.chunks.iter().enumerate() {
        for (index, change) in chunk.changes.iter().enumerate() {
            if change.positions.is_empty() {
                continue;
            }
            let offset = &mut offsets[change.bucket as usize];
            output[*offset] = (chunk_index, index);
            *offset += 1;
        }
    }
    output
}
