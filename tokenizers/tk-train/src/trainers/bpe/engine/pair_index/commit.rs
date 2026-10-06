//! Publish complete birth cohorts and retire old priorities after writes join.
use super::*;
impl<'a> PairIndex<'a> {
    /// Route aggregates to their count owners, then encode complete birth cohorts.
    /// Position nodes remain borrowed until every owner has joined.
    #[cfg(test)]
    pub(in super::super) fn commit_merges(
        &mut self,
        events: &super::super::merge::MergeEvents,
        identities: usize,
        execution: &super::super::execution::Execution,
        arena: &'a super::super::storage::AllocationArena,
    ) -> Result<()> {
        self.commit_merges_with_prepared(events, identities, execution, arena, Vec::new())
    }
    pub(in super::super) fn commit_merges_with_prepared(
        &mut self,
        events: &super::super::merge::MergeEvents,
        identities: usize,
        execution: &super::super::execution::Execution,
        arena: &'a super::super::storage::AllocationArena,
        births: Vec<(u64, PairState<'a>)>,
    ) -> Result<()> {
        use super::super::merge::ChangeAction as Action;
        use rayon::prelude::*;
        let policy = self.policy;
        let floor = self.minimum_frequency.max(1);
        let router = ShardRouter::new(self.shards.len());
        let mut routes = events.dispatch_with_router(router);
        debug_assert!(births.is_empty() || policy == IdentityPolicy::FirstActivationOnly);
        // Serial metadata routing moves complete states, with no regrouping or
        // codec operation. Owners publish within the existing commit phase.
        let mut prepared: Vec<Vec<(u64, PairState<'a>)>> =
            (0..self.shards.len()).map(|_| Vec::new()).collect();
        for (key, state) in births {
            prepared[router.owner(key)].push((key, state));
        }

        let candidates = self
            .shards
            .par_iter_mut()
            .zip(prepared.into_par_iter())
            .zip(routes.par_iter_mut())
            // Owners without changes keep their counts and valid priorities.
            // Leave their lazy queue refill to selection and avoid scheduling
            // empty codec/directory work, at every corpus and vocabulary scale.
            .filter(|((_, prepared), route)| !route.changes.is_empty() || !prepared.is_empty())
            .map(
                |((shard, prepared), route)| -> Result<Vec<MergeCandidate<'a>>> {
                    // PERF: Group births by output ID and direction first. Each
                    // bucket then uses one reusable neighbor directory, as merge
                    // preparation does. This avoids a hash table and allocation
                    // per complete pair while keeping the same commit mechanism
                    // for fresh and reusable identities.
                    {
                        // Pure metadata grouping shares the existing owner task.
                        // No arena/directory lease is held and no pool work nests.
                        route.group_births(events);
                    }

                    for reference in &route.changes {
                        let change = &events.chunks[reference.chunk].changes[reference.index()];
                        if matches!(reference.action(), Action::Remove | Action::Both) {
                            if policy == IdentityPolicy::AllowActiveReuse {
                                let count = shard.ledger.entry(change.removed_key).or_default();
                                let amount = i64::try_from(change.removed_weight)
                                    .map_err(|_| "BPE identity-reuse removal exceeds i64")?;
                                *count = (*count as i64)
                                    .checked_sub(amount)
                                    .ok_or("BPE identity-reuse count subtraction exceeds i64")?
                                    as u64;
                            } else {
                                shard.subtract_fresh(
                                    change.removed_key,
                                    change.removed_weight,
                                    floor,
                                )?;
                            }
                        }
                        if policy == IdentityPolicy::AllowActiveReuse
                            && matches!(reference.action(), Action::Birth | Action::Both)
                        {
                            let count = shard.ledger.entry(change.born_key).or_default();
                            let amount = i64::try_from(change.born_weight)
                                .map_err(|_| "BPE identity-reuse birth exceeds i64")?;
                            *count = (*count as i64)
                                .checked_add(amount)
                                .ok_or("BPE identity-reuse count addition exceeds i64")?
                                as u64;
                        }
                    }
                    for (key, state) in prepared {
                        let priority = PairPriority {
                            key,
                            priority_count: state.ledger_count_bits,
                        };
                        debug_assert!(
                            !shard.states.contains_key(&key),
                            "fresh birth has one producer rule"
                        );
                        shard.states.insert(key, state);
                        shard.priorities.push(priority);
                    }
                    if route.births.is_empty() {
                        if policy == IdentityPolicy::FirstActivationOnly {
                            shard.prepare_prefix(floor);
                        }
                        return Ok(Vec::new());
                    }
                    let worker = execution.current_worker();
                    let lease = arena.lease(worker);
                    let mut scratch = execution.encoding(worker);
                    let mut candidates = Vec::new();
                    struct BirthGroup {
                        weight: u64,
                        head: usize,
                        occurrences: usize,
                    }
                    impl Default for BirthGroup {
                        fn default() -> Self {
                            Self {
                                weight: 0,
                                head: usize::MAX,
                                occurrences: 0,
                            }
                        }
                    }
                    struct Fragment<'a> {
                        chunk: &'a super::super::merge::EventChunk,
                        index: usize,
                        next: usize,
                    }
                    execution.with_accumulator::<BirthGroup, _>(identities, 0, |neighbors| {
                        // PERF: One owner-level fragment allocation serves every key
                        // and rule/direction bucket. Per-key vectors would allocate for
                        // each key receiving positions from more than one producer.
                        let mut fragments = Vec::<Fragment<'_>>::new();
                        let mut remaining = route.births.as_slice();
                        while let Some(&first_index) = remaining.first() {
                            let first_ref = &route.changes[first_index];
                            let first = &events.chunks[first_ref.chunk].changes[first_ref.index()];
                            let bucket = first.bucket;
                            let end = remaining.partition_point(|&index| {
                                let reference = &route.changes[index];
                                events.chunks[reference.chunk].changes[reference.index()].bucket
                                    == bucket
                            });
                            let (births, next) = remaining.split_at(end);
                            remaining = next;
                            let left = bucket & 1 == 0;
                            let pair = key_pair(first.born_key);
                            let replacement = if left { pair.1 } else { pair.0 };
                            for &reference_index in births {
                                let reference = &route.changes[reference_index];
                                let chunk = &events.chunks[reference.chunk];
                                let index = reference.index();
                                let change = &chunk.changes[index];
                                let pair = key_pair(change.born_key);
                                let neighbor = if left { pair.0 } else { pair.1 };
                                let group = neighbors.touch(neighbor);
                                group.weight = group
                                    .weight
                                    .checked_add(change.born_weight)
                                    .ok_or("BPE birth frequency exceeds u64")?;
                                group.occurrences = group
                                    .occurrences
                                    .checked_add(change.positions.len())
                                    .ok_or("BPE birth position count exceeds resident bounds")?;
                                fragments.push(Fragment {
                                    chunk,
                                    index,
                                    next: group.head,
                                });
                                group.head = fragments.len() - 1;
                            }
                            for (neighbor, group) in neighbors.drain() {
                                let key = pair_key(if left {
                                    (neighbor, replacement)
                                } else {
                                    (replacement, neighbor)
                                });
                                // PERF: Fresh keys cannot revive. Reduce all producers
                                // and reject low counts before touching the global map;
                                // inserting then deleting them causes avoidable growth
                                // and tombstone churn. Signed ledgers already record
                                // ordered changes and retain every positive birth cohort,
                                // including counts below the selection floor.
                                let count = if policy == IdentityPolicy::FirstActivationOnly {
                                    group.weight
                                } else {
                                    shard.ledger[&key]
                                };
                                if if policy == IdentityPolicy::FirstActivationOnly {
                                    count < floor
                                } else {
                                    (count as i64) <= 0
                                } {
                                    continue;
                                }
                                let mut head = group.head;
                                // Fresh jobs supply spatially disjoint runs. Identity-reuse
                                // AA births may combine interleaved left/right chains;
                                // the common encoder merges those actual overlaps.
                                let fragments = &fragments;
                                let sources = std::iter::from_fn(move || {
                                    if head == usize::MAX {
                                        return None;
                                    }
                                    let fragment = &fragments[head];
                                    head = fragment.next;
                                    Some(fragment)
                                })
                                .map(|fragment| {
                                    (
                                        &fragment.chunk.chains,
                                        fragment.chunk.changes[fragment.index].positions,
                                    )
                                });
                                let positions = if policy == IdentityPolicy::FirstActivationOnly {
                                    // Fresh buckets own disjoint, spatially ordered jobs.
                                    // The count is already complete. Encode their reverse
                                    // traversal without rereading each source's endpoints.

                                    SortedPositions::from_reversed_iter(
                                        group.occurrences,
                                        sources.flat_map(|(owner, chain)| owner.reversed(chain)),
                                        &mut scratch,
                                        &lease,
                                    )?
                                } else {
                                    SortedPositions::from_reversed_chains(
                                        sources,
                                        &mut scratch,
                                        &lease,
                                    )?
                                };
                                debug_assert_eq!(positions.len(), group.occurrences);

                                let priority = PairPriority {
                                    key,
                                    priority_count: count,
                                };
                                if policy == IdentityPolicy::FirstActivationOnly {
                                    shard.states.insert(
                                        key,
                                        PairState {
                                            ledger_count_bits: count,
                                            positions,
                                        },
                                    );
                                    shard.priorities.push(priority);
                                } else {
                                    candidates.push(MergeCandidate {
                                        priority,
                                        positions,
                                    });
                                }
                            }
                            fragments.clear();
                        }

                        if policy == IdentityPolicy::FirstActivationOnly {
                            // PERF: Refill while this owner is already running. A
                            // separate pool phase would schedule the same owners again.
                            shard.prepare_prefix(floor);
                        }
                        Ok(candidates)
                    })
                },
            )
            .collect::<Result<Vec<_>>>()?;

        if let Selection::Cohorts { candidates: queue } = &mut self.selection {
            for births in candidates {
                queue.extend(births);
            }
        }
        Ok(())
    }
}
