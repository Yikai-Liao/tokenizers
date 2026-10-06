//! Read-only preparation: select a path, build writes and account neighbor events.
mod aa;
mod cohort;
mod ordinary;
use super::super::storage::{
    AllocationArena, IdAccumulator, IdDirectory, PositionBuffer, PositionChain, PositionChains,
    SortedPositions,
};
use super::super::{
    IdentityPolicy, WORD_SEPARATOR_ID, aa_parity,
    corpus::{Corpus, PairMatch, PairMatcher, SlotStorage, WordWeightCursor},
    execution::Execution,
    pair_index::{MergeCandidate, pair_key},
};
use super::{
    CompletedBirth, EventChunk, MergeEvents, MergeRule, PairChanges, PreparedJob, PreparedMerges,
    WritePlan,
};
use ahash::AHashMap;
use tk_encode::{Result, models::bpe::Pair};
/// One job's writes, buffered neighbor events, and already encoded fresh births.
struct PreparedOutput<'arena> {
    job: PreparedJob,
    chunks: Vec<EventChunk>,
    completed_births: Vec<CompletedBirth<'arena>>,
}
#[derive(Default)]
struct NeighborChanges {
    removed: u64,
    born: u64,
    positions: PositionChain,
}
pub(in super::super) struct MergeScratch {
    left: IdAccumulator<NeighborChanges>,
    right: IdAccumulator<NeighborChanges>,
    chains: PositionChains,
    changes: Vec<PairChanges>,
    remaining_nodes: usize,
}
impl MergeScratch {
    pub(in super::super) fn new(token_id_count: usize, directories: [IdDirectory; 2]) -> Self {
        let [left, right] = directories;
        Self {
            left: IdAccumulator::with_directory(token_id_count, left),
            right: IdAccumulator::with_directory(token_id_count, right),
            chains: PositionChains::new(),
            changes: Vec::new(),
            remaining_nodes: PositionChains::MAX_NODES,
        }
    }
    pub(in super::super) fn into_directories(self) -> [IdDirectory; 2] {
        [self.left.into_directory(), self.right.into_directory()]
    }
    // PERF: Rules in one job share their node allocation. Drain only the
    // neighbor directories between rules; handing off nodes here would create
    // a separately growing allocation for every rule and prevent buffer reuse.
    fn flush_rule(&mut self, rule: &MergeRule, rank: usize) {
        self.changes.extend(Self::neighbor_events(
            &mut self.left,
            &mut self.right,
            rule,
            rank,
        ));
    }
    // One direction/bucket rule for buffered events and complete producers.
    fn neighbor_events(
        left: &mut IdAccumulator<NeighborChanges>,
        right: &mut IdAccumulator<NeighborChanges>,
        rule: &MergeRule,
        rank: usize,
    ) -> impl Iterator<Item = PairChanges> {
        let left = left.drain().map(move |(neighbor, change)| PairChanges {
            removed_key: pair_key((neighbor, rule.pair.0)),
            born_key: pair_key((neighbor, rule.replacement)),
            removed_weight: change.removed,
            born_weight: change.born,
            positions: change.positions,

            bucket: (rank * 2) as u32,
        });
        let right = right.drain().map(move |(neighbor, change)| PairChanges {
            removed_key: pair_key((rule.pair.1, neighbor)),
            born_key: pair_key((rule.replacement, neighbor)),
            removed_weight: change.removed,
            born_weight: change.born,
            positions: change.positions,

            bucket: (rank * 2 + usize::from(neighbor != rule.replacement)) as u32,
        });
        left.chain(right)
    }
    /// Drain a complete producer's neighbors, pruning before direct encoding.
    /// The caller must cover the whole rule without a partial node-budget flush.
    /// Removal events survive; completed births are published only through the
    /// returned records, with no second event scan or position re-encoding.
    #[allow(clippy::too_many_arguments)]
    fn flush_rule_with_births<'arena>(
        &mut self,
        rule: &MergeRule,
        rank: usize,
        floor: u64,
        arena: &'arena AllocationArena,
        execution: &Execution,
        births: &mut Vec<CompletedBirth<'arena>>,
    ) -> Result<()> {
        let worker = execution.current_worker();
        let lease = arena.lease(worker);
        let chains = &self.chains;
        let changes = &mut self.changes;
        // These are the original neighbor-directory drains, not a second pass
        // over emitted events. Each birth already has its complete mass/chain.
        let mut emit = |mut event: PairChanges| -> Result<()> {
            if !event.positions.is_empty() && event.born_weight >= floor {
                let occurrences = event.positions.len();
                let positions = SortedPositions::from_reversed_iter_direct(
                    occurrences,
                    chains.reversed(event.positions),
                    &lease,
                )?;

                births.push(CompletedBirth {
                    key: event.born_key,
                    weight: event.born_weight,
                    positions,
                });
            }
            // No birth event is emitted for the completed producer. The count
            // owner's original removal actions retain their keys and weights.
            if event.removed_weight != 0 {
                event.positions = PositionChain::default();
                changes.push(event);
            }
            Ok(())
        };
        Self::neighbor_events(&mut self.left, &mut self.right, rule, rank).try_for_each(&mut emit)
    }
    fn take_chunk(&mut self) -> EventChunk {
        self.remaining_nodes = PositionChains::MAX_NODES;
        EventChunk {
            chains: std::mem::take(&mut self.chains),
            changes: std::mem::take(&mut self.changes),
        }
    }
    fn remove(group: &mut NeighborChanges, weight: u64) -> Result<()> {
        group.removed = group
            .removed
            .checked_add(weight)
            .ok_or("BPE neighbor removal mass exceeds u64")?;
        Ok(())
    }
    // PERF: This update runs for every newborn boundary. Expose its small
    // successful path to the caller so count and chain updates share local
    // values; overflow error construction must not keep it out of line.
    #[inline(always)]
    fn birth(
        group: &mut NeighborChanges,
        chains: &mut PositionChains,
        remaining_nodes: &mut usize,
        position: u64,
        weight: u64,
    ) -> Result<()> {
        group.born = group
            .born
            .checked_add(weight)
            .ok_or("BPE neighbor birth mass exceeds u64")?;
        chains.push(&mut group.positions, position)?;
        *remaining_nodes -= 1;
        Ok(())
    }
    fn left(&mut self, neighbor: u32, position: u64, weight: u64, birth: bool) -> Result<()> {
        let group = self.left.touch(neighbor);
        Self::remove(group, weight)?;
        if birth {
            Self::birth(
                group,
                &mut self.chains,
                &mut self.remaining_nodes,
                position,
                weight,
            )?;
        }
        Ok(())
    }
    fn right(
        &mut self,
        removed: u32,
        born: u32,
        position: u64,
        weight: u64,
        birth: bool,
    ) -> Result<()> {
        let group = self.right.touch(removed);
        Self::remove(group, weight)?;
        if birth {
            // A cohort keeps the same neighbor on removal and birth. Preserve
            // checked removal-before-birth order while sharing its lookup.
            let group = if removed == born {
                group
            } else {
                self.right.touch(born)
            };
            Self::birth(
                group,
                &mut self.chains,
                &mut self.remaining_nodes,
                position,
                weight,
            )?;
        }
        Ok(())
    }
}
// PERF: Prefetch a bounded distance ahead to overlap scattered endpoint loads
// without retaining a fully decoded position list.
const PREFETCH_DISTANCE: usize = 16;
const EMPTY: u64 = u64::MAX;
const MULTIPLE: u64 = u64::MAX - 1;
/// Dense directories recognize selected neighbors without a hash lookup in the
/// common case. Shared heads or tails use the complete-key map.
#[derive(Default)]
pub(in super::super) struct SelectedRuleIndex {
    heads: Vec<u64>,
    tails: Vec<u64>,
    multiple: AHashMap<u64, u32>,
    pairs: Vec<Pair>,
}
impl SelectedRuleIndex {
    fn reset(&mut self, rules: &[MergeRule], token_id_count: usize) {
        for (head, tail) in self.pairs.drain(..) {
            self.heads[head as usize] = EMPTY;
            self.tails[tail as usize] = EMPTY;
        }
        self.heads.resize(token_id_count, EMPTY);
        self.tails.resize(token_id_count, EMPTY);
        self.multiple.clear();
        for rule in rules {
            self.pairs.push(rule.pair);
            let head = &mut self.heads[rule.pair.0 as usize];
            *head = if *head == EMPTY {
                pair_key((rule.pair.1, rule.replacement))
            } else {
                MULTIPLE
            };
            let tail = &mut self.tails[rule.pair.1 as usize];
            *tail = if *tail == EMPTY {
                pair_key((rule.pair.0, rule.replacement))
            } else {
                MULTIPLE
            };
        }
        for rule in rules {
            if self.heads[rule.pair.0 as usize] == MULTIPLE
                || self.tails[rule.pair.1 as usize] == MULTIPLE
            {
                self.multiple.insert(pair_key(rule.pair), rule.replacement);
            }
        }
    }
    fn left_selected<S: SlotStorage>(&self, corpus: &Corpus<S>, before: u64, prior: u32) -> bool {
        let tail = self.tails[prior as usize];
        if tail == EMPTY {
            return false;
        }
        let previous = corpus.token(before - 1);
        if tail == MULTIPLE {
            self.multiple.contains_key(&pair_key((previous, prior)))
        } else {
            (tail >> 32) as u32 == previous
        }
    }
    fn final_next<S: SlotStorage>(&self, corpus: &Corpus<S>, after: u64, next: u32) -> u32 {
        let head = self.heads[next as usize];
        if head == EMPTY {
            return next;
        }
        let following = corpus.token(after + corpus.span_by_id(next));
        if head == MULTIPLE {
            self.multiple
                .get(&pair_key((next, following)))
                .copied()
                .unwrap_or(next)
        } else if (head >> 32) as u32 == following {
            head as u32
        } else {
            next
        }
    }
}
enum SelectedNeighbors<'rules> {
    Rules(&'rules SelectedRuleIndex),
    Adjacent {
        previous: Option<u64>,
        following: Option<u64>,
    },
}
struct RulePreparation<'prep, S: SlotStorage> {
    corpus: &'prep Corpus<S>,
    scratch: &'prep mut MergeScratch,
    rule: &'prep MergeRule,
    rank: usize,
    matcher: PairMatcher<'prep, S>,
    /// Strict admission gate for newborn neighbors, in retained-symbol slots.
    /// Initial candidates and the selected merge itself have no extra length gate.
    birth_span_limit: u64,
    chunks: Vec<EventChunk>,
    positions: PositionBuffer,
    // PERF: Weighted words are contiguous and often share weights. Cache the
    // current interval across sorted occurrence visits instead of searching
    // immutable boundaries for every rewrite. The cursor also handles resets.
    weights: WordWeightCursor<'prep>,
}
impl<'prep, S: SlotStorage> RulePreparation<'prep, S> {
    fn new(
        corpus: &'prep Corpus<S>,
        scratch: &'prep mut MergeScratch,
        rule: &'prep MergeRule,
        rank: usize,
        birth_span_limit: u64,
    ) -> Self {
        Self {
            corpus,
            scratch,
            rule,
            rank,
            matcher: corpus.matcher(rule.pair),
            birth_span_limit,
            chunks: Vec::new(),
            positions: PositionBuffer::default(),
            weights: corpus.weight_cursor(),
        }
    }
    fn room(&mut self) {
        if self.scratch.remaining_nodes < 2 {
            self.scratch.flush_rule(self.rule, self.rank);
            self.chunks.push(self.scratch.take_chunk());
        }
    }
    fn fresh(&mut self, matched: PairMatch, neighbors: SelectedNeighbors<'_>) -> Result<()> {
        self.room();
        let weight = self.weights.weight(matched.left_start);
        let prior = self.corpus.token(matched.left_start - 1);
        if prior != WORD_SEPARATOR_ID {
            let prior_span = self.corpus.span_by_id(prior);
            let before = matched.left_start - prior_span;
            let left_selected = match neighbors {
                SelectedNeighbors::Adjacent { previous, .. } => {
                    previous.is_some_and(|start| start + matched.merged_span == matched.left_start)
                }
                SelectedNeighbors::Rules(selected) => {
                    selected.left_selected(self.corpus, before, prior)
                }
            };
            if !left_selected {
                self.scratch.left(
                    prior,
                    before,
                    weight,
                    prior_span + matched.merged_span < self.birth_span_limit,
                )?;
            }
        }
        let next = self.corpus.token(matched.next_start);
        if next != WORD_SEPARATOR_ID {
            let final_next = match neighbors {
                SelectedNeighbors::Adjacent { following, .. } => {
                    if following == Some(matched.next_start) {
                        self.rule.replacement
                    } else {
                        next
                    }
                }
                SelectedNeighbors::Rules(selected) => {
                    selected.final_next(self.corpus, matched.next_start, next)
                }
            };
            self.scratch.right(
                next,
                final_next,
                matched.left_start,
                weight,
                matched.merged_span + self.corpus.span_by_id(final_next) < self.birth_span_limit,
            )?;
        }
        Ok(())
    }
    fn finish_with_births<'arena>(
        self,
        arena: &'arena AllocationArena,
        execution: &Execution,
        floor: u64,
        births: &mut Vec<CompletedBirth<'arena>>,
    ) -> Result<(WritePlan, Vec<EventChunk>)> {
        debug_assert!(
            self.chunks.is_empty(),
            "allocation node budget excludes partial flush"
        );
        self.scratch
            .flush_rule_with_births(self.rule, self.rank, floor, arena, execution, births)?;
        Ok((
            WritePlan {
                rule: *self.rule,
                positions: self.positions,
            },
            self.chunks,
        ))
    }
    fn finish(self) -> (WritePlan, Vec<EventChunk>) {
        self.scratch.flush_rule(self.rule, self.rank);
        (
            WritePlan {
                rule: *self.rule,
                positions: self.positions,
            },
            self.chunks,
        )
    }
}
#[derive(Clone, Copy, Debug)]
pub(in super::super) struct MergeOptions {
    pub(in super::super) single_producer_fast: bool,
}
impl Default for MergeOptions {
    fn default() -> Self {
        Self {
            single_producer_fast: true,
        }
    }
}
/// Prepare all jobs against one stable corpus and join every reader before return.
/// Rules and candidates must be nonempty and correspond in accepted rank order.
/// Complete fresh producers return encoded births; partial, AA, and reuse jobs
/// return event chains for owner reduction. An error returns no applicable plan.
#[allow(clippy::too_many_arguments)]
pub(in super::super) fn prepare_merges_with_births<'arena, S: SlotStorage>(
    corpus: &Corpus<S>,
    rules: &[MergeRule],
    candidates: &[MergeCandidate<'_>],
    policy: IdentityPolicy,
    token_id_count: usize,
    birth_span_limit: usize,
    execution: &Execution,
    arena: &'arena AllocationArena,
    floor: u64,
    options: MergeOptions,
) -> Result<(PreparedMerges, Vec<CompletedBirth<'arena>>)> {
    let floor = floor.max(1);
    let outputs = if policy == IdentityPolicy::AllowActiveReuse {
        cohort::prepare(
            corpus,
            &rules[0],
            &candidates[0],
            token_id_count,
            birth_span_limit as u64,
            execution,
        )?
        .into_iter()
        .map(|(job, chunks)| PreparedOutput {
            job,
            chunks,
            completed_births: Vec::new(),
        })
        .collect()
    } else if rules[0].pair.0 == rules[0].pair.1 {
        aa::prepare(
            corpus,
            &rules[0],
            &candidates[0],
            token_id_count,
            birth_span_limit as u64,
            execution,
        )?
        .into_iter()
        .map(|(job, chunks)| PreparedOutput {
            job,
            chunks,
            completed_births: Vec::new(),
        })
        .collect()
    } else {
        ordinary::prepare(
            corpus,
            rules,
            candidates,
            token_id_count,
            birth_span_limit,
            execution,
            arena,
            floor,
            options,
        )?
    };
    let mut jobs = Vec::new();
    let mut chunks = Vec::new();
    let mut births = Vec::new();
    for output in outputs {
        births.extend(output.completed_births);
        jobs.push(output.job);
        chunks.extend(output.chunks);
    }
    Ok((
        PreparedMerges {
            jobs,
            events: MergeEvents {
                chunks,
                buckets: rules.len() * 2,
            },
        },
        births,
    ))
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_rule_reuse_clears_shared_endpoints_before_domain_growth() {
        let mut selected = SelectedRuleIndex::default();
        selected.reset(
            &[
                MergeRule {
                    pair: (1, 2),
                    replacement: 4,
                },
                MergeRule {
                    pair: (1, 3),
                    replacement: 5,
                },
            ],
            6,
        );
        assert_eq!(selected.heads[1], MULTIPLE);
        assert_eq!(selected.multiple.len(), 2);
        selected.reset(
            &[MergeRule {
                pair: (3, 6),
                replacement: 7,
            }],
            8,
        );
        assert_eq!(selected.heads[1], EMPTY);
        assert_eq!(selected.tails[2], EMPTY);
        assert_eq!(selected.tails[3], EMPTY);
        assert!(selected.multiple.is_empty());
        assert_eq!(selected.heads[3], pair_key((6, 7)));
        assert_eq!(selected.tails[6], pair_key((3, 7)));
        selected.reset(&[], 8);
        assert_eq!(selected.heads[3], EMPTY);
        assert_eq!(selected.tails[6], EMPTY);
    }
}
