//! Prepare neighbor changes from one immutable snapshot, then join disjoint writes.
//! Fresh identities permit independent rules to share a batch. Birth cohorts
//! select one rule and preserve each intermediate boundary produced by HF BPE.
use super::{
    IdentityPolicy, WORD_SEPARATOR_ID, aa_parity,
    corpus::{Corpus, PairMatch, PairMatcher, SlotStorage, WordWeightCursor},
    execution::Execution,
    pair_index::{MergeCandidate, PairState, ShardRouter, pair_key},
};
use ahash::AHashMap;
use rayon::prelude::*;
use tk_collections::{AllocationArena, SortedPositions};
use tk_collections::{IdAccumulator, IdDirectory, PositionBuffer, PositionChain, PositionChains};
use tk_encode::{Result, models::bpe::Pair};

#[derive(Clone, Copy)]
pub(super) struct MergeRule {
    pub(super) pair: Pair,
    pub(super) replacement: u32,
}
#[derive(Default)]
struct NeighborChanges {
    removed: u64,
    born: u64,
    positions: PositionChain,
}
/// One neighbor's removal and birth are committed together. With reusable IDs,
/// the two keys can coincide.
pub(super) struct PairChanges {
    pub(super) removed_key: u64,
    pub(super) born_key: u64,
    pub(super) removed_weight: u64,
    pub(super) born_weight: u64,
    pub(super) positions: PositionChain,

    pub(super) bucket: u32,
}
#[derive(Clone, Copy)]
pub(super) enum ChangeAction {
    Remove,
    Birth,
    Both,
}
/// Two low tag bits share one word with a record index. A resident Vec of
/// 48-byte records bounds its indices well below the available upper bits.
pub(super) struct OwnerChange {
    pub(super) chunk: usize,
    index_and_action: usize,
}
impl OwnerChange {
    fn new(chunk: usize, index: usize, action: ChangeAction) -> Self {
        Self {
            chunk,
            index_and_action: (index << 2) | action as usize,
        }
    }
    pub(super) fn index(&self) -> usize {
        self.index_and_action >> 2
    }
    pub(super) fn action(&self) -> ChangeAction {
        match self.index_and_action & 3 {
            0 => ChangeAction::Remove,
            1 => ChangeAction::Birth,
            2 => ChangeAction::Both,
            _ => unreachable!("record tags are constructed from ChangeAction"),
        }
    }
}
pub(super) struct OwnerRoute {
    pub(super) changes: Vec<OwnerChange>,
    pub(super) births: Vec<usize>,
}
pub(super) struct EventChunk {
    pub(super) chains: PositionChains,
    pub(super) changes: Vec<PairChanges>,
}
pub(super) struct MergeEvents {
    pub(super) buckets: usize,
    pub(super) chunks: Vec<EventChunk>,
}
struct WritePlan {
    rule: MergeRule,
    positions: PositionBuffer,
}
struct PreparedJob {
    writes: Vec<WritePlan>,
    word_region: Option<std::ops::Range<u64>>,
}
pub(super) struct PreparedMerges {
    pub(super) diagnostics: Vec<JobDiagnostics>,
    jobs: Vec<PreparedJob>,
    events: MergeEvents,
}
#[derive(Debug)]
pub(super) struct TaskDiagnostic {
    pub(super) rank: usize,
    pub(super) begin: usize,
    pub(super) end: usize,
    pub(super) full: bool,
    pub(super) matched_positions: usize,
}
#[derive(Debug)]
pub(super) struct FastRankDiagnostic {
    pub(super) rank: usize,
    pub(super) input_positions: usize,
    pub(super) born_records: usize,
    pub(super) encoded_keys: usize,
    pub(super) encoded_positions: usize,
    pub(super) pruned_keys: usize,
}
#[derive(Debug)]
pub(super) struct JobDiagnostics {
    pub(super) worker: usize,
    pub(super) tasks: Vec<TaskDiagnostic>,
    pub(super) chunks: usize,
    pub(super) node_budget_fits: bool,
    pub(super) fast_ranks: Vec<FastRankDiagnostic>,
}
struct MergeScratch {
    left: IdAccumulator<NeighborChanges>,
    right: IdAccumulator<NeighborChanges>,
    pub(super) chains: PositionChains,
    changes: Vec<PairChanges>,
    remaining_nodes: usize,
}
impl MergeScratch {
    fn new(identities: usize, directories: [IdDirectory; 2]) -> Self {
        let [left, right] = directories;
        Self {
            left: IdAccumulator::with_directory(identities, left),
            right: IdAccumulator::with_directory(identities, right),
            chains: PositionChains::new(),
            changes: Vec::new(),
            remaining_nodes: PositionChains::new().remaining_nodes(),
        }
    }
    fn into_directories(self) -> [IdDirectory; 2] {
        [self.left.into_directory(), self.right.into_directory()]
    }
    // PERF: Rules in one job share their node allocation. Drain only the
    // neighbor directories between rules; handing off nodes here would create
    // a separately growing allocation for every rule and prevent buffer reuse.
    fn flush_rule(&mut self, rule: &MergeRule, rank: usize) {
        for (neighbor, change) in self.left.drain() {
            self.changes.push(PairChanges {
                removed_key: pair_key((neighbor, rule.pair.0)),
                born_key: pair_key((neighbor, rule.replacement)),
                removed_weight: change.removed,
                born_weight: change.born,
                positions: change.positions,

                bucket: (rank * 2) as u32,
            });
        }
        for (neighbor, change) in self.right.drain() {
            self.changes.push(PairChanges {
                removed_key: pair_key((rule.pair.1, neighbor)),
                born_key: pair_key((rule.replacement, neighbor)),
                removed_weight: change.removed,
                born_weight: change.born,
                positions: change.positions,

                bucket: (rank * 2 + usize::from(neighbor != rule.replacement)) as u32,
            });
        }
    }
    fn flush_rule_with_births<'a>(
        &mut self,
        rule: &MergeRule,
        rank: usize,
        floor: u64,
        arena: &'a AllocationArena,
        execution: &Execution,
        direct: bool,
        births: &mut Vec<(u64, PairState<'a>)>,
        mut diagnostic: Option<&mut FastRankDiagnostic>,
    ) -> Result<()> {
        let worker = execution.current_worker();
        let lease = arena.lease(worker);
        let mut encoding = (!direct).then(|| execution.encoding(worker));
        let chains = &self.chains;
        let changes = &mut self.changes;
        // These are the original neighbor-directory drains, not a second pass
        // over emitted events. Each birth already has its complete mass/chain.
        let mut emit = |mut event: PairChanges| -> Result<()> {
            if !event.positions.is_empty() {
                if let Some(stats) = diagnostic.as_deref_mut() {
                    stats.born_records += 1;
                }
                if event.born_weight >= floor {
                    let occurrences = event.positions.len();
                    let positions = if let Some(encoding) = encoding.as_deref_mut() {
                        SortedPositions::from_reversed_iter(
                            occurrences,
                            chains.reversed(event.positions),
                            encoding,
                            &lease,
                        )?
                    } else {
                        SortedPositions::from_reversed_iter_direct(
                            occurrences,
                            chains.reversed(event.positions),
                            &lease,
                        )?
                    };
                    if let Some(stats) = diagnostic.as_deref_mut() {
                        stats.encoded_keys += 1;
                        stats.encoded_positions += occurrences;
                    }
                    births.push((
                        event.born_key,
                        PairState {
                            ledger_count_bits: event.born_weight,
                            positions,
                        },
                    ));
                } else if let Some(stats) = diagnostic.as_deref_mut() {
                    stats.pruned_keys += 1;
                }
            }
            // No birth event is emitted for the completed producer. The count
            // owner's original removal actions retain their keys and weights.
            if event.removed_weight != 0 {
                event.positions = PositionChain::default();
                changes.push(event);
            }
            Ok(())
        };
        for (neighbor, change) in self.left.drain() {
            emit(PairChanges {
                removed_key: pair_key((neighbor, rule.pair.0)),
                born_key: pair_key((neighbor, rule.replacement)),
                removed_weight: change.removed,
                born_weight: change.born,
                positions: change.positions,
                bucket: (rank * 2) as u32,
            })?;
        }
        for (neighbor, change) in self.right.drain() {
            emit(PairChanges {
                removed_key: pair_key((rule.pair.1, neighbor)),
                born_key: pair_key((rule.replacement, neighbor)),
                removed_weight: change.removed,
                born_weight: change.born,
                positions: change.positions,
                bucket: (rank * 2 + usize::from(neighbor != rule.replacement)) as u32,
            })?;
        }
        Ok(())
    }
    fn take_chunk(&mut self) -> EventChunk {
        self.remaining_nodes = PositionChains::new().remaining_nodes();
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
    fn left(
        &mut self,
        _replacement: u32,
        neighbor: u32,
        position: u64,
        weight: u64,
        birth: bool,
    ) -> Result<()> {
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
        _replacement: u32,
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
impl MergeEvents {
    /// One directory per owner for the whole batch. Only actual actions and
    /// births occupy entries; producers do not allocate an owner/bucket matrix.
    pub(super) fn route(&self, workers: usize) -> Vec<OwnerRoute> {
        self.route_with_diagnostics(workers, None)
    }
    pub(super) fn route_with_diagnostics(
        &self,
        workers: usize,
        diagnostic_round: Option<&super::single_producer_diagnostics::RoundDiagnostics>,
    ) -> Vec<OwnerRoute> {
        self.route_with_router_diagnostics(ShardRouter::new(workers, false), diagnostic_round)
    }
    pub(super) fn route_with_router_diagnostics(
        &self,
        router: ShardRouter,
        diagnostic_round: Option<&super::single_producer_diagnostics::RoundDiagnostics>,
    ) -> Vec<OwnerRoute> {
        let mut routes = self.dispatch_with_router(router);
        // Count actions retain traversal order. Group only actual births on
        // their owners, keeping histograms out of the serial routing pass.
        // Stable counting distribution retains spatial producer order without
        // comparison sorting or repeated indirect key loads per comparison.
        let dispatched = diagnostic_round.and_then(|round| round.mark());
        let route_window = diagnostic_round.map(|round| round.parallel_span());
        routes
            .par_iter_mut()
            .enumerate()
            .for_each(|(owner, route)| {
                let _route_span = diagnostic_round.map(|round| {
                    round.task(
                        super::single_producer_diagnostics::Stage::RouteBirthGroup,
                        owner,
                        rayon::current_thread_index().expect("routing runs in the training pool"),
                        dispatched,
                    )
                });
                route.group_births(self);
            });
        drop(route_window);
        routes
    }
    /// Route only metadata. Birth grouping can run inside the owner task.
    pub(super) fn dispatch(&self, workers: usize) -> Vec<OwnerRoute> {
        self.dispatch_with_router(ShardRouter::new(workers, false))
    }
    pub(super) fn dispatch_with_router(&self, router: ShardRouter) -> Vec<OwnerRoute> {
        let mut routes: Vec<_> = (0..router.shards())
            .map(|_| OwnerRoute {
                changes: Vec::new(),
                births: Vec::new(),
            })
            .collect();
        for (chunk_index, chunk) in self.chunks.iter().enumerate() {
            for (index, change) in chunk.changes.iter().enumerate() {
                debug_assert!((change.bucket as usize) < self.buckets);
                let removed =
                    (change.removed_weight != 0).then(|| router.owner(change.removed_key));
                // Zero-weight identity-reuse births still own positions. Only an empty
                // chain has no birth action; weight alone cannot decide this.
                let born = (!change.positions.is_empty()).then(|| router.owner(change.born_key));
                match (removed, born) {
                    (Some(removed), Some(born)) if removed == born => {
                        let route = &mut routes[removed];
                        route.births.push(route.changes.len());
                        route.changes.push(OwnerChange::new(
                            chunk_index,
                            index,
                            ChangeAction::Both,
                        ));
                    }
                    (removed, born) => {
                        if let Some(owner) = removed {
                            routes[owner].changes.push(OwnerChange::new(
                                chunk_index,
                                index,
                                ChangeAction::Remove,
                            ));
                        }
                        if let Some(owner) = born {
                            let route = &mut routes[owner];
                            route.births.push(route.changes.len());
                            route.changes.push(OwnerChange::new(
                                chunk_index,
                                index,
                                ChangeAction::Birth,
                            ));
                        }
                    }
                }
            }
        }
        routes
    }
}
impl OwnerRoute {
    /// Fresh removal counts commute. Group original references by rule/direction
    /// only; this order is never used for reusable signed ledger actions.
    pub(super) fn grouped_removals(&self, events: &MergeEvents) -> Vec<usize> {
        let removals: Vec<_> = self
            .changes
            .iter()
            .enumerate()
            .filter_map(|(index, change)| {
                matches!(change.action(), ChangeAction::Remove | ChangeAction::Both)
                    .then_some(index)
            })
            .collect();
        if removals.len() < 2 {
            return removals;
        }
        let bucket_of = |index: usize| {
            let reference = &self.changes[index];
            events.chunks[reference.chunk].changes[reference.index()].bucket as usize
        };
        let mut offsets = vec![0_usize; events.buckets];
        for &index in &removals {
            offsets[bucket_of(index)] += 1;
        }
        let mut total = 0;
        for offset in &mut offsets {
            let count = *offset;
            *offset = total;
            total += count;
        }
        let mut grouped = vec![0; removals.len()];
        for index in removals {
            let offset = &mut offsets[bucket_of(index)];
            grouped[*offset] = index;
            *offset += 1;
        }
        grouped
    }
    pub(super) fn group_births(&mut self, events: &MergeEvents) {
        if self.births.len() < 2 {
            return;
        }
        let bucket_of = |index: usize| {
            let reference = &self.changes[index];
            events.chunks[reference.chunk].changes[reference.index()].bucket as usize
        };
        let mut offsets = vec![0_usize; events.buckets];
        for &index in &self.births {
            offsets[bucket_of(index)] += 1;
        }
        let mut total = 0;
        for offset in &mut offsets {
            let count = *offset;
            *offset = total;
            total += count;
        }
        let mut births = vec![0; self.births.len()];
        for &index in &self.births {
            let offset = &mut offsets[bucket_of(index)];
            births[*offset] = index;
            *offset += 1;
        }
        self.births = births;
    }
}
impl EventChunk {
    #[cfg(test)]
    pub(super) fn test(chains: PositionChains, changes: Vec<PairChanges>, _workers: usize) -> Self {
        Self { chains, changes }
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
pub(super) struct SelectedRules {
    heads: Vec<u64>,
    tails: Vec<u64>,
    multiple: AHashMap<u64, u32>,
    pairs: Vec<Pair>,
}
impl SelectedRules {
    fn reset(&mut self, rules: &[MergeRule], identities: usize) {
        for (head, tail) in self.pairs.drain(..) {
            self.heads[head as usize] = EMPTY;
            self.tails[tail as usize] = EMPTY;
        }
        self.heads.resize(identities, EMPTY);
        self.tails.resize(identities, EMPTY);
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
#[derive(Clone, Copy)]
struct LogicalToken {
    id: u32,
    start: u64,
    span: u64,
}
enum SelectedNeighbors<'a> {
    Rules(&'a SelectedRules),
    Adjacent {
        previous: Option<u64>,
        following: Option<u64>,
    },
}
struct Preparation<'a, S: SlotStorage> {
    corpus: &'a Corpus<S>,
    scratch: &'a mut MergeScratch,
    rule: &'a MergeRule,
    rank: usize,
    matcher: PairMatcher<'a, S>,
    limit: u64,
    chunks: Vec<EventChunk>,
    positions: PositionBuffer,
    // PERF: Weighted words are contiguous and often share weights. Cache the
    // current interval across sorted occurrence visits instead of searching
    // immutable boundaries for every rewrite. The cursor also handles resets.
    weights: WordWeightCursor<'a>,
}
impl<S: SlotStorage> Preparation<'_, S> {
    fn room(&mut self) {
        if self.scratch.remaining_nodes < 2 {
            self.scratch.flush_rule(self.rule, self.rank);
            self.chunks.push(self.scratch.take_chunk());
        }
    }
    fn record_cohort_match(
        &mut self,
        matched: PairMatch,
        previous: Option<LogicalToken>,
    ) -> Result<LogicalToken> {
        self.room();
        let weight = self.weights.weight(matched.left_start);
        if let Some(previous) = previous {
            self.scratch.left(
                self.rule.replacement,
                previous.id,
                previous.start,
                weight,
                previous.span + matched.merged_span < self.limit,
            )?;
        }
        let next = self.corpus.token(matched.next_start);
        if next != WORD_SEPARATOR_ID {
            self.scratch.right(
                self.rule.replacement,
                next,
                next,
                matched.left_start,
                weight,
                matched.merged_span + self.corpus.span(matched.next_start) < self.limit,
            )?;
        }
        self.positions.push(matched.left_start);
        Ok(LogicalToken {
            id: self.rule.replacement,
            start: matched.left_start,
            span: matched.merged_span,
        })
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
                    self.rule.replacement,
                    prior,
                    before,
                    weight,
                    prior_span + matched.merged_span < self.limit,
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
                self.rule.replacement,
                next,
                final_next,
                matched.left_start,
                weight,
                matched.merged_span + self.corpus.span_by_id(final_next) < self.limit,
            )?;
        }
        Ok(())
    }
    fn finish_with_births<'a>(
        self,
        arena: &'a AllocationArena,
        execution: &Execution,
        floor: u64,
        direct: bool,
        births: &mut Vec<(u64, PairState<'a>)>,
        diagnostic: Option<&mut FastRankDiagnostic>,
    ) -> Result<(WritePlan, Vec<EventChunk>)> {
        debug_assert!(
            self.chunks.is_empty(),
            "allocation node budget excludes partial flush"
        );
        self.scratch.flush_rule_with_births(
            self.rule, self.rank, floor, arena, execution, direct, births, diagnostic,
        )?;
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
struct PositionTask {
    rank: usize,
    begin: usize,
    end: usize,
    full: bool,
    fast: bool,
}
impl PositionTask {
    fn new(rank: usize, begin: usize, end: usize, total: usize) -> Self {
        Self {
            rank,
            begin,
            end,
            full: begin == 0 && end == total,
            fast: false,
        }
    }
}
fn job_node_budget_fits(tasks: &[PositionTask], capacity: usize) -> bool {
    tasks
        .iter()
        .try_fold(0usize, |total, task| {
            total.checked_add(task.end - task.begin)
        })
        .and_then(|total| total.checked_mul(2))
        .is_some_and(|nodes| nodes <= capacity)
}
#[derive(Clone, Copy, Debug, Default)]
pub(super) enum PairLayout {
    #[default]
    Grid,
    Tail,
    Whole,
    Pack,
}
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct MergeOptions {
    pub(super) single_producer_fast: bool,
    pub(super) direct_encoding: bool,
    pub(super) group_births_in_commit: bool,
    pub(super) direct_cold_encoding: bool,
    pub(super) layout: PairLayout,
    pub(super) diagnostics: bool,
    pub(super) fast_shard_router: bool,
    pub(super) logical_owners: usize,
    pub(super) removal_entry: bool,
    pub(super) removal_reduce: bool,
    pub(super) removal_statistics: bool,
    pub(super) removal_selective: bool,
    pub(super) batch_limit: usize,
}
impl MergeOptions {
    pub(super) fn from_env() -> Self {
        Self {
            single_producer_fast: std::env::var("TK_SINGLE_PRODUCER_FAST").is_ok_and(|v| v == "1"),
            diagnostics: std::env::var("TK_SINGLE_DIAG").is_ok_and(|v| v == "1"),
            direct_encoding: std::env::var("TK_SINGLE_DIRECT").map_or(true, |v| v != "0"),
            group_births_in_commit: std::env::var("TK_COMMIT_GROUP_FUSION").is_ok_and(|v| v == "1"),
            direct_cold_encoding: std::env::var("TK_COMMIT_DIRECT_COLD").is_ok_and(|v| v == "1"),
            batch_limit: std::env::var("TK_BATCH_LIMIT")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&n| (1..=256).contains(&n))
                .unwrap_or(256),
            removal_selective: std::env::var("TK_REMOVAL_SELECTIVE").is_ok_and(|v| v == "1"),
            removal_entry: std::env::var("TK_REMOVAL_ENTRY").is_ok_and(|v| v == "1"),
            removal_reduce: std::env::var("TK_REMOVAL_REDUCE").is_ok_and(|v| v == "1"),
            removal_statistics: std::env::var("TK_REMOVAL_STATS").is_ok_and(|v| v == "1"),
            fast_shard_router: std::env::var("TK_FAST_SHARD_ROUTER").is_ok_and(|v| v == "1"),
            logical_owners: std::env::var("TK_LOGICAL_OWNERS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&n| n <= 64)
                .unwrap_or(0),
            layout: match std::env::var("TK_PAIR_LAYOUT").as_deref() {
                Ok("tail") => PairLayout::Tail,
                Ok("whole") => PairLayout::Whole,
                Ok("pack") => PairLayout::Pack,
                _ => PairLayout::Grid,
            },
        }
    }
}
fn position_jobs(
    candidates: &[MergeCandidate<'_>],
    workers: usize,
    layout: PairLayout,
    fast_enabled: bool,
) -> Vec<Vec<PositionTask>> {
    let mut jobs = if workers == 1 {
        position_jobs_grid(candidates, workers)
    } else {
        match layout {
            PairLayout::Grid => position_jobs_grid(candidates, workers),
            PairLayout::Tail => position_jobs_tail(candidates, workers),
            PairLayout::Whole => position_jobs_whole(candidates, workers),
            PairLayout::Pack => position_jobs_pack(candidates, workers),
        }
    };
    if fast_enabled {
        let capacity = PositionChains::new().remaining_nodes();
        for job in &mut jobs {
            if job_node_budget_fits(job, capacity) {
                for task in job {
                    task.fast = task.full;
                }
            }
        }
    }
    jobs
}
fn position_jobs_grid(candidates: &[MergeCandidate<'_>], workers: usize) -> Vec<Vec<PositionTask>> {
    let total: usize = candidates
        .iter()
        .map(|candidate| candidate.positions.len())
        .sum();
    let chunk = total.div_ceil(workers).clamp(1, 1 << 26);
    let mut jobs = Vec::<Vec<PositionTask>>::new();
    let mut visited = 0;
    for (rank, candidate) in candidates.iter().enumerate() {
        let mut begin = 0;
        while begin < candidate.positions.len() {
            let job = visited / chunk;
            if job == jobs.len() {
                jobs.push(Vec::new());
            }
            let take = (chunk - visited % chunk).min(candidate.positions.len() - begin);
            jobs[job].push(PositionTask::new(
                rank,
                begin,
                begin + take,
                candidate.positions.len(),
            ));
            visited += take;
            begin += take;
        }
    }
    jobs
}
fn position_jobs_tail(candidates: &[MergeCandidate<'_>], workers: usize) -> Vec<Vec<PositionTask>> {
    let total: usize = candidates
        .iter()
        .map(|candidate| candidate.positions.len())
        .sum();
    let chunk = total.div_ceil(workers).clamp(1, 1 << 26);
    // Keep a very small remainder of one pair in the current job. Its extra
    // positions cost less than another producer, event run, and codec fragment.
    // Bound the deviation from equal position counts by both an absolute cap
    // and a small fraction of the target job size.
    let tail_budget = chunk.div_ceil(128).min(64);
    let mut jobs = Vec::<Vec<PositionTask>>::new();
    let mut filled = chunk;
    for (rank, candidate) in candidates.iter().enumerate() {
        let mut begin = 0;
        while begin < candidate.positions.len() {
            if filled >= chunk {
                jobs.push(Vec::new());
                filled = 0;
            }
            let remaining = candidate.positions.len() - begin;
            let mut take = (chunk - filled).min(remaining);
            if remaining - take <= tail_budget {
                take = remaining;
            }
            jobs.last_mut()
                .expect("the current job exists")
                .push(PositionTask::new(
                    rank,
                    begin,
                    begin + take,
                    candidate.positions.len(),
                ));
            filled += take;
            begin += take;
        }
    }
    jobs
}
fn position_jobs_whole(
    candidates: &[MergeCandidate<'_>],
    workers: usize,
) -> Vec<Vec<PositionTask>> {
    let total: usize = candidates
        .iter()
        .map(|candidate| candidate.positions.len())
        .sum();
    let chunk = total.div_ceil(workers).clamp(1, 1 << 26);
    let tail_budget = chunk.div_ceil(128).min(64);
    // Keep ordinary pairs intact and expose more ready jobs to the pool.
    // Small pairs share a job; large pairs retain the original maximum range.
    let grain = total
        .div_ceil(workers.saturating_mul(4))
        .max(4096)
        .min(chunk);
    let mut jobs = Vec::<Vec<PositionTask>>::new();
    let mut filled = chunk;
    for (rank, candidate) in candidates.iter().enumerate() {
        let count = candidate.positions.len();
        if count == 0 {
            continue;
        }
        if count <= chunk {
            if filled >= grain || filled + count > grain {
                jobs.push(Vec::new());
                filled = 0;
            }
            jobs.last_mut()
                .expect("the current job exists")
                .push(PositionTask::new(rank, 0, count, count));
            filled += count;
            continue;
        }
        // A large candidate gets spatially ordered ranges, without increasing
        // its producer count merely to reach the smaller scheduling grain.
        let mut begin = 0;
        while begin < count {
            let remaining = count - begin;
            let mut take = chunk.min(remaining);
            if remaining - take <= tail_budget {
                take = remaining;
            }
            jobs.push(vec![PositionTask::new(
                rank,
                begin,
                begin + take,
                candidate.positions.len(),
            )]);
            begin += take;
        }
        filled = chunk;
    }
    jobs
}

fn position_jobs_pack(candidates: &[MergeCandidate<'_>], workers: usize) -> Vec<Vec<PositionTask>> {
    let total: usize = candidates
        .iter()
        .map(|candidate| candidate.positions.len())
        .sum();
    let chunk = total.div_ceil(workers).clamp(1, 1 << 26);
    let tail_budget = chunk.div_ceil(128).min(64);
    let mut remaining: Vec<_> = candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| !candidate.positions.is_empty())
        .map(|(rank, candidate)| {
            PositionTask::new(
                rank,
                0,
                candidate.positions.len(),
                candidate.positions.len(),
            )
        })
        .collect();
    let mut jobs = Vec::<Vec<PositionTask>>::new();
    let mut filled = chunk;
    while !remaining.is_empty() {
        if filled >= chunk {
            jobs.push(Vec::new());
            filled = 0;
        }
        let space = chunk - filled;
        // Reorder preparation only. Prefer the largest whole remainder that
        // fits, with a bounded allowance for the existing tiny-tail policy.
        // Original ranks and every pair's increasing coordinate ranges remain.
        let selected = remaining
            .iter()
            .enumerate()
            .filter(|(_, task)| task.end - task.begin <= space + tail_budget)
            .max_by_key(|(_, task)| (task.end - task.begin, std::cmp::Reverse(task.rank)))
            .or_else(|| {
                remaining
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, task)| (task.end - task.begin, std::cmp::Reverse(task.rank)))
            })
            .map(|(index, _)| index)
            .expect("at least one pair remains");
        let task = &mut remaining[selected];
        let count = task.end - task.begin;
        let mut take = space.min(count);
        if count - take <= tail_budget {
            take = count;
        }
        jobs.last_mut()
            .expect("the current job exists")
            .push(PositionTask::new(
                task.rank,
                task.begin,
                task.begin + take,
                candidates[task.rank].positions.len(),
            ));
        task.begin += take;
        filled += take;
        if task.begin == task.end {
            remaining.swap_remove(selected);
        }
    }
    jobs
}

fn preparation<'a, S: SlotStorage>(
    corpus: &'a Corpus<S>,
    scratch: &'a mut MergeScratch,
    rule: &'a MergeRule,
    rank: usize,
    limit: u64,
) -> Preparation<'a, S> {
    Preparation {
        corpus,
        scratch,
        rule,
        rank,
        matcher: corpus.matcher(rule.pair),
        limit,
        chunks: Vec::new(),
        positions: PositionBuffer::new(),
        weights: corpus.weight_cursor(),
    }
}
pub(super) fn prepare_merges<S: SlotStorage>(
    corpus: &Corpus<S>,
    rules: &[MergeRule],
    candidates: &[MergeCandidate<'_>],
    policy: IdentityPolicy,
    identities: usize,
    limit: usize,
    execution: &Execution,
) -> Result<PreparedMerges> {
    prepare_merges_core(
        corpus,
        rules,
        candidates,
        policy,
        identities,
        limit,
        execution,
        None,
        1,
        MergeOptions::default(),
        None,
    )
    .map(|(prepared, _)| prepared)
}
pub(super) fn prepare_merges_with_births<'a, S: SlotStorage>(
    corpus: &Corpus<S>,
    rules: &[MergeRule],
    candidates: &[MergeCandidate<'_>],
    policy: IdentityPolicy,
    identities: usize,
    limit: usize,
    execution: &Execution,
    arena: &'a AllocationArena,
    floor: u64,
    options: MergeOptions,
    diagnostic_round: Option<&super::single_producer_diagnostics::RoundDiagnostics>,
) -> Result<(PreparedMerges, Vec<(u64, PairState<'a>)>)> {
    // Reuse and AA preserve their original preparation and ownership path.
    if policy == IdentityPolicy::Reusable || rules[0].pair.0 == rules[0].pair.1 {
        return prepare_merges(
            corpus, rules, candidates, policy, identities, limit, execution,
        )
        .map(|prepared| (prepared, Vec::new()));
    }
    prepare_merges_core(
        corpus,
        rules,
        candidates,
        policy,
        identities,
        limit,
        execution,
        Some(arena),
        floor.max(1),
        options,
        diagnostic_round,
    )
}
fn prepare_merges_core<'a, S: SlotStorage>(
    corpus: &Corpus<S>,
    rules: &[MergeRule],
    candidates: &[MergeCandidate<'_>],
    policy: IdentityPolicy,
    identities: usize,
    limit: usize,
    execution: &Execution,
    arena: Option<&'a AllocationArena>,
    floor: u64,
    options: MergeOptions,
    diagnostic_round: Option<&super::single_producer_diagnostics::RoundDiagnostics>,
) -> Result<(PreparedMerges, Vec<(u64, PairState<'a>)>)> {
    let outputs = if policy == IdentityPolicy::Reusable {
        prepare_cohort(
            corpus,
            &rules[0],
            &candidates[0],
            identities,
            limit as u64,
            execution,
        )?
        .into_iter()
        .map(|(job, chunks)| (job, chunks, Vec::new(), None))
        .collect()
    } else if rules[0].pair.0 == rules[0].pair.1 {
        prepare_aa(
            corpus,
            &rules[0],
            &candidates[0],
            identities,
            limit as u64,
            execution,
        )?
        .into_iter()
        .map(|(job, chunks)| (job, chunks, Vec::new(), None))
        .collect()
    } else {
        let jobs = position_jobs(
            candidates,
            execution.workers(),
            options.layout,
            options.single_producer_fast && arena.is_some(),
        );
        let mut selected = execution.selected_rules();
        selected.reset(rules, identities);
        let dispatched = diagnostic_round.and_then(|round| round.mark());
        let prep_window = diagnostic_round.map(|round| round.parallel_span());
        let outputs = jobs
            .into_par_iter()
            .enumerate()
            .map(|(job_index, tasks)| -> Result<_> {
                let _prep_span = diagnostic_round.map(|round| {
                    round.task(
                        super::single_producer_diagnostics::Stage::Prep,
                        job_index,
                        execution.current_worker(),
                        dispatched,
                    )
                });
                let mut diagnostic = options.diagnostics.then(|| JobDiagnostics {
                    worker: execution.current_worker(),
                    tasks: tasks
                        .iter()
                        .map(|task| TaskDiagnostic {
                            rank: task.rank,
                            begin: task.begin,
                            end: task.end,
                            matched_positions: 0,
                            full: task.full,
                        })
                        .collect(),
                    chunks: 0,
                    node_budget_fits: job_node_budget_fits(
                        &tasks,
                        PositionChains::new().remaining_nodes(),
                    ),
                    fast_ranks: Vec::new(),
                });
                let mut directories = execution.directories();
                let mut scratch = MergeScratch::new(identities, std::mem::take(&mut *directories));
                let mut outputs = Vec::new();
                let mut births = Vec::new();
                for (task_index, task) in tasks.into_iter().enumerate() {
                    let mut plan = preparation(
                        corpus,
                        &mut scratch,
                        &rules[task.rank],
                        task.rank,
                        limit as u64,
                    );
                    let mut cursor = candidates[task.rank].positions.cursor(task.begin..task.end);
                    // PERF: Interleave decoding and consumption through a small
                    // ring. Each newly decoded position is prefetched one ring
                    // ahead of its endpoint loads, without a full decoded tile.
                    let mut ring = [0; PREFETCH_DISTANCE];
                    let mut active = cursor.decode_into(&mut ring);
                    for &position in &ring[..active] {
                        corpus.prefetch(position);
                    }
                    let mut head = 0;
                    while active != 0 {
                        let position = ring[head];
                        if let Some(next) = cursor.next() {
                            ring[head] = next;
                            corpus.prefetch(next);
                        } else {
                            active -= 1;
                        }
                        head = (head + 1) % PREFETCH_DISTANCE;
                        if let Some(matched) = plan.matcher.get(position) {
                            plan.fresh(matched, SelectedNeighbors::Rules(&selected))?;
                            plan.positions.push(position);
                        }
                    }
                    if let Some(diagnostic) = &mut diagnostic {
                        diagnostic.tasks[task_index].matched_positions = plan.positions.len();
                    }
                    if task.fast {
                        let _codec_span = diagnostic_round.map(|round| {
                            round.task(
                                super::single_producer_diagnostics::Stage::FastCodec,
                                job_index,
                                execution.current_worker(),
                                None,
                            )
                        });
                        let mut rank_diagnostic = options.diagnostics.then(|| FastRankDiagnostic {
                            rank: task.rank,
                            input_positions: task.end - task.begin,
                            born_records: 0,
                            encoded_keys: 0,
                            encoded_positions: 0,
                            pruned_keys: 0,
                        });
                        outputs.push(plan.finish_with_births(
                            arena.expect("fast tasks have an arena"),
                            execution,
                            floor,
                            options.direct_encoding,
                            &mut births,
                            rank_diagnostic.as_mut(),
                        )?);
                        if let (Some(job), Some(rank)) = (&mut diagnostic, rank_diagnostic) {
                            job.fast_ranks.push(rank);
                        }
                    } else {
                        outputs.push(plan.finish());
                    }
                }
                if let Some((_, chunks)) = outputs.last_mut() {
                    chunks.push(scratch.take_chunk());
                }
                *directories = scratch.into_directories();
                drop(directories);
                let (writes, chunks): (Vec<_>, Vec<_>) = outputs.into_iter().unzip();
                let chunks = chunks.into_iter().flatten().collect::<Vec<_>>();
                if let Some(diagnostic) = &mut diagnostic {
                    diagnostic.chunks = chunks.len();
                }
                Ok((
                    PreparedJob {
                        writes,
                        word_region: None,
                    },
                    chunks,
                    births,
                    diagnostic,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        drop(prep_window);
        outputs
    };
    let mut jobs = Vec::new();
    let mut chunks = Vec::new();
    let mut births = Vec::new();
    let mut diagnostics = Vec::new();
    for (write, events, encoded, diagnostic) in outputs {
        if let Some(diagnostic) = diagnostic {
            diagnostics.push(diagnostic);
        }
        births.extend(encoded);
        jobs.push(write);
        chunks.extend(events);
    }
    Ok((
        PreparedMerges {
            diagnostics,
            jobs,
            events: MergeEvents {
                chunks,
                buckets: rules.len() * 2,
            },
        },
        births,
    ))
}
enum CohortSource {
    Words(Vec<usize>),
    Positions(std::ops::Range<usize>),
}
struct CohortTask {
    region: std::ops::Range<u64>,
    source: CohortSource,
}
fn prepare_cohort<S: SlotStorage>(
    corpus: &Corpus<S>,
    rule: &MergeRule,
    candidate: &MergeCandidate<'_>,
    identities: usize,
    limit: u64,
    execution: &Execution,
) -> Result<Vec<(PreparedJob, Vec<EventChunk>)>> {
    let count = candidate.positions.len();
    let chunk = count.div_ceil(execution.workers()).max(1);
    let tasks = if corpus.needs_word_scan() {
        // Ordered cohorts make word IDs nondecreasing. Each range caches its
        // current word, then adjacent dedup joins range boundaries as well.
        let parts: Vec<Vec<usize>> = (0..count)
            .step_by(chunk)
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|begin| {
                let mut words = Vec::new();
                let mut current = None;
                for position in candidate
                    .positions
                    .cursor(begin..(begin + chunk).min(count))
                {
                    let word = match current {
                        Some((word, end)) if position < end => word,
                        _ => corpus.word_containing(position),
                    };
                    current = Some((word, corpus.word_end(word)));
                    if words.last() != Some(&word) {
                        words.push(word);
                    }
                }
                words
            })
            .collect();
        let mut words: Vec<_> = parts.into_iter().flatten().collect();
        words.dedup();
        let chunk = words.len().div_ceil(execution.workers()).max(1);
        words
            .chunks(chunk)
            .enumerate()
            .map(|(index, part)| {
                let start = if index == 0 {
                    0
                } else {
                    corpus.word_start(part[0])
                };
                let end = words
                    .get((index + 1) * chunk)
                    .map_or(corpus.len() as u64, |&word| corpus.word_start(word));
                CohortTask {
                    region: start..end,
                    source: CohortSource::Words(part.to_vec()),
                }
            })
            .collect::<Vec<_>>()
    } else {
        // Before alias reuse or a length gate, only cut boundaries need a word
        // lookup. Move each cut to the first occurrence in that complete word.
        let mut cuts = vec![(0, 0)];
        for desired in (chunk..count).step_by(chunk) {
            let position = candidate
                .positions
                .cursor(desired..desired + 1)
                .next()
                .expect("the cut index belongs to the cohort");
            let pivot = corpus.word_start(corpus.word_containing(position));
            let begin = candidate.positions.lower_bound(pivot);
            if begin > cuts.last().expect("the first cut exists").0 {
                cuts.push((begin, pivot));
            }
        }
        cuts.push((count, corpus.len() as u64));
        cuts.windows(2)
            .map(|cuts| CohortTask {
                region: cuts[0].1..cuts[1].1,
                source: CohortSource::Positions(cuts[0].0..cuts[1].0),
            })
            .collect()
    };
    tasks
        .into_par_iter()
        .map(|task| -> Result<_> {
            let mut directories = execution.directories();
            let mut scratch = MergeScratch::new(identities, std::mem::take(&mut *directories));
            let mut plan = preparation(corpus, &mut scratch, rule, 0, limit);
            match task.source {
                CohortSource::Words(words) => {
                    for word in words {
                        let mut previous = None;
                        let mut position = corpus.word_start(word);
                        while corpus.token(position) != WORD_SEPARATOR_ID {
                            if let Some(matched) = plan.matcher.get(position) {
                                previous = Some(plan.record_cohort_match(matched, previous)?);
                                position = matched.next_start;
                            } else {
                                let span = corpus.span(position);
                                previous = Some(LogicalToken {
                                    id: corpus.token(position),
                                    start: position,
                                    span,
                                });
                                position += span;
                            }
                        }
                    }
                }
                CohortSource::Positions(range) => {
                    let mut previous = None;
                    let mut after = task.region.start;
                    for position in candidate.positions.cursor(range) {
                        if position < after {
                            continue;
                        }
                        if let Some(matched) = plan.matcher.get(position) {
                            if position != after {
                                let id = corpus.token(position - 1);
                                previous = (id != WORD_SEPARATOR_ID).then(|| {
                                    let span = corpus.span(position - 1);
                                    LogicalToken {
                                        id,
                                        start: position - span,
                                        span,
                                    }
                                });
                            }
                            previous = Some(plan.record_cohort_match(matched, previous)?);
                            after = matched.next_start;
                        }
                    }
                }
            }
            let (write, mut chunks) = plan.finish();
            chunks.push(scratch.take_chunk());
            *directories = scratch.into_directories();
            Ok((
                PreparedJob {
                    writes: vec![write],
                    word_region: Some(task.region),
                },
                chunks,
            ))
        })
        .collect()
}
fn prepare_aa<S: SlotStorage>(
    corpus: &Corpus<S>,
    rule: &MergeRule,
    candidate: &MergeCandidate<'_>,
    identities: usize,
    limit: u64,
    execution: &Execution,
) -> Result<Vec<(PreparedJob, Vec<EventChunk>)>> {
    let chunk = candidate
        .positions
        .len()
        .div_ceil(execution.workers())
        .clamp(1, 1 << 26);
    let ranges: Vec<_> = (0..candidate.positions.len())
        .step_by(chunk)
        .map(|begin| begin..(begin + chunk).min(candidate.positions.len()))
        .collect();
    let matcher = corpus.matcher(rule.pair);
    let valid: Vec<_> = ranges
        .into_par_iter()
        .map(|range| {
            let mut positions = PositionBuffer::new();
            for position in candidate.positions.cursor(range) {
                if matcher.get(position).is_some() {
                    positions.push(position);
                }
            }
            positions
        })
        .collect();
    let span = corpus.span_by_id(rule.pair.0);
    let summaries: Vec<_> = valid
        .iter()
        .map(|positions| {
            aa_parity::summarize_by(positions.len(), |index| positions.get(index), span)
        })
        .collect();
    let parities = aa_parity::incoming_parities(&summaries, span);
    let chosen: Vec<_> = valid
        .into_par_iter()
        .zip(parities)
        .map(|(positions, parity)| {
            let mut chosen = PositionBuffer::new();
            aa_parity::for_each_selected(positions.iter(), span, parity, |position| {
                chosen.push(position)
            });
            chosen
        })
        .collect();
    let mut previous = Vec::with_capacity(chosen.len());
    let mut last = None;
    for positions in &chosen {
        previous.push(last);
        if !positions.is_empty() {
            last = Some(positions.get(positions.len() - 1));
        }
    }
    let mut following = vec![None; chosen.len()];
    let mut first = None;
    for (index, positions) in chosen.iter().enumerate().rev() {
        following[index] = first;
        if !positions.is_empty() {
            first = Some(positions.get(0));
        }
    }
    chosen
        .into_par_iter()
        .enumerate()
        .map(|(index, positions)| -> Result<_> {
            let mut directories = execution.directories();
            let mut scratch = MergeScratch::new(identities, std::mem::take(&mut *directories));
            let mut plan = preparation(corpus, &mut scratch, rule, 0, limit);
            for (offset, position) in positions.iter().enumerate() {
                let previous = if offset == 0 {
                    previous[index]
                } else {
                    Some(positions.get(offset - 1))
                };
                let following = if offset + 1 == positions.len() {
                    following[index]
                } else {
                    Some(positions.get(offset + 1))
                };
                let matched = matcher.geometry(position);
                plan.fresh(
                    matched,
                    SelectedNeighbors::Adjacent {
                        previous,
                        following,
                    },
                )?;
            }
            plan.positions = positions;
            let (write, mut chunks) = plan.finish();
            chunks.push(scratch.take_chunk());
            *directories = scratch.into_directories();
            Ok((
                PreparedJob {
                    writes: vec![write],
                    word_region: None,
                },
                chunks,
            ))
        })
        .collect()
}
impl PreparedMerges {
    pub(super) fn apply<S: SlotStorage>(self, corpus: &mut Corpus<S>) -> MergeEvents {
        if corpus.has_occurrence_spans() {
            let regions: Vec<_> = self
                .jobs
                .iter()
                .map(|job| {
                    job.word_region
                        .clone()
                        .expect("occurrence spans require complete whole-word jobs")
                })
                .collect();
            let writers = corpus
                .word_writers(&regions)
                .expect("the occurrence plane is materialized");
            self.jobs
                .par_iter()
                .zip(writers.into_par_iter())
                .for_each(|(job, mut writer)| {
                    for write in &job.writes {
                        write
                            .positions
                            .iter()
                            .for_each(|position| writer.merge(position, write.rule.replacement));
                    }
                });
        } else {
            self.jobs.par_iter().for_each(|job| {
                for write in &job.writes {
                    let matcher = corpus.matcher(write.rule.pair);
                    write.positions.iter().for_each(|position| {
                        // SAFETY: preparation selected disjoint endpoint spans.
                        // apply holds the mutable corpus borrow until pool join;
                        // geometry reads immutable ID spans, never token IDs.
                        unsafe {
                            corpus.write_endpoints(
                                matcher.geometry(position),
                                write.rule.replacement,
                            );
                        }
                    });
                }
            });
        }
        self.events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_rule_reuse_clears_shared_endpoints_before_domain_growth() {
        let mut selected = SelectedRules::default();
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

#[cfg(test)]
mod producer_budget_tests {
    use super::*;
    #[test]
    fn node_budget_includes_partial_tasks_and_accepts_exact_capacity() {
        let tasks = [
            PositionTask::new(0, 0, 4, 4),
            PositionTask::new(1, 3, 8, 12),
        ];
        assert!(tasks[0].full);
        assert!(!tasks[1].full);
        assert!(job_node_budget_fits(&tasks, 18));
        assert!(!job_node_budget_fits(&tasks, 17));
        assert!(!job_node_budget_fits(
            &[PositionTask::new(0, 0, usize::MAX, usize::MAX)],
            usize::MAX
        ));
        assert!(!job_node_budget_fits(
            &[
                PositionTask::new(0, 0, usize::MAX, usize::MAX),
                PositionTask::new(1, 0, 1, 1)
            ],
            usize::MAX
        ));
    }
}
