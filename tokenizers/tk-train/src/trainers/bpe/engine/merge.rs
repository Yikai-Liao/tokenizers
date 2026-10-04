//! Prepare neighbor changes from one immutable snapshot, then join disjoint writes.
//! Fresh identities permit independent rules to share a batch. Birth cohorts
//! select one rule and preserve each intermediate boundary produced by HF BPE.
use super::{
    IdentityPolicy, WORD_SEPARATOR_ID, aa_parity,
    corpus::{Corpus, PairMatch, PairMatcher, WordWeightCursor},
    execution::Execution,
    pair_index::{MergeCandidate, pair_key, shard_for},
};
use ahash::AHashMap;
use rayon::prelude::*;
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
    owner: u32,
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
pub(super) struct OwnerChange(usize);
impl OwnerChange {
    fn new(index: usize, action: ChangeAction) -> Self {
        Self((index << 2) | action as usize)
    }
    pub(super) fn index(&self) -> usize {
        self.0 >> 2
    }
    pub(super) fn action(&self) -> ChangeAction {
        match self.0 & 3 {
            0 => ChangeAction::Remove,
            1 => ChangeAction::Birth,
            2 => ChangeAction::Both,
            _ => unreachable!("record tags are constructed from ChangeAction"),
        }
    }
}
pub(super) struct OwnerRoute {
    pub(super) changes: Vec<OwnerChange>,
    pub(super) births: Vec<Vec<usize>>,
}
pub(super) struct EventChunk {
    pub(super) chains: Vec<PositionChains>,
    pub(super) routes: Vec<OwnerRoute>,
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
    jobs: Vec<PreparedJob>,
    events: MergeEvents,
}
struct MergeScratch {
    left: IdAccumulator<NeighborChanges>,
    right: IdAccumulator<NeighborChanges>,
    pub(super) chains: Vec<PositionChains>,
    changes: Vec<PairChanges>,
    remaining_nodes: usize,
    buckets: usize,
}
impl MergeScratch {
    fn new(
        workers: usize,
        identities: usize,
        buckets: usize,
        directories: [IdDirectory; 2],
    ) -> Self {
        let [left, right] = directories;
        Self {
            left: IdAccumulator::with_directory(identities, left),
            right: IdAccumulator::with_directory(identities, right),
            chains: (0..workers).map(|_| PositionChains::new()).collect(),
            changes: Vec::new(),
            remaining_nodes: PositionChains::new().remaining_nodes(),
            buckets,
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
    fn take_chunk(&mut self) -> EventChunk {
        self.remaining_nodes = PositionChains::new().remaining_nodes();
        let empty = (0..self.chains.len())
            .map(|_| PositionChains::new())
            .collect();
        EventChunk::new(
            std::mem::replace(&mut self.chains, empty),
            std::mem::take(&mut self.changes),
            self.buckets,
        )
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
        chains: &mut [PositionChains],
        remaining_nodes: &mut usize,
        key: u64,
        position: u64,
        weight: u64,
    ) -> Result<()> {
        group.born = group
            .born
            .checked_add(weight)
            .ok_or("BPE neighbor birth mass exceeds u64")?;
        if group.positions.is_empty() {
            group.owner = shard_for(key, chains.len()) as u32;
        }
        chains[group.owner as usize].push(&mut group.positions, position)?;
        *remaining_nodes -= 1;
        Ok(())
    }
    fn left(
        &mut self,
        replacement: u32,
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
                pair_key((neighbor, replacement)),
                position,
                weight,
            )?;
        }
        Ok(())
    }
    fn right(
        &mut self,
        replacement: u32,
        removed: u32,
        born: u32,
        position: u64,
        weight: u64,
        birth: bool,
    ) -> Result<()> {
        Self::remove(self.right.touch(removed), weight)?;
        if birth {
            Self::birth(
                self.right.touch(born),
                &mut self.chains,
                &mut self.remaining_nodes,
                pair_key((replacement, born)),
                position,
                weight,
            )?;
        }
        Ok(())
    }
}
impl EventChunk {
    fn new(chains: Vec<PositionChains>, changes: Vec<PairChanges>, buckets: usize) -> Self {
        let mut routes: Vec<_> = (0..chains.len())
            .map(|_| OwnerRoute {
                changes: Vec::new(),
                births: (0..buckets).map(|_| Vec::new()).collect(),
            })
            .collect();
        // Producers route complete aggregates and build rule/direction buckets.
        // Commit owners consume these buckets directly, preserving job order.
        for (index, change) in changes.iter().enumerate() {
            let removed =
                (change.removed_weight != 0).then(|| shard_for(change.removed_key, chains.len()));
            // Zero-weight identity-reuse births still own positions. Only an empty
            // chain has no birth action; weight alone cannot decide this.
            let born =
                (!change.positions.is_empty()).then(|| shard_for(change.born_key, chains.len()));
            match (removed, born) {
                (Some(removed), Some(born)) if removed == born => routes[removed]
                    .changes
                    .push(OwnerChange::new(index, ChangeAction::Both)),
                (removed, born) => {
                    if let Some(owner) = removed {
                        routes[owner]
                            .changes
                            .push(OwnerChange::new(index, ChangeAction::Remove));
                    }
                    if let Some(owner) = born {
                        routes[owner]
                            .changes
                            .push(OwnerChange::new(index, ChangeAction::Birth));
                    }
                }
            }
            if let Some(owner) = born {
                routes[owner].births[change.bucket as usize].push(index);
            }
        }
        Self {
            chains,
            routes,
            changes,
        }
    }
    #[cfg(test)]
    pub(super) fn test(chains: PositionChains, changes: Vec<PairChanges>, workers: usize) -> Self {
        let owner = changes
            .iter()
            .find(|change| !change.positions.is_empty())
            .map_or(0, |change| shard_for(change.born_key, workers));
        let mut owners: Vec<_> = (0..workers).map(|_| PositionChains::new()).collect();
        owners[owner] = chains;
        Self::new(owners, changes, 2)
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
    fn left_selected(&self, corpus: &Corpus, before: u64, prior: u32) -> bool {
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
    fn final_next(&self, corpus: &Corpus, after: u64, next: u32) -> u32 {
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
struct Preparation<'a> {
    corpus: &'a Corpus,
    scratch: &'a mut MergeScratch,
    rule: &'a MergeRule,
    rank: usize,
    matcher: PairMatcher<'a>,
    limit: u64,
    chunks: Vec<EventChunk>,
    positions: PositionBuffer,
    // PERF: Weighted words are contiguous and often share weights. Cache the
    // current interval across sorted occurrence visits instead of searching
    // immutable boundaries for every rewrite. The cursor also handles resets.
    weights: WordWeightCursor<'a>,
}
impl Preparation<'_> {
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
}
fn position_jobs(candidates: &[MergeCandidate<'_>], workers: usize) -> Vec<Vec<PositionTask>> {
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
            jobs[job].push(PositionTask {
                rank,
                begin,
                end: begin + take,
            });
            visited += take;
            begin += take;
        }
    }
    jobs
}
fn preparation<'a>(
    corpus: &'a Corpus,
    scratch: &'a mut MergeScratch,
    rule: &'a MergeRule,
    rank: usize,
    limit: u64,
) -> Preparation<'a> {
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
pub(super) fn prepare_merges(
    corpus: &Corpus,
    rules: &[MergeRule],
    candidates: &[MergeCandidate<'_>],
    policy: IdentityPolicy,
    identities: usize,
    limit: usize,
    execution: &Execution,
) -> Result<PreparedMerges> {
    let outputs = if policy == IdentityPolicy::Reusable {
        prepare_cohort(
            corpus,
            &rules[0],
            &candidates[0],
            identities,
            limit as u64,
            execution,
        )?
    } else if rules[0].pair.0 == rules[0].pair.1 {
        prepare_aa(
            corpus,
            &rules[0],
            &candidates[0],
            identities,
            limit as u64,
            execution,
        )?
    } else {
        let jobs = position_jobs(candidates, execution.workers());
        let mut selected = execution.selected_rules();
        selected.reset(rules, identities);
        jobs.into_par_iter()
            .map(|tasks| -> Result<_> {
                let mut directories = execution.directories();
                let mut scratch = MergeScratch::new(
                    execution.workers(),
                    identities,
                    rules.len() * 2,
                    std::mem::take(&mut *directories),
                );
                let mut outputs = Vec::new();
                for task in tasks {
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
                    outputs.push(plan.finish());
                }
                if let Some((_, chunks)) = outputs.last_mut() {
                    chunks.push(scratch.take_chunk());
                }
                *directories = scratch.into_directories();
                let (writes, chunks): (Vec<_>, Vec<_>) = outputs.into_iter().unzip();
                Ok((
                    PreparedJob {
                        writes,
                        word_region: None,
                    },
                    chunks.into_iter().flatten().collect::<Vec<_>>(),
                ))
            })
            .collect::<Result<Vec<_>>>()?
    };
    let mut jobs = Vec::new();
    let mut chunks = Vec::new();
    for (write, events) in outputs {
        jobs.push(write);
        chunks.extend(events);
    }
    Ok(PreparedMerges {
        jobs,
        events: MergeEvents {
            chunks,
            buckets: rules.len() * 2,
        },
    })
}
enum CohortSource {
    Words(Vec<usize>),
    Positions(std::ops::Range<usize>),
}
struct CohortTask {
    region: std::ops::Range<u64>,
    source: CohortSource,
}
fn prepare_cohort(
    corpus: &Corpus,
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
            let mut scratch = MergeScratch::new(
                execution.workers(),
                identities,
                2,
                std::mem::take(&mut *directories),
            );
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
fn prepare_aa(
    corpus: &Corpus,
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
            let mut scratch = MergeScratch::new(
                execution.workers(),
                identities,
                2,
                std::mem::take(&mut *directories),
            );
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
    pub(super) fn apply(self, corpus: &mut Corpus) -> MergeEvents {
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
                        corpus.write_endpoints(matcher.geometry(position), write.rule.replacement);
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
