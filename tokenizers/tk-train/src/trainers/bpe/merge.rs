//! Select compatible rules and complete each batch with joined parallel phases.
use super::{
    BpeTrainer, WORD_SEPARATOR_ID, add,
    corpus::{Corpus, Match},
    index::{Candidate, PairIndex},
    positions::Positions,
    vocabulary::Vocabulary,
};
use ahash::{AHashMap, AHashSet};
use itertools::Itertools;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::collections::VecDeque;
use std::ops::Range;
use tk_encode::{Result, models::bpe::Pair};

/// Complete-round outcome. Only successful index publication returns records.
/// Restart discards the attempt, including any speculative selected identities.
#[derive(Debug)]
pub(super) enum MergeOutcome {
    Finished,
    RestartForReuse,
    Committed(Vec<MergeRecord>),
}

/// Selected-rule scalars returned only after the whole batch commits successfully.
/// No candidate positions survive preparation through these records.
#[derive(Debug)]
pub(super) struct MergeRecord {
    pub(super) pair: Pair,
    #[cfg(test)]
    pub(super) count: u64,
    #[cfg(test)]
    pub(super) replacement: u32,
}

impl BpeTrainer {
    /// Select and commit one complete round. All preparation joins before writes;
    /// all writes join before index owner updates and ordered publication. Errors
    /// discard the attempt rather than rolling back selection or endpoint writes.
    pub(super) fn merge_next(
        &self,
        vocabulary: &mut Vocabulary,
        corpus: &mut Corpus,
        index: &mut PairIndex,
    ) -> Result<MergeOutcome> {
        let reuse = index.reuse();
        let mut batch = Batch {
            rules: Vec::new(),
            reuse,
            floor: self.min_frequency.max(1),
        };
        let cap = if reuse {
            1
        } else {
            256.min(self.vocab_size - vocabulary.len())
        };
        let mut heads = AHashSet::new();
        let mut tails = AHashSet::new();
        // Take a priority prefix without skipping conflicts. Crossed endpoints
        // would consume another rule's input, so only shared heads/tails may batch.
        // Every newborn has an unselected old boundary as a frequency/tie witness;
        // see DESIGN.md's compatible batch proof for why it cannot overtake this prefix.
        while batch.rules.len() < cap {
            let Some(priority) = index.best() else {
                break;
            };
            let pair = priority.pair();
            if !batch.rules.is_empty()
                && (pair.0 == pair.1 || tails.contains(&pair.0) || heads.contains(&pair.1))
            {
                break;
            }
            let (replacement, reserved) = vocabulary.resolve_merge(pair)?;
            if !reuse && reserved && corpus.has_been_activated(replacement) {
                return Ok(MergeOutcome::RestartForReuse);
            }
            // A reserved ID need not follow old IDs lexicographically. A singleton
            // activation needs no argument about its position relative to later rules.
            if reserved && !batch.rules.is_empty() {
                break;
            }
            let candidate = index.take(priority);
            corpus.prepare_identity(pair, replacement);
            batch.rules.push(Rule {
                replacement,
                candidate,
            });
            if reserved || pair.0 == pair.1 {
                break;
            }
            heads.insert(pair.0);
            tails.insert(pair.1);
        }
        if batch.rules.is_empty() {
            return Ok(MergeOutcome::Finished);
        }

        // Selection scratch retires before preparation, as at the former boundary.
        drop(heads);
        drop(tails);

        // Capture only bounded scalars before consuming preparation releases the
        // selected candidates. Preserve selection order, including reuse cohorts.
        let records = batch
            .rules
            .iter()
            .map(|rule| MergeRecord {
                pair: rule.pair(),
                #[cfg(test)]
                count: rule.candidate.priority.count,
                #[cfg(test)]
                replacement: rule.replacement,
            })
            .collect();
        let jobs = batch.prepare(corpus, self.max_token_length.unwrap_or(usize::MAX))?;
        #[cfg(test)]
        tests::assert_write_orders(corpus, &jobs);
        let changes = jobs
            .into_par_iter()
            .map(|job| {
                corpus.apply_merges(&job.starts, job.replacement);
                job.changes
            })
            .collect();
        index.commit(changes)?;
        Ok(MergeOutcome::Committed(records))
    }
}

/// One selected pair and resolved replacement ID, owning its candidate positions.
/// Keeping the candidate binds preparation to the cohort selected by the queue.
struct Rule {
    replacement: u32,
    candidate: Candidate,
}

impl Rule {
    fn pair(&self) -> Pair {
        self.candidate.priority.pair()
    }
}

/// Priority-prefix rules selected for one joined preparation/application round.
/// Fresh rules have compatible endpoints; reuse mode selects a single cohort.
struct Batch {
    rules: Vec<Rule>,
    reuse: bool,
    floor: u64,
}

/// One neighbor-edge removal and replacement birth from preparation.
/// Weights remain separate so commit preserves intermediate signed adjustments;
/// P carries birth positions, or () for separately routed metadata. Bucket
/// identifies the producer group used by owner aggregation.
pub(super) struct Change<P> {
    pub(super) removed: Pair,
    pub(super) born: Pair,
    pub(super) removed_weight: u64,
    pub(super) born_weight: u64,
    pub(super) positions: P,
    pub(super) bucket: usize,
}

/// A birth stream's publication contract: partial aggregation or complete output.
/// Complete fresh producers have already applied the frequency floor and encoded
/// their full list; partial streams need owner aggregation before publication.
pub(super) enum Birth {
    // Local fresh counts await owner aggregation before floor admission.
    // Reuse fragments instead follow the owner's per-action signed ledger.
    Partial(SmallVec<[u64; 2]>),
    // A fresh ordinary producer covers the full candidate and has applied the floor.
    // Retained lists are ready for direct publication without another encoding pass.
    Complete(Positions),
}

impl Birth {
    pub(super) fn is_empty(&self) -> bool {
        match self {
            Self::Partial(values) => values.is_empty(),
            Self::Complete(values) => values.is_empty(),
        }
    }
}

/// One preparation task's disjoint writes and owned neighbor-event payloads.
/// The task returns both together so no corpus mutation precedes preparation join.
struct Job {
    starts: SmallVec<[u64; 2]>,
    replacement: u32,
    changes: Vec<Change<Birth>>,
    #[cfg(test)]
    frozen: Vec<super::corpus::FrozenMatch>,
    #[cfg(test)]
    rule: (Pair, u32),
}

/// Task-local left/right neighbor aggregation for one selected rule.
/// Borrows reusable lookup directories; owns its groups and decides whether
/// they cover a complete producer or must be routed as partial fragments.
struct Neighbors<'a> {
    rule: &'a Rule,
    rank: usize,
    directories: &'a mut Directories,
    changes: [Vec<Change<SmallVec<[u64; 2]>>>; 2],
    complete: bool,
}

/// Reusable direct-ID lookups from left/right neighbors to task-local group indices.
/// Cached indices are validated against the current task's group keys.
#[derive(Default)]
struct Directories {
    indices: [Vec<u32>; 2],
}

impl Directories {
    fn resize(&mut self, domain: usize) {
        for values in &mut self.indices {
            values.resize(domain, u32::MAX);
        }
    }
}

impl<'a> Neighbors<'a> {
    fn new(rule: &'a Rule, rank: usize, directories: &'a mut Directories, complete: bool) -> Self {
        Self {
            rule,
            rank,
            directories,
            changes: std::array::from_fn(|_| Vec::new()),
            complete,
        }
    }

    fn group(&mut self, neighbor: u32, left: bool) -> &mut Change<SmallVec<[u64; 2]>> {
        let side = usize::from(!left);
        let slot = &mut self.directories.indices[side][neighbor as usize];
        if self.changes[side]
            .get(*slot as usize)
            .is_none_or(|change| (if left { change.born.0 } else { change.born.1 }) != neighbor)
        {
            let id = self.rule.replacement;
            let pair = self.rule.pair();
            let index = self.changes[side].len();
            self.changes[side].push(Change {
                removed: if left {
                    (neighbor, pair.0)
                } else {
                    (pair.1, neighbor)
                },
                born: if left { (neighbor, id) } else { (id, neighbor) },
                removed_weight: 0,
                born_weight: 0,
                positions: SmallVec::new(),
                // Both sides can create (id, id); keep them in one birth cohort.
                // Other right births follow left births in the reference update order.
                bucket: 2 * self.rank + usize::from(!left && neighbor != id),
            });
            *slot =
                u32::try_from(index).expect("each neighbor directory is bounded by real token IDs");
        }
        &mut self.changes[side][*slot as usize]
    }

    fn record(
        &mut self,
        left: bool,
        removed: u32,
        born: u32,
        position: usize,
        weight: u64,
        admit: bool,
    ) -> Result<()> {
        let group = self.group(removed, left);
        add(&mut group.removed_weight, weight)?;
        if admit {
            let group = if removed == born {
                group
            } else {
                self.group(born, left)
            };
            add(&mut group.born_weight, weight)?;
            group.positions.push(position as u64);
        }
        Ok(())
    }

    fn finish(self, floor: u64) -> Result<Vec<Change<Birth>>> {
        // Complete ordinary producers prune and encode before owner routing.
        // Partial jobs retain raw positions until their counts have been reduced.
        let mut result = Vec::new();
        // Left changes precede right changes. Reuse count actions and the resulting
        // historical cohorts observe this order even when both sides name one key.
        for mut change in self.changes.into_iter().flatten() {
            if self.complete && change.born_weight < floor {
                change.positions = SmallVec::new();
                change.born_weight = 0;
            }
            if change.removed_weight == 0 && change.positions.is_empty() {
                continue;
            }
            let positions = if self.complete {
                Birth::Complete(Positions::from_sorted(&change.positions)?)
            } else {
                Birth::Partial(change.positions)
            };
            result.push(Change {
                removed: change.removed,
                born: change.born,
                removed_weight: change.removed_weight,
                born_weight: change.born_weight,
                bucket: change.bucket,
                positions,
            });
        }
        Ok(result)
    }
}

impl Batch {
    fn prepare(self, corpus: &Corpus, limit: usize) -> Result<Vec<Job>> {
        if self.reuse {
            return self.prepare_cohort(corpus, limit);
        }

        // 1. Resolve the selected rules against this joined corpus snapshot.
        let snapshot = FreshSnapshot::new(&self, corpus, limit);

        // 2. Split ordered candidate streams into independent preparation jobs.
        let tasks = snapshot.tasks();
        let jobs = tasks
            .into_par_iter()
            .map_init(Directories::default, |directories, (rank, source)| {
                snapshot.prepare_job(directories, rank, source)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(jobs)
    }

    fn prepare_cohort(self, corpus: &Corpus, limit: usize) -> Result<Vec<Job>> {
        // 1. Restrict the historical cohort to its unique word owners.
        let rule = &self.rules[0];
        let words: Vec<_> = rule
            .candidate
            .positions
            .iter()
            .map(|p| corpus.word(corpus.resident(p)))
            .dedup()
            .collect();

        // 2. Prepare separate word chunks; edits within each word remain ordered.
        let chunk = words.len().div_ceil(rayon::current_num_threads()).max(1);
        let jobs = words
            .par_chunks(chunk)
            .map_init(Directories::default, |directories, words| -> Result<_> {
                CohortPreparation::prepare(rule, corpus, words, limit, self.floor, directories)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(jobs)
    }
}

/// Lookup state for a selected head: absent, one right endpoint, or several rules.
/// Shared heads fall back to the selected-pair table without using a token sentinel.
#[derive(Clone, Copy)]
enum Head {
    Empty,
    Unique { right: u32, replacement: u32 },
    Shared,
}

/// Shared read-only lookup for fresh preparation against one joined corpus snapshot.
/// Resolves selected endpoints once; tasks own their directories, writes and
/// neighbor events. Self-pairs also retain nonoverlapping left-to-right starts.
struct FreshSnapshot<'a> {
    batch: &'a Batch,
    corpus: &'a Corpus,

    limit: usize,
    matches: FreshMatches,
}

enum FreshMatches {
    Ordinary(SelectedRules),
    SelfPair(Vec<u64>),
}

struct SelectedRules {
    selected: AHashMap<Pair, u32>,
    heads: Box<[Head]>,
    tails: Box<[bool]>,
}

impl SelectedRules {
    fn new(batch: &Batch, id_count: usize) -> Self {
        // Resolve shared endpoints once for all tasks in an ordinary batch.
        let selected: AHashMap<_, _> = batch
            .rules
            .iter()
            .map(|rule| (rule.pair(), rule.replacement))
            .collect();
        let mut heads = vec![Head::Empty; id_count].into_boxed_slice();
        let mut tails = vec![false; id_count].into_boxed_slice();
        for rule in &batch.rules {
            let slot = &mut heads[rule.pair().0 as usize];
            *slot = match slot {
                Head::Empty => Head::Unique {
                    right: rule.pair().1,
                    replacement: rule.replacement,
                },
                _ => Head::Shared,
            };
            tails[rule.pair().1 as usize] = true;
        }

        Self {
            selected,
            heads,
            tails,
        }
    }

    fn replacement(&self, pair: Pair) -> Option<u32> {
        match self.heads.get(pair.0 as usize) {
            Some(&Head::Unique { right, replacement }) => (right == pair.1).then_some(replacement),
            Some(Head::Shared) => self.selected.get(&pair).copied(),
            _ => None,
        }
    }
}

impl<'a> FreshSnapshot<'a> {
    fn new(batch: &'a Batch, corpus: &'a Corpus, limit: usize) -> Self {
        let pair = batch.rules[0].pair();
        let matches = if pair.0 == pair.1 {
            // Self-pairs overlap: retain left-to-right nonoverlapping starts.
            // They use start membership rather than ordinary endpoint lookup.
            let mut starts = Vec::new();
            let rule = &batch.rules[0];
            let matcher = corpus.fresh_matcher(rule.pair());
            let mut after = 0;
            for coordinate in rule.candidate.positions.iter() {
                let p = corpus.resident(coordinate);
                if p >= after
                    && let Some(matched) = matcher(p, corpus.token(p))
                {
                    starts.push(coordinate);
                    after = matched.after;
                }
            }
            FreshMatches::SelfPair(starts)
        } else {
            FreshMatches::Ordinary(SelectedRules::new(batch, corpus.id_count()))
        };
        Self {
            batch,
            corpus,

            limit,
            matches,
        }
    }

    fn tasks(&self) -> Vec<(usize, Source<'_>)> {
        let mut tasks = Vec::new();
        let total: usize = self
            .batch
            .rules
            .iter()
            .map(|rule| rule.candidate.positions.len())
            .sum();
        let chunk = total.div_ceil(rayon::current_num_threads() * 4).max(4096);
        let ordinary_chunk = total
            .div_ceil(rayon::current_num_threads())
            .clamp(1, 1 << 26);
        for (rank, rule) in self.batch.rules.iter().enumerate() {
            match &self.matches {
                FreshMatches::SelfPair(starts) => {
                    for part in starts.chunks(chunk) {
                        tasks.push((rank, Source::Slice(part)));
                    }
                }
                FreshMatches::Ordinary(_) => {
                    let positions = &rule.candidate.positions;
                    for range in positions.partition_ranges(ordinary_chunk) {
                        tasks.push((rank, Source::Range(range)));
                    }
                }
            }
        }
        tasks
    }

    fn prepare_job(
        &self,
        directories: &mut Directories,
        rank: usize,
        positions: Source<'_>,
    ) -> Result<Job> {
        #[cfg(test)]
        super::tests::observe_worker(super::tests::Phase::FreshPrepare);
        let rule = &self.batch.rules[rank];
        directories.resize(self.corpus.id_count());
        let mut neighbors = Neighbors::new(
            rule,
            rank,
            directories,
            positions.complete(rule.candidate.positions.len()),
        );
        let mut starts = SmallVec::new();
        #[cfg(test)]
        let mut frozen = Vec::new();
        let matcher = self.corpus.fresh_matcher(rule.pair());
        let mut weights = None;
        for p in positions.prefetched(&rule.candidate.positions, self.corpus) {
            let Some(matched) = matcher(p, self.corpus.token(p)) else {
                continue;
            };
            let (weight, end) = match weights {
                Some((weight, end)) if p < end => (weight, end),
                _ => self.corpus.weight_region(p),
            };
            weights = Some((weight, end));
            self.record_neighbors(&mut neighbors, p, matched, weight)?;
            starts.push(matched.start as u64);
            #[cfg(test)]
            frozen.push(self.corpus.freeze_match(matched));
        }
        Ok(Job {
            starts,
            replacement: rule.replacement,
            changes: neighbors.finish(self.batch.floor)?,
            #[cfg(test)]
            frozen,
            #[cfg(test)]
            rule: (rule.pair(), rule.replacement),
        })
    }

    // Between adjacent selected matches, the left match owns the shared boundary:
    // it emits the removal and the birth using both final replacement IDs. The
    // right match skips that left edge, including when matches span separate jobs.
    fn record_neighbors(
        &self,
        neighbors: &mut Neighbors<'_>,
        p: usize,
        matched: Match,
        weight: u64,
    ) -> Result<()> {
        // Left edge: register it only if the preceding match does not own it.
        let prior = self.corpus.token(p - 1);
        if prior != WORD_SEPARATOR_ID {
            let span = self.corpus.id_span(prior);
            let before = p - span;
            let merging = match &self.matches {
                FreshMatches::SelfPair(starts) => {
                    p >= matched.span()
                        && starts.binary_search(&((p - matched.span()) as u64)).is_ok()
                }
                FreshMatches::Ordinary(selected) => {
                    before != 0
                        && selected.tails[prior as usize]
                        && selected
                            .replacement((self.corpus.token(before - 1), prior))
                            .is_some()
                }
            };
            if !merging {
                neighbors.record(
                    true,
                    prior,
                    prior,
                    before,
                    weight,
                    span + matched.span() < self.limit,
                )?;
            }
        }

        // Right edge: include the following match's replacement when selected.
        let next = self.corpus.token(matched.after);
        if next != WORD_SEPARATOR_ID {
            let replacement = match &self.matches {
                FreshMatches::SelfPair(starts) => starts
                    .binary_search(&(matched.after as u64))
                    .is_ok()
                    .then_some(neighbors.rule.replacement),
                FreshMatches::Ordinary(selected)
                    if !matches!(selected.heads[next as usize], Head::Empty) =>
                {
                    selected.replacement((
                        next,
                        self.corpus.token(matched.after + self.corpus.id_span(next)),
                    ))
                }
                FreshMatches::Ordinary(_) => None,
            };
            let born = replacement.unwrap_or(next);
            neighbors.record(
                false,
                next,
                born,
                p,
                weight,
                matched.span() + self.corpus.id_span(born) < self.limit,
            )?;
        }
        Ok(())
    }
}

/// One reuse task's sequential word scans, deferred writes and neighbor events.
/// Borrows the joined corpus and rule; all words in this task share its admission
/// policy and directories. Finishing transfers both outputs without applying them.
struct CohortPreparation<'task> {
    #[cfg(test)]
    frozen: Vec<super::corpus::FrozenMatch>,
    starts: SmallVec<[u64; 2]>,
    neighbors: Neighbors<'task>,
    corpus: &'task Corpus,
    limit: usize,
}

impl<'task> CohortPreparation<'task> {
    fn prepare(
        rule: &'task Rule,
        corpus: &'task Corpus,
        words: &[usize],
        limit: usize,
        floor: u64,
        directories: &'task mut Directories,
    ) -> Result<Job> {
        #[cfg(test)]
        super::tests::observe_worker(super::tests::Phase::ReusePrepare);
        directories.resize(corpus.id_count());
        let mut task = Self {
            #[cfg(test)]
            frozen: Vec::new(),
            starts: SmallVec::new(),
            neighbors: Neighbors::new(rule, 0, directories, false),
            corpus,
            limit,
        };
        for &word in words {
            task.prepare_word(word)?;
        }
        Ok(Job {
            starts: task.starts,
            replacement: rule.replacement,
            changes: task.neighbors.finish(floor)?,
            #[cfg(test)]
            frozen: task.frozen,
            #[cfg(test)]
            rule: (rule.pair(), rule.replacement),
        })
    }

    // Within one word each accepted merge changes the next left neighbor. Keep
    // this sequential while separate word chunks prepare in parallel.
    fn prepare_word(&mut self, word: usize) -> Result<()> {
        let rule = self.neighbors.rule;
        let corpus = self.corpus;
        let limit = self.limit;
        let neighbors = &mut self.neighbors;
        let starts = &mut self.starts;
        let mut previous = None;
        let mut p = corpus.word_start(word);
        let mut positions = rule.candidate.positions.iter_from_value(p as u64);
        while p < corpus.word_end(word) {
            // Full-word scans are required only after alias reuse or a length
            // gate. Otherwise the selected cohort defines the scan domain.
            if !corpus.whole_words() {
                let Some(coordinate) = positions.find(|&start| start >= p as u64) else {
                    break;
                };
                let next = corpus.resident(coordinate);
                if next >= corpus.word_end(word) {
                    break;
                }
                if next != p {
                    let id = corpus.token(next - 1);
                    previous = (id != WORD_SEPARATOR_ID)
                        .then(|| (id, next - corpus.span(next - 1), corpus.span(next - 1)));
                    p = next;
                }
            }
            if let Some(matched) = corpus.matched(p, rule.pair()) {
                let weight = corpus.word_weight(word);
                if let Some((id, start, span)) = previous {
                    neighbors.record(true, id, id, start, weight, span + matched.span() < limit)?;
                }
                let next = corpus.token(matched.after);
                if next != WORD_SEPARATOR_ID {
                    neighbors.record(
                        false,
                        next,
                        next,
                        p,
                        weight,
                        matched.span() + corpus.span(matched.after) < limit,
                    )?;
                }
                previous = Some((rule.replacement, p, matched.span()));
                starts.push(matched.start as u64);
                #[cfg(test)]
                self.frozen.push(corpus.freeze_match(matched));
                p = matched.after;
            } else {
                let span = corpus.span(p);
                previous = Some((corpus.token(p), p, span));
                p += span.max(1);
            }
        }
        Ok(())
    }
}

/// Task input: filtered self-pair starts or element indices in the task's rule.
/// The accompanying rule rank binds each range to its immutable candidate list.
enum Source<'a> {
    Slice(&'a [u64]),
    Range(Range<usize>),
}

impl Source<'_> {
    fn complete(&self, candidate_len: usize) -> bool {
        match self {
            // Storage alignment can make a range exceed its target size and
            // cover the whole candidate. Filtered AA starts remain partial even
            // when one slice contains all starts.
            Self::Range(range) => *range == (0..candidate_len),
            Self::Slice(_) => false,
        }
    }

    // Decode one bounded queue ahead. Prefetch is a nonblocking
    // hint; matching still reads the same joined preparation snapshot in order.
    fn prefetched<'a>(
        &'a self,
        candidate: &'a Positions,
        corpus: &'a Corpus,
    ) -> impl Iterator<Item = usize> + 'a {
        let coordinates = match self {
            Self::Slice(values) => itertools::Either::Left(values.iter().copied()),
            Self::Range(range) => itertools::Either::Right(candidate.iter_range(range.clone())),
        };
        let mut positions = coordinates.map(move |coordinate| corpus.resident(coordinate));
        let mut ahead = VecDeque::with_capacity(16);
        ahead.extend(positions.by_ref().take(16));
        for &p in &ahead {
            corpus.prefetch(p);
        }
        std::iter::from_fn(move || {
            let position = ahead.pop_front()?;
            if let Some(next) = positions.next() {
                corpus.prefetch(next);
                ahead.push_back(next);
            }
            Some(position)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::{WordCountsView, corpus::CorpusPlan};
    use super::*;

    // Freeze the actual accepted preparation results, including rule and job
    // ownership. Replay on independent initial planes before production applies.
    pub(super) fn assert_write_orders(corpus: &Corpus, jobs: &[Job]) {
        let frozen: Vec<_> = jobs
            .iter()
            .flat_map(|job| job.frozen.iter().copied())
            .collect();
        for job in jobs {
            for matched in &job.frozen {
                assert_eq!(
                    (corpus.token(matched.start), corpus.token(matched.right)),
                    job.rule.0,
                    "accepted endpoints belong to the frozen rule"
                );
            }
        }
        let mut intervals = frozen.clone();
        intervals.sort_unstable_by_key(|matched| matched.start);
        assert!(
            intervals
                .windows(2)
                .all(|adjacent| adjacent[0].after <= adjacent[1].start),
            "accepted half-open intervals must be disjoint"
        );
        let expected = corpus.copy_for_writes(&frozen);
        for job in jobs {
            for &matched in &job.frozen {
                expected.apply_frozen(matched, job.rule.1);
            }
        }
        for reverse in [false, true] {
            let actual = corpus.copy_for_writes(&frozen);
            let ordered: Vec<_> = if reverse {
                jobs.iter().rev().collect()
            } else {
                jobs.iter().collect()
            };
            for job in ordered {
                apply_test_job(job, &actual, reverse);
            }
            assert_eq!(
                actual.write_slots(),
                expected.write_slots(),
                "all token and occurrence-span slots, reverse={reverse}"
            );
        }
        let actual = corpus.copy_for_writes(&frozen);
        jobs.par_iter()
            .for_each(|job| apply_test_job(job, &actual, false));
        assert_eq!(
            actual.write_slots(),
            expected.write_slots(),
            "all slots after parallel jobs"
        );
    }

    fn apply_test_job(job: &Job, corpus: &Corpus, reverse: bool) {
        let reversed;
        let starts = if reverse {
            reversed = job.starts.iter().rev().copied().collect::<Vec<_>>();
            &reversed[..]
        } else {
            &job.starts[..]
        };
        assert_eq!(starts.len(), job.frozen.len());
        corpus.apply_merges(starts, job.replacement);
    }

    #[test]
    fn deferred_writes_cover_empty_inline_and_spilled_neighboring_jobs() {
        for workers in [1, 4, 8] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap()
                .install(|| {
                    let trainer = BpeTrainer::builder()
                        .vocab_size(16)
                        .min_frequency(1)
                        .show_progress(false)
                        .build();
                    let words = [("ab".repeat(135).into(), 1)].into_iter().collect();
                    let view = WordCountsView::from_map(&words);
                    let (mut vocabulary, ids) =
                        Vocabulary::initialize(&trainer, view, workers, &mut None, &None).unwrap();
                    let plan = CorpusPlan::build(view, ids, &trainer, false).unwrap();
                    let mut index = PairIndex::build(&plan, 1, workers, false, &None).unwrap();
                    let mut corpus = plan.materialize();
                    let priority = index.best().unwrap();
                    let pair = priority.pair();
                    let (replacement, _) = vocabulary.resolve_merge(pair).unwrap();
                    corpus.prepare_identity(pair, replacement);
                    let batch = Batch {
                        rules: vec![Rule {
                            replacement,
                            candidate: index.take(priority),
                        }],
                        reuse: false,
                        floor: 1,
                    };
                    let starts: Vec<_> = batch.rules[0].candidate.positions.iter().collect();
                    assert_eq!(starts.len(), 135);
                    let snapshot = FreshSnapshot::new(&batch, &corpus, usize::MAX);
                    // The first task coordinate is rejected; initialize shared
                    // geometry from the first accepted start, not from the input.
                    let mut input = vec![0];
                    input.extend_from_slice(&starts);
                    let job = snapshot
                        .prepare_job(&mut Directories::default(), 0, Source::Slice(&input))
                        .unwrap();
                    assert_eq!(job.frozen[0].start as u64, starts[0]);
                    assert_write_orders(&corpus, &[job]);
                    let mut offset = 0;
                    let jobs: Vec<_> = [0, 1, 2, 3, 129]
                        .into_iter()
                        .map(|length| {
                            let job = snapshot
                                .prepare_job(
                                    &mut Directories::default(),
                                    0,
                                    Source::Slice(&starts[offset..offset + length]),
                                )
                                .unwrap();
                            offset += length;
                            assert_eq!(job.frozen.len(), length);
                            assert_eq!(job.starts.spilled(), length > 2);
                            job
                        })
                        .collect();
                    assert_eq!(offset, starts.len());
                    assert_write_orders(&corpus, &[]);
                    assert_write_orders(&corpus, &jobs);
                });
        }
    }

    #[test]
    fn empty_deferred_writes_still_commit_neighbor_deletions() {
        for workers in [1, 4, 8] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap()
                .install(|| {
                    let trainer = BpeTrainer::builder().show_progress(false).build();
                    let words = [("ab".into(), 1)].into_iter().collect();
                    let view = WordCountsView::from_map(&words);
                    let (_, ids) =
                        Vocabulary::initialize(&trainer, view, workers, &mut None, &None).unwrap();
                    let plan = CorpusPlan::build(view, ids, &trainer, false).unwrap();
                    let mut index = PairIndex::build(&plan, 1, workers, false, &None).unwrap();
                    let corpus = plan.materialize();
                    let pair = index.best().unwrap().pair();
                    let job = Job {
                        starts: SmallVec::new(),
                        replacement: 2,
                        changes: vec![Change {
                            removed: pair,
                            born: pair,
                            removed_weight: 1,
                            born_weight: 0,
                            positions: Birth::Partial(SmallVec::new()),
                            bucket: 0,
                        }],
                        frozen: Vec::new(),
                        rule: (pair, 2),
                    };
                    let before = corpus.write_slots();
                    corpus.apply_merges(&job.starts, job.replacement);
                    index.commit(vec![job.changes]).unwrap();
                    assert_eq!(corpus.write_slots(), before);
                    assert!(
                        index.best().is_none(),
                        "deletion must reach the owner even without writes"
                    );
                });
        }
    }

    #[test]
    fn neighbor_directories_recover_from_stale_indices_after_error() {
        let trainer = BpeTrainer::builder().show_progress(false).build();
        let words = [("abcxyz".into(), 1)].into_iter().collect();
        let view = WordCountsView::from_map(&words);
        let (mut vocabulary, ids) =
            Vocabulary::initialize(&trainer, view, 1, &mut None, &None).unwrap();
        let plan = CorpusPlan::build(view, ids, &trainer, false).unwrap();
        let mut index = PairIndex::build(&plan, 1, 1, false, &None).unwrap();
        let priority = index.best().unwrap();
        let pair = priority.pair();
        let (replacement, _) = vocabulary.resolve_merge(pair).unwrap();
        let rule = Rule {
            replacement,
            candidate: index.take(priority),
        };
        let mut directories = Directories::default();
        directories.resize(vocabulary.len());
        {
            let mut neighbors = Neighbors::new(&rule, 0, &mut directories, false);
            neighbors.record(true, 2, 2, 10, u64::MAX, true).unwrap();
            neighbors.record(true, 4, 4, 20, 4, true).unwrap();
            neighbors.record(false, 3, 3, 30, 7, true).unwrap();
            assert!(neighbors.record(true, 2, 2, 40, 1, true).is_err());
        }
        // No reset: old indices now point outside the groups or at different keys.
        let mut neighbors = Neighbors::new(&rule, 1, &mut directories, false);
        for (left, neighbor, position, weight) in [
            (true, 4, 70, 17),
            (true, 3, 50, 11),
            (true, 2, 60, 13),
            (false, 2, 80, 19),
            (false, 3, 90, 23),
            (true, 2, 100, 5),
        ] {
            neighbors
                .record(left, neighbor, neighbor, position, weight, true)
                .unwrap();
        }
        let changes = neighbors.finish(1).unwrap();
        let actual: Vec<_> = changes
            .into_iter()
            .map(|change| {
                let Birth::Partial(positions) = change.positions else {
                    panic!("expected partial neighbor events")
                };
                (
                    change.removed,
                    change.born,
                    change.removed_weight,
                    change.born_weight,
                    change.bucket,
                    positions.to_vec(),
                )
            })
            .collect();
        assert_eq!(
            actual,
            vec![
                ((4, pair.0), (4, replacement), 17, 17, 2, vec![70]),
                ((3, pair.0), (3, replacement), 11, 11, 2, vec![50]),
                ((2, pair.0), (2, replacement), 18, 18, 2, vec![60, 100]),
                ((pair.1, 2), (replacement, 2), 19, 19, 3, vec![80]),
                ((pair.1, 3), (replacement, 3), 23, 23, 3, vec![90]),
            ]
        );
    }

    #[test]
    fn aligned_fragments_keep_complete_and_partial_publication() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        for (repetitions, expected_lengths) in
            [(127, vec![127]), (129, vec![128, 1]), (257, vec![256, 1])]
        {
            let trainer = BpeTrainer::builder()
                .vocab_size(8)
                .min_frequency(repetitions as u64)
                .show_progress(false)
                .build();
            let words = [("abc".repeat(repetitions).into(), 1)]
                .into_iter()
                .collect();
            let view = WordCountsView::from_map(&words);
            pool.install(|| {
                let progress = trainer.setup_progress();
                let (mut vocabulary, ids) =
                    Vocabulary::initialize(&trainer, view, 2, &mut None, &progress).unwrap();
                let plan = CorpusPlan::build(view, ids, &trainer, false).unwrap();
                let mut index =
                    PairIndex::build(&plan, trainer.min_frequency, 2, false, &progress).unwrap();
                let mut corpus = plan.materialize();
                let priority = index.best().unwrap();
                let pair = priority.pair();
                let candidate = index.take(priority);
                let (replacement, _) = vocabulary.resolve_merge(pair).unwrap();
                corpus.prepare_identity(pair, replacement);
                let batch = Batch {
                    rules: vec![Rule {
                        replacement,
                        candidate,
                    }],
                    reuse: false,
                    floor: trainer.min_frequency.max(1),
                };
                assert_eq!(batch.rules.len(), 1);
                assert_eq!(batch.rules[0].candidate.positions.len(), repetitions);
                let snapshot = FreshSnapshot::new(&batch, &corpus, usize::MAX);
                let tasks = snapshot.tasks();
                assert_eq!(tasks.len(), expected_lengths.len());
                for ((rank, source), expected_len) in tasks.into_iter().zip(&expected_lengths) {
                    let Source::Range(range) = &source else {
                        panic!("expected ordinary positions fragment")
                    };
                    assert_eq!(range.len(), *expected_len);
                    let complete = expected_lengths.len() == 1;
                    assert_eq!(source.complete(repetitions), complete);
                    let job = snapshot
                        .prepare_job(&mut Directories::default(), rank, source)
                        .unwrap();
                    assert!(!job.changes.is_empty());
                    assert!(
                        job.changes
                            .iter()
                            .all(
                                |change| matches!(change.positions, Birth::Complete(_)) == complete
                            )
                    );
                }
            });
        }

        // An AA slice remains partial even if all filtered starts fit in one job.
        assert!(!Source::Slice(&[0, 2, 4]).complete(3));
    }
}
