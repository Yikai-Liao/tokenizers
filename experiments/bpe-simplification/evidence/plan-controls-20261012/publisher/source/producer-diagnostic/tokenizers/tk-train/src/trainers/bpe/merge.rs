//! Select compatible rules and complete each batch with joined parallel phases.
use super::{
    BpeTrainer, WORD_SEPARATOR_ID, add,
    corpus::{Corpus, Match},
    index::{Candidate, PairIndex},
    positions::{Positions, ReadTask},
    vocabulary::Vocabulary,
};
use ahash::{AHashMap, AHashSet};
use itertools::Itertools;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::collections::VecDeque;
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
        let prepare_reuse = batch.reuse;
        let prepare_started = std::time::Instant::now();
        let jobs = batch.prepare(corpus, self.max_token_length.unwrap_or(usize::MAX))?;
        BENCH_PREPARE_NS.fetch_add(prepare_started.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        if prepare_reuse { &BENCH_REUSE_BATCHES } else { &BENCH_FRESH_BATCHES }.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let apply_started = std::time::Instant::now();
        let changes: Vec<_> = jobs
            .into_par_iter()
            .map(|job| {
                corpus.apply_merges(&job.starts, job.replacement);
                (job.replacement, job.changes)
            })
            .collect();
        BENCH_APPLY_NS.fetch_add(apply_started.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        let commit_started = std::time::Instant::now();
        if reuse {
            index.commit(changes.into_iter().map(|(_, changes)| changes).collect())?;
        } else {
            index.commit_complete(changes, corpus.id_count())?;
        }
        BENCH_COMMIT_NS.fetch_add(commit_started.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
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

/// One preparation task's disjoint writes and owned neighbor-event payloads.
/// The task returns both together so no corpus mutation precedes preparation join.
struct Job {
    starts: SmallVec<[usize; 2]>,
    replacement: u32,
    changes: Vec<Change<Positions>>,
}

/// Task-local left/right neighbor aggregation for one selected rule.
/// Borrows reusable lookup directories and directly appends compressed positions.
struct Neighbors<'a> {
    rule: &'a Rule,
    directories: &'a mut Directories,
    changes: [Vec<Change<Positions>>; 2],
}

/// Reusable direct-ID lookups from left/right neighbors to task-local group indices.
/// Cached indices are validated against the current task's group keys.
struct Directories {
    bench_site: usize,
    indices: [Vec<u32>; 2],
}

impl Directories {
    fn resize(&mut self, domain: usize) {
        let stats = &DIRECTORY_SITES[self.bench_site];
        stats.resizes.fetch_add(1, Relaxed); stats.max_domain.fetch_max(domain as u64, Relaxed);
        for values in &mut self.indices {
            let (old_length, old_capacity) = (values.len(), values.capacity());
            assert!(domain >= old_length);
            values.resize(domain, u32::MAX);
            stats.filled.fetch_add((domain-old_length) as u64, Relaxed);
            let capacity = values.capacity();
            if capacity != old_capacity {
                if old_capacity == 0 { stats.allocations.fetch_add(1, Relaxed); }
                else { stats.growths.fetch_add(1, Relaxed); stats.old_on_growth.fetch_add(old_length as u64, Relaxed); }
                let delta = (capacity-old_capacity) as u64 * 4;
                let live = stats.live_capacity.fetch_add(delta, Relaxed) + delta;
                stats.peak_capacity.fetch_max(live, Relaxed);
            }
        }
    }
}

type CompleteBirths = (Vec<Vec<Change<Positions>>>, Vec<Candidate>);

/// Finish each fresh producer in original task order before routing its births.
/// Groups hold leaves, including zero-weight leaves; only a complete count may
/// reject a key. Concatenation happens once, and its possible Root is final.
pub(super) fn complete_births(
    changes: Vec<(u32, Vec<Change<Positions>>)>,
    domain: usize,
    floor: u64,
) -> Result<CompleteBirths> {
    struct Birth {
        neighbor: u32,
        count: u64,
        leaves: SmallVec<[Positions; 1]>,
    }
    let producers: Vec<_> = changes
        .into_iter()
        .chunk_by(|(replacement, _)| *replacement)
        .into_iter()
        .map(|(replacement, jobs)| {
            (
                replacement,
                jobs.map(|(_, changes)| changes).collect::<Vec<_>>(),
            )
        })
        .collect();
    let completed = producers
        .into_par_iter()
        .map_init(|| Directories::bench_new(1), |directories, (replacement, jobs)| {
            let bench_started = std::time::Instant::now();
            let bench_cpu = bench_thread_cpu();
            let mut bench_g = 0u64; let mut bench_h = 0u64;
            PRODUCER_INPUT_JOB_BYTES.fetch_add(jobs.capacity() as u64 * std::mem::size_of::<Vec<Change<Positions>>>() as u64 + jobs.iter().map(|v| v.capacity() as u64 * std::mem::size_of::<Change<Positions>>() as u64).sum::<u64>(), Relaxed);
            directories.resize(domain);
            let mut groups: [Vec<Birth>; 2] = Default::default();
            let mut losses = Vec::new();
            for mut change in jobs.into_iter().flatten() {
                bench_g += 1;
                if !change.positions.is_empty() {
                    bench_h += 1;
                    // XX belongs to the right side; each side has its own slots.
                    let left = change.born.0 != replacement;
                    let side = usize::from(!left);
                    let neighbor = if left { change.born.0 } else { change.born.1 };
                    let slot = &mut directories.indices[side][neighbor as usize];
                    if groups[side]
                        .get(*slot as usize)
                        .is_none_or(|birth| birth.neighbor != neighbor)
                    {
                        *slot = u32::try_from(groups[side].len())
                            .expect("birth groups are bounded by real neighbor IDs");
                        groups[side].push(Birth {
                            neighbor,
                            count: 0,
                            leaves: SmallVec::new(),
                        });
                    }
                    let birth = &mut groups[side][*slot as usize];
                    add(&mut birth.count, change.born_weight)?;
                    birth.leaves.push(std::mem::take(&mut change.positions));
                }
                if change.removed_weight != 0 {
                    losses.push(change);
                }
            }
            let bench_k = groups.iter().map(|v| v.len() as u64).sum::<u64>();
            let mut bench_staging = groups.iter().map(|v| v.capacity() as u64 * std::mem::size_of::<Birth>() as u64).sum::<u64>() + losses.capacity() as u64 * std::mem::size_of::<Change<Positions>>() as u64;
            for group in groups.iter().flatten() {
                if group.leaves.spilled() {
                    PRODUCER_SPILLS.fetch_add(1, Relaxed);
                    bench_staging += group.leaves.capacity() as u64 * std::mem::size_of::<Positions>() as u64;
                }
                bench_staging += group.leaves.iter().map(|p| p.bench_storage().1).sum::<u64>();
            }
            let mut births = Vec::new();
            for (side, groups) in groups.into_iter().enumerate() {
                for birth in groups {
                    if birth.count >= floor {
                        let pair = if side == 0 {
                            (birth.neighbor, replacement)
                        } else {
                            (replacement, birth.neighbor)
                        };
                        births.push(Candidate::new(
                            pair,
                            birth.count,
                            Positions::concat(birth.leaves)?,
                        ));
                    }
                }
            }
            PRODUCER_COUNT.fetch_add(1, Relaxed);
            PRODUCER_G.fetch_add(bench_g, Relaxed); PRODUCER_H.fetch_add(bench_h, Relaxed);
            PRODUCER_K.fetch_add(bench_k, Relaxed); PRODUCER_K_LIVE.fetch_add(births.len() as u64, Relaxed);
            PRODUCER_MAX_G.fetch_max(bench_g, Relaxed); PRODUCER_MAX_H.fetch_max(bench_h, Relaxed); PRODUCER_MAX_K.fetch_max(bench_k, Relaxed);
            PRODUCER_MAX_STAGING.fetch_max(bench_staging, Relaxed); PRODUCER_STAGING_SUM.fetch_add(bench_staging, Relaxed);
            let cpu = bench_thread_cpu() - bench_cpu;
            PRODUCER_SUM_CPU_NS.fetch_add(cpu, Relaxed); PRODUCER_MAX_CPU_NS.fetch_max(cpu, Relaxed);
            PRODUCER_MAX_WALL_NS.fetch_max(bench_started.elapsed().as_nanos() as u64, Relaxed);
            Ok((losses, births))
        })
        .collect::<Result<Vec<_>>>()?;
    let (losses, births): (Vec<_>, Vec<_>) = completed.into_iter().unzip();
    Ok((losses, births.into_iter().flatten().collect()))
}

impl<'a> Neighbors<'a> {
    fn new(rule: &'a Rule, directories: &'a mut Directories) -> Self {
        Self {
            rule,
            directories,
            changes: std::array::from_fn(|_| Vec::new()),
        }
    }

    fn group(&mut self, neighbor: u32, left: bool) -> &mut Change<Positions> {
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
                positions: Positions::default(),
                // Both sides can create (id, id); keep them in one birth cohort.
                // Other right births follow left births in the reference update order.
                bucket: usize::from(!left && neighbor != id),
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
            group.positions.push(position as u64)?;
        }
        Ok(())
    }

    fn finish(self) -> Vec<Change<Positions>> {
        // Preserve left-before-right count actions and historical cohort order.
        self.changes
            .into_iter()
            .flatten()
            .filter(|change| change.removed_weight != 0 || !change.positions.is_empty())
            .collect()
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
                CohortPreparation::prepare(rule, corpus, words, limit, directories)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(jobs)
    }
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
    Ordinary(AHashMap<Pair, u32>),
    SelfPair(Vec<u64>),
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
            FreshMatches::Ordinary(
                batch
                    .rules
                    .iter()
                    .map(|rule| (rule.pair(), rule.replacement))
                    .collect(),
            )
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
                    for task in positions.tasks(ordinary_chunk) {
                        tasks.push((rank, Source::Task(task)));
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
        let mut neighbors = Neighbors::new(rule, directories);
        let mut starts = SmallVec::new();
        let matcher = self.corpus.fresh_matcher(rule.pair());
        let mut weights = None;
        for p in positions.prefetched(self.corpus) {
            let Some(matched) = matcher(p, self.corpus.token(p)) else {
                continue;
            };
            let (weight, end) = match weights {
                Some((weight, end)) if p < end => (weight, end),
                _ => self.corpus.weight_region(p),
            };
            weights = Some((weight, end));
            self.record_neighbors(&mut neighbors, matched, weight)?;
            starts.push(matched.start);
        }
        Ok(Job {
            starts,
            replacement: rule.replacement,
            changes: neighbors.finish(),
        })
    }

    // Between adjacent selected matches, the left match owns the shared boundary:
    // it emits the removal and the birth using both final replacement IDs. The
    // right match skips that left edge, including when matches span separate jobs.
    fn record_neighbors(
        &self,
        neighbors: &mut Neighbors<'_>,
        matched: Match,
        weight: u64,
    ) -> Result<()> {
        let p = matched.start;
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
                    before != 0 && selected.contains_key(&(self.corpus.token(before - 1), prior))
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
                FreshMatches::Ordinary(selected) => selected
                    .get(&(
                        next,
                        self.corpus.token(matched.after + self.corpus.id_span(next)),
                    ))
                    .copied(),
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
    starts: SmallVec<[usize; 2]>,
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
        directories: &'task mut Directories,
    ) -> Result<Job> {
        #[cfg(test)]
        super::tests::observe_worker(super::tests::Phase::ReusePrepare);
        directories.resize(corpus.id_count());
        let mut task = Self {
            starts: SmallVec::new(),
            neighbors: Neighbors::new(rule, directories),
            corpus,
            limit,
        };
        for &word in words {
            task.prepare_word(word)?;
        }
        Ok(Job {
            starts: task.starts,
            replacement: rule.replacement,
            changes: task.neighbors.finish(),
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
                starts.push(matched.start);
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

/// Filtered self-pair starts or borrowed, directly positioned reading work.
enum Source<'a> {
    Slice(&'a [u64]),
    Task(ReadTask<'a>),
}

impl Source<'_> {
    // Decode one bounded queue ahead. Prefetch is a nonblocking
    // hint; matching still reads the same joined preparation snapshot in order.
    fn prefetched<'a>(&'a self, corpus: &'a Corpus) -> impl Iterator<Item = usize> + 'a {
        let coordinates = match self {
            Self::Slice(values) => itertools::Either::Left(values.iter().copied()),
            Self::Task(task) => itertools::Either::Right(task.iter()),
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
    use super::super::tests::{spans, tokens};
    use super::super::{
        tests::{counts, initialize, trainer},
        word_counts::WordCountsView,
    };
    use super::*;

    fn select(vocabulary: &mut Vocabulary, corpus: &mut Corpus, index: &mut PairIndex) -> Rule {
        let priority = index.best().unwrap();
        let (replacement, _) = vocabulary.resolve_merge(priority.pair()).unwrap();
        corpus.prepare_identity(priority.pair(), replacement);
        Rule {
            replacement,
            candidate: index.take(priority),
        }
    }

    #[test]
    fn shared_boundary_once() {
        let trainer = trainer();
        let words = counts(&[("abacdbdc", 2), ("ab", 10), ("ac", 9), ("db", 8), ("dc", 7)]);
        let (mut vocabulary, mut corpus, mut index) =
            initialize(&trainer, WordCountsView::from_map(&words), false);
        let mut batch = Batch {
            rules: Vec::new(),
            reuse: false,
        };
        // Shared heads/tails remain compatible; adjacent replacements publish once.
        for pair in [(0, 1), (0, 2), (3, 1), (3, 2)] {
            let rule = select(&mut vocabulary, &mut corpus, &mut index);
            assert_eq!(rule.pair(), pair);
            batch.rules.push(rule);
        }
        let jobs = batch.prepare(&corpus, usize::MAX).unwrap();
        let mut births: Vec<_> = jobs
            .iter()
            .flat_map(|j| &j.changes)
            .filter(|c| c.born_weight != 0)
            .map(|c| (c.born, c.born_weight))
            .collect();
        births.sort_unstable();
        assert_eq!(births, [((4, 5), 2), ((5, 6), 2), ((6, 7), 2)]);
        jobs.par_iter()
            .for_each(|j| corpus.apply_merges(&j.starts, j.replacement));
        let s = WORD_SEPARATOR_ID;
        assert_eq!(
            tokens(&corpus),
            [
                s, 4, 4, s, 5, 5, s, 6, 6, s, 7, 7, s, 4, 4, 5, 5, 6, 6, 7, 7, s
            ]
        );
    }

    #[test]
    fn rejected_start_and_aa_tasks() {
        let trainer = trainer();
        let words = [("aaaaaaaaa".into(), 1)].into_iter().collect();
        let (mut vocabulary, mut corpus, mut index) =
            initialize(&trainer, WordCountsView::from_map(&words), false);
        let batch = Batch {
            rules: vec![select(&mut vocabulary, &mut corpus, &mut index)],
            reuse: false,
        };
        let snapshot = FreshSnapshot::new(&batch, &corpus, usize::MAX);
        let FreshMatches::SelfPair(starts) = &snapshot.matches else {
            panic!("expected AA starts")
        };
        assert_eq!(starts, &[1, 3, 5, 7]);
        // Each task first sees a separator/stale coordinate, then valid AA starts.
        for (input, expected) in [(&[0, 1, 3][..], &[1, 3][..]), (&[0, 5, 7][..], &[5, 7][..])] {
            let job = snapshot
                .prepare_job(&mut Directories::default(), 0, Source::Slice(input))
                .unwrap();
            assert_eq!(job.starts.as_slice(), expected);
        }
    }

    #[test]
    fn reuse_spans() {
        let mut trainer = trainer();
        trainer.end_of_word_suffix = Some("a".into());
        let words = counts(&[("baaaaaba", 2), ("ba", 1)]);
        let (mut vocabulary, mut corpus, mut index) =
            initialize(&trainer, WordCountsView::from_map(&words), true);
        assert!(matches!(
            trainer
                .merge_next(&mut vocabulary, &mut corpus, &mut index)
                .unwrap(),
            MergeOutcome::Committed(_)
        ));
        let rule = select(&mut vocabulary, &mut corpus, &mut index);
        assert_eq!((rule.pair(), rule.replacement), ((1, 2), 3));
        // AA aliases the suffix ID: BA sees original and merged occurrences.
        let batch = Batch {
            rules: vec![rule],
            reuse: true,
        };
        let jobs = batch.prepare(&corpus, usize::MAX).unwrap();
        for job in jobs {
            corpus.apply_merges(&job.starts, job.replacement);
        }
        let s = WORD_SEPARATOR_ID;
        assert_eq!(tokens(&corpus), [s, 3, s, 3, 2, 2, 0, 3, 3, s, 1, 2, s]);
        assert_eq!(spans(&corpus), [0, 3, 0, 3, 2, 2, 1, 2, 2, 0, 1, 1, 0]);
    }

    #[test]
    fn deletion_only_birth_and_directory_recovery() {
        let trainer = trainer();
        let words = [("xab".into(), 1)].into_iter().collect();
        let (mut vocabulary, mut corpus, mut index) =
            initialize(&trainer, WordCountsView::from_map(&words), false);
        let rule = select(&mut vocabulary, &mut corpus, &mut index);
        let mut directories = Directories::default();
        directories.resize(vocabulary.len());
        {
            let mut neighbors = Neighbors::new(&rule, &mut directories);
            neighbors.record(true, 2, 2, 1, u64::MAX, true).unwrap();
            assert!(neighbors.record(true, 2, 2, 1, 1, true).is_err());
        }
        // The old directory index is reused without reset after the failed group.
        let mut neighbors = Neighbors::new(&rule, &mut directories);
        neighbors.record(true, 1, 1, 1, 3, true).unwrap();
        neighbors.record(true, 2, 2, 2, 4, true).unwrap();
        let changes = neighbors.finish();
        assert_eq!(
            changes
                .iter()
                .map(|c| (c.removed, c.born_weight))
                .collect::<Vec<_>>(),
            [((1, 0), 3), ((2, 0), 4)]
        );
        let batch = Batch {
            rules: vec![rule],
            reuse: false,
        };
        let job = batch.prepare(&corpus, 3).unwrap().pop().unwrap();
        let change = &job.changes[0];
        assert_eq!(
            (change.removed, change.removed_weight, change.born_weight),
            ((2, 0), 1, 0)
        );
        assert!(change.positions.is_empty());
        corpus.apply_merges(&job.starts, job.replacement);
        index.commit(vec![job.changes]).unwrap();
        assert!(index.best().is_none());
    }
}

static BENCH_PREPARE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BENCH_FRESH_BATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BENCH_REUSE_BATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(super) fn bench_report_prepare() {
    eprintln!("BPE_PREPARE_SECONDS {:.9}", BENCH_PREPARE_NS.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e9);
    eprintln!("BPE_FRESH_BATCHES {}", BENCH_FRESH_BATCHES.swap(0, std::sync::atomic::Ordering::Relaxed));
    eprintln!("BPE_REUSE_BATCHES {}", BENCH_REUSE_BATCHES.swap(0, std::sync::atomic::Ordering::Relaxed));
}

static BENCH_APPLY_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BENCH_COMMIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(super) fn bench_report_apply_commit() {
    eprintln!("BPE_APPLY_SECONDS {:.9}", BENCH_APPLY_NS.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e9);
    eprintln!("BPE_COMMIT_SECONDS {:.9}", BENCH_COMMIT_NS.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e9);
}


// Diagnostic-only accounting. Site 0 is preparation; site 1 is the producer reducer.
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
struct DirectoryStats {
    states: AtomicU64, resizes: AtomicU64, filled: AtomicU64,
    allocations: AtomicU64, growths: AtomicU64, old_on_growth: AtomicU64,
    live_capacity: AtomicU64, peak_capacity: AtomicU64, max_domain: AtomicU64,
    live_states: AtomicU64, peak_states: AtomicU64,
    threads: thread_local::ThreadLocal<()>,
}
impl DirectoryStats {
    const fn new() -> Self { Self {
        states: AtomicU64::new(0), resizes: AtomicU64::new(0), filled: AtomicU64::new(0),
        allocations: AtomicU64::new(0), growths: AtomicU64::new(0), old_on_growth: AtomicU64::new(0),
        live_capacity: AtomicU64::new(0), peak_capacity: AtomicU64::new(0), max_domain: AtomicU64::new(0),
        live_states: AtomicU64::new(0), peak_states: AtomicU64::new(0), threads: thread_local::ThreadLocal::new(),
    }}
}
static DIRECTORY_SITES: [DirectoryStats; 2] = [DirectoryStats::new(), DirectoryStats::new()];
impl Directories {
    fn bench_new(site: usize) -> Self {
        let stats = &DIRECTORY_SITES[site];
        stats.threads.get_or_default();
        stats.states.fetch_add(1, Relaxed);
        let live = stats.live_states.fetch_add(1, Relaxed) + 1;
        stats.peak_states.fetch_max(live, Relaxed);
        Self { indices: Default::default(), bench_site: site }
    }
}
impl Default for Directories { fn default() -> Self { Self::bench_new(0) } }
impl Drop for Directories {
    fn drop(&mut self) {
        let stats = &DIRECTORY_SITES[self.bench_site];
        stats.live_capacity.fetch_sub(self.indices.iter().map(|v| v.capacity() as u64 * 4).sum(), Relaxed);
        stats.live_states.fetch_sub(1, Relaxed);
    }
}
pub(super) fn bench_report_directories() {
    for (site, stats) in DIRECTORY_SITES.iter().enumerate() {
        eprintln!("BPE_DIRECTORIES {}", serde_json::json!({
            "site": if site == 0 { "prepare" } else { "producer" },
            "states_created": stats.states.load(Relaxed), "threads_observed": stats.threads.iter().count(),
            "resize_calls": stats.resizes.load(Relaxed), "sentinel_elements": stats.filled.load(Relaxed),
            "sentinel_logical_bytes": stats.filled.load(Relaxed) * 4,
            "vector_initial_allocations": stats.allocations.load(Relaxed),
            "vector_capacity_growths": stats.growths.load(Relaxed),
            "old_elements_at_capacity_growth": stats.old_on_growth.load(Relaxed),
            "peak_live_vector_capacity_bytes": stats.peak_capacity.load(Relaxed),
            "live_vector_capacity_bytes_at_report": stats.live_capacity.load(Relaxed),
            "max_domain": stats.max_domain.load(Relaxed),
            "peak_live_states": stats.peak_states.load(Relaxed), "live_states_at_report": stats.live_states.load(Relaxed),
            "diagnostic_header_bytes": std::mem::size_of::<Directories>(), "production_header_bytes": 48,
        }));
    }
}

static PRODUCER_G: AtomicU64 = AtomicU64::new(0);
static PRODUCER_H: AtomicU64 = AtomicU64::new(0);
static PRODUCER_K: AtomicU64 = AtomicU64::new(0);
static PRODUCER_K_LIVE: AtomicU64 = AtomicU64::new(0);
static PRODUCER_COUNT: AtomicU64 = AtomicU64::new(0);
static PRODUCER_MAX_WALL_NS: AtomicU64 = AtomicU64::new(0);
static PRODUCER_SUM_CPU_NS: AtomicU64 = AtomicU64::new(0);
static PRODUCER_MAX_CPU_NS: AtomicU64 = AtomicU64::new(0);
static PRODUCER_MAX_G: AtomicU64 = AtomicU64::new(0);
static PRODUCER_MAX_H: AtomicU64 = AtomicU64::new(0);
static PRODUCER_MAX_K: AtomicU64 = AtomicU64::new(0);
static PRODUCER_SPILLS: AtomicU64 = AtomicU64::new(0);
static PRODUCER_MAX_STAGING: AtomicU64 = AtomicU64::new(0);
static PRODUCER_STAGING_SUM: AtomicU64 = AtomicU64::new(0);
static PRODUCER_INPUT_JOB_BYTES: AtomicU64 = AtomicU64::new(0);
// Linux diagnostic binary only: clock_gettime writes the C timespec layout.
#[repr(C)] struct BenchTimespec { seconds: std::os::raw::c_long, nanos: std::os::raw::c_long }
unsafe extern "C" { fn clock_gettime(clock: std::os::raw::c_int, time: *mut BenchTimespec) -> std::os::raw::c_int; }
pub(super) fn bench_thread_cpu() -> u64 {
    let mut value = BenchTimespec { seconds: 0, nanos: 0 };
    // CLOCK_THREAD_CPUTIME_ID is 3 on this Linux benchmark host; value is writable.
    assert_eq!(unsafe { clock_gettime(3, &mut value) }, 0);
    value.seconds as u64 * 1_000_000_000 + value.nanos as u64
}
pub(super) fn bench_report_producer() {
    eprintln!("BPE_PRODUCER_DIAGNOSTIC {}", serde_json::json!({
        "producers": PRODUCER_COUNT.load(Relaxed), "changes_G": PRODUCER_G.load(Relaxed),
        "birth_fragments_H": PRODUCER_H.load(Relaxed), "complete_keys_K": PRODUCER_K.load(Relaxed),
        "live_keys_K": PRODUCER_K_LIVE.load(Relaxed), "max_G": PRODUCER_MAX_G.load(Relaxed),
        "max_H": PRODUCER_MAX_H.load(Relaxed), "max_K": PRODUCER_MAX_K.load(Relaxed),
        "max_wall_ns": PRODUCER_MAX_WALL_NS.load(Relaxed), "sum_cpu_ns": PRODUCER_SUM_CPU_NS.load(Relaxed),
        "max_cpu_ns": PRODUCER_MAX_CPU_NS.load(Relaxed), "spilled_leaf_groups": PRODUCER_SPILLS.load(Relaxed),
        "max_single_producer_staging_checkpoint_bytes": PRODUCER_MAX_STAGING.load(Relaxed),
        "sum_producer_staging_checkpoints_bytes": PRODUCER_STAGING_SUM.load(Relaxed),
        "sum_input_job_vector_capacity_bytes": PRODUCER_INPUT_JOB_BYTES.load(Relaxed),
    }));
}
