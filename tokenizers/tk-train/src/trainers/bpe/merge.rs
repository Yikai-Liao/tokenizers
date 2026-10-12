//! Select compatible rules and complete each batch with joined parallel phases.
use super::{
    BpeTrainer, add,
    corpus::{Corpus, FreshMatch},
    index::{Candidate, PairIndex},
    positions::Positions,
    vocabulary::Vocabulary,
};
use ahash::{AHashMap, AHashSet};
use rayon::prelude::*;
use smallvec::SmallVec;
use tk_encode::{Result, models::bpe::Pair};

/// Complete-round outcome. Only successful index publication commits history.
/// Restart discards the attempt, including any speculative selected identities.
#[derive(Debug)]
pub(super) enum MergeOutcome {
    Finished,
    RestartForReuse,
    Committed(usize),
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
        let records: Vec<_> = batch.rules.iter().map(Rule::pair).collect();
        let accepted = records.len();
        #[cfg(test)]
        let trace: super::Trace = batch
            .rules
            .iter()
            .map(|rule| {
                (
                    rule.pair(),
                    rule.candidate.priority.count(),
                    rule.replacement,
                )
            })
            .collect();
        let jobs = batch.prepare(corpus, self.max_token_length.unwrap_or(usize::MAX))?;
        let changes = jobs
            .into_par_iter()
            .map(|job| {
                corpus.apply_merges(&job.starts, job.replacement);
                job.changes
            })
            .collect();
        index.commit(changes)?;
        vocabulary.commit_merges(records);
        #[cfg(test)]
        vocabulary.trace.extend(trace);
        Ok(MergeOutcome::Committed(accepted))
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
#[derive(Default)]
struct Directories {
    indices: [Vec<u32>; 2],
}

impl<'a> Neighbors<'a> {
    fn new(rule: &'a Rule, domain: usize, directories: &'a mut Directories) -> Self {
        for values in &mut directories.indices {
            values.resize(domain, u32::MAX);
        }
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
        // Bind the historical stream and word owners to the joined snapshot.
        let rule = &self.rules[0];
        let cohort = corpus.cohort(&rule.candidate.positions, rule.pair(), rule.replacement);
        let jobs = cohort
            .chunks()
            .map_init(Directories::default, |directories, matches| -> Result<_> {
                #[cfg(test)]
                super::tests::observe_worker(super::tests::Phase::ReusePrepare);
                let mut neighbors = Neighbors::new(rule, corpus.id_count(), directories);
                let mut starts = SmallVec::new();
                for occurrence in matches {
                    let matched = occurrence.matched;
                    if let Some(previous) = occurrence.previous {
                        neighbors.record(
                            true,
                            previous.id,
                            previous.id,
                            previous.start,
                            occurrence.weight,
                            previous.span + matched.span() < limit,
                        )?;
                    }
                    if let Some(next) = occurrence.next {
                        neighbors.record(
                            false,
                            next.id,
                            next.id,
                            matched.start,
                            occurrence.weight,
                            matched.span() + next.span < limit,
                        )?;
                    }
                    starts.push(matched.start);
                }
                Ok(Job {
                    starts,
                    replacement: rule.replacement,
                    changes: neighbors.finish(),
                })
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
                if coordinate >= after
                    && let Some(matched) = matcher(coordinate)
                {
                    starts.push(coordinate);
                    after = matched.after as u64;
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

    fn tasks(&self) -> Vec<(usize, impl Iterator<Item = u64> + Send + '_)> {
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
                        tasks.push((rank, itertools::Either::Left(part.iter().copied())));
                    }
                }
                FreshMatches::Ordinary(_) => {
                    let positions = &rule.candidate.positions;
                    for positions in positions.chunks(ordinary_chunk) {
                        tasks.push((rank, itertools::Either::Right(positions)));
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
        positions: impl Iterator<Item = u64>,
    ) -> Result<Job> {
        #[cfg(test)]
        super::tests::observe_worker(super::tests::Phase::FreshPrepare);
        let rule = &self.batch.rules[rank];
        let mut neighbors = Neighbors::new(rule, self.corpus.id_count(), directories);
        let mut starts = SmallVec::new();
        for occurrence in self.corpus.fresh_matches(positions, rule.pair()) {
            let start = occurrence.matched.start;
            self.record_neighbors(&mut neighbors, occurrence)?;
            starts.push(start);
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
        occurrence: FreshMatch,
    ) -> Result<()> {
        let FreshMatch {
            matched,
            weight,
            previous,
            next,
            adjacent_pairs,
            preceding_start,
        } = occurrence;
        // Left edge: register it only if the preceding match does not own it.
        if let Some(prior) = previous {
            let merging = match &self.matches {
                FreshMatches::SelfPair(starts) => {
                    preceding_start.is_some_and(|start| starts.binary_search(&start).is_ok())
                }
                FreshMatches::Ordinary(selected) => {
                    adjacent_pairs[0].is_some_and(|pair| selected.contains_key(&pair))
                }
            };
            if !merging {
                neighbors.record(
                    true,
                    prior.id,
                    prior.id,
                    prior.start,
                    weight,
                    prior.span + matched.span() < self.limit,
                )?;
            }
        }

        // Right edge: include the following match's replacement when selected.
        if let Some(next) = next {
            let replacement = match &self.matches {
                FreshMatches::SelfPair(starts) => starts
                    .binary_search(&(matched.after as u64))
                    .is_ok()
                    .then_some(neighbors.rule.replacement),
                FreshMatches::Ordinary(selected) => {
                    adjacent_pairs[1].and_then(|pair| selected.get(&pair).copied())
                }
            };
            let born = replacement.unwrap_or(next.id);
            neighbors.record(
                false,
                next.id,
                born,
                matched.start,
                weight,
                matched.span() + self.corpus.id_span(born) < self.limit,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{spans, tokens};
    use super::super::{
        WORD_SEPARATOR_ID,
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
                .prepare_job(&mut Directories::default(), 0, input.iter().copied())
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
    fn directory_recovery() {
        let trainer = trainer();
        let words = [("xab".into(), 1)].into_iter().collect();
        let (mut vocabulary, mut corpus, mut index) =
            initialize(&trainer, WordCountsView::from_map(&words), false);
        let rule = select(&mut vocabulary, &mut corpus, &mut index);
        let mut directories = Directories::default();
        {
            let mut neighbors = Neighbors::new(&rule, corpus.id_count(), &mut directories);
            neighbors.record(true, 2, 2, 1, u64::MAX, true).unwrap();
            assert!(neighbors.record(true, 2, 2, 1, 1, true).is_err());
        }
        // The old directory index is reused without reset after the failed group.
        let mut neighbors = Neighbors::new(&rule, corpus.id_count(), &mut directories);
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
