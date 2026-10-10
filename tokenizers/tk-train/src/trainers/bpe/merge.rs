//! Compatible selection and snapshot preparation hide endpoint/event bookkeeping.
use super::{
    BpeTrainer, WORD_SEPARATOR_ID, add,
    corpus::{Corpus, Match},
    index::{Candidate, PairIndex},
    positions::{Builder, Codec, Input, Positions},
    vocabulary::Vocabulary,
};
use ahash::{AHashMap, AHashSet};
use itertools::Itertools;
use rayon::prelude::*;
use tk_encode::{Result, models::bpe::Pair};

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
pub(super) struct Batch {
    rules: Vec<Rule>,
    reuse: bool,
    floor: u64,
}

/// Selection outcome for the coordinator: finish, retry the attempt, or prepare.
/// Restart discards speculative rules and requests historical-cohort training.
pub(super) enum Selection {
    Finished,
    // The whole attempt is discarded, including rules already selected in this batch.
    Restart,
    Ready(Batch),
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
    Partial(Builder),
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

/// Joined preparation output, owning endpoint writes and neighbor events.
/// Apply joins all writes before returning events for the coordinator's index commit.
pub(super) struct Prepared {
    jobs: Vec<Job>,
}

/// One preparation task's disjoint writes and owned neighbor-event payloads.
/// The task returns both together so no corpus mutation precedes preparation join.
struct Job {
    writes: Writes,
    changes: Vec<Change<Birth>>,
}

/// Task-local left/right neighbor aggregation for one selected rule.
/// Borrows reusable lookup directories; owns its groups and decides whether
/// they cover a complete producer or must be routed as partial fragments.
struct Neighbors<'a> {
    rule: &'a Rule,
    rank: usize,
    directories: &'a mut Directories,
    changes: [Vec<Change<Builder>>; 2],
    complete: bool,
}

/// Reusable direct-ID lookups from left/right neighbors to task-local group indices.
/// Only touched slots are reset between tasks; u32::MAX marks an absent group.
#[derive(Default)]
struct Directories {
    indices: [Vec<u32>; 2],
    touched: Vec<(usize, usize)>,
}

impl Directories {
    fn reset(&mut self, domain: usize) {
        for (side, id) in self.touched.drain(..) {
            self.indices[side][id] = u32::MAX;
        }
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

    fn group(&mut self, neighbor: u32, left: bool) -> &mut Change<Builder> {
        let side = usize::from(!left);
        let slot = &mut self.directories.indices[side][neighbor as usize];
        if *slot == u32::MAX {
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
                positions: Builder::default(),
                // Both sides can create (id, id); keep them in one birth cohort.
                // Other right births follow left births in the reference update order.
                bucket: 2 * self.rank + usize::from(!left && neighbor != id),
            });
            self.directories.touched.push((side, neighbor as usize));
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
            group.positions.push_ordered(position as u64);
        }
        Ok(())
    }

    fn finish(self, floor: u64, codec: &Codec) -> Result<Vec<Change<Birth>>> {
        // Complete ordinary producers prune and encode before owner routing.
        // Partial jobs retain raw positions until their counts have been reduced.
        let mut result = Vec::new();
        let mut lease = codec.lease();
        // Left changes precede right changes. Reuse count actions and the resulting
        // historical cohorts observe this order even when both sides name one key.
        for mut change in self.changes.into_iter().flatten() {
            if self.complete && change.born_weight < floor {
                change.positions = Builder::default();
                change.born_weight = 0;
            }
            if change.removed_weight == 0 && change.positions.is_empty() {
                continue;
            }
            let positions = if self.complete {
                Birth::Complete(Positions::from_sorted(
                    Input::Builder(&change.positions),
                    &mut lease,
                )?)
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
    pub(super) fn select(
        trainer: &BpeTrainer,
        vocabulary: &mut Vocabulary,
        corpus: &mut Corpus,
        index: &mut PairIndex<'_>,
    ) -> Result<Selection> {
        let reuse = index.reuse();
        let mut batch = Self {
            rules: Vec::new(),
            reuse,
            floor: trainer.min_frequency.max(1),
        };
        let cap = if reuse {
            1
        } else {
            256.min(trainer.vocab_size - vocabulary.len())
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
            let token = vocabulary.merge_token(pair);
            if !reuse
                && token
                    .existing_id
                    .is_some_and(|id| corpus.has_been_activated(id))
            {
                return Ok(Selection::Restart);
            }
            let reserved = token.existing_id.is_some();
            // A reserved ID need not follow old IDs lexicographically. A singleton
            // activation needs no argument about its position relative to later rules.
            if reserved && !batch.rules.is_empty() {
                break;
            }
            let candidate = index.take(priority);
            let replacement = vocabulary.resolve_merge(token)?;
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
        Ok(if batch.rules.is_empty() {
            Selection::Finished
        } else {
            Selection::Ready(batch)
        })
    }

    pub(super) fn pairs(&self) -> impl Iterator<Item = Pair> + '_ {
        self.rules.iter().map(|rule| rule.pair())
    }

    #[cfg(test)]
    pub(super) fn trace(&self) -> impl Iterator<Item = (Pair, u64, u32)> + '_ {
        self.rules
            .iter()
            .map(|rule| (rule.pair(), rule.candidate.priority.count, rule.replacement))
    }

    pub(super) fn prepare(self, corpus: &Corpus, codec: &Codec, limit: usize) -> Result<Prepared> {
        if self.reuse {
            return self.prepare_cohort(corpus, codec, limit);
        }

        // 1. Resolve the selected rules against this joined corpus snapshot.
        let snapshot = FreshSnapshot::new(&self, corpus, codec, limit);

        // 2. Split ordered candidate streams into independent preparation jobs.
        let tasks = snapshot.tasks();
        let jobs = tasks
            .into_par_iter()
            .map_init(Directories::default, |directories, (rank, rule, source)| {
                snapshot.prepare_job(directories, rank, rule, source)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Prepared { jobs })
    }

    fn prepare_cohort(self, corpus: &Corpus, codec: &Codec, limit: usize) -> Result<Prepared> {
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
                CohortPreparation::new(rule, corpus, codec, limit, self.floor, directories)
                    .prepare(words)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Prepared { jobs })
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
    codec: &'a Codec,
    limit: usize,
    matches: FreshMatches,
}

enum FreshMatches {
    Ordinary(SelectedRules),
    SelfPair(Vec<u64>),
}

struct SelectedRules {
    selected: AHashMap<Pair, u32>,
    heads: Vec<Head>,
    tails: Vec<bool>,
}

impl SelectedRules {
    fn new(batch: &Batch, id_count: usize) -> Self {
        // Resolve shared endpoints once for all tasks in an ordinary batch.
        let selected: AHashMap<_, _> = batch
            .rules
            .iter()
            .map(|rule| (rule.pair(), rule.replacement))
            .collect();
        let mut heads = vec![Head::Empty; id_count];
        let mut tails = vec![false; id_count];
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
    fn new(batch: &'a Batch, corpus: &'a Corpus, codec: &'a Codec, limit: usize) -> Self {
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
            codec,
            limit,
            matches,
        }
    }

    fn tasks(&self) -> Vec<(usize, &Rule, Source<'_>)> {
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
                        tasks.push((rank, rule, Source::Slice(part)));
                    }
                }
                FreshMatches::Ordinary(_) => {
                    let positions = &rule.candidate.positions;
                    if positions.len() <= ordinary_chunk {
                        tasks.push((
                            rank,
                            rule,
                            Source::Blocks(positions, 0..positions.block_count()),
                        ));
                        continue;
                    }
                    for range in positions.block_ranges(ordinary_chunk) {
                        tasks.push((rank, rule, Source::Blocks(positions, range)));
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
        rule: &Rule,
        positions: Source<'_>,
    ) -> Result<Job> {
        #[cfg(test)]
        super::tests::observe_worker(super::tests::Phase::FreshPrepare);
        directories.reset(self.corpus.id_count());
        let mut neighbors = Neighbors::new(rule, rank, directories, positions.complete());
        let mut writes = Writes::fresh(rule, self.corpus);
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
            self.record_left(&mut neighbors, p, matched, weight)?;
            self.record_right(&mut neighbors, rule, p, matched, weight)?;
            writes.record(matched);
        }
        Ok(Job {
            writes,
            changes: neighbors.finish(self.batch.floor, self.codec)?,
        })
    }

    fn record_left(
        &self,
        neighbors: &mut Neighbors<'_>,
        p: usize,
        matched: Match,
        weight: u64,
    ) -> Result<()> {
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
            // The selected match on the left owns a shared boundary.
            // Its right event emits the final replacements of both rules;
            // this match must not emit a second left removal or birth.
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
        Ok(())
    }

    fn record_right(
        &self,
        neighbors: &mut Neighbors<'_>,
        rule: &Rule,
        p: usize,
        matched: Match,
        weight: u64,
    ) -> Result<()> {
        let next = self.corpus.token(matched.after);
        if next != WORD_SEPARATOR_ID {
            let replacement = match &self.matches {
                FreshMatches::SelfPair(starts) => starts
                    .binary_search(&(matched.after as u64))
                    .is_ok()
                    .then_some(rule.replacement),
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
    writes: Writes,
    neighbors: Neighbors<'task>,
    corpus: &'task Corpus,
    codec: &'task Codec,

    limit: usize,
    floor: u64,
}

impl<'task> CohortPreparation<'task> {
    fn new(
        rule: &'task Rule,
        corpus: &'task Corpus,
        codec: &'task Codec,

        limit: usize,
        floor: u64,
        directories: &'task mut Directories,
    ) -> Self {
        #[cfg(test)]
        super::tests::observe_worker(super::tests::Phase::ReusePrepare);
        directories.reset(corpus.id_count());
        let neighbors = Neighbors::new(rule, 0, directories, false);
        let writes = Writes::Occurrences {
            positions: Vec::new(),
            id: rule.replacement,
        };
        Self {
            writes,
            neighbors,
            corpus,
            codec,
            limit,
            floor,
        }
    }

    fn prepare(mut self, words: &[usize]) -> Result<Job> {
        for &word in words {
            self.prepare_word(word)?;
        }
        Ok(Job {
            writes: self.writes,
            changes: self.neighbors.finish(self.floor, self.codec)?,
        })
    }

    // Within one word each accepted merge changes the next left neighbor. Keep
    // this sequential while separate word chunks prepare in parallel.
    fn prepare_word(&mut self, word: usize) -> Result<()> {
        let rule = self.neighbors.rule;
        let corpus = self.corpus;
        let limit = self.limit;
        let neighbors = &mut self.neighbors;
        let writes = &mut self.writes;
        let mut previous = None;
        let mut p = corpus.word_start(word);
        let index = rule.candidate.positions.lower_bound(p as u64);
        let mut positions = rule.candidate.positions.iter_from(index).peekable();
        while p < corpus.word_end(word) {
            // Full-word scans are required only after alias reuse or a length
            // gate. Otherwise the selected cohort defines the scan domain.
            if !corpus.whole_words() {
                while positions.peek().is_some_and(|&start| start < p as u64) {
                    positions.next();
                }
                let Some(coordinate) = positions.peek().copied() else {
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
                positions.next();
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
                writes.record(matched);
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

impl Prepared {
    pub(super) fn apply(self, corpus: &Corpus) -> Vec<Vec<Change<Birth>>> {
        self.jobs
            .into_par_iter()
            .map(|job| {
                job.writes.apply(corpus);
                job.changes
            })
            .collect()
    }
}

/// Deferred endpoint writes, compact for fresh identities and explicit for reuse.
/// Fresh identities have one span per ID, so writes need only sorted starts and
/// one rule geometry. Alias reuse retains each occurrence's snapshot geometry.
enum Writes {
    Compact {
        positions: Builder,
        left: usize,
        total: usize,
        id: u32,
    },
    Occurrences {
        positions: Vec<Match>,
        id: u32,
    },
}

impl Writes {
    fn fresh(rule: &Rule, corpus: &Corpus) -> Self {
        let left = corpus.id_span(rule.pair().0);
        Self::Compact {
            positions: Builder::default(),
            left,
            total: left + corpus.id_span(rule.pair().1),
            id: rule.replacement,
        }
    }

    fn record(&mut self, matched: Match) {
        match self {
            Self::Compact { positions, .. } => positions.push_ordered(matched.start as u64),
            Self::Occurrences { positions, .. } => {
                positions.push(matched);
            }
        }
    }

    fn apply(self, corpus: &Corpus) {
        match self {
            Self::Compact {
                positions,
                left,
                total,
                id,
            } => {
                for coordinate in positions.iter() {
                    let start = corpus.resident(coordinate);
                    corpus.apply(
                        Match {
                            start,
                            right: start + left,
                            after: start + total,
                        },
                        id,
                    );
                }
            }
            Self::Occurrences { positions, id } => {
                for matched in positions {
                    corpus.apply(matched, id);
                }
            }
        }
    }
}

/// Borrowed task input: filtered self-pair starts or a candidate's restart blocks.
/// Source geometry stays private to preparation; both emit full-u64 positions.
enum Source<'a> {
    Slice(&'a [u64]),
    Blocks(&'a Positions, std::ops::Range<usize>),
}

impl Source<'_> {
    fn complete(&self) -> bool {
        match self {
            Self::Blocks(positions, range) => {
                range.start == 0 && range.end == positions.block_count()
            }
            Self::Slice(_) => false,
        }
    }

    // Decode one bounded ring ahead. Prefetch is a nonblocking
    // hint; matching still reads the same joined preparation snapshot in order.
    fn prefetched<'a>(&'a self, corpus: &'a Corpus) -> impl Iterator<Item = usize> + 'a {
        let coordinates = match self {
            Self::Slice(values) => itertools::Either::Left(values.iter().copied()),
            Self::Blocks(positions, range) => {
                itertools::Either::Right(positions.read_blocks(range.clone()))
            }
        };
        let mut positions = coordinates.map(move |coordinate| corpus.resident(coordinate));
        let mut ring: [Option<usize>; 16] = std::array::from_fn(|_| positions.next());
        for &p in ring.iter().flatten() {
            corpus.prefetch(p);
        }
        let mut head = 0;
        std::iter::from_fn(move || {
            let position = ring[head]?;
            ring[head] = positions.next();
            if let Some(next) = ring[head] {
                corpus.prefetch(next);
            }
            head = (head + 1) % ring.len();
            Some(position)
        })
    }
}
