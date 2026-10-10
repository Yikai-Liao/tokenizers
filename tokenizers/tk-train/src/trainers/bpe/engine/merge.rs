//! Compatible selection and snapshot preparation hide endpoint/event bookkeeping.
use super::{BpeTrainer, Corpus, PairIndex, Vocabulary, WORD_SEPARATOR_ID, add};
use super::{
    corpus::Match,
    index::Candidate,
    positions::{Arena, Builder, Input, Positions},
};
use ahash::{AHashMap, AHashSet};
use rayon::prelude::*;
use tk_encode::{Result, models::bpe::Pair};

struct Rule<'arena> {
    pair: Pair,
    replacement: u32,
    candidate: Candidate<'arena>,
}
pub(super) struct Batch<'arena> {
    rules: Vec<Rule<'arena>>,
    reuse: bool,
    floor: u64,
}
pub(super) enum Selection<'arena> {
    Finished,
    // The whole attempt is discarded, including rules already selected in this batch.
    Restart,
    Ready(Batch<'arena>),
}
pub(super) struct Change<P> {
    pub(super) removed: Pair,
    pub(super) born: Pair,
    pub(super) removed_weight: u64,
    pub(super) born_weight: u64,
    pub(super) positions: P,
    pub(super) bucket: usize,
}
pub(super) enum Birth<'arena> {
    // Local fresh counts await owner aggregation before floor admission.
    // Reuse fragments instead follow the owner's per-action signed ledger.
    Partial(Builder),
    // A fresh ordinary producer covers the full candidate and has applied the floor.
    // Retained lists are ready for direct publication without another encoding pass.
    Complete(Positions<'arena>),
}
impl Birth<'_> {
    pub(super) fn is_empty(&self) -> bool {
        match self {
            Self::Partial(values) => values.is_empty(),
            Self::Complete(values) => values.is_empty(),
        }
    }
}
pub(super) struct Prepared<'arena> {
    jobs: Vec<Job<'arena>>,
}
struct Job<'arena> {
    writes: Writes,
    changes: Vec<Change<Birth<'arena>>>,
}
struct Neighbors<'a, 'arena> {
    rule: &'a Rule<'arena>,
    rank: usize,
    directories: &'a mut Directories,
    changes: [Vec<Change<Builder>>; 2],
    complete: bool,
}
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
impl<'a, 'arena> Neighbors<'a, 'arena> {
    fn new(
        rule: &'a Rule<'arena>,
        rank: usize,
        directories: &'a mut Directories,
        complete: bool,
    ) -> Self {
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
            let pair = self.rule.pair;
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
            group.positions.push_ordered(position as u64)?;
        }
        Ok(())
    }
    fn finish(self, floor: u64, arena: &'arena Arena) -> Result<Vec<Change<Birth<'arena>>>> {
        // Complete ordinary producers prune and encode before owner routing.
        // Partial jobs retain raw positions until their counts have been reduced.
        let mut result = Vec::new();
        let mut lease = arena.lease();
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
impl<'arena> Batch<'arena> {
    pub(super) fn select(
        trainer: &BpeTrainer,
        vocabulary: &mut Vocabulary,
        corpus: &mut Corpus,
        index: &mut PairIndex<'arena>,
    ) -> Result<Selection<'arena>> {
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
            if !reuse && token.existing_id.is_some_and(|id| corpus.is_active(id)) {
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
                pair,
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
        self.rules.iter().map(|rule| rule.pair)
    }
    #[cfg(test)]
    pub(super) fn trace(&self) -> impl Iterator<Item = (Pair, u64, u32)> + '_ {
        self.rules
            .iter()
            .map(|rule| (rule.pair, rule.candidate.priority.count, rule.replacement))
    }
    pub(super) fn prepare(
        self,
        corpus: &Corpus,
        arena: &'arena Arena,
        limit: usize,
    ) -> Result<Prepared<'arena>> {
        if self.reuse {
            return self.prepare_cohort(corpus, arena, limit);
        }
        let selected: AHashMap<_, _> = self
            .rules
            .iter()
            .map(|rule| (rule.pair, rule.replacement))
            .collect();
        let mut heads = vec![None; corpus.id_count()];
        let mut tails = vec![false; corpus.id_count()];
        for rule in &self.rules {
            let slot = &mut heads[rule.pair.0 as usize];
            // The separator cannot be a real rule endpoint; it marks a shared head.
            *slot = Some(if slot.is_some() {
                (WORD_SEPARATOR_ID, 0)
            } else {
                (rule.pair.1, rule.replacement)
            });
            tails[rule.pair.1 as usize] = true;
        }
        let selected_id = |pair: Pair| match heads.get(pair.0 as usize).copied().flatten() {
            Some((right, id)) if right != WORD_SEPARATOR_ID => (right == pair.1).then_some(id),
            None => None,
            _ => selected.get(&pair).copied(),
        };
        let aa = self.rules[0].pair.0 == self.rules[0].pair.1;
        let mut starts = Vec::new();
        if aa {
            let rule = &self.rules[0];
            let matcher = corpus.fresh_matcher(rule.pair);
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
        }
        let mut tasks = Vec::new();
        let total: usize = self
            .rules
            .iter()
            .map(|rule| rule.candidate.positions.len())
            .sum();
        let chunk = total.div_ceil(rayon::current_num_threads() * 4).max(4096);
        let ordinary_chunk = total
            .div_ceil(rayon::current_num_threads())
            .clamp(1, 1 << 26);
        for (rank, rule) in self.rules.iter().enumerate() {
            if aa {
                for part in starts.chunks(chunk) {
                    tasks.push((rank, rule, Source::Slice(part)));
                }
            } else {
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
        let jobs = tasks
            .into_par_iter()
            .map_init(
                Directories::default,
                |directories, (rank, rule, positions)| -> Result<_> {
                    #[cfg(test)]
                    super::tests::observe_worker(super::tests::Phase::FreshPrepare);
                    directories.reset(corpus.id_count());
                    let mut neighbors =
                        Neighbors::new(rule, rank, directories, positions.complete());
                    let mut writes = Writes::fresh(rule, corpus);
                    let matcher = corpus.fresh_matcher(rule.pair);
                    let mut weights = None;
                    for p in positions.prefetched(corpus) {
                        let Some(matched) = matcher(p, corpus.token(p)) else {
                            continue;
                        };
                        let (weight, end) = match weights {
                            Some((weight, end)) if p < end => (weight, end),
                            _ => corpus.weight_region(p),
                        };
                        weights = Some((weight, end));
                        let prior = corpus.token(p - 1);
                        if prior != WORD_SEPARATOR_ID {
                            let span = corpus.id_span(prior);
                            let before = p - span;
                            let merging = if aa {
                                p >= matched.span()
                                    && starts.binary_search(&((p - matched.span()) as u64)).is_ok()
                            } else {
                                before != 0
                                    && tails[prior as usize]
                                    && selected_id((corpus.token(before - 1), prior)).is_some()
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
                                    span + matched.span() < limit,
                                )?;
                            }
                        }
                        let next = corpus.token(matched.after);
                        if next != WORD_SEPARATOR_ID {
                            let replacement = if aa {
                                starts
                                    .binary_search(&(matched.after as u64))
                                    .is_ok()
                                    .then_some(rule.replacement)
                            } else if heads[next as usize].is_some() {
                                selected_id((
                                    next,
                                    corpus.token(matched.after + corpus.id_span(next)),
                                ))
                            } else {
                                None
                            };
                            let born = replacement.unwrap_or(next);
                            neighbors.record(
                                false,
                                next,
                                born,
                                p,
                                weight,
                                matched.span() + corpus.id_span(born) < limit,
                            )?;
                        }
                        writes.record(matched)?;
                    }
                    Ok(Job {
                        writes,
                        changes: neighbors.finish(self.floor, arena)?,
                    })
                },
            )
            .collect::<Result<Vec<_>>>()?;
        Ok(Prepared { jobs })
    }
    fn prepare_cohort(
        self,
        corpus: &Corpus,
        arena: &'arena Arena,
        limit: usize,
    ) -> Result<Prepared<'arena>> {
        let rule = &self.rules[0];
        let mut words: Vec<_> = rule
            .candidate
            .positions
            .iter()
            .map(|p| corpus.word(corpus.resident(p)))
            .collect();
        words.dedup();
        let chunk = words.len().div_ceil(rayon::current_num_threads()).max(1);
        let jobs = words
            .par_chunks(chunk)
            .map_init(Directories::default, |directories, words| -> Result<_> {
                #[cfg(test)]
                super::tests::observe_worker(super::tests::Phase::ReusePrepare);
                directories.reset(corpus.id_count());
                let mut neighbors = Neighbors::new(rule, 0, directories, false);
                let mut writes = Writes::Occurrences {
                    positions: Vec::new(),
                    id: rule.replacement,
                };
                for &word in words {
                    let mut previous = None;
                    let mut p = corpus.word_start(word);
                    let start = rule.candidate.positions.lower_bound(p as u64);
                    let mut positions = rule.candidate.positions.from(start).peekable();
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
                                previous = (id != WORD_SEPARATOR_ID).then(|| {
                                    (id, next - corpus.span(next - 1), corpus.span(next - 1))
                                });
                                p = next;
                            }
                            positions.next();
                        }
                        if let Some(matched) = corpus.matched(p, rule.pair) {
                            let weight = corpus.word_weight(word);
                            if let Some((id, start, span)) = previous {
                                neighbors.record(
                                    true,
                                    id,
                                    id,
                                    start,
                                    weight,
                                    span + matched.span() < limit,
                                )?;
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
                            writes.record(matched)?;
                            p = matched.after;
                        } else {
                            let span = corpus.span(p);
                            previous = Some((corpus.token(p), p, span));
                            p += span.max(1);
                        }
                    }
                }
                Ok(Job {
                    writes,
                    changes: neighbors.finish(self.floor, arena)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Prepared { jobs })
    }
}
impl<'arena> Prepared<'arena> {
    pub(super) fn apply(self, corpus: &Corpus) -> Vec<Vec<Change<Birth<'arena>>>> {
        self.jobs
            .into_par_iter()
            .map(|job| {
                job.writes.apply(corpus);
                job.changes
            })
            .collect()
    }
}

// Fresh identities have one span per ID, so their snapshot writes need only
// sorted starts and one rule geometry. Alias reuse retains occurrence geometry.
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
    fn fresh(rule: &Rule<'_>, corpus: &Corpus) -> Self {
        let left = corpus.id_span(rule.pair.0);
        Self::Compact {
            positions: Builder::default(),
            left,
            total: left + corpus.id_span(rule.pair.1),
            id: rule.replacement,
        }
    }
    fn record(&mut self, matched: Match) -> Result<()> {
        match self {
            Self::Compact { positions, .. } => positions.push_ordered(matched.start as u64),
            Self::Occurrences { positions, .. } => {
                positions.push(matched);
                Ok(())
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

// Source geometry stays private to preparation; both paths emit full-u64 positions.
enum Source<'a, 'arena> {
    Slice(&'a [u64]),
    Blocks(&'a Positions<'arena>, std::ops::Range<usize>),
}
impl Source<'_, '_> {
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
