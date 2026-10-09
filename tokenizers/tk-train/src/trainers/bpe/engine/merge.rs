//! Compatible selection and snapshot preparation hide endpoint/event bookkeeping.
use super::{BpeTrainer, Corpus, PairIndex, Vocabulary, WORD_SEPARATOR_ID};
use super::{corpus::Match, index::Candidate, positions::Positions};
use ahash::{AHashMap, AHashSet};
use rayon::prelude::*;
use tk_encode::{Result, models::bpe::Pair};

struct Rule {
    pair: Pair,
    replacement: u32,
    candidate: Candidate,
}
pub(super) struct Batch {
    rules: Vec<Rule>,
    reuse: bool,
    restart: bool,
}
pub(super) struct Change {
    pub(super) removed: Pair,
    pub(super) born: Pair,
    pub(super) removed_weight: u64,
    pub(super) born_weight: u64,
    pub(super) positions: Vec<u64>,
    pub(super) bucket: usize,
}
pub(super) struct Prepared {
    jobs: Vec<Job>,
}
struct Job {
    writes: Vec<(Match, u32)>,
    changes: Vec<Change>,
}
struct Neighbors<'a> {
    rule: &'a Rule,
    rank: usize,
    left: AHashMap<u32, usize>,
    right: AHashMap<u32, usize>,
    changes: Vec<Change>,
}
impl<'a> Neighbors<'a> {
    fn new(rule: &'a Rule, rank: usize) -> Self {
        Self {
            rule,
            rank,
            left: AHashMap::new(),
            right: AHashMap::new(),
            changes: Vec::new(),
        }
    }
    fn group(&mut self, neighbor: u32, left: bool) -> &mut Change {
        let directory = if left {
            &mut self.left
        } else {
            &mut self.right
        };
        let index = *directory.entry(neighbor).or_insert_with(|| {
            let id = self.rule.replacement;
            let pair = self.rule.pair;
            let index = self.changes.len();
            self.changes.push(Change {
                removed: if left {
                    (neighbor, pair.0)
                } else {
                    (pair.1, neighbor)
                },
                born: if left { (neighbor, id) } else { (id, neighbor) },
                removed_weight: 0,
                born_weight: 0,
                positions: Vec::new(),
                bucket: 2 * self.rank + usize::from(!left && neighbor != id),
            });
            index
        });
        &mut self.changes[index]
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
        group.removed_weight = group
            .removed_weight
            .checked_add(weight)
            .ok_or("BPE neighbor removal mass exceeds u64")?;
        if admit {
            let group = self.group(born, left);
            group.born_weight = group
                .born_weight
                .checked_add(weight)
                .ok_or("BPE neighbor birth mass exceeds u64")?;
            group.positions.push(position as u64);
        }
        Ok(())
    }
    fn finish(self) -> Vec<Change> {
        // Match the reference's removal-before-birth order and left/right drains.
        let mut changes: Vec<_> = self.changes.into_iter().enumerate().collect();
        let left: AHashSet<_> = self.left.into_values().collect();
        changes.sort_unstable_by_key(|(i, _)| (!left.contains(i), *i));
        changes.into_iter().map(|(_, change)| change).collect()
    }
}
impl Batch {
    pub(super) fn select(
        trainer: &BpeTrainer,
        vocabulary: &mut Vocabulary,
        corpus: &mut Corpus,
        index: &mut PairIndex,
    ) -> Result<Option<Self>> {
        let reuse = index.reuse();
        let mut batch = Self {
            rules: Vec::new(),
            reuse,
            restart: false,
        };
        let cap = if reuse {
            1
        } else {
            256.min(trainer.vocab_size - vocabulary.len())
        };
        let mut heads = AHashSet::new();
        let mut tails = AHashSet::new();
        while batch.rules.len() < cap {
            let Some(priority) = index.best() else {
                break;
            };
            let pair = priority.pair;
            if !batch.rules.is_empty()
                && (pair.0 == pair.1 || tails.contains(&pair.0) || heads.contains(&pair.1))
            {
                break;
            }
            let token = vocabulary.merge_token(pair);
            if !reuse && vocabulary.reuses_active_id(&token) {
                batch.restart = true;
                return Ok(Some(batch));
            }
            let reserved = token.existing_id.is_some();
            if reserved && !batch.rules.is_empty() {
                break;
            }
            let candidate = index.take(priority);
            let identity = vocabulary.resolve_merge(token)?;
            corpus.prepare_identity(pair, identity.id, identity.reused_active_id);
            batch.rules.push(Rule {
                pair,
                replacement: identity.id,
                candidate,
            });
            if reserved || pair.0 == pair.1 {
                break;
            }
            heads.insert(pair.0);
            tails.insert(pair.1);
        }
        Ok((!batch.rules.is_empty()).then_some(batch))
    }
    pub(super) fn restart(&self) -> bool {
        self.restart
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
    pub(super) fn prepare(self, corpus: &Corpus, limit: usize) -> Result<Prepared> {
        if self.reuse {
            return self.prepare_cohort(corpus, limit);
        }
        let selected: AHashMap<_, _> = self
            .rules
            .iter()
            .map(|rule| (rule.pair, rule.replacement))
            .collect();
        let aa = self.rules[0].pair.0 == self.rules[0].pair.1;
        let mut starts = Vec::new();
        if aa {
            let rule = &self.rules[0];
            let mut after = 0;
            for coordinate in rule.candidate.positions.iter() {
                let p = corpus.resident(coordinate);
                if p >= after
                    && let Some(matched) = corpus.matched(p, rule.pair)
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
        for (rank, rule) in self.rules.iter().enumerate() {
            if aa {
                for part in starts.chunks(chunk) {
                    tasks.push((rank, rule, Source::Slice(part)));
                }
            } else {
                let positions = &rule.candidate.positions;
                let blocks = chunk.div_ceil(128);
                for begin in (0..positions.block_count()).step_by(blocks) {
                    tasks.push((
                        rank,
                        rule,
                        Source::Blocks(
                            positions,
                            begin..(begin + blocks).min(positions.block_count()),
                        ),
                    ));
                }
            }
        }
        let jobs = tasks
            .into_par_iter()
            .map(|(rank, rule, positions)| -> Result<_> {
                let mut neighbors = Neighbors::new(rule, rank);
                let mut writes = Vec::new();
                let mut word = None;
                for coordinate in positions.iter() {
                    let p = corpus.resident(coordinate);
                    let Some(matched) = corpus.matched(p, rule.pair) else {
                        continue;
                    };
                    let iw = match word {
                        Some((iw, end)) if p < end => iw,
                        _ => corpus.word(p),
                    };
                    word = Some((iw, corpus.word_end(iw) + 1));
                    let weight = corpus.word_weight(iw);
                    let prior = corpus.token(p - 1);
                    if prior != WORD_SEPARATOR_ID {
                        let span = corpus.id_span(prior);
                        let before = p - span;
                        let merging = if aa {
                            p >= matched.span()
                                && starts.binary_search(&((p - matched.span()) as u64)).is_ok()
                        } else {
                            before != 0 && selected.contains_key(&(corpus.token(before - 1), prior))
                        };
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
                        } else {
                            selected
                                .get(&(next, corpus.token(matched.after + corpus.id_span(next))))
                                .copied()
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
                    writes.push((matched, rule.replacement));
                }
                Ok(Job {
                    writes,
                    changes: neighbors.finish(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Prepared { jobs })
    }
    fn prepare_cohort(self, corpus: &Corpus, limit: usize) -> Result<Prepared> {
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
            .map(|words| -> Result<_> {
                let mut neighbors = Neighbors::new(rule, 0);
                let mut writes = Vec::new();
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
                            writes.push((matched, rule.replacement));
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
                    changes: neighbors.finish(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Prepared { jobs })
    }
}
impl Prepared {
    pub(super) fn apply(self, corpus: &Corpus) -> Vec<Change> {
        self.jobs
            .into_par_iter()
            .map(|job| {
                for (matched, id) in job.writes {
                    corpus.apply(matched, id);
                }
                job.changes
            })
            .collect::<Vec<_>>()
            .into_iter()
            .flatten()
            .collect()
    }
}

// Source geometry stays private to preparation; both paths emit full-u64 positions.
enum Source<'a> {
    Slice(&'a [u64]),
    Blocks(&'a Positions, std::ops::Range<usize>),
}
impl Source<'_> {
    fn iter(&self) -> Box<dyn Iterator<Item = u64> + '_> {
        match self {
            Self::Slice(values) => Box::new(values.iter().copied()),
            Self::Blocks(positions, range) => Box::new(positions.read_blocks(range.clone())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::WordCountsView;
    use super::*;
    use crate::progress::TrainingProgress;
    use compact_str::CompactString;
    #[test]
    fn independent_priority_rules_are_selected_together() {
        let trainer = BpeTrainer::builder()
            .vocab_size(9)
            .min_frequency(1)
            .show_progress(false)
            .build();
        let words: AHashMap<CompactString, u64> =
            [("ab".into(), 10), ("cd".into(), 9), ("ef".into(), 8)].into();
        let progress = TrainingProgress::new(false, trainer.progress_format).unwrap();
        let view = WordCountsView::from_map(&words);
        let mut vocabulary =
            Vocabulary::initialize(&trainer, view, 4, &progress, &mut None).unwrap();
        let mut corpus = Corpus::build(view, &mut vocabulary, &trainer, false, &progress).unwrap();
        let mut index = PairIndex::build(&corpus, 1, 4, false, &progress).unwrap();
        let batch = Batch::select(&trainer, &mut vocabulary, &mut corpus, &mut index)
            .unwrap()
            .unwrap();
        assert_eq!(
            batch.trace().collect::<Vec<_>>(),
            vec![((0, 1), 10, 6), ((2, 3), 9, 7), ((4, 5), 8, 8)]
        );
        let changes = batch.prepare(&corpus, usize::MAX).unwrap().apply(&corpus);
        index.commit(changes).unwrap();
        assert!(index.best().is_none());
    }
}
