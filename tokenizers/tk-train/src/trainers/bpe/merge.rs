//! Joined merge preparation and disjoint endpoint writes.
use super::changes::{EdgeSide, NeighborChanges, NeighborScratch};
use super::{Corpus, EMPTY, PairChange, Positions, ProducerChanges, SelectedMerge, TrainingMode};
use ahash::AHashMap;
use itertools::Itertools;
use smallvec::SmallVec;
use std::ops::Range;
use tk_encode::Result;
use tk_encode::models::bpe::Pair;
use tk_encode::parallelism::{MaybeParallelIterator, MaybeParallelSlice};
/// Prepared disjoint writes and checked neighboring pair groups.
struct PreparedChunk {
    merged_id: u32,
    whole_rule: bool,
    starts: SmallVec<[usize; 2]>,
    changes: Vec<PairChange<Positions>>,
}
#[cfg(test)]
mod preparation_tests {
    use super::*;
    #[test]
    fn adjacent_matches_prepare_final_boundaries_without_writes() {
        let mut corpus = Corpus::fixture(&[EMPTY, 0, 1, 0, 1, EMPTY], vec![1], vec![3]);
        corpus.activate((0, 1), 2);
        let positions = Positions::from_unsorted(vec![1, 3]).unwrap();
        let prepared = corpus
            .prepare_fresh(vec![SelectedMerge {
                pair: (0, 1),
                merged_id: 2,
                positions,
            }])
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(prepared.starts.as_slice(), [1, 3]);
        assert_eq!(
            prepared
                .changes
                .iter()
                .map(|c| (
                    c.removed,
                    c.born,
                    c.removed_weight,
                    c.born_weight,
                    c.positions.iter().collect::<Vec<_>>()
                ))
                .collect::<Vec<_>>(),
            [
                ((1, 0), (2, 0), 3, 0, vec![]),
                ((1, 2), (2, 2), 0, 3, vec![1])
            ]
        );
        assert_eq!(corpus.symbol(1).id, 0);
        corpus
            .merge(vec![SelectedMerge {
                pair: (0, 1),
                merged_id: 2,
                positions: Positions::from_unsorted(vec![1, 3]).unwrap(),
            }])
            .unwrap();
        assert_eq!(corpus.read_word(1..5).collect::<Vec<_>>(), [2, 2]);
    }
}
#[cfg(test)]
mod reader_duplicate_tests {
    use super::*;
    #[test]
    fn ab_duplicate_runs_cross_reader_cuts_once() {
        for split_leaf in [false, true] {
            let mut corpus = Corpus::fixture(&[EMPTY, 0, 1, 0, 1, EMPTY], vec![1], vec![1]);
            let make = || {
                let mut values = vec![1; 257];
                values.push(3);
                if split_leaf {
                    Positions::concat([
                        Positions::from_unsorted(values[..128].to_vec()).unwrap(),
                        Positions::from_unsorted(values[128..].to_vec()).unwrap(),
                    ])
                    .unwrap()
                } else {
                    Positions::from_unsorted(values).unwrap()
                }
            };
            corpus.activate((0, 1), 2);
            let prepared = corpus
                .prepare_fresh(vec![SelectedMerge {
                    pair: (0, 1),
                    merged_id: 2,
                    positions: make(),
                }])
                .unwrap();
            assert_eq!(
                prepared
                    .iter()
                    .flat_map(|chunk| chunk.starts.iter().copied())
                    .collect::<Vec<_>>(),
                [1, 3]
            );
            let changes = corpus
                .merge(vec![SelectedMerge {
                    pair: (0, 1),
                    merged_id: 2,
                    positions: make(),
                }])
                .unwrap();
            let total = changes
                .iter()
                .flat_map(|p| &p.changes)
                .fold((0, 0), |a, c| {
                    (
                        a.0 + if c.removed == (1, 0) {
                            c.removed_weight
                        } else {
                            0
                        },
                        a.1 + if c.born == (2, 2) { c.born_weight } else { 0 },
                    )
                });
            assert_eq!(total, (1, 1));
            assert_eq!(corpus.read_word(1..5).collect::<Vec<_>>(), [2, 2]);
            assert_eq!(
                (corpus.occurrence_span(1), corpus.occurrence_span(3)),
                (2, 2)
            );
        }
    }
}
#[derive(Clone, Copy)]
struct Neighbor {
    token_id: u32,
    start: usize,
    span: usize,
}
struct FreshPreparation<'corpus> {
    corpus: &'corpus Corpus,
    birth_pair_span_limit: usize,
}
impl FreshPreparation<'_> {
    fn prepare(
        &self,
        pair: Pair,
        merged_id: u32,
        positions: impl IntoIterator<Item = u64>,
        scratch: &mut NeighborScratch,
        left_merging: impl Fn(&Range<usize>, usize, u32) -> bool,
        right_merged_id: impl Fn(usize, u32) -> Option<u32>,
    ) -> Result<PreparedChunk> {
        self.corpus
            .prepare_chunk(pair, merged_id, scratch, |neighbors, starts| {
                // Borrow decoder state through the lookahead adapters.
                let mut positions = positions.into_iter().dedup();
                for (matched, weight) in self.corpus.fresh_matches(&mut positions, pair) {
                    self.record(neighbors, &matched, weight, &left_merging, &right_merged_id)?;
                    starts.push(matched.start);
                }
                Ok(())
            })
    }

    // The left match owns each shared edge and uses both final IDs. A later
    // match skips that left edge, even across rule or task boundaries.
    fn record(
        &self,
        neighbors: &mut NeighborChanges<'_>,
        matched: &Range<usize>,
        weight: u64,
        left_merging: &impl Fn(&Range<usize>, usize, u32) -> bool,
        right_merged_id: &impl Fn(usize, u32) -> Option<u32>,
    ) -> Result<()> {
        let p = matched.start;
        let prior = self.corpus.token(p - 1);
        if prior != EMPTY {
            let span = self.corpus.fixed_id_span(prior);
            let before = p - span;
            if !left_merging(matched, before, prior) {
                neighbors.record(
                    EdgeSide::Left,
                    prior,
                    prior,
                    before,
                    weight,
                    // Admit only a neighbor pair strictly shorter than the
                    // limit, using prior-token span plus merged-match span.
                    span + matched.len() < self.birth_pair_span_limit,
                )?;
            }
        }
        let next = self.corpus.token(matched.end);
        if next != EMPTY {
            let born = right_merged_id(matched.end, next).unwrap_or(next);
            neighbors.record(
                EdgeSide::Right,
                next,
                born,
                p,
                weight,
                // A selected right neighbor may also merge: use its final
                // ID's span in the strict combined-length gate.
                matched.len() + self.corpus.fixed_id_span(born) < self.birth_pair_span_limit,
            )?;
        }
        Ok(())
    }
}
#[derive(Default)]
pub(super) struct SelectedRuleIndex {
    heads: Vec<u64>,
    tails: Vec<u64>,
    multiple: AHashMap<Pair, u32>,
    pairs: Vec<Pair>,
}
impl SelectedRuleIndex {
    const EMPTY: u64 = u64::MAX;
    const MULTIPLE: u64 = u64::MAX - 1;

    fn reset(&mut self, rules: &[SelectedMerge], domain: usize) {
        for (head, tail) in self.pairs.drain(..) {
            self.heads[head as usize] = Self::EMPTY;
            self.tails[tail as usize] = Self::EMPTY;
        }
        self.heads.resize(domain, Self::EMPTY);
        self.tails.resize(domain, Self::EMPTY);
        self.multiple.clear();
        for rule in rules {
            self.pairs.push(rule.pair);
            let head = &mut self.heads[rule.pair.0 as usize];
            *head = if *head == Self::EMPTY {
                (u64::from(rule.pair.1) << 32) | u64::from(rule.merged_id)
            } else {
                Self::MULTIPLE
            };
            let tail = &mut self.tails[rule.pair.1 as usize];
            *tail = if *tail == Self::EMPTY {
                (u64::from(rule.pair.0) << 32) | u64::from(rule.merged_id)
            } else {
                Self::MULTIPLE
            };
        }
        for rule in rules {
            if self.heads[rule.pair.0 as usize] == Self::MULTIPLE
                || self.tails[rule.pair.1 as usize] == Self::MULTIPLE
            {
                self.multiple.insert(rule.pair, rule.merged_id);
            }
        }
    }

    fn left_selected(&self, corpus: &Corpus, before: usize, prior: u32) -> bool {
        let tail = self.tails[prior as usize];
        if before == 0 || tail == Self::EMPTY {
            return false;
        }
        let previous = corpus.token(before - 1);
        if tail == Self::MULTIPLE {
            self.multiple.contains_key(&(previous, prior))
        } else {
            (tail >> 32) as u32 == previous
        }
    }

    fn final_next(&self, corpus: &Corpus, after: usize, next: u32) -> Option<u32> {
        let head = self.heads[next as usize];
        if head == Self::EMPTY {
            return None;
        }
        let following = corpus.token(after + corpus.fixed_id_span(next));
        if head == Self::MULTIPLE {
            self.multiple.get(&(next, following)).copied()
        } else {
            ((head >> 32) as u32 == following).then_some(head as u32)
        }
    }
}
impl Corpus {
    /// Preparation errors leave endpoints unchanged; all writers join before return.
    /// Index errors after writes discard the attempt without local rollback.
    pub(in crate::trainers::bpe) fn merge(
        &mut self,
        selected: Vec<SelectedMerge>,
    ) -> Result<Vec<ProducerChanges>> {
        let prepared = match self.mode {
            TrainingMode::Fresh => self.prepare_fresh(selected)?,
            TrainingMode::Reuse => {
                debug_assert_eq!(selected.len(), 1);
                self.prepare_reuse(selected.into_iter().next().expect("reuse selects one rule"))?
            }
        };
        Ok(prepared
            .into_maybe_par_iter()
            .map(|chunk| {
                self.apply(&chunk.starts, chunk.merged_id);
                ProducerChanges {
                    whole_rule: chunk.whole_rule,
                    changes: chunk.changes,
                }
            })
            .collect())
    }
    fn prepare_reuse(&self, rule: SelectedMerge) -> Result<Vec<PreparedChunk>> {
        let pair = rule.pair;
        let id = rule.merged_id;
        let positions = &rule.positions;
        // Historical coordinates select domains before stale-match filtering.
        let words: Vec<_> = positions
            .iter()
            .map(|coordinate| {
                let position = self.resident(coordinate);
                self.starts.partition_point(|&start| start <= position) - 1
            })
            .dedup()
            .collect();
        let chunk = words.len().div_ceil(super::super::num_workers()).max(1);
        let prepared = words
            .maybe_par_chunks(chunk)
            .map(|words| {
                self.neighbor_scratch.with(|scratch| {
                    self.prepare_chunk(pair, id, scratch, |neighbors, starts| {
                        for &word in words {
                            self.scan_word(&rule, word, neighbors, starts)?;
                        }
                        Ok(())
                    })
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(prepared)
    }
    fn prepare_fresh(&mut self, rules: Vec<SelectedMerge>) -> Result<Vec<PreparedChunk>> {
        if rules[0].pair.0 != rules[0].pair.1 {
            self.selected_rules.reset(&rules, self.id_count());
            let preparation = FreshPreparation {
                corpus: self,
                birth_pair_span_limit: self.birth_pair_span_limit.unwrap_or(usize::MAX),
            };
            let selected = &self.selected_rules;
            let total: usize = rules.iter().map(|rule| rule.positions.len()).sum();
            let chunk = if cfg!(test) {
                1
            } else {
                total
                    .div_ceil(super::super::num_workers())
                    .clamp(1, 1 << 26)
            };
            let mut tasks = Vec::new();
            for rule in &rules {
                let mut parts = rule.positions.chunks(chunk).peekable();
                if let Some(first) = parts.next() {
                    let complete = parts.peek().is_none();
                    tasks.push((rule.pair, rule.merged_id, first, complete));
                    tasks.extend(
                        parts.map(|positions| (rule.pair, rule.merged_id, positions, false)),
                    );
                }
            }
            return tasks
                .into_maybe_par_iter()
                .map(|(pair, id, (previous, coordinates), complete)| {
                    let coordinates = coordinates.skip_while(move |&p| previous == Some(p));
                    self.neighbor_scratch.with(|scratch| {
                        let mut prepared = preparation.prepare(
                            pair,
                            id,
                            coordinates,
                            scratch,
                            |_, before, prior| selected.left_selected(self, before, prior),
                            |after, next| selected.final_next(self, after, next),
                        )?;
                        prepared.whole_rule = complete;
                        Ok(prepared)
                    })
                })
                .collect();
        }
        debug_assert_eq!(rules.len(), 1);
        let SelectedMerge {
            pair,
            merged_id: id,
            positions,
        } = rules.into_iter().next().expect("AA selects one rule");
        let preparation = FreshPreparation {
            corpus: self,
            birth_pair_span_limit: self.birth_pair_span_limit.unwrap_or(usize::MAX),
        };
        // Decide overlaps globally before fragmenting this rule's immutable snapshot.
        let mut accepted = Vec::new();
        let mut end = 0;
        let matcher = self.fixed_matcher(pair);
        for coordinate in positions.iter() {
            let p = self.resident(coordinate);
            if p >= end
                && let Some(matched) = matcher(p)
            {
                end = matched.end;
                accepted.push(coordinate);
            }
        }
        let chunk = if cfg!(test) {
            1
        } else {
            positions
                .len()
                .div_ceil(super::super::num_workers() * 4)
                .max(4096)
        };
        drop(positions);
        let prepared: Vec<_> = accepted
            .maybe_par_chunks(chunk)
            .map(|matches| {
                self.neighbor_scratch.with(|scratch| {
                    preparation.prepare(
                        pair,
                        id,
                        matches.iter().copied(),
                        scratch,
                        |matched, _, _| {
                            matched.start >= matched.len()
                                && accepted
                                    .binary_search(&((matched.start - matched.len()) as u64))
                                    .is_ok()
                        },
                        |after, _| {
                            accepted
                                .binary_search(&(after as u64))
                                .is_ok()
                                .then_some(id)
                        },
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(prepared)
    }
    fn whole_words(&self) -> bool {
        self.whole_words
    }
    fn prepare_chunk(
        &self,
        pair: Pair,
        merged_id: u32,
        scratch: &mut NeighborScratch,
        scan: impl FnOnce(&mut NeighborChanges<'_>, &mut SmallVec<[usize; 2]>) -> Result<()>,
    ) -> Result<PreparedChunk> {
        let mut neighbors = NeighborChanges::new(pair, merged_id, self.id_count(), scratch);
        let mut starts = SmallVec::new();
        scan(&mut neighbors, &mut starts)?;
        Ok(PreparedChunk {
            merged_id,
            whole_rule: false,
            starts,
            changes: neighbors.finish(),
        })
    }
    fn scan_word(
        &self,
        rule: &SelectedMerge,
        word: usize,
        neighbors: &mut NeighborChanges<'_>,
        starts: &mut SmallVec<[usize; 2]>,
    ) -> Result<()> {
        let limit = self.birth_pair_span_limit.unwrap_or(usize::MAX);
        let range = self.word_range(word);
        let weight = self.weight(range.start).weight;
        let mut p = range.start;
        let mut positions = rule.positions.iter_from(p as u64);
        let mut left_neighbor: Option<Neighbor> = None;
        while p < range.end {
            // Full scans are required after alias activation or any Some length gate.
            // Otherwise only eligible coordinates in this historical word are visited.
            if !self.whole_words() {
                let Some(coordinate) = positions.find(|&start| start >= p as u64) else {
                    break;
                };
                let next = self.resident(coordinate);
                if next >= range.end {
                    break;
                }
                // Before alias activation, consuming a historical left start
                // replaces its ID permanently. Reject stale starts before reading
                // next - 1: interior slots do not identify the current predecessor.
                if self.token(next) != rule.pair.0 {
                    continue;
                }
                if next != p {
                    // The slot just before a sought start is the preceding
                    // token's tail. Its span reconstructs that token's start.
                    let id = self.token(next - 1);
                    left_neighbor = (id != EMPTY).then(|| {
                        let span = self.occurrence_span(next - 1);
                        Neighbor {
                            token_id: id,
                            start: next - span,
                            span,
                        }
                    });
                    p = next;
                }
            }
            if let Some(matched) = self.match_at(p, rule.pair) {
                if let Some(left) = left_neighbor {
                    neighbors.record(
                        EdgeSide::Left,
                        left.token_id,
                        left.token_id,
                        left.start,
                        weight,
                        // Gate the new pair's combined occurrence length;
                        // equality is excluded by the strict birth-length limit.
                        left.span + matched.len() < limit,
                    )?;
                }
                let next = self.token(matched.end);
                if next != EMPTY {
                    neighbors.record(
                        EdgeSide::Right,
                        next,
                        next,
                        p,
                        weight,
                        // The right neighbor comes from the joined snapshot;
                        // its span plus this match is the new pair's length.
                        matched.len() + self.occurrence_span(matched.end) < limit,
                    )?;
                }
                left_neighbor = Some(Neighbor {
                    token_id: rule.merged_id,
                    start: matched.start,
                    span: matched.len(),
                });
                starts.push(matched.start);
                p = matched.end;
            } else {
                let span = self.occurrence_span(p);
                left_neighbor = Some(Neighbor {
                    token_id: self.token(p),
                    start: p,
                    span,
                });
                // Whole-word traversal and the surviving historical-head check
                // both keep p at a current token start with a nonzero span.
                p += span;
            }
        }
        Ok(())
    }
}
