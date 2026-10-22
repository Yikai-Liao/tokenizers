//! Joined merge preparation and disjoint endpoint writes.
use super::changes::{EdgeSide, NeighborChanges, NeighborScratch};
use super::{Corpus, EMPTY, PairChange, Positions, ProducerChanges, SelectedMerge, TrainingMode};
use itertools::Itertools;
use smallvec::SmallVec;
use std::ops::Range;
use tk_encode::Result;
use tk_encode::models::bpe::Pair;
use tk_encode::parallelism::{MaybeParallelIterator, MaybeParallelSlice};
/// Prepared disjoint writes and checked neighboring pair groups.
struct PreparedChunk {
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
        let mut scratch = NeighborScratch::default();
        let prepared = corpus
            .prepare_local(
                (0, 1),
                2,
                usize::MAX,
                positions.iter(),
                None,
                Some(&positions),
                None,
                &mut scratch,
            )
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
            .merge_local((0, 1), 2, usize::MAX, &positions)
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
            let changes = corpus
                .merge(SelectedMerge {
                    pair: (0, 1),
                    merged_id: 2,
                    positions: make(),
                })
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
impl Corpus {
    pub(in crate::trainers::bpe) fn merge(
        &mut self,
        rule: SelectedMerge,
    ) -> Result<Vec<ProducerChanges>> {
        let pair = rule.pair;
        let id = rule.merged_id;
        let positions = &rule.positions;
        let limit = self.birth_pair_span_limit.unwrap_or(usize::MAX);
        if self.mode == TrainingMode::Fresh {
            return self.merge_local(pair, id, limit, &positions);
        }

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
        Ok(prepared
            .into_maybe_par_iter()
            .map(|chunk| {
                self.apply(&chunk.starts, id);
                ProducerChanges {
                    whole_rule: false,
                    changes: chunk.changes,
                }
            })
            .collect())
    }
    fn merge_local(
        &mut self,
        pair: Pair,
        id: u32,
        limit: usize,
        positions: &Positions,
    ) -> Result<Vec<ProducerChanges>> {
        let chunk = if cfg!(test) {
            1
        } else {
            positions
                .len()
                .div_ceil(super::super::num_workers())
                .clamp(1, 1 << 26)
        };
        if pair.0 != pair.1 {
            let tasks: Vec<_> = positions.chunks(chunk).collect();
            let complete = tasks.len() == 1;
            let prepared: Vec<_> = tasks
                .into_maybe_par_iter()
                .map(|(previous, coordinates)| {
                    let coordinates = coordinates.skip_while(move |&p| previous == Some(p));
                    self.neighbor_scratch.with(|scratch| {
                        self.prepare_local(
                            pair,
                            id,
                            limit,
                            coordinates,
                            None,
                            Some(positions),
                            None,
                            scratch,
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            return Ok(prepared
                .into_maybe_par_iter()
                .map(|prepared| {
                    self.apply(&prepared.starts, id);
                    ProducerChanges {
                        whole_rule: complete,
                        changes: prepared.changes,
                    }
                })
                .collect());
        }
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
                accepted.push(matched);
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
        let prepared: Vec<_> = accepted
            .maybe_par_chunks(chunk)
            .enumerate()
            .map(|(part, matches)| {
                let seed = if part == 0 {
                    None
                } else {
                    let prior = &accepted[part * chunk - 1];
                    Some(prior.end)
                };
                self.neighbor_scratch.with(|scratch| {
                    self.prepare_local(
                        pair,
                        id,
                        limit,
                        matches.iter().map(|matched| matched.start as u64),
                        seed,
                        None,
                        Some(&accepted),
                        scratch,
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(prepared
            .into_maybe_par_iter()
            .map(|prepared| {
                self.apply(&prepared.starts, id);
                ProducerChanges {
                    whole_rule: false,
                    changes: prepared.changes,
                }
            })
            .collect())
    }
    fn prepare_local(
        &self,
        pair: Pair,
        id: u32,
        limit: usize,
        coordinates: impl IntoIterator<Item = u64>,
        previous: Option<usize>,
        cohort: Option<&Positions>,
        accepted: Option<&[Range<usize>]>,
        scratch: &mut NeighborScratch,
    ) -> Result<PreparedChunk> {
        let mut starts = SmallVec::new();
        let mut neighbors = NeighborChanges::new(pair, id, self.id_metadata.len(), scratch);
        let mut previous_end = previous.unwrap_or(0);
        for (matched, weight) in self.fresh_matches(coordinates, pair) {
            let p = matched.start;
            if p < previous_end {
                continue;
            }
            let span = matched.len();
            let after = matched.end;
            // The preceding accepted match has already merged in the sequential
            // contract. Read its virtual endpoint without mutating the snapshot.
            let adjacent_selected = cohort.is_some_and(|positions| {
                let total = self.fixed_id_span(pair.0) + self.fixed_id_span(pair.1);
                p.checked_sub(total).is_some_and(|before| {
                    self.match_at(before, pair)
                        .is_some_and(|matched| matched.end == p)
                        && positions.iter_from(before as u64).next() == Some(before as u64)
                })
            });
            let prior = self.symbol(p - 1);
            if prior.id != EMPTY && p != previous_end && !adjacent_selected {
                neighbors.record(
                    EdgeSide::Left,
                    prior.id,
                    prior.id,
                    p - prior.span,
                    weight,
                    prior.span + span < limit,
                )?;
            }
            let next = self.symbol(after);
            if next.id != EMPTY {
                let selected = cohort.is_some_and(|positions| {
                    self.match_at(after, pair).is_some()
                        && positions.iter_from(after as u64).next() == Some(after as u64)
                }) || accepted.is_some_and(|matches| {
                    matches
                        .binary_search_by_key(&after, |matched| matched.start)
                        .is_ok()
                });
                let born = if selected { id } else { next.id };
                neighbors.record(
                    EdgeSide::Right,
                    next.id,
                    born,
                    p,
                    weight,
                    span + self.fixed_id_span(born) < limit,
                )?;
            }
            starts.push(p);
            previous_end = after;
        }
        Ok(PreparedChunk {
            starts,
            changes: neighbors.finish(),
        })
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
