//! Joined merge preparation and disjoint endpoint writes.
use super::{Corpus, EMPTY, Positions, Symbol, WordChange};
use itertools::Itertools;
use tk_encode::models::bpe::Pair;
use tk_encode::parallelism::{MaybeParallelIterator, MaybeParallelSlice};
/// Read-only preparation owns accepted starts and the old sequential edge events.
struct PreparedChunk {
    starts: Vec<usize>,
    changes: Vec<WordChange>,
}
#[cfg(test)]
mod preparation_tests {
    use super::*;
    #[test]
    fn adjacent_matches_prepare_sequential_events_without_writes() {
        let mut corpus = Corpus::fixture(&[EMPTY, 0, 1, 0, 1, EMPTY], vec![1], vec![3]);
        let positions = Positions::from_unsorted(vec![1, 3]).unwrap();
        let prepared = corpus.prepare_local((0, 1), 2, usize::MAX, positions.iter());
        assert_eq!(prepared.starts, [1, 3]);
        assert_eq!(
            prepared
                .changes
                .iter()
                .map(|c| (c.pair, c.delta, c.coordinate, c.weight))
                .collect::<Vec<_>>(),
            [
                ((1, 0), -1, 2, 3),
                ((2, 0), 1, 1, 3),
                ((2, 0), -1, 1, 3),
                ((2, 2), 1, 1, 3)
            ]
        );
        assert_eq!(corpus.symbol(1).id, 0);
        corpus.activate((0, 1), 2);
        corpus.merge_local((0, 1), 2, usize::MAX, &positions);
        assert_eq!(corpus.read_word(1..5).collect::<Vec<_>>(), [2, 2]);
    }
}
impl Corpus {
    pub(in crate::trainers::bpe) fn merge(
        &mut self,
        pair: Pair,
        id: u32,
        limit: usize,
        positions: &Positions,
        existing_identity: bool,
    ) -> Vec<WordChange> {
        if existing_identity {
            self.whole_words |= self.has_been_activated(id);
        }
        self.activate(pair, id);
        if !self.whole_words {
            return self.merge_local(pair, id, limit, positions);
        }

        // Historical coordinates select domains before stale-match filtering.
        let words: Vec<_> = positions
            .iter()
            .map(|coordinate| {
                let position = usize::try_from(coordinate).expect("resident corpus coordinate");
                assert!(position < self.tokens.len());
                self.starts.partition_point(|&start| start <= position) - 1
            })
            .dedup()
            .collect();
        let prepared: Vec<_> = words
            .into_maybe_par_iter()
            .map(|word| self.prepare_word(word, pair, id, limit))
            .collect();
        prepared
            .into_maybe_par_iter()
            .flat_map(|chunk| {
                self.apply(&chunk.starts, id);
                chunk.changes
            })
            .collect()
    }
    fn merge_local(
        &mut self,
        pair: Pair,
        id: u32,
        limit: usize,
        positions: &Positions,
    ) -> Vec<WordChange> {
        let prepared = self.prepare_local(pair, id, limit, positions.iter());
        let chunk = if cfg!(test) { 1 } else { 4096 };
        prepared
            .starts
            .maybe_par_chunks(chunk)
            .for_each(|starts| self.apply(starts, id));
        prepared.changes
    }
    fn prepare_local(
        &self,
        pair: Pair,
        id: u32,
        limit: usize,
        coordinates: impl IntoIterator<Item = u64>,
    ) -> PreparedChunk {
        let mut prepared = PreparedChunk {
            starts: Vec::new(),
            changes: Vec::new(),
        };
        let mut previous_end = 0;
        let mut previous_span = 0;
        for coordinate in coordinates {
            let p = usize::try_from(coordinate).expect("resident corpus coordinate");
            if p < previous_end {
                continue;
            }
            let left = self.symbol(p);
            let right = p + left.span;
            if left.id != pair.0
                || left.span == 0
                || right >= self.tokens.len()
                || self.symbol(right).id != pair.1
            {
                continue;
            }
            let span = left.span + self.symbol(right).span;
            let after = p + span;
            let weight = self.counts[self.starts.partition_point(|&start| start <= p) - 1];
            let mut record = |pair, delta, coordinate: usize| {
                prepared.changes.push(WordChange {
                    pair,
                    delta,
                    coordinate: coordinate as u64,
                    weight,
                })
            };
            // The preceding accepted match has already merged in the sequential
            // contract. Read its virtual endpoint without mutating the snapshot.
            let prior = if p == previous_end {
                Symbol {
                    id,
                    span: previous_span,
                }
            } else {
                self.symbol(p - 1)
            };
            if prior.id != EMPTY {
                let before = p - prior.span;
                record((prior.id, pair.0), -1, before);
                if prior.span + span < limit {
                    record((prior.id, id), 1, before);
                }
            }
            let next = self.symbol(after);
            if next.id != EMPTY {
                record((pair.1, next.id), -1, right);
                if span + next.span < limit {
                    record((id, next.id), 1, p);
                }
            }
            prepared.starts.push(p);
            previous_end = after;
            previous_span = span;
        }
        prepared
    }
    fn prepare_word(&self, word: usize, pair: Pair, id: u32, limit: usize) -> PreparedChunk {
        let mut prepared = PreparedChunk {
            starts: Vec::new(),
            changes: Vec::new(),
        };
        let mut previous: Option<(usize, Symbol)> = None;
        let range = self.word_range(word);
        let mut p = range.start;
        while p < range.end {
            let left = self.symbol(p);
            let right = p + left.span;
            if left.id != pair.0 || right >= range.end || self.symbol(right).id != pair.1 {
                previous = Some((p, left));
                p = right;
                continue;
            }
            let span = left.span + self.symbol(right).span;
            let after = p + span;
            let weight = self.counts[word];
            let mut record = |pair, delta, coordinate: usize| {
                prepared.changes.push(WordChange {
                    pair,
                    delta,
                    coordinate: coordinate as u64,
                    weight,
                })
            };
            if let Some((before, prior)) = previous {
                record((prior.id, pair.0), -1, before);
                if prior.span + span < limit {
                    record((prior.id, id), 1, before);
                }
            }
            if after < range.end {
                let next = self.symbol(after);
                record((pair.1, next.id), -1, right);
                if span + next.span < limit {
                    record((id, next.id), 1, p);
                }
            }
            prepared.starts.push(p);
            previous = Some((p, Symbol { id, span }));
            p = after;
        }
        prepared
    }
}
