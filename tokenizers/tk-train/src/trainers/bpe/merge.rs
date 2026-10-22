//! Joined merge preparation and disjoint endpoint writes.
use super::{Corpus, EMPTY, Positions, Symbol, WordChange, merge_symbols};
use itertools::Itertools;
use tk_encode::models::bpe::Pair;
use tk_encode::parallelism::MaybeParallelIterator;

impl Corpus {
    pub(in crate::trainers::bpe) fn merge(
        &mut self,
        pair: Pair,
        id: u32,
        limit: usize,
        positions: &Positions,
        existing_identity: bool,
    ) -> Vec<WordChange> {
        // Existing identities conservatively retain historical whole-word semantics.
        self.whole_words |= existing_identity;
        if !self.whole_words {
            return self.merge_local(pair, id, limit, positions);
        }

        struct SlotPtr(*mut Symbol);
        // Candidate word IDs are unique; their fixed slot ranges never overlap.
        unsafe impl Sync for SlotPtr {}
        let pointer = SlotPtr(self.symbols.as_mut_ptr());
        let total = self.symbols.len();
        let starts = &self.starts;
        let weights = &self.counts;
        // Historical coordinates choose word domains even when their old match is stale.
        let words: Vec<_> = positions
            .iter()
            .map(|coordinate| {
                let position = usize::try_from(coordinate).expect("resident corpus coordinate");
                assert!(position < total);
                starts.partition_point(|&start| start <= position) - 1
            })
            .dedup()
            .collect();
        words
            .into_maybe_par_iter()
            .flat_map(|word| {
                let pointer = &pointer;
                assert!(word < starts.len());
                let begin = starts[word];
                let end = starts.get(word + 1).copied().unwrap_or(total) - 1;
                // SAFETY: only this task owns [begin,end); storage remains fixed until join.
                let symbols =
                    unsafe { std::slice::from_raw_parts_mut(pointer.0.add(begin), end - begin) };
                merge_symbols(symbols, pair.0, pair.1, id, limit)
                    .into_iter()
                    .map(|(pair, delta, coordinate)| WordChange {
                        pair,
                        delta,
                        coordinate: (begin + coordinate) as u64,
                        weight: weights[word],
                    })
                    .collect::<Vec<_>>()
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
        let mut changes = Vec::new();
        for coordinate in positions.iter() {
            let p = usize::try_from(coordinate).expect("resident corpus coordinate");
            let left = self.symbols[p];
            let right = p + left.span;
            if left.id != pair.0
                || left.span == 0
                || right >= self.symbols.len()
                || self.symbols[right].id != pair.1
            {
                continue;
            }
            let span = left.span + self.symbols[right].span;
            let after = p + span;
            let word = self.starts.partition_point(|&start| start <= p) - 1;
            let weight = self.counts[word];
            let mut record = |pair, delta, coordinate: usize| {
                changes.push(WordChange {
                    pair,
                    delta,
                    coordinate: coordinate as u64,
                    weight,
                })
            };
            let prior = self.symbols[p - 1];
            if prior.id != EMPTY {
                let before = p - prior.span;
                record((prior.id, pair.0), -1, before);
                if prior.span + span < limit {
                    record((prior.id, id), 1, before);
                }
            }
            let next = self.symbols[after];
            if next.id != EMPTY {
                record((pair.1, next.id), -1, right);
                if span + next.span < limit {
                    record((id, next.id), 1, p);
                }
            }
            self.symbols[p] = Symbol { id, span };
            if right + 1 != after {
                self.symbols[right] = Symbol { id: EMPTY, span: 0 };
            }
            self.symbols[after - 1] = Symbol { id, span };
        }
        changes
    }
}
