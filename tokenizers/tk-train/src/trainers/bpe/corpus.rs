//! Stable original-symbol coordinates for word-local training edits.
use super::vocabulary::InitialTokenIds;
use super::word_counts::WordCountsView;
use crate::progress::ProgressBar;
use ahash::{AHashMap, AHashSet};
use std::ops::Range;
use tk_encode::models::bpe::Pair;
use tk_encode::parallelism::MaybeParallelRefIterator;

const EMPTY: u32 = u32::MAX;

#[derive(Clone, Copy)]
struct Symbol {
    id: u32,
    span: usize,
}

/// Owns mutable word storage and the frequency attached to each word domain.
pub(super) struct Corpus {
    symbols: Vec<Symbol>,
    starts: Vec<usize>,
    counts: Vec<u64>,
}

pub(super) struct WordChange {
    pub(super) pair: Pair,
    pub(super) delta: i32,
    pub(super) word: usize,
    pub(super) weight: u64,
}

impl Corpus {
    pub(super) fn new(words: WordCountsView<'_>, ids: &InitialTokenIds) -> Self {
        let mut symbols = vec![Symbol { id: EMPTY, span: 0 }];
        let mut starts = Vec::with_capacity(words.len());
        let mut counts = Vec::with_capacity(words.len());
        for (word, &weight) in words.iter() {
            starts.push(symbols.len());
            counts.push(weight);
            symbols.extend(ids.symbols(word).map(|id| Symbol { id, span: 1 }));
            symbols.push(Symbol { id: EMPTY, span: 0 });
        }
        Self {
            symbols,
            starts,
            counts,
        }
    }

    pub(super) fn word_count(&self) -> usize {
        self.starts.len()
    }

    pub(super) fn count_pairs(
        &self,
        p: &Option<ProgressBar>,
    ) -> (AHashMap<Pair, i32>, AHashMap<Pair, AHashSet<usize>>) {
        self.starts
            .maybe_par_iter()
            .enumerate()
            .map(|(i, _)| {
                let mut pair_counts = AHashMap::new();
                let mut where_to_update: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();

                let chars: Vec<_> = read_symbols(&self.symbols[self.word_range(i)]).collect();
                for window in chars.windows(2) {
                    let cur_pair: Pair = (window[0], window[1]);

                    // Initialize pair_counts and where_to_update for this pair if we just saw it
                    // Then update counts
                    *pair_counts.entry(cur_pair).or_default() += self.counts[i] as i32;
                    where_to_update.entry(cur_pair).or_default().insert(i);
                }

                if let Some(p) = &p {
                    p.inc(1);
                }

                (pair_counts, where_to_update)
            })
            .reduce(
                || (AHashMap::new(), AHashMap::new()),
                |(mut pair_counts, mut where_to_update), (pc, wtu)| {
                    for (k, v) in pc {
                        *pair_counts.entry(k).or_default() += v;
                    }
                    for (k, v) in wtu {
                        where_to_update.entry(k).or_default().extend(v);
                    }
                    (pair_counts, where_to_update)
                },
            )
    }

    pub(super) fn merge(
        &mut self,
        pair: Pair,
        id: u32,
        limit: usize,
        positions: &AHashSet<usize>,
    ) -> Vec<WordChange> {
        struct SlotPtr(*mut Symbol);
        // Candidate word IDs are unique; their fixed slot ranges never overlap.
        unsafe impl Sync for SlotPtr {}
        let pointer = SlotPtr(self.symbols.as_mut_ptr());
        let total = self.symbols.len();
        let starts = &self.starts;
        let weights = &self.counts;
        positions
            .maybe_par_iter()
            .flat_map(|&word| {
                let pointer = &pointer;
                assert!(word < starts.len());
                let begin = starts[word];
                let end = starts.get(word + 1).copied().unwrap_or(total) - 1;
                // SAFETY: only this task owns [begin,end); storage remains fixed until join.
                let symbols =
                    unsafe { std::slice::from_raw_parts_mut(pointer.0.add(begin), end - begin) };
                merge_symbols(symbols, pair.0, pair.1, id, limit)
                    .into_iter()
                    .map(|(pair, delta)| WordChange {
                        pair,
                        delta,
                        word,
                        weight: weights[word],
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }
    fn word_range(&self, word: usize) -> Range<usize> {
        self.starts[word]
            ..self
                .starts
                .get(word + 1)
                .copied()
                .unwrap_or(self.symbols.len())
                - 1
    }
}

fn merge_symbols(
    symbols: &mut [Symbol],
    c1: u32,
    c2: u32,
    id: u32,
    limit: usize,
) -> Vec<(Pair, i32)> {
    let mut changes = Vec::new();
    let mut previous: Option<usize> = None;
    let mut position = 0;
    while position < symbols.len() {
        let left = symbols[position];
        let right = position + left.span;
        if left.id != c1 || right >= symbols.len() || symbols[right].id != c2 {
            previous = Some(position);
            position = right;
            continue;
        }
        let span = left.span + symbols[right].span;
        let after = position + span;
        if let Some(before) = previous {
            let neighbor = symbols[before];
            changes.push(((neighbor.id, c1), -1));
            if neighbor.span + span < limit {
                changes.push(((neighbor.id, id), 1));
            }
        }
        // Heads and tails identify each logical token; spans skip interior holes.
        symbols[position] = Symbol { id, span };
        if right + 1 != after {
            symbols[right] = Symbol { id: EMPTY, span: 0 };
        }
        symbols[after - 1] = Symbol { id, span };
        if after < symbols.len() {
            let neighbor = symbols[after];
            changes.push(((c2, neighbor.id), -1));
            if span + neighbor.span < limit {
                changes.push(((id, neighbor.id), 1));
            }
        }
        previous = Some(position);
        position = after;
    }
    changes
}

fn read_symbols(symbols: &[Symbol]) -> impl Iterator<Item = u32> + '_ {
    let mut position = 0;
    std::iter::from_fn(move || {
        let symbol = *symbols.get(position)?;
        position += symbol.span;
        Some(symbol.id)
    })
}
