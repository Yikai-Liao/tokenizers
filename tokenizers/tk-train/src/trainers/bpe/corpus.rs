//! Stable original-symbol coordinates for word-local training edits.
#[path = "merge.rs"]
mod merge;
use super::positions::Positions;
use super::vocabulary::InitialTokenIds;
use super::word_counts::WordCountsView;
use crate::progress::ProgressBar;
use ahash::AHashMap;
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
    whole_words: bool,
}

pub(super) struct WordChange {
    pub(super) pair: Pair,
    pub(super) delta: i32,
    pub(super) coordinate: u64,
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
            whole_words: false,
        }
    }

    pub(super) fn word_count(&self) -> usize {
        self.starts.len()
    }

    pub(super) fn count_pairs(
        &self,
        p: &Option<ProgressBar>,
    ) -> (AHashMap<Pair, i32>, AHashMap<Pair, Vec<u64>>) {
        self.starts
            .maybe_par_iter()
            .enumerate()
            .map(|(i, _)| {
                let mut pair_counts = AHashMap::new();
                let mut where_to_update: AHashMap<Pair, Vec<u64>> = AHashMap::new();

                let chars: Vec<_> = read_symbols(&self.symbols[self.word_range(i)]).collect();
                for (offset, window) in chars.windows(2).enumerate() {
                    let cur_pair: Pair = (window[0], window[1]);

                    // Initialize pair_counts and where_to_update for this pair if we just saw it
                    // Then update counts
                    *pair_counts.entry(cur_pair).or_default() += self.counts[i] as i32;
                    where_to_update
                        .entry(cur_pair)
                        .or_default()
                        .push((self.starts[i] + offset) as u64);
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
) -> Vec<(Pair, i32, usize)> {
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
            changes.push(((neighbor.id, c1), -1, before));
            if neighbor.span + span < limit {
                changes.push(((neighbor.id, id), 1, before));
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
            changes.push(((c2, neighbor.id), -1, right));
            if span + neighbor.span < limit {
                changes.push(((id, neighbor.id), 1, position));
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
