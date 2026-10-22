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

#[cfg(test)]
#[derive(Clone, Default)]
pub(super) struct Word {
    symbols: Vec<Symbol>,
}

#[cfg(test)]
impl Word {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn add(&mut self, id: u32) {
        self.symbols.push(Symbol { id, span: 1 });
    }

    pub(super) fn get_chars(&self) -> Vec<u32> {
        read_symbols(&self.symbols).collect()
    }

    pub(super) fn merge(&mut self, c1: u32, c2: u32, id: u32, limit: usize) -> Vec<(Pair, i32)> {
        merge_symbols(&mut self.symbols, c1, c2, id, limit)
            .into_iter()
            .map(|(pair, delta, _)| (pair, delta))
            .collect()
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge() {
        // Let's say we have the word 'hello' and a word-to-id vocab that looks
        // like this: {'h': 0, 'e': 1, 'l': 2, 'o': 3}.
        let mut word = Word::new();
        word.add(0); // 'h'
        word.add(1); // 'e'
        word.add(2); // 'l'
        word.add(2); // 'l'
        word.add(3); // 'o'

        // We're going to perform a merge on the pair ('l', 'l') ~= (2, 2). Let's
        // say that 'll' has the ID of 4 in the updated word-to-id vocab.
        let changes = word.merge(2, 2, 4, usize::MAX);

        // So the word should now look like this:
        assert_eq!(
            word.get_chars(),
            &[
                0u32, // 'h'
                1u32, // 'e'
                4u32, // 'll'
                3u32, // 'o'
            ]
        );

        // The return value `changes` will be used to update the pair counts during
        // training. This merge affects the counts for the pairs
        // ('e', 'l') ~= (1, 2),
        // ('e', 'll') ~= (1, 4),
        // ('l', 'o') ~= (2, 3), and
        // ('ll', 'o') ~= (4, 3).
        // So the changes should reflect that:
        assert_eq!(
            changes,
            &[
                ((1u32, 2u32), -1i32), // count for ('e', 'l') should be decreased by 1.
                ((1u32, 4u32), 1i32),  // count for ('e', 'll') should be increased by 1.
                ((2u32, 3u32), -1i32), // count for ('l', 'o') should be decreased by 1.
                ((4u32, 3u32), 1i32),  // count for ('ll', 'o') should be increased by 1.
            ]
        );
    }

    #[test]
    fn test_merge_max_length() {
        // Let's say we have the word 'hello' and a word-to-id vocab that looks
        // like this: {'h': 0, 'e': 1, 'l': 2, 'o': 3}.
        let mut word = Word::new();
        word.add(0); // 'h'
        word.add(1); // 'e'
        word.add(2); // 'l'
        word.add(2); // 'l'
        word.add(3); // 'o'

        // We're going to perform a merge on the pair ('l', 'l') ~= (2, 2). Let's
        // say that 'll' has the ID of 4 in the updated word-to-id vocab.
        let changes = word.merge(2, 2, 4, 2);
        assert_eq!(
            word.get_chars(),
            &[
                0u32, // 'h'
                1u32, // 'e'
                4u32, // 'll'
                3u32, // 'o'
            ]
        );

        assert_eq!(
            changes,
            &[
                ((1u32, 2u32), -1i32), // count for ('e', 'l') should be decreased by 1.
                // ((1u32, 4u32), 1i32),  Missing since this would be larger than 2
                ((2u32, 3u32), -1i32), // count for ('l', 'o') should be decreased by 1.
                                       // ((4u32, 3u32), 1i32), Missing since this would be larger than 2
            ]
        );
    }

    #[test]
    fn reused_id_retains_per_occurrence_geometry() {
        let mut word = Word::new();
        for id in [0, 0, 0, 0, 0] {
            word.add(id);
        }
        word.merge(0, 0, 0, usize::MAX);
        assert_eq!(word.get_chars(), [0, 0, 0]);
        word.merge(0, 0, 1, usize::MAX);
        assert_eq!(word.get_chars(), [1, 0]);
        assert_eq!((word.symbols[0].span, word.symbols[4].span), (4, 1));
    }

    #[test]
    fn adjacent_matches_preserve_intermediate_neighbor_events() {
        let mut word = Word::new();
        for id in [0, 1, 0, 1] {
            word.add(id);
        }
        assert_eq!(
            word.merge(0, 1, 2, usize::MAX),
            [((1, 0), -1), ((2, 0), 1), ((2, 0), -1), ((2, 2), 1)]
        );
        assert_eq!(word.get_chars(), [2, 2]);
    }
}
