//! Stable original-symbol coordinates for word-local training edits.

use super::positions::Positions;
use super::word::Word;
use crate::progress::ProgressBar;
use ahash::AHashMap;
use std::ops::Range;
use tk_encode::Result;
use tk_encode::models::bpe::Pair;
use tk_encode::parallelism::MaybeParallelRefIterator;

const EMPTY: u32 = u32::MAX;

#[derive(Clone, Copy)]
struct Symbol {
    id: u32,
    span: usize,
}

pub(super) struct Corpus {
    symbols: Vec<Symbol>,
    starts: Vec<usize>,
    counts: Vec<u64>,
    whole_words: bool,
}

pub(super) struct WordChange {
    pub(super) pair: Pair,
    pub(super) added: bool,
    pub(super) coordinate: u64,
    pub(super) weight: u64,
}

impl Corpus {
    pub(super) fn from_words(words: Vec<(Word, u64)>) -> Self {
        let total = words.iter().map(|(word, _)| word.len()).sum::<usize>();
        let mut symbols = Vec::with_capacity(total + words.len() + 1);
        symbols.push(Symbol { id: EMPTY, span: 0 });
        let mut starts = Vec::with_capacity(words.len());
        let mut counts = Vec::with_capacity(words.len());
        for (word, weight) in words {
            starts.push(symbols.len());
            counts.push(weight);
            symbols.extend(word.into_ids().map(|id| Symbol { id, span: 1 }));
            symbols.push(Symbol { id: EMPTY, span: 0 });
        }

        Self {
            symbols,
            starts,
            counts,
            whole_words: false,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.starts.len()
    }

    #[allow(clippy::type_complexity)]
    pub(super) fn count_pairs(
        &self,
        progress: &Option<ProgressBar>,
    ) -> Result<(AHashMap<Pair, u64>, AHashMap<Pair, Vec<u64>>)> {
        self.starts
            .maybe_par_iter()
            .enumerate()
            .map(|(i, _)| {
                let mut pair_counts = AHashMap::new();
                let mut where_to_update: AHashMap<Pair, Vec<u64>> = AHashMap::new();

                let chars: Vec<_> = read_symbols(&self.symbols[self.word_range(i)]).collect();
                for (offset, window) in chars.windows(2).enumerate() {
                    let pair: Pair = (window[0], window[1]);
                    let count = pair_counts.entry(pair).or_insert(0_u64);
                    *count = count
                        .checked_add(self.counts[i])
                        .ok_or("BPE pair count exceeds u64")?;
                    where_to_update
                        .entry(pair)
                        .or_default()
                        .push((self.starts[i] + offset) as u64);
                }

                if let Some(progress) = progress {
                    progress.inc(1);
                }

                Ok((pair_counts, where_to_update))
            })
            .reduce(
                || Ok((AHashMap::new(), AHashMap::new())),
                |counts, positions| {
                    let (mut pair_counts, mut where_to_update) = counts?;
                    let (pc, wtu) = positions?;
                    for (pair, count) in pc {
                        let total = pair_counts.entry(pair).or_insert(0_u64);
                        *total = total
                            .checked_add(count)
                            .ok_or("BPE pair count exceeds u64")?;
                    }
                    for (pair, word_positions) in wtu {
                        where_to_update
                            .entry(pair)
                            .or_insert_with(Vec::new)
                            .extend(word_positions);
                    }
                    Ok((pair_counts, where_to_update))
                },
            )
    }

    pub(super) fn merge(
        &mut self,
        pair: Pair,
        id: u32,
        limit: usize,
        positions: &Positions,
        existing_identity: bool,
    ) -> Vec<WordChange> {
        self.whole_words |= existing_identity;
        if self.whole_words {
            self.merge_whole_words(pair, id, limit, positions)
        } else {
            self.merge_local(pair, id, limit, positions)
        }
    }

    fn merge_whole_words(
        &mut self,
        pair: Pair,
        id: u32,
        limit: usize,
        positions: &Positions,
    ) -> Vec<WordChange> {
        let mut changes = Vec::new();
        let mut previous_word = None;
        for coordinate in positions.iter() {
            let position = usize::try_from(coordinate).expect("resident corpus coordinate");
            let word = self.word_for(position);
            if previous_word == Some(word) {
                continue;
            }
            previous_word = Some(word);

            let range = self.word_range(word);
            let local_changes = merge_symbols(&mut self.symbols[range.clone()], pair, id, limit);
            for (pair, added, coordinate) in local_changes {
                changes.push(WordChange {
                    pair,
                    added,
                    coordinate: (range.start + coordinate) as u64,
                    weight: self.counts[word],
                });
            }
        }
        changes
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
            let position = usize::try_from(coordinate).expect("resident corpus coordinate");
            let Some(left) = self.symbols.get(position).copied() else {
                continue;
            };
            if left.id != pair.0 || left.span == 0 {
                continue;
            }
            let right_position = position + left.span;
            let Some(right) = self.symbols.get(right_position).copied() else {
                continue;
            };
            if right.id != pair.1 || right.span == 0 {
                continue;
            }
            let span = left.span + right.span;
            let after = position + span;
            let word = self.word_for(position);
            let word_start = self.word_range(word).start;
            let weight = self.counts[word];

            if position > word_start {
                let before = position - 1;
                let neighbor = self.symbols[before];
                if neighbor.id != EMPTY && neighbor.span != 0 {
                    let neighbor_start = position - neighbor.span;
                    changes.push(WordChange {
                        pair: (neighbor.id, pair.0),
                        added: false,
                        coordinate: neighbor_start as u64,
                        weight,
                    });
                    if neighbor.span + span < limit {
                        changes.push(WordChange {
                            pair: (neighbor.id, id),
                            added: true,
                            coordinate: neighbor_start as u64,
                            weight,
                        });
                    }
                }
            }

            let next = self
                .symbols
                .get(after)
                .copied()
                .unwrap_or(Symbol { id: EMPTY, span: 0 });
            if next.id != EMPTY && next.span != 0 {
                changes.push(WordChange {
                    pair: (pair.1, next.id),
                    added: false,
                    coordinate: right_position as u64,
                    weight,
                });
                if span + next.span < limit {
                    changes.push(WordChange {
                        pair: (id, next.id),
                        added: true,
                        coordinate,
                        weight,
                    });
                }
            }

            self.symbols[position] = Symbol { id, span };
            if right_position + 1 != after {
                self.symbols[right_position] = Symbol { id: EMPTY, span: 0 };
            }
            self.symbols[after - 1] = Symbol { id, span };
        }
        changes
    }

    fn word_for(&self, position: usize) -> usize {
        self.starts.partition_point(|&start| start <= position) - 1
    }

    fn word_range(&self, word: usize) -> Range<usize> {
        let start = self.starts[word];
        let end = self
            .starts
            .get(word + 1)
            .copied()
            .unwrap_or(self.symbols.len());
        start..end - 1
    }
}

fn merge_symbols(
    symbols: &mut [Symbol],
    pair: Pair,
    id: u32,
    limit: usize,
) -> Vec<(Pair, bool, usize)> {
    let mut changes = Vec::new();
    let mut previous = None;
    let mut position = 0;
    while position < symbols.len() {
        let left = symbols[position];
        let right_position = position + left.span;
        if left.id != pair.0
            || right_position >= symbols.len()
            || symbols[right_position].id != pair.1
            || symbols[right_position].span == 0
        {
            previous = Some(position);
            position = right_position;
            continue;
        }

        let right = symbols[right_position];
        let span = left.span + right.span;
        let after = position + span;
        if let Some(before) = previous {
            let neighbor = symbols[before];
            changes.push(((neighbor.id, pair.0), false, before));
            if neighbor.span + span < limit {
                changes.push(((neighbor.id, id), true, before));
            }
        }

        symbols[position] = Symbol { id, span };
        if right_position + 1 != after {
            symbols[right_position] = Symbol { id: EMPTY, span: 0 };
        }
        symbols[after - 1] = Symbol { id, span };
        if after < symbols.len() {
            let neighbor = symbols[after];
            changes.push(((pair.1, neighbor.id), false, right_position));
            if span + neighbor.span < limit {
                changes.push(((id, neighbor.id), true, position));
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
