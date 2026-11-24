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

    pub(super) fn count_pairs(
        &self,
        progress: &Option<ProgressBar>,
    ) -> Result<AHashMap<Pair, (u64, Vec<u64>)>> {
        self.starts
            .maybe_par_iter()
            .enumerate()
            .map(|(i, _)| {
                let mut pairs = AHashMap::<Pair, (u64, Vec<u64>)>::new();

                let chars: Vec<_> = read_symbols(&self.symbols[self.word_range(i)]).collect();
                for (offset, window) in chars.windows(2).enumerate() {
                    let pair: Pair = (window[0], window[1]);
                    let (count, positions) = pairs.entry(pair).or_default();
                    *count = count
                        .checked_add(self.counts[i])
                        .ok_or("BPE pair count exceeds u64")?;
                    positions.push((self.starts[i] + offset) as u64);
                }

                if let Some(progress) = progress {
                    progress.inc(1);
                }

                Ok(pairs)
            })
            .reduce(
                || Ok(AHashMap::new()),
                |left, right| {
                    let mut pairs = left?;
                    for (pair, (count, positions)) in right? {
                        let (total, all_positions) = pairs.entry(pair).or_default();
                        *total = total
                            .checked_add(count)
                            .ok_or("BPE pair count exceeds u64")?;
                        all_positions.extend(positions);
                    }
                    Ok(pairs)
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

            self.merge_symbols(word, pair, id, limit, &mut changes);
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
            let after = position + (left.span + right.span);
            let word = self.word_for(position);
            let word_start = self.word_range(word).start;
            let weight = self.counts[word];

            let previous = if position > word_start {
                let neighbor = self.symbols[position - 1];
                (neighbor.id != EMPTY && neighbor.span != 0)
                    .then(|| (position - neighbor.span, neighbor))
            } else {
                None
            };
            let next = self
                .symbols
                .get(after)
                .copied()
                .filter(|neighbor| neighbor.id != EMPTY && neighbor.span != 0);
            self.apply_match(
                position..after,
                id,
                limit,
                (previous, next),
                weight,
                &mut changes,
            );
        }
        changes
    }

    fn merge_symbols(
        &mut self,
        word: usize,
        pair: Pair,
        id: u32,
        limit: usize,
        changes: &mut Vec<WordChange>,
    ) {
        let range = self.word_range(word);
        let weight = self.counts[word];
        let mut previous = None;
        let mut position = range.start;
        while position < range.end {
            let left = self.symbols[position];
            let right_position = position + left.span;
            if left.id != pair.0
                || right_position >= range.end
                || self.symbols[right_position].id != pair.1
                || self.symbols[right_position].span == 0
            {
                previous = Some(position);
                position = right_position;
                continue;
            }

            let after = position + (left.span + self.symbols[right_position].span);
            let neighbors = (
                previous.map(|before| (before, self.symbols[before])),
                (after < range.end).then(|| self.symbols[after]),
            );
            self.apply_match(position..after, id, limit, neighbors, weight, changes);
            previous = Some(position);
            position = after;
        }
    }

    fn apply_match(
        &mut self,
        matched: Range<usize>,
        id: u32,
        limit: usize,
        (previous, next): (Option<(usize, Symbol)>, Option<Symbol>),
        weight: u64,
        changes: &mut Vec<WordChange>,
    ) {
        let mut emit = |pair, added, position| {
            changes.push(WordChange {
                pair,
                added,
                coordinate: position as u64,
                weight,
            });
        };
        let position = matched.start;
        let after = matched.end;
        let span = matched.len();
        let left = self.symbols[position];
        let right_position = position + left.span;
        let right_id = self.symbols[right_position].id;
        if let Some((before, neighbor)) = previous {
            emit((neighbor.id, left.id), false, before);
            if neighbor.span + span < limit {
                emit((neighbor.id, id), true, before);
            }
        }

        self.symbols[position] = Symbol { id, span };
        if right_position + 1 != after {
            self.symbols[right_position] = Symbol { id: EMPTY, span: 0 };
        }
        self.symbols[after - 1] = Symbol { id, span };
        if let Some(neighbor) = next {
            emit((right_id, neighbor.id), false, right_position);
            if span + neighbor.span < limit {
                emit((id, neighbor.id), true, position);
            }
        }
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

fn read_symbols(symbols: &[Symbol]) -> impl Iterator<Item = u32> + '_ {
    let mut position = 0;
    std::iter::from_fn(move || {
        let symbol = *symbols.get(position)?;
        position += symbol.span;
        Some(symbol.id)
    })
}
