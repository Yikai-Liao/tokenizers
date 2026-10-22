//! Stable original-symbol coordinates for word-local training edits.
#[path = "merge.rs"]
mod merge;
use super::positions::Positions;
use super::vocabulary::InitialTokenIds;
use super::word_counts::WordCountsView;
use crate::progress::ProgressBar;
use ahash::AHashMap;
use std::ops::Range;
use std::sync::atomic::{AtomicU16, AtomicU32, AtomicUsize, Ordering};
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
    tokens: Slots,
    occurrence_spans: Option<Box<[AtomicUsize]>>,
    starts: Vec<usize>,
    counts: Vec<u64>,
    whole_words: bool,
    id_metadata: Vec<usize>,
}

pub(super) struct WordChange {
    pub(super) pair: Pair,
    pub(super) delta: i32,
    pub(super) coordinate: u64,
    pub(super) weight: u64,
}

impl Corpus {
    pub(super) fn new(
        words: WordCountsView<'_>,
        ids: &InitialTokenIds,
        target_vocab_size: usize,
    ) -> Self {
        let slots = 1 + words
            .iter()
            .map(|(word, _)| ids.symbol_count(word) + 1)
            .sum::<usize>();
        let mut symbols = Vec::with_capacity(slots);
        symbols.push(Symbol { id: EMPTY, span: 0 });
        let mut starts = Vec::with_capacity(words.len());
        let mut counts = Vec::with_capacity(words.len());
        for (word, &weight) in words.iter() {
            starts.push(symbols.len());
            counts.push(weight);
            symbols.extend(ids.symbols(word).map(|id| Symbol { id, span: 1 }));
            symbols.push(Symbol { id: EMPTY, span: 0 });
        }
        let tokens = Slots::new(symbols.len(), ids.token_count().max(target_vocab_size));
        for (p, symbol) in symbols.iter().enumerate() {
            tokens.set(p, symbol.id);
        }
        Self {
            tokens,
            occurrence_spans: None,
            starts,
            counts,
            whole_words: false,
            id_metadata: ids.metadata(),
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

                let chars: Vec<_> = self.read_word(self.word_range(i)).collect();
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
                .unwrap_or(self.tokens.len())
                - 1
    }
}

impl Corpus {
    fn apply(&self, starts: &[usize], merged_id: u32) {
        let Some(&first) = starts.first() else {
            return;
        };
        match &self.occurrence_spans {
            None => {
                // One rule has shared geometry. Cache it before any token store:
                // first + left locates its right token, whose span completes total.
                let left = self.fixed_id_span(self.token(first));
                let total = left + self.fixed_id_span(self.token(first + left));
                for &start in starts {
                    let right = start + left;
                    let after = start + total;
                    self.tokens.replace_pair(start, right, after, merged_id);
                }
            }
            Some(spans) => {
                for &start in starts {
                    // Read this occurrence's old geometry before changing its endpoints.
                    let left = spans[start].load(Ordering::Relaxed);
                    let right = start + left;
                    let total = left + spans[right].load(Ordering::Relaxed);
                    let after = start + total;
                    self.tokens.replace_pair(start, right, after, merged_id);
                    // A one-slot right token is the new tail itself; clear its
                    // old span only when that right start becomes an interior hole.
                    if right + 1 != after {
                        spans[right].store(0, Ordering::Relaxed);
                    }
                    spans[start].store(total, Ordering::Relaxed);
                    spans[after - 1].store(total, Ordering::Relaxed);
                }
            }
        }
    }
}

impl Corpus {
    fn symbol(&self, p: usize) -> Symbol {
        let id = self.token(p);
        let span = if id == EMPTY {
            0
        } else {
            self.occurrence_spans.as_ref().map_or_else(
                || self.fixed_id_span(id),
                |spans| spans[p].load(Ordering::Relaxed),
            )
        };
        Symbol { id, span }
    }

    fn read_word(&self, range: Range<usize>) -> impl Iterator<Item = u32> + '_ {
        let mut p = range.start;
        std::iter::from_fn(move || {
            if p >= range.end {
                return None;
            }
            let symbol = self.symbol(p);
            p += symbol.span;
            Some(symbol.id)
        })
    }
    #[cfg(test)]
    fn fixture(values: &[u32], starts: Vec<usize>, counts: Vec<u64>) -> Self {
        let bound = values
            .iter()
            .copied()
            .filter(|&id| id != EMPTY)
            .max()
            .map_or(0, |id| id as usize + 1);
        let tokens = Slots::new(values.len(), bound);
        for (p, &id) in values.iter().enumerate() {
            tokens.set(p, id);
        }
        Self {
            tokens,
            occurrence_spans: None,
            starts,
            counts,
            whole_words: false,
            id_metadata: vec![
                1;
                values
                    .iter()
                    .copied()
                    .filter(|&id| id != EMPTY)
                    .max()
                    .map_or(0, |id| id as usize + 1)
            ],
        }
    }
}

impl Corpus {
    fn token(&self, p: usize) -> u32 {
        self.tokens.get(p)
    }
    fn len(&self) -> usize {
        self.tokens.len()
    }
    fn has_been_activated(&self, id: u32) -> bool {
        self.id_metadata
            .get(id as usize)
            .is_some_and(|&span| span != 0)
    }
    pub(super) fn activate(&mut self, pair: Pair, id: u32) {
        // Once aliases need occurrence geometry, keep only activation here.
        // Summing representative ID spans would grow with repeated alias reuse.
        let span = if self.occurrence_spans.is_some() {
            1
        } else {
            self.fixed_id_span(pair.0) + self.fixed_id_span(pair.1)
        };
        if id as usize == self.id_metadata.len() {
            self.id_metadata.push(span);
        } else {
            let old = self.id_metadata[id as usize];
            if self.occurrence_spans.is_none() && old != 0 && old != span {
                // Preserve each occurrence's old geometry before changing this
                // ID's representative span. Reused IDs can have several lengths.
                self.materialize_occurrence_spans();
            }
            self.id_metadata[id as usize] = span;
        }
    }
    fn fixed_id_span(&self, id: u32) -> usize {
        self.id_metadata[id as usize]
    }
    fn materialize_occurrence_spans(&mut self) {
        let values: Vec<_> = (0..self.len()).map(|_| AtomicUsize::new(0)).collect();
        for &start in &self.starts {
            let mut p = start;
            while self.token(p) != EMPTY {
                let n = self.fixed_id_span(self.token(p));
                values[p].store(n, Ordering::Relaxed);
                // A span covers [p, p + n): its final endpoint is p + n - 1.
                values[p + n - 1].store(n, Ordering::Relaxed);
                p += n;
            }
        }
        self.occurrence_spans = Some(values.into_boxed_slice());
    }
}

enum Slots {
    Narrow(Box<[AtomicU16]>),
    Wide(Box<[AtomicU32]>),
}
impl Slots {
    fn new(length: usize, vocabulary_bound: usize) -> Self {
        if vocabulary_bound <= u16::MAX as usize {
            Self::Narrow(
                (0..length)
                    .map(|_| AtomicU16::new(u16::MAX))
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            )
        } else {
            Self::Wide(
                (0..length)
                    .map(|_| AtomicU32::new(EMPTY))
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            )
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Narrow(values) => values.len(),
            Self::Wide(values) => values.len(),
        }
    }

    #[inline]
    fn get(&self, position: usize) -> u32 {
        match self {
            Self::Narrow(values) => {
                let id = values[position].load(Ordering::Relaxed);
                if id == u16::MAX { EMPTY } else { u32::from(id) }
            }
            Self::Wide(values) => values[position].load(Ordering::Relaxed),
        }
    }

    #[inline]
    fn set(&self, position: usize, id: u32) {
        match self {
            Self::Narrow(values) => {
                let id = if id == EMPTY {
                    u16::MAX
                } else {
                    u16::try_from(id).expect("vocabulary bound admits narrow endpoint ID")
                };
                values[position].store(id, Ordering::Relaxed);
            }
            Self::Wide(values) => values[position].store(id, Ordering::Relaxed),
        }
    }
}

impl Slots {
    fn replace_pair(&self, start: usize, right: usize, after: usize, merged_id: u32) {
        self.set(start, merged_id);
        if right + 1 != after {
            self.set(right, EMPTY);
        }
        self.set(after - 1, merged_id);
    }
}

#[cfg(test)]
mod merge_contract_tests {
    use super::*;

    #[test]
    fn slot_storage_preserves_ids_at_the_narrow_boundary() {
        for bound in [65535, 65536] {
            let slots = Slots::new(7, bound);
            assert_eq!(slots.len(), 7);
            assert_eq!(slots.get(0), EMPTY);
            slots.set(1, 65534);
            slots.set(3, 2);
            slots.set(4, 2);
            slots.replace_pair(1, 3, 5, (bound - 1) as u32);
            assert_eq!(slots.get(0), EMPTY);
            assert_eq!(slots.get(1), (bound - 1) as u32);
            assert_eq!(slots.get(3), EMPTY);
            assert_eq!(slots.get(4), (bound - 1) as u32);
            assert_eq!(slots.get(5), EMPTY);
        }
    }
}
