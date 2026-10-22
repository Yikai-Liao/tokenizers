//! Stable original-symbol coordinates for word-local training edits.
#[path = "merge.rs"]
mod merge;
use super::positions::Positions;
use super::vocabulary::InitialTokenIds;
use super::word_counts::WordCountsView;
use crate::progress::ProgressBar;
use ahash::AHashMap;
use std::ops::Range;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
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
    tokens: Box<[AtomicU32]>,
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
    pub(super) fn new(words: WordCountsView<'_>, ids: &InitialTokenIds) -> Self {
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
        Self {
            tokens: symbols
                .iter()
                .map(|symbol| AtomicU32::new(symbol.id))
                .collect(),
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
    fn apply(&self, starts: &[usize], id: u32) {
        // Preparation owns disjoint accepted intervals. Every reader joins before
        // writes; atomics keep endpoint accesses safe if a task unwinds.
        for &p in starts {
            let left = self.symbol(p);
            let right = p + left.span;
            let span = left.span + self.symbol(right).span;
            let after = p + span;
            self.store(p, Symbol { id, span });
            if right + 1 != after {
                self.store(right, Symbol { id: EMPTY, span: 0 });
            }
            self.store(after - 1, Symbol { id, span });
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
    fn store(&self, p: usize, symbol: Symbol) {
        self.tokens[p].store(symbol.id, Ordering::Relaxed);
        if let Some(spans) = &self.occurrence_spans {
            spans[p].store(symbol.span, Ordering::Relaxed);
        }
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
        Self {
            tokens: values.iter().map(|&id| AtomicU32::new(id)).collect(),
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
        self.tokens[p].load(Ordering::Relaxed)
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
