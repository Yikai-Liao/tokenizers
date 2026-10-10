//! Endpoint tokens retain their original coordinates; span metadata skips holes.
use super::{
    BpeTrainer, WORD_SEPARATOR_ID,
    vocabulary::{InitialTokenIds, Vocabulary},
    word_counts::WordCountsView,
};
use crate::progress::TrainingProgress;
use compact_str::CompactString;
use rayon::prelude::*;
use std::{
    ops::ControlFlow,
    sync::atomic::{AtomicU16, AtomicU32, AtomicUsize, Ordering},
};
use tk_encode::{Result, models::bpe::Pair};

pub(super) struct Corpus {
    tokens: Slots,
    starts: Vec<usize>,
    weight_regions: Vec<(usize, u64)>,
    unit_region: Option<(usize, usize)>,
    // Nonzero marks an activated ID. Until alias geometry diverges it also
    // stores that ID's span; afterwards occurrence_spans owns all geometry.
    spans: Vec<usize>,
    occurrence_spans: Option<Vec<AtomicUsize>>,
    whole_words: bool,
}
#[derive(Clone, Copy)]
pub(super) struct Match {
    pub(super) start: usize,
    pub(super) right: usize,
    pub(super) after: usize,
}
impl Match {
    pub(super) fn span(self) -> usize {
        self.after - self.start
    }
}
pub(super) struct CorpusPlan<'input> {
    // Counting needs geometry and borrowed words, but no resident token plane.
    // materialize consumes this metadata and constructs the complete Corpus once.
    starts: Vec<usize>,
    weight_regions: Vec<(usize, u64)>,
    unit_region: Option<(usize, usize)>,
    spans: Vec<usize>,
    whole_words: bool,
    words: Vec<(&'input CompactString, &'input u64)>,
    ids: InitialTokenIds,
    length: usize,
    slot_bound: usize,
    reuse: bool,
}
impl<'input> CorpusPlan<'input> {
    pub(super) fn build(
        words: WordCountsView<'input>,
        vocabulary: &mut Vocabulary,
        trainer: &BpeTrainer,
        reuse: bool,
        progress: &TrainingProgress,
    ) -> Result<Self> {
        let ids =
            vocabulary.initial_ids(words, &progress.stage("Resolve initial IDs", words.len()))?;
        let mut words: Vec<_> = words.iter().collect();
        words.par_sort_unstable_by(|a, b| b.1.cmp(a.1));
        let measured: Vec<_> = words
            .par_iter()
            .map(|(word, _)| ids.symbol_count(word))
            .collect();
        let mut starts = Vec::with_capacity(words.len());
        let mut weight_regions = Vec::new();
        let mut length = 1usize;
        let mut mass = 0u128;
        for (&(_, &weight), &symbols) in words.iter().zip(&measured) {
            starts.push(length);
            if weight_regions
                .last()
                .is_none_or(|&(_, previous)| previous != weight)
            {
                weight_regions.push((length, weight));
            }
            length = length
                .checked_add(symbols)
                .and_then(|n| n.checked_add(1))
                .ok_or("BPE corpus exceeds resident index bounds")?;
            mass += u128::from(weight) * symbols.saturating_sub(1) as u128;
        }
        if length > isize::MAX as usize / std::mem::size_of::<AtomicU32>() {
            return Err("BPE corpus exceeds resident allocation bounds".into());
        }
        if (reuse || !ids.plain())
            && (mass > i64::MAX as u128
                || words.iter().any(|(_, weight)| **weight > i64::MAX as u64))
        {
            return Err("BPE identity-reuse weighted edge mass or word weight exceeds i64".into());
        }
        let unit_region = weight_regions
            .iter()
            .position(|&(_, weight)| weight == 1)
            .map(|i| {
                (
                    weight_regions[i].0,
                    weight_regions.get(i + 1).map_or(length, |r| r.0),
                )
            });
        let slot_bound = vocabulary.len().max(trainer.vocab_size);
        Ok(Self {
            starts,
            weight_regions,
            unit_region,
            spans: vocabulary.take_initial_spans(),
            whole_words: trainer.max_token_length.is_some(),
            words,
            ids,
            length,
            slot_bound,
            reuse,
        })
    }
    pub(super) fn items(&self) -> usize {
        self.length
    }
    pub(super) fn word_count(&self) -> usize {
        self.words.len()
    }
    // Whole words preserve symbol-scan semantics. A single oversized word is
    // admitted alone; otherwise each range has at most `slots` resident slots.
    pub(super) fn initial_ranges(&self, slots: usize) -> Vec<std::ops::Range<usize>> {
        let mut ranges = Vec::new();
        let mut begin = 0;
        while begin < self.word_count() {
            let mut end = begin + 1;
            while end < self.word_count()
                && self.starts.get(end + 1).copied().unwrap_or(self.length) - self.starts[begin]
                    <= slots
            {
                end += 1;
            }
            ranges.push(begin..end);
            begin = end;
        }
        ranges
    }
    pub(super) fn small_pair_domain(&self) -> Option<usize> {
        (self.spans.len() <= 256).then_some(self.spans.len())
    }
    pub(super) fn initial_edges(
        &self,
        range: std::ops::Range<usize>,
        mut emit: impl FnMut(Pair, u64, u64) -> Result<()>,
    ) -> Result<()> {
        for word in range {
            let (text, &weight) = self.words[word];
            let mut position = self.starts[word];
            let mut previous = None;
            let mut failure = None;
            self.ids.scan_symbols(text, |id| {
                if let Some(left) = previous
                    && let Err(error) = emit((left, id), (position - 1) as u64, weight)
                {
                    failure = Some(error);
                    return ControlFlow::Break(());
                }
                previous = Some(id);
                position += 1;
                ControlFlow::Continue(())
            });
            if let Some(error) = failure {
                return Err(error);
            }
        }
        Ok(())
    }
    pub(super) fn materialize(self, progress: &TrainingProgress) -> Corpus {
        let tokens = Slots::new(self.length, self.slot_bound);
        let work = progress.stage("Materialize corpus", self.length);
        self.words
            .par_iter()
            .enumerate()
            .for_each(|(word, (text, _))| {
                #[cfg(test)]
                super::tests::observe_worker(super::tests::Phase::Materialize);
                let start = self.starts[word];
                let mut position = start;
                self.ids.scan_symbols(text, |id| {
                    tokens.set(position, id);
                    position += 1;
                    ControlFlow::Continue(())
                });
                work.complete(position - start + 1);
            });
        // Only historical reuse cohorts need a word directory after counting.
        Corpus {
            tokens,
            starts: if self.reuse { self.starts } else { Vec::new() },
            weight_regions: self.weight_regions,
            unit_region: self.unit_region,
            spans: self.spans,
            occurrence_spans: None,
            whole_words: self.whole_words,
        }
    }
}
impl Corpus {
    pub(super) fn len(&self) -> usize {
        self.tokens.len()
    }
    pub(super) fn resident(&self, coordinate: u64) -> usize {
        let position =
            usize::try_from(coordinate).expect("corpus coordinate fits resident indexing");
        assert!(
            position < self.len(),
            "coordinate belongs to the resident corpus"
        );
        position
    }
    #[inline]
    pub(super) fn token(&self, position: usize) -> u32 {
        self.tokens.get(position)
    }
    pub(super) fn prefetch(&self, position: usize) {
        #[cfg(target_arch = "x86_64")]
        {
            let address = match &self.tokens {
                Slots::Narrow(values) => values.get(position).map(|v| std::ptr::from_ref(v).cast()),
                Slots::Wide(values) => values.get(position).map(|v| std::ptr::from_ref(v).cast()),
            };
            if let Some(address) = address {
                // SAFETY: get() supplies an address within this live borrowed
                // allocation. The intrinsic issues only a cache hint; it does
                // not read or write a logical token or extend its lifetime.
                unsafe {
                    std::arch::x86_64::_mm_prefetch(address, std::arch::x86_64::_MM_HINT_T0);
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = position;
    }
    pub(super) fn weight_region(&self, position: usize) -> (u64, usize) {
        if self.weight_regions.len() == 1 {
            return (self.weight_regions[0].1, self.len());
        }
        if let Some((begin, end)) = self.unit_region
            && position >= begin
            && position < end
        {
            return (1, end);
        }
        let i = self
            .weight_regions
            .partition_point(|&(start, _)| start <= position)
            - 1;
        (
            self.weight_regions[i].1,
            self.weight_regions.get(i + 1).map_or(self.len(), |r| r.0),
        )
    }
    pub(super) fn word(&self, position: usize) -> usize {
        self.starts.partition_point(|&start| start <= position) - 1
    }
    pub(super) fn word_start(&self, word: usize) -> usize {
        self.starts[word]
    }
    pub(super) fn word_end(&self, word: usize) -> usize {
        self.starts.get(word + 1).copied().unwrap_or(self.len()) - 1
    }
    pub(super) fn word_weight(&self, word: usize) -> u64 {
        self.weight_region(self.word_start(word)).0
    }
    #[inline]
    pub(super) fn span(&self, position: usize) -> usize {
        self.occurrence_spans.as_ref().map_or_else(
            || self.id_span(self.token(position)),
            |spans| spans[position].load(Ordering::Relaxed),
        )
    }
    pub(super) fn id_count(&self) -> usize {
        self.spans.len()
    }
    pub(super) fn id_span(&self, id: u32) -> usize {
        self.spans[id as usize]
    }
    // Fresh IDs have fixed geometry. Cache it once per preparation task,
    // keeping occurrence-dependent alias matching in matched().
    pub(super) fn fresh_matcher(&self, pair: Pair) -> impl Fn(usize, u32) -> Option<Match> + '_ {
        debug_assert!(self.occurrence_spans.is_none());
        let left = self.id_span(pair.0);
        let total = left + self.id_span(pair.1);
        move |position, head| {
            if head != pair.0
                || left == 0
                || total == left
                || position + total >= self.len()
                || self.token(position + left) != pair.1
            {
                return None;
            }
            Some(Match {
                start: position,
                right: position + left,
                after: position + total,
            })
        }
    }
    pub(super) fn matched(&self, position: usize, pair: Pair) -> Option<Match> {
        if self.token(position) != pair.0 {
            return None;
        }
        let span = self.span(position);
        let right = position + span;
        if span == 0 || right >= self.len() || self.token(right) != pair.1 {
            return None;
        }
        let right_span = self.span(right);
        let after = right + right_span;
        (right_span != 0 && after < self.len()).then_some(Match {
            start: position,
            right,
            after,
        })
    }
    pub(super) fn whole_words(&self) -> bool {
        self.whole_words
    }
    pub(super) fn is_active(&self, id: u32) -> bool {
        self.spans.get(id as usize).is_some_and(|&span| span != 0)
    }
    pub(super) fn prepare_identity(&mut self, pair: Pair, id: u32) {
        self.whole_words |= self.is_active(id);
        // Once aliases need occurrence geometry, keep only activation here.
        // Summing representative ID spans would grow with repeated alias reuse.
        let span = if self.occurrence_spans.is_some() {
            1
        } else {
            self.id_span(pair.0) + self.id_span(pair.1)
        };
        if id as usize == self.spans.len() {
            self.spans.push(span);
        } else {
            let old = self.spans[id as usize];
            if self.occurrence_spans.is_none() && old != 0 && old != span {
                let values: Vec<_> = (0..self.len()).map(|_| AtomicUsize::new(0)).collect();
                for &start in &self.starts {
                    let mut p = start;
                    while self.token(p) != WORD_SEPARATOR_ID {
                        let n = self.id_span(self.token(p));
                        values[p].store(n, Ordering::Relaxed);
                        values[p + n - 1].store(n, Ordering::Relaxed);
                        p += n;
                    }
                }
                self.occurrence_spans = Some(values);
            }
            self.spans[id as usize] = span;
        }
    }
    // Call only after all preparation readers join. Matches own disjoint endpoint
    // slots; atomics also keep every individual access safe on error or unwind.
    pub(super) fn apply(&self, matched: Match, replacement: u32) {
        self.tokens.set(matched.start, replacement);
        if matched.right + 1 != matched.after {
            self.tokens.set(matched.right, WORD_SEPARATOR_ID);
        }
        self.tokens.set(matched.after - 1, replacement);
        if let Some(spans) = &self.occurrence_spans {
            if matched.right + 1 != matched.after {
                spans[matched.right].store(0, Ordering::Relaxed);
            }
            spans[matched.start].store(matched.span(), Ordering::Relaxed);
            spans[matched.after - 1].store(matched.span(), Ordering::Relaxed);
        }
    }
}

// A reserved u16 sentinel leaves real IDs 0..65534 available. Wider vocabularies
// use u32 directly; both planes have the same safe endpoint access protocol.
enum Slots {
    Narrow(Vec<AtomicU16>),
    Wide(Vec<AtomicU32>),
}
impl Slots {
    fn new(length: usize, vocabulary_bound: usize) -> Self {
        if vocabulary_bound <= u16::MAX as usize {
            Self::Narrow((0..length).map(|_| AtomicU16::new(u16::MAX)).collect())
        } else {
            Self::Wide(
                (0..length)
                    .map(|_| AtomicU32::new(WORD_SEPARATOR_ID))
                    .collect(),
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
                if id == u16::MAX {
                    WORD_SEPARATOR_ID
                } else {
                    u32::from(id)
                }
            }
            Self::Wide(values) => values[position].load(Ordering::Relaxed),
        }
    }
    #[inline]
    fn set(&self, position: usize, id: u32) {
        match self {
            Self::Narrow(values) => {
                let id = if id == WORD_SEPARATOR_ID {
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
