//! Endpoint tokens retain their original coordinates; span metadata skips holes.
use super::{
    BpeTrainer, WORD_SEPARATOR_ID, vocabulary::InitialTokenIds, word_counts::WordCountsView,
};
use compact_str::CompactString;
use itertools::Itertools;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU16, AtomicU32, AtomicUsize, Ordering};
use tk_encode::{Result, models::bpe::Pair};

/// Resident token endpoints at fixed original coordinates, with weights and spans.
/// Merges leave holes; span metadata skips them without moving other occurrences.
/// Preparation reads a joined snapshot; application writes only disjoint matches.
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

/// One accepted occurrence's left start, right start and exclusive merged end.
/// Coordinates describe the preparation snapshot, before endpoint writes apply.
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

/// Borrowed input and measured coordinate layout for initial pair counting.
/// Defers the resident token plane until temporary counting buffers retire;
/// materialization consumes the plan and transfers its metadata to the Corpus.
pub(super) struct CorpusPlan<'input> {
    starts: Vec<usize>,
    weight_regions: Vec<(usize, u64)>,
    unit_region: Option<(usize, usize)>,
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
        ids: InitialTokenIds,
        trainer: &BpeTrainer,
        reuse: bool,
    ) -> Result<Self> {
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
        let slot_bound = ids.token_count().max(trainer.vocab_size);
        Ok(Self {
            starts,
            weight_regions,
            unit_region,
            whole_words: trainer.max_token_length.is_some(),
            words,
            ids,
            length,
            slot_bound,
            reuse,
        })
    }

    pub(super) fn word_count(&self) -> usize {
        self.words.len()
    }

    // Whole words preserve symbol-scan semantics. A single oversized word is
    // admitted alone; otherwise each range has at most `slots` resident slots.
    pub(super) fn initial_ranges(&self, slots: usize) -> Vec<std::ops::Range<usize>> {
        let mut ranges = Vec::new();
        let mut begin = 0;
        let n = self.word_count();
        while begin < n {
            let start = self.starts[begin];
            let one_word_end = begin + 1;
            let end = if one_word_end == n || self.length - start <= slots {
                n
            } else {
                // The first word is admitted even when oversized. Each later
                // boundary closes one additional word; an exact-budget end fits.
                let additional_words = match slice_group_by::exponential_search_by(
                    &self.starts[one_word_end + 1..],
                    |&next| (next - start).cmp(&slots),
                ) {
                    Ok(exact) => exact + 1,
                    Err(insertion) => insertion,
                };
                one_word_end + additional_words
            };
            ranges.push(begin..end);
            begin = end;
        }
        ranges
    }

    pub(super) fn small_pair_domain(&self) -> Option<usize> {
        (self.ids.token_count() <= 256).then_some(self.ids.token_count())
    }

    pub(super) fn initial_edges(
        &self,
        range: std::ops::Range<usize>,
        mut emit: impl FnMut(Pair, u64, u64) -> Result<()>,
    ) -> Result<()> {
        for word in range {
            let (text, &weight) = self.words[word];
            self.ids
                .symbols(text)
                .tuple_windows()
                .enumerate()
                .try_for_each(|(offset, pair)| {
                    emit(pair, (self.starts[word] + offset) as u64, weight)
                })?;
        }
        Ok(())
    }

    pub(super) fn materialize(self) -> Corpus {
        let tokens = Slots::new(self.length, self.slot_bound);
        self.words
            .par_iter()
            .enumerate()
            .for_each(|(word, (text, _))| {
                #[cfg(test)]
                super::tests::observe_worker(super::tests::Phase::Materialize);
                let start = self.starts[word];
                for (offset, id) in self.ids.symbols(text).enumerate() {
                    tokens.set(start + offset, id);
                }
            });
        // Only historical reuse cohorts need a word directory after counting.
        Corpus {
            tokens,
            starts: if self.reuse { self.starts } else { Vec::new() },
            weight_regions: self.weight_regions,
            unit_region: self.unit_region,
            spans: self.ids.into_spans(),
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
        let address: Option<*const u8> = match &self.tokens {
            Slots::Narrow(values) => values.get(position).map(|v| std::ptr::from_ref(v).cast()),
            Slots::Wide(values) => values.get(position).map(|v| std::ptr::from_ref(v).cast()),
        };
        if let Some(address) = address {
            // Locality zero is L1 on both stable and nightly branches builds,
            // matching the former x86 T0 hint and covering AArch64 as well.
            branches::prefetch_read_data::<_, 0>(address);
        }
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

    /// Activation history persists after an ID's last occurrence is consumed.
    pub(super) fn has_been_activated(&self, id: u32) -> bool {
        self.spans.get(id as usize).is_some_and(|&span| span != 0)
    }

    pub(super) fn prepare_identity(&mut self, pair: Pair, id: u32) {
        self.whole_words |= self.has_been_activated(id);
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
                // Preserve each occurrence's old geometry before changing this
                // ID's representative span. Reused IDs can have several lengths.
                self.materialize_occurrence_spans();
            }
            self.spans[id as usize] = span;
        }
    }

    fn materialize_occurrence_spans(&mut self) {
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

/// Atomic endpoint plane selected once from the vocabulary ID bound.
/// A reserved u16 sentinel leaves real IDs 0..65534 available. Wider vocabularies
/// use u32 directly; both planes have the same safe endpoint access protocol.
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

#[cfg(test)]
mod tests {
    use super::*;
    use ahash::AHashMap;

    // Preserve the previous whole-word linear search as an independent oracle.
    fn linear_ranges(starts: &[usize], length: usize, slots: usize) -> Vec<std::ops::Range<usize>> {
        let mut ranges = Vec::new();
        let mut begin = 0;
        while begin < starts.len() {
            let mut end = begin + 1;
            while end < starts.len()
                && starts.get(end + 1).copied().unwrap_or(length) - starts[begin] <= slots
            {
                end += 1;
            }
            ranges.push(begin..end);
            begin = end;
        }
        ranges
    }

    #[test]
    fn initial_ranges_match_linear_search_at_slot_and_separator_boundaries() {
        let words = AHashMap::new();
        let view = WordCountsView::from_map(&words);
        let trainer = BpeTrainer::new(1, 0);
        let (_, ids) =
            super::super::vocabulary::Vocabulary::initialize(&trainer, view, 1, &mut None, &None)
                .unwrap();
        let mut plan = CorpusPlan::build(view, ids, &trainer, false).unwrap();
        let text = CompactString::default();
        let weight = 1;
        let mut layouts = vec![(vec![usize::MAX - 8, usize::MAX - 3], usize::MAX)];
        for n in [0, 1, 2, 3, 7, 128, 257] {
            for seed in 0..16 {
                let mut state = seed as u64;
                let mut starts = Vec::new();
                let mut length = 1;
                for _ in 0..n {
                    starts.push(length);
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    length += 1 + (state >> 32) as usize % 65;
                }
                layouts.push((starts, length));
            }
        }
        for (starts, length) in layouts {
            plan.words = vec![(&text, &weight); starts.len()];
            plan.starts = starts;
            plan.length = length;
            let mut budgets = vec![0, 1, 2, 3, 4, 5, 7, 8, 16, 17, 63, 64, 65, 1024, usize::MAX];
            if let Some(&start) = plan.starts.first() {
                budgets.extend(plan.starts.iter().map(|&next| next - start));
                budgets.push(length - start);
            }
            for slots in budgets {
                let ranges = plan.initial_ranges(slots);
                assert_eq!(ranges, linear_ranges(&plan.starts, length, slots));
                let mut end = 0;
                for range in ranges {
                    assert_eq!(range.start, end);
                    assert!(!range.is_empty());
                    let span = plan.starts.get(range.end).copied().unwrap_or(length)
                        - plan.starts[range.start];
                    assert!(span <= slots || range.len() == 1);
                    end = range.end;
                }
                assert_eq!(end, plan.word_count());
            }
        }

        // Slot widths 3, 4, 2, 6: exact-budget boundaries stay in the segment.
        plan.starts = vec![1, 4, 8, 10];
        plan.length = 16;
        plan.words = vec![(&text, &weight); 4];
        assert_eq!(plan.initial_ranges(7), vec![0..2, 2..3, 3..4]);
        assert_eq!(plan.initial_ranges(9), vec![0..3, 3..4]);
        assert_eq!(plan.initial_ranges(15), vec![0..4]);
        assert_eq!(plan.initial_ranges(2), vec![0..1, 1..2, 2..3, 3..4]);
    }
}
