//! Stable original-symbol coordinates for word-local training edits.
#[path = "merge.rs"]
mod merge;
use merge::SelectedRuleIndex;
#[path = "merge_changes.rs"]
mod changes;
use changes::NeighborScratchPool;

use super::TrainingMode;
use super::positions::Positions;
use super::vocabulary::InitialTokenIds;
use super::word_counts::WordCountsView;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::ops::Range;
use std::sync::atomic::{AtomicU16, AtomicU32, AtomicUsize, Ordering};
use tk_encode::Result;
use tk_encode::models::bpe::Pair;
use tk_encode::parallelism;
use tk_encode::parallelism::MaybeParallelRefIterator;

const EMPTY: u32 = u32::MAX;

#[cfg(test)]
#[derive(Clone, Copy)]
struct Symbol {
    id: u32,
    span: usize,
}

/// Owns mutable word storage and the frequency attached to each word domain.
pub(super) struct Corpus {
    tokens: Slots,
    neighbor_scratch: NeighborScratchPool,
    selected_rules: SelectedRuleIndex,
    occurrence_spans: Option<Box<[AtomicUsize]>>,
    starts: Box<[usize]>,
    weight_runs: Box<[WeightRun]>,
    unit_weight_range: Option<Range<usize>>,
    whole_words: bool,
    birth_pair_span_limit: Option<usize>,
    mode: TrainingMode,
    id_metadata: Vec<usize>,
}

/// Borrowed layout and initial IDs; resident endpoints are deferred until counting joins.
pub(super) struct CorpusPlan<'input> {
    words: Box<[(&'input compact_str::CompactString, &'input u64)]>,
    starts: Box<[usize]>,
    weight_runs: Box<[WeightRun]>,
    unit_weight_range: Option<Range<usize>>,
    length: usize,
    ids: InitialTokenIds,
    slot_bound: usize,
    birth_pair_span_limit: Option<usize>,
    mode: TrainingMode,
}
impl<'input> CorpusPlan<'input> {
    pub(super) fn build(
        words: WordCountsView<'input>,
        ids: InitialTokenIds,
        target_vocab_size: usize,
        mode: TrainingMode,
        birth_pair_span_limit: Option<usize>,
    ) -> Result<Self> {
        let mut words: Vec<_> = words.iter().collect();
        if parallelism::get_parallelism() {
            words.par_sort_unstable_by(|a, b| b.1.cmp(a.1));
        } else {
            words.sort_unstable_by(|a, b| b.1.cmp(a.1));
        }
        let words = words.into_boxed_slice();
        let mut starts: Vec<_> = words
            .maybe_par_iter()
            .map(|(word, _)| ids.symbol_count(word))
            .collect();
        let mut weight_runs = Vec::<WeightRun>::new();
        let mut length = 1usize;
        // Reuse the measured symbol counts as the checked word-start directory.
        for (&(_, &weight), start) in words.iter().zip(&mut starts) {
            let symbols = *start;
            *start = length;
            if weight_runs.last().is_none_or(|run| run.weight != weight) {
                weight_runs.push(WeightRun {
                    start: length,
                    weight,
                });
            }
            // Each word contributes its retained symbols and one trailing
            // separator; length already includes the leading separator.
            length = length
                .checked_add(symbols)
                .and_then(|n| n.checked_add(1))
                .ok_or("BPE corpus exceeds resident index bounds")?;
        }
        // Bound bytes using the widest slot representation. Division avoids
        // overflowing the multiplication that an allocation-size check needs.
        if length > isize::MAX as usize / std::mem::size_of::<AtomicU32>() {
            return Err("BPE corpus exceeds resident allocation bounds".into());
        }
        let mass = words
            .iter()
            .enumerate()
            .map(|(i, (_, weight))| {
                let end = starts.get(i + 1).copied().unwrap_or(length) - 1;
                u128::from(**weight) * (end - starts[i]).saturating_sub(1) as u128
            })
            .sum::<u128>();
        if (mode == TrainingMode::Reuse || !ids.is_plain())
            && (mass > i64::MAX as u128
                || words.iter().any(|(_, weight)| **weight > i64::MAX as u64))
        {
            return Err("BPE identity-reuse weighted edge mass or word weight exceeds i64".into());
        }
        // Descending weights put all frequency-one words in one contiguous run.
        // Its exclusive end is the next run's start, or the corpus end.
        let unit_weight_range = weight_runs
            .iter()
            .position(|run| run.weight == 1)
            .map(|i| weight_runs[i].start..weight_runs.get(i + 1).map_or(length, |run| run.start));
        let slot_bound = ids.token_count().max(target_vocab_size);
        Ok(Self {
            starts: starts.into_boxed_slice(),
            weight_runs: weight_runs.into_boxed_slice(),
            unit_weight_range,
            words,
            ids,
            length,
            slot_bound,
            birth_pair_span_limit,
            mode,
        })
    }
    pub(super) fn word_count(&self) -> usize {
        self.words.len()
    }
    pub(super) fn materialize(self) -> Corpus {
        let tokens = Slots::new(self.length, self.slot_bound);
        self.words
            .maybe_par_iter()
            .enumerate()
            .for_each(|(word, (text, _))| {
                let start = self.starts[word];
                self.ids.symbols(text).enumerate().for_each(|(offset, id)| {
                    tokens.set(start + offset, id);
                });
            });
        Corpus {
            tokens,
            neighbor_scratch: NeighborScratchPool::default(),
            selected_rules: SelectedRuleIndex::default(),
            occurrence_spans: None,
            starts: if self.mode == TrainingMode::Reuse || !self.ids.is_plain() {
                self.starts
            } else {
                Box::default()
            },
            weight_runs: self.weight_runs,
            unit_weight_range: self.unit_weight_range,
            whole_words: self.birth_pair_span_limit.is_some(),
            birth_pair_span_limit: self.birth_pair_span_limit,
            mode: self.mode,
            id_metadata: self.ids.into_metadata(),
        }
    }
}

impl Corpus {
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
    #[cfg(test)]
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

    #[cfg(test)]
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
            neighbor_scratch: NeighborScratchPool::default(),
            selected_rules: SelectedRuleIndex::default(),
            occurrence_spans: None,
            weight_runs: starts
                .iter()
                .zip(&counts)
                .map(|(&start, &weight)| WeightRun { start, weight })
                .collect(),
            unit_weight_range: None,
            starts: starts.into_boxed_slice(),
            whole_words: false,
            birth_pair_span_limit: None,
            mode: TrainingMode::Fresh,
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
    pub(super) fn has_been_activated(&self, id: u32) -> bool {
        self.id_metadata
            .get(id as usize)
            .is_some_and(|&span| span != 0)
    }
    pub(super) fn activate(&mut self, pair: Pair, id: u32) {
        self.whole_words |= self.has_been_activated(id);
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

struct WeightRun {
    start: usize,
    weight: u64,
}
#[derive(Clone, Copy)]
struct WeightLookup {
    weight: u64,
    end: usize,
}
impl Corpus {
    fn weight(&self, position: usize) -> WeightLookup {
        if self.weight_runs.len() == 1 {
            return WeightLookup {
                weight: self.weight_runs[0].weight,
                end: self.len(),
            };
        }
        // Cached runs are half-open: a coordinate at end belongs to the next
        // weight run and must fall through to the directory lookup.
        if let Some(range) = &self.unit_weight_range
            && position >= range.start
            && position < range.end
        {
            return WeightLookup {
                weight: 1,
                end: range.end,
            };
        }
        // partition_point returns the first later run; its predecessor owns
        // this coordinate. Callers supply a coordinate at or after the first start.
        let i = self
            .weight_runs
            .partition_point(|run| run.start <= position)
            - 1;
        WeightLookup {
            weight: self.weight_runs[i].weight,
            end: self
                .weight_runs
                .get(i + 1)
                .map_or(self.len(), |run| run.start),
        }
    }
}

pub(super) struct InitialEdge {
    pub(super) pair: Pair,
    pub(super) coordinate: u64,
    pub(super) weight: u64,
}
pub(super) struct InitialChunk<'plan, 'input> {
    plan: &'plan CorpusPlan<'input>,
    words: Range<usize>,
}

impl<'input> CorpusPlan<'input> {
    pub(super) fn chunks(&self, slots: usize) -> Vec<InitialChunk<'_, 'input>> {
        let mut chunks = Vec::new();
        let mut begin = 0;
        let n = self.word_count();
        while begin < n {
            let start = self.starts[begin];
            let one_word_end = begin + 1;
            // The last word must be admitted; otherwise take the whole suffix
            // only when all remaining symbols and separators fit the slot budget.
            let end = if one_word_end == n || self.length - start <= slots {
                n
            } else {
                // The first word is admitted even when oversized. Each later
                // boundary closes one additional word; an exact-budget end fits.
                // Search starts after the first word's end. An exact hit at
                // offset k closes k + 1 additional words; an insertion index
                // counts only earlier boundaries whose spans fit the budget.
                let additional_words = match slice_group_by::exponential_search_by(
                    &self.starts[one_word_end + 1..],
                    |&next| (next - start).cmp(&slots),
                ) {
                    Ok(exact) => exact + 1,
                    Err(insertion) => insertion,
                };
                one_word_end + additional_words
            };
            chunks.push(InitialChunk {
                plan: self,
                words: begin..end,
            });
            begin = end;
        }
        chunks
    }
    pub(super) fn slot_count(&self) -> usize {
        self.length
    }
}

impl InitialChunk<'_, '_> {
    /// Return the first visitor error; successful scans return word progress.
    // Either::fold dispatches once per word, rather than once per symbol.
    #[allow(clippy::manual_try_fold)]
    pub(super) fn scan(self, mut visit: impl FnMut(InitialEdge) -> Result<()>) -> Result<usize> {
        let words = self.words.len();
        for word in self.words {
            let (text, &weight) = self.plan.words[word];
            let mut symbols = self.plan.ids.symbols(text);
            let Some(mut left) = symbols.next() else {
                continue;
            };
            let mut coordinate = self.plan.starts[word];
            // Either forwards fold once per word. After an error, decode only
            // the remaining symbols of this word without visiting further edges.
            symbols.fold(
                Ok(()),
                #[inline(always)]
                |result, right| {
                    result?;
                    let result = visit(InitialEdge {
                        pair: (left, right),
                        coordinate: coordinate as u64,
                        weight,
                    });
                    left = right;
                    coordinate += 1;
                    result
                },
            )?;
        }
        Ok(words)
    }
}

impl CorpusPlan<'_> {
    pub(super) fn token_count(&self) -> usize {
        self.ids.token_count()
    }
}

impl Corpus {
    fn occurrence_span(&self, position: usize) -> usize {
        self.occurrence_spans.as_ref().map_or_else(
            || self.fixed_id_span(self.token(position)),
            |spans| spans[position].load(Ordering::Relaxed),
        )
    }
    fn match_at(&self, position: usize, pair: Pair) -> Option<Range<usize>> {
        if self.token(position) != pair.0 {
            return None;
        }
        let span = self.occurrence_span(position);
        let right = position + span;
        // Reject an inactive left endpoint or a right start outside the plane
        // before reading it; holes are skipped by the left occurrence's span.
        if span == 0 || right >= self.len() || self.token(right) != pair.1 {
            return None;
        }
        let right_span = self.occurrence_span(right);
        let after = right + right_span;
        // The right occurrence must be active and its exclusive end must remain
        // resident, because preparation reads the neighbor at that coordinate.
        (right_span != 0 && after < self.len()).then_some(position..after)
    }
}

pub(super) struct PairChange<P> {
    pub(super) removed: Pair,
    pub(super) born: Pair,
    pub(super) removed_weight: u64,
    pub(super) born_weight: u64,
    pub(super) positions: P,
    pub(super) birth_group: usize,
}
pub(super) struct ProducerChanges {
    pub(super) whole_rule: bool,
    pub(super) changes: Vec<PairChange<Positions>>,
}

impl CorpusPlan<'_> {
    pub(super) fn mode(&self) -> TrainingMode {
        self.mode
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

pub(super) struct SelectedMerge {
    pub(super) pair: Pair,
    pub(super) merged_id: u32,
    pub(super) positions: Positions,
}

impl Corpus {
    fn resident(&self, coordinate: u64) -> usize {
        let position =
            usize::try_from(coordinate).expect("corpus coordinate fits resident indexing");
        assert!(
            position < self.len(),
            "coordinate {coordinate} is outside the resident corpus of {} slots",
            self.len()
        );
        position
    }
}

impl Corpus {
    fn id_count(&self) -> usize {
        self.id_metadata.len()
    }
}

impl Corpus {
    fn prefetch(&self, position: usize) {
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
    fn fixed_matcher(&self, pair: Pair) -> impl Fn(usize) -> Option<Range<usize>> + '_ {
        debug_assert!(self.occurrence_spans.is_none());
        let left = self.fixed_id_span(pair.0);
        let total = left + self.fixed_id_span(pair.1);
        move |position| {
            // Zero spans mark inactive IDs (total == left means an inactive
            // right ID). The exclusive end must still name a resident slot,
            // and the right token must begin exactly one left span away.
            if self.token(position) != pair.0
                || left == 0
                || total == left
                || position + total >= self.len()
                || self.token(position + left) != pair.1
            {
                return None;
            }
            Some(position..position + total)
        }
    }
    fn fresh_matches<'a>(
        &'a self,
        coordinates: impl IntoIterator<Item = u64> + 'a,
        pair: Pair,
    ) -> impl Iterator<Item = (Range<usize>, u64)> + 'a {
        let matcher = self.fixed_matcher(pair);
        let mut positions = coordinates
            .into_iter()
            .map(move |coordinate| self.resident(coordinate))
            .inspect(move |&p| self.prefetch(p));
        let mut ahead = VecDeque::with_capacity(16);
        ahead.extend(positions.by_ref().take(16));
        let mut cached_weight: Option<WeightLookup> = None;
        std::iter::from_fn(move || {
            let position = ahead.pop_front()?;
            ahead.extend(positions.next());
            Some(position)
        })
        .filter_map(move |p| {
            let matched = matcher(p)?;
            let lookup = match cached_weight {
                Some(lookup) if p < lookup.end => lookup,
                _ => self.weight(p),
            };
            cached_weight = Some(lookup);
            Some((matched, lookup.weight))
        })
    }
}

#[cfg(test)]
mod merge_contract_tests {
    use super::*;

    #[test]
    fn joined_merge_updates_endpoints_and_strict_neighbor_counts() {
        for (limit, born) in [(3, 0), (4, 7)] {
            let mut corpus = Corpus::fixture(&[EMPTY, 0, 1, 2, 2, 3, EMPTY], vec![1], vec![7]);
            corpus.birth_pair_span_limit = Some(limit);
            corpus.activate((2, 2), 4);
            let changes = corpus
                .merge(vec![SelectedMerge {
                    pair: (2, 2),
                    merged_id: 4,
                    positions: Positions::from_unsorted(vec![3]).unwrap(),
                }])
                .unwrap();
            let counts: Vec<_> = changes
                .iter()
                .flat_map(|producer| &producer.changes)
                .map(|change| {
                    (
                        change.removed,
                        change.born,
                        change.removed_weight,
                        change.born_weight,
                    )
                })
                .collect();
            assert_eq!(
                counts,
                [((1, 2), (1, 4), 7, born), ((2, 3), (4, 3), 7, born)]
            );
            assert_eq!(corpus.read_word(1..6).collect::<Vec<_>>(), [0, 1, 4, 3]);
            assert_eq!(
                (corpus.occurrence_span(3), corpus.occurrence_span(4)),
                (2, 2)
            );
        }
    }

    #[test]
    fn reused_identity_keeps_distinct_occurrence_spans() {
        for (values, pair, coordinate, expected, birth, spans) in [
            (
                [EMPTY, 0, 0, 0, 0, 2, EMPTY],
                (0, 2),
                3,
                [0, 3],
                (0, 3),
                (2, 3),
            ),
            (
                [EMPTY, 2, 0, 0, 0, 0, EMPTY],
                (2, 0),
                1,
                [3, 0],
                (3, 0),
                (3, 2),
            ),
        ] {
            for (limit, born_weight) in [(5, 0), (6, 11)] {
                let mut corpus = Corpus::fixture(&values, vec![1], vec![11]);
                corpus.mode = TrainingMode::Reuse;
                corpus.activate((0, 0), 0);
                corpus
                    .merge(vec![SelectedMerge {
                        pair: (0, 0),
                        merged_id: 0,
                        positions: Positions::from_unsorted(vec![1, 2, 3, 4]).unwrap(),
                    }])
                    .unwrap();
                corpus.birth_pair_span_limit = Some(limit);
                corpus.activate(pair, 3);
                let changes = corpus
                    .merge(vec![SelectedMerge {
                        pair,
                        merged_id: 3,
                        positions: Positions::from_unsorted(vec![coordinate]).unwrap(),
                    }])
                    .unwrap();
                let counts: Vec<_> = changes
                    .into_iter()
                    .flat_map(|producer| producer.changes)
                    .map(|change| {
                        (
                            change.removed,
                            change.born,
                            change.removed_weight,
                            change.born_weight,
                        )
                    })
                    .collect();
                assert_eq!(counts, [((0, 0), birth, 11, born_weight)]);
                assert_eq!(corpus.read_word(1..6).collect::<Vec<_>>(), expected);
                assert_eq!(
                    (corpus.occurrence_span(1), corpus.occurrence_span(5)),
                    spans
                );
            }
        }
        // Historical coordinates select a word even after alias rewrites have
        // turned the coordinate into an interior slot of a longer token.
        let mut corpus = Corpus::fixture(&[EMPTY, 0, 0, 0, 0, 0, EMPTY], vec![1], vec![1]);
        corpus.mode = TrainingMode::Reuse;
        let apply = |corpus: &mut Corpus, id, coordinates| {
            corpus.activate((0, 0), id);
            corpus
                .merge(vec![SelectedMerge {
                    pair: (0, 0),
                    merged_id: id,
                    positions: Positions::from_unsorted(coordinates).unwrap(),
                }])
                .unwrap();
        };
        apply(&mut corpus, 0, vec![1, 2, 3, 4]);
        apply(&mut corpus, 0, vec![1]);
        apply(&mut corpus, 1, vec![2]);
        assert_eq!(corpus.read_word(1..6).collect::<Vec<_>>(), [1]);
        assert_eq!(
            (corpus.occurrence_span(1), corpus.occurrence_span(5)),
            (5, 5)
        );
    }

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

    #[test]
    fn initial_chunks_and_weight_runs_preserve_zero_and_unit_weights() {
        use super::super::{BpeTrainer, vocabulary::Vocabulary, word_counts::WordCounts};
        let words =
            WordCounts::from_unique(vec![("ef".into(), 0), ("ab".into(), 9), ("cd".into(), 1)]);
        let view = WordCountsView::from(&words);
        let trainer = BpeTrainer::new(1, 64);
        let (_, ids) =
            Vocabulary::initialize(&trainer, view, &trainer.select_alphabet(view), &None).unwrap();
        let plan = CorpusPlan::build(view, ids, 64, TrainingMode::Fresh, None).unwrap();
        assert_eq!(plan.slot_count(), 10);
        for budget in [0, 2, 3, 6, usize::MAX] {
            let mut edges = Vec::new();
            let mut word_count = 0;
            for chunk in plan.chunks(budget) {
                word_count += chunk
                    .scan(|edge| {
                        edges.push((edge.pair, edge.coordinate, edge.weight));
                        Ok(())
                    })
                    .unwrap();
            }
            assert_eq!(word_count, 3);
            assert_eq!(edges, [((0, 1), 1, 9), ((2, 3), 4, 1), ((4, 5), 7, 0)]);
        }
        let corpus = plan.materialize();
        let expected = [
            (1, 9, 4),
            (2, 9, 4),
            (4, 1, 7),
            (5, 1, 7),
            (7, 0, 10),
            (8, 0, 10),
        ];
        for (position, weight, end) in expected {
            let actual = corpus.weight(position);
            assert_eq!((actual.weight, actual.end), (weight, end));
        }
    }
}
