//! Stable token endpoints over one fixed slot plane.
//!
//! A live token retains its ID at its first and last slots. Its span locates the
//! next token; the preceding endpoint locates the previous token. Rewrites never
//! shift a word's suffix. Unequal identity reuse materializes occurrence spans
//! before changing the corpus, while immutable boundaries retain word identity.
use super::{
    IdentityPolicy, WORD_SEPARATOR_ID,
    vocabulary::{InitialIds, Vocabulary},
};
use crate::progress::TrainingProgress;
use ahash::AHashMap;
use compact_str::CompactString;
use rayon::prelude::*;
use std::{
    mem::{ManuallyDrop, MaybeUninit},
    ops::Range,
    sync::atomic::{AtomicU32, Ordering},
};
use tk_collections::{IntervalCursor, IntervalIndex};
use tk_encode::{Result, models::bpe::Pair};

pub(super) struct Corpus {
    slots: Vec<AtomicU32>,
    word_starts: Vec<u64>,
    weights: IntervalIndex<u64>,
    unit_weight: Option<(u64, u64)>,
    spans_by_id: Vec<u64>,
    occurrence_spans: Option<Vec<u64>>,
    scan_whole_words: bool,
}
#[cfg(test)]
pub(super) struct InitialCorpus<'a> {
    pub(super) token_ids: &'a [AtomicU32],
    pub(super) word_weights: &'a IntervalIndex<u64>,
}
// A word keeps its original byte coordinates and its measured global start.
// Initial pair construction borrows this plan; no mutable slot allocation exists
// until all raw records have retired.
struct PlannedWord<'a> {
    word: &'a CompactString,
    start: u64,
}
struct SymbolCheckpoint {
    position: usize,
    byte: usize,
}
pub(super) struct PreparedCorpus<'a> {
    words: Vec<PlannedWord<'a>>,
    checkpoints: Vec<SymbolCheckpoint>,
    initial_ids: InitialIds,
    len: usize,
    weights: IntervalIndex<u64>,
    unit_weight: Option<(u64, u64)>,
    spans_by_id: Vec<u64>,
    scan_whole_words: bool,
    edges: usize,
}
/// Initial routing visits complete keys in ascending physical-coordinate order.
/// A range owns left endpoints; its final edge may read one token past the range.
pub(super) trait InitialPairSource: Sync {
    fn len(&self) -> usize;
    fn word_weights(&self) -> &IntervalIndex<u64>;
    fn for_each_edge(&self, range: Range<usize>, emit: impl FnMut(usize, u64));
}
#[cfg(test)]
impl InitialPairSource for InitialCorpus<'_> {
    fn len(&self) -> usize {
        self.token_ids.len()
    }
    fn word_weights(&self) -> &IntervalIndex<u64> {
        self.word_weights
    }
    fn for_each_edge(&self, range: Range<usize>, mut emit: impl FnMut(usize, u64)) {
        for position in range {
            let left = self.token_ids[position].load(Ordering::Relaxed);
            let right = self.token_ids[position + 1].load(Ordering::Relaxed);
            if left != WORD_SEPARATOR_ID && right != WORD_SEPARATOR_ID {
                emit(position, super::pair_index::pair_key((left, right)));
            }
        }
    }
}
impl InitialPairSource for &PreparedCorpus<'_> {
    fn len(&self) -> usize {
        self.len
    }
    fn word_weights(&self) -> &IntervalIndex<u64> {
        &self.weights
    }
    fn for_each_edge(&self, range: Range<usize>, mut emit: impl FnMut(usize, u64)) {
        let first = self
            .words
            .partition_point(|word| word.start <= range.start as u64)
            .saturating_sub(1);
        for index in first..self.words.len() {
            let planned = &self.words[index];
            let word_start = planned.start as usize;
            if word_start >= range.end {
                break;
            }
            let separator = self
                .words
                .get(index + 1)
                .map_or(self.len, |word| word.start as usize)
                - 1;
            if separator <= range.start {
                continue;
            }
            let mut position = word_start;
            let mut byte_start = 0;
            if word_start < range.start {
                let after = self
                    .checkpoints
                    .partition_point(|point| point.position <= range.start);
                if let Some(point) = after.checked_sub(1).map(|index| &self.checkpoints[index])
                    && point.position >= word_start
                {
                    position = point.position;
                    byte_start = point.byte;
                }
            }
            let mut previous = None;
            for (offset, character) in planned.word[byte_start..].char_indices() {
                let byte = byte_start + offset;
                let id = if self.initial_ids.plain() {
                    self.initial_ids.plain_id(character)
                } else {
                    self.initial_ids.id(
                        character,
                        byte == 0,
                        byte + character.len_utf8() == planned.word.len(),
                    )
                };
                let Some(id) = id else {
                    continue;
                };
                if let Some((left_position, left)) = previous
                    && left_position >= range.start
                    && left_position < range.end
                {
                    emit(left_position, super::pair_index::pair_key((left, id)));
                }
                previous = Some((position, id));
                // The right endpoint at range.end has supplied the lookahead.
                if position >= range.end {
                    break;
                }
                position += 1;
            }
        }
    }
}
#[derive(Clone, Copy)]
pub(super) struct PairMatch {
    pub(super) left_start: u64,
    pub(super) right_start: u64,
    pub(super) next_start: u64,
    pub(super) merged_span: u64,
}
pub(super) struct WordWeightCursor<'a> {
    intervals: IntervalCursor<'a, u64>,
    uniform_weight: Option<u64>,
    unit_weight: Option<(u64, u64)>,
}
impl WordWeightCursor<'_> {
    #[inline]
    pub(super) fn weight(&mut self, position: u64) -> u64 {
        if let Some(weight) = self.uniform_weight {
            return weight;
        }
        // PERF: Weight ordering gives one complete range for weight one. The
        // original trainer checks it before touching the general cursor.
        if self
            .unit_weight
            .is_some_and(|(start, length)| position.wrapping_sub(start) < length)
        {
            return 1;
        }
        *self
            .intervals
            .get(position)
            .expect("a matched edge belongs to a word")
    }
}
impl<'a> PreparedCorpus<'a> {
    pub(super) fn build(
        word_counts: &'a AHashMap<CompactString, u64>,
        vocabulary: &mut Vocabulary,
        policy: IdentityPolicy,
        length_limited: bool,
        progress: &TrainingProgress,
    ) -> Result<Self> {
        let work = progress.stage("Resolve initial IDs", word_counts.len());
        let initial_ids = vocabulary.initial_ids(word_counts, &work)?;
        let mut words: Vec<_> = word_counts
            .iter()
            .map(|(word, &weight)| (word, weight))
            .collect();
        let work = progress.stage("Arrange weighted words", words.len());
        words.par_sort_unstable_by(|left, right| right.1.cmp(&left.1));
        work.complete(words.len());
        let work = progress.stage("Measure corpus", words.len());
        let measured: Vec<usize> = words
            .par_iter()
            .map(|(word, _)| {
                if initial_ids.complete_alphabet() {
                    word.chars().count()
                } else {
                    word.chars()
                        .filter(|&character| initial_ids.retained(character))
                        .count()
                }
            })
            .collect();
        work.complete(words.len());
        let mut word_starts = Vec::with_capacity(words.len());
        let mut interval_starts = Vec::new();
        let mut interval_weights = Vec::new();
        let mut slots = 1_usize;
        let mut edges = 0_usize;
        let mut weighted_mass = 0_u128;
        let mut maximum_weight = 0_u64;
        for ((_, weight), &symbols) in words.iter().zip(&measured) {
            weighted_mass += u128::from(*weight) * symbols.saturating_sub(1) as u128;
            maximum_weight = maximum_weight.max(*weight);
            word_starts.push(slots as u64);
            if interval_weights.last() != Some(weight) {
                interval_starts.push(slots as u64);
                interval_weights.push(*weight);
            }
            slots = slots
                .checked_add(symbols)
                .and_then(|size| size.checked_add(1))
                .ok_or("BPE corpus exceeds resident index bounds")?;
            edges = edges
                .checked_add(symbols.saturating_sub(1))
                .ok_or("BPE edge count exceeds usize")?;
        }
        if slots > isize::MAX as usize / std::mem::size_of::<AtomicU32>() {
            return Err("BPE corpus exceeds resident allocation bounds".into());
        }
        // Preserve the signed input bound for configurations that may reuse
        // an identity, even while their first attempt accepts fresh IDs only.
        if (policy == IdentityPolicy::Reusable || vocabulary.has_affixes())
            && (weighted_mass > i64::MAX as u128 || maximum_weight > i64::MAX as u64)
        {
            return Err("BPE identity-reuse weighted edge mass or word weight exceeds i64".into());
        }
        let words: Vec<_> = words
            .into_iter()
            .zip(word_starts)
            .map(|((word, _), start)| PlannedWord { word, start })
            .collect();
        let mut checkpoints = Vec::new();
        // Byte checkpoints also bound seeks through heavily filtered words.
        // Repeated positions are valid: a filtered byte range consumes no slots;
        // partition_point selects the final anchor before the target token.
        const CHECKPOINT_BYTES: usize = 4096;
        for planned in &words {
            if planned.word.len() <= CHECKPOINT_BYTES {
                continue;
            }
            let mut retained = 0_usize;
            let mut last_byte = 0_usize;
            for (byte, character) in planned.word.char_indices() {
                if byte - last_byte >= CHECKPOINT_BYTES {
                    checkpoints.push(SymbolCheckpoint {
                        position: planned.start as usize + retained,
                        byte,
                    });
                    last_byte = byte;
                }
                retained += usize::from(initial_ids.retained(character));
            }
        }
        let unit_weight = interval_weights
            .iter()
            .position(|&weight| weight == 1)
            .map(|index| {
                let start = interval_starts[index];
                let end = interval_starts
                    .get(index + 1)
                    .copied()
                    .unwrap_or(slots as u64);
                (start, end - start)
            });
        Ok(Self {
            words,
            checkpoints,
            initial_ids,
            len: slots,
            weights: IntervalIndex::new(interval_starts, interval_weights),
            unit_weight,
            spans_by_id: vocabulary.initial_spans(),
            scan_whole_words: length_limited,
            edges,
        })
    }
    pub(super) fn initial_edges(&self) -> usize {
        self.edges
    }
    pub(super) fn materialize(
        self,
        workers: usize,
        policy: IdentityPolicy,
        progress: &TrainingProgress,
    ) -> Result<Corpus> {
        let slots = self.len;
        let words = &self.words;
        let initial_ids = &self.initial_ids;
        let mut tokens = Vec::<MaybeUninit<AtomicU32>>::new();
        tokens
            .try_reserve_exact(slots)
            .map_err(|_| "BPE corpus allocation failed")?;
        // PERF: Fill each slot once in its owning word job. Preinitializing the
        // whole plane would add a serial store pass before the parallel writes.
        // SAFETY: MaybeUninit elements may be uninitialized. The prefix and every
        // disjoint job region are fully written before conversion below.
        unsafe {
            tokens.set_len(slots);
        }
        tokens[0].write(AtomicU32::new(WORD_SEPARATOR_ID));
        let chunk = words.len().div_ceil(workers * 8).max(1);
        let work = progress.stage("Fill corpus", slots - 1);
        let mut jobs = Vec::new();
        let mut remaining = &mut tokens[1..];
        for start in (0..words.len()).step_by(chunk) {
            let end = (start + chunk).min(words.len());
            let base = words[start].start as usize;
            let after = words.get(end).map_or(slots, |word| word.start as usize);
            let (region, next) = remaining.split_at_mut(after - base);
            remaining = next;
            jobs.push((start, end, region));
        }
        jobs.into_par_iter().for_each(|(start, end, region)| {
            let mut position = 0;
            for planned in &words[start..end] {
                let word = planned.word;
                if initial_ids.plain() {
                    for character in word.chars() {
                        if let Some(id) = initial_ids.plain_id(character) {
                            region[position].write(AtomicU32::new(id));
                            position += 1;
                        }
                    }
                } else {
                    for (byte, character) in word.char_indices() {
                        if let Some(id) = initial_ids.id(
                            character,
                            byte == 0,
                            byte + character.len_utf8() == word.len(),
                        ) {
                            region[position].write(AtomicU32::new(id));
                            position += 1;
                        }
                    }
                }
                // Empty filtered words also own one initialized separator.
                region[position].write(AtomicU32::new(WORD_SEPARATOR_ID));
                position += 1;
            }
            work.complete(region.len());
        });
        let mut tokens = ManuallyDrop::new(tokens);
        // SAFETY: All word jobs joined after writing every measured token and
        // separator. MaybeUninit<AtomicU32> has the same layout as AtomicU32;
        // the new vector takes sole ownership of the allocation.
        let tokens = unsafe {
            Vec::from_raw_parts(
                tokens.as_mut_ptr().cast::<AtomicU32>(),
                tokens.len(),
                tokens.capacity(),
            )
        };
        let word_starts = if policy == IdentityPolicy::Reusable {
            self.words.iter().map(|word| word.start).collect()
        } else {
            Vec::new()
        };
        Ok(Corpus {
            slots: tokens,
            word_starts,
            weights: self.weights,
            unit_weight: self.unit_weight,
            spans_by_id: self.spans_by_id,
            occurrence_spans: None,
            scan_whole_words: self.scan_whole_words,
        })
    }
}
impl Corpus {
    #[cfg(test)]
    pub(super) fn initial_view(&self) -> InitialCorpus<'_> {
        InitialCorpus {
            token_ids: &self.slots,
            word_weights: &self.weights,
        }
    }
    pub(super) fn len(&self) -> usize {
        self.slots.len()
    }
    #[inline]
    pub(super) fn token(&self, position: u64) -> u32 {
        self.slots[position as usize].load(Ordering::Relaxed)
    }
    #[inline]
    pub(super) fn span(&self, position: u64) -> u64 {
        match &self.occurrence_spans {
            Some(spans) => spans[position as usize],
            None => self.spans_by_id[self.token(position) as usize],
        }
    }
    pub(super) fn span_by_id(&self, id: u32) -> u64 {
        self.spans_by_id[id as usize]
    }
    pub(super) fn matcher(&self, pair: Pair) -> PairMatcher<'_> {
        PairMatcher {
            corpus: self,
            pair,
            left_span: self.spans_by_id[pair.0 as usize],
            right_span: self.spans_by_id[pair.1 as usize],
        }
    }
    pub(super) fn word_containing(&self, position: u64) -> usize {
        self.word_starts.partition_point(|&start| start <= position) - 1
    }
    pub(super) fn word_end(&self, word: usize) -> u64 {
        self.word_starts
            .get(word + 1)
            .copied()
            .unwrap_or(self.len() as u64)
    }
    pub(super) fn word_start(&self, word: usize) -> u64 {
        self.word_starts[word]
    }

    pub(super) fn weight_cursor(&self) -> WordWeightCursor<'_> {
        WordWeightCursor {
            intervals: self.weights.cursor(),
            uniform_weight: match self.weights.values() {
                [weight] => Some(*weight),
                _ => None,
            },
            unit_weight: self.unit_weight,
        }
    }
    pub(super) fn needs_word_scan(&self) -> bool {
        self.scan_whole_words
    }
    pub(super) fn prepare_spans(&mut self, pair: Pair, replacement: u32, reused_active: bool) {
        self.scan_whole_words |= reused_active;
        let span = self.spans_by_id[pair.0 as usize] + self.spans_by_id[pair.1 as usize];
        if replacement as usize == self.spans_by_id.len() {
            self.spans_by_id.push(if self.occurrence_spans.is_some() {
                0
            } else {
                span
            });
        } else if self.occurrence_spans.is_none() {
            let previous = self.spans_by_id[replacement as usize];
            if previous == 0 || previous == span {
                self.spans_by_id[replacement as usize] = span;
                return;
            }
            let mut spans = vec![0; self.len()];
            for &start in &self.word_starts {
                let mut position = start;
                while self.token(position) != WORD_SEPARATOR_ID {
                    let span = self.spans_by_id[self.token(position) as usize];
                    spans[position as usize] = span;
                    spans[(position + span - 1) as usize] = span;
                    position += span;
                }
            }
            self.occurrence_spans = Some(spans);
        }
    }
    #[inline]
    pub(super) fn write_endpoints(&self, matched: PairMatch, replacement: u32) {
        write_endpoints(&self.slots, matched, replacement);
    }
    pub(super) fn word_writers(
        &mut self,
        regions: &[std::ops::Range<u64>],
    ) -> Option<Vec<WordWriter<'_>>> {
        let spans = self.occurrence_spans.as_mut()?;
        let mut remaining = spans.as_mut_slice();
        let mut writers = Vec::with_capacity(regions.len());
        for region in regions {
            let (spans, next) = remaining.split_at_mut((region.end - region.start) as usize);
            remaining = next;
            writers.push(WordWriter {
                base: region.start,
                slots: &self.slots[region.start as usize..region.end as usize],
                spans,
            });
        }
        Some(writers)
    }
    pub(super) fn has_occurrence_spans(&self) -> bool {
        self.occurrence_spans.is_some()
    }
    #[inline]
    pub(super) fn prefetch(&self, position: u64) {
        #[cfg(target_arch = "x86_64")]
        if position < self.len() as u64 {
            // SAFETY: this address is inside the live slot allocation. Prefetch
            // does not load a value or change the phase-ordering contract.
            unsafe {
                std::arch::x86_64::_mm_prefetch(
                    self.slots.as_ptr().add(position as usize).cast(),
                    std::arch::x86_64::_MM_HINT_T0,
                );
            }
        }
    }
}

// Prepared plans own disjoint live-token spans. Pool joins delimit the
// Relaxed stores; this helper also serves exclusive whole-word regions.
#[inline]
fn write_endpoints(slots: &[AtomicU32], matched: PairMatch, replacement: u32) {
    slots[matched.left_start as usize].store(replacement, Ordering::Relaxed);
    if matched.right_start + 1 != matched.next_start {
        slots[matched.right_start as usize].store(WORD_SEPARATOR_ID, Ordering::Relaxed);
    }
    slots[(matched.next_start - 1) as usize].store(replacement, Ordering::Relaxed);
}
/// A complete immutable word region owns its occurrence-span writes as well.
pub(super) struct WordWriter<'a> {
    base: u64,
    slots: &'a [AtomicU32],
    spans: &'a mut [u64],
}
impl WordWriter<'_> {
    pub(super) fn merge(&mut self, position: u64, replacement: u32) {
        let left = (position - self.base) as usize;
        let right = left + self.spans[left] as usize;
        let after = right + self.spans[right] as usize;
        let merged = (after - left) as u64;
        write_endpoints(
            self.slots,
            PairMatch {
                left_start: left as u64,
                right_start: right as u64,
                next_start: after as u64,
                merged_span: merged,
            },
            replacement,
        );
        if right + 1 != after {
            self.spans[right] = 0;
        }
        self.spans[left] = merged;
        self.spans[after - 1] = merged;
    }
}
/// Fixed rule geometry is cached once. Unequal identity reuse reads its
/// occurrence plane from the same immutable snapshot instead.
pub(super) struct PairMatcher<'a> {
    corpus: &'a Corpus,
    pair: Pair,
    left_span: u64,
    right_span: u64,
}
impl PairMatcher<'_> {
    #[inline]
    pub(super) fn geometry(&self, left_start: u64) -> PairMatch {
        let left_span = self
            .corpus
            .occurrence_spans
            .as_ref()
            .map_or(self.left_span, |spans| spans[left_start as usize]);
        let right_start = left_start + left_span;
        let right_span = self
            .corpus
            .occurrence_spans
            .as_ref()
            .map_or(self.right_span, |spans| spans[right_start as usize]);
        PairMatch {
            left_start,
            right_start,
            next_start: right_start + right_span,
            merged_span: left_span + right_span,
        }
    }
    #[inline]
    pub(super) fn get(&self, left_start: u64) -> Option<PairMatch> {
        if self.corpus.token(left_start) != self.pair.0 {
            return None;
        }
        let left_span = self
            .corpus
            .occurrence_spans
            .as_ref()
            .map_or(self.left_span, |spans| spans[left_start as usize]);
        let right_start = left_start + left_span;
        if left_span == 0
            || right_start >= self.corpus.len() as u64
            || self.corpus.token(right_start) != self.pair.1
        {
            return None;
        }
        let right_span = self
            .corpus
            .occurrence_spans
            .as_ref()
            .map_or(self.right_span, |spans| spans[right_start as usize]);
        let next_start = right_start + right_span;
        if right_span == 0 || next_start >= self.corpus.len() as u64 {
            return None;
        }
        Some(PairMatch {
            left_start,
            right_start,
            next_start,
            merged_span: left_span + right_span,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        IdentityPolicy,
        execution::Execution,
        initial_pairs,
        merge::{self, MergeRule},
        pair_index::{PairIndex, key_pair},
    };
    use super::*;

    #[test]
    fn unit_weight_range_preserves_boundaries_and_cursor_resets() {
        let start = 1_u64 << 32;
        let weights = IntervalIndex::new(vec![1, start, start + 4], vec![3, 1, 0]);
        let mut cursor = WordWeightCursor {
            intervals: weights.cursor(),
            uniform_weight: None,
            unit_weight: Some((start, 4)),
        };
        for (position, expected) in [
            (start, 1),
            (start + 3, 1),
            (start + 4, 0),
            (start - 1, 3),
            (start + 1, 1),
            (u64::MAX, 0),
            (1, 3),
        ] {
            assert_eq!(cursor.weight(position), expected);
        }
        for weight in [0, 2, u64::MAX] {
            let weights = IntervalIndex::new(vec![1], vec![weight]);
            let mut cursor = WordWeightCursor {
                intervals: weights.cursor(),
                uniform_weight: Some(weight),
                unit_weight: None,
            };
            for position in [1, start, u64::MAX, 2] {
                assert_eq!(cursor.weight(position), weight);
            }
        }
    }
    use tk_collections::AllocationArena;
    use tk_encode::utils::progress::ProgressFormat;

    #[test]
    fn cohort_cohorts_keep_distinct_words_and_unequal_occurrence_spans() {
        // This synthetic identity state tests the corpus/cohort contract directly.
        // Plain concatenation cannot reuse an active ID this way.
        let mut corpus = Corpus {
            slots: [
                WORD_SEPARATOR_ID,
                0,
                1,
                2,
                WORD_SEPARATOR_ID,
                3,
                2,
                WORD_SEPARATOR_ID,
            ]
            .into_iter()
            .map(AtomicU32::new)
            .collect(),
            word_starts: vec![1, 5],
            weights: IntervalIndex::new(vec![1, 5], vec![3, 1]),
            unit_weight: Some((5, 4)),
            spans_by_id: vec![1; 4],
            occurrence_spans: None,
            scan_whole_words: false,
        };
        let execution = Execution::new(2).unwrap();
        let arena = AllocationArena::new(2, 3);
        let progress = TrainingProgress::new(false, ProgressFormat::Silent).unwrap();
        execution.pool.install(|| {
            let initial = initial_pairs::build_initial_pairs(
                corpus.initial_view(),
                0,
                &execution,
                &arena,
                &progress,
            )
            .unwrap();
            let mut index =
                PairIndex::from_initial_pairs(initial, IdentityPolicy::Reusable, 2).unwrap();
            for (round, (pair, count, position, replacement, reused)) in [
                ((0, 1), 3, 1, 3, true),
                ((3, 2), 4, 1, 4, false),
                ((3, 2), 4, 5, 4, true),
            ]
            .into_iter()
            .enumerate()
            {
                let priority = index.best().unwrap();
                assert_eq!(
                    (key_pair(priority.key), priority.priority_count),
                    (pair, count)
                );
                let candidate = index.take_best();
                assert_eq!(candidate.positions.iter().collect::<Vec<_>>(), [position]);
                corpus.prepare_spans(pair, replacement, reused);
                let rules = [MergeRule { pair, replacement }];
                let prepared = merge::prepare_merges(
                    &corpus,
                    &rules,
                    &[candidate],
                    IdentityPolicy::Reusable,
                    corpus.spans_by_id.len(),
                    usize::MAX,
                    &execution,
                )
                .unwrap();
                let events = prepared.apply(&mut corpus);
                index
                    .commit_merges(&events, corpus.spans_by_id.len(), &execution, &arena)
                    .unwrap();
                drop(events);
                if round == 0 {
                    assert_eq!((corpus.span(1), corpus.span(5)), (2, 1));
                }
                if round == 1 {
                    assert_eq!((corpus.token(5), corpus.span(1)), (3, 3));
                }
            }
            assert_eq!((corpus.token(1), corpus.span(1)), (4, 3));
            assert_eq!((corpus.token(5), corpus.span(5)), (4, 2));
            assert!(index.best().is_none());
        });
    }

    #[test]
    fn intermediate_birth_survives_its_later_boundary_removal() {
        let mut corpus = Corpus {
            slots: [WORD_SEPARATOR_ID, 0, 0, 0, 0, WORD_SEPARATOR_ID]
                .into_iter()
                .map(AtomicU32::new)
                .collect(),
            word_starts: vec![1],
            weights: IntervalIndex::new(vec![1], vec![1]),
            unit_weight: Some((1, 5)),
            spans_by_id: vec![1],
            occurrence_spans: None,
            scan_whole_words: true,
        };
        // Independent mainline Word semantics: the first rewrite births a
        // length-three boundary; the next rewrite removes it. Its cohort still
        // owns the word even though the final length-four boundary is gated out.
        let mut reference = super::super::super::word::Word::new();
        for _ in 0..4 {
            reference.add(0, 1);
        }
        assert_eq!(
            reference.merge(0, 0, 0, 4),
            [((0, 0), -1), ((0, 0), 1), ((0, 0), -1)]
        );
        let execution = Execution::new(2).unwrap();
        let arena = AllocationArena::new(2, 3);
        let progress = TrainingProgress::new(false, ProgressFormat::Silent).unwrap();
        execution.pool.install(|| {
            let initial = initial_pairs::build_initial_pairs(
                corpus.initial_view(),
                0,
                &execution,
                &arena,
                &progress,
            )
            .unwrap();
            let mut index =
                PairIndex::from_initial_pairs(initial, IdentityPolicy::Reusable, 1).unwrap();
            assert_eq!(index.best().unwrap().priority_count, 3);
            let candidate = index.take_best();
            corpus.prepare_spans((0, 0), 0, true);
            let rules = [MergeRule {
                pair: (0, 0),
                replacement: 0,
            }];
            let prepared = merge::prepare_merges(
                &corpus,
                &rules,
                &[candidate],
                IdentityPolicy::Reusable,
                1,
                4,
                &execution,
            )
            .unwrap();
            let events = prepared.apply(&mut corpus);
            let changes: Vec<_> = events
                .chunks
                .iter()
                .flat_map(|chunk| chunk.changes.iter())
                .collect();
            assert_eq!(
                changes
                    .iter()
                    .map(|change| change.removed_weight)
                    .sum::<u64>(),
                2
            );
            assert_eq!(
                changes.iter().map(|change| change.born_weight).sum::<u64>(),
                1
            );
            let births: Vec<_> = events
                .chunks
                .iter()
                .flat_map(|chunk| {
                    chunk.changes.iter().flat_map(|change| {
                        chunk.chains[super::super::pair_index::shard_for(
                            change.born_key,
                            chunk.chains.len(),
                        )]
                        .reversed(change.positions)
                    })
                })
                .collect();
            assert_eq!(births, [1]);
            index
                .commit_merges(&events, corpus.spans_by_id.len(), &execution, &arena)
                .unwrap();
            drop(events);
            assert_eq!(index.best().unwrap().priority_count, 2);
            let cohort = index.take_best();
            assert_eq!(cohort.positions.iter().collect::<Vec<_>>(), [1]);
            assert_eq!((corpus.span(1), corpus.span(3)), (2, 2));
        });
    }
}
