//! Immutable corpus planning, checkpoint seeking and deferred materialization.
use super::super::{
    IdentityPolicy, WORD_SEPARATOR_ID,
    pair_index::pair_key,
    storage::IntervalIndex,
    vocabulary::{InitialTokenIds, Vocabulary},
};
use super::{Corpus, InitialPairSource, SlotStorage};
use crate::progress::TrainingProgress;
use crate::trainers::bpe::word_counts::WordCountsView;
use compact_str::CompactString;
use rayon::prelude::*;
use std::{
    mem::{ManuallyDrop, MaybeUninit},
    ops::{ControlFlow, Range},
    sync::atomic::AtomicU32,
};
use tk_encode::Result;

// A word keeps its original byte coordinates and its measured global start.
// Initial pair construction borrows this plan; no mutable slot allocation exists
// until all raw records have retired.
struct PlannedWord<'input> {
    word: &'input CompactString,
    start: u64,
}
struct SymbolCheckpoint {
    slot_position: usize,
    byte_offset: usize,
}
pub(in super::super) struct CorpusPlan<'input> {
    words: Vec<PlannedWord<'input>>,
    checkpoints: Vec<SymbolCheckpoint>,
    initial_ids: InitialTokenIds,
    len: usize,
    weights: IntervalIndex<u64>,
    unit_weight: Option<(u64, u64)>,
    spans_by_id: Vec<u64>,
    scan_whole_words: bool,
    edges: usize,
    weighted_mass: u128,
}
impl InitialPairSource for &CorpusPlan<'_> {
    fn len(&self) -> usize {
        self.len
    }
    fn word_weights(&self) -> &IntervalIndex<u64> {
        &self.weights
    }
    fn edge_count(&self, range: Range<usize>) -> usize {
        // The plan has already measured retained symbols. Each word owns its
        // left endpoints through the penultimate symbol, including wave cuts.
        let first = self
            .words
            .partition_point(|word| word.start <= range.start as u64)
            .saturating_sub(1);
        let mut count = 0;
        for index in first..self.words.len() {
            let start = self.words[index].start as usize;
            if start >= range.end {
                break;
            }
            let after = self
                .words
                .get(index + 1)
                .map_or(self.len, |word| word.start as usize);
            let end = after.saturating_sub(2).max(start);
            count += end.min(range.end).saturating_sub(start.max(range.start));
        }
        count
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
            let (mut position, byte_start) = self.scan_start(index, range.start);
            let mut previous = None;
            self.initial_ids
                .scan_symbols(planned.word, byte_start, |id| {
                    if let Some((left_position, left)) = previous
                        && left_position >= range.start
                        && left_position < range.end
                    {
                        emit(left_position, pair_key((left, id)));
                    }
                    previous = Some((position, id));
                    // The right endpoint at range.end has supplied the lookahead.
                    if position >= range.end {
                        return ControlFlow::Break(());
                    }
                    position += 1;
                    ControlFlow::Continue(())
                });
        }
    }
}
impl<'input> CorpusPlan<'input> {
    pub(in super::super) fn build(
        word_counts: WordCountsView<'input>,
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
        // Long words are measured by independent UTF-8 byte chunks. Retain
        // these counts for checkpoints instead of rescanning their characters.
        const CHECKPOINT_BYTES: usize = 4096;
        let mut byte_chunks = Vec::new();
        for (index, (word, _)) in words.iter().enumerate() {
            if word.len() <= CHECKPOINT_BYTES {
                continue;
            }
            for byte in (0..word.len()).step_by(CHECKPOINT_BYTES) {
                let mut begin = byte;
                while !word.is_char_boundary(begin) {
                    begin += 1;
                }
                if begin == word.len() {
                    break;
                }
                let mut end = (byte + CHECKPOINT_BYTES).min(word.len());
                while !word.is_char_boundary(end) {
                    end += 1;
                }
                byte_chunks.push((index, begin..end));
            }
        }
        let count_symbols = |text: &str| {
            if initial_ids.complete_alphabet() {
                text.chars().count()
            } else {
                text.chars()
                    .filter(|&character| initial_ids.retained(character))
                    .count()
            }
        };
        let counts: Vec<usize> = byte_chunks
            .par_iter()
            .map(|(index, range)| count_symbols(&words[*index].0[range.clone()]))
            .collect();
        let mut measured: Vec<usize> = words
            .par_iter()
            .map(|(word, _)| {
                if word.len() <= CHECKPOINT_BYTES {
                    count_symbols(word)
                } else {
                    0
                }
            })
            .collect();
        for ((index, _), count) in byte_chunks.iter().zip(&counts) {
            measured[*index] += count;
        }
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
        if (policy == IdentityPolicy::AllowActiveReuse || !initial_ids.plain())
            && (weighted_mass > i64::MAX as u128 || maximum_weight > i64::MAX as u64)
        {
            return Err("BPE identity-reuse weighted edge mass or word weight exceeds i64".into());
        }
        let words: Vec<_> = words
            .into_iter()
            .zip(word_starts)
            .map(|((word, _), start)| PlannedWord { word, start })
            .collect();
        // Ordered per-word prefixes restore global anchors, including repeated
        // positions across byte chunks whose characters were all filtered out.
        let mut checkpoints = Vec::new();
        let mut current_word = None;
        let mut retained = 0;
        for ((index, range), count) in byte_chunks.into_iter().zip(counts) {
            if current_word != Some(index) {
                current_word = Some(index);
                retained = 0;
            }
            if range.start != 0 {
                checkpoints.push(SymbolCheckpoint {
                    slot_position: words[index].start as usize + retained,
                    byte_offset: range.start,
                });
            }
            retained += count;
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
            weighted_mass,
        })
    }
    pub(in super::super) fn initial_counts_fit_u64(&self) -> bool {
        self.weighted_mass <= u128::from(u64::MAX)
    }
    pub(in super::super) fn initial_edges(&self) -> usize {
        self.edges
    }
    pub(in super::super) fn materialize<S: SlotStorage>(
        self,
        workers: usize,
        policy: IdentityPolicy,
        progress: &TrainingProgress,
    ) -> Result<Corpus<S>> {
        let work = progress.stage("Fill corpus", self.len - 1);
        let tokens = S::from_prepared(&self, workers, &work)?;
        let word_starts = if policy == IdentityPolicy::AllowActiveReuse {
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
    fn scan_start(&self, index: usize, slot_position: usize) -> (usize, usize) {
        let word_start = self.words[index].start as usize;
        if word_start < slot_position {
            let after = self
                .checkpoints
                .partition_point(|point| point.slot_position <= slot_position);
            if let Some(point) = after.checked_sub(1).map(|index| &self.checkpoints[index])
                && point.slot_position >= word_start
            {
                return (point.slot_position, point.byte_offset);
            }
        }
        (word_start, 0)
    }
    fn for_each_word_token(
        &self,
        index: usize,
        range: Range<usize>,
        mut emit: impl FnMut(usize, u32),
    ) {
        let planned = &self.words[index];
        let separator = self
            .words
            .get(index + 1)
            .map_or(self.len, |word| word.start as usize)
            - 1;
        if range.start < separator {
            let (mut position, byte_start) = self.scan_start(index, range.start);
            self.initial_ids
                .scan_symbols(planned.word, byte_start, |id| {
                    if position >= range.end {
                        return ControlFlow::Break(());
                    }
                    if position >= range.start {
                        emit(position, id);
                    }
                    position += 1;
                    ControlFlow::Continue(())
                });
        }
        if range.contains(&separator) {
            emit(separator, WORD_SEPARATOR_ID);
        }
    }
    pub(super) fn fill_tokens<T: Send>(
        &self,
        workers: usize,
        work: &crate::progress::WorkProgress,
        make: impl Fn(usize, u32) -> T + Sync,
    ) -> Result<Vec<T>> {
        let slots = self.len;
        let words = &self.words;
        let initial_ids = &self.initial_ids;
        let mut tokens = Vec::<MaybeUninit<T>>::new();
        tokens
            .try_reserve_exact(slots.checked_add(1).ok_or("BPE corpus size overflow")?)
            .map_err(|_| "BPE corpus allocation failed")?;
        // PERF: Fill each slot once in its owning word job. Preinitializing the
        // whole plane would add a serial store pass before the parallel writes.
        // SAFETY: MaybeUninit elements may be uninitialized. The prefix and every
        // disjoint job region are fully written before conversion below.
        unsafe {
            tokens.set_len(slots);
        }
        tokens[0].write(make(0, WORD_SEPARATOR_ID));
        let chunk = (slots - 1).div_ceil(workers * 8).max(1);
        let mut specs = Vec::new();
        let mut start = 0;
        while start < words.len() {
            let base = words[start].start as usize;
            let after = words
                .get(start + 1)
                .map_or(slots, |word| word.start as usize);
            if after - base > chunk {
                for begin in (base..after).step_by(chunk) {
                    specs.push((begin, (begin + chunk).min(after), start, start + 1, true));
                }
                start += 1;
                continue;
            }
            let target = base + chunk;
            let mut end = words.partition_point(|word| (word.start as usize) < target);
            let last = end - 1;
            let last_after = words.get(end).map_or(slots, |word| word.start as usize);
            // Leave a crossing large word for the next iteration to split.
            if last > start && last_after - words[last].start as usize > chunk {
                end -= 1;
            }
            let after = words.get(end).map_or(slots, |word| word.start as usize);
            specs.push((base, after, start, end, false));
            start = end;
        }
        let mut jobs = Vec::with_capacity(specs.len());
        let mut remaining = &mut tokens[1..];
        for (base, after, start, end, segment) in specs {
            let (region, next) = remaining.split_at_mut(after - base);
            remaining = next;
            jobs.push((base, start, end, segment, region));
        }
        debug_assert!(remaining.is_empty());
        jobs.into_par_iter()
            .for_each(|(base, start, end, segment, region)| {
                if segment {
                    let mut written = 0;
                    self.for_each_word_token(start, base..base + region.len(), |position, id| {
                        region[position - base].write(make(position, id));
                        written += 1;
                    });
                    debug_assert_eq!(written, region.len());
                    work.complete(region.len());
                    return;
                }
                let mut position = 0;
                for planned in &words[start..end] {
                    let word = planned.word;
                    initial_ids.scan_symbols(word, 0, |id| {
                        region[position].write(make(base + position, id));
                        position += 1;
                        ControlFlow::Continue(())
                    });
                    // Empty filtered words also own one initialized separator.
                    region[position].write(make(base + position, WORD_SEPARATOR_ID));
                    position += 1;
                }
                debug_assert_eq!(position, region.len());
                work.complete(region.len());
            });
        let mut tokens = ManuallyDrop::new(tokens);
        // SAFETY: All word jobs joined after writing every measured token and
        // separator. MaybeUninit<T> has the same layout as T;
        // the new vector takes sole ownership of the allocation.
        let tokens = unsafe {
            Vec::from_raw_parts(
                tokens.as_mut_ptr().cast::<T>(),
                tokens.len(),
                tokens.capacity(),
            )
        };
        Ok(tokens)
    }
}
