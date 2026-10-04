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
    sync::atomic::{AtomicU8, AtomicU16, AtomicU32, Ordering},
};
use tk_collections::{IntervalCursor, IntervalIndex};
use tk_encode::{Result, models::bpe::Pair};

// Reserve one code beyond admitted IDs for the separator. Layout is chosen
// once: sixteen bits for small vocabularies, three bytes through 24 bits, and
// the complete u32 domain beyond that.
pub(super) fn slot_bits(id_count: usize) -> u8 {
    let required = (usize::BITS - id_count.leading_zeros()) as u8;
    if required <= 16 {
        16
    } else if required <= 24 {
        24
    } else {
        32
    }
}
// The coordinator chooses one storage type before training. Static dispatch
// keeps layout tests and variable-width bit arithmetic outside endpoint loops.
pub(super) trait SlotStorage: Send + Sync + Sized {
    fn from_prepared(
        prepared: &PreparedCorpus<'_>,
        workers: usize,
        work: &crate::progress::WorkProgress,
    ) -> Result<Self>;
    fn len(&self) -> usize;
    fn load(&self, position: usize) -> u32;
    /// # Safety
    /// Stores belong to a joined write phase with no concurrent token readers.
    /// Each writer owns distinct logical slots until that phase joins.
    unsafe fn store(&self, position: usize, id: u32);
    #[cfg(target_arch = "x86_64")]
    fn prefetch_pointer(&self, position: usize) -> *const i8;
}
pub(super) type FullSlots = Vec<AtomicU32>;
impl SlotStorage for FullSlots {
    fn from_prepared(
        prepared: &PreparedCorpus<'_>,
        workers: usize,
        work: &crate::progress::WorkProgress,
    ) -> Result<Self> {
        prepared.fill_tokens(workers, work, |_, id| AtomicU32::new(id))
    }
    #[inline]
    fn len(&self) -> usize {
        self.as_slice().len()
    }
    #[inline]
    fn load(&self, position: usize) -> u32 {
        self[position].load(Ordering::Relaxed)
    }
    #[inline]
    unsafe fn store(&self, position: usize, id: u32) {
        self[position].store(id, Ordering::Relaxed);
    }
    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn prefetch_pointer(&self, position: usize) -> *const i8 {
        self.as_ptr().wrapping_add(position).cast()
    }
}
pub(super) struct HalfSlots(Vec<AtomicU16>);
impl SlotStorage for HalfSlots {
    fn from_prepared(
        prepared: &PreparedCorpus<'_>,
        workers: usize,
        work: &crate::progress::WorkProgress,
    ) -> Result<Self> {
        Ok(Self(prepared.fill_tokens(workers, work, |_, id| {
            AtomicU16::new(id as u16)
        })?))
    }
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }
    #[inline]
    fn load(&self, position: usize) -> u32 {
        let id = self.0[position].load(Ordering::Relaxed);
        if id == u16::MAX {
            WORD_SEPARATOR_ID
        } else {
            u32::from(id)
        }
    }
    #[inline]
    unsafe fn store(&self, position: usize, id: u32) {
        debug_assert!(id == WORD_SEPARATOR_ID || id < u32::from(u16::MAX));
        self.0[position].store(id as u16, Ordering::Relaxed);
    }
    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn prefetch_pointer(&self, position: usize) -> *const i8 {
        self.0.as_ptr().wrapping_add(position).cast()
    }
}
// The last initialized guard slot makes the final scalar four-byte read
// valid. It is outside the logical plane and is never rewritten.
pub(super) struct ThreeByteSlots {
    tokens: Vec<[AtomicU8; 3]>,
}
impl ThreeByteSlots {
    const SEPARATOR: u32 = 0x00ff_ffff;
    fn bytes(id: u32) -> [AtomicU8; 3] {
        let bytes = id.to_le_bytes();
        std::array::from_fn(|index| AtomicU8::new(bytes[index]))
    }
}
impl SlotStorage for ThreeByteSlots {
    fn from_prepared(
        prepared: &PreparedCorpus<'_>,
        workers: usize,
        work: &crate::progress::WorkProgress,
    ) -> Result<Self> {
        let mut tokens = prepared.fill_tokens(workers, work, |_, id| Self::bytes(id))?;
        // fill_tokens reserves this guard before its parallel initialization,
        // avoiding a second allocation or a full-plane copy here.
        tokens.push(Self::bytes(0));
        Ok(Self { tokens })
    }
    #[inline]
    fn len(&self) -> usize {
        self.tokens.len() - 1
    }
    #[inline]
    fn load(&self, position: usize) -> u32 {
        assert!(position < self.len());
        // SAFETY: AtomicU8 has u8's size, alignment and valid representations.
        // All four bytes are initialized inside this allocation, including
        // the guard for the last slot. Writes require an exclusive joined
        // phase through unsafe store; concurrent read-only accesses are valid.
        // The next slot's low byte is masked off after this scalar read.
        let raw = unsafe {
            self.tokens
                .as_ptr()
                .add(position)
                .cast::<u32>()
                .read_unaligned()
        };
        let id = u32::from_le(raw) & Self::SEPARATOR;
        if id == Self::SEPARATOR {
            WORD_SEPARATOR_ID
        } else {
            id
        }
    }
    #[inline]
    unsafe fn store(&self, position: usize, id: u32) {
        debug_assert!(id == WORD_SEPARATOR_ID || id < Self::SEPARATOR);
        let bytes = id.to_le_bytes();
        let slot = &self.tokens[..self.len()][position];
        for index in 0..3 {
            slot[index].store(bytes[index], Ordering::Relaxed);
        }
    }
    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn prefetch_pointer(&self, position: usize) -> *const i8 {
        self.tokens.as_ptr().wrapping_add(position).cast()
    }
}
pub(super) struct Corpus<S: SlotStorage = FullSlots> {
    slots: S,
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
    weighted_mass: u128,
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
#[cfg(test)]
impl<S: SlotStorage> InitialPairSource for &Corpus<S> {
    fn len(&self) -> usize {
        self.slots.len()
    }
    fn word_weights(&self) -> &IntervalIndex<u64> {
        &self.weights
    }
    fn for_each_edge(&self, range: Range<usize>, mut emit: impl FnMut(usize, u64)) {
        for position in range {
            let left = self.slots.load(position);
            let right = self.slots.load(position + 1);
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
                    position: words[index].start as usize + retained,
                    byte: range.start,
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
    pub(super) fn initial_counts_fit_u64(&self) -> bool {
        self.weighted_mass <= u128::from(u64::MAX)
    }
    pub(super) fn initial_edges(&self) -> usize {
        self.edges
    }
    pub(super) fn materialize<S: SlotStorage>(
        self,
        workers: usize,
        policy: IdentityPolicy,
        progress: &TrainingProgress,
    ) -> Result<Corpus<S>> {
        let work = progress.stage("Fill corpus", self.len - 1);
        let tokens = S::from_prepared(&self, workers, &work)?;
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
    fn for_each_word_token(
        &self,
        index: usize,
        range: Range<usize>,
        mut emit: impl FnMut(usize, u32),
    ) {
        let planned = &self.words[index];
        let start = planned.start as usize;
        let separator = self
            .words
            .get(index + 1)
            .map_or(self.len, |word| word.start as usize)
            - 1;
        if range.start < separator {
            let mut position = start;
            let mut byte_start = 0;
            let after = self
                .checkpoints
                .partition_point(|point| point.position <= range.start);
            if let Some(point) = after.checked_sub(1).map(|index| &self.checkpoints[index])
                && point.position >= start
            {
                position = point.position;
                byte_start = point.byte;
            }
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
                if position >= range.end {
                    break;
                }
                if position >= range.start {
                    emit(position, id);
                }
                position += 1;
            }
        }
        if range.contains(&separator) {
            emit(separator, WORD_SEPARATOR_ID);
        }
    }
    fn fill_tokens<T: Send>(
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
                    if initial_ids.plain() {
                        for character in word.chars() {
                            if let Some(id) = initial_ids.plain_id(character) {
                                region[position].write(make(base + position, id));
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
                                region[position].write(make(base + position, id));
                                position += 1;
                            }
                        }
                    }
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
impl<S: SlotStorage> Corpus<S> {
    #[cfg(test)]
    pub(super) fn initial_view(&self) -> impl InitialPairSource + '_ {
        self
    }
    pub(super) fn len(&self) -> usize {
        self.slots.len()
    }
    #[inline]
    pub(super) fn token(&self, position: u64) -> u32 {
        self.slots.load(position as usize)
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
    pub(super) fn matcher(&self, pair: Pair) -> PairMatcher<'_, S> {
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
    /// # Safety
    /// No token readers may overlap this joined write phase. The prepared
    /// geometry must own disjoint endpoint slots across all active writers.
    pub(super) unsafe fn write_endpoints(&self, matched: PairMatch, replacement: u32) {
        // SAFETY: the caller supplies the write-phase and ownership proof.
        unsafe {
            write_endpoints(&self.slots, matched, replacement);
        }
    }
    pub(super) fn word_writers(
        &mut self,
        regions: &[std::ops::Range<u64>],
    ) -> Option<Vec<WordWriter<'_, S>>> {
        let spans = self.occurrence_spans.as_mut()?;
        let mut remaining = spans.as_mut_slice();
        let mut writers = Vec::with_capacity(regions.len());
        for region in regions {
            let (spans, next) = remaining.split_at_mut((region.end - region.start) as usize);
            remaining = next;
            writers.push(WordWriter {
                base: region.start,
                slots: &self.slots,
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
                    self.slots.prefetch_pointer(position as usize),
                    std::arch::x86_64::_MM_HINT_T0,
                );
            }
        }
    }
}

// Prepared plans own disjoint live-token spans. Pool joins delimit the
// Relaxed stores; this helper also serves exclusive whole-word regions.
#[inline]
unsafe fn write_endpoints<S: SlotStorage>(slots: &S, matched: PairMatch, replacement: u32) {
    // SAFETY: caller guarantees a reader-free joined phase and disjoint spans.
    unsafe {
        slots.store(matched.left_start as usize, replacement);
        if matched.right_start + 1 != matched.next_start {
            slots.store(matched.right_start as usize, WORD_SEPARATOR_ID);
        }
        slots.store((matched.next_start - 1) as usize, replacement);
    }
}
/// A complete immutable word region owns its occurrence-span writes as well.
pub(super) struct WordWriter<'a, S: SlotStorage> {
    base: u64,
    slots: &'a S,
    spans: &'a mut [u64],
}
impl<S: SlotStorage> WordWriter<'_, S> {
    pub(super) fn merge(&mut self, position: u64, replacement: u32) {
        let left = (position - self.base) as usize;
        let right = left + self.spans[left] as usize;
        let after = right + self.spans[right] as usize;
        let merged = (after - left) as u64;
        // SAFETY: word_writers borrows Corpus mutably until all writers drop.
        // Their checked occurrence-span regions are disjoint; merge never reads
        // token IDs and changes only endpoints inside this writer's region.
        unsafe {
            write_endpoints(
                self.slots,
                PairMatch {
                    left_start: self.base + left as u64,
                    right_start: self.base + right as u64,
                    next_start: self.base + after as u64,
                    merged_span: merged,
                },
                replacement,
            );
        }
        if right + 1 != after {
            self.spans[right] = 0;
        }
        self.spans[left] = merged;
        self.spans[after - 1] = merged;
    }
}
/// Fixed rule geometry is cached once. Unequal identity reuse reads its
/// occurrence plane from the same immutable snapshot instead.
pub(super) struct PairMatcher<'a, S: SlotStorage> {
    corpus: &'a Corpus<S>,
    pair: Pair,
    left_span: u64,
    right_span: u64,
}
impl<S: SlotStorage> PairMatcher<'_, S> {
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
    fn compact_slot_domains_preserve_ids_separators_and_adjacent_parallel_writes() {
        for (count, expected_bits) in [
            (0, 16),
            (65_535, 16),
            (65_536, 24),
            (100_000, 24),
            (131_071, 24),
            (131_072, 24),
            (200_000, 24),
            (524_288, 24),
            (1_048_576, 24),
            (16_777_215, 24),
            (16_777_216, 32),
            (u32::MAX as usize, 32),
            (usize::MAX, 32),
        ] {
            assert_eq!(slot_bits(count), expected_bits, "ID count {count}");
        }
        fn check<S: SlotStorage>(plane: S, count: usize) {
            const SLOTS: usize = 512;
            const ROUNDS: usize = 256;
            let mut values = vec![0, (count - 1) as u32, WORD_SEPARATOR_ID];
            values.extend(
                [65_534, 65_535, 65_536, 131_071, 131_072]
                    .into_iter()
                    .filter(|&id| (id as usize) < count),
            );
            std::thread::scope(|scope| {
                for lane in 0..8 {
                    let plane = &plane;
                    let values = &values;
                    scope.spawn(move || {
                        for round in 0..ROUNDS {
                            for position in (lane..SLOTS).step_by(8) {
                                // SAFETY: each thread owns its lane's slots;
                                // all reads follow this scope's join.
                                unsafe {
                                    plane
                                        .store(position, values[(position + round) % values.len()]);
                                }
                            }
                        }
                    });
                }
            });
            for position in 0..SLOTS {
                assert_eq!(
                    plane.load(position),
                    values[(position + ROUNDS - 1) % values.len()],
                    "count {count}, slot {position}"
                );
            }
        }
        for count in [100_000, 200_000, 16_777_215] {
            check(
                ThreeByteSlots {
                    tokens: (0..513).map(|_| ThreeByteSlots::bytes(0)).collect(),
                },
                count,
            );
        }
        check(
            HalfSlots((0..512).map(|_| AtomicU16::new(0)).collect()),
            65_535,
        );
        check(
            (0..512).map(|_| AtomicU32::new(0)).collect::<FullSlots>(),
            u32::MAX as usize,
        );
    }

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
            .collect::<Vec<_>>(),
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
                .collect::<Vec<_>>(),
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
                    chunk
                        .changes
                        .iter()
                        .flat_map(|change| chunk.chains.reversed(change.positions))
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
