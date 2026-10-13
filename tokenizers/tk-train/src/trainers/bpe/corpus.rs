//! Endpoint tokens retain their original coordinates; span metadata skips holes.
#[path = "merge.rs"]
mod merge;

use super::{
    TrainingMode, WORD_SEPARATOR_ID, positions::Positions, vocabulary::InitialTokenIds,
    word_counts::WordCountsView,
};
use compact_str::CompactString;
use itertools::Itertools;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::ops::Range;
use std::sync::atomic::{AtomicU16, AtomicU32, AtomicUsize, Ordering};
use tk_encode::{Result, models::bpe::Pair, parallelism::*};

/// Resident token endpoints at fixed original coordinates, with weights and spans.
/// Merges leave holes; span metadata skips them without moving other occurrences.
/// Preparation reads a joined snapshot; application writes only disjoint matches.
pub(super) struct Corpus {
    tokens: Slots,
    starts: Box<[usize]>,
    weight_runs: Box<[WeightRun]>,
    unit_weight_range: Option<(Range<usize>, usize)>,
    // Nonzero marks an activated ID. Until alias geometry diverges it also
    // stores that ID's span; afterwards occurrence_spans owns all geometry.
    id_metadata: Vec<usize>,
    occurrence_spans: Option<Box<[AtomicUsize]>>,
    whole_words: bool,
    mode: TrainingMode,
    birth_pair_span_limit: Option<usize>,
}

/// The selected historical payload moves here without its priority snapshot.
pub(super) struct SelectedMerge {
    pub(super) pair: Pair,
    pub(super) merged_id: u32,
    pub(super) positions: Positions,
}

/// Ordered removal and birth; checked weights must remain separate at commit.
pub(super) struct PairChange<P> {
    pub(super) removed: Pair,
    pub(super) born: Pair,
    pub(super) removed_weight: u64,
    pub(super) born_weight: u64,
    pub(super) positions: P,
    pub(super) birth_group: usize,
}

/// A run in the sorted word-weight directory.
struct WeightRun {
    start: usize,
    weight: u64,
}

/// Cached weight through an exclusive coordinate end.
#[derive(Clone, Copy)]
struct WeightLookup {
    weight: u64,
    end: usize,
    index: usize,
}

/// Borrowed input and measured coordinate layout for initial pair counting.
/// Defers the resident token plane until temporary counting buffers retire;
/// materialization consumes the plan and transfers its metadata to the Corpus.
pub(super) struct CorpusPlan<'input> {
    starts: Box<[usize]>,
    weight_runs: Box<[WeightRun]>,
    unit_weight_range: Option<(Range<usize>, usize)>,
    whole_words: bool,
    words: Box<[(&'input CompactString, &'input u64)]>,
    ids: InitialTokenIds,
    length: usize,
    slot_bound: usize,
    mode: TrainingMode,
    birth_pair_span_limit: Option<usize>,
}

/// One original adjacent edge in the filtered symbol coordinate space.
pub(super) struct InitialEdge {
    pub(super) pair: Pair,
    pub(super) coordinate: u64,
    pub(super) weight: u64,
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
        if get_parallelism() {
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
        let mut mass = 0u128;
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
            // A word with n retained symbols has max(n - 1, 0) adjacent edges.
            // Weight each edge by word frequency; u128 checks the total mass
            // without imposing a total-u64 limit on fresh pair counts.
            mass += u128::from(weight) * symbols.saturating_sub(1) as u128;
        }
        // Bound bytes using the widest slot representation. Division avoids
        // overflowing the multiplication that an allocation-size check needs.
        if length > isize::MAX as usize / std::mem::size_of::<AtomicU32>() {
            return Err("BPE corpus exceeds resident allocation bounds".into());
        }
        // Reuse and nonempty affixes require the signed ledger contract: both
        // total edge mass and each word's weight must fit i64 independently.
        if (mode == TrainingMode::Reuse || !ids.is_plain())
            && (mass > i64::MAX as u128
                || words.iter().any(|(_, weight)| **weight > i64::MAX as u64))
        {
            return Err("BPE identity-reuse weighted edge mass or word weight exceeds i64".into());
        }
        // Descending weights put all frequency-one words in one contiguous run.
        // Its exclusive end is the next run's start, or the corpus end.
        let unit_weight_range = weight_runs.iter().position(|run| run.weight == 1).map(|i| {
            let range =
                weight_runs[i].start..weight_runs.get(i + 1).map_or(length, |run| run.start);
            (range, i)
        });
        let slot_bound = ids.token_count().max(target_vocab_size);
        Ok(Self {
            starts: starts.into_boxed_slice(),
            weight_runs: weight_runs.into_boxed_slice(),
            unit_weight_range,
            whole_words: birth_pair_span_limit.is_some(),
            words,
            ids,
            length,
            slot_bound,
            mode,
            birth_pair_span_limit,
        })
    }

    pub(super) fn word_count(&self) -> usize {
        self.words.len()
    }

    // Whole words preserve symbol-scan semantics. A single oversized word is
    // admitted alone; otherwise each range has at most `slots` resident slots.
    fn ranges(&self, slots: usize) -> Vec<Range<usize>> {
        let mut ranges = Vec::new();
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
            ranges.push(begin..end);
            begin = end;
        }
        ranges
    }

    pub(super) fn token_count(&self) -> usize {
        self.ids.token_count()
    }

    pub(super) fn mode(&self) -> TrainingMode {
        self.mode
    }

    /// Bound whole-word edge streams in coordinate order, with progress units.
    /// A count task consumes its stream directly and owns any counting error.
    pub(super) fn chunks(
        &self,
        slots: usize,
    ) -> Vec<(usize, impl Iterator<Item = InitialEdge> + Send + '_)> {
        self.ranges(slots)
            .into_iter()
            .map(move |range| (range.len(), self.edges(range)))
            .collect()
    }

    fn edges(&self, range: Range<usize>) -> impl Iterator<Item = InitialEdge> + '_ {
        range.flat_map(move |word| {
            let (text, &weight) = self.words[word];
            let start = self.starts[word];
            self.ids
                .symbols(text)
                .tuple_windows()
                .enumerate()
                .map(move |(offset, pair)| InitialEdge {
                    pair,
                    coordinate: (start + offset) as u64,
                    weight,
                })
        })
    }

    pub(super) fn materialize(self) -> Corpus {
        let tokens = Slots::new(self.length, self.slot_bound);
        self.words
            .maybe_par_iter()
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
            starts: if self.mode == TrainingMode::Reuse {
                self.starts
            } else {
                Box::default()
            },
            weight_runs: self.weight_runs,
            unit_weight_range: self.unit_weight_range,
            id_metadata: self.ids.into_metadata(),
            occurrence_spans: None,
            whole_words: self.whole_words,
            mode: self.mode,
            birth_pair_span_limit: self.birth_pair_span_limit,
        }
    }
}

impl Corpus {
    #[cfg(test)]
    pub(super) fn tokens(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.len()).map(|p| self.token(p))
    }

    #[cfg(test)]
    pub(super) fn spans(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.len()).map(|p| self.occurrence_span(p))
    }

    fn len(&self) -> usize {
        self.tokens.len()
    }

    fn resident(&self, coordinate: u64) -> usize {
        let position =
            usize::try_from(coordinate).expect("corpus coordinate fits resident indexing");
        assert!(
            position < self.len(),
            "coordinate belongs to the resident corpus"
        );
        position
    }

    #[inline]
    fn token(&self, position: usize) -> u32 {
        self.tokens.get(position)
    }

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

    fn weight(&self, position: usize, first_run: usize) -> WeightLookup {
        if self.weight_runs.len() == 1 {
            return WeightLookup {
                weight: self.weight_runs[0].weight,
                end: self.len(),
                index: 0,
            };
        }
        // Cached runs are half-open: a coordinate at end belongs to the next
        // weight run and must fall through to the directory lookup.
        if let Some((range, index)) = &self.unit_weight_range
            && position >= range.start
            && position < range.end
        {
            return WeightLookup {
                weight: 1,
                end: range.end,
                index: *index,
            };
        }
        // Fresh readers start anywhere, then only move forward. An expired
        // cache guarantees position >= the next run's start, so its suffix has
        // a predecessor; reuse and first queries search the whole directory.
        let i = first_run
            + self.weight_runs[first_run..].partition_point(|run| run.start <= position)
            - 1;
        WeightLookup {
            weight: self.weight_runs[i].weight,
            end: self
                .weight_runs
                .get(i + 1)
                .map_or(self.len(), |run| run.start),
            index: i,
        }
    }

    fn word(&self, position: usize) -> usize {
        // The last word start at or before this coordinate is its owning word;
        // valid historical positions guarantee that a predecessor exists.
        self.starts.partition_point(|&start| start <= position) - 1
    }

    fn word_range(&self, word: usize) -> Range<usize> {
        // The slot immediately before the next word (or corpus end) is this
        // word's separator. Subtract it from the exclusive end of the scan.
        self.starts[word]..self.starts.get(word + 1).copied().unwrap_or(self.len()) - 1
    }

    #[inline]
    fn occurrence_span(&self, position: usize) -> usize {
        self.occurrence_spans.as_ref().map_or_else(
            || self.fixed_id_span(self.token(position)),
            |spans| spans[position].load(Ordering::Relaxed),
        )
    }

    fn id_count(&self) -> usize {
        self.id_metadata.len()
    }

    fn fixed_id_span(&self, id: u32) -> usize {
        self.id_metadata[id as usize]
    }

    // Fresh IDs have fixed geometry cached once per reader. Aliases use match_at.
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

    /// Read valid fresh matches and their weights from nondecreasing coordinates.
    /// Cache fixed geometry and weight regions within this reader; prefetch at
    /// most 16 coordinates ahead. The preparation snapshot must remain unchanged
    /// until every reader joins. AA overlap selection stays with the caller.
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
                Some(lookup) => self.weight(p, lookup.index + 1),
                None => self.weight(p, 0),
            };
            cached_weight = Some(lookup);
            Some((matched, lookup.weight))
        })
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

    fn whole_words(&self) -> bool {
        self.whole_words
    }

    /// Activation history persists after an ID's last occurrence is consumed.
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

    fn materialize_occurrence_spans(&mut self) {
        let values: Vec<_> = (0..self.len()).map(|_| AtomicUsize::new(0)).collect();
        for &start in &self.starts {
            let mut p = start;
            while self.token(p) != WORD_SEPARATOR_ID {
                let n = self.fixed_id_span(self.token(p));
                values[p].store(n, Ordering::Relaxed);
                // A span covers [p, p + n): its final endpoint is p + n - 1.
                values[p + n - 1].store(n, Ordering::Relaxed);
                p += n;
            }
        }
        self.occurrence_spans = Some(values.into_boxed_slice());
    }

    /// Apply one rule's accepted starts after all preparation tasks join.
    /// Match intervals must be disjoint across every PreparedChunk in the round; reads
    /// touch only this match's own endpoints. Atomics keep individual accesses
    /// safe on error or unwind, but overlapping writes invalidate reconstruction.
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

/// Atomic endpoint plane selected once from the vocabulary ID bound.
/// A reserved u16 sentinel leaves real IDs 0..65534 available. Wider vocabularies
/// use u32 directly; both planes have the same safe endpoint access protocol.
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
                    .map(|_| AtomicU32::new(WORD_SEPARATOR_ID))
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
                if id == u16::MAX {
                    WORD_SEPARATOR_ID
                } else {
                    u32::from(id)
                }
            }
            Self::Wide(values) => values[position].load(Ordering::Relaxed),
        }
    }

    // Publish the merged endpoints and clear a former right start only when it
    // is interior. A one-slot right token shares that start with the new tail.
    fn replace_pair(&self, start: usize, right: usize, after: usize, merged_id: u32) {
        self.set(start, merged_id);
        if right + 1 != after {
            self.set(right, WORD_SEPARATOR_ID);
        }
        self.set(after - 1, merged_id);
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
    use super::super::tests::{spans, tokens};
    use super::*;

    fn fixture(input: &[u32], spans: &[usize], occurrences: Option<&[usize]>) -> Corpus {
        let corpus = Corpus {
            tokens: Slots::new(input.len(), spans.len()),
            starts: vec![1].into_boxed_slice(),
            weight_runs: vec![WeightRun {
                start: 1,
                weight: 1,
            }]
            .into_boxed_slice(),
            unit_weight_range: Some((1..input.len(), 0)),
            id_metadata: spans.to_vec(),
            occurrence_spans: occurrences
                .map(|values| values.iter().copied().map(AtomicUsize::new).collect()),
            whole_words: occurrences.is_some(),
            mode: TrainingMode::Reuse,
            birth_pair_span_limit: None,
        };
        for (position, &id) in input.iter().enumerate() {
            corpus.tokens.set(position, id);
        }
        corpus
    }

    #[test]
    fn adjacent_writes() {
        let s = WORD_SEPARATOR_ID;
        for order in 0..3 {
            let corpus = fixture(&[s, 0, 0, 1, 0, 0, 1, s], &[2, 1, 3], None);
            corpus.apply(&[], 2);
            assert_eq!(tokens(&corpus), [s, 0, 0, 1, 0, 0, 1, s]);
            match order {
                0 => corpus.apply(&[1, 4], 2),
                1 => corpus.apply(&[4, 1], 2),
                _ => [1, 4].maybe_par_iter().for_each(|&p| corpus.apply(&[p], 2)),
            }
            assert_eq!(tokens(&corpus), [s, 2, 0, 2, 2, 0, 2, s]);
            assert_eq!(corpus.id_metadata, [2, 1, 3]);
        }
        // One pair has 2+1 and 1+3 spans; merged_id reuses ID 0.
        let corpus = fixture(
            &[s, 0, 0, 1, 0, 1, 1, 1, s],
            &[1, 1],
            Some(&[0, 2, 2, 1, 1, 3, 0, 3, 0]),
        );
        corpus.apply(&[1, 4], 0);
        assert_eq!(tokens(&corpus), [s, 0, 0, 0, 0, s, 1, 0, s]);
        assert_eq!(spans(&corpus), [0, 3, 2, 3, 4, 0, 0, 4, 0]);
    }

    #[test]
    fn fresh_reader() {
        let s = WORD_SEPARATOR_ID;
        let input: Vec<_> = std::iter::once(s)
            .chain((0..40).flat_map(|_| [0, 1, s]))
            .collect();
        let mut corpus = fixture(&input, &[1, 1, 2], None);
        corpus.apply(&[1, 64], 2);
        for runs in [
            vec![(1, 9), (31, 1), (91, 0)],
            vec![
                (1, 9),
                (4, 8),
                (7, 7),
                (10, 6),
                (13, 5),
                (16, 4),
                (19, 3),
                (22, 2),
                (25, 1),
                (91, 0),
            ],
            vec![(1, 0)],
            vec![(1, 1)],
            vec![(1, 9)],
        ] {
            corpus.weight_runs = runs
                .iter()
                .map(|&(start, weight)| WeightRun { start, weight })
                .collect();
            corpus.unit_weight_range = runs
                .iter()
                .position(|&(_, weight)| weight == 1)
                .map(|i| (runs[i].0..runs.get(i + 1).map_or(input.len(), |r| r.0), i));
            for coordinates in [
                (0..input.len() as u64).collect::<Vec<_>>(),
                vec![7, 7, 31, 34, 91, 118],
                vec![31, 91, 118],
                vec![7, 91],
                vec![118, 118],
                vec![],
            ] {
                // Original [a, b, sep] words begin at 1 + 3*word. Starts 1 and
                // 64 were consumed above; every other tail, hole and separator
                // must be filtered before querying weights. Linear reference
                // lookup checks arbitrary task starts and duplicate coordinates.
                let expected: Vec<_> = coordinates
                    .iter()
                    .copied()
                    .filter(|&p| p % 3 == 1 && p != 1 && p != 64)
                    .map(|p| {
                        let weight = runs
                            .iter()
                            .rev()
                            .find(|&&(start, _)| start <= p as usize)
                            .unwrap()
                            .1;
                        (p as usize, p as usize + 2, weight)
                    })
                    .collect();
                let matches: Vec<_> = corpus
                    .fresh_matches(coordinates, (0, 1))
                    .map(|(matched, weight)| (matched.start, matched.end, weight))
                    .collect();
                assert_eq!(matches, expected, "runs={runs:?}");
            }
        }
    }

    #[test]
    fn initial_ranges() {
        let words = ahash::AHashMap::new();
        let view = WordCountsView::from(&words);
        let trainer = super::super::BpeTrainer::new(1, 0);
        let (_, ids) = super::super::vocabulary::Vocabulary::initialize(
            &trainer,
            view,
            &trainer.select_alphabet(view),
            &None,
        )
        .unwrap();
        let mut plan = CorpusPlan::build(
            view,
            ids,
            trainer.vocab_size,
            TrainingMode::Fresh,
            trainer.max_token_length,
        )
        .unwrap();
        assert!(plan.ranges(0).is_empty());
        let text = CompactString::default();
        let weight = 1;
        let maximum = usize::MAX;
        for (starts, length, budget, expected) in [
            (vec![1], 1025, 0, std::iter::once(0..1).collect()),
            (vec![1, 4, 8, 10], 16, 0, vec![0..1, 1..2, 2..3, 3..4]),
            (vec![1, 4, 8, 10], 16, 7, vec![0..2, 2..3, 3..4]),
            (vec![1, 4, 8, 10], 16, 9, vec![0..3, 3..4]),
            (
                vec![maximum - 8, maximum - 3],
                maximum,
                8,
                std::iter::once(0..2).collect(),
            ),
        ] {
            plan.words = vec![(&text, &weight); starts.len()].into_boxed_slice();
            plan.starts = starts.into_boxed_slice();
            plan.length = length;
            assert_eq!(plan.ranges(budget), expected);
        }
    }
}
