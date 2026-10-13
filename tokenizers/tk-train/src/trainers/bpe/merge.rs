//! Private joined preparation and application over corpus geometry.
use super::super::{TrainingMode, WORD_SEPARATOR_ID, add, positions::Positions};
use super::{Corpus, PairChange, SelectedMerge};
use ahash::AHashMap;
use itertools::Itertools;
#[cfg(test)]
use rayon::prelude::*;
use smallvec::SmallVec;
use std::ops::Range;
use tk_encode::parallelism::*;
use tk_encode::{Result, models::bpe::Pair};

impl Corpus {
    /// Join all readers before writes and all writers before returning changes.
    /// Preparation errors leave endpoints unchanged. Later index errors can
    /// occur after these writes; the owning attempt provides no rollback.
    pub(in crate::trainers::bpe) fn merge(
        &mut self,
        selected: Vec<SelectedMerge>,
    ) -> Result<Vec<Vec<PairChange<Positions>>>> {
        let chunks = self.prepare(selected)?;
        let corpus: &Corpus = self;
        Ok(chunks
            .into_maybe_par_iter()
            .map(|chunk| {
                corpus.apply(&chunk.starts, chunk.merged_id);
                chunk.changes
            })
            .collect())
    }

    // Owning preparation retires every selected stream and shared lookup before
    // application begins. The selector establishes the nonempty input shapes.
    fn prepare(&self, selected: Vec<SelectedMerge>) -> Result<Vec<PreparedChunk>> {
        match self.mode {
            TrainingMode::Fresh => self.prepare_fresh(selected),
            TrainingMode::Reuse => {
                debug_assert_eq!(selected.len(), 1);
                self.prepare_reuse(selected.into_iter().next().expect("reuse selects one rule"))
            }
        }
    }

    fn prepare_fresh(&self, selected: Vec<SelectedMerge>) -> Result<Vec<PreparedChunk>> {
        let batch = if selected[0].pair.0 == selected[0].pair.1 {
            debug_assert_eq!(selected.len(), 1);
            FreshBatch::AA(selected.into_iter().next().expect("AA selects one rule"))
        } else {
            FreshBatch::AB(selected)
        };
        let preparation = FreshPreparation {
            corpus: self,
            birth_pair_span_limit: self.birth_pair_span_limit.unwrap_or(usize::MAX),
        };
        match batch {
            FreshBatch::AA(SelectedMerge {
                pair,
                merged_id,
                positions,
            }) => {
                let original_positions_len = positions.len();
                let mut accepted_starts = Vec::new();
                let matcher = self.fixed_matcher(pair);
                let mut previous_end = 0;
                for coordinate in positions.iter() {
                    if coordinate >= previous_end
                        && let Some(matched) = matcher(self.resident(coordinate))
                    {
                        accepted_starts.push(coordinate);
                        previous_end = matched.end as u64;
                    }
                }
                drop(positions);
                // Budget from the original historical payload, not accepted starts.
                let chunk = original_positions_len
                    .div_ceil(super::super::num_workers() * 4)
                    .max(4096);
                let tasks: Vec<_> = accepted_starts
                    .chunks(chunk)
                    .map(|part| (pair, merged_id, part.iter().copied()))
                    .collect();
                tasks
                    .into_maybe_par_iter()
                    .map_init(
                        NeighborScratch::default,
                        |scratch, (pair, merged_id, positions)| {
                            preparation.prepare(
                                pair,
                                merged_id,
                                positions,
                                scratch,
                                |matched, _, _| {
                                    matched.start >= matched.len()
                                        && accepted_starts
                                            .binary_search(
                                                &((matched.start - matched.len()) as u64),
                                            )
                                            .is_ok()
                                },
                                |after, _| {
                                    accepted_starts
                                        .binary_search(&(after as u64))
                                        .is_ok()
                                        .then_some(merged_id)
                                },
                            )
                        },
                    )
                    .collect()
            }
            FreshBatch::AB(rules) => {
                let merged_ids_by_pair: AHashMap<_, _> = rules
                    .iter()
                    .map(|rule| (rule.pair, rule.merged_id))
                    .collect();
                let total: usize = rules.iter().map(|rule| rule.positions.len()).sum();
                let chunk = total
                    .div_ceil(super::super::num_workers())
                    .clamp(1, 1 << 26);
                let mut tasks = Vec::new();
                for rule in &rules {
                    for positions in rule.positions.chunks(chunk) {
                        tasks.push((rule.pair, rule.merged_id, positions));
                    }
                }
                // Rule order, then restart-aligned chunk order, remains indexed.
                tasks
                    .into_maybe_par_iter()
                    .map_init(
                        NeighborScratch::default,
                        |scratch, (pair, merged_id, positions)| {
                            preparation.prepare(
                                pair,
                                merged_id,
                                positions,
                                scratch,
                                |_, before, prior| {
                                    before != 0
                                        && merged_ids_by_pair
                                            .contains_key(&(self.token(before - 1), prior))
                                },
                                |after, next| {
                                    merged_ids_by_pair
                                        .get(&(next, self.token(after + self.fixed_id_span(next))))
                                        .copied()
                                },
                            )
                        },
                    )
                    .collect()
            }
        }
    }

    fn prepare_reuse(&self, rule: SelectedMerge) -> Result<Vec<PreparedChunk>> {
        // Historical positions name the unique word domains. One word belongs
        // to one task; its virtual left neighbors are interpreted sequentially.
        let words: Vec<_> = rule
            .positions
            .iter()
            .map(|p| self.word(self.resident(p)))
            .dedup()
            .collect();
        let chunk = words.len().div_ceil(super::super::num_workers()).max(1);
        let limit = self.birth_pair_span_limit.unwrap_or(usize::MAX);
        words
            .maybe_par_chunks(chunk)
            .map_init(NeighborScratch::default, |scratch, words| {
                #[cfg(test)]
                super::super::tests::observe_worker(super::super::tests::Phase::ReusePrepare);
                let mut neighbors =
                    NeighborChanges::new(rule.pair, rule.merged_id, self.id_count(), scratch);
                let mut starts = SmallVec::new();
                for &word in words {
                    scan_word(self, &rule, word, limit, &mut neighbors, &mut starts)?;
                }
                Ok(PreparedChunk {
                    starts,
                    merged_id: rule.merged_id,
                    changes: neighbors.finish(),
                })
            })
            .collect()
    }
}

enum FreshBatch {
    AA(SelectedMerge),
    AB(Vec<SelectedMerge>),
}

struct PreparedChunk {
    starts: SmallVec<[usize; 2]>,
    merged_id: u32,
    changes: Vec<PairChange<Positions>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum EdgeSide {
    Left,
    Right,
}

/// A logical neighboring token; reuse scans can produce it before endpoint writes.
#[derive(Clone, Copy)]
struct Neighbor {
    token_id: u32,
    start: usize,
    span: usize,
}

/// Task-local aggregation reuses borrowed neighbor directories.
struct NeighborChanges<'a> {
    pair: Pair,
    merged_id: u32,
    scratch: &'a mut NeighborScratch,
    changes: [Vec<PairChange<Positions>>; 2],
}

#[derive(Default)]
struct NeighborScratch {
    indices: [Vec<u32>; 2],
}

impl<'a> NeighborChanges<'a> {
    fn new(pair: Pair, merged_id: u32, domain: usize, scratch: &'a mut NeighborScratch) -> Self {
        for values in &mut scratch.indices {
            values.resize(domain, u32::MAX);
        }
        Self {
            pair,
            merged_id,
            scratch,
            changes: std::array::from_fn(|_| Vec::new()),
        }
    }

    fn group(&mut self, neighbor: u32, side: EdgeSide) -> &mut PairChange<Positions> {
        let left = side == EdgeSide::Left;
        let side_index = side as usize;
        let slot = &mut self.scratch.indices[side_index][neighbor as usize];
        if self.changes[side_index]
            .get(*slot as usize)
            .is_none_or(|change| (if left { change.born.0 } else { change.born.1 }) != neighbor)
        {
            let id = self.merged_id;
            let pair = self.pair;
            let index = self.changes[side_index].len();
            self.changes[side_index].push(PairChange {
                removed: if left {
                    (neighbor, pair.0)
                } else {
                    (pair.1, neighbor)
                },
                born: if left { (neighbor, id) } else { (id, neighbor) },
                removed_weight: 0,
                born_weight: 0,
                positions: Positions::default(),
                // Both sides' (id,id) births share a group; other right births follow left births.
                birth_group: usize::from(!left && neighbor != id),
            });
            *slot =
                u32::try_from(index).expect("each neighbor directory is bounded by real token IDs");
        }
        &mut self.changes[side_index][*slot as usize]
    }

    fn record(
        &mut self,
        side: EdgeSide,
        removed: u32,
        born: u32,
        position: usize,
        weight: u64,
        admit_birth: bool,
    ) -> Result<()> {
        let group = self.group(removed, side);
        add(&mut group.removed_weight, weight)?;
        if admit_birth {
            let group = if removed == born {
                group
            } else {
                self.group(born, side)
            };
            add(&mut group.born_weight, weight)?;
            group.positions.push(position as u64)?;
        }
        Ok(())
    }

    fn finish(self) -> Vec<PairChange<Positions>> {
        self.changes
            .into_iter()
            .flatten()
            .filter(|change| change.removed_weight != 0 || !change.positions.is_empty())
            .collect()
    }
}

/// Common fresh reader and event loop; AA/AB closures provide global adjacency.
struct FreshPreparation<'a> {
    corpus: &'a Corpus,
    birth_pair_span_limit: usize,
}

impl FreshPreparation<'_> {
    fn prepare(
        &self,
        pair: Pair,
        merged_id: u32,
        positions: impl IntoIterator<Item = u64>,
        scratch: &mut NeighborScratch,
        left_merging: impl Fn(&Range<usize>, usize, u32) -> bool,
        right_merged_id: impl Fn(usize, u32) -> Option<u32>,
    ) -> Result<PreparedChunk> {
        #[cfg(test)]
        super::super::tests::observe_worker(super::super::tests::Phase::FreshPrepare);
        let mut neighbors = NeighborChanges::new(pair, merged_id, self.corpus.id_count(), scratch);
        let mut starts = SmallVec::new();
        // PERF: Borrow the local decoder so lookahead adapters hold a pointer.
        // In the measured x86-64 release build, owning decoder state added a
        // 144-byte reader-state copy per AB task and increased the stack frame.
        let mut positions = positions.into_iter();
        for (matched, weight) in self.corpus.fresh_matches(&mut positions, pair) {
            self.record(
                &mut neighbors,
                &matched,
                weight,
                &left_merging,
                &right_merged_id,
            )?;
            starts.push(matched.start);
        }
        Ok(PreparedChunk {
            starts,
            merged_id,
            changes: neighbors.finish(),
        })
    }

    // The left match owns each shared edge and uses both final IDs. A later
    // match skips that left edge, even across rule or task boundaries.
    fn record(
        &self,
        neighbors: &mut NeighborChanges<'_>,
        matched: &Range<usize>,
        weight: u64,
        left_merging: &impl Fn(&Range<usize>, usize, u32) -> bool,
        right_merged_id: &impl Fn(usize, u32) -> Option<u32>,
    ) -> Result<()> {
        let p = matched.start;
        let prior = self.corpus.token(p - 1);
        if prior != WORD_SEPARATOR_ID {
            let span = self.corpus.fixed_id_span(prior);
            let before = p - span;
            if !left_merging(matched, before, prior) {
                neighbors.record(
                    EdgeSide::Left,
                    prior,
                    prior,
                    before,
                    weight,
                    span + matched.len() < self.birth_pair_span_limit,
                )?;
            }
        }
        let next = self.corpus.token(matched.end);
        if next != WORD_SEPARATOR_ID {
            let born = right_merged_id(matched.end, next).unwrap_or(next);
            neighbors.record(
                EdgeSide::Right,
                next,
                born,
                p,
                weight,
                matched.len() + self.corpus.fixed_id_span(born) < self.birth_pair_span_limit,
            )?;
        }
        Ok(())
    }
}

fn scan_word(
    corpus: &Corpus,
    rule: &SelectedMerge,
    word: usize,
    limit: usize,
    neighbors: &mut NeighborChanges<'_>,
    starts: &mut SmallVec<[usize; 2]>,
) -> Result<()> {
    let range = corpus.word_range(word);
    let mut p = range.start;
    let mut positions = rule.positions.iter_from(p as u64);
    let mut left_neighbor: Option<Neighbor> = None;
    while p < range.end {
        // Full scans are required after alias activation or any Some length gate.
        // Otherwise only eligible coordinates in this historical word are visited.
        if !corpus.whole_words() {
            let Some(coordinate) = positions.find(|&start| start >= p as u64) else {
                break;
            };
            let next = corpus.resident(coordinate);
            if next >= range.end {
                break;
            }
            if next != p {
                let id = corpus.token(next - 1);
                left_neighbor = (id != WORD_SEPARATOR_ID).then(|| Neighbor {
                    token_id: id,
                    start: next - corpus.occurrence_span(next - 1),
                    span: corpus.occurrence_span(next - 1),
                });
                p = next;
            }
        }
        if let Some(matched) = corpus.match_at(p, rule.pair) {
            let weight = corpus.word_weight(word);
            if let Some(left) = left_neighbor {
                neighbors.record(
                    EdgeSide::Left,
                    left.token_id,
                    left.token_id,
                    left.start,
                    weight,
                    left.span + matched.len() < limit,
                )?;
            }
            let next = corpus.token(matched.end);
            if next != WORD_SEPARATOR_ID {
                neighbors.record(
                    EdgeSide::Right,
                    next,
                    next,
                    p,
                    weight,
                    matched.len() + corpus.occurrence_span(matched.end) < limit,
                )?;
            }
            left_neighbor = Some(Neighbor {
                token_id: rule.merged_id,
                start: matched.start,
                span: matched.len(),
            });
            starts.push(matched.start);
            p = matched.end;
        } else {
            let span = corpus.occurrence_span(p);
            left_neighbor = Some(Neighbor {
                token_id: corpus.token(p),
                start: p,
                span,
            });
            p += span.max(1);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::super::tests::{spans, tokens};
    use super::super::super::{MergeOutcome, index::PairIndex, vocabulary::Vocabulary};
    use super::super::super::{
        WORD_SEPARATOR_ID,
        tests::{counts, initialize, trainer},
        word_counts::WordCountsView,
    };
    use super::*;

    fn select(
        vocabulary: &mut Vocabulary,
        corpus: &mut Corpus,
        index: &mut PairIndex,
    ) -> SelectedMerge {
        let priority = index.peek().unwrap();
        let merged_id = vocabulary.resolve(priority.pair()).unwrap().merged_id;
        corpus.activate(priority.pair(), merged_id);
        SelectedMerge {
            pair: priority.pair(),
            merged_id,
            positions: index.take(priority).positions,
        }
    }

    #[test]
    fn shared_boundary_once() {
        let trainer = trainer();
        let words = counts(&[("abacdbdc", 2), ("ab", 10), ("ac", 9), ("db", 8), ("dc", 7)]);
        let (mut vocabulary, mut corpus, mut index) =
            initialize(&trainer, WordCountsView::from(&words), false);
        let mut selected = Vec::new();
        // Shared heads/tails remain compatible; adjacent replacements publish once.
        for pair in [(0, 1), (0, 2), (3, 1), (3, 2)] {
            let rule = select(&mut vocabulary, &mut corpus, &mut index);
            assert_eq!(rule.pair, pair);
            selected.push(rule);
        }
        let jobs = corpus.prepare(selected).unwrap();
        let mut births: Vec<_> = jobs
            .iter()
            .flat_map(|j| &j.changes)
            .filter(|c| c.born_weight != 0)
            .map(|c| (c.born, c.born_weight))
            .collect();
        births.sort_unstable();
        assert_eq!(births, [((4, 5), 2), ((5, 6), 2), ((6, 7), 2)]);
        jobs.par_iter()
            .for_each(|j| corpus.apply(&j.starts, j.merged_id));
        let s = WORD_SEPARATOR_ID;
        assert_eq!(
            tokens(&corpus),
            [
                s, 4, 4, s, 5, 5, s, 6, 6, s, 7, 7, s, 4, 4, 5, 5, 6, 6, 7, 7, s
            ]
        );
    }

    #[test]
    fn rejected_start_and_aa_tasks() {
        for workers in [1, 6] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            pool.install(|| {
                let trainer = trainer();
                let word = "a".repeat(20_000 * workers + 1);
                let words = counts(&[(&word, 1)]);
                let (mut vocabulary, mut corpus, mut index) =
                    initialize(&trainer, WordCountsView::from(&words), false);
                let mut rule = select(&mut vocabulary, &mut corpus, &mut index);
                // A stale separator precedes the overlapping valid AA starts.
                let coordinates: Vec<_> = std::iter::once(0).chain(rule.positions.iter()).collect();
                rule.positions = Positions::from_sorted(&coordinates).unwrap();
                let chunks = corpus.prepare(vec![rule]).unwrap();
                // Original length gives 5001; accepted length would incorrectly give 4096.
                assert_eq!(chunks[0].starts.len(), 5001);
                assert_eq!(chunks.len(), 2 * workers);
                let accepted: Vec<_> = chunks
                    .iter()
                    .flat_map(|c| c.starts.iter().copied())
                    .collect();
                assert_eq!(accepted, (1..word.len()).step_by(2).collect::<Vec<_>>());
                let changes: Vec<_> = chunks.iter().flat_map(|c| &c.changes).collect();
                let shared: Vec<_> = changes.iter().filter(|c| c.born == (1, 1)).collect();
                assert_eq!(
                    shared.iter().map(|c| c.born_weight).sum::<u64>(),
                    (accepted.len() - 1) as u64
                );
                assert_eq!(
                    shared
                        .iter()
                        .flat_map(|c| c.positions.iter())
                        .collect::<Vec<_>>(),
                    accepted[..accepted.len() - 1]
                        .iter()
                        .map(|&p| p as u64)
                        .collect::<Vec<_>>()
                );
                assert!(
                    !changes
                        .iter()
                        .any(|c| c.born == (0, 1) && !c.positions.is_empty()),
                    "every later chunk must consult the complete AA acceptance set"
                );
            });
        }
    }

    #[test]
    fn reuse_spans() {
        let mut trainer = trainer();
        trainer.end_of_word_suffix = Some("a".into());
        let words = counts(&[("baaaaaba", 2), ("ba", 1)]);
        let (mut vocabulary, mut corpus, mut index) =
            initialize(&trainer, WordCountsView::from(&words), true);
        assert!(matches!(
            trainer
                .merge_next(&mut vocabulary, &mut corpus, &mut index)
                .unwrap(),
            MergeOutcome::Committed { .. }
        ));
        let rule = select(&mut vocabulary, &mut corpus, &mut index);
        assert_eq!((rule.pair, rule.merged_id), ((1, 2), 3));
        // AA aliases the suffix ID: BA sees original and merged occurrences.
        let selected = vec![rule];
        let jobs = corpus.prepare(selected).unwrap();
        for job in jobs {
            corpus.apply(&job.starts, job.merged_id);
        }
        let s = WORD_SEPARATOR_ID;
        assert_eq!(tokens(&corpus), [s, 3, s, 3, 2, 2, 0, 3, 3, s, 1, 2, s]);
        assert_eq!(spans(&corpus), [0, 3, 0, 3, 2, 2, 1, 2, 2, 0, 1, 1, 0]);
    }

    #[test]
    fn directory_recovery() {
        let mut trainer = trainer();
        trainer.max_token_length = Some(3);
        let words = counts(&[("xab", 1)]);
        let (mut vocabulary, mut corpus, mut index) =
            initialize(&trainer, WordCountsView::from(&words), false);
        let rule = select(&mut vocabulary, &mut corpus, &mut index);
        let mut directories = NeighborScratch::default();
        {
            let mut neighbors = NeighborChanges::new(
                rule.pair,
                rule.merged_id,
                corpus.id_count(),
                &mut directories,
            );
            neighbors
                .record(EdgeSide::Left, 2, 2, 1, u64::MAX, true)
                .unwrap();
            assert!(neighbors.record(EdgeSide::Left, 2, 2, 1, 1, true).is_err());
        }
        // The old directory index is reused without reset after the failed group.
        let mut neighbors = NeighborChanges::new(
            rule.pair,
            rule.merged_id,
            corpus.id_count(),
            &mut directories,
        );
        neighbors.record(EdgeSide::Left, 1, 1, 1, 3, true).unwrap();
        neighbors.record(EdgeSide::Left, 2, 2, 2, 4, true).unwrap();
        let changes = neighbors.finish();
        assert_eq!(
            changes
                .iter()
                .map(|c| (c.removed, c.born_weight))
                .collect::<Vec<_>>(),
            [((1, 0), 3), ((2, 0), 4)]
        );
        let job = corpus.prepare(vec![rule]).unwrap().pop().unwrap();
        let change = &job.changes[0];
        assert_eq!(
            (change.removed, change.removed_weight, change.born_weight),
            ((2, 0), 1, 0)
        );
        assert!(change.positions.is_empty());
        corpus.apply(&job.starts, job.merged_id);
        index.commit(vec![job.changes]).unwrap();
        assert!(index.peek().is_none());
    }

    #[test]
    fn reuse_scan_domain_and_bound_configuration() {
        for limit in [None, Some(usize::MAX)] {
            let mut trainer = trainer();
            trainer.max_token_length = limit;
            let words = counts(&[("abab", 2), ("ab", 1)]);
            let (mut vocabulary, mut corpus, mut index) =
                initialize(&trainer, WordCountsView::from(&words), true);
            assert_eq!(corpus.word_range(0), 1..5);
            assert_eq!(corpus.word_range(1), 6..8);
            let mut rule = select(&mut vocabulary, &mut corpus, &mut index);
            assert_eq!(rule.pair, (0, 1));
            rule.positions = Positions::from_sorted(&[1]).unwrap();
            let chunks = corpus.prepare(vec![rule]).unwrap();
            let starts: Vec<_> = chunks
                .iter()
                .flat_map(|c| c.starts.iter().copied())
                .collect();
            assert_eq!(starts, if limit.is_some() { vec![1, 3] } else { vec![1] });
            for chunk in chunks {
                corpus.apply(&chunk.starts, chunk.merged_id);
            }
            assert_eq!(
                &tokens(&corpus)[6..8],
                &[0, 1],
                "an unrepresented word is never scanned"
            );
        }
    }
}
