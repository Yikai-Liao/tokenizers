//! Private joined preparation and application over corpus geometry.
use super::super::{TrainingMode, WORD_SEPARATOR_ID, add, positions::Positions};
use super::{Corpus, PairChange, ProducerChanges, SelectedMerge};
use ahash::AHashMap;
use itertools::Itertools;
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
    ) -> Result<Vec<ProducerChanges>> {
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
                    // A historical start must be at or after the previous accepted end
                    // before matching; overlapping AA starts never read endpoints.
                    if coordinate >= previous_end
                        && let Some(matched) = matcher(self.resident(coordinate))
                    {
                        accepted_starts.push(coordinate);
                        previous_end = matched.end as u64;
                    }
                }
                drop(positions);
                // Budget from the original historical payload, not accepted starts.
                // Aim for four tasks per worker, with at least 4096 starts per task;
                // ceil division covers any remainder in the original payload.
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
                                    // All AA matches have the same length. The
                                    // preceding adjacent match would start one
                                    // length earlier; guard subtraction, then check
                                    // the global acceptance set across task cuts.
                                    matched.start >= matched.len()
                                        && accepted_starts
                                            .binary_search(
                                                &((matched.start - matched.len()) as u64),
                                            )
                                            .is_ok()
                                },
                                |after, _| {
                                    // An accepted start at this match's exclusive
                                    // end replaces the right neighbor with the
                                    // same merged ID, even in another task.
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
                // Aim for one task per worker; keep nonempty chunks and cap
                // decoding work at 2^26 historical positions per chunk.
                let chunk = total
                    .div_ceil(super::super::num_workers())
                    .clamp(1, 1 << 26);
                let mut tasks = Vec::new();
                for rule in &rules {
                    let mut parts = rule.positions.chunks(chunk).peekable();
                    if let Some(first) = parts.next() {
                        let complete = parts.peek().is_none();
                        tasks.push((rule.pair, rule.merged_id, first, complete));
                        tasks.extend(
                            parts.map(|positions| (rule.pair, rule.merged_id, positions, false)),
                        );
                    }
                }
                // Rule order, then restart-aligned chunk order, remains indexed.
                tasks
                    .into_maybe_par_iter()
                    .map_init(
                        NeighborScratch::default,
                        |scratch, (pair, merged_id, positions, complete)| {
                            let mut prepared = preparation.prepare(
                                pair,
                                merged_id,
                                positions,
                                scratch,
                                |_, before, prior| {
                                    // before is the prior token's start. The token
                                    // preceding prior ends at before - 1; a selected
                                    // pair ending in prior owns this shared left edge.
                                    // Coordinate zero has no preceding endpoint.
                                    before != 0
                                        && merged_ids_by_pair
                                            .contains_key(&(self.token(before - 1), prior))
                                },
                                |after, next| {
                                    // Skip the whole right neighbor to read the
                                    // token after it. If that pair is selected,
                                    // this edge must use its final merged ID.
                                    merged_ids_by_pair
                                        .get(&(next, self.token(after + self.fixed_id_span(next))))
                                        .copied()
                                },
                            )?;
                            if complete {
                                prepared.changes = match prepared.changes {
                                    ProducerChanges::Fragment(changes) => {
                                        ProducerChanges::WholeRule(changes)
                                    }
                                    ProducerChanges::WholeRule(_) => unreachable!(),
                                };
                            }
                            Ok(prepared)
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
        // Divide whole word domains across workers, rounding up and retaining
        // a nonzero chunk size even when the historical list contains no words.
        let chunk = words.len().div_ceil(super::super::num_workers()).max(1);
        words
            .maybe_par_chunks(chunk)
            .map_init(NeighborScratch::default, |scratch, words| {
                self.prepare_chunk(rule.pair, rule.merged_id, scratch, |neighbors, starts| {
                    for &word in words {
                        self.scan_word(&rule, word, neighbors, starts)?;
                    }
                    Ok(())
                })
            })
            .collect()
    }

    // Each task owns one accumulator and result. Failed scans publish no chunk;
    // borrowed scratch remains reusable through its checked directory entries.
    fn prepare_chunk(
        &self,
        pair: Pair,
        merged_id: u32,
        scratch: &mut NeighborScratch,
        scan: impl FnOnce(&mut NeighborChanges<'_>, &mut SmallVec<[usize; 2]>) -> Result<()>,
    ) -> Result<PreparedChunk> {
        let mut neighbors = NeighborChanges::new(pair, merged_id, self.id_count(), scratch);
        let mut starts = SmallVec::new();
        scan(&mut neighbors, &mut starts)?;
        Ok(PreparedChunk {
            starts,
            merged_id,
            changes: ProducerChanges::Fragment(neighbors.finish()),
        })
    }

    fn scan_word(
        &self,
        rule: &SelectedMerge,
        word: usize,
        neighbors: &mut NeighborChanges<'_>,
        starts: &mut SmallVec<[usize; 2]>,
    ) -> Result<()> {
        let limit = self.birth_pair_span_limit.unwrap_or(usize::MAX);
        let range = self.word_range(word);
        let weight = self.weight(range.start).weight;
        let mut p = range.start;
        let mut positions = rule.positions.iter_from(p as u64);
        let mut left_neighbor: Option<Neighbor> = None;
        while p < range.end {
            // Full scans are required after alias activation or any Some length gate.
            // Otherwise only eligible coordinates in this historical word are visited.
            if !self.whole_words() {
                let Some(coordinate) = positions.find(|&start| start >= p as u64) else {
                    break;
                };
                let next = self.resident(coordinate);
                if next >= range.end {
                    break;
                }
                // Before alias activation, consuming a historical left start
                // replaces its ID permanently. Reject stale starts before reading
                // next - 1: interior slots do not identify the current predecessor.
                if self.token(next) != rule.pair.0 {
                    continue;
                }
                if next != p {
                    // The slot just before a sought start is the preceding
                    // token's tail. Its span reconstructs that token's start.
                    let id = self.token(next - 1);
                    left_neighbor = (id != WORD_SEPARATOR_ID).then(|| {
                        let span = self.occurrence_span(next - 1);
                        Neighbor {
                            token_id: id,
                            start: next - span,
                            span,
                        }
                    });
                    p = next;
                }
            }
            if let Some(matched) = self.match_at(p, rule.pair) {
                if let Some(left) = left_neighbor {
                    neighbors.record(
                        EdgeSide::Left,
                        left.token_id,
                        left.token_id,
                        left.start,
                        weight,
                        // Gate the new pair's combined occurrence length;
                        // equality is excluded by the strict birth-length limit.
                        left.span + matched.len() < limit,
                    )?;
                }
                let next = self.token(matched.end);
                if next != WORD_SEPARATOR_ID {
                    neighbors.record(
                        EdgeSide::Right,
                        next,
                        next,
                        p,
                        weight,
                        // The right neighbor comes from the joined snapshot;
                        // its span plus this match is the new pair's length.
                        matched.len() + self.occurrence_span(matched.end) < limit,
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
                let span = self.occurrence_span(p);
                left_neighbor = Some(Neighbor {
                    token_id: self.token(p),
                    start: p,
                    span,
                });
                // Whole-word traversal and the surviving historical-head check
                // both keep p at a current token start with a nonzero span.
                p += span;
            }
        }
        Ok(())
    }
}

enum FreshBatch {
    AA(SelectedMerge),
    AB(Vec<SelectedMerge>),
}

struct PreparedChunk {
    starts: SmallVec<[usize; 2]>,
    merged_id: u32,
    changes: ProducerChanges,
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
        let changes = &mut self.changes[side as usize];
        let slot = &mut self.scratch.indices[side as usize][neighbor as usize];
        // Scratch indices survive earlier tasks. Reuse a slot only when it
        // names a current group whose neighbor occupies this side's pair field.
        if changes
            .get(*slot as usize)
            .is_none_or(|change| (if left { change.born.0 } else { change.born.1 }) != neighbor)
        {
            let id = self.merged_id;
            let pair = self.pair;
            let index = changes.len();
            changes.push(PairChange {
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
        &mut changes[*slot as usize]
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
        // Removal-only events need a nonzero weight; zero-weight births still
        // carry historical positions and must reach the index.
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
        self.corpus
            .prepare_chunk(pair, merged_id, scratch, |neighbors, starts| {
                // PERF: Borrow the local decoder so lookahead adapters hold a pointer.
                // In the measured x86-64 release build, owning decoder state added a
                // 144-byte reader-state copy per AB task and increased the stack frame.
                let mut positions = positions.into_iter();
                for (matched, weight) in self.corpus.fresh_matches(&mut positions, pair) {
                    self.record(neighbors, &matched, weight, &left_merging, &right_merged_id)?;
                    starts.push(matched.start);
                }
                Ok(())
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
                    // Admit only a neighbor pair strictly shorter than the
                    // limit, using prior-token span plus merged-match span.
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
                // A selected right neighbor may also merge: use its final
                // ID's span in the strict combined-length gate.
                matched.len() + self.corpus.fixed_id_span(born) < self.birth_pair_span_limit,
            )?;
        }
        Ok(())
    }
}
