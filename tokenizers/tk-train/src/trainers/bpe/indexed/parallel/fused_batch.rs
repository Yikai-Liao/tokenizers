//! Flat, non-AA batches: read and route once, join, then write disjoint spans.
//! Jobs cover consecutive sections of (rule rank, ordered posting) so each
//! birth key's unique producer also supplies ordered positions across jobs.
use super::weight_lookup::WeightLookup;

mod aggregate;
use super::*;

const EMPTY: u64 = u64::MAX;
const MULTIPLE: u64 = u64::MAX - 1;
struct Selected {
    heads: Vec<u64>,
    tails: Vec<u64>,
    multiple: AHashMap<u64, u32>,
}
impl Selected {
    fn new(rules: &[Rule], identities: usize) -> Self {
        let mut heads = vec![EMPTY; identities];
        let mut tails = vec![EMPTY; identities];
        let mut multiple = AHashMap::new();
        for r in rules {
            let head = &mut heads[r.edge.0 as usize];
            *head = if *head == EMPTY {
                key(r.edge.1, r.replacement)
            } else {
                MULTIPLE
            };
            let tail = &mut tails[r.edge.1 as usize];
            *tail = if *tail == EMPTY {
                key(r.edge.0, r.replacement)
            } else {
                MULTIPLE
            };
        }
        for r in rules {
            if heads[r.edge.0 as usize] == MULTIPLE || tails[r.edge.1 as usize] == MULTIPLE {
                multiple.insert(key(r.edge.0, r.edge.1), r.replacement);
            }
        }
        Self {
            heads,
            tails,
            multiple,
        }
    }
    fn bytes(&self) -> usize {
        (self.heads.capacity() + self.tails.capacity()) * 8
            + table_bytes(self.multiple.capacity(), 16)
    }
    fn left_selected<C: Slot>(&self, corpus: &[C], before: usize, prior: u32) -> bool {
        let tail = self.tails[prior as usize];
        if tail == EMPTY {
            return false;
        }
        let previous = corpus[before - 1].token();
        if tail == MULTIPLE {
            self.multiple.contains_key(&key(previous, prior))
        } else {
            (tail >> 32) as u32 == previous
        }
    }
    fn final_next<C: Slot>(&self, corpus: &[C], after: usize, next: u32, lengths: &[usize]) -> u32 {
        let head = self.heads[next as usize];
        if head == EMPTY {
            return next;
        }
        let following = corpus[after + lengths[next as usize]].token();
        if head == MULTIPLE {
            self.multiple
                .get(&key(next, following))
                .copied()
                .unwrap_or(next)
        } else if (head >> 32) as u32 == following {
            head as u32
        } else {
            next
        }
    }
}

struct Task<'a> {
    rank: usize,
    positions: &'a [u32],
}
struct Valid {
    rank: usize,
    positions: Vec<u32>,
}
pub(super) struct Prepared<O: Offset, const INLINE: usize> {
    valid: Vec<Vec<Valid>>,
    pub(super) outputs: Vec<Output<O, INLINE>>,
    pub(super) valid_bytes: usize,
    pub(super) selected_bytes: usize,
    pub(super) aggregate_bytes: usize,
}

pub(super) fn prepare<C: Slot, O: Offset, const INLINE: usize>(
    corpus: &[C],
    rules: &[Rule],
    postings: &[SmallPosting],
    block: &Block<O, INLINE>,
    lengths: &[usize],
    uniform: Option<u64>,
    weight_lookup: Option<&WeightLookup>,
    max_length: usize,
    workers: usize,
) -> Result<Prepared<O, INLINE>> {
    // Cap per-job directory storage; this does not narrow canonical IDs.
    if lengths.len() <= 65_536 {
        prepare_with_mode::<C, O, INLINE, true>(
            corpus,
            rules,
            postings,
            block,
            lengths,
            uniform,
            weight_lookup,
            max_length,
            workers,
        )
    } else {
        prepare_with_mode::<C, O, INLINE, false>(
            corpus,
            rules,
            postings,
            block,
            lengths,
            uniform,
            weight_lookup,
            max_length,
            workers,
        )
    }
}

fn prepare_with_mode<C: Slot, O: Offset, const INLINE: usize, const GROUPED: bool>(
    corpus: &[C],
    rules: &[Rule],
    postings: &[SmallPosting],
    block: &Block<O, INLINE>,
    lengths: &[usize],
    uniform: Option<u64>,
    weight_lookup: Option<&WeightLookup>,
    max_length: usize,
    workers: usize,
) -> Result<Prepared<O, INLINE>> {
    debug_assert!(C::SHARED);
    debug_assert!(rules.iter().all(|r| r.edge.0 != r.edge.1));
    let selected = Selected::new(rules, lengths.len());
    let total: usize = postings.iter().map(SmallPosting::len).sum();
    let chunk = total.div_ceil(workers).max(1);
    // Each job owns one route set, even when a batch contains many tiny rules.
    let mut jobs: Vec<Vec<Task<'_>>> = Vec::new();
    let mut visited = 0;
    for (rank, posting) in postings.iter().enumerate() {
        let mut positions = posting.as_slice();
        while !positions.is_empty() {
            let job = visited / chunk;
            if jobs.len() == job {
                jobs.push(Vec::new());
            }
            let take = (chunk - visited % chunk).min(positions.len());
            jobs[job].push(Task {
                rank,
                positions: &positions[..take],
            });
            visited += take;
            positions = &positions[take..];
        }
    }
    let prepared: Vec<_> = jobs
        .par_iter()
        .map(|tasks| -> Result<_> {
            let mut output = Output::new(workers, true);
            let mut left_cache = aggregate::Scratch::new(if GROUPED { lengths.len() } else { 0 });
            let mut right_cache = aggregate::Scratch::new(if GROUPED { lengths.len() } else { 0 });
            let mut valid = Vec::with_capacity(tasks.len());
            for task in tasks {
                let rule = &rules[task.rank];
                let mut positions = Vec::new();
                let mut cursor = 0;
                for &position in task.positions {
                    let p = position as usize;
                    let right = p + rule.left_len;
                    if corpus[p].token() != rule.edge.0
                        || right >= corpus.len()
                        || corpus[right].token() != rule.edge.1
                    {
                        continue;
                    }
                    positions.push(position);
                    let after = right + rule.right_len;
                    let weight = if let Some(weight) = uniform {
                        weight
                    } else if let Some(lookup) = weight_lookup {
                        lookup.weight(block, position)
                    } else {
                        block.weight_forward(p, None, &mut cursor)
                    };
                    let prior = corpus[p - 1].token();
                    if prior != NONE {
                        let before = p - lengths[prior as usize];
                        // If the left neighbor is itself selected, its right delta
                        // accounts for this boundary using the final two outputs.
                        let left_selected = selected.left_selected(corpus, before, prior);
                        if !left_selected {
                            if GROUPED {
                                left_cache.remove(prior, weight);
                            } else {
                                output.remove(key(prior, rule.edge.0), weight);
                            }
                            if lengths[prior as usize] + rule.length() < max_length {
                                if GROUPED {
                                    left_cache.birth(
                                        &mut output,
                                        prior,
                                        key(prior, rule.replacement),
                                        before,
                                        weight,
                                    )?;
                                } else {
                                    output.birth(
                                        key(prior, rule.replacement),
                                        before,
                                        weight,
                                        32,
                                    )?;
                                }
                            }
                        }
                    }
                    let next = corpus[after].token();
                    if next != NONE {
                        if GROUPED {
                            right_cache.remove(next, weight);
                        } else {
                            output.remove(key(rule.edge.1, next), weight);
                        }
                        let final_next = selected.final_next(corpus, after, next, lengths);
                        if rule.length() + lengths[final_next as usize] < max_length {
                            if GROUPED {
                                right_cache.birth(
                                    &mut output,
                                    final_next,
                                    key(rule.replacement, final_next),
                                    p,
                                    weight,
                                )?;
                            } else {
                                output.birth(key(rule.replacement, final_next), p, weight, 32)?;
                            }
                        }
                    }
                }
                if GROUPED {
                    left_cache.flush(&mut output, rule, true)?;
                    right_cache.flush(&mut output, rule, false)?;
                }
                valid.push(Valid {
                    rank: task.rank,
                    positions,
                });
            }
            Ok((valid, output, left_cache.bytes() + right_cache.bytes()))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut valid = Vec::with_capacity(prepared.len());
    let mut outputs = Vec::with_capacity(prepared.len());
    let mut valid_bytes = 0;
    let mut aggregate_bytes = 0;
    for (v, output, bytes) in prepared {
        aggregate_bytes += bytes;
        valid_bytes += v.iter().map(|v| v.positions.capacity() * 4).sum::<usize>();
        valid.push(v);
        outputs.push(output);
    }
    Ok(Prepared {
        valid,
        outputs,
        valid_bytes,
        selected_bytes: selected.bytes(),
        aggregate_bytes,
    })
}

impl<O: Offset, const INLINE: usize> Prepared<O, INLINE> {
    pub(super) fn apply<C: Slot>(&self, corpus: &[C], rules: &[Rule]) {
        // All prepare reads have joined. Non-AA certificate gives disjoint
        // merged spans. Atomic stores permit mutation through shared references;
        // this par_iter joins before commit or the next batch can read corpus.
        debug_assert!(C::SHARED);
        self.valid.par_iter().for_each(|job| {
            for valid in job {
                let rule = &rules[valid.rank];
                for &position in &valid.positions {
                    let p = position as usize;
                    let right = p + rule.left_len;
                    corpus[p].set_shared(rule.replacement);
                    if rule.right_len == 1 {
                        corpus[right].set_shared(rule.replacement);
                    } else {
                        corpus[right].set_shared(NONE);
                        corpus[right + rule.right_len - 1].set_shared(rule.replacement);
                    }
                }
            }
        });
    }
}
