//! Flat, non-AA batches: read and route once, join, then write disjoint spans.
//! Jobs cover consecutive sections of (rule rank, ordered posting) so each
//! birth key's unique producer also supplies ordered positions across jobs.
use super::*;

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
}

pub(super) fn prepare<C: Slot, O: Offset, const INLINE: usize>(
    corpus: &[C],
    rules: &[Rule],
    postings: &[SmallPosting],
    block: &Block<O, INLINE>,
    lengths: &[usize],
    uniform: Option<u64>,
    max_length: usize,
    workers: usize,
) -> Result<Prepared<O, INLINE>> {
    debug_assert!(C::SHARED);
    debug_assert!(rules.iter().all(|r| r.edge.0 != r.edge.1));
    let selected: AHashMap<u64, u32> = rules
        .iter()
        .map(|r| (key(r.edge.0, r.edge.1), r.replacement))
        .collect();
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
                    let weight = block.weight_forward(p, uniform, &mut cursor);
                    let prior = corpus[p - 1].token();
                    if prior != NONE {
                        let before = p - lengths[prior as usize];
                        let previous = corpus[before - 1].token();
                        // If the left neighbor is itself selected, its right delta
                        // accounts for this boundary using the final two outputs.
                        let left_selected =
                            previous != NONE && selected.contains_key(&key(previous, prior));
                        if !left_selected {
                            output.remove(key(prior, rule.edge.0), weight);
                            if lengths[prior as usize] + rule.length() < max_length {
                                output.birth(key(prior, rule.replacement), before, weight, 32)?;
                            }
                        }
                    }
                    let next = corpus[after].token();
                    if next != NONE {
                        output.remove(key(rule.edge.1, next), weight);
                        let following = corpus[after + lengths[next as usize]].token();
                        let final_next = if following == NONE {
                            next
                        } else {
                            selected.get(&key(next, following)).copied().unwrap_or(next)
                        };
                        if rule.length() + lengths[final_next as usize] < max_length {
                            output.birth(key(rule.replacement, final_next), p, weight, 32)?;
                        }
                    }
                }
                valid.push(Valid {
                    rank: task.rank,
                    positions,
                });
            }
            Ok((valid, output))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut valid = Vec::with_capacity(prepared.len());
    let mut outputs = Vec::with_capacity(prepared.len());
    let mut valid_bytes = 0;
    for (v, output) in prepared {
        valid_bytes += v.iter().map(|v| v.positions.capacity() * 4).sum::<usize>();
        valid.push(v);
        outputs.push(output);
    }
    Ok(Prepared {
        valid,
        outputs,
        valid_bytes,
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
