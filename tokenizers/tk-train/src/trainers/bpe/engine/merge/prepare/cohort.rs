//! Identity-reuse cohorts preserve intermediate word boundaries and birth semantics.
use super::*;
use rayon::prelude::*;
enum CohortSource {
    WordIndices(Vec<usize>),
    Positions(std::ops::Range<usize>),
}
struct CohortTask {
    region: std::ops::Range<u64>,
    source: CohortSource,
}
pub(super) fn prepare<S: SlotStorage>(
    corpus: &Corpus<S>,
    rule: &MergeRule,
    candidate: &MergeCandidate<'_>,
    token_id_count: usize,
    birth_span_limit: u64,
    execution: &Execution,
) -> Result<Vec<(PreparedJob, Vec<EventChunk>)>> {
    let count = candidate.positions.len();
    let chunk = count.div_ceil(execution.workers()).max(1);
    let tasks = if corpus.needs_word_scan() {
        // Ordered cohorts make word IDs nondecreasing. Each range caches its
        // current word, then adjacent dedup joins range boundaries as well.
        let parts: Vec<Vec<usize>> = (0..count)
            .step_by(chunk)
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|begin| {
                let mut words = Vec::new();
                let mut current = None;
                for position in candidate
                    .positions
                    .cursor(begin..(begin + chunk).min(count))
                {
                    let word = match current {
                        Some((word, end)) if position < end => word,
                        _ => corpus.word_containing(position),
                    };
                    current = Some((word, corpus.word_end(word)));
                    if words.last() != Some(&word) {
                        words.push(word);
                    }
                }
                words
            })
            .collect();
        let mut words: Vec<_> = parts.into_iter().flatten().collect();
        words.dedup();
        let chunk = words.len().div_ceil(execution.workers()).max(1);
        words
            .chunks(chunk)
            .enumerate()
            .map(|(index, part)| {
                let start = if index == 0 {
                    0
                } else {
                    corpus.word_start(part[0])
                };
                let end = words
                    .get((index + 1) * chunk)
                    .map_or(corpus.len() as u64, |&word| corpus.word_start(word));
                CohortTask {
                    region: start..end,
                    source: CohortSource::WordIndices(part.to_vec()),
                }
            })
            .collect::<Vec<_>>()
    } else {
        // Before alias reuse or a length gate, only cut boundaries need a word
        // lookup. Move each cut to the first occurrence in that complete word.
        let mut cuts = vec![(0, 0)];
        for desired in (chunk..count).step_by(chunk) {
            let position = candidate
                .positions
                .cursor(desired..desired + 1)
                .next()
                .expect("the cut index belongs to the cohort");
            let pivot = corpus.word_start(corpus.word_containing(position));
            let begin = candidate.positions.lower_bound(pivot);
            if begin > cuts.last().expect("the first cut exists").0 {
                cuts.push((begin, pivot));
            }
        }
        cuts.push((count, corpus.len() as u64));
        cuts.windows(2)
            .map(|cuts| CohortTask {
                region: cuts[0].1..cuts[1].1,
                source: CohortSource::Positions(cuts[0].0..cuts[1].0),
            })
            .collect()
    };
    tasks
        .into_par_iter()
        .map(|task| -> Result<_> {
            execution.with_merge_scratch(token_id_count, |scratch| {
                let mut plan = RulePreparation::new(corpus, scratch, rule, 0, birth_span_limit);
                match task.source {
                    CohortSource::WordIndices(words) => {
                        for word in words {
                            let mut previous = None;
                            let mut position = corpus.word_start(word);
                            while corpus.token(position) != WORD_SEPARATOR_ID {
                                if let Some(matched) = plan.matcher.get(position) {
                                    previous = Some(plan.record_cohort_match(matched, previous)?);
                                    position = matched.next_start;
                                } else {
                                    let span = corpus.span(position);
                                    previous = Some(LogicalToken {
                                        id: corpus.token(position),
                                        start: position,
                                        span,
                                    });
                                    position += span;
                                }
                            }
                        }
                    }
                    CohortSource::Positions(range) => {
                        let mut previous = None;
                        let mut after = task.region.start;
                        for position in candidate.positions.cursor(range) {
                            if position < after {
                                continue;
                            }
                            if let Some(matched) = plan.matcher.get(position) {
                                if position != after {
                                    let id = corpus.token(position - 1);
                                    previous = (id != WORD_SEPARATOR_ID).then(|| {
                                        let span = corpus.span(position - 1);
                                        LogicalToken {
                                            id,
                                            start: position - span,
                                            span,
                                        }
                                    });
                                }
                                previous = Some(plan.record_cohort_match(matched, previous)?);
                                after = matched.next_start;
                            }
                        }
                    }
                }
                let (write, mut chunks) = plan.finish();
                chunks.push(scratch.take_chunk());

                Ok((
                    PreparedJob {
                        writes: vec![write],
                        word_region: Some(task.region),
                    },
                    chunks,
                ))
            })
        })
        .collect()
}

/// One logical token after earlier matches in this word, before endpoint writes.
#[derive(Clone, Copy)]
struct LogicalToken {
    id: u32,
    start: u64,
    span: u64,
}

impl<S: SlotStorage> RulePreparation<'_, S> {
    /// Record a reuse match against the logical preceding token; preserve
    /// intermediate boundaries and removal-before-birth neighbor accounting.
    fn record_cohort_match(
        &mut self,
        matched: PairMatch,
        previous: Option<LogicalToken>,
    ) -> Result<LogicalToken> {
        self.room::<false>();
        let weight = self.weights.weight(matched.left_start);
        if let Some(previous) = previous {
            self.scratch.left::<false>(
                previous.id,
                previous.start,
                weight,
                previous.span + matched.merged_span < self.birth_span_limit,
            )?;
        }
        let next = self.corpus.token(matched.next_start);
        if next != WORD_SEPARATOR_ID {
            self.scratch.right::<false>(
                next,
                next,
                matched.left_start,
                weight,
                matched.merged_span + self.corpus.span(matched.next_start) < self.birth_span_limit,
            )?;
        }
        self.positions.push_position(matched.left_start);
        Ok(LogicalToken {
            id: self.rule.replacement,
            start: matched.left_start,
            span: matched.merged_span,
        })
    }
}
