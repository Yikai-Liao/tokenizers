//! Merge contracts and application of prepared writes after all readers join.
mod events;
mod prepare;
use super::{
    corpus::{Corpus, SlotStorage},
    storage::PositionBuffer,
};
pub(super) use events::{ChangeAction, EventChunk, MergeEvents, PairChanges};
pub(super) use prepare::{
    MergeOptions, MergeScratch, SelectedRuleIndex, prepare_merges_with_births,
};
use rayon::prelude::*;
use tk_encode::models::bpe::Pair;
#[derive(Clone, Copy)]
pub(super) struct MergeRule {
    pub(super) pair: Pair,
    pub(super) replacement: u32,
}
struct WritePlan {
    rule: MergeRule,
    positions: PositionBuffer,
}
struct PreparedJob {
    writes: Vec<WritePlan>,
    word_region: Option<std::ops::Range<u64>>,
}
pub(super) struct PreparedMerges {
    jobs: Vec<PreparedJob>,
    events: MergeEvents,
}
impl PreparedMerges {
    pub(super) fn apply<S: SlotStorage>(self, corpus: &mut Corpus<S>) -> MergeEvents {
        if corpus.has_occurrence_spans() {
            let regions: Vec<_> = self
                .jobs
                .iter()
                .map(|job| {
                    job.word_region
                        .clone()
                        .expect("occurrence spans require complete whole-word jobs")
                })
                .collect();
            let writers = corpus
                .word_writers(&regions)
                .expect("the occurrence plane is materialized");
            self.jobs
                .par_iter()
                .zip(writers.into_par_iter())
                .for_each(|(job, mut writer)| {
                    for write in &job.writes {
                        write
                            .positions
                            .positions()
                            .for_each(|position| writer.merge(position, write.rule.replacement));
                    }
                });
        } else {
            self.jobs.par_iter().for_each(|job| {
                for write in &job.writes {
                    let matcher = corpus.matcher(write.rule.pair);
                    write.positions.positions().for_each(|position| {
                        // SAFETY: preparation selected disjoint endpoint spans.
                        // apply holds the mutable corpus borrow until pool join;
                        // geometry reads immutable ID spans, never token IDs.
                        unsafe {
                            corpus.write_endpoints(
                                matcher.geometry(position),
                                write.rule.replacement,
                            );
                        }
                    });
                }
            });
        }
        self.events
    }
}
