//! Joined training rounds over a fixed-coordinate corpus and an occurrence index.
mod corpus;
mod index;
mod merge;
mod positions;
mod vocabulary;

use super::{BpeTrainer, word_counts::WordCountsView};
use crate::progress::TrainingProgress;
use corpus::{Corpus, CorpusPlan};
use index::PairIndex;
use merge::{Batch, Scratch};
#[cfg(test)]
use tk_encode::models::bpe::Pair;
use tk_encode::{
    Result,
    models::bpe::{Merges, Vocab},
    vocab::bucket_added_vocabulary::AddedToken,
};
use vocabulary::Vocabulary;

const WORD_SEPARATOR_ID: u32 = u32::MAX;
type ModelParts = (Vocab, Merges, Vec<AddedToken>);
#[cfg(test)]
type Trace = Vec<(Pair, u64, u32)>;

pub(super) fn train(
    trainer: &BpeTrainer,
    words: WordCountsView<'_>,
    workers: usize,
    #[cfg(test)] mut observe: Option<&mut (dyn FnMut(Pair, u64, u32) + Send)>,
) -> Result<ModelParts> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()?;
    pool.install(|| {
        let progress = TrainingProgress::new(trainer.show_progress, trainer.progress_format)?;
        let scratch = Scratch::new(workers);
        let mut alphabet = None;
        let mut reuse = false;
        loop {
            let mut vocabulary =
                Vocabulary::initialize(trainer, words, workers, &progress, &mut alphabet)?;
            let plan = CorpusPlan::build(words, &mut vocabulary, trainer, reuse, &progress)?;
            let mut index =
                PairIndex::build(&plan, trainer.min_frequency, workers, reuse, &progress)?;
            if vocabulary.len() >= trainer.vocab_size {
                progress.stage("Compute merges", trainer.vocab_size);
                let (vocab, merges) = vocabulary.into_model_parts(Vec::new());
                return Ok((vocab, merges, trainer.special_tokens.clone()));
            }
            let mut corpus = plan.materialize(&progress);
            let work = progress.stage("Compute merges", trainer.vocab_size);
            let mut merges = Vec::new();
            #[cfg(test)]
            let mut trace = Trace::new();
            let mut restart = false;
            while vocabulary.len() < trainer.vocab_size {
                let Some(batch) = Batch::select(trainer, &mut vocabulary, &mut corpus, &mut index)?
                else {
                    break;
                };
                if batch.restart() {
                    restart = true;
                    break;
                }
                #[cfg(test)]
                trace.extend(batch.trace());
                merges.extend(batch.pairs());
                let prepared = batch.prepare(
                    &corpus,
                    trainer.max_token_length.unwrap_or(usize::MAX),
                    &scratch,
                )?;
                let changes = prepared.apply(&corpus);
                index.commit(changes)?;
                work.learned(merges.len());
            }
            if restart {
                reuse = true;
                continue;
            }
            drop(index);
            drop(corpus);
            #[cfg(test)]
            if let Some(observer) = observe.as_mut() {
                for (pair, count, id) in trace {
                    observer(pair, count, id);
                }
            }
            drop(scratch);
            let (vocab, merges) = vocabulary.into_model_parts(merges);
            return Ok((vocab, merges, trainer.special_tokens.clone()));
        }
    })
}

#[cfg(test)]
mod tests;
