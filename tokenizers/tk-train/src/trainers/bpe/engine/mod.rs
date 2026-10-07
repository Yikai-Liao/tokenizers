//! One BPE coordinator over shared vocabulary, corpus, and occurrence storage.
//! The visible round is select, prepare, release candidates, apply, commit, then
//! release events. Preparation readers and corpus writers join before the next
//! phase. Errors discard the attempt; commit may fail after writes/partial counts.
//! Active-ID reuse restarts from original words with the already selected alphabet.
use crate::trainers::bpe::word_counts::WordCountsView;
mod aa_parity;
mod batch;
mod corpus;
mod execution;
mod initial_pairs;
mod merge;
mod pair_index;
mod storage;
mod vocabulary;
use super::BpeTrainer;
use crate::progress::TrainingProgress;
#[cfg(test)]
use ahash::AHashMap;
use batch::{BatchSelection, RuleBatch};
#[cfg(test)]
use compact_str::CompactString;
use storage::AllocationArena;
use tk_encode::{
    Result,
    models::bpe::{Merges, Vocab},
    vocab::bucket_added_vocabulary::AddedToken,
};
const WORD_SEPARATOR_ID: u32 = u32::MAX;
#[derive(Clone, Copy, PartialEq, Eq)]
enum IdentityPolicy {
    /// New IDs and first activations of reserved IDs; existing keys cannot revive.
    FirstActivationOnly,
    /// Active identities may revive keys; rebuild with a signed ledger and cohorts.
    AllowActiveReuse,
}
type ModelParts = (Vocab, Merges, Vec<AddedToken>);
enum AttemptOutcome {
    Complete(ModelParts),
    RestartForReuse,
}
pub(super) fn train(
    trainer: &BpeTrainer,
    word_counts: WordCountsView<'_>,
    workers: usize,
    #[cfg(test)] observe: Option<&mut (dyn FnMut(tk_encode::models::bpe::Pair, u64, u32) + Send)>,
) -> Result<ModelParts> {
    train_with_merge_options(
        trainer,
        word_counts,
        workers,
        merge::MergeOptions::default(),
        #[cfg(test)]
        observe,
    )
}
fn train_with_merge_options(
    trainer: &BpeTrainer,
    word_counts: WordCountsView<'_>,
    workers: usize,
    merge_options: merge::MergeOptions,
    #[cfg(test)] mut observe: Option<
        &mut (dyn FnMut(tk_encode::models::bpe::Pair, u64, u32) + Send),
    >,
) -> Result<ModelParts> {
    let execution = execution::Execution::new(workers)?;
    execution.pool.install(|| {
        let progress = TrainingProgress::new(trainer.show_progress, trainer.progress_format)?;
        // Every attempt begins with first activations. A nonempty affix does
        // not by itself require one-rule cohort execution. Stop before accepting
        // the first reused active ID, then rebuild with the same coordinator.
        let mut policy = IdentityPolicy::FirstActivationOnly;
        // Alphabet frequency ties are intentionally unchanged. A restart must
        // retain this call's choice rather than sample the selector again.
        let mut retained_alphabet = None;
        #[cfg(test)]
        let mut trace = Vec::new();
        loop {
            #[cfg(test)]
            trace.clear();
            match train_attempt(
                trainer,
                word_counts,
                policy,
                &execution,
                merge_options,
                &progress,
                &mut retained_alphabet,
                #[cfg(test)]
                &mut trace,
            )? {
                AttemptOutcome::Complete(parts) => {
                    // Publish only the successfully completed attempt's trace;
                    // traces from abandoned attempts were discarded.
                    #[cfg(test)]
                    if let Some(observer) = observe.as_mut() {
                        for (pair, count, id) in trace {
                            observer(pair, count, id);
                        }
                    }
                    return Ok(parts);
                }
                AttemptOutcome::RestartForReuse => {
                    // All position lists and their arena were dropped by the attempt.
                    // Retain neither speculative values nor encoding allocations.
                    execution.release_scratch();
                    policy = IdentityPolicy::AllowActiveReuse;
                }
            }
        }
    })
}
#[cfg_attr(test, allow(clippy::too_many_arguments))]
fn train_attempt(
    trainer: &BpeTrainer,
    word_counts: WordCountsView<'_>,
    policy: IdentityPolicy,
    execution: &execution::Execution,
    merge_options: merge::MergeOptions,
    progress: &TrainingProgress,
    retained_alphabet: &mut Option<Vec<char>>,
    #[cfg(test)] trace: &mut Vec<(tk_encode::models::bpe::Pair, u64, u32)>,
) -> Result<AttemptOutcome> {
    let workers = execution.workers();
    let mut vocabulary = vocabulary::Vocabulary::initialize(
        trainer,
        word_counts,
        workers,
        progress,
        retained_alphabet,
    )?;
    let mut prepared_corpus = corpus::CorpusPlan::build(
        word_counts,
        &mut vocabulary,
        policy,
        trainer.max_token_length.is_some(),
        progress,
    )?;
    if vocabulary.len() >= trainer.vocab_size && prepared_corpus.initial_counts_fit_u64() {
        drop(prepared_corpus);
        execution.release_scratch();
        progress.stage("Compute merges", trainer.vocab_size);
        return Ok(complete_model(trainer, vocabulary, Vec::new()));
    }
    prepared_corpus.prepare_initial_symbols(workers, progress)?;
    match corpus::slot_bits(trainer.vocab_size.max(vocabulary.len())) {
        16 => train_with_slots::<corpus::U16Slots>(
            trainer,
            vocabulary,
            prepared_corpus,
            policy,
            execution,
            merge_options,
            progress,
            #[cfg(test)]
            trace,
        ),
        24 => train_with_slots::<corpus::PackedU24Slots>(
            trainer,
            vocabulary,
            prepared_corpus,
            policy,
            execution,
            merge_options,
            progress,
            #[cfg(test)]
            trace,
        ),
        _ => train_with_slots::<corpus::U32Slots>(
            trainer,
            vocabulary,
            prepared_corpus,
            policy,
            execution,
            merge_options,
            progress,
            #[cfg(test)]
            trace,
        ),
    }
}
#[cfg_attr(test, allow(clippy::too_many_arguments))]
fn train_with_slots<S: corpus::SlotStorage>(
    trainer: &BpeTrainer,
    mut vocabulary: vocabulary::Vocabulary,
    prepared_corpus: corpus::CorpusPlan<'_>,
    policy: IdentityPolicy,
    execution: &execution::Execution,
    merge_options: merge::MergeOptions,
    progress: &TrainingProgress,
    #[cfg(test)] trace: &mut Vec<(tk_encode::models::bpe::Pair, u64, u32)>,
) -> Result<AttemptOutcome> {
    let workers = execution.workers();

    let arena = AllocationArena::new(workers, prepared_corpus.initial_edges());
    let initial = initial_pairs::InitialPairTable::build(
        &prepared_corpus,
        if policy == IdentityPolicy::FirstActivationOnly {
            trainer.min_frequency.max(1)
        } else {
            0
        },
        execution,
        &arena,
        progress,
    )?;
    if vocabulary.len() >= trainer.vocab_size {
        // The total-mass proof was inconclusive. Initial construction has now
        // retained every checked per-key and signed-policy validation.
        drop(initial);
        drop(prepared_corpus);
        drop(arena);
        execution.release_scratch();

        progress.stage("Compute merges", trainer.vocab_size);
        return Ok(complete_model(trainer, vocabulary, Vec::new()));
    }
    let mut corpus = prepared_corpus.materialize::<S>(workers, policy, progress)?;
    let mut index =
        pair_index::PairIndex::from_initial_pairs(initial, policy, trainer.min_frequency)?;

    let mut merges = Vec::new();
    // PERF: Reuse bounded selection workspace across all rounds. Clearing
    // candidates releases their position lists before commit without reallocating
    // the vector; rule and conflict storage never exceeds the batch limit.
    let mut batch = RuleBatch::default();
    let work = progress.stage("Compute merges", trainer.vocab_size);
    while vocabulary.len() < trainer.vocab_size {
        match batch.select(
            trainer,
            &mut vocabulary,
            &mut corpus,
            &mut index,
            policy,
            #[cfg(test)]
            trace,
        )? {
            BatchSelection::Ready => {}
            BatchSelection::Finished => break,
            BatchSelection::RestartForReuse => return Ok(AttemptOutcome::RestartForReuse),
        }
        merges.extend(batch.rules.iter().map(|rule| rule.pair));
        let (prepared, prepared_births) = merge::prepare_merges_with_births(
            &corpus,
            &batch.rules,
            &batch.candidates,
            policy,
            vocabulary.len(),
            trainer.max_token_length.unwrap_or(usize::MAX),
            execution,
            &arena,
            trainer.min_frequency.max(1),
            merge_options,
        )?;

        // PERF: Preparation owns all writes and birth events. Selected
        // position lists have no remaining reader; release them before allocating
        // the next generation during commit.
        batch.candidates.clear();
        let events = prepared.apply(&mut corpus);

        index.commit_merges_with_prepared(
            &events,
            vocabulary.len(),
            execution,
            &arena,
            prepared_births,
        )?;

        drop(events);
        work.learned(merges.len());
    }
    // Training state does not participate in model output. Release position
    // owners before their arena, and free the corpus and scratch before
    // constructing the public vocabulary and merge strings.
    drop(batch);

    drop(index);
    drop(corpus);
    drop(arena);
    execution.release_scratch();

    Ok(complete_model(trainer, vocabulary, merges))
}
fn complete_model(
    trainer: &BpeTrainer,
    vocabulary: vocabulary::Vocabulary,
    merges: Vec<tk_encode::models::bpe::Pair>,
) -> AttemptOutcome {
    let (vocab, merges) = vocabulary.into_model_parts(merges);
    AttemptOutcome::Complete((vocab, merges, trainer.special_tokens.clone()))
}

#[cfg(test)]
mod tests;
