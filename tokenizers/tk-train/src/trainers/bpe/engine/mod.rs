//! One BPE coordinator over shared vocabulary, corpus, and occurrence storage.
mod aa_parity;
mod corpus;
mod execution;
mod initial_pairs;
mod merge;
mod pair_index;
mod single_producer_diagnostics;
mod vocabulary;
use super::BpeTrainer;
use crate::progress::TrainingProgress;
use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use tk_collections::AllocationArena;
use tk_encode::{
    Result,
    models::bpe::{Merges, Vocab},
    vocab::bucket_added_vocabulary::AddedToken,
};
const WORD_SEPARATOR_ID: u32 = u32::MAX;
#[derive(Clone, Copy, PartialEq, Eq)]
enum IdentityPolicy {
    Fresh,
    Reusable,
}
type ModelParts = (Vocab, Merges, Vec<AddedToken>);
enum AttemptOutcome {
    Complete(ModelParts),
    RestartForReuse,
}
pub(super) fn train(
    trainer: &BpeTrainer,
    word_counts: &AHashMap<CompactString, u64>,
    workers: usize,
    #[cfg(test)] observe: Option<&mut (dyn FnMut(tk_encode::models::bpe::Pair, u64, u32) + Send)>,
) -> Result<ModelParts> {
    train_with_merge_options(
        trainer,
        word_counts,
        workers,
        merge::MergeOptions::from_env(),
        #[cfg(test)]
        observe,
    )
}
fn train_with_merge_options(
    trainer: &BpeTrainer,
    word_counts: &AHashMap<CompactString, u64>,
    workers: usize,
    merge_options: merge::MergeOptions,
    #[cfg(test)] mut observe: Option<
        &mut (dyn FnMut(tk_encode::models::bpe::Pair, u64, u32) + Send),
    >,
) -> Result<ModelParts> {
    let execution = execution::Execution::with_owners(
        workers,
        if merge_options.logical_owners == 0 {
            workers
        } else {
            merge_options.logical_owners
        },
        merge_options.fast_shard_router,
    )?;
    execution.pool.install(|| {
        let progress = TrainingProgress::new(trainer.show_progress, trainer.progress_format)?;
        // Every attempt begins with first activations. A nonempty affix does
        // not by itself require one-rule cohort execution. Stop before accepting
        // the first reused active ID, then rebuild with the same coordinator.
        let mut policy = IdentityPolicy::Fresh;
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
                    // A restarted attempt has no externally visible merge trace.
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
                    policy = IdentityPolicy::Reusable;
                }
            }
        }
    })
}
fn train_attempt(
    trainer: &BpeTrainer,
    word_counts: &AHashMap<CompactString, u64>,
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
    let prepared_corpus = corpus::PreparedCorpus::build(
        word_counts,
        &mut vocabulary,
        policy,
        trainer.max_token_length.is_some(),
        progress,
    )?;
    if vocabulary.len() >= trainer.vocab_size && prepared_corpus.initial_counts_fit_u64() {
        drop(prepared_corpus);
        execution.release_scratch();
        return Ok(complete_model(trainer, vocabulary, Vec::new()));
    }
    match corpus::slot_bits(trainer.vocab_size.max(vocabulary.len())) {
        16 => train_with_slots::<corpus::HalfSlots>(
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
        24 => train_with_slots::<corpus::ThreeByteSlots>(
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
        _ => train_with_slots::<corpus::FullSlots>(
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
fn train_with_slots<S: corpus::SlotStorage>(
    trainer: &BpeTrainer,
    mut vocabulary: vocabulary::Vocabulary,
    prepared_corpus: corpus::PreparedCorpus<'_>,
    policy: IdentityPolicy,
    execution: &execution::Execution,
    merge_options: merge::MergeOptions,
    progress: &TrainingProgress,
    #[cfg(test)] trace: &mut Vec<(tk_encode::models::bpe::Pair, u64, u32)>,
) -> Result<AttemptOutcome> {
    let workers = execution.workers();
    let diagnostics =
        single_producer_diagnostics::AttemptDiagnostics::new(merge_options.diagnostics, workers);
    let arena = AllocationArena::new(workers, prepared_corpus.initial_edges());
    let initial = initial_pairs::build_initial_pairs(
        &prepared_corpus,
        if policy == IdentityPolicy::Fresh {
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
        diagnostics.finish(true);
        return Ok(complete_model(trainer, vocabulary, Vec::new()));
    }
    let mut corpus = prepared_corpus.materialize::<S>(workers, policy, progress)?;
    let mut index =
        pair_index::PairIndex::from_initial_pairs(initial, policy, trainer.min_frequency)?;
    index.configure_commit(
        merge_options.group_births_in_commit,
        merge_options.direct_cold_encoding,
        merge_options.fast_shard_router,
    );
    index.configure_removals(
        merge_options.removal_entry,
        merge_options.removal_reduce,
        merge_options.removal_statistics,
        merge_options.removal_selective,
    );
    let mut merges = Vec::new();
    // PERF: Reuse bounded selection workspace across all rounds. Clearing
    // candidates releases their position lists before commit without reallocating
    // the vector; rule and conflict storage never exceeds the batch limit.
    let mut rules = Vec::new();
    let mut candidates = Vec::new();
    let mut heads = AHashSet::new();
    let mut tails = AHashSet::new();
    let work = progress.merges(trainer.vocab_size, vocabulary.len());
    while vocabulary.len() < trainer.vocab_size {
        let cap = if policy == IdentityPolicy::Fresh {
            let limit = if merge_options.batch_limit == 0 {
                256
            } else {
                merge_options.batch_limit
            };
            limit.min(trainer.vocab_size - vocabulary.len())
        } else {
            1
        };
        rules.clear();
        heads.clear();
        tails.clear();
        index.begin_selection();
        while rules.len() < cap {
            let Some(priority) = index.best() else {
                break;
            };
            let pair = pair_index::key_pair(priority.key);
            if !rules.is_empty()
                && (pair.0 == pair.1 || tails.contains(&pair.0) || heads.contains(&pair.1))
            {
                break;
            }
            let token = vocabulary.merge_token(pair);
            if policy == IdentityPolicy::Fresh && vocabulary.reuses_active_id(&token) {
                // Fresh pruning and fused batches omit intermediate cohorts.
                // Switching this index in place would lose observable births.
                // Input words remain unchanged: rebuild all cohorts instead,
                // before consuming this candidate or writing its batch.
                diagnostics.finish(false);
                return Ok(AttemptOutcome::RestartForReuse);
            }
            let reserved = token.existing_id.is_some();
            // A reserved ID can precede the old witness in a birth tie.
            if reserved && !rules.is_empty() {
                break;
            }
            let candidate = index.take_best();
            let identity = vocabulary.resolve_merge(token)?;
            corpus.prepare_spans(pair, identity.id, identity.reused_active_id);
            rules.push(merge::MergeRule {
                pair,
                replacement: identity.id,
            });
            candidates.push(candidate);
            #[cfg(test)]
            trace.push((pair, priority.priority_count, identity.id));
            merges.push(pair);
            if reserved || pair.0 == pair.1 || rules.len() == cap {
                break;
            }
            // Only a following rule needs these conflict checks. A
            // single-rule round never allocates the two hash tables.
            heads.insert(pair.0);
            tails.insert(pair.1);
        }
        index.end_selection();
        if rules.is_empty() {
            break;
        }
        let inputs = if merge_options.diagnostics {
            candidates
                .iter()
                .map(|candidate| candidate.positions.len())
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let diagnostic_round = diagnostics.begin_round(&inputs);
        let diagnostic = merge_options.diagnostics.then_some(&diagnostic_round);
        let prep_phase = diagnostic
            .map(|round| round.phase("prepare_mixed_serial_layout_parallel_jobs_serial_fanin"));
        let (mut prepared, prepared_births) = merge::prepare_merges_with_births(
            &corpus,
            &rules,
            &candidates,
            policy,
            vocabulary.len(),
            trainer.max_token_length.unwrap_or(usize::MAX),
            execution,
            &arena,
            trainer.min_frequency.max(1),
            merge_options,
            diagnostic,
        )?;
        drop(prep_phase);
        let job_diagnostics = std::mem::take(&mut prepared.diagnostics);
        // PERF: Preparation owns all writes and birth events. Selected
        // position lists have no remaining reader; release them before allocating
        // the next generation during commit.
        candidates.clear();
        let events = prepared.apply(&mut corpus);
        let commit_phase =
            diagnostic.map(|round| round.phase("commit_mixed_serial_route_parallel_owner"));
        index.commit_merges_with_prepared(
            &events,
            vocabulary.len(),
            execution,
            &arena,
            prepared_births,
            diagnostic,
        )?;
        drop(commit_phase);
        diagnostic_round.finish(&job_diagnostics);
        drop(events);
        work.learned(merges.len(), vocabulary.len());
    }
    // Training state does not participate in model output. Release position
    // owners before their arena, and free the corpus and scratch before
    // constructing the public vocabulary and merge strings.
    drop((rules, candidates, heads, tails));
    index.report_removal_statistics();
    drop(index);
    drop(corpus);
    drop(arena);
    execution.release_scratch();
    diagnostics.finish(true);
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
