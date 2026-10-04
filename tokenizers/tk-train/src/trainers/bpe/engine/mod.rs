//! One BPE coordinator over shared vocabulary, corpus, and occurrence storage.
mod aa_parity;
mod corpus;
mod execution;
mod initial_pairs;
mod merge;
mod pair_index;
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
impl IdentityPolicy {
    fn for_trainer(trainer: &BpeTrainer) -> Self {
        // Plain concatenation preserves irreversible segmentation. Affix removal
        // changes that proof: retain the signed ledger from initialization.
        if trainer
            .continuing_subword_prefix
            .as_deref()
            .is_none_or(str::is_empty)
            && trainer
                .end_of_word_suffix
                .as_deref()
                .is_none_or(str::is_empty)
        {
            Self::Fresh
        } else {
            Self::Reusable
        }
    }
}
pub(super) fn train(
    trainer: &BpeTrainer,
    word_counts: &AHashMap<CompactString, u64>,
    workers: usize,
    #[cfg(test)] mut observe: Option<
        &mut (dyn FnMut(tk_encode::models::bpe::Pair, u64, u32) + Send),
    >,
) -> Result<(Vocab, Merges, Vec<AddedToken>)> {
    let execution = execution::Execution::new(workers)?;
    execution.pool.install(|| {
        let progress = TrainingProgress::new(trainer.show_progress, trainer.progress_format)?;
        let policy = IdentityPolicy::for_trainer(trainer);
        let mut vocabulary =
            vocabulary::Vocabulary::initialize(trainer, word_counts, workers, &progress)?;
        let mut corpus = corpus::Corpus::build(
            word_counts,
            &mut vocabulary,
            workers,
            policy,
            trainer.max_token_length.is_some(),
            &progress,
        )?;
        let arena = AllocationArena::new(workers, corpus.initial_edges());
        let initial = initial_pairs::build_initial_pairs(
            corpus.initial_view(),
            if policy == IdentityPolicy::Fresh {
                trainer.min_frequency.max(1)
            } else {
                0
            },
            &execution,
            &arena,
            &progress,
        )?;
        let mut index =
            pair_index::PairIndex::from_initial_pairs(initial, policy, trainer.min_frequency)?;
        let mut merges = Vec::new();
        // PERF: Reuse bounded selection workspace across all rounds. Clearing
        // candidates releases their postings before commit without reallocating
        // the vector; rule and conflict storage never exceeds the batch limit.
        let mut rules = Vec::new();
        let mut candidates = Vec::new();
        let mut heads = AHashSet::new();
        let mut tails = AHashSet::new();
        let work = progress.merges(trainer.vocab_size, vocabulary.len());
        while vocabulary.len() < trainer.vocab_size {
            let cap = if policy == IdentityPolicy::Fresh {
                256.min(trainer.vocab_size - vocabulary.len())
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
                if let Some(observer) = observe.as_mut() {
                    observer(pair, priority.priority_count, identity.id);
                }
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
            let prepared = merge::prepare_merges(
                &corpus,
                &rules,
                &candidates,
                policy,
                vocabulary.len(),
                trainer.max_token_length.unwrap_or(usize::MAX),
                &execution,
            )?;
            // PERF: Preparation owns all writes and birth events. Selected
            // postings have no remaining reader; release them before allocating
            // the next generation during commit.
            candidates.clear();
            let events = prepared.apply(&mut corpus);
            index.commit_merges(&events, vocabulary.len(), &execution, &arena)?;
            drop(events);
            work.learned(merges.len(), vocabulary.len());
        }
        // Training state does not participate in model output. Release position
        // owners before their arena, and free the corpus and scratch before
        // constructing the public vocabulary and merge strings.
        drop((rules, candidates, heads, tails));
        drop(index);
        drop(corpus);
        drop(arena);
        execution.release_scratch();
        let (vocab, merges) = vocabulary.into_model_parts(merges);
        Ok((vocab, merges, trainer.special_tokens.clone()))
    })
}

#[cfg(test)]
mod tests;
