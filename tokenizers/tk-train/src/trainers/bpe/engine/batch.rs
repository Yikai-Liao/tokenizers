//! Reusable selection workspace and the rules that certify one safe merge batch.
use super::{IdentityPolicy, corpus, merge, pair_index, vocabulary};
use crate::trainers::bpe::BpeTrainer;
use ahash::AHashSet;
use tk_encode::Result;

pub(super) enum BatchSelection {
    Ready,
    Finished,
    RestartForReuse,
}
#[derive(Default)]
pub(super) struct RuleBatch<'a> {
    pub(super) rules: Vec<merge::MergeRule>,
    pub(super) candidates: Vec<pair_index::MergeCandidate<'a>>,
    heads: AHashSet<u32>,
    tails: AHashSet<u32>,
}
impl<'a> RuleBatch<'a> {
    #[cfg_attr(test, allow(clippy::too_many_arguments))]
    pub(super) fn select<S: corpus::SlotStorage>(
        &mut self,
        trainer: &BpeTrainer,
        vocabulary: &mut vocabulary::Vocabulary,
        corpus: &mut corpus::Corpus<S>,
        index: &mut pair_index::PairIndex<'a>,
        policy: IdentityPolicy,
        #[cfg(test)] trace: &mut Vec<(tk_encode::models::bpe::Pair, u64, u32)>,
    ) -> Result<BatchSelection> {
        let cap = if policy == IdentityPolicy::FirstActivationOnly {
            256.min(trainer.vocab_size - vocabulary.len())
        } else {
            1
        };
        debug_assert!(self.candidates.is_empty());
        self.rules.clear();
        self.heads.clear();
        self.tails.clear();
        index.begin_selection();
        while self.rules.len() < cap {
            let Some(priority) = index.best() else {
                break;
            };
            let pair = pair_index::key_pair(priority.key);
            if !self.rules.is_empty()
                && (pair.0 == pair.1
                    || self.tails.contains(&pair.0)
                    || self.heads.contains(&pair.1))
            {
                break;
            }
            let token = vocabulary.merge_token(pair);
            if policy == IdentityPolicy::FirstActivationOnly && vocabulary.reuses_active_id(&token)
            {
                // Fresh pruning and fused batches omit intermediate cohorts.
                // Switching this index in place would lose observable births.
                // Input words remain unchanged: rebuild all cohorts instead,
                // before consuming this candidate or writing its batch.

                return Ok(BatchSelection::RestartForReuse);
            }
            let reserved = token.existing_id.is_some();
            // A reserved ID can precede the old witness in a birth tie.
            if reserved && !self.rules.is_empty() {
                break;
            }
            let candidate = index.take_best();
            let identity = vocabulary.resolve_merge(token)?;
            corpus.prepare_spans(pair, identity.id, identity.reused_active_id);
            self.rules.push(merge::MergeRule {
                pair,
                replacement: identity.id,
            });
            self.candidates.push(candidate);
            #[cfg(test)]
            trace.push((pair, priority.priority_count, identity.id));
            if reserved || pair.0 == pair.1 || self.rules.len() == cap {
                break;
            }
            // Only a following rule needs these conflict checks. A
            // single-rule round never allocates the two hash tables.
            self.heads.insert(pair.0);
            self.tails.insert(pair.1);
        }
        index.end_selection();
        Ok(if self.rules.is_empty() {
            BatchSelection::Finished
        } else {
            BatchSelection::Ready
        })
    }
}
