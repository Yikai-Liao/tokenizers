//! Select a compatible priority prefix, then commit corpus, index and history.
use super::{
    BpeTrainer, TrainingMode,
    corpus::{Corpus, SelectedMerge},
    index::PairIndex,
    vocabulary::Vocabulary,
};
use ahash::AHashSet;
use tk_encode::Result;

/// Only successful index publication commits the current round's history.
#[derive(Debug)]
pub(super) enum MergeOutcome {
    Finished,
    RestartForReuse,
    Committed { rule_count: usize },
}

impl BpeTrainer {
    /// Selection can reserve identities and consume candidates. Errors or restart
    /// discard the attempt; successful corpus writes precede index publication.
    pub(super) fn merge_next(
        &self,
        vocabulary: &mut Vocabulary,
        corpus: &mut Corpus,
        index: &mut PairIndex,
    ) -> Result<MergeOutcome> {
        let reuse = index.mode() == TrainingMode::Reuse;
        let mut selected = Vec::new();
        let cap = if reuse {
            1
        } else {
            256.min(self.vocab_size - vocabulary.len())
        };
        let mut heads = AHashSet::new();
        let mut tails = AHashSet::new();
        #[cfg(test)]
        let mut pending_trace: super::Trace = Vec::new();

        // Keep the priority prefix: crossed endpoints stop selection, while shared
        // heads/tails may batch. DESIGN.md records the newborn priority argument.
        while selected.len() < cap {
            let Some(priority) = index.peek() else {
                break;
            };
            let pair = priority.pair();
            if !selected.is_empty()
                && (pair.0 == pair.1 || tails.contains(&pair.0) || heads.contains(&pair.1))
            {
                break;
            }
            let resolved = vocabulary.resolve(pair)?;
            if !reuse
                && resolved.already_in_vocabulary
                && corpus.has_been_activated(resolved.merged_id)
            {
                return Ok(MergeOutcome::RestartForReuse);
            }
            // An existing identity may precede old IDs; run it alone without the
            // compatible-prefix argument about newly appended IDs.
            if resolved.already_in_vocabulary && !selected.is_empty() {
                break;
            }
            let candidate = index.take(priority);
            corpus.activate(pair, resolved.merged_id);
            #[cfg(test)]
            pending_trace.push((pair, candidate.priority.count(), resolved.merged_id));
            selected.push(SelectedMerge {
                pair,
                merged_id: resolved.merged_id,
                positions: candidate.positions,
            });
            if resolved.already_in_vocabulary || pair.0 == pair.1 {
                break;
            }
            heads.insert(pair.0);
            tails.insert(pair.1);
        }
        if selected.is_empty() {
            return Ok(MergeOutcome::Finished);
        }
        drop(heads);
        drop(tails);

        let merges: Vec<_> = selected.iter().map(|rule| rule.pair).collect();
        let rule_count = merges.len();
        let changes = corpus.merge(selected)?;
        index.commit(changes)?;
        vocabulary.commit(merges);
        #[cfg(test)]
        vocabulary.trace.extend(pending_trace);
        Ok(MergeOutcome::Committed { rule_count })
    }
}
