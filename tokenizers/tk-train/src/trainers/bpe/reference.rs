//! Small-input oracle with sequential word edits and historical birth cohorts.
//! It shares alphabet selection, but no engine index, codec, matching or batch code.
use super::*;
use dary_heap::OctonaryHeap;
use indexmap::IndexSet;
use std::cmp::{Ordering, Reverse};

struct Merge {
    pair: Pair,
    count: u64,
    words: AHashSet<usize>,
}

impl Ord for Merge {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.count, Reverse(self.pair)).cmp(&(other.count, Reverse(other.pair)))
    }
}

impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Merge {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Merge {}

// Literal left-to-right edits preserve intermediate removals and births. Net
// recounting would lose alias cohorts and the strict neighbor-length birth gate.
fn merge_word(word: &mut Vec<(u32, usize)>, pair: Pair, id: u32, limit: usize) -> Vec<(Pair, i64)> {
    let mut changes = Vec::new();
    let mut i = 0;
    while i + 1 < word.len() {
        if (word[i].0, word[i + 1].0) == pair {
            let length = word[i].1 + word[i + 1].1;
            if i > 0 {
                changes.push(((word[i - 1].0, pair.0), -1));
                if word[i - 1].1 + length < limit {
                    changes.push(((word[i - 1].0, id), 1));
                }
            }
            word[i] = (id, length);
            word.remove(i + 1);
            if i + 1 < word.len() {
                changes.push(((pair.1, word[i + 1].0), -1));
                if length + word[i + 1].1 < limit {
                    changes.push(((id, word[i + 1].0), 1));
                }
            }
        }
        i += 1;
    }
    changes
}

impl BpeTrainer {
    pub(super) fn do_train_observed(
        &self,
        counts: &AHashMap<CompactString, u64>,
        mut observe: impl FnMut(Pair, u64, u32),
    ) -> Result<(Vocab, Merges, Vec<AddedToken>)> {
        let mut tokens = IndexSet::new();
        for token in &self.special_tokens {
            tokens.insert(token.content.clone());
        }
        for character in self.select_alphabet(WordCountsView::from_map(counts)) {
            tokens.insert(character.to_string());
        }
        let mut weights = Vec::new();
        let mut words = Vec::new();
        for (text, &weight) in counts {
            let mut word = Vec::new();
            let length = text.chars().count();
            for (i, character) in text.chars().enumerate() {
                let mut token = character.to_string();
                if !tokens.contains(&token) {
                    continue;
                }
                if i != 0
                    && let Some(prefix) = &self.continuing_subword_prefix
                {
                    token.insert_str(0, prefix);
                }
                if i + 1 == length
                    && let Some(suffix) = &self.end_of_word_suffix
                {
                    token.push_str(suffix);
                }
                word.push((tokens.insert_full(token).0 as u32, 1));
            }
            words.push(word);
            weights.push(i64::try_from(weight).expect("oracle fixtures use small weights"));
        }
        let mut ledger = AHashMap::<Pair, i64>::new();
        let mut births = AHashMap::<Pair, AHashSet<usize>>::new();
        for (i, word) in words.iter().enumerate() {
            for edge in word.windows(2) {
                let pair = (edge[0].0, edge[1].0);
                *ledger.entry(pair).or_default() += weights[i];
                births.entry(pair).or_default().insert(i);
            }
        }
        let mut queue = OctonaryHeap::new();
        let mut merges = Vec::new();
        loop {
            for (pair, words) in births.drain() {
                if ledger[&pair] > 0 {
                    queue.push(Merge {
                        pair,
                        count: ledger[&pair] as u64,
                        words,
                    });
                }
            }
            if tokens.len() >= self.vocab_size {
                break;
            }
            let Some(mut top) = queue.pop() else { break };
            if top.count != ledger[&top.pair] as u64 {
                top.count = ledger[&top.pair] as u64;
                queue.push(top);
                continue;
            }
            if top.count < self.min_frequency.max(1) {
                break;
            }
            let mut right = tokens[top.pair.1 as usize].as_str();
            if let Some(prefix) = &self.continuing_subword_prefix {
                right = right.strip_prefix(prefix).unwrap_or(right);
            }
            let token = format!("{}{right}", tokens[top.pair.0 as usize]);
            let id = tokens.insert_full(token).0 as u32;
            merges.push((
                tokens[top.pair.0 as usize].clone(),
                tokens[top.pair.1 as usize].clone(),
            ));
            observe(top.pair, top.count, id);
            for i in top.words {
                for (pair, change) in merge_word(
                    &mut words[i],
                    top.pair,
                    id,
                    self.max_token_length.unwrap_or(usize::MAX),
                ) {
                    *ledger.entry(pair).or_default() += change * weights[i];
                    if change > 0 {
                        births.entry(pair).or_default().insert(i);
                    }
                }
            }
        }
        let vocab = tokens
            .into_iter()
            .enumerate()
            .map(|(id, token)| (token, id as u32))
            .collect();
        Ok((vocab, merges, self.special_tokens.clone()))
    }
}
