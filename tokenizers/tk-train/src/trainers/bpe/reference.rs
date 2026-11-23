//! Test-only copy of the previous word-based merge loop.

use super::{BpeTrainer, Pair};
use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use dary_heap::OctonaryHeap;
use std::cmp::Ordering;
use tk_encode::Result;
use tk_encode::models::bpe::{Merges, Vocab};
use tk_encode::vocab::bucket_added_vocabulary::AddedToken;

#[derive(Eq)]
struct Merge {
    pair: Pair,
    count: u64,
    words: AHashSet<usize>,
}

impl PartialEq for Merge {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.pair == other.pair
    }
}

impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Merge {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count
            .cmp(&other.count)
            .then_with(|| other.pair.cmp(&self.pair))
    }
}

pub(super) fn train(
    trainer: &BpeTrainer,
    word_counts: &AHashMap<CompactString, u64>,
) -> Result<(Vocab, Merges, Vec<AddedToken>)> {
    let mut word_to_id = AHashMap::with_capacity(trainer.vocab_size);
    let mut id_to_word = Vec::with_capacity(trainer.vocab_size);
    trainer.add_special_tokens(&mut word_to_id, &mut id_to_word)?;
    trainer.compute_alphabet(word_counts, &mut word_to_id, &mut id_to_word)?;

    let mut words = trainer.tokenize_words(word_counts, &mut word_to_id, &mut id_to_word, &None)?;
    let mut pair_counts: AHashMap<Pair, i32> = AHashMap::new();
    let mut where_to_update: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();
    for (index, (word, weight)) in words.iter().enumerate() {
        for window in word.get_chars().windows(2) {
            let pair = (window[0], window[1]);
            *pair_counts.entry(pair).or_default() += *weight as i32;
            where_to_update.entry(pair).or_default().insert(index);
        }
    }

    let mut queue = OctonaryHeap::with_capacity(pair_counts.len());
    for (pair, words) in where_to_update.drain() {
        let count = pair_counts[&pair];
        if count > 0 {
            queue.push(Merge {
                pair,
                count: count as u64,
                words,
            });
        }
    }

    let max_token_length = trainer.max_token_length.unwrap_or(usize::MAX);
    let mut merges = Vec::new();
    while word_to_id.len() < trainer.vocab_size {
        let Some(mut top) = queue.pop() else { break };
        if top.count != pair_counts[&top.pair] as u64 {
            top.count = pair_counts[&top.pair] as u64;
            queue.push(top);
            continue;
        }
        if top.count < 1 || trainer.min_frequency > top.count {
            break;
        }

        let left = &id_to_word[top.pair.0 as usize];
        let mut right = id_to_word[top.pair.1 as usize].as_str();
        if let Some(prefix) = &trainer.continuing_subword_prefix
            && let Some(rest) = right.strip_prefix(prefix)
        {
            right = rest;
        }
        let token = format!("{left}{right}");
        let id = trainer.intern_token(&token, &mut word_to_id, &mut id_to_word)?;
        merges.push((top.pair, id));

        let mut changes = Vec::new();
        for word_index in top.words {
            let (word, weight) = &mut words[word_index];
            changes.extend(
                word.merge(top.pair.0, top.pair.1, id, max_token_length)
                    .into_iter()
                    .map(|change| (change, word_index, *weight)),
            );
        }
        for ((pair, change), word_index, weight) in changes {
            let change = change * weight as i32;
            *pair_counts.entry(pair).or_default() += change;
            if change > 0 {
                where_to_update.entry(pair).or_default().insert(word_index);
            }
        }
        for (pair, words) in where_to_update.drain() {
            let count = pair_counts[&pair];
            if count > 0 {
                queue.push(Merge {
                    pair,
                    count: count as u64,
                    words,
                });
            }
        }
    }

    let vocab = word_to_id
        .into_iter()
        .map(|(_, id)| (id_to_word[id as usize].to_string(), id))
        .collect();
    let merges = merges
        .into_iter()
        .map(|(pair, _)| {
            (
                id_to_word[pair.0 as usize].to_string(),
                id_to_word[pair.1 as usize].to_string(),
            )
        })
        .collect();
    Ok((vocab, merges, trainer.special_tokens.clone()))
}
