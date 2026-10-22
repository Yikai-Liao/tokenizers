//! One insertion-ordered owner for token strings and their IDs.
use super::BpeTrainer;
use super::word_counts::WordCountsView;
use ahash::{AHashMap, RandomState};
use compact_str::CompactString;
use fixedbitset::FixedBitSet;
use indexmap::IndexSet;
use tk_encode::models::bpe::{Merges, Pair, Vocab};
use tk_encode::parallelism::MaybeParallelSlice;

pub(super) struct Vocabulary {
    tokens: IndexSet<CompactString, RandomState>,
    prefix: Option<String>,
    merges: Vec<Pair>,
}

impl Vocabulary {
    pub(super) fn initialize(trainer: &BpeTrainer, wc: WordCountsView<'_>) -> Self {
        let mut vocabulary = Self::new(
            trainer.vocab_size,
            trainer.continuing_subword_prefix.clone(),
        );
        for token in &trainer.special_tokens {
            vocabulary.intern(token.content.as_str().into());
        }

        for character in trainer.select_alphabet(wc) {
            vocabulary.intern(character.to_string().into());
        }

        vocabulary
    }

    fn new(capacity: usize, prefix: Option<String>) -> Self {
        Self {
            tokens: IndexSet::with_capacity_and_hasher(capacity, RandomState::default()),
            prefix,
            merges: Vec::new(),
        }
    }

    pub(super) fn intern(&mut self, token: CompactString) -> u32 {
        self.tokens.insert_full(token).0 as u32
    }

    pub(super) fn contains(&self, token: &str) -> bool {
        self.tokens.contains(token)
    }

    pub(super) fn len(&self) -> usize {
        self.tokens.len()
    }

    pub(super) fn resolve(&mut self, pair: Pair) -> u32 {
        let left = self.tokens[pair.0 as usize].as_str();
        let right = self.tokens[pair.1 as usize].as_str();
        let right = self
            .prefix
            .as_deref()
            .and_then(|p| right.strip_prefix(p))
            .unwrap_or(right);
        let mut text = CompactString::with_capacity(left.len() + right.len());
        text.push_str(left);
        text.push_str(right);
        self.intern(text)
    }

    pub(super) fn commit(&mut self, pair: Pair) {
        self.merges.push(pair);
    }

    pub(super) fn into_parts(self) -> (Vocab, Merges) {
        let Self { tokens, merges, .. } = self;
        let merges = merges
            .into_iter()
            .map(|(left, right)| {
                (
                    tokens[left as usize].to_string(),
                    tokens[right as usize].to_string(),
                )
            })
            .collect();
        (
            tokens
                .into_iter()
                .enumerate()
                .map(|(id, token)| (token.to_string(), id as u32))
                .collect(),
            merges,
        )
    }
}

impl BpeTrainer {
    pub(super) fn select_alphabet(&self, wc: WordCountsView<'_>) -> Vec<char> {
        if self.limit_alphabet.is_none() {
            // Keep the unlimited fast path: parallel presence, without frequencies.
            let observed = observed_characters(wc);
            let mut present = observed.clone();
            for &character in &self.initial_alphabet {
                present.insert(character as usize);
            }
            return present
                .ones()
                .map(|codepoint| char::from_u32(codepoint as u32).expect("observed scalar"))
                .collect();
        }

        // Compute the alphabet from seen words
        // Full-u64 weights can exceed u64 after repeated character occurrences.
        // Resident input byte counts bound the sum below u128::MAX, leaving that
        // value available as the initial alphabet's highest retention priority.
        let mut alphabet: AHashMap<char, u128> = AHashMap::new();
        for (word, count) in wc.iter() {
            for c in word.chars() {
                *alphabet.entry(c).or_default() += u128::from(*count);
            }
        }

        // Also include anything from the provided initial alphabet
        for c in &self.initial_alphabet {
            *alphabet.entry(*c).or_default() = u128::MAX;
        }

        let mut kept = alphabet.iter().collect::<Vec<_>>();

        // Compute the number of chars to remove from the alphabet
        // If `limit_alphabet < initial_alphabet.len()`, some of these initial characters
        // will be removed
        let to_remove = self
            .limit_alphabet
            .map(|limit| alphabet.len().saturating_sub(limit))
            .unwrap_or(0);

        // Remove the unwanted chars
        if to_remove > 0 {
            kept.sort_unstable_by_key(|k| *k.1);
            kept.drain(..to_remove);
        }

        // Keep the initial alphabet (sorted for determinism)
        kept.sort_unstable_by_key(|k| *k.0 as u32);
        kept.into_iter().map(|(&character, _)| character).collect()
    }
}

fn observed_characters(word_counts: WordCountsView<'_>) -> FixedBitSet {
    let words: Vec<_> = word_counts.iter().map(|(word, _)| word).collect();
    let workers = super::num_workers();
    // Round up to cover the remainder, and keep the chunk width valid for
    // empty input; each worker scans whole words into its private presence bitmap.
    let chunk = words.len().div_ceil(workers).max(1);
    words
        .maybe_par_chunks(chunk)
        .map(|chunk| {
            let mut present = FixedBitSet::with_capacity(0x110000);
            for word in chunk {
                for character in word.chars() {
                    present.insert(character as usize);
                }
            }
            present
        })
        .reduce_with(|mut left, right| {
            left.union_with(&right);
            left
        })
        .unwrap_or_else(|| FixedBitSet::with_capacity(0x110000))
}
