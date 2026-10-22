//! One insertion-ordered owner for token strings and their IDs.
use super::BpeTrainer;
use super::word_counts::WordCountsView;
use crate::progress::ProgressBar;
use ahash::{AHashMap, RandomState};
use compact_str::CompactString;
use fixedbitset::FixedBitSet;
use indexmap::IndexSet;
use tk_encode::Result;
use tk_encode::models::bpe::{Merges, Pair, Vocab};
use tk_encode::parallelism::MaybeParallelSlice;

pub(super) struct Vocabulary {
    tokens: IndexSet<CompactString, RandomState>,
    prefix: Option<String>,
    merges: Vec<Pair>,
}

impl Vocabulary {
    pub(super) fn initialize(
        trainer: &BpeTrainer,
        wc: WordCountsView<'_>,
        progress: &Option<ProgressBar>,
    ) -> Result<(Self, InitialTokenIds)> {
        if trainer.vocab_size > u32::MAX as usize {
            return Err("BPE target vocabulary size exceeds the usable u32 ID domain".into());
        }
        let mut vocabulary = Self::new(
            trainer.vocab_size,
            trainer.continuing_subword_prefix.clone(),
        )?;
        for token in &trainer.special_tokens {
            vocabulary.intern(token.content.as_str().into())?;
        }

        let alphabet = trainer.select_alphabet(wc);
        let mut spans = vec![0; vocabulary.len()];
        for &character in &alphabet.observed_specials {
            let id = vocabulary.intern(character.to_string().into())?;
            spans[id as usize] = 1;
        }
        for entry in &alphabet.characters {
            let id = vocabulary.intern(entry.character.to_string().into())?;
            spans.resize(vocabulary.len(), 0);
            spans[id as usize] = usize::from(entry.observed_in_input);
        }

        trainer.update_progress(progress, wc.len(), "Tokenize words");
        let ids = InitialTokenIds::build(
            &mut vocabulary,
            wc,
            spans,
            alphabet.complete,
            trainer.end_of_word_suffix.as_deref(),
            progress,
        )?;
        Ok((vocabulary, ids))
    }

    fn new(capacity: usize, prefix: Option<String>) -> Result<Self> {
        let mut tokens = IndexSet::with_hasher(RandomState::default());
        tokens
            .try_reserve(capacity)
            .map_err(|error| format!("BPE vocabulary reservation failed: {error}"))?;
        Ok(Self {
            tokens,
            prefix,
            merges: Vec::new(),
        })
    }

    fn intern(&mut self, token: CompactString) -> Result<u32> {
        self.insert(token).map(|(id, _)| id)
    }

    pub(super) fn len(&self) -> usize {
        self.tokens.len()
    }

    pub(super) fn resolve(&mut self, pair: Pair) -> Result<ResolvedMerge> {
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
        let (merged_id, already_in_vocabulary) = self.insert(text)?;
        Ok(ResolvedMerge {
            merged_id,
            already_in_vocabulary,
        })
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

/// Immutable plain and observed decorated IDs for the input used to build this table.
pub(super) struct InitialTokenIds {
    spans: Vec<usize>,
    complete_alphabet: bool,
    characters: Box<[u32]>,
    decorated: Box<[DecoratedIds]>,
    prefix: bool,
    suffix: bool,
}

impl InitialTokenIds {
    fn build(
        vocabulary: &mut Vocabulary,
        words: WordCountsView<'_>,
        spans: Vec<usize>,
        complete_alphabet: bool,
        suffix: Option<&str>,
        progress: &Option<ProgressBar>,
    ) -> Result<Self> {
        let mut characters = vec![u32::MAX; 0x110000].into_boxed_slice();
        for (id, token) in vocabulary.tokens.iter().enumerate() {
            let mut chars = token.chars();
            if let Some(character) = chars.next()
                && chars.next().is_none()
            {
                characters[character as usize] = id as u32;
            }
        }
        let mut ids = Self {
            spans,
            complete_alphabet,
            characters,
            prefix: vocabulary.prefix.as_deref().is_some_and(|p| !p.is_empty()),
            suffix: suffix.is_some_and(|s| !s.is_empty()),
            decorated: Box::default(),
        };
        if ids.is_plain() {
            if let Some(p) = progress {
                p.inc(words.len() as u64);
            }
            return Ok(ids);
        }
        ids.decorated = vec![DecoratedIds([u32::MAX; 3]); vocabulary.len()].into_boxed_slice();
        ids.spans.fill(0);
        let mut text = String::new();
        for (word, _) in words.iter() {
            for (byte, character) in word.char_indices() {
                let Some(plain) = ids.get_plain(character) else {
                    continue;
                };
                let affix = ids.affix(byte == 0, byte + character.len_utf8() == word.len());
                if affix == Affix::Plain {
                    ids.spans[plain as usize] = 1;
                    continue;
                }
                let cached = ids.decorated[plain as usize].get(affix);
                if cached != u32::MAX {
                    ids.spans[cached as usize] = 1;
                    continue;
                }
                text.clear();
                if matches!(affix, Affix::Prefix | Affix::Both) {
                    text.push_str(vocabulary.prefix.as_deref().unwrap());
                }
                text.push_str(vocabulary.tokens[plain as usize].as_str());
                if matches!(affix, Affix::Suffix | Affix::Both) {
                    text.push_str(suffix.unwrap());
                }
                let id = vocabulary.intern(text.as_str().into())?;
                ids.decorated[plain as usize].remember(affix, id);
                ids.spans.resize(vocabulary.len(), 0);
                ids.spans[id as usize] = 1;
            }
            if let Some(p) = progress {
                p.inc(1);
            }
        }
        Ok(ids)
    }

    fn get_plain(&self, character: char) -> Option<u32> {
        let id = self.characters[character as usize];
        (id != u32::MAX).then_some(id)
    }

    fn affix(&self, first: bool, last: bool) -> Affix {
        // Prefixes apply behind the original first character; suffixes apply
        // to the original last character. Both decorators can apply together.
        match (self.prefix && !first, self.suffix && last) {
            (false, false) => Affix::Plain,
            (true, false) => Affix::Prefix,
            (false, true) => Affix::Suffix,
            (true, true) => Affix::Both,
        }
    }
    fn is_plain(&self) -> bool {
        // The plain lookup path is valid only when neither decorator is active.
        !self.prefix && !self.suffix
    }
    /// Read only words from the initialization input; other words may need unbuilt affix states.
    #[inline]
    pub(super) fn symbols<'a>(&'a self, word: &'a str) -> impl Iterator<Item = u32> + 'a {
        if self.is_plain() {
            itertools::Either::Left(
                word.chars()
                    .filter_map(|character| self.get_plain(character)),
            )
        } else {
            itertools::Either::Right(word.char_indices().filter_map(move |(byte, character)| {
                // UTF-8 byte offset plus this character's width identifies the
                // original last character; filtering never moves the word boundary.
                self.get(
                    character,
                    byte == 0,
                    byte + character.len_utf8() == word.len(),
                )
            }))
        }
    }
    fn get(&self, character: char, first: bool, last: bool) -> Option<u32> {
        let plain = self.characters[character as usize];
        if plain == u32::MAX {
            return None;
        }
        let affix = self.affix(first, last);
        Some(if affix == Affix::Plain {
            plain
        } else {
            self.decorated[plain as usize].get(affix)
        })
    }
}

#[cfg(test)]
mod initial_tests {
    use super::super::word_counts::WordCounts;
    use super::*;

    #[test]
    fn filtering_keeps_original_affix_positions_and_id_order() {
        let words = WordCounts::from_unique(vec![("qé🙂z".into(), 1), ("🙂é".into(), 9)]);
        let view = WordCountsView::from(&words);
        let mut trainer = BpeTrainer::new(1, 32);
        trainer.limit_alphabet = Some(2);
        trainer.initial_alphabet = ['é', '🙂'].into();
        trainer.continuing_subword_prefix = Some("前".into());
        trainer.end_of_word_suffix = Some("尾".into());
        let (vocabulary, ids) = Vocabulary::initialize(&trainer, view, &None).unwrap();
        assert_eq!(ids.symbols("qé🙂z").collect::<Vec<_>>(), [2, 3]);
        assert_eq!(ids.symbols("🙂é").collect::<Vec<_>>(), [1, 4]);
        assert_eq!(
            vocabulary
                .tokens
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
            ["é", "🙂", "前é", "前🙂", "前é尾"]
        );
        assert_eq!(ids.get_plain('q'), None);
        assert_eq!(ids.get_plain('🙂'), Some(1));
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
enum Affix {
    Prefix = 0,
    Suffix = 1,
    Both = 2,
    Plain = 3,
}

#[derive(Clone, Copy)]
struct DecoratedIds([u32; 3]);

impl DecoratedIds {
    fn get(&self, affix: Affix) -> u32 {
        self.0[affix as usize]
    }
    fn remember(&mut self, affix: Affix, id: u32) {
        self.0[affix as usize] = id;
    }
}

pub(super) struct ResolvedMerge {
    pub(super) merged_id: u32,
    pub(super) already_in_vocabulary: bool,
}

impl BpeTrainer {
    pub(super) fn select_alphabet(&self, wc: WordCountsView<'_>) -> Alphabet {
        if self.limit_alphabet.is_none() {
            // Keep the unlimited fast path: parallel presence, without frequencies.
            let observed = observed_characters(wc);
            let mut present = observed.clone();
            for &character in &self.initial_alphabet {
                present.insert(character as usize);
            }
            let characters = present
                .ones()
                .map(|codepoint| {
                    let character = char::from_u32(codepoint as u32)
                        .expect("alphabet bits come from valid characters");
                    AlphabetCharacter {
                        character,
                        observed_in_input: observed.contains(codepoint),
                    }
                })
                .collect();
            return Alphabet {
                characters,
                complete: true,
                observed_specials: self.observed_special_characters(&observed),
            };
        }
        let mut observed = FixedBitSet::with_capacity(0x110000);
        // Compute the alphabet from seen words
        // Full-u64 weights can exceed u64 after repeated character occurrences.
        // Resident input byte counts bound the sum below u128::MAX, leaving that
        // value available as the initial alphabet's highest retention priority.
        let mut alphabet: AHashMap<char, u128> = AHashMap::new();
        for (word, count) in wc.iter() {
            for c in word.chars() {
                *alphabet.entry(c).or_default() += u128::from(*count);
                observed.insert(c as usize);
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
        let characters = kept
            .into_iter()
            .map(|(&character, _)| AlphabetCharacter {
                character,
                observed_in_input: observed.contains(character as usize),
            })
            .collect();
        // The 136 KiB bitmap is selection scratch, never retained across attempts.
        Alphabet {
            characters,
            complete: false,
            observed_specials: self.observed_special_characters(&observed),
        }
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Alphabet {
    characters: Vec<AlphabetCharacter>,
    complete: bool,
    observed_specials: Vec<char>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct AlphabetCharacter {
    pub(super) character: char,
    observed_in_input: bool,
}

impl BpeTrainer {
    fn observed_special_characters(&self, observed: &FixedBitSet) -> Vec<char> {
        self.special_tokens
            .iter()
            .filter_map(|token| {
                let mut chars = token.content.chars();
                let character = chars.next()?;
                // A special token admits a symbol only when it is exactly one
                // character and that character actually occurs in the input.
                (chars.next().is_none() && observed.contains(character as usize))
                    .then_some(character)
            })
            .collect()
    }
}

impl InitialTokenIds {
    pub(super) fn metadata(&self) -> Vec<usize> {
        self.spans.clone()
    }
    pub(super) fn symbol_count(&self, text: &str) -> usize {
        if self.complete_alphabet {
            text.chars().count()
        } else {
            text.chars()
                .filter(|&ch| self.characters[ch as usize] != u32::MAX)
                .count()
        }
    }
}

impl InitialTokenIds {
    pub(super) fn token_count(&self) -> usize {
        self.spans.len()
    }
}

impl Vocabulary {
    fn insert(&mut self, token: CompactString) -> Result<(u32, bool)> {
        let (index, inserted) = self.tokens.insert_full(token);
        let id = u32::try_from(index).map_err(|_| "BPE vocabulary exceeds u32")?;
        if id == u32::MAX {
            return Err("BPE token ID collides with the word separator".into());
        }
        Ok((id, !inserted))
    }
}

#[cfg(test)]
mod domain_tests {
    use super::*;
    #[test]
    fn oversized_target_returns_an_error() {
        if usize::BITS > 32 {
            let trainer = BpeTrainer::builder()
                .vocab_size(usize::MAX)
                .show_progress(false)
                .build();
            assert!(trainer.do_train(&AHashMap::new()).is_err());
        }
    }
}
