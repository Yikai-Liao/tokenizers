//! Vocabulary identity and canonical output strings, independent of position storage.
use super::word_counts::WordCountsView;
use super::{BpeTrainer, WORD_SEPARATOR_ID};
use ahash::{AHashMap, RandomState};
use compact_str::CompactString;
use fixedbitset::FixedBitSet;
use indexmap::IndexSet;
use rayon::prelude::*;
use tk_encode::{
    Result,
    models::bpe::{Merges, Pair, Vocab},
    utils::progress::ProgressBar,
};

/// Append-only token identities and canonical text for one training attempt.
/// Reserves special/alphabet IDs, resolves decorated merges, and publishes output;
/// initial activation spans move to the corpus plan once tokenization is resolved.
pub(super) struct Vocabulary {
    // Append-only insertion indices are token IDs. Store each string once while
    // supporting both text lookup and direct lookup by ID with the same hasher.
    tokens: IndexSet<CompactString, RandomState>,
    prefix: Option<String>,
}

/// Immutable character/decorated-ID tables shared by counting and materialization.
/// Both stages scan the same filtered alphabet and affix interpretation, preserving
/// identical token sequences without interning strings during parallel scans.
pub(super) struct InitialTokenIds {
    spans: Vec<usize>,
    characters: Box<[u32]>,
    decorated: Box<[[u32; 3]]>,
    prefix: bool,
    suffix: bool,
    complete_alphabet: bool,
}

/// Selected once per training call; retries borrow the same cutoff decisions.
/// Presence is retained per selected character so the selection bitmap can retire
/// before token lookup tables and corpus storage are allocated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Alphabet {
    pub(super) characters: Vec<(char, bool)>,
    complete: bool,
    observed_specials: Vec<char>,
}

impl Vocabulary {
    pub(super) fn initialize(
        trainer: &BpeTrainer,
        word_counts: WordCountsView<'_>,
        alphabet: &Alphabet,
        progress: &Option<ProgressBar>,
    ) -> Result<(Self, InitialTokenIds)> {
        let mut vocabulary = Self {
            tokens: IndexSet::with_capacity_and_hasher(trainer.vocab_size, RandomState::default()),
            prefix: trainer.continuing_subword_prefix.clone(),
        };
        let mut spans = Vec::new();
        // 1. Reserve special IDs before inserting the codepoint-ordered alphabet.
        for token in &trainer.special_tokens {
            vocabulary.intern(token.content.as_str(), &mut spans)?;
        }

        // A single-character special token remains usable even outside the
        // selected alphabet. Multi-character special tokens are never symbols.
        for &character in &alphabet.observed_specials {
            let mut utf8 = [0; 4];
            let id = vocabulary.intern(character.encode_utf8(&mut utf8), &mut spans)?;
            spans[id as usize] = 1;
        }

        // 2. Reserve selected characters and activate those observed in the input.
        // Special tokens sharing a character identity use the same presence bit.
        for &(character, observed) in &alphabet.characters {
            let mut utf8 = [0; 4];
            let id = vocabulary.intern(character.encode_utf8(&mut utf8), &mut spans)?;
            spans[id as usize] = usize::from(observed);
        }
        // Resolve decorations before CorpusPlan sorts input by weight. Keep the
        // plain-alphabet shortcut and the original progress boundary.
        trainer.update_progress(progress, word_counts.len(), "Tokenize words");
        let ids = vocabulary.initial_ids(
            word_counts,
            spans,
            alphabet.complete,
            trainer.end_of_word_suffix.as_deref(),
            progress,
        )?;
        Ok((vocabulary, ids))
    }

    fn intern(&mut self, text: &str, spans: &mut Vec<usize>) -> Result<u32> {
        if let Some(id) = self.tokens.get_index_of(text) {
            return Ok(id as u32);
        }
        let (id, _) = self.insert_token(CompactString::from(text))?;
        spans.push(0);
        Ok(id)
    }

    // ID errors discard the whole attempt, including any insertion made here.
    fn insert_token(&mut self, token: CompactString) -> Result<(u32, bool)> {
        let (index, inserted) = self.tokens.insert_full(token);
        let id = u32::try_from(index).map_err(|_| "BPE vocabulary exceeds u32")?;
        if id == WORD_SEPARATOR_ID {
            return Err("BPE token ID collides with the word separator".into());
        }
        Ok((id, !inserted))
    }

    fn initial_ids(
        &mut self,
        word_counts: WordCountsView<'_>,
        spans: Vec<usize>,
        complete_alphabet: bool,
        suffix: Option<&str>,
        progress: &Option<ProgressBar>,
    ) -> Result<InitialTokenIds> {
        // 1. Map existing character IDs before any decorated identities are inserted.
        let mut ids = InitialTokenIds {
            spans,
            characters: vec![WORD_SEPARATOR_ID; 0x110000].into_boxed_slice(),
            decorated: Box::default(),
            prefix: self.prefix.as_deref().is_some_and(|p| !p.is_empty()),
            suffix: suffix.is_some_and(|s| !s.is_empty()),
            complete_alphabet,
        };
        for (id, token) in self.tokens.iter().enumerate() {
            let mut chars = token.chars();
            if let Some(character) = chars.next()
                && chars.next().is_none()
            {
                ids.characters[character as usize] = id as u32;
            }
        }
        if ids.prefix || ids.suffix {
            let mut decorated = Vec::new();
            decorated.resize(self.tokens.len(), [WORD_SEPARATOR_ID; 3]);
            ids.decorated = decorated.into_boxed_slice();
        }
        if ids.plain() {
            if let Some(p) = progress {
                p.inc(word_counts.len() as u64);
            }
            return Ok(ids);
        }

        // 2. Activate retained symbols in input traversal order. Sorting words first
        // would change the IDs allocated for previously unseen decorated strings.
        ids.spans.fill(0);
        let mut decorated = String::new();
        for (index, word) in word_counts.keys().enumerate() {
            self.activate_word(word, &mut ids, suffix, &mut decorated)?;
            if (index + 1) % 1024 == 0
                && let Some(p) = progress
            {
                p.inc(1024);
            }
        }
        if let Some(p) = progress {
            p.inc((word_counts.len() % 1024) as u64);
        }
        Ok(ids)
    }

    fn activate_word(
        &mut self,
        word: &str,
        ids: &mut InitialTokenIds,
        suffix: Option<&str>,
        decorated: &mut String,
    ) -> Result<()> {
        for (byte, character) in word.char_indices() {
            let plain_id = ids.characters[character as usize];
            if plain_id == WORD_SEPARATOR_ID {
                continue;
            }
            let add_prefix = ids.prefix && byte != 0;
            let add_suffix = ids.suffix && byte + character.len_utf8() == word.len();
            let id =
                self.decorated_id(ids, suffix, plain_id, (add_prefix, add_suffix), decorated)?;
            ids.spans[id as usize] = 1;
        }

        Ok(())
    }

    // Cache each affix combination by plain character ID. The reusable string
    // buffer avoids allocating again when a decorated identity already exists.
    fn decorated_id(
        &mut self,
        ids: &mut InitialTokenIds,
        suffix: Option<&str>,
        plain_id: u32,
        (add_prefix, add_suffix): (bool, bool),
        decorated: &mut String,
    ) -> Result<u32> {
        let Some(slot) = decoration_slot(add_prefix, add_suffix) else {
            return Ok(plain_id);
        };
        let cached = ids.decorated[plain_id as usize][slot];
        if cached != WORD_SEPARATOR_ID {
            return Ok(cached);
        }

        decorated.clear();
        if add_prefix {
            decorated.push_str(
                self.prefix
                    .as_deref()
                    .expect("prefix decoration requires a prefix"),
            );
        }
        decorated.push_str(self.tokens[plain_id as usize].as_str());
        if add_suffix {
            decorated.push_str(suffix.expect("suffix decoration requires a suffix"));
        }
        let id = self.intern(decorated, &mut ids.spans)?;
        ids.decorated[plain_id as usize][slot] = id;
        Ok(id)
    }

    pub(super) fn len(&self) -> usize {
        self.tokens.len()
    }

    pub(super) fn resolve_merge(&mut self, pair: Pair) -> Result<(u32, bool)> {
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
        self.insert_token(text)
    }

    pub(super) fn into_model_parts(self, merges: Vec<Pair>) -> (Vocab, Merges) {
        let merges = merges
            .into_iter()
            .map(|(left, right)| {
                (
                    self.tokens[left as usize].to_string(),
                    self.tokens[right as usize].to_string(),
                )
            })
            .collect();
        (
            self.tokens
                .into_iter()
                .enumerate()
                .map(|(id, token)| (token.to_string(), id as u32))
                .collect(),
            merges,
        )
    }
}

impl InitialTokenIds {
    pub(super) fn token_count(&self) -> usize {
        self.spans.len()
    }

    // Move activation spans and retire symbol lookup tables before merge rounds.
    pub(super) fn into_spans(self) -> Vec<usize> {
        self.spans
    }

    /// Interpret filtering and decorations against original UTF-8 coordinates.
    /// The plain path chooses its iterator once and avoids per-symbol affix checks.
    #[inline]
    pub(super) fn symbols<'a>(&'a self, word: &'a str) -> impl Iterator<Item = u32> + 'a {
        if self.plain() {
            itertools::Either::Left(
                word.chars()
                    .filter_map(|character| self.plain_id(character)),
            )
        } else {
            itertools::Either::Right(word.char_indices().filter_map(move |(byte, character)| {
                self.id(
                    character,
                    byte == 0,
                    byte + character.len_utf8() == word.len(),
                )
            }))
        }
    }

    pub(super) fn symbol_count(&self, text: &str) -> usize {
        if self.complete_alphabet {
            text.chars().count()
        } else {
            text.chars()
                .filter(|&ch| self.characters[ch as usize] != WORD_SEPARATOR_ID)
                .count()
        }
    }

    pub(super) fn plain(&self) -> bool {
        !self.prefix && !self.suffix
    }

    pub(super) fn plain_id(&self, character: char) -> Option<u32> {
        let id = self.characters[character as usize];
        (id != WORD_SEPARATOR_ID).then_some(id)
    }

    pub(super) fn id(&self, character: char, first: bool, last: bool) -> Option<u32> {
        let plain = self.characters[character as usize];
        if plain == WORD_SEPARATOR_ID {
            return None;
        }
        let add_prefix = self.prefix && !first;
        let add_suffix = self.suffix && last;
        Some(match decoration_slot(add_prefix, add_suffix) {
            None => plain,
            Some(slot) => self.decorated[plain as usize][slot],
        })
    }
}

// The three cached combinations retain their existing slots and token IDs.
fn decoration_slot(add_prefix: bool, add_suffix: bool) -> Option<usize> {
    match (add_prefix, add_suffix) {
        (false, false) => None,
        (true, false) => Some(0),
        (false, true) => Some(1),
        (true, true) => Some(2),
    }
}

/// Select the alphabet with the existing frequency-tie and codepoint order.
pub(super) fn select_alphabet(
    trainer: &BpeTrainer,
    wc: WordCountsView<'_>,
    workers: usize,
) -> Alphabet {
    if trainer.limit_alphabet.is_none() {
        // Keep the unlimited fast path: parallel presence, without frequencies.
        let observed = observed_characters(wc, workers);
        let mut present = observed.clone();
        for &character in &trainer.initial_alphabet {
            present.insert(character as usize);
        }
        let characters = present
            .ones()
            .map(|codepoint| {
                let character = char::from_u32(codepoint as u32)
                    .expect("alphabet bits come from valid characters");
                (character, observed.contains(codepoint))
            })
            .collect();
        return Alphabet {
            characters,
            complete: true,
            observed_specials: observed_special_characters(trainer, &observed),
        };
    }
    let mut observed = FixedBitSet::with_capacity(0x110000);
    // Compute the alphabet from seen words
    let mut alphabet: AHashMap<char, usize> = AHashMap::new();
    for (word, count) in wc.iter() {
        for c in word.chars() {
            *alphabet.entry(c).or_default() += *count as usize;
            observed.insert(c as usize);
        }
    }

    // Also include anything from the provided initial alphabet
    for c in &trainer.initial_alphabet {
        *alphabet.entry(*c).or_default() = usize::MAX;
    }

    let mut kept = alphabet.iter().collect::<Vec<_>>();

    // Compute the number of chars to remove from the alphabet
    // If `limit_alphabet < initial_alphabet.len()`, some of these initial characters
    // will be removed
    let to_remove = trainer
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
        .map(|(&character, _)| (character, observed.contains(character as usize)))
        .collect();
    // The 136 KiB bitmap is selection scratch, never retained across attempts.
    Alphabet {
        characters,
        complete: false,
        observed_specials: observed_special_characters(trainer, &observed),
    }
}

// Special identities can admit characters removed by the frequency cutoff.
fn observed_special_characters(trainer: &BpeTrainer, observed: &FixedBitSet) -> Vec<char> {
    trainer
        .special_tokens
        .iter()
        .filter_map(|token| {
            let mut chars = token.content.chars();
            let character = chars.next()?;
            (chars.next().is_none() && observed.contains(character as usize)).then_some(character)
        })
        .collect()
}

fn observed_characters(word_counts: WordCountsView<'_>, workers: usize) -> FixedBitSet {
    let words: Vec<_> = word_counts.keys().collect();
    let chunk = words.len().div_ceil(workers).max(1);
    words
        .par_chunks(chunk)
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

#[cfg(test)]
mod tests {
    use super::*;
    use ahash::AHashSet;

    #[test]
    fn initialization_completes_decorated_ids_before_weight_sorting() {
        use super::super::word_counts::WordCounts;
        let words = WordCounts::from_entries(
            [("qé🙂z", 1), ("🙂é", 9)]
                .into_iter()
                .map(|(word, count)| (word.into(), count))
                .collect(),
        );
        let mut trainer = BpeTrainer::new(1, 32);
        trainer.show_progress = false;
        trainer.limit_alphabet = Some(2);
        trainer.initial_alphabet = ['é', '🙂'].into();
        trainer.special_tokens = vec![tk_encode::vocab::bucket_added_vocabulary::AddedToken::from(
            "<s>", true,
        )];
        trainer.continuing_subword_prefix = Some("前".into());
        trainer.end_of_word_suffix = Some("尾".into());
        let (vocabulary, ids) = Vocabulary::initialize(
            &trainer,
            words.view(),
            &select_alphabet(&trainer, words.view(), 4),
            &None,
        )
        .unwrap();
        // Four observed characters exceed the two-character cutoff. Decorations
        // use original byte boundaries even after q/z are filtered away.
        assert_eq!(
            vocabulary
                .tokens
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
            ["<s>", "é", "🙂", "前é", "前🙂", "前é尾"]
        );
        assert_eq!(ids.symbols("qé🙂z").collect::<Vec<_>>(), [3, 4]);
        assert_eq!(ids.symbols("🙂é").collect::<Vec<_>>(), [2, 5]);
        assert_eq!(ids.token_count(), vocabulary.len());
        let spans = ids.into_spans();
        assert_eq!(spans, [0, 0, 1, 1, 1, 1]);
    }

    #[test]
    fn plain_activation_preserves_forced_unused_and_cutoff_special_characters() {
        use super::super::word_counts::WordCounts;
        use tk_encode::vocab::bucket_added_vocabulary::AddedToken;
        // Presence includes zero-weight words, independently of frequencies.
        let words = WordCounts::from_entries(vec![("猫é".into(), 0)]);
        for limit in [None, Some(1), Some(0)] {
            let mut trainer = BpeTrainer::new(1, 32);
            trainer.show_progress = false;
            trainer.limit_alphabet = limit;
            trainer.initial_alphabet = ['z'].into();
            trainer.special_tokens = ["猫", "z", "<s>"]
                .into_iter()
                .map(|text| AddedToken::from(text, true))
                .collect();
            // Empty affixes must also use the plain activation shortcut.
            trainer.continuing_subword_prefix = Some(String::new());
            trainer.end_of_word_suffix = Some(String::new());
            let alphabet = select_alphabet(&trainer, words.view(), 4);
            let (vocabulary, ids) =
                Vocabulary::initialize(&trainer, words.view(), &alphabet, &None).unwrap();
            assert_eq!(ids.complete_alphabet, limit.is_none());
            assert!(ids.plain());
            assert!(ids.decorated.is_empty());
            assert_eq!(&ids.spans[..3], [1, 0, 0]);
            assert_eq!(ids.symbols("猫z").collect::<Vec<_>>(), [0, 1]);
            assert_eq!(ids.symbol_count("猫é"), if limit.is_none() { 2 } else { 1 });
            assert_eq!(vocabulary.len(), if limit.is_none() { 4 } else { 3 });
        }
    }

    #[test]
    fn all_decoration_slots_preserve_text_and_cached_identity() {
        use super::super::word_counts::WordCounts;
        let words = WordCounts::from_entries(vec![("aaa".into(), 1), ("a".into(), 1)]);
        for prefix in [None, Some(""), Some("前")] {
            for suffix in [None, Some(""), Some("尾")] {
                let mut trainer = BpeTrainer::new(1, 32);
                trainer.show_progress = false;
                trainer.continuing_subword_prefix = prefix.map(str::to_owned);
                trainer.end_of_word_suffix = suffix.map(str::to_owned);
                let alphabet = select_alphabet(&trainer, words.view(), 1);
                let (vocabulary, ids) =
                    Vocabulary::initialize(&trainer, words.view(), &alphabet, &None).unwrap();
                for (first, last) in [(true, false), (false, false), (false, true), (true, true)] {
                    let expected = format!(
                        "{}a{}",
                        if first { "" } else { prefix.unwrap_or("") },
                        if last { suffix.unwrap_or("") } else { "" }
                    );
                    let id = ids.id('a', first, last).unwrap();
                    assert_eq!(vocabulary.tokens[id as usize].as_str(), expected);
                    assert_eq!(ids.spans[id as usize], 1);
                }
            }
        }
    }

    #[test]
    fn observed_reduction_preserves_presence_activation_and_model_ids() {
        for input in [
            vec![],
            vec![("aaaa", 1)],
            vec![("ASCII", 2), ("中文中文", 3), ("é🙂𐀀é", 7)],
        ] {
            let words = input
                .into_iter()
                .map(|(word, count)| (CompactString::from(word), count))
                .collect();
            let view = WordCountsView::from_map(&words);
            let expected: AHashSet<_> = view.keys().flat_map(|word| word.chars()).collect();
            let mut trainer = BpeTrainer::new(1, 64);
            trainer.show_progress = false;
            trainer.initial_alphabet = ['z', '🚀'].into();
            let model = trainer.do_train_impl(view, Some(1)).unwrap().parts;
            for workers in [1, 4, 8] {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(workers)
                    .build()
                    .unwrap()
                    .install(|| {
                        let observed = observed_characters(view, workers);
                        assert_eq!(observed.len(), 0x110000);
                        assert_eq!(
                            observed
                                .ones()
                                .map(|bit| char::from_u32(bit as u32).unwrap())
                                .collect::<AHashSet<_>>(),
                            expected
                        );
                        let (vocabulary, ids) = Vocabulary::initialize(
                            &trainer,
                            view,
                            &select_alphabet(&trainer, view, workers),
                            &None,
                        )
                        .unwrap();
                        for (id, token) in vocabulary.tokens.iter().enumerate() {
                            let character = token.chars().next().unwrap();
                            assert_eq!(ids.spans[id], usize::from(expected.contains(&character)));
                        }
                    });
                assert_eq!(
                    trainer.do_train_impl(view, Some(workers)).unwrap().parts,
                    model
                );
            }
        }
    }
}
