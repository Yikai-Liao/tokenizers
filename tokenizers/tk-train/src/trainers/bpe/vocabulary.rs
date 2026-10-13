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
    merges: Vec<Pair>,
    #[cfg(test)]
    pub(super) trace: super::Trace,
}

/// Canonical result identity; existence is independent of corpus activation.
pub(super) struct ResolvedMerge {
    pub(super) merged_id: u32,
    pub(super) already_in_vocabulary: bool,
}

/// Immutable character/decorated-ID tables shared by counting and materialization.
/// Both stages scan the same filtered alphabet and affix interpretation, preserving
/// identical token sequences without interning strings during parallel scans.
pub(super) struct InitialTokenIds {
    spans: Vec<usize>,
    characters: Box<[u32]>,
    decorated: Box<[DecoratedIds]>,
    prefix: bool,
    suffix: bool,
    complete_alphabet: bool,
}

// Only decorated states have columns; Plain returns the caller's ordinary ID.
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
    fn get(&self, affix: Affix, plain: u32) -> u32 {
        match affix {
            Affix::Plain => plain,
            _ => self.0[affix as usize],
        }
    }

    fn remember(&mut self, affix: Affix, id: u32) {
        if affix != Affix::Plain {
            self.0[affix as usize] = id;
        }
    }
}

/// Selected once per training call; retries borrow the same cutoff decisions.
/// Presence is retained per selected character so the selection bitmap can retire
/// before token lookup tables and corpus storage are allocated.
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

impl Alphabet {
    #[cfg(test)]
    pub(super) fn characters(&self) -> &[AlphabetCharacter] {
        &self.characters
    }
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
            merges: Vec::new(),
            #[cfg(test)]
            trace: Vec::new(),
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
        for entry in &alphabet.characters {
            let mut utf8 = [0; 4];
            let id = vocabulary.intern(entry.character.encode_utf8(&mut utf8), &mut spans)?;
            spans[id as usize] = usize::from(entry.observed_in_input);
        }
        // Resolve decorations before CorpusPlan sorts input by weight. Keep the
        // plain-alphabet shortcut and the original progress boundary.
        trainer.update_progress(progress, word_counts.len(), "Tokenize words");
        let ids = InitialTokenIds::build(
            &mut vocabulary,
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
        let (id, _) = self.insert(CompactString::from(text))?;
        spans.push(0);
        Ok(id)
    }

    // ID errors discard the whole attempt, including any insertion made here.
    fn insert(&mut self, token: CompactString) -> Result<(u32, bool)> {
        let (index, inserted) = self.tokens.insert_full(token);
        let id = u32::try_from(index).map_err(|_| "BPE vocabulary exceeds u32")?;
        if id == WORD_SEPARATOR_ID {
            return Err("BPE token ID collides with the word separator".into());
        }
        Ok((id, !inserted))
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

    /// Record a complete successful round in selection order, including reused IDs.
    pub(super) fn commit(&mut self, pairs: Vec<Pair>) {
        self.merges.extend(pairs);
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

impl InitialTokenIds {
    fn build(
        vocabulary: &mut Vocabulary,
        word_counts: WordCountsView<'_>,
        spans: Vec<usize>,
        complete_alphabet: bool,
        suffix: Option<&str>,
        progress: &Option<ProgressBar>,
    ) -> Result<Self> {
        // 1. Map existing character IDs before any decorated identities are inserted.
        let mut ids = Self {
            spans,
            characters: vec![WORD_SEPARATOR_ID; 0x110000].into_boxed_slice(),
            decorated: Box::default(),
            prefix: vocabulary.prefix.as_deref().is_some_and(|p| !p.is_empty()),
            suffix: suffix.is_some_and(|s| !s.is_empty()),
            complete_alphabet,
        };
        for (id, token) in vocabulary.tokens.iter().enumerate() {
            let mut chars = token.chars();
            if let Some(character) = chars.next()
                && chars.next().is_none()
            {
                ids.characters[character as usize] = id as u32;
            }
        }
        if ids.prefix || ids.suffix {
            let mut decorated = Vec::new();
            decorated.resize(
                vocabulary.tokens.len(),
                DecoratedIds([WORD_SEPARATOR_ID; 3]),
            );
            ids.decorated = decorated.into_boxed_slice();
        }
        if ids.is_plain() {
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
            ids.activate(word, vocabulary, suffix, &mut decorated)?;
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

    fn activate(
        &mut self,
        word: &str,
        vocabulary: &mut Vocabulary,
        suffix: Option<&str>,
        decorated: &mut String,
    ) -> Result<()> {
        let mut original = word.chars();
        let mut first = true;
        while let Some(character) = original.next() {
            let is_first = first;
            first = false;
            let plain_id = self.characters[character as usize];
            if plain_id == WORD_SEPARATOR_ID {
                continue;
            }
            let affix = self.affix(is_first, original.as_str().is_empty());
            let id = self.decorate(vocabulary, suffix, plain_id, affix, decorated)?;
            self.spans[id as usize] = 1;
        }

        Ok(())
    }

    // Cache each affix combination by plain character ID. The reusable string
    // buffer avoids allocating again when a decorated identity already exists.
    fn decorate(
        &mut self,
        vocabulary: &mut Vocabulary,
        suffix: Option<&str>,
        plain_id: u32,
        affix: Affix,
        decorated: &mut String,
    ) -> Result<u32> {
        // Plain never indexes the table, which is empty without affixes.
        if affix == Affix::Plain {
            return Ok(plain_id);
        }
        let cached = self.decorated[plain_id as usize].get(affix, plain_id);
        if cached != WORD_SEPARATOR_ID {
            return Ok(cached);
        }

        decorated.clear();
        if matches!(affix, Affix::Prefix | Affix::Both) {
            decorated.push_str(
                vocabulary
                    .prefix
                    .as_deref()
                    .expect("prefix decoration requires a prefix"),
            );
        }
        decorated.push_str(vocabulary.tokens[plain_id as usize].as_str());
        if matches!(affix, Affix::Suffix | Affix::Both) {
            decorated.push_str(suffix.expect("suffix decoration requires a suffix"));
        }
        let id = vocabulary.intern(decorated, &mut self.spans)?;
        self.decorated[plain_id as usize].remember(affix, id);
        Ok(id)
    }

    fn affix(&self, first: bool, last: bool) -> Affix {
        match (self.prefix && !first, self.suffix && last) {
            (false, false) => Affix::Plain,
            (true, false) => Affix::Prefix,
            (false, true) => Affix::Suffix,
            (true, true) => Affix::Both,
        }
    }

    pub(super) fn token_count(&self) -> usize {
        self.spans.len()
    }

    // Move activation spans and retire symbol lookup tables before merge rounds.
    pub(super) fn into_metadata(self) -> Vec<usize> {
        self.spans
    }

    /// Interpret filtering and decorations against original UTF-8 coordinates.
    /// The plain path chooses its iterator once and avoids per-symbol affix checks.
    #[inline]
    pub(super) fn symbols<'a>(&'a self, word: &'a str) -> impl Iterator<Item = u32> + 'a {
        if self.is_plain() {
            itertools::Either::Left(
                word.chars()
                    .filter_map(|character| self.get_plain(character)),
            )
        } else {
            itertools::Either::Right(word.char_indices().filter_map(move |(byte, character)| {
                self.get(
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

    pub(super) fn is_plain(&self) -> bool {
        !self.prefix && !self.suffix
    }

    fn get_plain(&self, character: char) -> Option<u32> {
        let id = self.characters[character as usize];
        (id != WORD_SEPARATOR_ID).then_some(id)
    }

    fn get(&self, character: char, first: bool, last: bool) -> Option<u32> {
        let plain = self.characters[character as usize];
        if plain == WORD_SEPARATOR_ID {
            return None;
        }
        let affix = self.affix(first, last);
        Some(if affix == Affix::Plain {
            plain
        } else {
            self.decorated[plain as usize].get(affix, plain)
        })
    }
}

/// Select the alphabet with the existing frequency-tie and codepoint order.
pub(super) fn select_alphabet(trainer: &BpeTrainer, wc: WordCountsView<'_>) -> Alphabet {
    if trainer.limit_alphabet.is_none() {
        // Keep the unlimited fast path: parallel presence, without frequencies.
        let observed = observed_characters(wc);
        let mut present = observed.clone();
        for &character in &trainer.initial_alphabet {
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
        .map(|(&character, _)| AlphabetCharacter {
            character,
            observed_in_input: observed.contains(character as usize),
        })
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

fn observed_characters(word_counts: WordCountsView<'_>) -> FixedBitSet {
    let words: Vec<_> = word_counts.keys().collect();
    let workers = rayon::current_num_threads();
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
    use super::super::word_counts::WordCounts;
    use super::*;
    use ahash::AHashSet;
    use tk_encode::vocab::bucket_added_vocabulary::AddedToken;

    #[test]
    fn decoration_row() {
        assert_eq!(std::mem::size_of::<DecoratedIds>(), 12);
        let mut row = DecoratedIds([WORD_SEPARATOR_ID; 3]);
        for (affix, id) in [(Affix::Prefix, 0), (Affix::Suffix, 1), (Affix::Both, 2)] {
            assert_eq!(row.get(affix, 9), WORD_SEPARATOR_ID);
            row.remember(affix, id);
        }
        assert_eq!(row.0, [0, 1, 2]);
        row.remember(Affix::Plain, 8);
        assert_eq!(row.0, [0, 1, 2]);
        assert_eq!(row.get(Affix::Plain, 9), 9);
    }

    #[test]
    fn affix_ids() {
        let inputs = ["a", "axa", "zaaz", "aa", ""];
        let words = WordCounts::from_entries(inputs.map(|s| (s.into(), 1)).to_vec());
        let mut trainer = BpeTrainer::new(1, 32);
        trainer.limit_alphabet = Some(1);
        trainer.initial_alphabet = ['a'].into();
        trainer.special_tokens = ["##a", "a</w>", "a"]
            .map(|s| AddedToken::from(s, true))
            .to_vec();
        for (prefix, suffix, single, inner, last, spans) in [
            ("##", "</w>", 1, 0, 3, &[1, 1, 1, 1][..]),
            ("##", "", 2, 0, 0, &[1, 0, 1][..]),
            ("", "</w>", 1, 2, 1, &[0, 1, 1][..]),
            ("", "", 2, 2, 2, &[0, 0, 1][..]),
        ] {
            let expected = [
                vec![single],
                vec![2, last],
                vec![inner, inner],
                vec![2, last],
                vec![],
            ];
            trainer.continuing_subword_prefix = Some(prefix.into());
            trainer.end_of_word_suffix = Some(suffix.into());
            let alphabet = select_alphabet(&trainer, words.view());
            let (mut vocabulary, mut ids) =
                Vocabulary::initialize(&trainer, words.view(), &alphabet, &None).unwrap();
            let tokens: Vec<_> = vocabulary.tokens.iter().map(|s| s.as_str()).collect();
            assert_eq!(&tokens[..3], ["##a", "a</w>", "a"]);
            assert_eq!(tokens.len(), spans.len());
            if let Some(token) = tokens.get(3) {
                assert_eq!(*token, "##a</w>");
            }
            let actual = inputs.map(|s| ids.symbols(s).collect::<Vec<_>>());
            assert_eq!(actual, expected);
            assert_eq!(ids.spans, spans);
            let count = vocabulary.len();
            ids.activate("aa", &mut vocabulary, Some(suffix), &mut String::new())
                .unwrap();
            assert_eq!(vocabulary.len(), count);
            assert_eq!(ids.symbols("aa").collect::<Vec<_>>(), expected[3]);
            if prefix.is_empty() && suffix.is_empty() {
                assert!(ids.decorated.is_empty());
                assert_eq!(ids.get('a', true, true), Some(2));
            }
        }
    }

    #[test]
    fn unicode_id_order() {
        let words = WordCounts::from_entries(vec![("qé🙂z".into(), 1), ("🙂é".into(), 9)]);
        let mut trainer = BpeTrainer::new(1, 32);
        trainer.limit_alphabet = Some(2);
        trainer.initial_alphabet = ['é', '🙂'].into();
        trainer.continuing_subword_prefix = Some("前".into());
        trainer.end_of_word_suffix = Some("尾".into());
        let alphabet = select_alphabet(&trainer, words.view());
        let (vocabulary, ids) =
            Vocabulary::initialize(&trainer, words.view(), &alphabet, &None).unwrap();
        assert_eq!(
            vocabulary
                .tokens
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
            ["é", "🙂", "前é", "前🙂", "前é尾"]
        );
        assert_eq!(ids.symbols("qé🙂z").collect::<Vec<_>>(), [2, 3]);
        assert_eq!(ids.symbols("🙂é").collect::<Vec<_>>(), [1, 4]);
    }

    #[test]
    fn alphabet_activation() {
        // Presence includes zero-weight words, independently of frequencies.
        let words = WordCounts::from_entries(vec![("猫é".into(), 0)]);
        let mut trainer = BpeTrainer::new(1, 32);
        trainer.limit_alphabet = Some(0);
        trainer.initial_alphabet = ['z'].into();
        trainer.special_tokens = ["猫", "z", "<s>"]
            .into_iter()
            .map(|text| AddedToken::from(text, true))
            .collect();
        let alphabet = select_alphabet(&trainer, words.view());
        let (_, ids) = Vocabulary::initialize(&trainer, words.view(), &alphabet, &None).unwrap();
        assert_eq!(&ids.spans[..3], [1, 0, 0]);
        assert_eq!(ids.symbols("猫z").collect::<Vec<_>>(), [0, 1]);
    }

    #[test]
    fn unicode_presence() {
        let words = WordCounts::from_entries(
            ["ASCII", "中文中文", "é🙂𐀀é"]
                .map(|s| (s.into(), 0))
                .to_vec(),
        );
        let expected: AHashSet<_> = "ASCII中文é🙂𐀀".chars().collect();
        for workers in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            let observed = pool.install(|| observed_characters(words.view()));
            assert_eq!(
                observed
                    .ones()
                    .map(|bit| char::from_u32(bit as u32).unwrap())
                    .collect::<AHashSet<_>>(),
                expected
            );
        }
        let words = WordCounts::default();
        assert_eq!(observed_characters(words.view()).count_ones(..), 0);
    }
}
