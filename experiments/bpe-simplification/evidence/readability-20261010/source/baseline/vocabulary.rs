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
    initial_spans: Vec<usize>,
    prefix: Option<String>,
    suffix: Option<String>,
    plain_ids_resolved: bool,
}

/// Canonical merge text plus any existing ID, before identity resolution.
/// Selection checks an existing ID's activation before committing this token.
pub(super) struct MergeToken {
    pub(super) existing_id: Option<u32>,
    text: CompactString,
}

/// Immutable character/decorated-ID tables shared by counting and materialization.
/// Both stages scan the same filtered alphabet and affix interpretation, preserving
/// identical token sequences without interning strings during parallel scans.
pub(super) struct InitialTokenIds {
    characters: Vec<u32>,
    decorated: Vec<[u32; 3]>,
    prefix: bool,
    suffix: bool,
    complete_alphabet: bool,
}

impl Vocabulary {
    pub(super) fn initialize(
        trainer: &BpeTrainer,
        word_counts: WordCountsView<'_>,
        workers: usize,
        retained_alphabet: &mut Option<Vec<char>>,
    ) -> Result<Self> {
        let mut vocabulary = Self {
            tokens: IndexSet::with_capacity_and_hasher(trainer.vocab_size, RandomState::default()),
            initial_spans: Vec::new(),
            prefix: trainer.continuing_subword_prefix.clone(),
            suffix: trainer.end_of_word_suffix.clone(),
            plain_ids_resolved: false,
        };
        // 1. Reserve special IDs before inserting the codepoint-ordered alphabet.
        for token in &trainer.special_tokens {
            vocabulary.intern(token.content.as_str())?;
        }

        // 2. Preserve the limited selector, or collect all observed characters.
        if trainer.limit_alphabet.is_some() {
            // Preserve the existing frequency-tie selector for limited alphabets.
            let characters =
                retained_alphabet.get_or_insert_with(|| select_alphabet(trainer, word_counts));
            for &character in characters.iter() {
                let mut utf8 = [0; 4];
                vocabulary.intern(character.encode_utf8(&mut utf8))?;
            }
        } else {
            vocabulary.initialize_unlimited_alphabet(trainer, word_counts, workers)?;
        }
        Ok(vocabulary)
    }

    // With no limit, a parallel presence bitmap replaces weighted frequency
    // counting. Codepoint traversal still determines the alphabet's token IDs.
    fn initialize_unlimited_alphabet(
        &mut self,
        trainer: &BpeTrainer,
        word_counts: WordCountsView<'_>,
        workers: usize,
    ) -> Result<()> {
        let observed = observed_characters(word_counts, workers);
        let mut present = observed.clone();
        for &character in &trainer.initial_alphabet {
            present.insert(character as usize);
        }
        for codepoint in present.ones() {
            let character =
                char::from_u32(codepoint as u32).expect("alphabet bits come from valid characters");
            let mut utf8 = [0; 4];
            let id = self.intern(character.encode_utf8(&mut utf8))?;
            self.initial_spans[id as usize] = usize::from(observed.contains(codepoint));
        }
        self.plain_ids_resolved = true;
        Ok(())
    }

    fn intern(&mut self, text: &str) -> Result<u32> {
        if let Some(id) = self.tokens.get_index_of(text) {
            return Ok(id as u32);
        }
        let id = self.insert_new_token(CompactString::from(text))?;
        self.initial_spans.push(0);
        Ok(id)
    }

    // The caller has established that this text is absent. No vocabulary
    // mutation can intervene before this insertion on the coordinator.
    fn insert_new_token(&mut self, token: CompactString) -> Result<u32> {
        let id = u32::try_from(self.tokens.len()).map_err(|_| "BPE vocabulary exceeds u32")?;
        if id == WORD_SEPARATOR_ID {
            return Err("BPE token ID collides with the word separator".into());
        }
        let (index, inserted) = self.tokens.insert_full(token);
        debug_assert!(inserted && index == id as usize);
        Ok(id)
    }

    pub(super) fn initial_ids(
        &mut self,
        word_counts: WordCountsView<'_>,
        progress: &Option<ProgressBar>,
    ) -> Result<InitialTokenIds> {
        // 1. Map existing character IDs before any decorated identities are inserted.
        let mut ids = InitialTokenIds {
            characters: vec![WORD_SEPARATOR_ID; 0x110000],
            decorated: Vec::new(),
            prefix: self.prefix.as_deref().is_some_and(|p| !p.is_empty()),
            suffix: self.suffix.as_deref().is_some_and(|s| !s.is_empty()),
            complete_alphabet: self.plain_ids_resolved,
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
            ids.decorated
                .resize(self.tokens.len(), [WORD_SEPARATOR_ID; 3]);
        }
        if !ids.prefix && !ids.suffix && self.plain_ids_resolved {
            if let Some(p) = progress {
                p.inc(word_counts.len() as u64);
            }
            return Ok(ids);
        }

        // 2. Activate retained symbols in input traversal order. Sorting words first
        // would change the IDs allocated for previously unseen decorated strings.
        self.initial_spans.fill(0);
        let mut decorated = String::new();
        for (index, word) in word_counts.keys().enumerate() {
            self.activate_word(word, &mut ids, &mut decorated)?;
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
        decorated: &mut String,
    ) -> Result<()> {
        for (byte, character) in word.char_indices() {
            let plain_id = ids.characters[character as usize];
            if plain_id == WORD_SEPARATOR_ID {
                continue;
            }
            let flags = usize::from(ids.prefix && byte != 0)
                | (usize::from(ids.suffix && byte + character.len_utf8() == word.len()) << 1);
            let id = self.decorated_id(ids, plain_id, character, flags, decorated)?;
            self.initial_spans[id as usize] = 1;
        }

        Ok(())
    }

    // Cache each affix combination by plain character ID. The reusable string
    // buffer avoids allocating again when a decorated identity already exists.
    fn decorated_id(
        &mut self,
        ids: &mut InitialTokenIds,
        plain_id: u32,
        character: char,
        flags: usize,
        decorated: &mut String,
    ) -> Result<u32> {
        if flags == 0 {
            return Ok(plain_id);
        }
        let cached = ids.decorated[plain_id as usize][flags - 1];
        if cached != WORD_SEPARATOR_ID {
            return Ok(cached);
        }

        decorated.clear();
        if flags & 1 != 0 {
            decorated.push_str(
                self.prefix
                    .as_deref()
                    .expect("prefix flag requires a prefix"),
            );
        }
        decorated.push(character);
        if flags & 2 != 0 {
            decorated.push_str(
                self.suffix
                    .as_deref()
                    .expect("suffix flag requires a suffix"),
            );
        }
        let id = self.intern(decorated)?;
        ids.decorated[plain_id as usize][flags - 1] = id;
        Ok(id)
    }

    pub(super) fn take_initial_spans(&mut self) -> Vec<usize> {
        std::mem::take(&mut self.initial_spans)
    }

    pub(super) fn len(&self) -> usize {
        self.tokens.len()
    }

    pub(super) fn merge_token(&self, pair: Pair) -> MergeToken {
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
        MergeToken {
            existing_id: self.tokens.get_index_of(text.as_str()).map(|id| id as u32),
            text,
        }
    }

    pub(super) fn resolve_merge(&mut self, token: MergeToken) -> Result<u32> {
        match token.existing_id {
            Some(id) => Ok(id),
            // PERF: merge_token already checked this key. Consume its complete
            // string instead of probing again and copying another owned string.
            None => self.insert_new_token(token.text),
        }
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
    /// Interpret filtering and decorations against original UTF-8 coordinates.
    /// The plain path chooses its loop once and avoids per-symbol affix checks.
    #[inline]
    pub(super) fn scan_symbols<B>(
        &self,
        word: &str,
        mut emit: impl FnMut(u32) -> std::ops::ControlFlow<B>,
    ) -> std::ops::ControlFlow<B> {
        use std::ops::ControlFlow;
        if self.plain() {
            for character in word.chars() {
                if let Some(id) = self.plain_id(character) {
                    emit(id)?;
                }
            }
        } else {
            for (byte, character) in word.char_indices() {
                if let Some(id) = self.id(
                    character,
                    byte == 0,
                    byte + character.len_utf8() == word.len(),
                ) {
                    emit(id)?;
                }
            }
        }
        ControlFlow::Continue(())
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
        let flags = usize::from(self.prefix && !first) | (usize::from(self.suffix && last) << 1);
        Some(if flags == 0 {
            plain
        } else {
            self.decorated[plain as usize][flags - 1]
        })
    }
}

/// Select the alphabet with the existing frequency-tie and codepoint order.
pub(super) fn select_alphabet(trainer: &BpeTrainer, wc: WordCountsView<'_>) -> Vec<char> {
    // Compute the alphabet from seen words
    let mut alphabet: AHashMap<char, usize> = AHashMap::new();
    for (word, count) in wc.iter() {
        for c in word.chars() {
            *alphabet.entry(c).or_default() += *count as usize;
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
    kept.into_iter().map(|(&character, _)| character).collect()
}

fn observed_characters(word_counts: WordCountsView<'_>, workers: usize) -> FixedBitSet {
    let words: Vec<_> = word_counts.keys().collect();
    let chunk = words.len().div_ceil(workers).max(1);
    let bitmaps: Vec<_> = words
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
        .collect();
    let mut present = FixedBitSet::with_capacity(0x110000);
    for bitmap in bitmaps {
        present.union_with(&bitmap);
    }
    present
}
