//! Vocabulary identity and canonical output strings, independent of position storage.
use super::{BpeTrainer, WORD_SEPARATOR_ID};
use crate::progress::{TrainingProgress, WorkProgress};
use ahash::AHashMap;
use compact_str::CompactString;
use rayon::prelude::*;
use tk_encode::{
    Result,
    models::bpe::{Merges, Pair, Vocab},
};

pub(super) struct Vocabulary {
    token_to_id: AHashMap<CompactString, u32>,
    tokens: Vec<CompactString>,
    active: Vec<bool>,
    prefix: Option<String>,
    suffix: Option<String>,
    plain_ids_resolved: bool,
}
pub(super) struct MergeToken {
    pub(super) existing_id: Option<u32>,
    text: CompactString,
}
pub(super) struct MergeIdentity {
    pub(super) id: u32,
    pub(super) reused_active_id: bool,
}
pub(super) struct InitialIds {
    characters: Vec<u32>,
    decorated: Vec<[u32; 3]>,
    prefix: bool,
    suffix: bool,
    complete_alphabet: bool,
}
impl Vocabulary {
    pub(super) fn initialize(
        trainer: &BpeTrainer,
        word_counts: &AHashMap<CompactString, u64>,
        workers: usize,
        progress: &TrainingProgress,
    ) -> Result<Self> {
        let mut vocabulary = Self {
            token_to_id: AHashMap::with_capacity(trainer.vocab_size),
            tokens: Vec::with_capacity(trainer.vocab_size),
            active: Vec::new(),
            prefix: trainer.continuing_subword_prefix.clone(),
            suffix: trainer.end_of_word_suffix.clone(),
            plain_ids_resolved: false,
        };
        for token in &trainer.special_tokens {
            vocabulary.intern(token.content.as_str())?;
        }
        let work = progress.stage("Compute alphabet", word_counts.len());
        if trainer.limit_alphabet.is_some() {
            // Preserve the existing frequency-tie selector for limited alphabets.
            trainer.compute_alphabet(
                word_counts,
                &mut vocabulary.token_to_id,
                &mut vocabulary.tokens,
            );
            vocabulary.active.resize(vocabulary.tokens.len(), false);
            work.complete(word_counts.len());
        } else {
            let words: Vec<_> = word_counts.keys().collect();
            let chunk = words.len().div_ceil(workers).max(1);
            let bitmaps: Vec<_> = words
                .par_chunks(chunk)
                .map(|chunk| {
                    let mut present = vec![0_u64; 0x110000 / 64];
                    for word in chunk {
                        for character in word.chars() {
                            let codepoint = character as usize;
                            present[codepoint / 64] |= 1_u64 << (codepoint % 64);
                        }
                    }
                    work.complete(chunk.len());
                    present
                })
                .collect();
            let mut present = vec![0_u64; 0x110000 / 64];
            for bitmap in bitmaps {
                for (merged, bits) in present.iter_mut().zip(bitmap) {
                    *merged |= bits;
                }
            }
            let observed = present.clone();
            for &character in &trainer.initial_alphabet {
                let codepoint = character as usize;
                present[codepoint / 64] |= 1_u64 << (codepoint % 64);
            }
            for (word, mut bits) in present.into_iter().enumerate() {
                while bits != 0 {
                    let codepoint = word * 64 + bits.trailing_zeros() as usize;
                    let character = char::from_u32(codepoint as u32)
                        .expect("alphabet bits come from valid characters");
                    let mut utf8 = [0; 4];
                    let id = vocabulary.intern(character.encode_utf8(&mut utf8))?;
                    vocabulary.active[id as usize] =
                        observed[codepoint / 64] & (1_u64 << (codepoint % 64)) != 0;
                    bits &= bits - 1;
                }
            }
            vocabulary.plain_ids_resolved = true;
        }
        Ok(vocabulary)
    }
    fn intern(&mut self, text: &str) -> Result<u32> {
        if let Some(&id) = self.token_to_id.get(text) {
            return Ok(id);
        }
        let id = u32::try_from(self.tokens.len()).map_err(|_| "BPE vocabulary exceeds u32")?;
        if id == WORD_SEPARATOR_ID {
            return Err("BPE token ID collides with the word separator".into());
        }
        let token = CompactString::from(text);
        self.tokens.push(token.clone());
        self.token_to_id.insert(token, id);
        self.active.push(false);
        Ok(id)
    }
    pub(super) fn initial_ids(
        &mut self,
        word_counts: &AHashMap<CompactString, u64>,
        work: &WorkProgress,
    ) -> Result<InitialIds> {
        let mut ids = InitialIds {
            characters: vec![WORD_SEPARATOR_ID; 0x110000],
            decorated: Vec::new(),
            prefix: self.prefix.as_deref().is_some_and(|p| !p.is_empty()),
            suffix: self.suffix.as_deref().is_some_and(|s| !s.is_empty()),
            complete_alphabet: self.plain_ids_resolved,
        };
        for (token, &id) in &self.token_to_id {
            let mut chars = token.chars();
            if let Some(character) = chars.next()
                && chars.next().is_none()
            {
                ids.characters[character as usize] = id;
            }
        }
        if ids.prefix || ids.suffix {
            ids.decorated
                .resize(self.tokens.len(), [WORD_SEPARATOR_ID; 3]);
        }
        if !ids.prefix && !ids.suffix && self.plain_ids_resolved {
            work.complete(word_counts.len());
            return Ok(ids);
        }
        self.active.fill(false);
        let mut decorated = String::new();
        // Allocate decorated IDs in the original map traversal before reordering words.
        for word in word_counts.keys() {
            for (byte, character) in word.char_indices() {
                let plain_id = ids.characters[character as usize];
                if plain_id == WORD_SEPARATOR_ID {
                    continue;
                }
                let flags = usize::from(ids.prefix && byte != 0)
                    | (usize::from(ids.suffix && byte + character.len_utf8() == word.len()) << 1);
                let id = if flags == 0 {
                    plain_id
                } else {
                    let cached = ids.decorated[plain_id as usize][flags - 1];
                    if cached != WORD_SEPARATOR_ID {
                        cached
                    } else {
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
                        let id = self.intern(&decorated)?;
                        ids.decorated[plain_id as usize][flags - 1] = id;
                        id
                    }
                };
                self.active[id as usize] = true;
            }
            work.complete(1);
        }
        Ok(ids)
    }
    pub(super) fn initial_spans(&self) -> Vec<u64> {
        self.active
            .iter()
            .map(|&active| u64::from(active))
            .collect()
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
            existing_id: self.token_to_id.get(text.as_str()).copied(),
            text,
        }
    }
    pub(super) fn resolve_merge(&mut self, token: MergeToken) -> Result<MergeIdentity> {
        let id = match token.existing_id {
            Some(id) => id,
            None => self.intern(token.text.as_str())?,
        };
        let reused_active_id = self.active[id as usize];
        self.active[id as usize] = true;
        Ok(MergeIdentity {
            id,
            reused_active_id,
        })
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
            self.token_to_id
                .into_iter()
                .map(|(token, id)| (token.to_string(), id))
                .collect(),
            merges,
        )
    }
}
impl InitialIds {
    pub(super) fn complete_alphabet(&self) -> bool {
        self.complete_alphabet
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
    pub(super) fn retained(&self, character: char) -> bool {
        self.characters[character as usize] != WORD_SEPARATOR_ID
    }
}
