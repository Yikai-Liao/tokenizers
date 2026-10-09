#![allow(clippy::map_entry)]

mod feed;
mod word_counts;
use word_counts::{WordCounts, WordCountsView};

mod engine;
#[cfg(feature = "parity-aware-bpe")]
pub mod parity_trainer;
#[cfg(test)]
mod reference;
#[cfg(feature = "parity-aware-bpe")]
mod word;
#[cfg(feature = "parity-aware-bpe")]
pub use parity_trainer::{ParityBpeTrainer, ParityBpeTrainerBuilder, ParityVariant};

use crate::Trainer;
use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use tk_encode::vocab::bucket_added_vocabulary::AddedToken;
// The optional parity trainer uses linked words; ordinary training owns a
// fixed-coordinate corpus. The test oracle uses independent sequential vectors.
#[cfg(feature = "parity-aware-bpe")]
use word::{WithFirstLastIterator, Word};

use tk_encode::Result;
#[cfg(any(test, feature = "parity-aware-bpe"))]
use tk_encode::models::bpe::Pair;
use tk_encode::models::bpe::{BpeConfig, Merges, PipelineBPE, Vocab};
use tk_encode::parallelism::*;
use tk_encode::utils::progress::ProgressFormat;

struct Config {
    min_frequency: u64,
    vocab_size: usize,
    show_progress: bool,
    progress_format: ProgressFormat,
    special_tokens: Vec<AddedToken>,
    limit_alphabet: Option<usize>,
    initial_alphabet: AHashSet<char>,
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
    max_token_length: Option<usize>,
}

/// A `BpeTrainerBuilder` can be used to create a `BpeTrainer` with a custom
/// configuration.
pub struct BpeTrainerBuilder {
    config: Config,
}

impl Default for BpeTrainerBuilder {
    fn default() -> Self {
        Self {
            config: Config {
                min_frequency: 0,
                vocab_size: 30000,
                show_progress: true,
                progress_format: ProgressFormat::default(),
                special_tokens: vec![],
                limit_alphabet: None,
                initial_alphabet: AHashSet::new(),
                continuing_subword_prefix: None,
                end_of_word_suffix: None,
                max_token_length: None,
            },
        }
    }
}

impl BpeTrainerBuilder {
    /// Constructs a new `BpeTrainerBuilder`
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the expected minimum frequency
    #[must_use]
    pub fn min_frequency(mut self, frequency: u64) -> Self {
        self.config.min_frequency = frequency;
        self
    }

    /// Set the vocabulary size
    #[must_use]
    pub fn vocab_size(mut self, size: usize) -> Self {
        self.config.vocab_size = size;
        self
    }

    /// Set whether to show progress
    #[must_use]
    pub fn show_progress(mut self, show: bool) -> Self {
        self.config.show_progress = show;
        self
    }

    /// Set the progress output format
    ///
    /// Controls how progress information is reported during training.
    /// - `Indicatif` (default): Interactive terminal progress bars
    /// - `JsonLines`: Machine-readable JSON lines to stderr
    /// - `Silent`: No progress output
    #[must_use]
    pub fn progress_format(mut self, format: ProgressFormat) -> Self {
        self.config.progress_format = format;
        self
    }

    /// Set the special tokens
    #[must_use]
    pub fn special_tokens(mut self, tokens: Vec<AddedToken>) -> Self {
        self.config.special_tokens = tokens;
        self
    }

    /// Set whether to limit the alphabet
    #[must_use]
    pub fn limit_alphabet(mut self, limit: usize) -> Self {
        self.config.limit_alphabet = Some(limit);
        self
    }

    /// Set the initial alphabet. See [`BpeTrainer::initial_alphabet`] for truncation.
    #[must_use]
    pub fn initial_alphabet(mut self, alphabet: HashSet<char>) -> Self {
        let mut initial_alphabet = AHashSet::with_capacity(alphabet.len());
        initial_alphabet.extend(alphabet);
        self.config.initial_alphabet = initial_alphabet;
        self
    }

    /// Set the continuing_subword_prefix
    #[must_use]
    pub fn continuing_subword_prefix(mut self, prefix: String) -> Self {
        self.config.continuing_subword_prefix = Some(prefix);
        self
    }

    /// Set the end_of_word_suffix
    #[must_use]
    pub fn end_of_word_suffix(mut self, suffix: String) -> Self {
        self.config.end_of_word_suffix = Some(suffix);
        self
    }
    /// Set the exclusive span limit for newly created adjacent pairs.
    ///
    /// See [`BpeTrainer::max_token_length`] for units and the initial-pair exception.
    #[must_use]
    pub fn max_token_length(mut self, max_token_length: Option<usize>) -> Self {
        self.config.max_token_length = max_token_length;
        self
    }

    /// Constructs the final BpeTrainer
    pub fn build(self) -> BpeTrainer {
        BpeTrainer {
            min_frequency: self.config.min_frequency,
            vocab_size: self.config.vocab_size,
            show_progress: self.config.show_progress,
            progress_format: self.config.progress_format,
            special_tokens: self.config.special_tokens,
            limit_alphabet: self.config.limit_alphabet,
            initial_alphabet: self.config.initial_alphabet,
            continuing_subword_prefix: self.config.continuing_subword_prefix,
            end_of_word_suffix: self.config.end_of_word_suffix,
            max_token_length: self.config.max_token_length,
            words: WordCounts::default(),
        }
    }
}

/// In charge of training a `BPE` model
///
/// # Examples
///
/// ```
/// use tk_train::BpeTrainer;
/// use tk_train::Trainer;
/// use tk_encode::models::bpe::{PipelineBPE, BpeConfig};
///
/// let sequences = vec![ "Hello", "World" ];
///
/// let mut trainer = BpeTrainer::default();
/// trainer.feed(sequences.iter(), |s| Ok(vec![s.to_owned()]));
///
/// // `PipelineBPE` has no empty state to train *into* -- it only exists once there is a
/// // vocabulary and a merge list -- so take the parts and build it.
/// let (vocab, merges, special_tokens) = trainer.train_vocab().unwrap();
/// let model = PipelineBPE::from_config(BpeConfig { vocab, merges, ..BpeConfig::default() }).unwrap();
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Eq)]
pub struct BpeTrainer {
    /// The minimum frequency a pair must have to produce a merge operation
    pub min_frequency: u64,
    /// The target vocabulary size
    pub vocab_size: usize,
    /// Whether to show progress while training
    pub show_progress: bool,
    /// Progress output format (Indicatif, JsonLines, or Silent)
    ///
    /// Progress display is not serialized; deserialization uses its default.
    /// It does not change training results.
    #[serde(skip)]
    pub progress_format: ProgressFormat,
    /// A list of special tokens that the model should know of
    #[serde(with = "crate::added_token_serde")]
    pub special_tokens: Vec<AddedToken>,
    /// Whether to limit the number of initial tokens that can be kept before computing merges
    pub limit_alphabet: Option<usize>,
    /// Characters prioritized during alphabet selection, including characters
    /// absent from the training input. If `limit_alphabet` is smaller than this
    /// set, some of these characters can still be removed.
    pub initial_alphabet: AHashSet<char>,
    /// An optional prefix to use on any subword that exist only behind another one
    pub continuing_subword_prefix: Option<String>,
    /// An optional suffix to characterize and end-of-word subword
    pub end_of_word_suffix: Option<String>,
    /// An exclusive span limit for newly created adjacent pairs, measured in
    /// retained input characters. Affix text and UTF-8 byte widths do not count.
    /// `None` disables this limit.
    ///
    /// Initial pairs bypass the limit, and selected pairs have no additional
    /// length check. For example, `Some(3)` rejects a newborn pair spanning three
    /// characters, while `Some(1)` still permits an initial two-character merge.
    pub max_token_length: Option<usize>,

    words: WordCounts,
}

impl Default for BpeTrainer {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl BpeTrainer {
    pub fn new(min_frequency: u64, vocab_size: usize) -> Self {
        Self {
            min_frequency,
            vocab_size,
            ..Default::default()
        }
    }

    pub fn builder() -> BpeTrainerBuilder {
        BpeTrainerBuilder::new()
    }

    /// Returns the number of unique words in the corpus after feeding.
    /// This can be used to estimate training time before starting.
    pub fn get_word_count(&self) -> usize {
        self.words.len()
    }

    /// Select the alphabet with the existing frequency-tie and codepoint order.
    fn select_alphabet(&self, wc: WordCountsView<'_>) -> Vec<char> {
        // Compute the alphabet from seen words
        let mut alphabet: AHashMap<char, usize> = AHashMap::new();
        for (word, count) in wc.iter() {
            for c in word.chars() {
                *alphabet.entry(c).or_default() += *count as usize;
            }
        }

        // Also include anything from the provided initial alphabet
        for c in &self.initial_alphabet {
            *alphabet.entry(*c).or_default() = usize::MAX;
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

    /// Train the collected weighted words and return vocabulary entries, ordered
    /// merges, and special tokens.
    ///
    /// Stored counts remain available for subsequent calls. Execution policy and
    /// numeric limits are the same as for [`Self::do_train`]. The WordPiece trainer
    /// uses these parts to reinterpret the vocabulary without building BPE merge tables.
    ///
    /// # Errors
    ///
    /// Returns the training errors described in [`Self::do_train`], including the
    /// signed input limits for nonempty affixes even when no merge is needed.
    pub fn train_vocab(&self) -> Result<(Vocab, Merges, Vec<AddedToken>)> {
        self.train_counts(self.words.view())
    }

    /// The runtime options a trained model is built with.
    ///
    /// The two affixes are the only settings a BPE trainer decides: everything else in
    /// [`BpeConfig`] describes how to *read* a model (unknown-token handling, dropout,
    /// caching) and is the reader's business, not the trainer's, so it stays at its default.
    fn model_options(&self) -> BpeConfig {
        BpeConfig {
            continuing_subword_prefix: self.continuing_subword_prefix.clone(),
            end_of_word_suffix: self.end_of_word_suffix.clone(),
            ..Default::default()
        }
    }

    /// Train weighted words and return vocabulary entries, ordered merges, and
    /// special tokens for registration by the caller.
    ///
    /// These parts populate [`BpeConfig`] for [`PipelineBPE::from_config`]. The
    /// WordPiece trainer consumes the vocabulary without building BPE merge tables.
    /// The input map is borrowed and remains unchanged.
    ///
    /// Training uses a dedicated Rayon pool sized by
    /// [`tk_encode::parallelism::num_threads`], or one worker when parallelism is
    /// disabled. [`Trainer::feed`] uses the ambient pool, including one installed
    /// by the caller.
    ///
    /// # Errors
    ///
    /// Returns an error on overflow or underflow in checked `u64` pair, birth,
    /// or removal arithmetic, or in checked `i64` active-reuse ledger updates.
    /// Nonempty affixes and active reuse also require the maximum word weight and
    /// initial weighted edge mass (the sum of each word's weight times its retained
    /// adjacent-pair count) to fit in `i64::MAX`. The affix check applies even when
    /// the initial vocabulary already meets the target size. Plain first-activation
    /// input has no total-`u64` mass limit when every individual pair count fits.
    ///
    /// Pool creation, progress setup, vocabulary or corpus size bounds, and fallible position
    /// storage operations can also return errors. Feed counting and limited-alphabet
    /// frequency accumulation use ordinary addition rather than these checked rules.
    pub fn do_train(
        &self,
        word_counts: &AHashMap<CompactString, u64>,
    ) -> Result<(Vocab, Merges, Vec<AddedToken>)> {
        self.train_counts(WordCountsView::from_map(word_counts))
    }

    fn train_counts(
        &self,
        word_counts: WordCountsView<'_>,
    ) -> Result<(Vocab, Merges, Vec<AddedToken>)> {
        let workers = if get_parallelism() {
            num_threads().max(1)
        } else {
            1
        };
        engine::train(
            self,
            word_counts,
            workers,
            #[cfg(test)]
            None,
        )
    }
}

impl Trainer for BpeTrainer {
    type Model = PipelineBPE;

    /// Train the collected words and replace the model using the trainer's affixes.
    /// Return special tokens for registration by the caller.
    fn train(&self, model: &mut PipelineBPE) -> Result<Vec<AddedToken>> {
        let (vocab, merges, special_tokens) = self.train_counts(self.words.view())?;
        *model = PipelineBPE::from_config(BpeConfig {
            vocab,
            merges,
            ..self.model_options()
        })?;
        Ok(special_tokens)
    }

    /// Whether we should show progress
    fn should_show_progress(&self) -> bool {
        self.show_progress
    }

    /// Apply `process` to each input and collect the resulting weighted words.
    /// Successful collection replaces the words used by `train` and `train_vocab`.
    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync,
    {
        self.words = feed::count(iterator, &process)?;
        Ok(())
    }
}
