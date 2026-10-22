#![allow(clippy::map_entry)]

mod corpus;
mod index;
mod positions;
use index::PairIndex;
mod feed;
use corpus::CorpusPlan;
mod vocabulary;
mod word_counts;
use vocabulary::Vocabulary;

#[cfg(feature = "parity-aware-bpe")]
pub mod parity_trainer;
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
use word_counts::{WordCounts, WordCountsView};
// The `Word` machinery a trainer merges into is training-only, so it lives here rather than in the
// inference crate. `PipelineBPE` is the only BPE left; a trainer reaches it through
// `from_config`, the same serde-free door a reader walks through, because its fields are
// private to `tk-encode`.
#[cfg(feature = "parity-aware-bpe")]
use word::WithFirstLastIterator;
#[cfg(feature = "parity-aware-bpe")]
use word::Word;

use crate::progress::{ProgressBar, ProgressFormat, ProgressStyle};
use tk_encode::Result;
use tk_encode::models::bpe::{BpeConfig, Merges, PipelineBPE, Vocab};
use tk_encode::parallelism;

/// A `BpeTrainerBuilder` can be used to create a `BpeTrainer` with a custom
/// configuration.
#[derive(Default)]
pub struct BpeTrainerBuilder {
    trainer: BpeTrainer,
}

impl BpeTrainerBuilder {
    /// Constructs a new `BpeTrainerBuilder`
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the expected minimum frequency
    #[must_use]
    pub fn min_frequency(mut self, frequency: u64) -> Self {
        self.trainer.min_frequency = frequency;
        self
    }

    /// Set the vocabulary size
    #[must_use]
    pub fn vocab_size(mut self, size: usize) -> Self {
        self.trainer.vocab_size = size;
        self
    }

    /// Set whether to show progress
    #[must_use]
    pub fn show_progress(mut self, show: bool) -> Self {
        self.trainer.show_progress = show;
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
        self.trainer.progress_format = format;
        self
    }

    /// Set the special tokens
    #[must_use]
    pub fn special_tokens(mut self, tokens: Vec<AddedToken>) -> Self {
        self.trainer.special_tokens = tokens;
        self
    }

    /// Set whether to limit the alphabet
    #[must_use]
    pub fn limit_alphabet(mut self, limit: usize) -> Self {
        self.trainer.limit_alphabet = Some(limit);
        self
    }

    /// Set characters considered even when absent from the input.
    /// A smaller alphabet limit may still remove some of these characters.
    #[must_use]
    pub fn initial_alphabet(mut self, alphabet: HashSet<char>) -> Self {
        let mut initial_alphabet = AHashSet::with_capacity(alphabet.len());
        initial_alphabet.extend(alphabet);
        self.trainer.initial_alphabet = initial_alphabet;
        self
    }

    /// Set the continuing_subword_prefix
    #[must_use]
    pub fn continuing_subword_prefix(mut self, prefix: String) -> Self {
        self.trainer.continuing_subword_prefix = Some(prefix);
        self
    }

    /// Set the end_of_word_suffix
    #[must_use]
    pub fn end_of_word_suffix(mut self, suffix: String) -> Self {
        self.trainer.end_of_word_suffix = Some(suffix);
        self
    }
    /// Set the strict span limit used to admit newly formed neighboring pairs.
    ///
    /// See [`BpeTrainer::max_token_length`] for units and initial-pair behavior.
    #[must_use]
    pub fn max_token_length(mut self, max_token_length: Option<usize>) -> Self {
        self.trainer.max_token_length = max_token_length;
        self
    }

    /// Constructs the final BpeTrainer
    #[must_use]
    pub fn build(self) -> BpeTrainer {
        self.trainer
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
/// # fn main() -> tk_encode::Result<()> {
/// let sequences = vec![ "Hello", "World" ];
///
/// let mut trainer = BpeTrainer::default();
/// trainer.feed(sequences.iter(), |s| Ok(vec![s.to_owned()]))?;
///
/// // `PipelineBPE` has no empty state to train *into* -- it only exists once there is a
/// // vocabulary and a merge list -- so take the parts and build it.
/// let (vocab, merges, special_tokens) = trainer.train_vocab()?;
/// let model = PipelineBPE::from_config(BpeConfig { vocab, merges, ..BpeConfig::default() })?;
/// # Ok(())
/// # }
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
    /// Skipped when (de)serializing, and read back as its `Default`: it only decides how progress
    /// is displayed.
    #[serde(skip)]
    pub progress_format: ProgressFormat,
    /// A list of special tokens that the model should know of
    #[serde(with = "crate::added_token_serde")]
    pub special_tokens: Vec<AddedToken>,
    /// Whether to limit the number of initial tokens that can be kept before computing merges
    pub limit_alphabet: Option<usize>,
    /// Characters to consider even when they are absent from the training set.
    /// A smaller `limit_alphabet` may still remove some of these characters.
    pub initial_alphabet: AHashSet<char>,
    /// An optional prefix to use on any subword that exist only behind another one
    pub continuing_subword_prefix: Option<String>,
    /// An optional suffix to characterize and end-of-word subword
    pub end_of_word_suffix: Option<String>,
    /// Strict upper bound for the combined span of newly formed neighboring pairs.
    ///
    /// Spans count retained input symbols, excluding prefix/suffix text. A new pair
    /// is admitted only when its span is strictly below this limit. Initial pairs
    /// are counted without this filter, so this is not a bound on every token.
    pub max_token_length: Option<usize>,

    words: WordCounts,
}

impl Default for BpeTrainer {
    fn default() -> Self {
        Self {
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
            words: WordCounts::default(),
        }
    }
}

impl BpeTrainer {
    #[must_use]
    pub fn new(min_frequency: u64, vocab_size: usize) -> Self {
        Self {
            min_frequency,
            vocab_size,
            ..Default::default()
        }
    }

    #[must_use]
    pub fn builder() -> BpeTrainerBuilder {
        BpeTrainerBuilder::new()
    }

    /// Returns the number of unique words in the corpus after feeding.
    /// This can be used to estimate training time before starting.
    #[must_use]
    pub fn get_word_count(&self) -> usize {
        self.words.len()
    }

    /// Setup a progress bar if asked to show progress (only for Indicatif format)
    fn setup_progress(&self) -> Option<ProgressBar> {
        if self.show_progress && self.progress_format == ProgressFormat::Indicatif {
            let p = ProgressBar::new(0);
            p.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {msg:<30!} {wide_bar} {pos:<9!}/{len:>9!}")
                    .expect("Invalid progress template"),
            );
            Some(p)
        } else {
            None
        }
    }

    /// Emit JSON progress line to stderr (for JsonLines format)
    fn emit_json_progress(&self, stage: &str, current: usize, total: usize) {
        if self.progress_format == ProgressFormat::JsonLines {
            eprintln!(
                r#"{{"stage":"{}","current":{},"total":{}}}"#,
                stage, current, total
            );
        }
    }

    /// Set the progress bar in the finish state
    fn finalize_progress(&self, p: &Option<ProgressBar>, final_len: usize, stage: &str) {
        if let Some(p) = p {
            p.set_length(final_len as u64);
            p.finish();
            println!();
        }
        self.emit_json_progress(stage, final_len, final_len);
    }

    /// Update the progress bar with the new provided length and message
    fn update_progress(&self, p: &Option<ProgressBar>, len: usize, message: &'static str) {
        if let Some(p) = p {
            p.set_message(message);
            p.set_length(len as u64);
            p.reset();
        }
        // Emit initial JSON progress for this stage
        self.emit_json_progress(message, 0, len);
    }

    /// Train and hand back the raw parts, for a caller that wants them rather than a built model.
    ///
    /// The WordPiece trainer is the one caller: it trains a BPE and reinterprets the vocabulary as
    /// WordPiece pieces, so building a `PipelineBPE` first -- merge tables and all -- would be work
    /// thrown away.
    ///
    /// # Errors
    ///
    /// Returns an error if checked position encoding or storage reservation fails.
    /// Initial and merged token identities must fit the supported ID domain.
    /// Checked corpus layouts and weighted pair counts must fit their numeric domains.
    pub fn train_vocab(&self) -> Result<(Vocab, Merges, Vec<AddedToken>)> {
        self.train_counts(WordCountsView::from(&self.words))
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

    /// Runs the training and returns the vocabulary and merge list it produced, plus the special
    /// tokens the caller has to add alongside them.
    ///
    /// It hands back `(Vocab, Merges)` rather than filling in a model because that pair *is* what a
    /// `tokenizer.json` stores, and `PipelineBPE` can only be built from it -- see
    /// [`PipelineBPE::from_config`]. The WordPiece trainer wants the vocabulary alone, so
    /// splitting the two also saves it building merge tables it would throw away.
    ///
    /// # Errors
    ///
    /// Reports the same failure conditions as [`Self::train_vocab`].
    pub fn do_train(
        &self,
        word_counts: &AHashMap<CompactString, u64>,
    ) -> Result<(Vocab, Merges, Vec<AddedToken>)> {
        self.train_counts(WordCountsView::from(word_counts))
    }

    fn train_counts(
        &self,
        word_counts: WordCountsView<'_>,
    ) -> Result<(Vocab, Merges, Vec<AddedToken>)> {
        let max_token_length: usize = self.max_token_length.unwrap_or(usize::MAX);

        let progress = self.setup_progress();

        let (mut vocabulary, ids) = Vocabulary::initialize(self, word_counts, &progress)?;

        //
        // 3. Tokenize words
        //
        let plan = CorpusPlan::build(word_counts, ids, self.vocab_size)?;
        self.finalize_progress(&progress, plan.word_count(), "Tokenize words");

        //
        // 4. Count pairs in words
        //
        self.update_progress(&progress, plan.word_count(), "Count pairs");
        let mut index = PairIndex::build(&plan, self.min_frequency, &progress)?;
        self.finalize_progress(&progress, plan.word_count(), "Count pairs");

        //
        // 5. Do merges
        //
        let mut corpus = plan.materialize();
        self.update_progress(&progress, self.vocab_size, "Compute merges");
        let mut completed = 0;
        loop {
            // Stop as soon as we have a big enough vocabulary
            if vocabulary.len() >= self.vocab_size {
                break;
            }

            let Some(top) = index.take() else { break };

            let resolved = vocabulary.resolve(top.pair)?;
            let changes = corpus.merge(
                top.pair,
                resolved.merged_id,
                max_token_length,
                &top.positions,
                resolved.already_in_vocabulary,
            );
            index.commit(changes)?;

            vocabulary.commit(top.pair);
            completed += 1;
            if let Some(p) = &progress {
                p.inc(1);
            }
            self.emit_json_progress("Compute merges", completed, self.vocab_size);
        }
        self.finalize_progress(&progress, completed, "Compute merges");

        let (vocab, merges) = vocabulary.into_parts();
        Ok((vocab, merges, self.special_tokens.clone()))
    }
}

impl Trainer for BpeTrainer {
    type Model = PipelineBPE;

    /// Train a BPE model
    fn train(&self, model: &mut PipelineBPE) -> Result<Vec<AddedToken>> {
        let (vocab, merges, special_tokens) =
            self.train_counts(WordCountsView::from(&self.words))?;
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

#[cfg(test)]
mod tests {
    use super::{BpeTrainer, Merges};
    use ahash::AHashMap;
    use compact_str::CompactString;

    #[test]
    fn test_train() {
        let word_counts: AHashMap<CompactString, u64> = [
            ("roses".into(), 1),
            ("are".into(), 2),
            ("red".into(), 1),
            ("voilets".into(), 1),
            ("blue".into(), 1),
            ("BERT".into(), 1),
            ("is".into(), 2),
            ("big".into(), 1),
            ("and".into(), 1),
            ("so".into(), 1),
            ("GPT-2".into(), 1),
        ]
        .iter()
        .cloned()
        .collect();
        let trainer = BpeTrainer::builder()
            .show_progress(false)
            .min_frequency(2)
            .build();
        let (trained_vocab, merges, _special_tokens) = trainer.do_train(&word_counts).unwrap();

        // Vocab should contain all of the characters from the `word_counts` mapping
        // as well as three merges: 're', 'are', and 'is'.
        let expected_vocab: AHashMap<String, u32> = [
            ("-".into(), 0),
            ("2".into(), 1),
            ("B".into(), 2),
            ("E".into(), 3),
            ("G".into(), 4),
            ("P".into(), 5),
            ("R".into(), 6),
            ("T".into(), 7),
            ("a".into(), 8),
            ("b".into(), 9),
            ("d".into(), 10),
            ("e".into(), 11),
            ("g".into(), 12),
            ("i".into(), 13),
            ("l".into(), 14),
            ("n".into(), 15),
            ("o".into(), 16),
            ("r".into(), 17),
            ("s".into(), 18),
            ("t".into(), 19),
            ("u".into(), 20),
            ("v".into(), 21),
            ("re".into(), 22),
            ("are".into(), 23),
            ("is".into(), 24),
        ]
        .iter()
        .cloned()
        .collect();
        assert_eq!(trained_vocab, expected_vocab);

        // `merges` is the pair of symbol *strings* per merge, highest priority first -- the on-disk
        // form, and what `PipelineBPE::from_config` re-derives its ranks from. Position in
        // the list is the rank, so the order is part of what is being asserted.
        let expected_merges: Merges = vec![
            ("r".into(), "e".into()),  // 'r' + 'e'  -> 're'
            ("a".into(), "re".into()), // 'a' + 're' -> 'are'
            ("i".into(), "s".into()),  // 'i' + 's'  -> 'is'
        ];
        assert_eq!(merges, expected_merges);
    }
    #[test]
    fn bpe_test_max_token_length_16() {
        /* bpe_test_max_token_length series of tests test the max_token_length flag of bpetrainer
        // this is the more robust version that only tests max length of learned tokens
        // (pre) tokenizer settings or vocab can be easily modified when necessary
         */

        let max_token_length = 16;
        let long_word_counts: AHashMap<CompactString, u64> = [
            ("singlelongtokenwithoutcasechange", 2),
            ("singleLongTokenWithCamelCaseChange", 2),
            ("Longsingletokenwithpunctu@t!onwithin", 2),
            ("Anotherlongsingletokenwithnumberw1th1n", 2),
            ("짧은한글문자열짧은한", 2),             // korean 10 char
            ("긴한글문자열긴한글문자열긴한글문", 2), // korean 16 char
            ("短字符串短字符串短字", 2),             //simplified chinese 10 char
            ("长字符串长字符串长字符串长字符串", 2), // simp. chinese 16 char
            ("短い文字列短い文字列", 2),             // japanese 10 char
            ("長い文字列長い文字列長い文字列長", 2), // japanese 16 char
            ("so", 2),
            ("GPT-2", 2),
        ]
        .iter()
        .map(|(key, value)| (CompactString::from(key.to_string()), *value))
        .collect();
        let trainer = BpeTrainer::builder()
            .max_token_length(Some(max_token_length))
            .show_progress(false)
            .min_frequency(0)
            .build();
        let (vocab, _merges, _special_tokens) = trainer.do_train(&long_word_counts).unwrap();
        for token in vocab.keys() {
            assert!(
                token.chars().count() <= max_token_length,
                "token too long : {} , chars().count() = {}",
                token,
                token.chars().count()
            )
        }
    }
    #[test]
    fn bpe_test_max_token_length_direct_assert() {
        /* more direct version of bpe_test_max_token_length test
        // directly compares tokens with known expected values.
        // maybe unstable depending on specific settings or changes.
         */
        let long_word_counts: AHashMap<CompactString, u64> = [
            ("sin", 2),
            ("Sin", 2),
            ("Lon", 2),
            ("Ano", 2),
            ("짧은한", 2),
            ("긴한글", 2),
            ("短字符", 2),
            ("长字符", 2),
            ("短い文", 2),
            ("長い文", 2),
            ("so", 2),
            ("GP", 2),
        ]
        .iter()
        .map(|(key, value)| (CompactString::from(key.to_string()), *value))
        .collect();
        let trainer = BpeTrainer::builder()
            .max_token_length(Some(2))
            .show_progress(false)
            .min_frequency(0)
            .build();
        let (trained_vocab, _merges, _special_tokens) =
            trainer.do_train(&long_word_counts).unwrap();
        let expected_vocab: AHashMap<String, u32> = [
            ("短", 12),
            ("n", 6),
            ("i", 5),
            ("s", 8),
            ("字符", 23),
            ("長", 14),
            ("긴", 17),
            ("い文", 22),
            ("L", 2),
            ("in", 21),
            ("o", 7),
            ("은한", 29),
            ("S", 4),
            ("P", 3),
            ("so", 27),
            ("符", 13),
            ("文", 11),
            ("字", 10),
            ("짧", 19),
            ("GP", 25),
            ("글", 16),
            ("G", 1),
            ("An", 24),
            ("长", 15),
            ("A", 0),
            ("Lo", 26),
            ("긴한", 28),
            ("い", 9),
            ("한", 20),
            ("은", 18),
        ]
        .iter()
        .cloned()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        assert_eq!(trained_vocab, expected_vocab)
    }
}

fn num_workers() -> usize {
    if parallelism::get_parallelism() {
        parallelism::current_num_threads()
    } else {
        1
    }
}

#[cfg(test)]
mod identity_tests {
    use super::*;

    #[test]
    fn active_alias_keeps_historical_merge_order() {
        let words = [("aba".into(), 7), ("bbbbbbbaba".into(), 6)].into();
        let trainer = BpeTrainer::builder()
            .show_progress(false)
            .vocab_size(64)
            .min_frequency(1)
            .continuing_subword_prefix("ab".into())
            .build();
        let (_, merges, _) = trainer.do_train(&words).unwrap();
        assert_eq!(
            merges,
            [
                ("abb".into(), "abb".into()),
                ("abb".into(), "aba".into()),
                ("abbb".into(), "abbb".into()),
                ("a".into(), "abba".into()),
                ("b".into(), "abbbbb".into()),
                ("aba".into(), "abba".into()),
                ("abbb".into(), "ababa".into()),
                ("bbbbb".into(), "abbbaba".into()),
            ]
        );
    }
}

#[cfg(test)]
mod builder_tests {
    use super::*;
    #[test]
    fn builder_serialization() {
        let default = BpeTrainer::default();
        let json = serde_json::json!({
            "min_frequency": 0, "vocab_size": 30000, "show_progress": true,
            "special_tokens": [], "limit_alphabet": null, "initial_alphabet": [],
            "continuing_subword_prefix": null, "end_of_word_suffix": null,
            "max_token_length": null, "words": {}
        });
        assert_eq!(serde_json::to_value(&default).unwrap(), json);
        assert_eq!(serde_json::from_value::<BpeTrainer>(json).unwrap(), default);
        let configured = BpeTrainer::builder()
            .progress_format(ProgressFormat::JsonLines)
            .build();
        let json = serde_json::to_value(&configured).unwrap();
        assert!(json.get("progress_format").is_none());
        assert_eq!(serde_json::from_value::<BpeTrainer>(json).unwrap(), default);
    }
}

#[cfg(feature = "parity-aware-bpe")]
use tk_encode::models::bpe::Pair;

#[cfg(test)]
mod historical_coordinate_tests {
    use super::*;

    #[test]
    fn stale_coordinate_keeps_its_historical_word_domain() {
        let words = [("cbbbbb".into(), 6), ("aba".into(), 4)].into();
        let trainer = BpeTrainer::builder()
            .show_progress(false)
            .vocab_size(64)
            .min_frequency(1)
            .continuing_subword_prefix("ab".into())
            .build();
        let (_, merges, _) = trainer.do_train(&words).unwrap();
        assert_eq!(
            merges,
            [
                ("abb".into(), "abb".into()),
                ("c".into(), "abbb".into()),
                ("abbb".into(), "abb".into()),
                ("cbb".into(), "abbbb".into()),
                ("a".into(), "abb".into()),
                ("ab".into(), "aba".into()),
            ]
        );
    }
}

fn add(count: &mut u64, amount: u64) -> Result<()> {
    *count = count
        .checked_add(amount)
        .ok_or("BPE weighted frequency exceeds u64")?;
    Ok(())
}

#[cfg(test)]
mod weighted_count_tests {
    use super::*;
    #[test]
    fn wide_frequencies_and_signed_guard() {
        let trainer = BpeTrainer::builder()
            .vocab_size(8)
            .min_frequency(1)
            .show_progress(false)
            .build();
        let words = [("abab".into(), u32::MAX as u64)].into();
        let (_, merges, _) = trainer.do_train(&words).unwrap();
        assert_eq!(
            merges,
            [("a".into(), "b".into()), ("ab".into(), "ab".into())]
        );
        let words = [("abab".into(), i64::MAX as u64)].into();
        assert!(trainer.do_train(&words).is_err());
    }
}

#[cfg(test)]
mod local_edit_tests {
    use super::*;
    #[test]
    fn weighted_adjacent_edits_keep_literal_merge_order() {
        let trainer = BpeTrainer::builder()
            .show_progress(false)
            .min_frequency(1)
            .vocab_size(5)
            .build();
        let words = [("abab".into(), 10), ("ababa".into(), 1)].into();
        let (_, merges, _) = trainer.do_train(&words).unwrap();
        assert_eq!(
            merges,
            [
                ("a".into(), "b".into()),
                ("ab".into(), "ab".into()),
                ("abab".into(), "a".into())
            ]
        );
    }
}
