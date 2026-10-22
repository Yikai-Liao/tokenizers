#![allow(clippy::map_entry)]

mod corpus;
mod index;
mod positions;
use index::PairIndex;
mod feed;
use corpus::Corpus;
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

        let (mut vocabulary, ids) = Vocabulary::initialize(self, word_counts, &progress);

        //
        // 3. Tokenize words
        //
        let mut corpus = Corpus::new(word_counts, &ids);
        self.finalize_progress(&progress, corpus.word_count(), "Tokenize words");
        drop(ids);

        //
        // 4. Count pairs in words
        //
        self.update_progress(&progress, corpus.word_count(), "Count pairs");
        let mut index = PairIndex::build(&corpus, self.min_frequency, &progress)?;
        self.finalize_progress(&progress, corpus.word_count(), "Count pairs");

        //
        // 5. Do merges
        //
        self.update_progress(&progress, self.vocab_size, "Compute merges");
        let mut completed = 0;
        loop {
            // Stop as soon as we have a big enough vocabulary
            if vocabulary.len() >= self.vocab_size {
                break;
            }

            let Some(top) = index.take() else { break };

            let resolved = vocabulary.resolve(top.pair);
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
        assert_eq!(configured.progress_format, ProgressFormat::JsonLines);
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

#[cfg(test)]
mod trainer_contract_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn counts_survive_flushes_and_repeated_callbacks() {
        for unique in [0, 2047, 2048, 2049, 4097] {
            let mut batch = vec!["repeat".to_owned()];
            batch.extend((0..unique).map(|i| format!("词{i}🙂")));
            batch.push("repeat".into());
            let expected: AHashMap<CompactString, u64> = batch
                .iter()
                .map(|s| (s.clone().into(), if s == "repeat" { 3 } else { 2 }))
                .collect();
            batch[0] = batch.join(" ");
            let actual = feed::count(batch.iter(), &|s| {
                Ok(s.split(' ').map(String::from).collect())
            });
            assert_eq!(actual.unwrap(), WordCounts::from(expected));
        }
    }

    #[test]
    fn equality_and_serde() {
        let original = WordCounts::from_unique(vec![("zero".into(), 0), ("wide".into(), u64::MAX)]);
        let map = AHashMap::from([("wide".into(), u64::MAX), ("zero".into(), 0)]);
        assert_eq!(original, WordCounts::from(map));
        for json in [
            r#"{"zero":1,"wide":18446744073709551615}"#,
            r#"{"other":0,"wide":18446744073709551615}"#,
            r#"{"zero":0}"#,
        ] {
            assert_ne!(original, serde_json::from_str::<WordCounts>(json).unwrap());
        }
        let json = serde_json::json!({"zero": 0, "wide": u64::MAX});
        assert_eq!(serde_json::to_value(&original).unwrap(), json);
        assert_eq!(
            serde_json::from_value::<WordCounts>(json).unwrap(),
            original
        );
        let duplicate: WordCounts = serde_json::from_str(r#"{"word":1,"word":9}"#).unwrap();
        assert_eq!(duplicate, WordCounts::from_unique(vec![("word".into(), 9)]));
        assert!(serde_json::from_str::<WordCounts>(r#"{"word":-1}"#).is_err());
    }

    #[test]
    fn training_replaces_model_and_preserves_fed_counts() {
        let mut trainer = BpeTrainer::builder()
            .show_progress(false)
            .min_frequency(1)
            .vocab_size(5)
            .special_tokens(vec![AddedToken::from("!", true)])
            .continuing_subword_prefix("##".into())
            .end_of_word_suffix("$".into())
            .build();
        trainer
            .feed(["ab", "ab"].into_iter(), |s| Ok(vec![s.into()]))
            .unwrap();
        let previous = trainer.clone();
        let called = AtomicUsize::new(0);
        let result = trainer.feed(["ok", "error", "tail"].into_iter(), |text| {
            called.fetch_add(1, Ordering::Relaxed);
            (text != "error")
                .then(|| vec![text.into()])
                .ok_or("expected".into())
        });
        assert!(result.is_err());
        assert_eq!(called.load(Ordering::Relaxed), 3);
        assert_eq!(trainer, previous);
        let mut model = PipelineBPE::from_config(BpeConfig {
            vocab: [("z".into(), 0)].into(),
            ..Default::default()
        })
        .unwrap();
        let expected = ["!", "a", "b", "##b$", "ab$"]
            .into_iter()
            .enumerate()
            .map(|(id, token)| (token.to_owned(), id as u32))
            .collect();
        for _ in 0..2 {
            let special = trainer.train(&mut model).unwrap();
            assert_eq!(special, [AddedToken::from("!", true)]);
            let config = model.to_config().unwrap();
            assert_eq!(config.vocab, expected);
            assert_eq!(config.merges, [("a".into(), "##b$".into())]);
            assert_eq!(config.continuing_subword_prefix.as_deref(), Some("##"));
            assert_eq!(config.end_of_word_suffix.as_deref(), Some("$"));
            let trained = (config.vocab, config.merges, special);
            assert_eq!(trainer.train_vocab().unwrap(), trained);
            assert_eq!(trainer.get_word_count(), 1);
        }
        trainer
            .feed(std::iter::empty::<&str>(), |_| unreachable!())
            .unwrap();
        assert_eq!(trainer.get_word_count(), 0);
    }

    #[test]
    fn progress_and_alphabet_options_survive_builder_configuration() {
        let trainer = BpeTrainer::builder()
            .show_progress(false)
            .initial_alphabet(['é', '🙂'].into())
            .limit_alphabet(2)
            .vocab_size(0)
            .build();
        assert!(!trainer.should_show_progress());
        assert!(BpeTrainer::default().should_show_progress());
        let (vocab, merges, _) = trainer.do_train(&[("abé".into(), 10)].into()).unwrap();
        assert_eq!(vocab, [("é".into(), 0), ("🙂".into(), 1)].into());
        assert!(merges.is_empty());
    }

    #[test]
    fn observed_special_characters_survive_alphabet_filtering() {
        let trainer = BpeTrainer::builder()
            .show_progress(false)
            .min_frequency(1)
            .vocab_size(5)
            .initial_alphabet(['b'].into())
            .limit_alphabet(1)
            .special_tokens(vec![
                AddedToken::from("a", true),
                AddedToken::from("xy", true),
                AddedToken::from("z", true),
            ])
            .build();
        let words = [("ab".into(), 3), ("xy".into(), 100)].into();
        let (vocab, merges, _) = trainer.do_train(&words).unwrap();
        assert_eq!(
            vocab,
            [
                ("a".into(), 0),
                ("xy".into(), 1),
                ("z".into(), 2),
                ("b".into(), 3),
                ("ab".into(), 4)
            ]
            .into()
        );
        assert_eq!(merges, [("a".into(), "b".into())]);
    }

    #[test]
    fn weighted_alphabet_selection_preserves_wide_frequencies() {
        let trainer = BpeTrainer::builder()
            .show_progress(false)
            .vocab_size(0)
            .limit_alphabet(1)
            .build();
        for (input, expected) in [
            (vec![("aaa", 1), ("b", 4)], "b"),
            (
                vec![
                    ("ax", i64::MAX as u64),
                    ("ay", i64::MAX as u64),
                    ("az", i64::MAX as u64),
                    ("b", i64::MAX as u64),
                ],
                "a",
            ),
        ] {
            let words = input
                .into_iter()
                .map(|(word, weight)| (CompactString::from(word), weight))
                .collect();
            let (vocab, merges, _) = trainer.do_train(&words).unwrap();
            assert_eq!(vocab, [(expected.into(), 0)].into());
            assert!(merges.is_empty());
        }
    }
}

#[cfg(test)]
mod reference_tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    // Recount every adjacent pair after each left-to-right rewrite. This small
    // reference has no queue, historical positions, shards, or batched writes.
    // Inputs here have plain symbols, bounded weights, and no special tokens.
    fn reference(words: &AHashMap<CompactString, u64>, trainer: &BpeTrainer) -> (Vocab, Merges) {
        let alphabet: BTreeSet<_> = words.keys().flat_map(|word| word.chars()).collect();
        let mut tokens: Vec<_> = alphabet.into_iter().map(|c| c.to_string()).collect();
        let initial = tokens.len();
        let mut corpus: Vec<_> = words
            .iter()
            .map(|(word, &weight)| {
                let ids: Vec<_> = word
                    .chars()
                    .map(|c| tokens.iter().position(|s| s == &c.to_string()).unwrap() as u32)
                    .collect();
                (ids, weight)
            })
            .collect();
        let mut merges = Vec::new();
        while tokens.len() < trainer.vocab_size {
            let mut counts = BTreeMap::<(u32, u32), u64>::new();
            for (word, weight) in &corpus {
                for edge in word.windows(2) {
                    let length = tokens[edge[0] as usize].chars().count()
                        + tokens[edge[1] as usize].chars().count();
                    let original_pair =
                        (edge[0] as usize) < initial && (edge[1] as usize) < initial;
                    if original_pair || trainer.max_token_length.is_none_or(|limit| length < limit)
                    {
                        *counts.entry((edge[0], edge[1])).or_default() += weight;
                    }
                }
            }
            let Some((pair, _)) = counts
                .into_iter()
                .filter(|(_, count)| *count >= trainer.min_frequency.max(1))
                .max_by_key(|&(pair, count)| (count, std::cmp::Reverse(pair)))
            else {
                break;
            };
            let left = tokens[pair.0 as usize].clone();
            let right = tokens[pair.1 as usize].clone();
            let merged = left.clone() + &right;
            let id = match tokens.iter().position(|s| s == &merged) {
                Some(id) => id as u32,
                None => {
                    tokens.push(merged);
                    (tokens.len() - 1) as u32
                }
            };
            merges.push((left, right));
            for (word, _) in &mut corpus {
                let mut output = Vec::new();
                let mut index = 0;
                while index < word.len() {
                    if word[index..].starts_with(&[pair.0, pair.1]) {
                        output.push(id);
                        index += 2;
                    } else {
                        output.push(word[index]);
                        index += 1;
                    }
                }
                *word = output;
            }
        }
        (
            tokens
                .into_iter()
                .enumerate()
                .map(|(id, text)| (text, id as u32))
                .collect(),
            merges,
        )
    }

    #[test]
    fn plain_training_matches_recounting_reference() {
        let mut state = 0x4632_9ab1u32;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            state
        };
        let characters = ['a', 'b', 'é', '字', '🙂'];
        for case in 0..96 {
            let mut words = AHashMap::<CompactString, u64>::new();
            let domain = 2 + case % 4;
            for _ in 0..2 + case % 7 {
                let length = 1 + next() as usize % 12;
                let text: String = (0..length)
                    .map(|_| characters[next() as usize % domain])
                    .collect();
                *words.entry(text.into()).or_default() += u64::from(next() % 8);
            }
            if case == 0 {
                words.clear();
            } else if case % 8 == 0 {
                words.insert("".into(), 0);
            }
            let trainer = BpeTrainer::builder()
                .show_progress(false)
                .vocab_size([0, 5, 9, 20][case % 4])
                .min_frequency([0, 2, 7][case % 3])
                .max_token_length([None, Some(1), Some(3), Some(6)][case / 4 % 4])
                .build();
            let expected = reference(&words, &trainer);
            let (vocab, merges, _) = trainer.do_train(&words).unwrap();
            assert_eq!((vocab, merges), expected, "case {case}: {words:?}");
        }
        // Four rules are compatible, but only two new vocabulary IDs fit.
        let words = [
            ("ab".into(), 8),
            ("cd".into(), 7),
            ("ef".into(), 6),
            ("gh".into(), 5),
        ]
        .into();
        let trainer = BpeTrainer::builder()
            .show_progress(false)
            .min_frequency(1)
            .vocab_size(10)
            .build();
        let (vocab, merges, _) = trainer.do_train(&words).unwrap();
        assert_eq!(vocab.len(), 10);
        assert_eq!((vocab, merges), reference(&words, &trainer));
    }

    #[test]
    fn large_alphabets_match_recounting_reference() {
        for domain in [256, 257] {
            let text: String = (0..domain)
                .flat_map(|offset| {
                    let character = char::from_u32(0x100 + offset).unwrap();
                    [character, character]
                })
                .collect();
            let words = [(CompactString::from(text), 1)].into();
            let trainer = BpeTrainer::builder()
                .show_progress(false)
                .min_frequency(1)
                .vocab_size(domain as usize + 3)
                .build();
            let expected = reference(&words, &trainer);
            let (vocab, merges, _) = trainer.do_train(&words).unwrap();
            assert_eq!((vocab, merges), expected, "alphabet size {domain}");
        }
    }
}
