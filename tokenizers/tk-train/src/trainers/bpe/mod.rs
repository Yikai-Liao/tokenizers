#![allow(clippy::map_entry)]

mod feed;
mod vocabulary;
mod word_counts;
use vocabulary::Vocabulary;

#[cfg(feature = "parity-aware-bpe")]
pub mod parity_trainer;
mod word;
#[cfg(feature = "parity-aware-bpe")]
pub use parity_trainer::{ParityBpeTrainer, ParityBpeTrainerBuilder, ParityVariant};

use crate::Trainer;
use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use dary_heap::OctonaryHeap;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashSet;
use tk_encode::vocab::bucket_added_vocabulary::AddedToken;
use word_counts::{WordCounts, WordCountsView};
// The `Word` machinery a trainer merges into is training-only, so it lives here rather than in the
// inference crate. `PipelineBPE` is the only BPE left; a trainer reaches it through
// `from_config`, the same serde-free door a reader walks through, because its fields are
// private to `tk-encode`.
use word::{WithFirstLastIterator, Word};

use crate::progress::{ProgressBar, ProgressFormat, ProgressStyle};
use tk_encode::Result;
use tk_encode::models::bpe::{BpeConfig, Merges, Pair, PipelineBPE, Vocab};
use tk_encode::parallelism;
use tk_encode::parallelism::MaybeParallelRefIterator;

#[derive(Debug, Eq)]
struct Merge {
    pair: Pair,
    count: u64,
    pos: AHashSet<usize>,
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
        if self.count != other.count {
            self.count.cmp(&other.count)
        } else {
            // Here we want ascending order
            other.pair.cmp(&self.pair)
        }
    }
}

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

    /// Add the provided special tokens to the initial vocabulary
    fn add_special_tokens(&self, vocabulary: &mut Vocabulary) {
        for token in &self.special_tokens {
            vocabulary.intern(token.content.as_str().into());
        }
    }

    /// Compute the initial alphabet and limit it if relevant
    fn compute_alphabet(&self, wc: WordCountsView<'_>, vocabulary: &mut Vocabulary) {
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
        for (character, _) in kept {
            vocabulary.intern(character.to_string().into());
        }
    }

    /// Tokenize words and add subwords to the vocabulary when relevant
    fn tokenize_words(
        &self,
        wc: WordCountsView<'_>,
        vocabulary: &mut Vocabulary,
        p: &Option<ProgressBar>,
    ) -> (Vec<Word>, Vec<u64>) {
        let mut words: Vec<Word> = Vec::with_capacity(wc.len());
        let mut counts: Vec<u64> = Vec::with_capacity(wc.len());

        for (word, count) in wc.iter() {
            let mut current_word = Word::new();
            counts.push(*count);

            for (is_first, is_last, c) in word.chars().with_first_and_last() {
                let mut s = c.to_string();
                if vocabulary.contains(s.as_str()) {
                    // Found the initial char in the authorized alphabet

                    // Add the `continuing_subword_prefix` if relevant
                    if !is_first && let Some(prefix) = &self.continuing_subword_prefix {
                        s.insert_str(0, prefix);
                    }
                    // Add the `end_of_word_suffix` if relevant
                    if is_last && let Some(suffix) = &self.end_of_word_suffix {
                        s.push_str(suffix);
                    }

                    // Insert the new formed string if necessary
                    let id = vocabulary.intern(s.into());
                    current_word.add(id, 1); // Training spans count retained symbols.
                }
            }
            words.push(current_word);

            if let Some(p) = p {
                p.inc(1);
            }
        }

        (words, counts)
    }

    fn count_pairs(
        &self,
        words: &[Word],
        counts: &[u64],
        p: &Option<ProgressBar>,
    ) -> (AHashMap<Pair, i32>, AHashMap<Pair, AHashSet<usize>>) {
        words
            .maybe_par_iter()
            .enumerate()
            .map(|(i, word)| {
                let mut pair_counts = AHashMap::new();
                let mut where_to_update: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();

                for window in word.get_chars().windows(2) {
                    let cur_pair: Pair = (window[0], window[1]);

                    // Initialize pair_counts and where_to_update for this pair if we just saw it
                    // Then update counts
                    *pair_counts.entry(cur_pair).or_default() += counts[i] as i32;
                    where_to_update.entry(cur_pair).or_default().insert(i);
                }

                if let Some(p) = &p {
                    p.inc(1);
                }

                (pair_counts, where_to_update)
            })
            .reduce(
                || (AHashMap::new(), AHashMap::new()),
                |(mut pair_counts, mut where_to_update), (pc, wtu)| {
                    for (k, v) in pc {
                        *pair_counts.entry(k).or_default() += v;
                    }
                    for (k, v) in wtu {
                        where_to_update.entry(k).or_default().extend(v);
                    }
                    (pair_counts, where_to_update)
                },
            )
    }

    /// Train and hand back the raw parts, for a caller that wants them rather than a built model.
    ///
    /// The WordPiece trainer is the one caller: it trains a BPE and reinterprets the vocabulary as
    /// WordPiece pieces, so building a `PipelineBPE` first -- merge tables and all -- would be work
    /// thrown away.
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
        let mut vocabulary =
            Vocabulary::new(self.vocab_size, self.continuing_subword_prefix.clone());
        let max_token_length: usize = self.max_token_length.unwrap_or(usize::MAX);

        let progress = self.setup_progress();

        //
        // 1. Add all special tokens to the vocabulary
        //
        self.add_special_tokens(&mut vocabulary);

        //
        // 2. Compute the initial alphabet
        //
        self.compute_alphabet(word_counts, &mut vocabulary);

        //
        // 3. Tokenize words
        //
        self.update_progress(&progress, word_counts.len(), "Tokenize words");
        let (mut words, counts) = self.tokenize_words(word_counts, &mut vocabulary, &progress);
        self.finalize_progress(&progress, words.len(), "Tokenize words");

        //
        // 4. Count pairs in words
        //
        self.update_progress(&progress, words.len(), "Count pairs");
        let (mut pair_counts, mut where_to_update) = self.count_pairs(&words, &counts, &progress);
        // Insert them in the queue
        let mut queue = OctonaryHeap::with_capacity(pair_counts.len());
        where_to_update.drain().for_each(|(pair, pos)| {
            let count = pair_counts[&pair];
            if count > 0 {
                queue.push(Merge {
                    pair,
                    count: count as u64,
                    pos,
                });
            }
        });
        self.finalize_progress(&progress, words.len(), "Count pairs");

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

            let Some(mut top) = queue.pop() else {
                break;
            };

            if top.count != pair_counts[&top.pair] as u64 {
                top.count = pair_counts[&top.pair] as u64;
                queue.push(top);
                continue;
            }

            if top.count < 1 || self.min_frequency > top.count {
                break;
            }

            let new_token_id = vocabulary.resolve(top.pair);
            // Merge the new pair in every words
            // Safety: This is just a type assertion, the code below may no longer be safe
            // if the type of `pos` changes
            let pos: &AHashSet<usize> = &top.pos;

            let words_len = words.len();
            // FIXME: doesn't look great
            struct WordPtr(*mut Word);
            // Safety: We do not actually use this for concurrent access to the same memory,
            // only to different chunks within the same allocation.
            unsafe impl Sync for WordPtr {}
            let word_start = WordPtr(words.as_mut_ptr());

            let changes = pos
                .maybe_par_iter()
                .flat_map(|&i| {
                    // We can merge each of these words in parallel here because each position
                    // can be there only once (AHashSet). So this is safe.
                    unsafe {
                        // Edition ≥2021 closures capture the `.0` field (a non-Sync raw
                        // pointer) unless we force whole-struct capture of the Sync wrapper.
                        let word_start = &word_start;
                        assert!(i < words_len);
                        // This is words[i], but avoids needing to go through &T (which triggers UB)
                        let word = word_start.0.add(i);
                        // let word: &mut Word = &mut (*word);
                        (*word)
                            .merge(top.pair.0, top.pair.1, new_token_id, max_token_length)
                            .into_iter()
                            .map(|c| (c, i))
                            .collect::<Vec<_>>()
                    }
                })
                .collect::<Vec<_>>();

            // Introduce new formed pairs
            for ((pair, change), iw) in changes {
                let count = change * counts[iw] as i32;
                *pair_counts.entry(pair).or_default() += count;
                if change > 0 {
                    where_to_update.entry(pair).or_default().insert(iw);
                }
            }
            where_to_update.drain().for_each(|(pair, pos)| {
                let count = pair_counts[&pair];
                if count > 0 {
                    queue.push(Merge {
                        pair,
                        count: count as u64,
                        pos,
                    });
                }
            });

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

#[cfg(test)]
mod trainer_contract_tests {
    use super::*;

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
        let mut model = PipelineBPE::from_config(BpeConfig {
            vocab: [("z".into(), 0)].into(),
            ..Default::default()
        })
        .unwrap();
        for _ in 0..2 {
            let special = trainer.train(&mut model).unwrap();
            assert_eq!(special, [AddedToken::from("!", true)]);
            let config = model.to_config().unwrap();
            assert_eq!(
                config.vocab,
                [
                    ("!".into(), 0),
                    ("a".into(), 1),
                    ("b".into(), 2),
                    ("##b$".into(), 3),
                    ("ab$".into(), 4)
                ]
                .into()
            );
            assert_eq!(config.merges, [("a".into(), "##b$".into())]);
            assert_eq!(config.continuing_subword_prefix.as_deref(), Some("##"));
            assert_eq!(config.end_of_word_suffix.as_deref(), Some("$"));
            assert_eq!(
                trainer.train_vocab().unwrap(),
                (config.vocab, config.merges, special)
            );
            assert_eq!(trainer.get_word_count(), 1);
        }
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
}
