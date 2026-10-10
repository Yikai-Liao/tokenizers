#![allow(clippy::map_entry)]

mod corpus;
mod feed;
mod index;
mod merge;
#[cfg(feature = "parity-aware-bpe")]
pub mod parity_trainer;
mod positions;
#[cfg(test)]
mod tests;
mod vocabulary;
#[cfg(feature = "parity-aware-bpe")]
mod word;
mod word_counts;
#[cfg(feature = "parity-aware-bpe")]
pub use parity_trainer::{ParityBpeTrainer, ParityBpeTrainerBuilder, ParityVariant};

use crate::Trainer;
use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use corpus::CorpusPlan;
use index::PairIndex;
use merge::{Batch, Selection};
use positions::Codec;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use tk_encode::vocab::bucket_added_vocabulary::AddedToken;
use vocabulary::Vocabulary;
use word_counts::{WordCounts, WordCountsView};

// The optional parity trainer uses linked words; ordinary training owns a
// fixed-coordinate corpus. The test oracle uses independent sequential vectors.
#[cfg(feature = "parity-aware-bpe")]
use word::{WithFirstLastIterator, Word};

use tk_encode::Result;

#[cfg(any(test, feature = "parity-aware-bpe"))]
use tk_encode::models::bpe::Pair;
use tk_encode::models::bpe::{BpeConfig, Merges, PipelineBPE, Vocab};
use tk_encode::parallelism::*;
use tk_encode::utils::progress::{ProgressBar, ProgressFormat, ProgressStyle};

/// A `BpeTrainerBuilder` can be used to create a `BpeTrainer` with a custom
/// configuration.
#[derive(Default)]
pub struct BpeTrainerBuilder {
    trainer: BpeTrainer,
}

impl BpeTrainerBuilder {
    /// Constructs a new `BpeTrainerBuilder`
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

    /// Set the initial alphabet. See [`BpeTrainer::initial_alphabet`] for truncation.
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

    /// Set the exclusive span limit for newly created adjacent pairs.
    ///
    /// See [`BpeTrainer::max_token_length`] for units and the initial-pair exception.
    #[must_use]
    pub fn max_token_length(mut self, max_token_length: Option<usize>) -> Self {
        self.trainer.max_token_length = max_token_length;
        self
    }

    /// Constructs the final BpeTrainer
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
        Self {
            min_frequency: 0,
            vocab_size: 30000,
            show_progress: true,
            progress_format: ProgressFormat::default(),
            special_tokens: Vec::new(),
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
    /// or removal arithmetic, or in checked `i64` identity-reuse ledger updates.
    /// A merge reuses an identity when its result string resolves to an ID already
    /// activated as an input symbol or an earlier merge result. This includes IDs
    /// whose last occurrence has since disappeared. A new ID or an unactivated
    /// reserved ID is a first activation.
    ///
    /// Nonempty affixes and identity reuse require the maximum word weight and
    /// initial weighted edge mass (the sum of each word's weight times its retained
    /// adjacent-pair count) to fit in `i64::MAX`. The affix check applies even when
    /// the initial vocabulary already meets the target size. Without nonempty
    /// affixes or identity reuse, there is no total-`u64` mass limit when every
    /// individual pair count fits.
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
        train(
            self,
            word_counts,
            workers,
            #[cfg(test)]
            None,
        )
    }
}

fn add(count: &mut u64, amount: u64) -> Result<()> {
    *count = count
        .checked_add(amount)
        .ok_or("BPE weighted frequency exceeds u64")?;
    Ok(())
}

const WORD_SEPARATOR_ID: u32 = u32::MAX;
type ModelParts = (Vocab, Merges, Vec<AddedToken>);

enum AttemptOutcome {
    Complete(ModelParts),
    RestartForReuse,
}

#[cfg(test)]
type Trace = Vec<(Pair, u64, u32)>;

fn train(
    trainer: &BpeTrainer,
    words: WordCountsView<'_>,
    workers: usize,
    #[cfg(test)] mut observe: Option<&mut (dyn FnMut(Pair, u64, u32) + Send)>,
) -> Result<ModelParts> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()?;
    pool.install(|| {
        let progress = trainer.setup_progress();
        let mut alphabet = None;
        let mut reuse = false;
        loop {
            match train_attempt(
                trainer,
                words,
                workers,
                reuse,
                &mut alphabet,
                &progress,
                #[cfg(test)]
                &mut observe,
            )? {
                AttemptOutcome::Complete(parts) => return Ok(parts),
                // An active-ID collision discards the fresh attempt. Retain the
                // selected alphabet so frequency ties cannot change on the retry.
                AttemptOutcome::RestartForReuse => reuse = true,
            }
        }
    })
}

// Only a completed attempt publishes model parts and its test trace;
// fresh attempts that request historical cohorts stay private.
fn train_attempt(
    trainer: &BpeTrainer,
    words: WordCountsView<'_>,
    workers: usize,
    reuse: bool,
    alphabet: &mut Option<Vec<char>>,
    progress: &Option<ProgressBar>,
    #[cfg(test)] observe: &mut Option<&mut (dyn FnMut(Pair, u64, u32) + Send)>,
) -> Result<AttemptOutcome> {
    // 1. Resolve the vocabulary and tokenize borrowed input into a corpus plan.
    let mut vocabulary = Vocabulary::initialize(trainer, words, workers, alphabet)?;
    trainer.update_progress(progress, words.len(), "Tokenize words");
    let plan = CorpusPlan::build(words, &mut vocabulary, trainer, reuse, progress)?;
    trainer.finalize_progress(progress, plan.word_count(), "Tokenize words");

    // 2. Count and freeze initial pairs before allocating resident token slots.
    trainer.update_progress(progress, plan.word_count(), "Count pairs");
    let codec = Codec::new(workers);
    let mut index = PairIndex::build(
        &codec,
        &plan,
        trainer.min_frequency,
        workers,
        reuse,
        progress,
    )?;
    trainer.finalize_progress(progress, plan.word_count(), "Count pairs");
    if vocabulary.len() >= trainer.vocab_size {
        trainer.update_progress(progress, trainer.vocab_size, "Compute merges");
        trainer.finalize_progress(progress, 0, "Compute merges");
        drop(index);
        drop(plan);
        drop(codec);
        let (vocab, merges) = vocabulary.into_model_parts(Vec::new());
        return Ok(AttemptOutcome::Complete((
            vocab,
            merges,
            trainer.special_tokens.clone(),
        )));
    }

    // 3. Select compatible batches, prepare snapshot edits, then commit joined jobs.
    let mut corpus = plan.materialize();
    trainer.update_progress(progress, trainer.vocab_size, "Compute merges");
    let mut merges = Vec::new();
    #[cfg(test)]
    let mut trace = Trace::new();
    while vocabulary.len() < trainer.vocab_size {
        let batch = match Batch::select(trainer, &mut vocabulary, &mut corpus, &mut index)? {
            Selection::Finished => break,
            Selection::Restart => {
                trainer.finalize_progress(progress, merges.len(), "Compute merges");
                return Ok(AttemptOutcome::RestartForReuse);
            }
            Selection::Ready(batch) => batch,
        };
        #[cfg(test)]
        trace.extend(batch.trace());
        let previous_len = merges.len();
        merges.extend(batch.pairs());
        let prepared = batch.prepare(
            &corpus,
            &codec,
            trainer.max_token_length.unwrap_or(usize::MAX),
        )?;
        let changes = prepared.apply(&corpus);
        index.commit(changes)?;
        if let Some(p) = progress {
            p.inc((merges.len() - previous_len) as u64);
        }
        trainer.emit_json_progress("Compute merges", merges.len(), trainer.vocab_size);
    }
    trainer.finalize_progress(progress, merges.len(), "Compute merges");
    drop(index);
    drop(corpus);
    drop(codec);
    #[cfg(test)]
    if let Some(observer) = observe.as_mut() {
        for (pair, count, id) in trace {
            observer(pair, count, id);
        }
    }
    let (vocab, merges) = vocabulary.into_model_parts(merges);
    Ok(AttemptOutcome::Complete((
        vocab,
        merges,
        trainer.special_tokens.clone(),
    )))
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
