#![allow(clippy::map_entry)]

use super::{WithFirstLastIterator, BPE};
use crate::parallelism::*;
use crate::tokenizer::{AddedToken, Result, Trainer};
use crate::utils::progress::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
struct Token {
    id: u32,
}

impl Token {
    fn new(id: u32, x: &Token, y: &Token) -> Self {
        // create a new token merging from x and y
        // id stands for the token id without any marks
        // the new token will inherit the marks from x and y
        Self {
            id: id | x.marks() | y.marks(),
        }
    }

    fn mark_begin(&mut self) {
        self.id |= 1 << 31;
    }

    fn mark_end(&mut self) {
        self.id |= 1 << 30;
    }

    fn marks(&self) -> u32 {
        self.id & ((1 << 31) | (1 << 30))
    }
    
    fn is_begin(&self) -> bool {
        self.id & (1 << 31) != 0
    }
    
    fn is_end(&self) -> bool {
        self.id & (1 << 30) != 0
    }
    
    fn pure_id(&self) -> u32 {
        self.id & !(3 << 30)
    }
}

type Pair = (Token, Token);

#[derive(Debug, Eq)]
struct Merge {
    count: u64, // the cached frequency when this merge was pushed to the heap
    pair: Pair, // the pair of tokens that will be merged
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

struct TokenManager {
    vocab: HashMap<String, u32>,
    vocab_r: Vec<String>,
    token_len: Vec<usize>,
    continual_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
}

impl TokenManager {
    fn with_capacity(capacity: usize, continual_subword_prefix: Option<String>, end_of_word_suffix: Option<String>) -> Self {
        let mut manager = Self {
            vocab: HashMap::with_capacity(capacity + 2),
            vocab_r: Vec::with_capacity(capacity + 2),
            token_len: Vec::with_capacity(capacity + 2),
            continual_subword_prefix,
            end_of_word_suffix,
        };
        manager.vocab_r.push("<unk>".to_owned());
        manager.vocab_r.push("<pad>".to_owned());
        manager.token_len.push(1);
        manager.token_len.push(1);
        manager
    }
    
    fn len(&self) -> usize {
        self.vocab.len()    // used for progress bar
    }
    
    fn add_token(&mut self, s: &str, len: usize) -> Token {
        let id = self.vocab.entry(s.to_owned()).or_insert_with(|| {
            let id = self.vocab_r.len() as u32;
            self.vocab_r.push(s.to_owned());
            self.token_len.push(len);
            id
        });
        Token{id: *id}
    }
    
    
    fn to_string(&self, token: &Token) -> String {
        let mut s = self.vocab_r[token.pure_id() as usize].clone();
        if let Some(prefix) = &self.continual_subword_prefix {
            if !token.is_begin() {
                s = format!("{}{}", prefix, s);
            }
        }
        if let Some(suffix) = &self.end_of_word_suffix {
            if token.is_end() {
                s = format!("{}{}", s, suffix);
            }
        }
        s
    }
    
    fn build_token_from_pair(&mut self, pair: &Pair) -> Token {
        let pure_x = pair.0.pure_id() as usize;
        let pure_y = pair.1.pure_id() as usize;
        let raw_x_str = &self.vocab_r[pure_x];
        let raw_y_str = &self.vocab_r[pure_y];
        let raw_pair_str = format!("{}{}", raw_x_str, raw_y_str);
        // check if pair_str is in vocab
        let raw_id = if let Some(&id) = self.vocab.get(&raw_pair_str) {
            id
        } else {
            let id = self.vocab_r.len() as u32;
            self.vocab_r.push(raw_pair_str.clone());
            self.vocab.insert(raw_pair_str, id);
            self.token_len.push(self.token_len[pure_x] + self.token_len[pure_y]);
            id
        };
        Token::new(raw_id, &pair.0, &pair.1)
    }
    
    fn mark_begin(&self) -> bool {
        self.continual_subword_prefix.is_some()
    }
    
    fn mark_end(&self) -> bool {
        self.end_of_word_suffix.is_some()
    }
}

struct FreqStatus {
    freq: u64,
    next_pivot: usize,
    next_index: usize,
}

struct Corpus {
    data: Vec<Token>,
    freq_change_pivot: Vec<usize>,
    freq_change_value: Vec<u64>,
}

impl Corpus {
    fn get_freq(&self, pos: usize) -> FreqStatus {
        // search for the frequency of the token at position pos
        // using binary search
        // each pivot means freq change at that position
        // the freq_change_pivot begins with 0 and ends with INF
        // the freq_change_value is the frequency at that pivot
        let i = self.freq_change_pivot.partition_point(|&x| x <= pos) - 1;
        FreqStatus {
            freq: self.freq_change_value[i],
            next_pivot: self.freq_change_pivot[i + 1],
            next_index: i + 1,
        }
    }
    
    fn get_next_freq(&self, pos: usize, prev_status: &mut FreqStatus) -> u64 {
        if pos < prev_status.next_pivot {
            return prev_status.freq;
        } 
        
        if pos < self.freq_change_pivot[prev_status.next_index + 1] {
            prev_status.freq = self.freq_change_value[prev_status.next_index];
            prev_status.next_index += 1;
            prev_status.next_pivot = self.freq_change_pivot[prev_status.next_index];
            return prev_status.freq;
        }

        let i = self.freq_change_pivot[prev_status.next_index + 1..]
            .partition_point(|&x| x <= pos)
            + prev_status.next_index;
        
        prev_status.freq = self.freq_change_value[i];
        prev_status.next_index = i;
        prev_status.next_pivot = self.freq_change_pivot[i + 1];
        
        prev_status.freq
    }
    
    fn add_freq_info(&mut self, value: u64, pivot: usize) {
        self.freq_change_value.push(value);
        self.freq_change_pivot.push(pivot);
    }
    
    fn len(&self) -> usize {
        self.data.len()
    }
}

struct PairStatus {
    pair_pos: HashMap<Pair, Vec<usize>>,
    pair_counts: HashMap<Pair, i64>,
    queue: BinaryHeap<Merge>,
}

impl PairStatus {
    fn new(mut pair_pos: HashMap<Pair, Vec<usize>>, pair_counts: HashMap<Pair, i64>, min_freq: u64) -> Self {
        let mut queue = BinaryHeap::with_capacity(pair_pos.len());
        let mut final_pair_counts = HashMap::with_capacity(pair_pos.len());
        for (pair, count) in pair_counts.into_iter() {
            if count < min_freq as i64 {
                // remove the pair in pair_pos
                pair_pos.remove(&pair);
                final_pair_counts.insert(pair, count);
                queue.push(Merge {
                    count: count as u64,
                    pair,
                });
            }
        }
        Self{pair_pos, pair_counts: final_pair_counts, queue}
    }

}


struct Config {
    min_frequency: u64,
    vocab_size: usize,
    show_progress: bool,
    special_tokens: Vec<AddedToken>,
    limit_alphabet: Option<usize>,
    initial_alphabet: HashSet<char>,
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
                special_tokens: vec![],
                limit_alphabet: None,
                initial_alphabet: HashSet::new(),
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

    /// Set the initial alphabet
    #[must_use]
    pub fn initial_alphabet(mut self, alphabet: HashSet<char>) -> Self {
        self.config.initial_alphabet = alphabet;
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
    /// Set max_token_length
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
            special_tokens: self.config.special_tokens,
            limit_alphabet: self.config.limit_alphabet,
            initial_alphabet: self.config.initial_alphabet,
            continuing_subword_prefix: self.config.continuing_subword_prefix,
            end_of_word_suffix: self.config.end_of_word_suffix,
            max_token_length: self.config.max_token_length,
            words: HashMap::new(),
        }
    }
}

/// In charge of training a `BPE` model
///
/// # Examples
///
/// ```
/// use tokenizers::tokenizer::Trainer;
/// use tokenizers::models::bpe::{BPE, BpeTrainer};
///
/// let sequences = vec![ "Hello", "World" ];
///
/// let mut trainer = BpeTrainer::default();
/// trainer.feed(sequences.iter(), |s| Ok(vec![s.to_owned()]));
///
/// let mut model = BPE::default();
/// let special_tokens = trainer.train(&mut model).unwrap();
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
    /// A list of special tokens that the model should know of
    pub special_tokens: Vec<AddedToken>,
    /// Whether to limit the number of initial tokens that can be kept before computing merges
    pub limit_alphabet: Option<usize>,
    /// The initial alphabet we want absolutely to include. This allows to cover
    /// some characters that are not necessarily in the training set
    pub initial_alphabet: HashSet<char>,
    /// An optional prefix to use on any subword that exist only behind another one
    pub continuing_subword_prefix: Option<String>,
    /// An optional suffix to characterize and end-of-word subword
    pub end_of_word_suffix: Option<String>,
    /// An optional parameter to limit the max length of any single token
    pub max_token_length: Option<usize>,
    
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

    /// Setup a progress bar if asked to show progress
    fn setup_progress(&self) -> Option<ProgressBar> {
        if self.show_progress {
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

    /// Set the progress bar in the finish state
    fn finalize_progress(&self, p: &Option<ProgressBar>, final_len: usize) {
        if let Some(p) = p {
            p.set_length(final_len as u64);
            p.finish();
            println!();
        }
    }

    /// Update the progress bar with the new provided length and message
    fn update_progress(&self, p: &Option<ProgressBar>, len: usize, message: &'static str) {
        if let Some(p) = p {
            p.set_message(message);
            p.set_length(len as u64);
            p.reset();
        }
    }

    /// Add the provided special tokens to the initial vocabulary
    fn add_special_tokens(&self, tk_manager: &mut TokenManager) {
        for token in &self.special_tokens {
            tk_manager.add_token(&token.content, 1);
        }
    }

    /// Compute the initial alphabet and limit it if relevant
    fn compute_alphabet(
        &self,
        wc: &HashMap<String, u64>,
        tk_manager: &mut TokenManager,
    ) -> HashMap<char, Token> {
        // Compute the alphabet from seen words
        let mut alphabet: HashMap<char, usize> = HashMap::new();
        for (word, count) in wc {
            for c in word.chars() {
                alphabet
                    .entry(c)
                    .and_modify(|cnt| *cnt += *count as usize)
                    .or_insert(*count as usize);
            }
        }

        // Also include anything from the provided initial alphabet
        for c in &self.initial_alphabet {
            alphabet
                .entry(*c)
                .and_modify(|cnt| *cnt = usize::MAX)
                .or_insert(usize::MAX);
        }

        let mut kept = alphabet.iter().collect::<Vec<_>>();

        // Compute the number of chars to remove from the alphabet
        // If `limit_alphabet < initial_alphabet.len()`, some of these initial characters
        // will be removed
        let to_remove = self
            .limit_alphabet
            .map(|limit| {
                if alphabet.len() > limit {
                    alphabet.len() - limit
                } else {
                    0
                }
            })
            .unwrap_or(0);

        // Remove the unwanted chars
        if to_remove > 0 {
            kept.sort_unstable_by_key(|k| *k.1);
            kept.drain(..to_remove);
        }

        // Keep the initial alphabet (sorted for determinism)
        let mut alphabet: HashMap<char, Token> = HashMap::with_capacity(kept.len());
        kept.sort_unstable_by_key(|k| (*k.0) as u32);
        kept.into_iter().for_each(|(c, _)| {
            let token = tk_manager.add_token(&c.to_string(), 1);
            alphabet.insert(c.clone(), token);
        });
        alphabet
    }
    
    fn build_corpus(
        &self,
        wc: &HashMap<String, u64>,
        alphabet: &HashMap<char, Token>,
        tk_manager: &mut TokenManager,
        p: &Option<ProgressBar>,
    ) -> (Corpus, PairStatus) {
        let mut data = Vec::with_capacity(wc.len());
        data.push(Token{id: 0}); // A mask
        
        let mut pair_pos: HashMap<Pair, Vec<usize>> = HashMap::new();
        let mut pair_counts: HashMap<Pair, i64> = HashMap::new();
        
        let mut words = wc.iter().map(| w, c| { (-*c, w) }).collect();
        // sort by frequency, descending
        words.sort_unstable_by_key(|x| x.0);
        let mut freq_change_pivot = vec![0];
        let mut freq_change_value = vec![-words[0].0];
        
        let mut prev_token = Token{id: 0};
        let (mark_begin, mark_end) = (tk_manager.mark_begin(), tk_manager.mark_end());
        for (mut count, word) in words {
            count = -count;
            if count < freq_change_value.last().unwrap() {
                freq_change_pivot.push(data.len());
                freq_change_value.push(count);
            }
            for (is_first, is_last, c) in word.chars().with_first_and_last() {
                let mut token = alphabet
                    .get(&c).copied()
                    .unwrap_or_else(|| { Token{id: 1} });
                // I believe that compiler could optimize this
                // by moving the if statement out of the loop
                if mark_begin && is_first {
                    token.mark_begin()
                }
                if mark_end && is_last {
                    token.mark_end()
                }
                // I still believe that compiler could unroll this loop
                // And we actually only need to check id > 1
                if !(is_first && is_last) && token.id > 1 && prev_token.id > 1 {
                    let pair = (prev_token, token);
                    pair_pos.entry(pair).or_insert_with(Vec::new).push(data.len()-1);
                    pair_counts.entry(pair).and_modify(|c| *c += count).or_insert(count as i64);
                }
                data.push(token);
                prev_token = token;
            }
            data.push(Token{id: 0}); // padding
            prev_token = Token{id: 0};
            if let Some(p) = p {
                p.inc(1);
            }
        }
        
        freq_change_pivot.push(data.len() + 1);
        freq_change_value.push(0);
        
        let corpus = Corpus {
            data,
            freq_change_pivot,
            freq_change_value,
        };
        let pair_status = PairStatus::new(pair_pos, pair_counts, self.min_frequency);
        
        (corpus, pair_status)
    }

    pub fn do_train(
        &self,
        word_counts: &HashMap<String, u64>,
        model: &mut BPE,
    ) -> Result<Vec<AddedToken>> {
        let mut tk_manager = TokenManager::with_capacity(
            self.vocab_size,
            self.continuing_subword_prefix.clone(),
            self.end_of_word_suffix.clone(),
        );
        
        let max_token_length: usize = self.max_token_length.unwrap_or(usize::MAX);

        let progress = self.setup_progress();

        //
        // 1. Add all special tokens to the vocabulary
        //
        self.add_special_tokens(&mut tk_manager);

        //
        // 2. Compute the initial alphabet
        //
        let alphabet = self.compute_alphabet(word_counts, &mut tk_manager);

        //
        // 3. Build the corpus
        //
        self.update_progress(&progress, word_counts.len(), "Tokenize words");
        let (corpus, pair_stat) =
            self.build_corpus(word_counts, &alphabet, &mut tk_manager, &progress);
        self.finalize_progress(&progress, corpus.len());

        //
        // 5. Do merges
        //
        self.update_progress(&progress, self.vocab_size, "Compute merges");
       
        self.finalize_progress(&progress, merges.len());

        // Transfer new vocab & options to model
        model.vocab = word_to_id;
        model.vocab_r = model
            .vocab
            .iter()
            .map(|(key, val)| (*val, key.to_owned()))
            .collect();
        model.merges = merges
            .into_iter()
            .enumerate()
            .map(|(i, (pair, new_token_id))| (pair, (i as u32, new_token_id)))
            .collect();

        if let Some(prefix) = &self.continuing_subword_prefix {
            model.continuing_subword_prefix = Some(prefix.to_owned());
        } else {
            model.continuing_subword_prefix = None;
        }
        if let Some(suffix) = &self.end_of_word_suffix {
            model.end_of_word_suffix = Some(suffix.to_owned());
        } else {
            model.end_of_word_suffix = None;
        }

        Ok(self.special_tokens.clone())
    }
}

impl Trainer for BpeTrainer {
    type Model = BPE;

    /// Train a BPE model
    fn train(&self, model: &mut BPE) -> Result<Vec<AddedToken>> {
        self.do_train(&self.words, model)
    }

    /// Whether we should show progress
    fn should_show_progress(&self) -> bool {
        self.show_progress
    }

    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item=S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync,
    {
        let words: Result<HashMap<String, u64>> = iterator
            .maybe_par_bridge()
            .map(|sequence| {
                let words = process(sequence.as_ref())?;
                let mut map = HashMap::new();
                for word in words {
                    map.entry(word).and_modify(|c| *c += 1).or_insert(1);
                }
                Ok(map)
            })
            .reduce(
                || Ok(HashMap::new()),
                |acc, ws| {
                    let mut acc = acc?;
                    for (k, v) in ws? {
                        acc.entry(k).and_modify(|c| *c += v).or_insert(v);
                    }
                    Ok(acc)
                },
            );

        self.words = words?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{BpeTrainer, Pair, BPE};
    use std::collections::HashMap;

    #[test]
    fn test_train() {
        let word_counts: HashMap<String, u64> = [
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
        let mut model = BPE::default();
        trainer.do_train(&word_counts, &mut model).unwrap();

        // Vocab should contain all of the characters from the `word_counts` mapping
        // as well as three merges: 're', 'are', and 'is'.
        let expected_vocab: HashMap<String, u32> = [
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
        assert_eq!(model.vocab, expected_vocab);

        // The keys in `merges` are pairs of symbols, the values are tuples of (rank, id),
        // where 'rank' determines the order in which this merge will be applied during
        // tokenization, and 'id' is the vocab id of the symbol resulting from merging
        // the pair of symbols in the corresponding key.
        let expected_merges: HashMap<Pair, (u32, u32)> = [
            ((17, 11), (0, 22)), // 'r' + 'e'  -> 're'
            ((8, 22), (1, 23)),  // 'a' + 're' -> 'are'
            ((13, 18), (2, 24)), // 'i' + 's'  -> 'is'
        ]
            .iter()
            .cloned()
            .collect();
        assert_eq!(model.merges, expected_merges);
    }
    #[test]
    fn bpe_test_max_token_length_16() {
        /* bpe_test_max_token_length series of tests test the max_token_length flag of bpetrainer
        // this is the more robust version that only tests max length of learned tokens
        // (pre) tokenizer settings or vocab can be easily modified when necessary
         */

        let max_token_length = 16;
        let long_word_counts: HashMap<String, u64> = [
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
            .map(|(key, value)| (key.to_string(), *value))
            .collect();
        let trainer = BpeTrainer::builder()
            .max_token_length(Some(max_token_length))
            .show_progress(false)
            .min_frequency(0)
            .build();
        let mut model = BPE::default();
        trainer.do_train(&long_word_counts, &mut model).unwrap();
        let vocab = model.get_vocab();
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
        let long_word_counts: HashMap<String, u64> = [
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
            .map(|(key, value)| (key.to_string(), *value))
            .collect();
        let trainer = BpeTrainer::builder()
            .max_token_length(Some(2))
            .show_progress(false)
            .min_frequency(0)
            .build();
        let mut model = BPE::default();
        trainer.do_train(&long_word_counts, &mut model).unwrap();
        let trained_vocab: HashMap<String, u32> = model.get_vocab();
        let expected_vocab: HashMap<String, u32> = [
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
