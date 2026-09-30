//! Sequential HF adaptation of Efficient BPE's occurrence index and local rewrites.
//!
//! HF heap entries own historical cohorts of words. Its counts deliberately omit
//! the selected edge's removal. Affix aliases can make the ledger and cohorts
//! observable, so the general path retains them and scans cohort words as needed.
//! Ordinary character BPE has a stronger invariant: every replacement identity
//! appears in the corpus for the first time, even when its string already has a
//! reserved vocabulary ID. Old pairs then only lose edges. The guarded path uses
//! that proof to retire low-frequency pairs, including finite length limits;
//! nonempty affixes keep the general ledger. See
//! benchmarks/hf-bpe/PAIR_MONOTONICITY.md for the proof and its scope.
use super::*;
use std::time::Instant;

mod aa_parity;
mod compact;
mod parallel;
mod small_posting;

const NONE: u32 = u32::MAX;

/// Measurements for the experimental sequential occurrence-index trainer.
#[derive(Debug, Default, Clone, Serialize)]
pub(super) struct IndexedTrainingStats {
    pub initialize_ms: f64,
    pub merge_ms: f64,
    pub initial_symbols: usize,
    pub posting_visits: usize,
    pub stale_posting_visits: usize,
    pub word_scan_steps: usize,
    pub reused_ids: usize,
    pub monotone_pairs: bool,
    pub pruned_pairs: usize,
    pub layout: &'static str,
    pub corpus_bytes: usize,
    pub posting_bytes: usize,
    pub peak_birth_bytes: usize,
    pub workers: usize,
    pub atomic_corpus: bool,
    pub initialization_workers: usize,
    pub batch_rounds: usize,
    pub max_batch_rules: usize,
    pub initial_corpus_bytes: usize,
    pub initial_posting_bytes: usize,
    pub initial_pair_table_bytes: usize,
    pub initial_block_table_bytes: usize,
    pub initial_directory_bytes: usize,
    pub initial_heap_bytes: usize,
    pub initial_slots: usize,
    pub initial_edges: usize,
    pub initial_pairs: usize,
    pub initial_blocks: usize,
    pub initial_block_pairs: usize,
    pub initial_slot_bytes: usize,
    pub initial_length_bytes: usize,
    pub initial_weight_bytes: usize,
    pub tokenize_ms: f64,
    pub alphabet_ms: f64,
    pub alphabet_scratch_bytes: usize,
    pub character_table_bytes: usize,
    pub corpus_measure_ms: f64,
    pub corpus_allocate_ms: f64,
    pub corpus_fill_ms: f64,
    pub initial_route_ms: f64,
    pub initial_count_ms: f64,
    pub initial_count_backend: &'static str,
    pub initial_weight_lookup_ms: f64,
    pub initial_weight_lookup_bytes: usize,
    pub initial_route_compact_ms: f64,
    pub initial_radix_sort_ms: f64,
    pub initial_group_count_ms: f64,
    pub initial_posting_install_ms: f64,
    pub initial_route_buffer_bytes: usize,
    pub peak_initial_route_buffer_bytes: usize,
    pub initial_radix_scratch_bytes: usize,
    pub initial_group_buffer_bytes: usize,
    pub initial_heap_ms: f64,
    pub select_ms: f64,
    pub plan_ms: f64,
    pub fused_prepare_ms: f64,
    pub fused_batches: usize,
    pub peak_valid_start_bytes: usize,
    pub weight_lookup_build_ms: f64,
    pub weight_lookup_bytes: usize,
    pub peak_selected_lookup_bytes: usize,
    pub delta_ms: f64,
    pub rewrite_ms: f64,
    pub commit_ms: f64,
    pub route_ms: f64,
}

/// Explicit controls for the experimental parallel occurrence trainer.
#[derive(Debug, Clone, Copy)]
pub(super) struct IndexedParallelConfig {
    pub workers: usize,
    /// Optional initialization pool for controlled comparisons; merges keep workers.
    pub initialization_workers: Option<usize>,
    /// 16 uses u16 local posting offsets; 32 uses u32 offsets.
    pub posting_block_bits: u8,
    /// Use u16 corpus IDs when the complete possible ID domain fits.
    pub narrow_corpus: bool,
    /// Experimental Relaxed atomic slot accesses with otherwise identical work.
    pub atomic_corpus: bool,
    /// Maximum certified rules per batch, at most 256.
    pub batch_size: usize,
}

impl Default for IndexedParallelConfig {
    fn default() -> Self {
        Self {
            workers: 4,
            initialization_workers: None,
            posting_block_bits: 32,
            narrow_corpus: true,
            atomic_corpus: false,
            batch_size: 256,
        }
    }
}

/// Raw HF model parts and measurements; build a `PipelineBPE` using the trainer's affixes.
pub(super) struct IndexedTraining {
    pub vocab: Vocab,
    pub merges: Merges,
    pub special_tokens: Vec<AddedToken>,
    pub stats: IndexedTrainingStats,
    #[cfg(test)]
    trace: Vec<(Pair, u64, u32)>,
}

// Like Efficient BPE, live tokens occupy their original start/end positions in
// one u32 corpus. NONE separates words (HF's token ID zero is a valid token).
// With uniquely represented raw spans, a per-ID length table finds neighbours.
// Affix identity collisions need a position-wide span table: the same ID can have
// different physical lengths. Historical positions retain their word identity
// through the immutable pivots, without a word ID on every symbol.
struct Context {
    pos: usize,
    right: usize,
    after: usize,
    before: Option<usize>,
    len: u32,
}

#[derive(Eq)]
struct Candidate {
    pair: Pair,
    count: u64,
    positions: Vec<u32>,
}
impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.pair == other.pair && self.count == other.count
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count
            .cmp(&other.count)
            .then_with(|| other.pair.cmp(&self.pair))
    }
}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct Index {
    corpus: Vec<u32>,
    lengths: Vec<u32>,
    spans: Option<Vec<u32>>,
    pivots: Vec<u32>,
    weights: Vec<u64>,
    counts: AHashMap<Pair, i64>,
    queue: OctonaryHeap<Candidate>,
    births: AHashMap<Pair, Vec<u32>>,
    // Some(floor) certifies ordinary global character BPE, not fresh numeric IDs.
    monotone_floor: Option<u64>,
    pruned_pairs: usize,
}

struct PreparedCorpus {
    corpus: Vec<u32>,
    lengths: Vec<u32>,
    pivots: Vec<u32>,
    weights: Vec<u64>,
    monotone_floor: Option<u64>,
}

fn supports_compact(trainer: &BpeTrainer) -> bool {
    trainer
        .continuing_subword_prefix
        .as_deref()
        .is_none_or(str::is_empty)
        && trainer
            .end_of_word_suffix
            .as_deref()
            .is_none_or(str::is_empty)
}

impl PreparedCorpus {
    fn tokenize(
        trainer: &BpeTrainer,
        wc: &AHashMap<CompactString, u64>,
        w2id: &mut AHashMap<CompactString, u32>,
        id2w: &mut Vec<CompactString>,
        progress: &Option<ProgressBar>,
    ) -> Result<Self> {
        // Count Unicode scalars, rather than reserving the UTF-8 byte bound,
        // so Chinese input does not reserve three times its corpus payload.
        let capacity = wc
            .keys()
            .try_fold(1_usize, |size, word| {
                size.checked_add(word.chars().count())?.checked_add(1)
            })
            .ok_or("indexed BPE corpus size exceeds usize")?;
        if capacity >= NONE as usize {
            return Err("indexed BPE corpus including separators must fit u32 positions".into());
        }
        let mut corpus = Vec::with_capacity(capacity);
        corpus.push(NONE);
        let mut pivots = Vec::with_capacity(wc.len());
        let mut weights = Vec::with_capacity(wc.len());
        // Zero means this identity has never been live, including long strings
        // reserved by special_tokens. A vocabulary entry alone is not activation.
        let mut lengths = vec![0; id2w.len()];
        let mut decorated = String::new();
        for (word, &weight) in wc {
            pivots.push(corpus.len() as u32);
            weights.push(weight);
            for (is_first, is_last, c) in word.chars().with_first_and_last() {
                let mut utf8 = [0_u8; 4];
                let plain: &str = c.encode_utf8(&mut utf8);
                // Use the original character's position for affixes, even if
                // alphabet filtering removed the first or last character.
                let Some(&plain_id) = w2id.get(plain) else {
                    continue;
                };
                let prefix = (!is_first)
                    .then_some(trainer.continuing_subword_prefix.as_deref())
                    .flatten();
                let suffix = is_last
                    .then_some(trainer.end_of_word_suffix.as_deref())
                    .flatten();
                let key = if prefix.is_some() || suffix.is_some() {
                    decorated.clear();
                    decorated.push_str(prefix.unwrap_or(""));
                    decorated.push_str(plain);
                    decorated.push_str(suffix.unwrap_or(""));
                    decorated.as_str()
                } else {
                    plain
                };
                let id = if prefix.is_none_or(str::is_empty) && suffix.is_none_or(str::is_empty) {
                    plain_id
                } else if let Some(&id) = w2id.get(key) {
                    id
                } else {
                    let id = u32::try_from(id2w.len()).map_err(|_| "BPE vocabulary exceeds u32")?;
                    if id == NONE {
                        return Err("indexed BPE reserves u32::MAX for word separators".into());
                    }
                    let token = CompactString::from(key);
                    id2w.push(token.clone());
                    w2id.insert(token, id);
                    lengths.push(0);
                    id
                };
                lengths[id as usize] = 1;
                corpus.push(id);
            }
            corpus.push(NONE);
            if let Some(p) = progress {
                p.inc(1);
            }
        }
        trainer.finalize_progress(progress, wc.len(), "Tokenize words");
        trainer.update_progress(progress, wc.len(), "Count pairs");
        // Nonempty affixes can alias distinct raw spans. Without affixes, the
        // length predicate is constant per pair identity, so the proof still
        // applies to finite HF limits (including the initial-pair exception).
        let plain = supports_compact(trainer);
        Ok(Self {
            corpus,
            pivots,
            weights,
            lengths,
            monotone_floor: plain.then_some(trainer.min_frequency.max(1)),
        })
    }
}

impl Index {
    fn tokenize(
        trainer: &BpeTrainer,
        wc: &AHashMap<CompactString, u64>,
        w2id: &mut AHashMap<CompactString, u32>,
        id2w: &mut Vec<CompactString>,
        progress: &Option<ProgressBar>,
    ) -> Result<Self> {
        let input = PreparedCorpus::tokenize(trainer, wc, w2id, id2w, progress)?;
        Self::new(
            input.corpus,
            input.pivots,
            input.weights,
            input.lengths,
            input.monotone_floor,
        )
    }
    fn new(
        corpus: Vec<u32>,
        pivots: Vec<u32>,
        weights: Vec<u64>,
        lengths: Vec<u32>,
        monotone_floor: Option<u64>,
    ) -> Result<Self> {
        let mut counts = AHashMap::<Pair, i64>::new();
        let mut postings = AHashMap::<Pair, Vec<u32>>::new();
        let mut total = 0_i64;
        for (&pivot, &weight) in pivots.iter().zip(&weights) {
            let weight =
                i64::try_from(weight).map_err(|_| "indexed BPE weight exceeds i64::MAX")?;
            let mut pos = pivot as usize;
            while corpus[pos] != NONE {
                if corpus[pos + 1] != NONE {
                    total = total
                        .checked_add(weight)
                        .ok_or("indexed BPE weighted pair counts exceed i64::MAX")?;
                    let pair = (corpus[pos], corpus[pos + 1]);
                    *counts.entry(pair).or_default() += weight;
                    postings.entry(pair).or_default().push(pos as u32);
                }
                pos += 1;
            }
        }
        let queue = postings
            .into_iter()
            .filter_map(|(pair, positions)| {
                let count = counts[&pair];
                (count > 0 && count as u64 >= monotone_floor.unwrap_or(1)).then_some(Candidate {
                    pair,
                    count: count as u64,
                    positions,
                })
            })
            .collect();
        let mut pruned_pairs = 0;
        if let Some(floor) = monotone_floor {
            let before = counts.len();
            counts.retain(|_, count| *count > 0 && *count as u64 >= floor);
            pruned_pairs = before - counts.len();
        }
        Ok(Self {
            corpus,
            lengths,
            spans: None,
            pivots,
            weights,
            counts,
            queue,
            births: AHashMap::new(),
            monotone_floor,
            pruned_pairs,
        })
    }

    fn word_at(&self, pos: u32) -> usize {
        self.pivots.partition_point(|&pivot| pivot <= pos) - 1
    }

    fn span(&self, pos: usize) -> u32 {
        match &self.spans {
            Some(spans) => spans[pos],
            None => self.lengths[self.corpus[pos] as usize],
        }
    }

    fn inspect(&self, pos: usize, pair: Pair) -> Option<Context> {
        if self.corpus[pos] != pair.0 {
            return None;
        }
        let left_len = self.span(pos);
        let right = pos + left_len as usize;
        if left_len == 0 || right >= self.corpus.len() || self.corpus[right] != pair.1 {
            return None;
        }
        let right_len = self.span(right);
        let after = right + right_len as usize;
        if right_len == 0 || after >= self.corpus.len() {
            return None;
        }
        let before = if self.corpus[pos - 1] == NONE {
            None
        } else {
            Some(pos - self.span(pos - 1) as usize)
        };
        Some(Context {
            pos,
            right,
            after,
            before,
            len: left_len + right_len,
        })
    }

    fn prepare_replacement(&mut self, pair: Pair, replacement: u32) {
        if self.monotone_floor.is_some() {
            let length = self.lengths[pair.0 as usize] + self.lengths[pair.1 as usize];
            if replacement as usize == self.lengths.len() {
                self.lengths.push(length);
            } else {
                // The isolation lemma rules out a second activation of a raw
                // string. Reusing a reserved ID is its first live occurrence.
                debug_assert_eq!(self.lengths[replacement as usize], 0);
                self.lengths[replacement as usize] = length;
            }
            return;
        }
        if replacement as usize == self.lengths.len() {
            let length = if self.spans.is_none() {
                self.lengths[pair.0 as usize] + self.lengths[pair.1 as usize]
            } else {
                0 // Occurrence spans supply lengths after the first ID reuse.
            };
            self.lengths.push(length);
        } else if self.spans.is_none() {
            // Materialize only live endpoints before changing an existing ID's
            // physical span. No corpus-sized compatibility array in the fresh path.
            let mut spans = vec![0; self.corpus.len()];
            for &pivot in &self.pivots {
                let mut pos = pivot as usize;
                while self.corpus[pos] != NONE {
                    let len = self.lengths[self.corpus[pos] as usize];
                    spans[pos] = len;
                    spans[pos + len as usize - 1] = len;
                    pos += len as usize;
                }
            }
            self.spans = Some(spans);
        }
    }

    fn change(&mut self, pair: Pair, change: i64, pos: u32, weight: i64) -> Result<()> {
        let count = match self.counts.entry(pair) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(_)
                if change < 0 && self.monotone_floor.is_some() =>
            {
                // A retired old pair can only lose occurrences. Its forgotten
                // count cannot affect eligibility or create another candidate.
                return Ok(());
            }
            std::collections::hash_map::Entry::Vacant(entry) => entry.insert(0),
        };
        *count = count
            .checked_add(change * weight)
            .ok_or("indexed BPE pair ledger exceeds i64 range")?;
        if change > 0 {
            self.births.entry(pair).or_default().push(pos);
        }
        Ok(())
    }

    fn merge_at(&mut self, context: Context, replacement: u32, max_length: usize) -> Result<usize> {
        let Context {
            pos,
            right,
            after,
            before,
            len,
        } = context;
        let left_id = self.corpus[pos];
        let right_id = self.corpus[right];
        let weight = self.weights[self.word_at(pos as u32)] as i64;
        // Match Word::merge: remove both old neighbours unconditionally, and
        // introduce a new neighbour only for a STRICTLY smaller combined span.
        // The chosen pair's own frequency is not subtracted in the HF ledger.
        if let Some(before) = before {
            let prior_id = self.corpus[before];
            self.change((prior_id, left_id), -1, before as u32, weight)?;
            if (self.span(before) as usize) + (len as usize) < max_length {
                self.change((prior_id, replacement), 1, before as u32, weight)?;
            }
        }
        if self.corpus[after] != NONE {
            let after_id = self.corpus[after];
            self.change((right_id, after_id), -1, right as u32, weight)?;
            if (len as usize) + (self.span(after) as usize) < max_length {
                self.change((replacement, after_id), 1, pos as u32, weight)?;
            }
        }
        // Same endpoint rewrite as the prototype: retain the new ID at the
        // start/end, and clear a retired right start when it is not also the end.
        self.corpus[pos] = replacement;
        self.corpus[right] = if right + 1 == after {
            replacement
        } else {
            NONE
        };
        self.corpus[after - 1] = replacement;
        if let Some(spans) = &mut self.spans {
            if right + 1 != after {
                spans[right] = 0;
            }
            spans[pos] = len;
            spans[after - 1] = len;
        }
        Ok(after)
    }

    fn apply(
        &mut self,
        mut top: Candidate,
        replacement: u32,
        scan_words: bool,
        max_length: usize,
        stats: &mut IndexedTrainingStats,
    ) -> Result<()> {
        self.prepare_replacement(top.pair, replacement);
        stats.posting_visits += top.positions.len();
        if scan_words {
            // An indexed edge stands for its entire word cohort in HF. Even a
            // retired or length-excluded occurrence can identify that word.
            let mut words: Vec<u32> = top
                .positions
                .iter()
                .map(|&p| self.word_at(p) as u32)
                .collect();
            words.sort_unstable();
            words.dedup();
            for word in words {
                let mut pos = self.pivots[word as usize] as usize;
                while self.corpus[pos] != NONE {
                    stats.word_scan_steps += 1;
                    pos = if let Some(context) = self.inspect(pos, top.pair) {
                        self.merge_at(context, replacement, max_length)?
                    } else {
                        pos + self.span(pos) as usize
                    };
                }
            }
        } else {
            // AB occurrences cannot overlap. AA must choose the leftmost edge
            // in each run; births are not necessarily globally ordered.
            if top.pair.0 == top.pair.1 {
                top.positions.sort_unstable();
            }
            for pos in top.positions {
                if let Some(context) = self.inspect(pos as usize, top.pair) {
                    self.merge_at(context, replacement, max_length)?;
                } else {
                    stats.stale_posting_visits += 1;
                }
            }
        }
        for (pair, positions) in self.births.drain() {
            let count = self.counts[&pair];
            // Aggregate the entire round first: multiple births of a NEW pair
            // can increase its count before it becomes a completed old key.
            if count > 0 && count as u64 >= self.monotone_floor.unwrap_or(1) {
                self.queue.push(Candidate {
                    pair,
                    count: count as u64,
                    positions,
                });
            } else if self.monotone_floor.is_some() {
                self.counts.remove(&pair);
                self.pruned_pairs += 1;
            }
        }
        Ok(())
    }
}

impl BpeTrainer {
    pub(super) fn train_vocab_indexed_parallel(
        &self,
        config: IndexedParallelConfig,
    ) -> Result<IndexedTraining> {
        self.do_train_indexed_parallel(&self.words, config)
    }

    /// Nonempty affixes retain the serial HF cohort engine.
    /// Workers are explicit and independent of the global Rayon pool.
    pub(super) fn do_train_indexed_parallel(
        &self,
        word_counts: &AHashMap<CompactString, u64>,
        config: IndexedParallelConfig,
    ) -> Result<IndexedTraining> {
        if config.workers == 0
            || config.initialization_workers == Some(0)
            || !matches!(config.posting_block_bits, 16 | 32)
            || config.batch_size == 0
            || config.batch_size > 256
        {
            return Err("invalid indexed parallel configuration".into());
        }
        if supports_compact(self) {
            parallel::train(self, word_counts, config)
        } else {
            self.do_train_indexed(word_counts)
        }
    }
    /// Experimental sequential position-index trainer. The ordinary `train_vocab`
    /// remains the reference backend. Initialization preserves HF's token identities
    /// while writing directly into Efficient BPE's flat endpoint layout.
    pub(super) fn train_vocab_indexed(&self) -> Result<IndexedTraining> {
        self.do_train_indexed(&self.words)
    }

    /// Combine stable endpoint indexing for long pieces with the PR's compact
    /// word arena and forward rewrite for short pieces. Affixes keep HF handling.
    pub(super) fn train_vocab_fused(&self) -> Result<IndexedTraining> {
        self.do_train_fused(&self.words)
    }

    pub(super) fn do_train_fused(
        &self,
        word_counts: &AHashMap<CompactString, u64>,
    ) -> Result<IndexedTraining> {
        if supports_compact(self) {
            compact::train(self, word_counts, true)
        } else {
            self.do_train_indexed(word_counts)
        }
    }

    /// Train weighted pretokenized words with the experimental occurrence index.
    /// Nonempty affixes retain HF word-cohort handling. Ordinary character BPE
    /// uses the proven monotone-pair path, including reserved IDs and length gates.
    pub(super) fn do_train_indexed(
        &self,
        word_counts: &AHashMap<CompactString, u64>,
    ) -> Result<IndexedTraining> {
        if supports_compact(self) {
            return compact::train(self, word_counts, false);
        }
        let begin = Instant::now();
        let mut word_to_id = AHashMap::with_capacity(self.vocab_size);
        let mut id_to_word = Vec::with_capacity(self.vocab_size);
        let progress = self.setup_progress();
        self.add_special_tokens(&mut word_to_id, &mut id_to_word);
        self.compute_alphabet(word_counts, &mut word_to_id, &mut id_to_word);
        self.update_progress(&progress, word_counts.len(), "Tokenize words");
        let mut index = Index::tokenize(
            self,
            word_counts,
            &mut word_to_id,
            &mut id_to_word,
            &progress,
        )?;
        self.finalize_progress(&progress, index.pivots.len(), "Count pairs");
        let mut stats = IndexedTrainingStats {
            initialize_ms: begin.elapsed().as_secs_f64() * 1000.0,
            initial_symbols: index.corpus.len() - index.pivots.len() - 1,
            monotone_pairs: index.monotone_floor.is_some(),
            layout: "hf_cohorts",
            corpus_bytes: index.corpus.capacity() * std::mem::size_of::<u32>(),
            ..Default::default()
        };
        let begin = Instant::now();
        let mut scan_words = self.max_token_length.is_some();
        let max_length = self.max_token_length.unwrap_or(usize::MAX);
        let mut merges = Vec::new();
        #[cfg(test)]
        let mut trace = Vec::new();
        self.update_progress(&progress, self.vocab_size, "Compute merges");
        while word_to_id.len() < self.vocab_size {
            let Some(mut top) = index.queue.pop() else {
                break;
            };
            let Some(&count) = index.counts.get(&top.pair) else {
                debug_assert!(index.monotone_floor.is_some());
                continue;
            };
            if let Some(floor) = index.monotone_floor
                && (count <= 0 || (count as u64) < floor)
            {
                index.counts.remove(&top.pair);
                index.pruned_pairs += 1;
                continue;
            }
            let count = count as u64;
            if top.count != count {
                top.count = count;
                index.queue.push(top);
                continue;
            }
            if count == 0 || count < self.min_frequency {
                break;
            }
            let a = &id_to_word[top.pair.0 as usize];
            let b = id_to_word[top.pair.1 as usize].as_str();
            let b = self
                .continuing_subword_prefix
                .as_deref()
                .and_then(|p| b.strip_prefix(p))
                .unwrap_or(b);
            let token = CompactString::from(format!("{a}{b}"));
            let id = if let Some(&id) = word_to_id.get(&token) {
                stats.reused_ids += 1;
                scan_words |= index.monotone_floor.is_none();
                id
            } else {
                let id = u32::try_from(id_to_word.len())
                    .map_err(|_| "BPE vocabulary exceeds u32::MAX")?;
                if id == NONE {
                    return Err("indexed BPE reserves u32::MAX for word separators".into());
                }
                id_to_word.push(token.clone());
                word_to_id.insert(token, id);
                id
            };
            #[cfg(test)]
            trace.push((top.pair, top.count, id));
            merges.push(top.pair);
            index.apply(top, id, scan_words, max_length, &mut stats)?;
            if let Some(p) = &progress {
                p.inc(1)
            }
            self.emit_json_progress("Compute merges", merges.len(), self.vocab_size);
        }
        self.finalize_progress(&progress, merges.len(), "Compute merges");
        stats.merge_ms = begin.elapsed().as_secs_f64() * 1000.0;
        stats.pruned_pairs = index.pruned_pairs;
        let vocab = word_to_id
            .into_iter()
            .map(|(token, id)| (token.to_string(), id))
            .collect();
        let merges = merges
            .into_iter()
            .map(|(a, b)| {
                (
                    id_to_word[a as usize].to_string(),
                    id_to_word[b as usize].to_string(),
                )
            })
            .collect();
        Ok(IndexedTraining {
            vocab,
            merges,
            special_tokens: self.special_tokens.clone(),
            stats,
            #[cfg(test)]
            trace,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tk_encode::pipeline::Model;

    fn counts(items: &[(&str, u64)]) -> AHashMap<CompactString, u64> {
        items.iter().map(|&(s, n)| (s.into(), n)).collect()
    }

    fn check(trainer: &BpeTrainer, wc: &AHashMap<CompactString, u64>) -> IndexedTraining {
        let mut trace = Vec::new();
        let (vocab, merges, special) = trainer
            .do_train_observed(wc, |p, c, id| trace.push((p, c, id)))
            .unwrap();
        let indexed = trainer.do_train_indexed(wc).unwrap();
        assert_eq!(
            indexed.trace, trace,
            "trace mismatch; words={wc:?}; trainer={trainer:?}"
        );
        assert_eq!(indexed.vocab, vocab);
        assert_eq!(indexed.merges, merges);
        assert_eq!(indexed.special_tokens, special);
        let fused = trainer.do_train_fused(wc).unwrap();
        assert_eq!(
            fused.trace, trace,
            "fused trace mismatch; words={wc:?}; trainer={trainer:?}"
        );
        assert_eq!(fused.vocab, vocab);
        assert_eq!(fused.merges, merges);
        assert_eq!(fused.special_tokens, special);
        let parallel = trainer
            .do_train_indexed_parallel(
                wc,
                IndexedParallelConfig {
                    workers: 2,
                    initialization_workers: None,
                    posting_block_bits: 16,
                    narrow_corpus: true,
                    atomic_corpus: false,
                    batch_size: 256,
                },
            )
            .unwrap();
        assert_eq!(
            parallel.trace, trace,
            "parallel trace mismatch; words={wc:?}; trainer={trainer:?}"
        );
        assert_eq!(parallel.vocab, vocab);
        assert_eq!(parallel.merges, merges);
        assert_eq!(parallel.special_tokens, special);
        let fused_parallel = trainer
            .do_train_indexed_parallel(
                wc,
                IndexedParallelConfig {
                    workers: 4,
                    initialization_workers: None,
                    posting_block_bits: 32,
                    narrow_corpus: false,
                    atomic_corpus: true,
                    batch_size: 256,
                },
            )
            .unwrap();
        assert_eq!(
            fused_parallel.trace, trace,
            "fused parallel trace mismatch; words={wc:?}; trainer={trainer:?}"
        );
        assert_eq!(fused_parallel.vocab, vocab);
        assert_eq!(fused_parallel.merges, merges);
        assert_eq!(fused_parallel.special_tokens, special);
        indexed
    }

    #[test]
    fn weighted_ties_overlap_empty_unicode_and_long_spans() {
        let trainer = BpeTrainer::builder()
            .vocab_size(100)
            .show_progress(false)
            .build();
        check(
            &trainer,
            &counts(&[
                ("", 1),
                ("a", 10),
                ("aaaaaaa", 3),
                ("abababab", 4),
                ("abcabc", 4),
                ("测试测试", 7),
                ("ééé", 2),
            ]),
        );
        let wc = counts(&[("a".repeat(1024).as_str(), 3)]);
        let result = check(&trainer, &wc);
        assert!(result.vocab.contains_key(&"a".repeat(1024)));
        assert_eq!(result.stats.word_scan_steps, 0);
    }

    #[test]
    fn special_token_identity_reuse_and_duplicate_specials() {
        let trainer = BpeTrainer::builder()
            .vocab_size(30)
            .show_progress(false)
            .special_tokens(
                ["ab", "aba", "ab", "aa", "aaaa"]
                    .map(|s| AddedToken::from(s, true))
                    .to_vec(),
            )
            .build();
        let result = check(
            &trainer,
            &counts(&[("abababa", 5), ("aaaaaaa", 7), ("baab", 2)]),
        );
        assert!(result.stats.reused_ids >= 2);
        assert!(result.stats.monotone_pairs);
        assert_eq!(result.stats.word_scan_steps, 0);
        assert!(result.merges.len() > result.vocab.len() - 7);
    }

    #[test]
    fn monotone_pruning_waits_for_global_birth_count() {
        let trainer = BpeTrainer::builder()
            .vocab_size(30)
            .min_frequency(2)
            .show_progress(false)
            .build();
        let result = check(&trainer, &counts(&[("xabp", 1), ("xabq", 1)]));
        assert!(result.stats.monotone_pairs);
        assert!(result.stats.pruned_pairs > 0);
        // Each word contributes only one new (x,ab) edge, but their global
        // birth count is two. Per-word filtering would lose the second rule.
        assert_eq!(
            result.merges,
            vec![("a".into(), "b".into()), ("x".into(), "ab".into())]
        );
        let mut empty_affixes = trainer.clone();
        empty_affixes.continuing_subword_prefix = Some(String::new());
        empty_affixes.end_of_word_suffix = Some(String::new());
        assert!(
            check(&empty_affixes, &counts(&[("xabp", 1), ("xabq", 1)]))
                .stats
                .monotone_pairs
        );
    }

    #[test]
    fn suffix_alias_really_increases_an_old_pair() {
        let trainer = BpeTrainer::builder()
            .vocab_size(10)
            .end_of_word_suffix("a".into())
            .show_progress(false)
            .build();
        let result = check(&trainer, &counts(&[("baaba", 1)]));
        assert!(!result.stats.monotone_pairs);
        assert_eq!(result.stats.pruned_pairs, 0);
        // a=0,b=1,aa=2; the terminal a already has the string identity aa.
        // [b,a,a,b,aa] -> [b,aa,b,aa] makes old (b,aa) grow from one to two.
        assert_eq!(result.trace[0], ((0, 0), 1, 2));
        assert_eq!(result.trace[1], ((1, 2), 2, 3));
        assert!(result.stats.reused_ids > 0);
        assert!(result.stats.word_scan_steps > 0);
    }

    #[test]
    fn affixes_alphabet_and_length_boundaries() {
        let wc = counts(&[
            ("aaaaaaa", 11),
            ("abab", 7),
            ("caba", 4),
            ("测试测试", 3),
            ("ccc", 1),
        ]);
        for prefix in [None, Some("##"), Some("a"), Some("")] {
            for suffix in [None, Some("</w>"), Some("a"), Some("")] {
                for limit in [None, Some(0), Some(1), Some(2), Some(3), Some(5), Some(16)] {
                    let mut trainer = BpeTrainer::builder()
                        .vocab_size(70)
                        .show_progress(false)
                        .special_tokens(vec![AddedToken::from("aa", true)])
                        .build();
                    trainer.continuing_subword_prefix = prefix.map(str::to_owned);
                    trainer.end_of_word_suffix = suffix.map(str::to_owned);
                    trainer.max_token_length = limit;
                    check(&trainer, &wc);
                }
            }
        }
        let trainer = BpeTrainer::builder()
            .vocab_size(30)
            .show_progress(false)
            .limit_alphabet(3)
            .initial_alphabet(['a', '测'].into())
            .build();
        check(&trainer, &wc);
    }

    fn next(rng: &mut u64) -> u64 {
        *rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *rng >> 32
    }

    #[test]
    fn randomized_round_by_round_hf_differential() {
        let mut rng = 2348;
        for case in 0..1500 {
            let mut wc = AHashMap::<CompactString, u64>::new();
            for _ in 0..(1 + next(&mut rng) % 12) {
                let word: String = (0..next(&mut rng) % 24)
                    .map(|_| ['a', 'b', 'c', '测'][next(&mut rng) as usize % 4])
                    .collect();
                *wc.entry(word.into()).or_default() += 1 + next(&mut rng) % 9;
            }
            let mut trainer = BpeTrainer::builder()
                .show_progress(false)
                .vocab_size((4 + next(&mut rng) % 50) as usize)
                .min_frequency(next(&mut rng) % 6)
                .build();
            if case % 3 == 0 {
                trainer.special_tokens = ["aa", "ab", "aba", "abc", "ba", "测测"]
                    .map(|s| AddedToken::from(s, true))
                    .to_vec();
            }
            if case % 4 == 0 {
                trainer.continuing_subword_prefix =
                    Some(["##", "a", "", "测"][(case / 4) % 4].into());
            }
            if case % 5 == 0 {
                trainer.end_of_word_suffix = Some(["</w>", "a", ""][case % 3].into());
            }
            if case % 2 == 0 {
                trainer.max_token_length = Some((next(&mut rng) % 10) as usize);
            }
            check(&trainer, &wc);
        }
    }

    #[test]
    fn encode_and_json_reload_match_reference() {
        let wc = counts(&[("aaaaa", 3), ("abab", 9), ("测试测试", 5), ("cababa", 2)]);
        for affix in [false, true] {
            let mut trainer = BpeTrainer::builder()
                .vocab_size(50)
                .show_progress(false)
                .special_tokens(vec![AddedToken::from("ab", true)])
                .build();
            if affix {
                trainer.continuing_subword_prefix = Some("##".into());
                trainer.end_of_word_suffix = Some("</w>".into());
            }
            let got = check(&trainer, &wc);
            // Persist the exact raw parts the prototype produces, including rank order.
            let tmp = tempfile::NamedTempFile::new().unwrap();
            serde_json::to_writer(tmp.as_file(), &(&got.vocab, &got.merges)).unwrap();
            let (vocab, merges): (Vocab, Merges) =
                serde_json::from_reader(tmp.reopen().unwrap()).unwrap();
            let reloaded = PipelineBPE::from_config(BpeConfig {
                vocab,
                merges,
                ..trainer.model_options()
            })
            .unwrap();
            let (vocab, merges, _) = trainer.do_train(&wc).unwrap();
            let reference = PipelineBPE::from_config(BpeConfig {
                vocab,
                merges,
                ..trainer.model_options()
            })
            .unwrap();
            for input in [
                "",
                "aaaaa",
                "abab",
                "cababa",
                "测试测试",
                "aaab测c",
                "unknown",
            ] {
                let encode = |model: &PipelineBPE| {
                    let mut scratch = model.init_scratch();
                    let mut output = Vec::new();
                    model
                        .tokenize_pipeline(input, &mut scratch, &mut output)
                        .unwrap();
                    output.iter().map(|t| t.id()).collect::<Vec<_>>()
                };
                assert_eq!(encode(&reloaded), encode(&reference), "{input:?}");
            }
        }
    }

    #[test]
    fn fed_input_and_thread_counts_match() {
        let wc = counts(&[
            ("aaaaaaa", 3),
            ("abababab", 7),
            ("cababa", 2),
            ("测试测试", 4),
        ]);
        let trainer = BpeTrainer::builder()
            .vocab_size(50)
            .show_progress(false)
            .build();
        for threads in [1, 2, 4] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    check(&trainer, &wc);
                });
        }
        let mut trainer = trainer;
        trainer
            .feed(["aa aa", "ab aa"].into_iter(), |s| {
                Ok(s.split_whitespace().map(str::to_owned).collect())
            })
            .unwrap();
        let (vocab, merges, special) = trainer.train_vocab().unwrap();
        let got = trainer.train_vocab_indexed().unwrap();
        assert_eq!(got.vocab, vocab);
        assert_eq!(got.merges, merges);
        assert_eq!(got.special_tokens, special);
    }

    #[test]
    fn wide_weights_and_overflow_guard() {
        let trainer = BpeTrainer::builder()
            .vocab_size(10)
            .show_progress(false)
            .build();
        let got = trainer
            .do_train_indexed(&counts(&[("ab", u32::MAX as u64)]))
            .unwrap();
        assert_eq!(got.trace, vec![((0, 1), u32::MAX as u64, 2)]);
        assert!(
            trainer
                .do_train_indexed(&counts(&[("ab", u64::MAX)]))
                .is_err()
        );
    }
}

#[cfg(test)]
mod greedy_tests {
    use super::*;

    // Independent oracle: recompute every live edge and rewrite whole vectors.
    // It intentionally has no heap, historical posting, or delta maintenance.
    fn greedy(trainer: &BpeTrainer, wc: &AHashMap<CompactString, u64>) -> Vec<(Pair, u64, u32)> {
        let mut ids = AHashMap::new();
        let mut strings = Vec::new();
        trainer.add_special_tokens(&mut ids, &mut strings);
        trainer.compute_alphabet(wc, &mut ids, &mut strings);
        let (words, weights) = trainer.tokenize_words(wc, &mut ids, &mut strings, &None);
        let mut words: Vec<Vec<u32>> = words.iter().map(Word::get_chars).collect();
        let mut trace = Vec::new();
        while ids.len() < trainer.vocab_size {
            let mut counts = AHashMap::<Pair, u64>::new();
            for (word, &weight) in words.iter().zip(&weights) {
                for edge in word.windows(2) {
                    *counts.entry((edge[0], edge[1])).or_default() += weight;
                }
            }
            let Some((pair, count)) = counts
                .into_iter()
                .max_by(|(a, ac), (b, bc)| ac.cmp(bc).then_with(|| b.cmp(a)))
            else {
                break;
            };
            if count == 0 || count < trainer.min_frequency {
                break;
            }
            let token = CompactString::from(format!(
                "{}{}",
                strings[pair.0 as usize], strings[pair.1 as usize]
            ));
            let id = if let Some(&id) = ids.get(&token) {
                id
            } else {
                let id = strings.len() as u32;
                strings.push(token.clone());
                ids.insert(token, id);
                id
            };
            trace.push((pair, count, id));
            for word in &mut words {
                let mut output = Vec::new();
                let mut i = 0;
                while i < word.len() {
                    if i + 1 < word.len() && (word[i], word[i + 1]) == pair {
                        output.push(id);
                        i += 2;
                    } else {
                        output.push(word[i]);
                        i += 1;
                    }
                }
                *word = output;
            }
        }
        trace
    }

    #[test]
    fn recomputing_greedy_oracle_without_affixes_or_length_filter() {
        let mut rng = 1400_u64;
        for _ in 0..250 {
            let mut wc = AHashMap::<CompactString, u64>::new();
            for _ in 0..12 {
                let word: String = (0..30)
                    .map(|_| {
                        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                        ['a', 'b', 'c', '测'][(rng >> 32) as usize % 4]
                    })
                    .collect();
                *wc.entry(word.into()).or_default() += 1 + (rng >> 32) % 9;
            }
            let trainer = BpeTrainer::builder()
                .vocab_size(100)
                .show_progress(false)
                .build();
            let out = trainer.do_train_indexed(&wc).unwrap();
            assert_eq!(out.trace, greedy(&trainer, &wc), "words={wc:?}");
        }
    }
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;

    #[test]
    fn reused_id_resurrects_low_frequency_pair_and_keeps_hf_cohorts() {
        // A synthetic index state, NOT a reachable ordinary-character BPE
        // counterexample (see PAIR_MONOTONICITY.md). It checks the general
        // ledger's handling of aliases, which affixes can actually produce.
        // One word still contains IDs (0,1,2), another
        // already contains the identity 3 that (0,1) will reuse. The old (3,2)
        // frequency is below min_frequency=2 and must remain indexed.
        let mut index = Index::new(
            vec![NONE, 0, 1, 2, NONE, 3, 2, NONE],
            vec![1, 5],
            vec![3, 1],
            vec![1; 4],
            None,
        )
        .unwrap();
        let select = |index: &mut Index| -> Candidate {
            loop {
                let mut top = index.queue.pop().unwrap();
                let count = index.counts[&top.pair] as u64;
                if top.count == count {
                    return top;
                }
                top.count = count;
                index.queue.push(top);
            }
        };
        let mut stats = IndexedTrainingStats::default();
        let first = select(&mut index);
        assert_eq!(first.pair, (0, 1));
        assert!(index.spans.is_none());
        index.apply(first, 3, true, usize::MAX, &mut stats).unwrap();
        // Same ID, distinct physical spans: the merged occurrence is two
        // characters, while the other word's preexisting occurrence is one.
        assert_eq!(index.span(1), 2);
        assert_eq!(index.span(5), 1);
        assert_eq!(index.inspect(1, (3, 2)).unwrap().after, 4);
        assert_eq!(index.inspect(5, (3, 2)).unwrap().after, 7);
        assert_eq!(index.counts[&(3, 2)], 4);
        let birth_cohort = select(&mut index);
        assert_eq!((birth_cohort.pair, birth_cohort.count), ((3, 2), 4));
        index
            .apply(birth_cohort, 4, true, usize::MAX, &mut stats)
            .unwrap();
        assert_eq!(index.corpus[index.pivots[1] as usize], 3);
        // HF's older candidate owns the other word, and keeps its global ledger
        // snapshot even after the first cohort was consumed. It is still used.
        let old_cohort = select(&mut index);
        assert_eq!((old_cohort.pair, old_cohort.count), ((3, 2), 4));
        index
            .apply(old_cohort, 4, true, usize::MAX, &mut stats)
            .unwrap();
        assert_eq!(index.corpus[index.pivots[1] as usize], 4);
    }
}
