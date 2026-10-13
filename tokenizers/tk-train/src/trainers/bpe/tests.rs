//! Differential semantics, explicit numeric contracts, and public integration.
use super::*;
use crate::Trainer;
use ahash::AHashMap;
use compact_str::CompactString;
use rayon::iter::ParallelIterator;
use std::time::{Duration, Instant};

pub(super) fn counts(items: &[(&str, u64)]) -> AHashMap<CompactString, u64> {
    items
        .iter()
        .map(|&(word, weight)| (word.into(), weight))
        .collect()
}

pub(super) fn trainer() -> BpeTrainer {
    BpeTrainer::builder()
        .vocab_size(64)
        .min_frequency(1)
        .show_progress(false)
        .build()
}

fn check(trainer: &BpeTrainer, words: &AHashMap<CompactString, u64>) -> Vec<(Pair, u64, u32)> {
    let expected = trainer.do_train_reference(words).unwrap();
    for workers in [1, 4, 8] {
        let actual = trainer
            .do_train_impl(WordCountsView::from_map(words), Some(workers))
            .unwrap();
        assert_eq!(actual.trace, expected.trace, "workers={workers}");
        let (vocab, merges) = actual.into_parts();
        assert_eq!(vocab, expected.parts.0, "workers={workers}");
        assert_eq!(merges, expected.parts.1, "workers={workers}");
    }
    expected.trace
}

pub(super) fn initialize(
    trainer: &BpeTrainer,
    words: WordCountsView<'_>,
    reuse: bool,
) -> (Vocabulary, Corpus, PairIndex) {
    let alphabet = vocabulary::select_alphabet(trainer, words);
    let (vocabulary, ids) = Vocabulary::initialize(trainer, words, &alphabet, &None).unwrap();
    let plan = CorpusPlan::build(words, ids, trainer, reuse).unwrap();
    let index = PairIndex::build(&plan, trainer.min_frequency.max(1), &None).unwrap();
    (vocabulary, plan.materialize(), index)
}

pub(super) fn tokens(corpus: &Corpus) -> Vec<u32> {
    corpus.tokens().collect()
}

pub(super) fn spans(corpus: &Corpus) -> Vec<usize> {
    corpus.spans().collect()
}

#[test]
fn models_match_reference() {
    let mut seed = 0x58e4_a91du64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let alphabet = ['a', 'b', 'c', '猫', '\u{301}', '🙂'];
    for case in 0..64 {
        let mut words = AHashMap::with_hasher(ahash::RandomState::with_seeds(11, 13, 17, 19));
        for _ in 0..24 {
            let length = next() as usize % if case % 8 == 0 { 256 } else { 32 };
            let text: String = (0..length)
                .map(|_| alphabet[next() as usize % alphabet.len()])
                .collect();
            *words.entry(text.into()).or_default() += next() % 8;
        }
        let mut trainer = trainer();
        trainer.min_frequency = next() % 4;
        trainer.max_token_length =
            [None, Some(0), Some(1), Some(2), Some(3), Some(9)][case as usize % 6];
        trainer.continuing_subword_prefix = [None, Some(""), Some("##"), Some("a"), Some("ab")]
            [case as usize % 5]
            .map(str::to_owned);
        trainer.end_of_word_suffix =
            [None, Some(""), Some("</w>"), Some("a")][case as usize / 4 % 4].map(str::to_owned);
        trainer.special_tokens = ["ab", "aa", "##a", "a🙂", "ab"]
            .into_iter()
            .take(case as usize % 6)
            .map(|s| AddedToken::from(s, true))
            .collect();
        trainer.limit_alphabet = [None, Some(3)][case as usize % 2];
        trainer.initial_alphabet = ['a', '猫'].into();
        check(&trainer, &words);
    }
}

#[test]
fn overlap_and_alias() {
    check(
        &trainer(),
        &counts(&[("abac", 4), ("dbdc", 4), ("abdb", 3), ("acdc", 3)]),
    );
    check(
        &trainer(),
        &counts(&[("ab", 3), ("cd", 3), ("aaaaa", 0), ("", 7)]),
    );
    // 4097 AA starts cross the 4096-start task boundary, even with one worker.
    let words = counts(&[
        (&"ab".repeat(12_000), 1),
        ("xy", 12_000),
        (&"a".repeat(8195), 3),
    ]);
    let mut trainer = trainer();
    trainer.min_frequency = 10_000;
    check(&trainer, &words);
    trainer.min_frequency = 1;
    trainer.end_of_word_suffix = Some("a".into());
    let trace = check(&trainer, &counts(&[("baaba", 1)]));
    assert_eq!(&trace[..2], &[((0, 0), 1, 2), ((1, 2), 2, 3)]);
}

#[test]
fn wide_ids() {
    let mut trainer = trainer();
    trainer.vocab_size = 65550;
    trainer.special_tokens = std::iter::once("a".to_owned())
        .chain((1..65536).map(|i| format!("<reserved{i}>")))
        .map(|s| AddedToken::from(s, true))
        .collect();
    let words = counts(&[("abab", 3), ("bbaa", 2)]);
    check(&trainer, &words);
    assert_eq!(trainer.do_train(&words).unwrap().0["b"], 65536);
}

#[test]
fn count_boundaries() {
    let mut trainer = trainer();
    for weight in [u32::MAX as u64 + 17, u64::MAX] {
        let result = trainer
            .do_train_impl(
                WordCountsView::from_map(&counts(&[("ab", weight)])),
                Some(2),
            )
            .unwrap();
        assert_eq!(result.trace, [((0, 1), weight, 2)]);
    }
    for target in [0, 2, 64] {
        trainer.vocab_size = target;
        for (word, weight, suffix, succeeds) in [
            ("aba", u64::MAX, None, true),
            ("abab", u64::MAX, None, false),
            ("ab", i64::MAX as u64, Some("a"), true),
            ("ab", i64::MAX as u64 + 1, Some("a"), false),
            ("abc", i64::MAX as u64, Some("a"), false),
        ] {
            trainer.end_of_word_suffix = suffix.map(str::to_owned);
            assert_eq!(
                trainer.do_train(&counts(&[(word, weight)])).is_ok(),
                succeeds,
                "target={target}, word={word}, weight={weight}"
            );
        }
    }
}

#[test]
fn inputs_remain_reusable() {
    let words = counts(&[("ab测", 2), ("测", 1)]);
    let original = words.clone();
    let mut trainer = trainer();
    trainer
        .feed(["ab测", "测", "ab测"].into_iter(), |s| {
            Ok(vec![s.into()])
        })
        .unwrap();
    let stored = trainer.clone();
    for _ in 0..2 {
        trainer.do_train(&words).unwrap();
        trainer.train_vocab().unwrap();
        assert_eq!(trainer, stored);
    }
    assert_eq!(words, original);
}

#[test]
fn failed_training_keeps_model() {
    let mut model = PipelineBPE::from_config(BpeConfig {
        vocab: [("old".into(), 0)].into(),
        ..Default::default()
    })
    .unwrap();
    let original = model.to_config().unwrap().vocab;
    let mut trainer = trainer();
    trainer.words = WordCounts::from_map(counts(&[("abab", u64::MAX)]));
    assert!(trainer.train(&mut model).is_err());
    assert_eq!(model.to_config().unwrap().vocab, original);

    trainer.words = WordCounts::from_map(counts(&[("ab", 1)]));
    trainer.end_of_word_suffix = Some("s".repeat(64));
    // Training succeeds, but the model reader rejects its oversized affix.
    assert!(trainer.train_vocab().is_ok());
    assert!(
        trainer
            .train(&mut model)
            .unwrap_err()
            .to_string()
            .contains("affixes too long")
    );
    assert_eq!(model.to_config().unwrap().vocab, original);
}

#[test]
fn restart_keeps_alphabet() {
    let mut trainer = trainer();
    trainer.end_of_word_suffix = Some("a".into());
    trainer.limit_alphabet = Some(3);
    trainer.initial_alphabet = ['a', 'b'].into();
    // Four characters exceed the cutoff; q/z tie for the one unforced slot.
    let words = counts(&[("baaba", 1), ("q", 1), ("z", 1)]);
    let view = WordCountsView::from_map(&words);
    let retained = vocabulary::select_alphabet(&trainer, view);
    let selected: Vec<_> = retained
        .characters()
        .iter()
        .map(|entry| entry.character)
        .collect();
    assert_eq!(selected.len(), 3);
    assert!(selected.contains(&'a') && selected.contains(&'b'));
    assert_ne!(selected.contains(&'q'), selected.contains(&'z'));
    assert!(matches!(
        trainer
            .train_attempt(view, false, &retained, &None)
            .unwrap(),
        AttemptOutcome::RestartForReuse
    ));
    let AttemptOutcome::Complete(result) =
        trainer.train_attempt(view, true, &retained, &None).unwrap()
    else {
        panic!("reuse must complete")
    };
    let mut oracle = trainer.clone();
    oracle.initial_alphabet = selected.into_iter().collect();
    let expected = oracle.do_train_reference(&words).unwrap();
    assert_eq!(result.trace, check(&oracle, &words));
    assert_eq!(result.into_parts(), (expected.parts.0, expected.parts.1));
}

#[test]
fn restart_discards_ids() {
    let mut trainer = trainer();
    trainer.continuing_subword_prefix = Some("a".into());
    trainer.special_tokens = vec![AddedToken::from("xy", true)];
    let words = counts(&[("xy", 3), ("uv", 2), ("ab", 1)]);
    let (mut vocabulary, mut corpus, mut index) =
        initialize(&trainer, WordCountsView::from_map(&words), false);
    let MergeOutcome::Committed(accepted) = trainer
        .merge_next(&mut vocabulary, &mut corpus, &mut index)
        .unwrap()
    else {
        panic!("reserved XY must commit")
    };
    assert_eq!(accepted, 1);
    assert_eq!((vocabulary.trace[0].1, vocabulary.trace[0].2), (3, 0));
    let committed_len = vocabulary.len();
    // UV allocates an ID before AB aliases activated B; both speculative rules
    // and the previously committed attempt must be discarded by the restart.
    assert!(matches!(
        trainer
            .merge_next(&mut vocabulary, &mut corpus, &mut index)
            .unwrap(),
        MergeOutcome::RestartForReuse
    ));
    assert_eq!(vocabulary.len(), committed_len + 1);
    let trace = check(&trainer, &words);
    assert_eq!(trace.iter().filter(|rule| rule.1 == 3).count(), 1);
}

#[test]
fn tokenizer_roundtrip() {
    use tk_encode::{
        models::bpe::{BpeConfig, PipelineBPE},
        pipeline::{EncodeOptions, PipelineModel, PipelinePreTokenizer, PipelineTokenizer},
    };
    let mut trainer = trainer();
    trainer.continuing_subword_prefix = Some("##".into());
    trainer.end_of_word_suffix = Some("</w>".into());
    trainer.special_tokens = vec![AddedToken::from("[UNK]", true)];
    trainer
        .feed(["ab测 测", "ab测"].into_iter(), |s| {
            Ok(s.split_whitespace().map(str::to_owned).collect())
        })
        .unwrap();
    let restored: BpeTrainer =
        serde_json::from_value(serde_json::to_value(&trainer).unwrap()).unwrap();
    assert_eq!(trainer, restored);
    let mut model = PipelineBPE::from_config(BpeConfig {
        vocab: [("old".into(), 0)].into(),
        ..Default::default()
    })
    .unwrap();
    assert_eq!(trainer.train(&mut model).unwrap(), trainer.special_tokens);
    let config = model.to_config().unwrap();
    let expected = [
        ("测", vec![config.vocab["测</w>"]]),
        ("a测", vec![config.vocab["a"], config.vocab["##测</w>"]]),
        ("ab测", vec![config.vocab["ab测</w>"]]),
    ];
    let tokenizer = PipelineTokenizer::from_parts(
        Default::default(),
        vec![],
        PipelinePreTokenizer::None,
        PipelineModel::BPE(model),
        Default::default(),
        None,
        Default::default(),
        None,
        None,
    );
    let json = tk_serialize::to_json(&tokenizer).unwrap();
    let written: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(written["model"]["continuing_subword_prefix"], "##");
    assert_eq!(written["model"]["end_of_word_suffix"], "</w>");
    let reloaded = tk_serialize::from_json(&json).unwrap();
    let rewritten: serde_json::Value =
        serde_json::from_str(&tk_serialize::to_json(&reloaded).unwrap()).unwrap();
    assert_eq!(rewritten["model"], written["model"]);
    for tokenizer in [&tokenizer, &reloaded] {
        for (input, expected) in &expected {
            let encoded = tokenizer
                .encode(*input, &EncodeOptions::no_specials())
                .wait()
                .unwrap();
            assert_eq!(encoded[0].ids(), expected.as_slice(), "{input}");
        }
    }
}

// Child processes isolate public settings; their ambient pool deliberately differs
// from the pool requested for training. Both fresh and alias preparation execute.
use std::sync::atomic::{AtomicUsize, Ordering};
static EXPECTED_WORKERS: AtomicUsize = AtomicUsize::new(0);
static OBSERVED_PHASES: AtomicUsize = AtomicUsize::new(0);
/// Worker-executed stages observed by the public training subprocess test.
/// These markers check the requested pool inside materialization and both
/// preparation paths, rather than inferring it from the caller's ambient pool.
pub(super) enum Phase {
    Materialize,
    FreshPrepare,
    ReusePrepare,
}

pub(super) fn observe_worker(phase: Phase) {
    let expected = EXPECTED_WORKERS.load(Ordering::Relaxed);
    if expected != 0 {
        assert_eq!(rayon::current_num_threads(), expected);
        OBSERVED_PHASES.fetch_or(1 << phase as usize, Ordering::Relaxed);
    }
}

fn run_child(test: &str, child: &str, setting: &str) -> std::process::Output {
    let result = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test, "--nocapture"])
        .env(child, setting)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{setting}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    result
}

#[test]
fn public_feed_pools_and_progress() {
    const CHILD: &str = "BPE_PUBLIC_TEST_CHILD";
    const TEST: &str = "trainers::bpe::tests::public_feed_pools_and_progress";
    if let Ok(setting) = std::env::var(CHILD) {
        run_public_training_child(&setting);
        return;
    }
    for setting in ["parallel", "serial", "one", "zero", "silent", "restart"] {
        check_progress_records(setting, run_child(TEST, CHILD, setting));
    }
}

// The child owns process-wide parallelism settings; the parent checks its output.
fn run_public_training_child(setting: &str) {
    use tk_encode::utils::progress::ProgressFormat;
    let parallel = setting != "serial";
    let workers = if setting == "one" { 1 } else { 4 };
    tk_encode::parallelism::set_num_threads(workers);
    tk_encode::parallelism::set_parallelism(parallel);
    EXPECTED_WORKERS.store(if parallel { workers } else { 1 }, Ordering::Relaxed);
    rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap()
        .install(|| {
            if matches!(setting, "parallel" | "serial" | "one") {
                check_feed_success(2, parallel);
                if setting == "parallel" {
                    check_feed_partial_failure();
                }
            }
            let mut trainer = trainer();
            trainer.progress_format = if setting == "silent" {
                ProgressFormat::Silent
            } else {
                ProgressFormat::JsonLines
            };
            trainer.vocab_size = if setting == "zero" { 2 } else { 64 };
            trainer.continuing_subword_prefix = Some("a".into());
            trainer.special_tokens = vec![AddedToken::from("xy", true)];
            let words = counts(&[("xy", 3), ("uv", 2), ("ab", 1)]);
            OBSERVED_PHASES.store(0, Ordering::Relaxed);
            let (_, merges, _) = trainer.do_train(&words).unwrap();
            assert_eq!(
                OBSERVED_PHASES.load(Ordering::Relaxed),
                if setting == "zero" { 0 } else { 0b111 }
            );
            println!("BPE_MERGE_COUNT={}", merges.len());
        });
}

fn check_progress_records(setting: &str, result: std::process::Output) {
    let output = String::from_utf8(result.stderr).unwrap();
    if setting == "silent" {
        assert!(output.is_empty());
        return;
    }
    let records: Vec<serde_json::Value> = output
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    let stdout = String::from_utf8(result.stdout).unwrap();
    let count: u64 = stdout
        .lines()
        .find_map(|s| s.strip_prefix("BPE_MERGE_COUNT="))
        .unwrap()
        .parse()
        .unwrap();
    let finished = records
        .iter()
        .rev()
        .find(|r| r["stage"] == "Compute merges")
        .unwrap();
    assert_eq!(finished["current"], count);
    assert_eq!(finished["total"], count);
    if setting == "zero" {
        assert_eq!(count, 0);
    }
}

mod reference {
    //! Small-input oracle with sequential word edits and historical birth cohorts.
    //! It shares alphabet selection, but no engine index, codec, matching or batch code.
    use super::*;
    use dary_heap::OctonaryHeap;
    use indexmap::IndexSet;
    use std::cmp::{Ordering, Reverse};

    /// Oracle queue snapshot with the historical set of words containing the pair.
    /// Sequential word edits update a separate ledger used to repair its queued count.
    struct Merge {
        pair: Pair,
        count: u64,
        words: AHashSet<usize>,
    }

    impl Ord for Merge {
        fn cmp(&self, other: &Self) -> Ordering {
            (self.count, Reverse(self.pair)).cmp(&(other.count, Reverse(other.pair)))
        }
    }

    impl PartialOrd for Merge {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl PartialEq for Merge {
        fn eq(&self, other: &Self) -> bool {
            self.cmp(other) == Ordering::Equal
        }
    }

    impl Eq for Merge {}

    type Word = Vec<(u32, usize)>;

    // Literal left-to-right edits preserve intermediate removals and births. Net
    // recounting would lose alias cohorts and the strict neighbor-length birth gate.
    fn merge_word(word: &mut Word, pair: Pair, id: u32, limit: usize) -> Vec<(Pair, i64)> {
        let mut changes = Vec::new();
        let mut i = 0;
        while i + 1 < word.len() {
            if (word[i].0, word[i + 1].0) == pair {
                let length = word[i].1 + word[i + 1].1;
                if i > 0 {
                    changes.push(((word[i - 1].0, pair.0), -1));
                    if word[i - 1].1 + length < limit {
                        changes.push(((word[i - 1].0, id), 1));
                    }
                }
                word[i] = (id, length);
                word.remove(i + 1);
                if i + 1 < word.len() {
                    changes.push(((pair.1, word[i + 1].0), -1));
                    if length + word[i + 1].1 < limit {
                        changes.push(((id, word[i + 1].0), 1));
                    }
                }
            }
            i += 1;
        }
        changes
    }

    impl BpeTrainer {
        pub(super) fn do_train_reference(
            &self,
            counts: &AHashMap<CompactString, u64>,
        ) -> Result<TrainingResult> {
            // 1. Insert special tokens and the selected alphabet before word tokenization.
            let mut tokens = IndexSet::new();
            for token in &self.special_tokens {
                tokens.insert(token.content.clone());
            }
            for entry in
                vocabulary::select_alphabet(self, WordCountsView::from_map(counts)).characters()
            {
                tokens.insert(entry.character.to_string());
            }

            // 2. Tokenize in input traversal order, allocating decorated IDs as seen.
            let (mut words, weights) = initialize_words(self, counts, &mut tokens);

            // 3. Count initial edges and retain independent historical word cohorts.
            let mut ledger = AHashMap::<Pair, i64>::new();
            let mut births = AHashMap::<Pair, AHashSet<usize>>::new();
            for (i, word) in words.iter().enumerate() {
                for edge in word.windows(2) {
                    let pair = (edge[0].0, edge[1].0);
                    *ledger.entry(pair).or_default() += weights[i];
                    births.entry(pair).or_default().insert(i);
                }
            }
            let mut queue = OctonaryHeap::new();
            let mut merges = Vec::new();
            let mut trace = Trace::new();

            // 4. Repair priorities and apply one rule at a time with literal word edits.
            loop {
                for (pair, words) in births.drain() {
                    if ledger[&pair] > 0 {
                        queue.push(Merge {
                            pair,
                            count: ledger[&pair] as u64,
                            words,
                        });
                    }
                }
                if tokens.len() >= self.vocab_size {
                    break;
                }
                let Some(mut top) = queue.pop() else { break };
                if top.count != ledger[&top.pair] as u64 {
                    top.count = ledger[&top.pair] as u64;
                    queue.push(top);
                    continue;
                }
                if top.count < self.min_frequency.max(1) {
                    break;
                }
                let mut right = tokens[top.pair.1 as usize].as_str();
                if let Some(prefix) = &self.continuing_subword_prefix {
                    right = right.strip_prefix(prefix).unwrap_or(right);
                }
                let token = format!("{}{right}", tokens[top.pair.0 as usize]);
                let id = tokens.insert_full(token).0 as u32;
                merges.push((
                    tokens[top.pair.0 as usize].clone(),
                    tokens[top.pair.1 as usize].clone(),
                ));
                trace.push((top.pair, top.count, id));
                for i in top.words {
                    for (pair, change) in merge_word(
                        &mut words[i],
                        top.pair,
                        id,
                        self.max_token_length.unwrap_or(usize::MAX),
                    ) {
                        *ledger.entry(pair).or_default() += change * weights[i];
                        if change > 0 {
                            births.entry(pair).or_default().insert(i);
                        }
                    }
                }
            }
            let vocab = tokens
                .into_iter()
                .enumerate()
                .map(|(id, token)| (token, id as u32))
                .collect();
            Ok(TrainingResult {
                parts: (vocab, merges, self.special_tokens.clone()),
                trace,
            })
        }
    }

    fn initialize_words(
        trainer: &BpeTrainer,
        counts: &AHashMap<CompactString, u64>,
        tokens: &mut IndexSet<String>,
    ) -> (Vec<Word>, Vec<i64>) {
        let mut weights = Vec::new();
        let mut words = Vec::new();
        for (text, &weight) in counts {
            let mut word = Vec::new();
            let length = text.chars().count();
            for (i, character) in text.chars().enumerate() {
                let mut token = character.to_string();
                if !tokens.contains(&token) {
                    continue;
                }
                if i != 0
                    && let Some(prefix) = &trainer.continuing_subword_prefix
                {
                    token.insert_str(0, prefix);
                }
                if i + 1 == length
                    && let Some(suffix) = &trainer.end_of_word_suffix
                {
                    token.push_str(suffix);
                }
                word.push((tokens.insert_full(token).0 as u32, 1));
            }
            words.push(word);
            weights.push(i64::try_from(weight).expect("oracle fixtures use small weights"));
        }
        (words, weights)
    }
}

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

#[test]
fn initial_edges() {
    let words = counts(&[("qé🙂z", 7), ("é🙂é", 6), ("🙂é", 5), ("", 4), ("é", 3)]);
    let mut trainer = trainer();
    trainer.limit_alphabet = Some(2);
    let view = WordCountsView::from_map(&words);
    let (_, ids) = Vocabulary::initialize(
        &trainer,
        view,
        &vocabulary::select_alphabet(&trainer, view),
        &None,
    )
    .unwrap();
    let plan = CorpusPlan::build(view, ids, &trainer, true).unwrap();
    let expected = [
        ((0, 1), 1, 7),
        ((0, 1), 4, 6),
        ((1, 0), 5, 6),
        ((1, 0), 8, 5),
    ];
    let read = || {
        let (words, edges) = plan
            .initial_chunks(usize::MAX)
            .find_first(|_| true)
            .unwrap();
        assert_eq!(words, 5);
        edges.map(|edge| (edge.pair, edge.position, edge.weight))
    };
    let mut actual: Vec<_> = read().collect();
    assert_eq!(actual, expected);
    actual.clear();
    let mut edges = read();
    let result: Result<()> = edges.by_ref().try_for_each(|edge| {
        actual.push(edge);
        if actual.len() == 2 {
            Err("designated edge failure".into())
        } else {
            Ok(())
        }
    });
    assert_eq!(result.unwrap_err().to_string(), "designated edge failure");
    assert_eq!(actual, expected[..2]);
    assert_eq!(edges.collect::<Vec<_>>(), expected[2..]);
}

fn check_feed_success(workers: usize, parallel: bool) {
    let length = workers + 33;
    let reads = AtomicUsize::new(0);
    let seen: Vec<_> = (0..length).map(|_| AtomicUsize::new(0)).collect();
    let participants = AtomicUsize::new(0);
    let mut index = 0;
    // The iterator resumes after None; Feed must stop at the first exhaustion.
    let input = std::iter::from_fn(|| {
        reads.fetch_add(1, Ordering::Relaxed);
        let value = index;
        index += 1;
        (value != length && value <= length + 1).then(|| value.to_string())
    });
    let counted = super::feed::count(input, &|text| {
        assert_eq!(rayon::current_num_threads(), workers);
        participants.fetch_or(
            1 << rayon::current_thread_index().unwrap(),
            Ordering::Relaxed,
        );
        let id = text.parse::<usize>().unwrap();
        seen[id].fetch_add(1, Ordering::Relaxed);
        Ok(if id % 3 == 0 {
            vec![]
        } else {
            vec![format!("词🙂{id}"), "repeat".into(), "repeat".into()]
        })
    })
    .unwrap();
    assert_eq!(reads.load(Ordering::Relaxed), length + 1);
    assert!(seen.iter().all(|n| n.load(Ordering::Relaxed) == 1));
    assert!(parallel || participants.load(Ordering::Relaxed).count_ones() <= 1);
    let mut expected = AHashMap::<CompactString, u64>::new();
    for id in (0..length).filter(|id| id % 3 != 0) {
        expected.insert(format!("词🙂{id}").into(), 1);
        *expected.entry("repeat".into()).or_default() += 2;
    }
    assert_eq!(counted, WordCounts::from_map(expected));
    for unique in [2047, 2048, 2049, 4097] {
        let mut words: Vec<_> = (0..unique).map(|id| format!("key{id}")).collect();
        words.extend(["key0".into(), format!("key{}", unique - 1), "key0".into()]);
        let counted = super::feed::count(std::iter::once("input"), &|_| Ok(words.clone())).unwrap();
        let mut expected = AHashMap::<CompactString, u64>::new();
        for word in &words {
            *expected.entry(word.as_str().into()).or_default() += 1;
        }
        assert_eq!(counted, WordCounts::from_map(expected));
    }
}

fn check_feed_partial_failure() {
    let mut trainer = trainer();
    trainer
        .feed(std::iter::once("saved"), |s| Ok(vec![s.into()]))
        .unwrap();
    let previous = trainer.clone();
    let participants = AtomicUsize::new(0);
    let first_worker = AtomicUsize::new(usize::MAX);
    let checkpoint = AtomicUsize::new(0);
    let error = trainer
        .feed(["0", "1", "2"].into_iter(), |text| {
            let id = text.parse::<usize>().unwrap();
            let worker = rayon::current_thread_index().unwrap();
            if id < 2 {
                participants.fetch_or(1 << worker, Ordering::Relaxed);
            }
            let deadline = Instant::now() + Duration::from_secs(5);
            match id {
                0 => {
                    first_worker.store(worker, Ordering::Relaxed);
                    while participants.load(Ordering::Relaxed).count_ones() < 2 {
                        assert!(
                            Instant::now() < deadline,
                            "startup callbacks were serialized"
                        );
                        std::thread::yield_now();
                    }
                    // Returning allows this fold to publish full caches in A,
                    // or individual updates in B, before its next callback.
                    Ok((0..4097).map(|n| format!("published{n}")).collect())
                }
                1 => {
                    while checkpoint.load(Ordering::Relaxed) == 0 {
                        assert!(Instant::now() < deadline, "successful fold did not advance");
                        std::thread::yield_now();
                    }
                    Err("failure after shared updates".into())
                }
                2 => {
                    assert_eq!(worker, first_worker.load(Ordering::Relaxed));
                    checkpoint.store(1, Ordering::Relaxed);
                    // This successful fold also has a nonempty tail. The other
                    // fold fails; count must not replace stored trainer words.
                    Ok(vec!["successful-tail".into()])
                }
                _ => unreachable!(),
            }
        })
        .unwrap_err();
    assert_eq!(error.to_string(), "failure after shared updates");
    assert_eq!(trainer, previous);
}
