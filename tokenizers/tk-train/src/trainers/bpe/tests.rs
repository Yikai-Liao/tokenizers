//! Differential semantics, explicit numeric contracts, and public integration.
use super::*;
use crate::Trainer;
use ahash::AHashMap;
use compact_str::CompactString;
use std::time::{Duration, Instant};

// Explicit pools belong to tests and callers, never to the trainer.
impl BpeTrainer {
    fn do_train_with_workers(
        &self,
        words: WordCountsView<'_>,
        workers: usize,
    ) -> Result<Vocabulary> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()?
            .install(|| self.do_train_impl(words))
    }
}

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
            .do_train_with_workers(WordCountsView::from(words), workers)
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
    let alphabet = trainer.select_alphabet(words);
    let (vocabulary, ids) = Vocabulary::initialize(trainer, words, &alphabet, &None).unwrap();
    let plan = CorpusPlan::build(
        words,
        ids,
        trainer.vocab_size,
        if reuse {
            TrainingMode::Reuse
        } else {
            TrainingMode::Fresh
        },
        trainer.max_token_length,
    )
    .unwrap();
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
fn parity() {
    let mut seed = 0x58e4_a91du64;
    // Fixed-seed xorshift64 keeps generated cases reproducible. The three
    // shifts mix the same state; discarding shifted-out bits is intentional.
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let alphabet = ['a', 'b', 'c', '猫', '\u{301}', '🙂'];
    // Include case 84, whose reuse cohorts retain consumed interior starts.
    for case in 0..128 {
        let mut words = AHashMap::with_hasher(ahash::RandomState::with_seeds(11, 13, 17, 19));
        for _ in 0..24 {
            // Every eighth case allows longer words to cross restart boundaries;
            // the other cases keep small words and frequent alias interactions.
            let length = next() as usize % if case % 8 == 0 { 256 } else { 32 };
            let text: String = (0..length)
                .map(|_| alphabet[next() as usize % alphabet.len()])
                .collect();
            *words.entry(text.into()).or_default() += next() % 8;
        }
        let mut trainer = trainer();
        trainer.min_frequency = next() % 4;
        // Cycle length gates and affixes at different strides so the fixed
        // corpus cases exercise varied combinations. Suffix changes every four
        // cases, instead of staying locked to the faster prefix/limit cycles.
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
fn targeted_models() {
    check(
        &trainer(),
        &counts(&[("abacdbdc", 2), ("ab", 10), ("ac", 9), ("db", 8), ("dc", 7)]),
    );
    // Distinct lengths keep all forty words present across the weight regions.
    let words = (0..40)
        .map(|i| {
            (
                format!("{}ababa", "x".repeat(i)).into(),
                [9, 1, 1, 0][i / 10],
            )
        })
        .collect();
    check(&trainer(), &words);
    let mut trainer = trainer();
    trainer.end_of_word_suffix = Some("a".into());
    for limit in [None, Some(3), Some(usize::MAX)] {
        trainer.max_token_length = limit;
        check(&trainer, &counts(&[("baaaaaba", 2), ("ba", 1)]));
    }
    trainer.end_of_word_suffix = None;
    trainer.max_token_length = None;
    trainer.limit_alphabet = Some(2);
    let words = counts(&[("qé🙂z", 7), ("é🙂é", 6), ("🙂é", 5), ("", 4), ("é", 3)]);
    let (vocab, _, _) = trainer.do_train(&words).unwrap();
    // The oracle shares alphabet selection; keep an independent cutoff expectation.
    assert_eq!((vocab["é"], vocab["🙂"]), (0, 1));
    check(&trainer, &words);
    check(&trainer, &counts(&[]));
}

#[test]
fn fragmented_birth_floor() {
    // Each parallel AB fragment is below the floor, while the complete birth
    // (AB, c) reaches it. Only whole producers may prune before aggregation.
    let mut trainer = trainer();
    trainer.min_frequency = 3072;
    let word = "abc".repeat(4096);
    let trace = check(&trainer, &counts(&[(&word, 1)]));
    assert_eq!(&trace[..2], &[((0, 1), 4096, 3), ((3, 2), 4096, 4)]);
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
    // The long odd AA run crosses preparation tasks, even with one worker.
    let words = counts(&[
        (&"ab".repeat(12_000), 1),
        ("xy", 12_000),
        (&"a".repeat(20001), 3),
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
fn reuse_stale_starts() {
    let mut trainer = trainer();
    trainer.continuing_subword_prefix = Some("ab".into());
    let words = counts(&[("cbbbbb", 6), ("aba", 4)]);
    // The later ab + aba identity collision restarts this attempt in reuse mode.
    // Before that collision, c + abbb consumes a historical abbb + abb start;
    // its predecessor is now the cbb head, rather than a token tail.
    let trace = check(&trainer, &words);
    assert_eq!(
        trace.iter().map(|rule| rule.1).collect::<Vec<_>>(),
        [24, 6, 6, 6, 4, 4]
    );
    assert_eq!(
        trainer.do_train(&words).unwrap().1,
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

#[test]
fn reuse_stale_models() {
    let mut trainer = trainer();
    trainer.continuing_subword_prefix = Some("ab".into());
    let words = counts(&[("aba", 7), ("bbbbbbbaba", 6)]);
    // Seeking a consumed historical start also used to invent a left neighbor
    // without panicking, publishing aba + abb instead of aba + abba.
    check(&trainer, &words);
    assert_eq!(
        trainer.do_train(&words).unwrap().1,
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
            .do_train_with_workers(WordCountsView::from(&counts(&[("ab", weight)])), 2)
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
fn vocabulary_bounds() {
    let mut trainer = trainer();
    let words = counts(&[("ab", 1)]);
    trainer.vocab_size = u32::MAX as usize;
    assert_eq!(trainer.do_train(&words).unwrap().0.len(), 3);
    assert!(trainer.do_train(&counts(&[])).unwrap().0.is_empty());
    if usize::BITS > 32 {
        trainer.vocab_size = usize::MAX;
        assert!(
            trainer
                .do_train(&counts(&[]))
                .unwrap_err()
                .to_string()
                .contains("target vocabulary size exceeds")
        );
    }
}

#[test]
fn limited_alphabet_wide_counts() {
    let mut trainer = trainer();
    trainer.limit_alphabet = Some(1);
    let words = counts(&[("aa", u64::MAX), ("b", u64::MAX)]);
    let (vocab, merges, _) = trainer.do_train(&words).unwrap();
    assert_eq!(vocab, [("a".into(), 0), ("aa".into(), 1)].into());
    assert_eq!(merges, [("a".into(), "a".into())]);
    trainer.initial_alphabet = ['b'].into();
    assert_eq!(
        trainer.do_train(&words).unwrap().0,
        [("b".into(), 0)].into()
    );

    // Initial characters have highest priority but still obey the alphabet cap.
    trainer.initial_alphabet = ['a', 'b'].into();
    assert_eq!(trainer.do_train(&counts(&[])).unwrap().0.len(), 1);
}

#[test]
fn initial_pairs_bypass_length_gate() {
    let mut trainer = trainer();
    for limit in [0, 1, 2] {
        trainer.max_token_length = Some(limit);
        let (vocab, merges, _) = trainer.do_train(&counts(&[("ab", 1)])).unwrap();
        assert_eq!(
            vocab,
            [("a".into(), 0), ("b".into(), 1), ("ab".into(), 2)].into()
        );
        assert_eq!(merges, [("a".into(), "b".into())]);
    }
}

#[test]
fn reusable_input() {
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
fn feed_replaces_input() {
    let mut trainer = trainer();
    trainer
        .feed(["old", "old"].into_iter(), |word| Ok(vec![word.into()]))
        .unwrap();
    trainer
        .feed(["ab", "ab", "c"].into_iter(), |word| Ok(vec![word.into()]))
        .unwrap();
    assert_eq!(trainer.get_word_count(), 2);
    let (vocab, merges, _) = trainer.train_vocab().unwrap();
    assert_eq!(
        vocab,
        [
            ("a".into(), 0),
            ("b".into(), 1),
            ("c".into(), 2),
            ("ab".into(), 3)
        ]
        .into()
    );
    assert_eq!(merges, [("a".into(), "b".into())]);

    let previous = trainer.clone();
    let calls = std::sync::atomic::AtomicUsize::new(0);
    assert!(
        trainer
            .feed(["bad", "ab", "bad"].into_iter(), |word| {
                calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if word == "bad" {
                    Err("processing failed".into())
                } else {
                    Ok(vec![word.into()])
                }
            })
            .is_err()
    );
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 3);
    assert_eq!(trainer, previous);

    trainer
        .feed(std::iter::empty::<&str>(), |_| unreachable!())
        .unwrap();
    assert_eq!(trainer.get_word_count(), 0);
    let (vocab, merges, _) = trainer.train_vocab().unwrap();
    assert!(vocab.is_empty());
    assert!(merges.is_empty());
}

#[test]
fn empty_training() {
    for target in [0, 64] {
        let mut trainer = trainer();
        trainer.vocab_size = target;
        let mut model = PipelineBPE::from_config(BpeConfig {
            vocab: [("old".into(), 0)].into(),
            ..Default::default()
        })
        .unwrap();
        assert!(trainer.train(&mut model).unwrap().is_empty());
        let config = model.to_config().unwrap();
        assert!(config.vocab.is_empty());
        assert!(config.merges.is_empty());

        trainer.initial_alphabet = ['测'].into();
        trainer.special_tokens = vec![AddedToken::from("[UNK]", true)];
        assert_eq!(trainer.train(&mut model).unwrap(), trainer.special_tokens);
        let config = model.to_config().unwrap();
        assert_eq!(config.vocab, [("[UNK]".into(), 0), ("测".into(), 1)].into());
        assert!(config.merges.is_empty());
    }
}

#[test]
fn failed_train() {
    let mut model = PipelineBPE::from_config(BpeConfig {
        vocab: [("old".into(), 0)].into(),
        ..Default::default()
    })
    .unwrap();
    let original = model.to_config().unwrap().vocab;
    let mut trainer = trainer();
    trainer.words = WordCounts::from(counts(&[("abab", u64::MAX)]));
    assert!(trainer.train(&mut model).is_err());
    assert_eq!(model.to_config().unwrap().vocab, original);

    trainer.words = WordCounts::from(counts(&[("ab", 1)]));
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
    let view = WordCountsView::from(&words);
    let retained = trainer.select_alphabet(view);
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
            .train_attempt(view, TrainingMode::Fresh, &retained, &None)
            .unwrap(),
        AttemptOutcome::RestartForReuse
    ));
    let AttemptOutcome::Complete(result) = trainer
        .train_attempt(view, TrainingMode::Reuse, &retained, &None)
        .unwrap()
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
        initialize(&trainer, WordCountsView::from(&words), false);
    let MergeOutcome::Committed {
        rule_count: accepted,
    } = trainer
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

// Child processes isolate public settings. Training must reuse the ambient pool
// and stay on the calling worker when repository parallelism is disabled.
use std::sync::atomic::{AtomicUsize, Ordering};

fn run_child(test: &str, child: &str, setting: &str) -> std::process::Output {
    let result = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test, "--nocapture"])
        .env(child, setting)
        .env(
            "TOKENIZERS_PARALLELISM",
            if setting == "env_false" {
                "false"
            } else {
                "true"
            },
        )
        .env("RAYON_NUM_THREADS", "3")
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
fn public_behavior() {
    const CHILD: &str = "BPE_PUBLIC_TEST_CHILD";
    const TEST: &str = "trainers::bpe::tests::public_behavior";
    if let Ok(setting) = std::env::var(CHILD) {
        run_public_training_child(&setting);
        return;
    }
    for setting in [
        "parallel",
        "serial",
        "env_false",
        "one",
        "zero",
        "silent",
        "restart",
    ] {
        check_progress_records(setting, run_child(TEST, CHILD, setting));
    }
}

// The child owns process-wide parallelism settings; the parent checks its output.
fn run_public_training_child(setting: &str) {
    use tk_encode::utils::progress::ProgressFormat;
    let parallel = !matches!(setting, "serial" | "env_false");
    let workers = if setting == "one" { 1 } else { 2 };
    // The encoder's separate pool setting must not replace the training caller's pool.
    tk_encode::parallelism::set_num_threads(4);
    if setting != "env_false" {
        tk_encode::parallelism::set_parallelism(parallel);
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .unwrap()
        .install(|| {
            if matches!(setting, "parallel" | "serial" | "env_false" | "one") {
                check_feed_success(workers, parallel);
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
            let (_, merges, _) = trainer.do_train(&words).unwrap();
            assert_eq!(has_parallelism_been_used(), parallel);
            println!("BPE_MERGE_COUNT={}", merges.len());
            let trainer = BpeTrainer::builder()
                .vocab_size(96)
                .min_frequency(1)
                .show_progress(false)
                .build();
            let words = counts(&[("ab".repeat(8192).as_str(), 3), ("xyxy", 2)]);
            let actual = trainer.do_train(&words).unwrap();
            assert_eq!(actual, trainer.do_train_reference(&words).unwrap().parts);
            assert_eq!(has_parallelism_been_used(), parallel);
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
        // i + 1 must exist before comparing adjacent IDs; spans are separate
        // occurrence lengths and are summed only after the pair matches.
        while i + 1 < word.len() {
            if (word[i].0, word[i + 1].0) == pair {
                let length = word[i].1 + word[i + 1].1;
                if i > 0 {
                    changes.push(((word[i - 1].0, pair.0), -1));
                    // Birth admission uses the combined left-neighbor and
                    // merged occurrence length, excluding equality at the limit.
                    if word[i - 1].1 + length < limit {
                        changes.push(((word[i - 1].0, id), 1));
                    }
                }
                word[i] = (id, length);
                word.remove(i + 1);
                // Removing the old right token shifts the surviving neighbor
                // to i + 1. Apply the same strict combined-length gate on that side.
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
            for entry in self
                .select_alphabet(WordCountsView::from(counts))
                .characters()
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
                        // Each +/-1 occurrence event represents this word's
                        // entire frequency, including intermediate alias events.
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
                // Count original characters, including filtered ones: only
                // the original final character receives the end-of-word suffix.
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

fn check_feed_success(workers: usize, parallel: bool) {
    let caller = rayon::current_thread_index();
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
        // length produces the first None; length + 1 would revive the iterator.
        // Feed must fuse it, so that resurrected item is never observed.
        (value != length && value <= length + 1).then(|| value.to_string())
    });
    let counted = super::feed::count(input, &|text| {
        assert_eq!(rayon::current_num_threads(), workers);
        assert!(parallel || rayon::current_thread_index() == caller);
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
    // Successful items plus the first exhaustion check account for length + 1 reads.
    assert_eq!(reads.load(Ordering::Relaxed), length + 1);
    assert!(seen.iter().all(|n| n.load(Ordering::Relaxed) == 1));
    // Enabled parallelism need not fan out; disabled parallelism must keep
    // every callback on one caller, represented by at most one participant bit.
    assert!(parallel || participants.load(Ordering::Relaxed).count_ones() <= 1);
    let mut expected = AHashMap::<CompactString, u64>::new();
    for id in (0..length).filter(|id| id % 3 != 0) {
        expected.insert(format!("词🙂{id}").into(), 1);
        *expected.entry("repeat".into()).or_default() += 2;
    }
    assert_eq!(counted, WordCounts::from(expected));
    for unique in [2047, 2048, 2049, 4097] {
        let mut words: Vec<_> = (0..unique).map(|id| format!("key{id}")).collect();
        words.extend(["key0".into(), format!("key{}", unique - 1), "key0".into()]);
        let counted = super::feed::count(std::iter::once("input"), &|_| Ok(words.clone())).unwrap();
        let mut expected = AHashMap::<CompactString, u64>::new();
        for word in &words {
            *expected.entry(word.as_str().into()).or_default() += 1;
        }
        assert_eq!(counted, WordCounts::from(expected));
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
