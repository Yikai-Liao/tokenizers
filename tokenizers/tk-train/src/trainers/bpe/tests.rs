//! Differential semantics, explicit numeric contracts, and public integration.
use super::*;
use crate::Trainer;
use ahash::AHashMap;
use compact_str::CompactString;
use std::time::{Duration, Instant};

fn counts(items: &[(&str, u64)]) -> AHashMap<CompactString, u64> {
    items
        .iter()
        .map(|&(word, weight)| (word.into(), weight))
        .collect()
}

fn trainer() -> BpeTrainer {
    BpeTrainer::builder()
        .vocab_size(64)
        .min_frequency(1)
        .show_progress(false)
        .build()
}

fn check(trainer: &BpeTrainer, words: &AHashMap<CompactString, u64>) -> Vec<(Pair, u64, u32)> {
    let mut expected_trace = Vec::new();
    let expected = trainer
        .do_train_observed(words, |p, n, id| expected_trace.push((p, n, id)))
        .unwrap();
    for workers in [1, 4, 8] {
        let mut actual_trace = Vec::new();
        let actual = train(
            trainer,
            WordCountsView::from_map(words),
            workers,
            Some(&mut |p, n, id| actual_trace.push((p, n, id))),
        )
        .unwrap();
        assert_eq!(actual_trace, expected_trace, "workers={workers}");
        assert_eq!(actual, expected, "workers={workers}");
    }
    expected_trace
}

#[test]
fn generated_models_and_every_rule_match_the_sequential_oracle() {
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
fn overlap_floor_alias_and_alphabet_boundaries() {
    check(
        &trainer(),
        &counts(&[("abac", 4), ("dbdc", 4), ("abdb", 3), ("acdc", 3)]),
    );
    // A split producer's local pieces are below the floor; its complete birth is not.
    // Long AA runs separately exercise greedy starts across position restart blocks.
    check(
        &trainer(),
        &counts(&[("ab", 3), ("cd", 3), ("aaaaa", 0), ("", 7)]),
    );
    let repeated = "ab".repeat(12_000);
    let aa = "a".repeat(4097);
    let words = counts(&[(&repeated, 1), ("xy", 12_000), (&aa, 3)]);
    let mut trainer = trainer();
    trainer.min_frequency = 10_000;
    check(&trainer, &words);
    trainer.min_frequency = 1;
    trainer.end_of_word_suffix = Some("a".into());
    let trace = check(&trainer, &counts(&[("baaba", 1)]));
    assert_eq!(&trace[..2], &[((0, 0), 1, 2), ((1, 2), 2, 3)]);
    let aliases = (0..128)
        .map(|i| {
            (
                format!("{i}{}baaba", "aaaaaaaaabcd".repeat(16)).into(),
                [0, 1, 7, 13][i % 4],
            )
        })
        .collect();
    // Reserved merges must still activate after alias geometry becomes per-occurrence.
    trainer.special_tokens = vec![AddedToken::from("aaaaaaaaabcd", true)];
    for (prefix, suffix, limit) in [
        (None, Some("a"), None),
        (Some("ab"), None, Some(9)),
        (Some("##"), Some("</w>"), Some(3)),
    ] {
        trainer.continuing_subword_prefix = prefix.map(str::to_owned);
        trainer.end_of_word_suffix = suffix.map(str::to_owned);
        trainer.max_token_length = limit;
        check(&trainer, &aliases);
    }
    trainer.continuing_subword_prefix = None;
    trainer.special_tokens.clear();
    trainer.max_token_length = None;
    trainer.end_of_word_suffix = None;
    trainer.limit_alphabet = Some(3);
    trainer.initial_alphabet = ['a', '测'].into();
    check(
        &trainer,
        &counts(&[("caba", 4), ("测试测试", 3), ("abab", 7)]),
    );
    trainer.vocab_size = 3;
    let (vocab, merges, _) = trainer
        .do_train(&counts(&[("aaaa", 11), ("abab", 7), ("测试", 3)]))
        .unwrap();
    assert_eq!(
        vocab,
        [("a".into(), 0), ("b".into(), 1), ("测".into(), 2)].into()
    );
    assert!(merges.is_empty());
}

#[test]
fn real_wide_ids_preserve_pair_order_and_separator_distinction() {
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
fn count_domains_and_zero_merge_validation() {
    let mut trainer = trainer();
    for weight in [u32::MAX as u64 + 17, u64::MAX] {
        let mut trace = Vec::new();
        train(
            &trainer,
            WordCountsView::from_map(&counts(&[("ab", weight)])),
            2,
            Some(&mut |p, n, id| trace.push((p, n, id))),
        )
        .unwrap();
        assert_eq!(trace, [((0, 1), weight, 2)]);
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
fn ready_batch_applies_independent_rules_together() {
    let trainer = trainer();
    let words = counts(&[("ab", 10), ("cd", 9), ("ef", 8)]);
    let view = WordCountsView::from_map(&words);
    let progress = trainer.setup_progress();
    let mut vocabulary = Vocabulary::initialize(&trainer, view, 4, &mut None).unwrap();
    let plan = CorpusPlan::build(view, &mut vocabulary, &trainer, false, &progress).unwrap();
    let arena = Arena::new(4, plan.items());
    let mut index = PairIndex::build(&arena, &plan, 1, 4, false, &progress).unwrap();
    let mut corpus = plan.materialize();
    let Selection::Ready(batch) =
        Batch::select(&trainer, &mut vocabulary, &mut corpus, &mut index).unwrap()
    else {
        panic!("expected ready batch")
    };
    assert_eq!(batch.pairs().collect::<Vec<_>>(), [(0, 1), (2, 3), (4, 5)]);
    let prepared = batch.prepare(&corpus, &arena, usize::MAX).unwrap();
    index.commit(prepared.apply(&corpus)).unwrap();
    assert!(matches!(
        Batch::select(&trainer, &mut vocabulary, &mut corpus, &mut index).unwrap(),
        Selection::Finished
    ));
}

#[test]
fn feed_training_tokenizer_json_and_encoding_roundtrip() {
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

fn check_feed_contracts(trainer: &mut BpeTrainer, parallel: bool) {
    for boundary in [0, 31, 32, 33, 127, 128, 129, 257] {
        // One callback alone crosses the local key limit; the public
        // reference is a flat multiset, independent of batching and flushes.
        let outputs = [
            Vec::new(),
            vec!["shared".into(), "shared".into()],
            vec![
                "".into(),
                "中文🙂".into(),
                "e\u{301}".into(),
                "long".repeat(2048),
            ],
            (0..2047 + boundary % 3)
                .map(|i| format!("word{i}"))
                .collect(),
        ];
        let mut index = 0;
        let input = std::iter::from_fn(move || {
            let current = index;
            index += 1;
            (current != boundary && current <= boundary + 1).then(|| current.to_string())
        });
        let participants = AtomicUsize::new(0);
        let calls = AtomicUsize::new(0);
        trainer
            .feed(input, |s| {
                assert_eq!(rayon::current_num_threads(), 2);
                calls.fetch_add(1, Ordering::Relaxed);
                let index = s.parse::<usize>().unwrap();
                participants.fetch_or(
                    1 << rayon::current_thread_index().unwrap(),
                    Ordering::Relaxed,
                );
                if parallel && index < 2 {
                    let deadline = Instant::now() + Duration::from_secs(5);
                    while participants.load(Ordering::Relaxed).count_ones() < 2 {
                        assert!(
                            Instant::now() < deadline,
                            "feed callbacks did not run independently"
                        );
                        std::thread::yield_now();
                    }
                }
                Ok(outputs[index % outputs.len()].clone())
            })
            .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), boundary);
        assert!(parallel || participants.load(Ordering::Relaxed).count_ones() <= 1);
        let mut expected = AHashMap::<CompactString, u64>::new();
        for index in 0..boundary {
            for word in &outputs[index % outputs.len()] {
                *expected.entry(word.as_str().into()).or_default() += 1;
            }
        }
        let mut equivalent = trainer.clone();
        equivalent.words = super::word_counts::WordCounts::from_map(expected.clone());
        assert_eq!(*trainer, equivalent);
        let stored = serde_json::to_value(&trainer).unwrap();
        assert_eq!(stored["words"], serde_json::to_value(expected).unwrap());
        assert_eq!(
            *trainer,
            serde_json::from_value::<BpeTrainer>(stored).unwrap()
        );
    }
    let calls = AtomicUsize::new(0);
    let previous = trainer.clone();
    let error = trainer
        .feed((0..257).map(|i| i.to_string()), |s| {
            calls.fetch_add(1, Ordering::Relaxed);
            if s == "7" {
                Err("process failed".into())
            } else {
                Ok(vec![s.into()])
            }
        })
        .unwrap_err();
    assert_eq!(error.to_string(), "process failed");
    assert_eq!(calls.load(Ordering::Relaxed), 257);
    assert_eq!(*trainer, previous);
}

#[test]
fn public_pools_feed_flush_errors_and_progress_matrix() {
    use tk_encode::utils::progress::ProgressFormat;
    const CHILD: &str = "BPE_PUBLIC_TEST_CHILD";
    const TEST: &str = "trainers::bpe::tests::public_pools_feed_flush_errors_and_progress_matrix";
    if let Ok(setting) = std::env::var(CHILD) {
        let parallel = setting != "serial";
        tk_encode::parallelism::set_num_threads(4);
        tk_encode::parallelism::set_parallelism(parallel);
        EXPECTED_WORKERS.store(if parallel { 4 } else { 1 }, Ordering::Relaxed);
        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap()
            .install(|| {
                let mut trainer = trainer();
                if matches!(setting.as_str(), "parallel" | "serial") {
                    check_feed_contracts(&mut trainer, parallel);
                }
                trainer.end_of_word_suffix = Some("a".into());
                trainer.do_train(&counts(&[("baaba", 1)])).unwrap();
                // Materialization and historical-cohort preparation both ran.
                assert_eq!(OBSERVED_PHASES.load(Ordering::Relaxed), 0b101);
                trainer.end_of_word_suffix = None;
                trainer.show_progress = setting != "no-bar";
                trainer.progress_format = if setting == "silent" {
                    ProgressFormat::Silent
                } else {
                    ProgressFormat::JsonLines
                };
                trainer.vocab_size = if setting == "zero" { 2 } else { 64 };
                let words = if setting == "empty" {
                    AHashMap::new()
                } else {
                    counts(&[("ab", 3), ("cd", 2)])
                };
                let (_, merges, _) = trainer.do_train(&words).unwrap();
                if !merges.is_empty() {
                    assert_eq!(OBSERVED_PHASES.load(Ordering::Relaxed), 0b111);
                }
                println!("BPE_MERGE_COUNT={}", merges.len());
            });
        return;
    }
    for setting in ["parallel", "serial", "no-bar", "zero", "empty", "silent"] {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", TEST, "--nocapture"])
            .env(CHILD, setting)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let output = String::from_utf8(result.stderr).unwrap();
        if setting == "silent" {
            assert!(output.is_empty());
            continue;
        }
        let records: Vec<serde_json::Value> = output
            .lines()
            .map(|s| serde_json::from_str(s).unwrap())
            .collect();
        let mut stages = std::collections::HashSet::new();
        for record in &records {
            assert_eq!(record.as_object().unwrap().len(), 3);
            assert!(record["current"].is_u64() && record["total"].is_u64());
            if stages.insert(record["stage"].as_str().unwrap()) {
                assert_eq!(record["current"], 0);
            }
        }
        assert_eq!(
            stages,
            ["Tokenize words", "Count pairs", "Compute merges"].into()
        );
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
    }
}
