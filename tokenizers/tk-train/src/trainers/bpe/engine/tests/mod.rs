//! Differential semantics, explicit numeric contracts, and public integration.
use super::*;
use ahash::AHashMap;
use compact_str::CompactString;

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
    check(
        &trainer,
        &counts(&[("aaaaa", 1), ("aaaaaaa", 1), ("baaba", 0)]),
    );
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
        assert!(trainer.do_train(&counts(&[("aba", u64::MAX)])).is_ok());
        assert!(trainer.do_train(&counts(&[("abab", u64::MAX)])).is_err());
        trainer.end_of_word_suffix = Some("a".into());
        assert!(
            trainer
                .do_train(&counts(&[("ab", i64::MAX as u64)]))
                .is_ok()
        );
        assert!(
            trainer
                .do_train(&counts(&[("ab", i64::MAX as u64 + 1)]))
                .is_err()
        );
        assert!(
            trainer
                .do_train(&counts(&[("abc", i64::MAX as u64)]))
                .is_err()
        );
        trainer.end_of_word_suffix = None;
    }
}

#[test]
fn ready_batch_applies_independent_rules_together() {
    let trainer = trainer();
    let words = counts(&[("ab", 10), ("cd", 9), ("ef", 8)]);
    let view = WordCountsView::from_map(&words);
    let progress = TrainingProgress::new(false, trainer.progress_format).unwrap();
    let mut vocabulary = Vocabulary::initialize(&trainer, view, 4, &progress, &mut None).unwrap();
    let plan = CorpusPlan::build(view, &mut vocabulary, &trainer, false, &progress).unwrap();
    let arena = Arena::new(4, plan.items());
    let mut index = PairIndex::build(&arena, &plan, 1, 4, false, &progress).unwrap();
    let mut corpus = plan.materialize(&progress);
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
fn feed_model_configuration_reload_and_errors() {
    use crate::Trainer;
    use tk_encode::models::bpe::{BpeConfig, PipelineBPE};
    let mut trainer = trainer();
    trainer.continuing_subword_prefix = Some("##".into());
    trainer.end_of_word_suffix = Some("</w>".into());
    trainer
        .feed(["ab测 ab测", "ab测"].into_iter(), |s| {
            Ok(s.split_whitespace().map(str::to_owned).collect())
        })
        .unwrap();
    let restored: BpeTrainer =
        serde_json::from_value(serde_json::to_value(&trainer).unwrap()).unwrap();
    assert_eq!(trainer, restored);
    assert_eq!(
        trainer.train_vocab().unwrap(),
        restored.train_vocab().unwrap()
    );
    let mut model = PipelineBPE::from_config(BpeConfig {
        vocab: [("old".into(), 0)].into(),
        ..Default::default()
    })
    .unwrap();
    trainer.train(&mut model).unwrap();
    let config = model.to_config().unwrap();
    assert_eq!(config.continuing_subword_prefix.as_deref(), Some("##"));
    assert_eq!(config.end_of_word_suffix.as_deref(), Some("</w>"));
    let expected = (config.vocab.clone(), config.merges.clone());
    let reloaded = PipelineBPE::from_config(config)
        .unwrap()
        .to_config()
        .unwrap();
    assert_eq!((reloaded.vocab, reloaded.merges), expected);
    let previous = trainer.clone();
    assert!(
        trainer
            .feed(["bad"].into_iter(), |_| Err("process failed".into()))
            .is_err()
    );
    assert_eq!(trainer, previous);
}

// The child process isolates public parallelism settings from concurrent tests.
use std::sync::atomic::{AtomicUsize, Ordering};
static EXPECTED_WORKERS: AtomicUsize = AtomicUsize::new(0);
pub(super) fn observe_worker() {
    let expected = EXPECTED_WORKERS.load(Ordering::Relaxed);
    if expected != 0 {
        assert_eq!(rayon::current_num_threads(), expected);
    }
}

#[test]
fn pool_progress_and_feed_boundaries() {
    use crate::Trainer;
    const CHILD: &str = "BPE_POOL_TEST_CHILD";
    const TEST: &str = "trainers::bpe::engine::tests::pool_progress_and_feed_boundaries";
    if let Ok(setting) = std::env::var(CHILD) {
        let parallel = setting == "parallel";
        tk_encode::parallelism::set_num_threads(4);
        tk_encode::parallelism::set_parallelism(parallel);
        EXPECTED_WORKERS.store(if parallel { 4 } else { 1 }, Ordering::Relaxed);
        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap()
            .install(|| {
                let mut trainer = trainer();
                for boundary in [0, 31, 32, 33, 127, 128, 129, 257] {
                    let mut index = 0;
                    let input = std::iter::from_fn(move || {
                        let current = index;
                        index += 1;
                        (current != boundary && current <= boundary + 1)
                            .then(|| format!("word{current}"))
                    });
                    trainer
                        .feed(input, |s| {
                            Ok(vec![s.into(), "shared".into(), "shared".into()])
                        })
                        .unwrap();
                    let stored = serde_json::to_value(&trainer).unwrap();
                    assert_eq!(
                        trainer.get_word_count(),
                        boundary + usize::from(boundary != 0)
                    );
                    if boundary != 0 {
                        assert_eq!(stored["words"]["shared"], 2 * boundary);
                    }
                    assert!(stored["words"][format!("word{}", boundary + 1)].is_null());
                }
                let calls = AtomicUsize::new(0);
                let previous = trainer.clone();
                let failure = trainer.feed((0..257).map(|i| i.to_string()), |i| {
                    calls.fetch_add(1, Ordering::Relaxed);
                    if i == "7" {
                        Err("process failed".into())
                    } else {
                        Ok(vec![i.to_string()])
                    }
                });
                assert!(failure.is_err());
                assert_eq!(calls.load(Ordering::Relaxed), 257);
                assert_eq!(trainer, previous);
                trainer.show_progress = true;
                trainer.progress_format = tk_encode::utils::progress::ProgressFormat::JsonLines;
                let (_, merges, _) = trainer.do_train(&counts(&[("ab", 3), ("cd", 2)])).unwrap();
                assert_eq!(merges.len(), 2);
            });
        return;
    }
    for setting in ["parallel", "serial"] {
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
        let records: Vec<serde_json::Value> = output
            .lines()
            .filter_map(|s| serde_json::from_str(s).ok())
            .collect();
        let finished = records
            .iter()
            .rev()
            .find(|r| r["stage"] == "Compute merges")
            .unwrap();
        assert_eq!(finished["current"], 2);
        assert_eq!(finished["total"], 2);
    }
}
