use super::*;
use tk_encode::models::bpe::Pair;
fn counts(items: &[(&str, u64)]) -> AHashMap<CompactString, u64> {
    items
        .iter()
        .map(|&(word, count)| (word.into(), count))
        .collect()
}
fn check(trainer: &BpeTrainer, words: &AHashMap<CompactString, u64>) {
    check_with_workers(trainer, words, &[1, 2, 4]);
}
fn check_with_workers(
    trainer: &BpeTrainer,
    words: &AHashMap<CompactString, u64>,
    workers: &[usize],
) {
    let mut expected_trace = Vec::new();
    let expected = trainer
        .do_train_observed(words, |pair, count, id| {
            expected_trace.push((pair, count, id))
        })
        .unwrap();
    for &workers in workers {
        let mut trace = Vec::<(Pair, u64, u32)>::new();
        let actual = train(
            trainer,
            words,
            workers,
            Some(&mut |pair, count, id| trace.push((pair, count, id))),
        )
        .unwrap();
        assert_eq!(
            trace, expected_trace,
            "workers={workers}, prefix={:?}, suffix={:?}, limit={:?}",
            trainer.continuing_subword_prefix, trainer.end_of_word_suffix, trainer.max_token_length
        );
        assert_eq!(actual, expected, "workers={workers}");
    }
}
#[test]
fn routing_keeps_stable_births_and_original_count_actions() {
    use merge::{ChangeAction, EventChunk, MergeEvents, PairChanges};
    use pair_index::{pair_key, shard_for};
    use tk_collections::{PositionChain, PositionChains};

    let chunks = (0..7_u32)
        .map(|producer| {
            let mut chains = PositionChains::new();
            let changes = (0..97_u32)
                .map(|index| {
                    let removed_key = pair_key((u32::MAX - producer, index));
                    let born_key = if index % 5 == 0 {
                        removed_key
                    } else {
                        pair_key((index, u32::MAX - producer))
                    };
                    let mut positions = PositionChain::default();
                    if index % 3 != 0 {
                        chains
                            .push(
                                &mut positions,
                                (u64::from(producer) << 32) + u64::from(index),
                            )
                            .unwrap();
                    }
                    PairChanges {
                        removed_key,
                        born_key,
                        removed_weight: u64::from(index % 4 != 0),
                        born_weight: 0,
                        positions,
                        bucket: (producer * 41 + index * 79) % 512,
                    }
                })
                .collect();
            EventChunk { chains, changes }
        })
        .collect();
    let events = MergeEvents {
        buckets: 512,
        chunks,
    };
    for workers in [1, 4, 16, 64] {
        let routes = events.route(workers);
        for (owner, route) in routes.iter().enumerate() {
            let mut expected_actions = Vec::new();
            for (producer, chunk) in events.chunks.iter().enumerate() {
                for (index, change) in chunk.changes.iter().enumerate() {
                    let removal = change.removed_weight != 0
                        && shard_for(change.removed_key, workers) == owner;
                    let birth = !change.positions.is_empty()
                        && shard_for(change.born_key, workers) == owner;
                    if removal || birth {
                        expected_actions.push((producer, index, removal, birth));
                    }
                }
            }
            let actual_actions: Vec<_> = route
                .changes
                .iter()
                .map(|reference| {
                    (
                        reference.chunk,
                        reference.index(),
                        matches!(
                            reference.action(),
                            ChangeAction::Remove | ChangeAction::Both
                        ),
                        matches!(reference.action(), ChangeAction::Birth | ChangeAction::Both),
                    )
                })
                .collect();
            assert_eq!(
                actual_actions, expected_actions,
                "workers={workers}, owner={owner}"
            );

            // Enumerate buckets explicitly as the oracle. Inside each bucket,
            // the original producer/record order is the required stable order.
            let mut expected_births = Vec::new();
            for bucket in 0..512 {
                for (producer, chunk) in events.chunks.iter().enumerate() {
                    for (index, change) in chunk.changes.iter().enumerate() {
                        if change.bucket == bucket
                            && !change.positions.is_empty()
                            && shard_for(change.born_key, workers) == owner
                        {
                            expected_births.push((producer, index));
                        }
                    }
                }
            }
            let actual_births: Vec<_> = route
                .births
                .iter()
                .map(|&index| {
                    let reference = &route.changes[index];
                    (reference.chunk, reference.index())
                })
                .collect();
            assert_eq!(
                actual_births, expected_births,
                "workers={workers}, owner={owner}"
            );
        }
    }
    assert!(
        MergeEvents {
            buckets: 0,
            chunks: Vec::new()
        }
        .route(4)
        .iter()
        .all(|route| route.changes.is_empty() && route.births.is_empty())
    );
}
#[test]
fn routed_batches_preserve_rule_and_position_order_with_many_workers() {
    let mut words = counts(&[("aaaaaaa", 5), ("abcabc", 3), ("baab", 0), ("", 1)]);
    for index in 0..128_u32 {
        let word: String = (0..3)
            .map(|offset| char::from_u32(0x4000 + index * 3 + offset).unwrap())
            .collect();
        words.insert(word.into(), 2 + u64::from(index % 3));
    }
    let trainer = BpeTrainer::builder()
        .vocab_size(700)
        .min_frequency(2)
        .show_progress(false)
        .build();
    check_with_workers(&trainer, &words, &[1, 4, 16, 64]);

    let mut aliases = trainer;
    aliases.continuing_subword_prefix = Some("a".into());
    aliases.end_of_word_suffix = Some("a".into());
    aliases.max_token_length = Some(5);
    check_with_workers(
        &aliases,
        &counts(&[("aaaaaaa", 5), ("abcabc", 3), ("baab", 0)]),
        &[1, 16, 64],
    );
}
#[test]
fn weighted_ties_unicode_aa_and_reserved_id_activations() {
    let words = counts(&[
        ("", 1),
        ("a", 10),
        ("aaaaaaa", 3),
        ("abababab", 4),
        ("abcabc", 4),
        ("测试测试", 7),
        ("ééé", 2),
        ("baab", 0),
    ]);
    for special in [
        vec![],
        ["ab", "aba", "ab", "aa", "aaaa"]
            .map(|s| AddedToken::from(s, true))
            .to_vec(),
    ] {
        let trainer = BpeTrainer::builder()
            .vocab_size(100)
            .special_tokens(special)
            .show_progress(false)
            .build();
        check(&trainer, &words);
        check(&trainer, &counts(&[(&"a".repeat(1024), 3)]));
    }
}
#[test]
fn affixes_aliases_and_strict_length_boundaries() {
    let words = counts(&[
        ("aaaaaaa", 11),
        ("abab", 7),
        ("baaba", 4),
        ("测试测试", 3),
        ("ccc", 1),
    ]);
    // Named cases retain the distinct birth gates, identity collisions, and
    // empty-affix behavior without repeating every unrelated combination.
    for (_name, prefix, suffix, limit) in [
        ("plain", None, None, None),
        ("zero birth gate", None, None, Some(0)),
        ("unit birth gate", None, None, Some(1)),
        ("exact pair gate", None, None, Some(2)),
        ("short birth gate", None, None, Some(3)),
        ("middle birth gate", None, None, Some(5)),
        ("long birth gate", None, None, Some(16)),
        ("prefix", Some("##"), None, None),
        ("suffix", None, Some("</w>"), None),
        ("both affixes", Some("##"), Some("</w>"), None),
        ("prefix identity reuse", Some("a"), None, Some(3)),
        ("suffix unequal spans", None, Some("a"), Some(5)),
        ("both identity reuse", Some("a"), Some("a"), Some(3)),
        ("decorated pair gate", Some("##"), Some("</w>"), Some(2)),
        ("decorated long gate", Some("##"), Some("</w>"), Some(16)),
        ("empty prefix", Some(""), Some("</w>"), Some(3)),
        ("empty suffix", Some("##"), Some(""), Some(3)),
        ("both empty", Some(""), Some(""), None),
    ] {
        let mut trainer = BpeTrainer::builder()
            .vocab_size(70)
            .show_progress(false)
            .special_tokens(vec![AddedToken::from("aa", true)])
            .build();
        trainer.continuing_subword_prefix = prefix.map(str::to_owned);
        trainer.end_of_word_suffix = suffix.map(str::to_owned);
        trainer.max_token_length = limit;
        check(&trainer, &words);
    }
}
#[test]
fn pruning_waits_for_all_birth_producers_and_alphabet_filtering() {
    let trainer = BpeTrainer::builder()
        .vocab_size(30)
        .min_frequency(2)
        .show_progress(false)
        .build();
    check(&trainer, &counts(&[("xabp", 1), ("xabq", 1)]));
    let trainer = BpeTrainer::builder()
        .vocab_size(30)
        .limit_alphabet(3)
        .initial_alphabet(['a', '测'].into())
        .show_progress(false)
        .build();
    let words = counts(&[
        ("aaaaaaa", 11),
        ("abab", 7),
        ("caba", 4),
        ("测试测试", 3),
        ("ccc", 1),
    ]);
    check(&trainer, &words);
    // The mainline oracle shares alphabet construction. A literal expectation
    // independently checks the frequency limit and forced characters.
    let mut alphabet_only = trainer;
    alphabet_only.vocab_size = 3;
    let (vocab, merges, _) = train(&alphabet_only, &words, 1, None).unwrap();
    assert_eq!(
        vocab,
        [("a".into(), 0), ("b".into(), 1), ("测".into(), 2)]
            .into_iter()
            .collect::<AHashMap<_, _>>()
    );
    assert!(merges.is_empty());
}
#[test]
fn cohort_cohort_words_include_stale_addresses_and_zero_weights() {
    // The suffix creates an active "aa" ID before AA -> aa. Across words,
    // left and right birth chains for that identity interleave spatially.
    let trainer = BpeTrainer::builder()
        .vocab_size(30)
        .min_frequency(2)
        .end_of_word_suffix("a".into())
        .show_progress(false)
        .build();
    check(&trainer, &counts(&[("aaaaa", 1), ("aaaaaaa", 1)]));
    let words = (0..128)
        .map(|i| {
            (
                format!("{i}{}baaba", "aaaaaaaaabcd".repeat(16)).into(),
                [0, 1, 7, 13][i % 4],
            )
        })
        .collect();
    for (prefix, suffix, limit) in [
        (None, Some("a"), None),
        (Some("ab"), None, Some(9)),
        (Some("##"), Some("</w>"), Some(3)),
    ] {
        let mut trainer = BpeTrainer::builder()
            .vocab_size(80)
            .min_frequency(2)
            .show_progress(false)
            .max_token_length(limit)
            .build();
        trainer.continuing_subword_prefix = prefix.map(str::to_owned);
        trainer.end_of_word_suffix = suffix.map(str::to_owned);
        check(&trainer, &words);
    }
}
#[test]
fn full_pair_keys_remain_distinct_in_initial_counting() {
    use std::sync::atomic::AtomicU32;
    use tk_collections::IntervalIndex;
    let ids = [
        WORD_SEPARATOR_ID,
        1,
        2,
        WORD_SEPARATOR_ID,
        0x1_0001,
        2,
        WORD_SEPARATOR_ID,
        u32::MAX - 1,
        u32::MAX - 2,
        WORD_SEPARATOR_ID,
    ];
    let slots: Vec<_> = ids.into_iter().map(AtomicU32::new).collect();
    let weights = IntervalIndex::new(vec![1], vec![7]);
    for workers in [1, 2, 4] {
        let execution = execution::Execution::new(workers).unwrap();
        let arena = AllocationArena::new(workers, 3);
        let progress =
            TrainingProgress::new(false, tk_encode::utils::progress::ProgressFormat::Silent)
                .unwrap();
        execution.pool.install(|| {
            let initial = initial_pairs::build_initial_pairs(
                corpus::InitialCorpus {
                    token_ids: &slots,
                    word_weights: &weights,
                },
                1,
                &execution,
                &arena,
                &progress,
            )
            .unwrap();
            assert_eq!(
                initial
                    .shards
                    .iter()
                    .map(|shard| shard.len())
                    .sum::<usize>(),
                3
            );
            for (pair, position) in [
                ((1, 2), 1),
                ((0x1_0001, 2), 4),
                ((u32::MAX - 1, u32::MAX - 2), 7),
            ] {
                let key = pair_index::pair_key(pair);
                let state = &initial.shards[pair_index::shard_for(key, workers)][&key];
                assert_eq!(state.ledger_count_bits, 7);
                assert_eq!(state.positions.iter().collect::<Vec<_>>(), [position]);
            }
        });
    }
}

#[test]
fn initial_waves_preserve_crossing_edges_and_filter_complete_counts() {
    use std::sync::atomic::AtomicU32;
    use tk_collections::IntervalIndex;
    let mut slots: Vec<_> = [
        WORD_SEPARATOR_ID,
        1,
        2,
        1,
        2,
        1,
        2,
        WORD_SEPARATOR_ID,
        1,
        2,
        1,
        2,
        WORD_SEPARATOR_ID,
        u32::MAX - 1,
        u32::MAX - 2,
        WORD_SEPARATOR_ID,
    ]
    .into_iter()
    .map(AtomicU32::new)
    .collect();
    slots.extend([1, 2, WORD_SEPARATOR_ID].into_iter().map(AtomicU32::new));
    let weights = IntervalIndex::new(vec![1, 8, 13, 16], vec![3, 5, 0, 0]);
    for workers in [1, 4] {
        let execution = execution::Execution::new(workers).unwrap();
        let arena = AllocationArena::new(workers, 10);
        let progress =
            TrainingProgress::new(false, tk_encode::utils::progress::ProgressFormat::Silent)
                .unwrap();
        execution.pool.install(|| {
            // Use the real wave algorithm with a small resource bound. The
            // (1,2) edge at slot 3 crosses a wave; every partial count is below
            // 12, while its complete weighted count is 19.
            for floor in [0, 12] {
                let initial = initial_pairs::build_in_waves(
                    corpus::InitialCorpus {
                        token_ids: &slots,
                        word_weights: &weights,
                    },
                    floor,
                    &execution,
                    &arena,
                    &progress,
                    4,
                )
                .unwrap();
                assert_eq!(initial.weighted_mass, 30);
                let key = pair_index::pair_key((1, 2));
                let state = &initial.shards[pair_index::shard_for(key, workers)][&key];
                assert_eq!(state.ledger_count_bits, 19);
                assert_eq!(
                    state.positions.iter().collect::<Vec<_>>(),
                    if floor == 0 {
                        vec![1, 3, 5, 8, 10, 16]
                    } else {
                        vec![1, 3, 5, 8, 10]
                    }
                );
                assert_eq!(
                    initial
                        .shards
                        .iter()
                        .map(|shard| shard.len())
                        .sum::<usize>(),
                    if floor == 0 { 3 } else { 1 }
                );
                if floor == 0 {
                    let key = pair_index::pair_key((u32::MAX - 1, u32::MAX - 2));
                    let state = &initial.shards[pair_index::shard_for(key, workers)][&key];
                    assert_eq!(state.ledger_count_bits, 0);
                    assert_eq!(state.positions.iter().collect::<Vec<_>>(), [13]);
                }
            }
        });
    }
}

fn next(rng: &mut u64) -> u64 {
    *rng = rng
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *rng >> 32
}

#[test]
fn randomized_round_by_round_hf_differential() {
    hf_differential_cases(64);
}

#[test]
#[ignore = "extended differential check; run explicitly before release"]
fn randomized_round_by_round_hf_stress() {
    hf_differential_cases(1500);
}

fn hf_differential_cases(cases: usize) {
    let mut rng = 2348;
    for case in 0..cases {
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
            trainer.continuing_subword_prefix = Some(["##", "a", "", "测"][(case / 4) % 4].into());
        }
        if case % 5 == 0 {
            trainer.end_of_word_suffix = Some(["</w>", "a", ""][case % 3].into());
        }
        if case % 2 == 0 {
            trainer.max_token_length = Some((next(&mut rng) % 10) as usize);
        }
        check_with_workers(&trainer, &wc, &[[1, 2, 4][case % 3]]);
    }
}

fn greedy(trainer: &BpeTrainer, wc: &AHashMap<CompactString, u64>) -> Vec<(Pair, u64, u32)> {
    let mut ids = AHashMap::new();
    let mut strings = Vec::new();
    trainer.add_special_tokens(&mut ids, &mut strings);
    trainer.compute_alphabet(wc, &mut ids, &mut strings);
    let (words, weights) = trainer.tokenize_words(wc, &mut ids, &mut strings, &None);
    let mut words: Vec<Vec<u32>> = words
        .iter()
        .map(super::super::word::Word::get_chars)
        .collect();
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
        let mut trace = Vec::new();
        train(
            &trainer,
            &wc,
            4,
            Some(&mut |pair, count, id| trace.push((pair, count, id))),
        )
        .unwrap();
        assert_eq!(trace, greedy(&trainer, &wc), "words={wc:?}");
    }
}

#[test]
fn public_feed_train_and_model_reload_preserve_affixes() {
    use crate::Trainer;
    use tk_encode::{
        models::bpe::{BpeConfig, PipelineBPE},
        pipeline::{
            EncodeOptions, PipelineModel, PipelinePostProcessor, PipelinePreTokenizer,
            PipelineTokenizer,
        },
        vocab::bucket_added_vocabulary::AddedVocabulary,
    };
    let special = vec![AddedToken::from("[UNK]", true)];
    let mut trainer = BpeTrainer::builder()
        .vocab_size(20)
        .min_frequency(1)
        .continuing_subword_prefix("##".into())
        .end_of_word_suffix("</w>".into())
        .special_tokens(special.clone())
        .show_progress(false)
        .build();
    trainer
        .feed(["ab测 测", "ab测"].into_iter(), |text| {
            Ok(text.split_whitespace().map(str::to_owned).collect())
        })
        .unwrap();
    assert_eq!(trainer.get_word_count(), 2);
    let mut model = PipelineBPE::from_config(BpeConfig {
        vocab: [("[old]".into(), 0)].into_iter().collect(),
        ..Default::default()
    })
    .unwrap();
    assert_eq!(trainer.train(&mut model).unwrap(), special);
    let config = model.to_config().unwrap();
    assert_eq!(config.continuing_subword_prefix.as_deref(), Some("##"));
    assert_eq!(config.end_of_word_suffix.as_deref(), Some("</w>"));
    let expected = [
        ("测", vec![config.vocab["测</w>"]]),
        ("a测", vec![config.vocab["a"], config.vocab["##测</w>"]]),
        ("ab测", vec![config.vocab["ab测</w>"]]),
    ];
    let tokenizer = PipelineTokenizer::from_parts(
        AddedVocabulary::new(),
        vec![],
        PipelinePreTokenizer::None,
        PipelineModel::BPE(model),
        PipelinePostProcessor::default(),
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
    for model in [&tokenizer, &reloaded] {
        for (input, expected) in &expected {
            let encoded = model
                .encode(*input, &EncodeOptions::no_specials())
                .wait()
                .unwrap();
            assert_eq!(
                encoded[0]
                    .ids()
                    .iter()
                    .map(|token| token.id())
                    .collect::<Vec<_>>(),
                *expected,
                "{input}"
            );
        }
    }
}

#[test]
fn wide_frequencies_and_signed_ledger_boundaries() {
    let trainer = BpeTrainer::builder()
        .vocab_size(8)
        .min_frequency(1)
        .show_progress(false)
        .build();
    for weight in [u64::from(u32::MAX) + 17, u64::MAX] {
        let mut trace = Vec::new();
        train(
            &trainer,
            &counts(&[("ab", weight)]),
            2,
            Some(&mut |pair, count, id| trace.push((pair, count, id))),
        )
        .unwrap();
        assert_eq!(trace, [((0, 1), weight, 2)]);
        let (vocab, merges, _) = trainer.do_train(&counts(&[("ab", weight)])).unwrap();
        assert_eq!(vocab["ab"], 2);
        assert_eq!(merges, [("a".into(), "b".into())]);
    }
    assert!(trainer.do_train(&counts(&[("aba", u64::MAX)])).is_ok()); // separate keys, not a global u64 mass cap
    assert!(trainer.do_train(&counts(&[("abab", u64::MAX)])).is_err()); // one key's frequency overflows
    let trainer = BpeTrainer::builder()
        .vocab_size(8)
        .min_frequency(1)
        .show_progress(false)
        .end_of_word_suffix("a".into())
        .build();
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
    ); // edge mass exceeds the signed policy domain
}

#[test]
fn public_thread_policy_in_isolated_processes() {
    const CHILD: &str = "BPE_THREAD_POLICY_TEST_CHILD";
    if let Ok(setting) = std::env::var(CHILD) {
        let (parallel, workers) = setting.split_once(':').unwrap();
        tk_encode::parallelism::set_num_threads(workers.parse().unwrap());
        tk_encode::parallelism::set_parallelism(parallel == "true");
        public_feed_train_and_model_reload_preserve_affixes();
        let trainer = BpeTrainer::builder()
            .vocab_size(10)
            .min_frequency(1)
            .show_progress(false)
            .build();
        let words = counts(&[("aaaaa", 3), ("abcabc", 2), ("测测测", 1)]);
        assert_eq!(
            trainer.do_train(&words).unwrap(),
            trainer.do_train_observed(&words, |_, _, _| {}).unwrap()
        );
        return;
    }
    for setting in ["false:4", "true:1", "true:2", "true:4"] {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "trainers::bpe::engine::tests::public_thread_policy_in_isolated_processes",
                "--nocapture",
            ])
            .env(CHILD, setting)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{setting}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
}

#[test]
fn suffix_identity_reuse_has_literal_mainline_merge_choices() {
    let trainer = BpeTrainer::builder()
        .vocab_size(8)
        .min_frequency(1)
        .show_progress(false)
        .end_of_word_suffix("a".into())
        .build();
    let words = counts(&[("baaba", 1)]);
    let mut trace = Vec::new();
    let (_, merges, _) = train(
        &trainer,
        &words,
        2,
        Some(&mut |pair, count, id| trace.push((pair, count, id))),
    )
    .unwrap();
    assert_eq!(&trace[..2], &[((0, 0), 1, 2), ((1, 2), 2, 3)]);
    assert_eq!(
        &merges[..2],
        &[("a".into(), "a".into()), ("b".into(), "aa".into())]
    );
    assert_eq!(trainer.do_train(&words).unwrap().1, merges);
}

#[test]
fn active_id_reuse_rebuilds_without_publishing_speculative_rules() {
    let mut trainer = BpeTrainer::builder()
        .vocab_size(48)
        .min_frequency(1)
        .show_progress(false)
        .end_of_word_suffix("a".into())
        .build();
    for (late, limited) in [(false, false), (true, false), (false, true)] {
        trainer.limit_alphabet = limited.then_some(2);
        trainer.initial_alphabet = if limited {
            ['a', 'b'].into()
        } else {
            Default::default()
        };
        let mut words = counts(&[("baaba", 1)]);
        if late {
            words.insert("xyxyxy".into(), 100);
        }
        let execution = execution::Execution::new(2).unwrap();
        let mut trace = Vec::new();
        let mut retained_alphabet = None;
        let outcome = execution.pool.install(|| {
            let progress = TrainingProgress::new(false, trainer.progress_format).unwrap();
            train_attempt(
                &trainer,
                &words,
                IdentityPolicy::Fresh,
                &execution,
                merge::MergeOptions::default(),
                &progress,
                &mut retained_alphabet,
                &mut trace,
            )
            .unwrap()
        });
        assert!(matches!(outcome, AttemptOutcome::RestartForReuse));
        assert_eq!(trace.is_empty(), !late);
        if limited {
            assert_eq!(retained_alphabet.as_deref(), Some(['a', 'b'].as_slice()));
        }
        // Exact oracle traces expose duplicate publication on a late restart.
        // They also cover a first-rule collision and rebuilt vocabulary IDs.
        check_with_workers(&trainer, &words, &[1, 4]);
    }
}

#[test]
fn affix_first_activations_preserve_reserved_ids_and_model_order() {
    let words = counts(&[
        (&"xabcdab中abab".repeat(40), 7),
        (&"abababaaaa中文".repeat(20), 11),
        ("zeroaaaa🙂", 0),
        ("", 1),
    ]);
    let trainer = BpeTrainer::builder()
        .vocab_size(80)
        .min_frequency(2)
        .show_progress(false)
        .max_token_length(Some(7))
        .continuing_subword_prefix("##".into())
        .end_of_word_suffix("</w>".into())
        .special_tokens(vec![AddedToken::from("##ab", true)])
        .build();
    let execution = execution::Execution::new(2).unwrap();
    let mut trace = Vec::new();
    let outcome = execution.pool.install(|| {
        let progress = TrainingProgress::new(false, trainer.progress_format).unwrap();
        train_attempt(
            &trainer,
            &words,
            IdentityPolicy::Fresh,
            &execution,
            merge::MergeOptions::default(),
            &progress,
            &mut None,
            &mut trace,
        )
        .unwrap()
    });
    let AttemptOutcome::Complete((vocab, _, _)) = outcome else {
        panic!("first activations do not require ID-reuse execution");
    };
    assert_eq!(vocab["##ab"], 0);
    assert!(trace.iter().any(|&(_, _, id)| id == 0));
    check_with_workers(&trainer, &words, &[1, 4]);
}

#[test]
fn planned_edges_match_materialized_slots_across_word_and_seek_boundaries() {
    use corpus::InitialPairSource;
    let long = "a测éxb".repeat(2500);
    let mut words = counts(&[("", 1), ("x", 0), ("a测éxb", 7), ("xa", 3), ("aé", 0)]);
    words.insert(long.into(), 2);
    words.insert(
        format!(
            "{}a{}测{}",
            "x".repeat(16000),
            "x".repeat(16000),
            "x".repeat(16000)
        )
        .into(),
        5,
    );
    for affixes in [false, true] {
        for limited in [false, true] {
            let mut trainer = BpeTrainer::builder()
                .vocab_size(100)
                .show_progress(false)
                .build();
            if affixes {
                trainer.continuing_subword_prefix = Some("##".into());
                trainer.end_of_word_suffix = Some("</w>".into());
            }
            if limited {
                trainer.limit_alphabet = Some(2);
                trainer.initial_alphabet = ['a', '测'].into();
            }
            let execution = execution::Execution::new(4).unwrap();
            let progress = TrainingProgress::new(false, trainer.progress_format).unwrap();
            execution.pool.install(|| {
                let mut retained = None;
                let mut vocab = vocabulary::Vocabulary::initialize(
                    &trainer,
                    &words,
                    4,
                    &progress,
                    &mut retained,
                )
                .unwrap();
                let plan = corpus::PreparedCorpus::build(
                    &words,
                    &mut vocab,
                    IdentityPolicy::Reusable,
                    false,
                    &progress,
                )
                .unwrap();
                let len = (&plan).len();
                let mut ranges = vec![0..len - 1, 0..1, len - 2..len - 1];
                for start in [1, 2, 3, 4095, 4096, 4097, 8191, 8192, len / 2] {
                    if start < len - 1 {
                        ranges.push(start..(start + 3).min(len - 1));
                        ranges.push(start..start);
                    }
                }
                // The complete comparison includes empty words and every separator.
                let mut all = Vec::new();
                (&plan).for_each_edge(0..len - 1, |position, key| all.push((position, key)));
                let observed: Vec<_> = ranges
                    .iter()
                    .map(|range| {
                        let mut pairs = Vec::new();
                        (&plan).for_each_edge(range.clone(), |position, key| {
                            pairs.push((position, key))
                        });
                        pairs
                    })
                    .collect();
                let corpus = plan
                    .materialize::<corpus::FullSlots>(4, IdentityPolicy::Reusable, &progress)
                    .unwrap();
                let source = corpus.initial_view();
                for (range, actual) in ranges.into_iter().zip(observed) {
                    let mut expected = Vec::new();
                    source.for_each_edge(range.clone(), |position, key| {
                        expected.push((position, key))
                    });
                    assert_eq!(
                        actual, expected,
                        "affixes={affixes} limited={limited} range={range:?}"
                    );
                }
                let mut expected = Vec::new();
                source.for_each_edge(0..len - 1, |position, key| expected.push((position, key)));
                assert_eq!(all, expected);
            });
        }
    }
}

#[test]
fn planned_wave_tables_preserve_coordinates_weights_and_filtering() {
    #[derive(Debug, PartialEq, Eq)]
    struct InitialSnapshot {
        weighted_mass: u128,
        maximum_word_weight: u64,
        entries: Vec<(u64, u64, Vec<u64>)>,
    }
    fn snapshot(table: initial_pairs::InitialPairTable<'_>) -> InitialSnapshot {
        let mut entries: Vec<_> = table
            .shards
            .into_iter()
            .flat_map(|shard| shard.into_iter())
            .map(|(key, state)| {
                (
                    key,
                    state.ledger_count_bits,
                    state.positions.iter().collect(),
                )
            })
            .collect();
        entries.sort_unstable_by_key(|entry| entry.0);
        InitialSnapshot {
            weighted_mass: table.weighted_mass,
            maximum_word_weight: table.maximum_word_weight,
            entries,
        }
    }
    let words = counts(&[
        ("", 1),
        ("x", 0),
        ("ab测éab测éab", 7),
        ("ab测é", 0),
        ("baab", 2),
    ]);
    let mut trainer = BpeTrainer::builder()
        .vocab_size(100)
        .show_progress(false)
        .build();
    trainer.continuing_subword_prefix = Some("##".into());
    trainer.end_of_word_suffix = Some("</w>".into());
    trainer.limit_alphabet = Some(3);
    trainer.initial_alphabet = ['a', 'b', '测'].into();
    for workers in [1, 2, 4] {
        let execution = execution::Execution::new(workers).unwrap();
        let progress = TrainingProgress::new(false, trainer.progress_format).unwrap();
        execution.pool.install(|| {
            for wave in [2, 7, 1 << 28] {
                for minimum in [0, 2, 8] {
                    let mut retained = None;
                    let mut vocab = vocabulary::Vocabulary::initialize(
                        &trainer,
                        &words,
                        workers,
                        &progress,
                        &mut retained,
                    )
                    .unwrap();
                    let plan = corpus::PreparedCorpus::build(
                        &words,
                        &mut vocab,
                        IdentityPolicy::Reusable,
                        false,
                        &progress,
                    )
                    .unwrap();
                    let arena = AllocationArena::new(workers, plan.initial_edges());
                    let actual = snapshot(
                        initial_pairs::build_in_waves(
                            &plan, minimum, &execution, &arena, &progress, wave,
                        )
                        .unwrap(),
                    );
                    let corpus = plan
                        .materialize::<corpus::FullSlots>(
                            workers,
                            IdentityPolicy::Reusable,
                            &progress,
                        )
                        .unwrap();
                    let expected = snapshot(
                        initial_pairs::build_in_waves(
                            corpus.initial_view(),
                            minimum,
                            &execution,
                            &arena,
                            &progress,
                            wave,
                        )
                        .unwrap(),
                    );
                    assert_eq!(
                        actual, expected,
                        "workers={workers} wave={wave} minimum={minimum}"
                    );
                }
            }
        });
    }
}

#[test]
fn zero_merge_paths_preserve_count_overflow_and_signed_policy_checks() {
    let mut trainer = BpeTrainer::builder()
        .vocab_size(2)
        .min_frequency(1)
        .show_progress(false)
        .build();
    for words in [counts(&[("ab", 7)]), counts(&[("aba", u64::MAX)])] {
        let (vocab, merges, _) = trainer.do_train(&words).unwrap();
        assert_eq!(vocab.len(), 2);
        assert!(merges.is_empty());
    }
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
}

#[test]
fn long_word_split_preserves_unicode_filtering_affixes_and_merge_trace() {
    let long = "ab测é".repeat(2049);
    let filtered = format!(
        "{}a{}测{}",
        "x".repeat(8193),
        "x".repeat(8193),
        "x".repeat(8193)
    );
    let words = counts(&[(&long, 3), (&filtered, 2), ("", 1), ("ab", 0)]);
    for filtered in [false, true] {
        for affixes in [false, true] {
            let mut trainer = BpeTrainer::builder()
                .vocab_size(48)
                .min_frequency(2)
                .max_token_length(Some(32))
                .show_progress(false)
                .build();
            if filtered {
                trainer.limit_alphabet = Some(2);
                trainer.initial_alphabet = ['a', '测'].into();
            }
            if affixes {
                trainer.continuing_subword_prefix = Some("##".into());
                trainer.end_of_word_suffix = Some("</w>".into());
            }
            check_with_workers(&trainer, &words, &[1, 4, 16]);
        }
    }
}

#[test]
fn owner_single_producer_layout_factors_preserve_hf_trace_and_fallbacks() {
    use merge::{MergeOptions, PairLayout};
    let fixtures = [
        counts(&[
            ("ab", 1000),
            ("cd", 900),
            ("abcd", 20),
            ("cdab", 15),
            ("ababcdcd", 9),
        ]),
        counts(&[
            ("aaaaaa", 23),
            ("aaabaaa", 17),
            ("abaaab", 11),
            ("cdabcd", 7),
        ]),
        counts(&[
            ("猫猫猫鱼", 19),
            ("猫鱼猫鱼", 13),
            ("abcdef", 7),
            ("abcabc", 3),
        ]),
    ];
    for (fixture_index, words) in fixtures.iter().enumerate() {
        for affixes in [false, true] {
            let mut builder = BpeTrainer::builder()
                .vocab_size(45)
                .min_frequency(3)
                .show_progress(false);
            if affixes {
                builder = builder
                    .continuing_subword_prefix("##".into())
                    .end_of_word_suffix("</w>".into())
                    .max_token_length(Some(9));
            }
            let trainer = builder.build();
            let mut expected_trace = Vec::new();
            let expected = trainer
                .do_train_observed(words, |pair, count, id| {
                    expected_trace.push((pair, count, id))
                })
                .unwrap();
            for layout in [
                PairLayout::Grid,
                PairLayout::Tail,
                PairLayout::Whole,
                PairLayout::Pack,
            ] {
                for single_producer_fast in [false, true] {
                    for direct_encoding in [false, true] {
                        for workers in [1, 4] {
                            let options = MergeOptions {
                                layout,
                                single_producer_fast,
                                direct_encoding,
                                group_births_in_commit: false,
                                direct_cold_encoding: false,
                                diagnostics: true,
                                fast_shard_router: false,
                                logical_owners: 0,
                                removal_entry: false,
                                removal_reduce: false,
                                removal_statistics: false,
                                removal_selective: false,
                                batch_limit: 0,
                            };
                            let mut trace = Vec::new();
                            let actual = train_with_merge_options(
                                &trainer,
                                words,
                                workers,
                                options,
                                Some(&mut |pair, count, id| trace.push((pair, count, id))),
                            )
                            .unwrap();
                            assert_eq!(
                                trace, expected_trace,
                                "fixture={fixture_index}, affixes={affixes}, options={options:?}, workers={workers}"
                            );
                            assert_eq!(actual, expected);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn owner_single_producer_requires_full_task_and_prunes_complete_births() {
    use merge::{MergeOptions, PairLayout};
    let words = counts(&[
        ("ab", 1000),
        ("cd", 900),
        ("abcd", 20),
        ("cdab", 15),
        ("ababcdcd", 9),
    ]);
    let trainer = BpeTrainer::builder()
        .vocab_size(30)
        .min_frequency(2)
        .show_progress(false)
        .build();
    for workers in [1, 4] {
        for floor in [2, 100] {
            for direct_encoding in [false, true] {
                let execution = execution::Execution::new(workers).unwrap();
                let progress = TrainingProgress::new(false, trainer.progress_format).unwrap();
                execution.pool.install(|| {
                    let mut retained = None;
                    let mut vocab = vocabulary::Vocabulary::initialize(
                        &trainer,
                        &words,
                        workers,
                        &progress,
                        &mut retained,
                    )
                    .unwrap();
                    let plan = corpus::PreparedCorpus::build(
                        &words,
                        &mut vocab,
                        IdentityPolicy::Fresh,
                        false,
                        &progress,
                    )
                    .unwrap();
                    let arena = AllocationArena::new(workers, plan.initial_edges());
                    let initial =
                        initial_pairs::build_initial_pairs(&plan, 2, &execution, &arena, &progress)
                            .unwrap();
                    let mut corpus = plan
                        .materialize::<corpus::FullSlots>(workers, IdentityPolicy::Fresh, &progress)
                        .unwrap();
                    let mut index = pair_index::PairIndex::from_initial_pairs(
                        initial,
                        IdentityPolicy::Fresh,
                        2,
                    )
                    .unwrap();
                    let mut rules = Vec::new();
                    let mut candidates = Vec::new();
                    index.begin_selection();
                    for pair in [(0, 1), (2, 3)] {
                        assert_eq!(pair_index::key_pair(index.best().unwrap().key), pair);
                        candidates.push(index.take_best());
                        let identity = vocab.resolve_merge(vocab.merge_token(pair)).unwrap();
                        assert!(!identity.reused_active_id);
                        corpus.prepare_spans(pair, identity.id, false);
                        rules.push(merge::MergeRule {
                            pair,
                            replacement: identity.id,
                        });
                    }
                    index.end_selection();
                    let (prepared, births) = merge::prepare_merges_with_births(
                        &corpus,
                        &rules,
                        &candidates,
                        IdentityPolicy::Fresh,
                        vocab.len(),
                        usize::MAX,
                        &execution,
                        &arena,
                        floor,
                        MergeOptions {
                            single_producer_fast: true,
                            direct_encoding,
                            group_births_in_commit: false,
                            direct_cold_encoding: false,
                            layout: PairLayout::Grid,
                            diagnostics: true,
                            fast_shard_router: false,
                            logical_owners: 0,
                            removal_entry: false,
                            removal_reduce: false,
                            removal_statistics: false,
                            removal_selective: false,
                            batch_limit: 0,
                        },
                        None,
                    )
                    .unwrap();
                    if workers == 1 {
                        assert_eq!(prepared.diagnostics[0].chunks, 1);
                        assert_eq!(prepared.diagnostics[0].fast_ranks.len(), 2);
                        if floor == 2 {
                            let mut actual = AHashMap::new();
                            for (key, state) in &births {
                                assert!(
                                    actual
                                        .insert(
                                            *key,
                                            (state.ledger_count_bits, state.positions.len())
                                        )
                                        .is_none()
                                );
                            }
                            assert_eq!(
                                actual[&pair_index::pair_key((
                                    rules[0].replacement,
                                    rules[1].replacement
                                ))],
                                (29, 2)
                            );
                            assert_eq!(
                                actual[&pair_index::pair_key((
                                    rules[1].replacement,
                                    rules[0].replacement
                                ))],
                                (15, 1)
                            );
                            assert_eq!(
                                actual[&pair_index::pair_key((
                                    rules[0].replacement,
                                    rules[0].replacement
                                ))],
                                (9, 1)
                            );
                            assert_eq!(
                                actual[&pair_index::pair_key((
                                    rules[1].replacement,
                                    rules[1].replacement
                                ))],
                                (9, 1)
                            );
                        } else {
                            assert!(births.is_empty());
                        }
                    } else {
                        assert!(
                            prepared
                                .diagnostics
                                .iter()
                                .flat_map(|job| &job.tasks)
                                .all(|task| !task.full)
                        );
                        assert!(
                            births.is_empty(),
                            "partial tasks cannot take producer fast path"
                        );
                    }
                    let events = prepared.apply(&mut corpus);
                    if workers == 1 {
                        assert!(
                            events
                                .chunks
                                .iter()
                                .flat_map(|chunk| &chunk.changes)
                                .all(|change| change.positions.is_empty())
                        );
                        assert!(
                            events
                                .chunks
                                .iter()
                                .flat_map(|chunk| &chunk.changes)
                                .any(|change| change.removed_weight != 0)
                        );
                    }
                    candidates.clear();
                    index
                        .commit_merges_with_prepared(
                            &events,
                            vocab.len(),
                            &execution,
                            &arena,
                            births,
                            None,
                        )
                        .unwrap();
                    index.begin_selection();
                    if workers == 1 && floor == 100 {
                        assert!(index.best().is_none());
                    } else {
                        assert_eq!(index.best().unwrap().priority_count, 29);
                    }
                });
            }
        }
    }
}

#[test]
fn owner_single_producer_active_reuse_restarts_through_original_cohorts() {
    use merge::{MergeOptions, PairLayout};
    let trainer = BpeTrainer::builder()
        .vocab_size(48)
        .min_frequency(1)
        .show_progress(false)
        .end_of_word_suffix("a".into())
        .build();
    for words in [
        counts(&[("baaba", 1)]),
        counts(&[("baaba", 1), ("xyxyxy", 100)]),
    ] {
        let mut expected_trace = Vec::new();
        let expected = trainer
            .do_train_observed(&words, |pair, count, id| {
                expected_trace.push((pair, count, id))
            })
            .unwrap();
        for layout in [
            PairLayout::Grid,
            PairLayout::Tail,
            PairLayout::Whole,
            PairLayout::Pack,
        ] {
            let mut trace = Vec::new();
            let actual = train_with_merge_options(
                &trainer,
                &words,
                4,
                MergeOptions {
                    layout,
                    single_producer_fast: true,
                    direct_encoding: true,
                    group_births_in_commit: false,
                    direct_cold_encoding: false,
                    diagnostics: true,
                    fast_shard_router: false,
                    logical_owners: 0,
                    removal_entry: false,
                    removal_reduce: false,
                    removal_statistics: false,
                    removal_selective: false,
                    batch_limit: 0,
                },
                Some(&mut |pair, count, id| trace.push((pair, count, id))),
            )
            .unwrap();
            assert_eq!(trace, expected_trace, "layout={layout:?}");
            assert_eq!(actual, expected);
        }
    }
}

#[test]
fn commit_fused_group_and_direct_cold_preserve_hf_trace_and_reuse() {
    use merge::{MergeOptions, PairLayout};
    let fixtures = [
        counts(&[
            ("ab", 1000),
            ("cd", 900),
            ("abcd", 20),
            ("cdab", 15),
            ("ababcdcd", 9),
        ]),
        counts(&[
            ("aaaaaa", 23),
            ("aaabaaa", 17),
            ("abaaab", 11),
            ("cdabcd", 7),
        ]),
        counts(&[
            ("猫猫猫鱼", 19),
            ("猫鱼猫鱼", 13),
            ("abcdef", 7),
            ("abcabc", 3),
        ]),
    ];
    for words in &fixtures {
        for affixes in [false, true] {
            let mut builder = BpeTrainer::builder()
                .vocab_size(50)
                .min_frequency(3)
                .show_progress(false);
            if affixes {
                builder = builder
                    .continuing_subword_prefix("##".into())
                    .end_of_word_suffix("</w>".into())
                    .max_token_length(Some(9));
            }
            let trainer = builder.build();
            let mut expected_trace = Vec::new();
            let expected = trainer
                .do_train_observed(words, |pair, count, id| {
                    expected_trace.push((pair, count, id))
                })
                .unwrap();
            for workers in [1, 4] {
                for group_births_in_commit in [false, true] {
                    for direct_cold_encoding in [false, true] {
                        let options = MergeOptions {
                            single_producer_fast: true,
                            direct_encoding: true,
                            group_births_in_commit,
                            direct_cold_encoding,
                            layout: PairLayout::Whole,
                            diagnostics: false,
                            fast_shard_router: false,
                            logical_owners: 0,
                            removal_entry: false,
                            removal_reduce: false,
                            removal_statistics: false,
                            removal_selective: false,
                            batch_limit: 0,
                        };
                        let mut trace = Vec::new();
                        let actual = train_with_merge_options(
                            &trainer,
                            words,
                            workers,
                            options,
                            Some(&mut |pair, count, id| trace.push((pair, count, id))),
                        )
                        .unwrap();
                        assert_eq!(
                            trace, expected_trace,
                            "workers={workers} options={options:?}"
                        );
                        assert_eq!(actual, expected);
                    }
                }
            }
        }
    }
}

#[test]
fn owner_router_balance_preserves_trace_and_reuse() {
    use merge::{MergeOptions, PairLayout};
    for affixes in [false, true] {
        let words = counts(&[
            ("aaaaaa", 23),
            ("aaabaaa", 17),
            ("abccdabcd", 11),
            ("猫猫猫鱼", 19),
            ("猫鱼猫鱼", 13),
        ]);
        let mut builder = BpeTrainer::builder()
            .vocab_size(50)
            .min_frequency(3)
            .show_progress(false);
        if affixes {
            builder = builder
                .continuing_subword_prefix("##".into())
                .end_of_word_suffix("</w>".into())
                .max_token_length(Some(9));
        }
        let trainer = builder.build();
        let mut expected_trace = Vec::new();
        let expected = trainer
            .do_train_observed(&words, |pair, count, id| {
                expected_trace.push((pair, count, id))
            })
            .unwrap();
        for workers in [1, 6] {
            for logical_owners in [0, 8, 12] {
                for fast_shard_router in [false, true] {
                    let options = MergeOptions {
                        single_producer_fast: true,
                        direct_encoding: true,
                        group_births_in_commit: true,
                        layout: PairLayout::Whole,
                        fast_shard_router,
                        logical_owners,
                        ..Default::default()
                    };
                    let mut trace = Vec::new();
                    let actual = train_with_merge_options(
                        &trainer,
                        &words,
                        workers,
                        options,
                        Some(&mut |pair, count, id| trace.push((pair, count, id))),
                    )
                    .unwrap();
                    assert_eq!(
                        trace, expected_trace,
                        "workers={workers} options={options:?}"
                    );
                    assert_eq!(actual, expected);
                }
            }
        }
    }
}

#[test]
fn commit_removal_reduce_matches_hf_trace_and_reuse() {
    use merge::{MergeOptions, PairLayout};
    for affixes in [false, true] {
        let words = counts(&[
            ("ab", 1000),
            ("cd", 900),
            ("abcd", 20),
            ("cdab", 15),
            ("ababcdcd", 9),
            ("aaaaaa", 23),
            ("aaabaaa", 17),
            ("猫鱼猫鱼", 13),
        ]);
        let mut builder = BpeTrainer::builder()
            .vocab_size(60)
            .min_frequency(3)
            .show_progress(false);
        if affixes {
            builder = builder
                .continuing_subword_prefix("##".into())
                .end_of_word_suffix("</w>".into())
                .max_token_length(Some(9));
        }
        let trainer = builder.build();
        let mut expected_trace = Vec::new();
        let expected = trainer
            .do_train_observed(&words, |pair, count, id| {
                expected_trace.push((pair, count, id))
            })
            .unwrap();
        for workers in [1, 6] {
            for removal_entry in [false, true] {
                for removal_reduce in [false, true] {
                    let options = MergeOptions {
                        single_producer_fast: true,
                        direct_encoding: true,
                        group_births_in_commit: true,
                        fast_shard_router: true,
                        layout: PairLayout::Whole,
                        removal_entry,
                        removal_reduce,
                        removal_selective: true,
                        ..Default::default()
                    };
                    let mut trace = Vec::new();
                    let actual = train_with_merge_options(
                        &trainer,
                        &words,
                        workers,
                        options,
                        Some(&mut |pair, count, id| trace.push((pair, count, id))),
                    )
                    .unwrap();
                    assert_eq!(
                        trace, expected_trace,
                        "workers={workers} options={options:?}"
                    );
                    assert_eq!(actual, expected);
                }
            }
        }
    }
}

#[test]
fn commit_removal_reduce_batch_limits_preserve_hf_trace_and_reuse() {
    use merge::{MergeOptions, PairLayout};
    let words = counts(&[
        ("abcdabcd", 41),
        ("abcde", 37),
        ("cdefcdef", 31),
        ("ghijghij", 29),
        ("klmnklmn", 23),
        ("aaaaaaa", 17),
        ("猫猫鱼鱼", 13),
    ]);
    for affixes in [false, true] {
        let mut builder = BpeTrainer::builder()
            .vocab_size(75)
            .min_frequency(2)
            .show_progress(false);
        if affixes {
            builder = builder
                .continuing_subword_prefix("##".into())
                .end_of_word_suffix("</w>".into())
                .max_token_length(Some(9));
        }
        let trainer = builder.build();
        let mut expected_trace = Vec::new();
        let expected = trainer
            .do_train_observed(&words, |pair, count, id| {
                expected_trace.push((pair, count, id))
            })
            .unwrap();
        for workers in [1, 6] {
            for batch_limit in [1, 8, 256] {
                let options = MergeOptions {
                    single_producer_fast: true,
                    direct_encoding: true,
                    group_births_in_commit: true,
                    fast_shard_router: true,
                    layout: PairLayout::Whole,
                    removal_reduce: true,
                    removal_selective: true,
                    batch_limit,
                    ..Default::default()
                };
                let mut trace = Vec::new();
                let actual = train_with_merge_options(
                    &trainer,
                    &words,
                    workers,
                    options,
                    Some(&mut |pair, count, id| trace.push((pair, count, id))),
                )
                .unwrap();
                assert_eq!(
                    trace, expected_trace,
                    "workers={workers} options={options:?}"
                );
                assert_eq!(actual, expected);
            }
        }
    }
}
