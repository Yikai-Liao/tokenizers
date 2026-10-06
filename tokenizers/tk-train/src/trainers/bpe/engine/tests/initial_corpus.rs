//! Coordinate planning, full pair keys, wave boundaries, and materialization.
use super::*;

#[test]
fn full_pair_keys_remain_distinct_in_initial_counting() {
    use super::storage::IntervalIndex;
    use std::sync::atomic::AtomicU32;
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
    for workers in [1, 4, 65] {
        let execution = execution::Execution::new(workers).unwrap();
        let arena = AllocationArena::new(workers, 3);
        let progress =
            TrainingProgress::new(false, tk_encode::utils::progress::ProgressFormat::Silent)
                .unwrap();
        execution.pool.install(|| {
            let initial = initial_pairs::InitialPairTable::build(
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
    use super::storage::IntervalIndex;
    use std::sync::atomic::AtomicU32;
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
                let initial = initial_pairs::InitialPairTable::build_in_waves(
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
                    WordCountsView::from_map(&words),
                    4,
                    &progress,
                    &mut retained,
                )
                .unwrap();
                let plan = corpus::CorpusPlan::build(
                    WordCountsView::from_map(&words),
                    &mut vocab,
                    IdentityPolicy::AllowActiveReuse,
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
                let observed: Vec<_> = ranges
                    .iter()
                    .map(|range| {
                        let mut pairs = Vec::new();
                        (&plan).for_each_edge(range.clone(), |position, key| {
                            pairs.push((position, key))
                        });
                        assert_eq!(
                            (&plan).edge_count(range.clone()),
                            pairs.len(),
                            "planned capacity for {range:?}"
                        );
                        pairs
                    })
                    .collect();
                let corpus = plan
                    .materialize::<corpus::U32Slots>(4, IdentityPolicy::AllowActiveReuse, &progress)
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
            });
        }
    }
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
