//! Complete-producer encoding and partial fallback equivalence.
use super::*;

#[test]
fn producer_encoding_preserves_hf_trace_and_fallbacks() {
    use merge::MergeOptions;
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

            for single_producer_fast in [false, true] {
                for workers in [1, 4] {
                    let options = MergeOptions {
                        single_producer_fast,
                    };
                    let mut trace = Vec::new();
                    let actual = train_with_merge_options(
                        &trainer,
                        WordCountsView::from_map(words),
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

#[test]
fn owner_single_producer_requires_full_task_and_prunes_complete_births() {
    use merge::MergeOptions;
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
            let execution = execution::Execution::new(workers).unwrap();
            let progress = TrainingProgress::new(false, trainer.progress_format).unwrap();
            execution.pool.install(|| {
                let mut retained = None;
                let mut vocab = vocabulary::Vocabulary::initialize(
                    &trainer,
                    WordCountsView::from_map(&words),
                    workers,
                    &progress,
                    &mut retained,
                )
                .unwrap();
                let plan = corpus::CorpusPlan::build(
                    WordCountsView::from_map(&words),
                    &mut vocab,
                    IdentityPolicy::FirstActivationOnly,
                    false,
                    &progress,
                )
                .unwrap();
                let arena = AllocationArena::new(workers, plan.initial_edges());
                let initial =
                    initial_pairs::InitialPairTable::build(&plan, 2, &execution, &arena, &progress)
                        .unwrap();
                let mut corpus = plan
                    .materialize::<corpus::U32Slots>(
                        workers,
                        IdentityPolicy::FirstActivationOnly,
                        &progress,
                    )
                    .unwrap();
                let mut index = pair_index::PairIndex::from_initial_pairs(
                    initial,
                    IdentityPolicy::FirstActivationOnly,
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
                    IdentityPolicy::FirstActivationOnly,
                    vocab.len(),
                    usize::MAX,
                    &execution,
                    &arena,
                    floor,
                    MergeOptions {
                        single_producer_fast: true,
                    },
                )
                .unwrap();
                if workers == 1 {
                    if floor == 2 {
                        let mut actual = AHashMap::new();
                        for birth in &births {
                            assert!(
                                actual
                                    .insert(birth.key, (birth.weight, birth.positions.len()))
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
                    .commit_merges_with_prepared(&events, vocab.len(), &execution, &arena, births)
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
