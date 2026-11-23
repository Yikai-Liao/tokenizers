use super::{BpeTrainer, reference};
use super::{corpus::Corpus, index::PairIndex, word::Word};
use ahash::AHashMap;
use compact_str::CompactString;
use std::collections::HashSet;
use tk_encode::vocab::bucket_added_vocabulary::AddedToken;

fn counts(entries: &[(&str, u64)]) -> AHashMap<CompactString, u64> {
    entries
        .iter()
        .map(|(word, count)| (CompactString::from(*word), *count))
        .collect()
}

fn assert_matches_reference(trainer: &BpeTrainer, word_counts: &AHashMap<CompactString, u64>) {
    let actual = trainer.do_train(word_counts).unwrap();
    let expected = reference::train(trainer, word_counts).unwrap();
    assert_eq!(actual.0, expected.0);
    assert_eq!(actual.1, expected.1);
    assert_eq!(actual.2, expected.2);
}

#[test]
fn deterministic_corpora_cover_trainer_options() {
    let word_counts = counts(&[
        ("abab", 7),
        ("ababa", 4),
        ("baba", 5),
        ("a🙂b", 3),
        ("aaaa", 2),
        ("<s>a", 1),
        ("", 1),
    ]);
    let mut cases = vec![
        BpeTrainer::builder().show_progress(false).build(),
        BpeTrainer::builder()
            .show_progress(false)
            .min_frequency(4)
            .vocab_size(12)
            .build(),
        BpeTrainer::builder()
            .show_progress(false)
            .vocab_size(5)
            .build(),
        BpeTrainer::builder()
            .show_progress(false)
            .initial_alphabet(HashSet::from(['z', '🙂']))
            .build(),
        BpeTrainer::builder()
            .show_progress(false)
            .continuing_subword_prefix("##".into())
            .end_of_word_suffix("</w>".into())
            .build(),
        BpeTrainer::builder()
            .show_progress(false)
            .continuing_subword_prefix("a".into())
            .max_token_length(Some(3))
            .build(),
        BpeTrainer::builder()
            .show_progress(false)
            .max_token_length(Some(1))
            .min_frequency(0)
            .build(),
        BpeTrainer::builder()
            .show_progress(false)
            .special_tokens(vec![
                AddedToken::from("ab", false),
                AddedToken::from("<s>", true),
                AddedToken::from("ab", true),
            ])
            .build(),
    ];
    cases.push(
        BpeTrainer::builder()
            .show_progress(false)
            .vocab_size(0)
            .build(),
    );
    cases.push(
        BpeTrainer::builder()
            .progress_format(super::ProgressFormat::JsonLines)
            .show_progress(true)
            .vocab_size(12)
            .build(),
    );

    for trainer in cases {
        assert_matches_reference(&trainer, &word_counts);
    }

    let limited_counts = counts(&[("a", 25), ("bb", 7), ("ccc", 3), ("dddd", 1), ("🙂", 2)]);
    let limited = BpeTrainer::builder()
        .show_progress(false)
        .limit_alphabet(3)
        .build();
    assert_matches_reference(&limited, &limited_counts);

    let alias_counts = counts(&[("aba", 7), ("bbbbbbbaba", 6)]);
    let alias = BpeTrainer::builder()
        .show_progress(false)
        .vocab_size(64)
        .min_frequency(1)
        .continuing_subword_prefix("ab".into())
        .build();
    assert_matches_reference(&alias, &alias_counts);
}

#[test]
fn fixed_seed_corpora_match_the_previous_merge_loop() {
    let alphabet = ['a', 'b', 'c', 'd', 'é', '🙂'];
    let mut state = 0x74b4_2c31_d8a9_5e07_u64;

    for case in 0..96 {
        let mut word_counts = AHashMap::new();
        for _ in 0..12 {
            let length = (next(&mut state) % 8) as usize;
            let word = (0..length)
                .map(|_| alphabet[(next(&mut state) as usize) % alphabet.len()])
                .collect::<String>();
            let count = next(&mut state) % 5 + 1;
            *word_counts.entry(CompactString::from(word)).or_default() += count;
        }

        let mut builder = BpeTrainer::builder()
            .show_progress(false)
            .min_frequency(next(&mut state) % 8)
            .vocab_size((next(&mut state) % 18 + 2) as usize);
        if case % 2 == 0 {
            builder = builder.limit_alphabet(20);
        }
        if case % 3 == 0 {
            builder = builder.initial_alphabet(HashSet::from(['a', '🙂']));
        }
        if case % 4 == 0 {
            builder = builder.continuing_subword_prefix("a".into());
        }
        if case % 5 == 0 {
            builder = builder.end_of_word_suffix("$".into());
        }
        if case % 6 == 0 {
            builder = builder.max_token_length(Some((next(&mut state) % 7) as usize));
        }
        if case % 7 == 0 {
            builder = builder.special_tokens(vec![AddedToken::from("ab", false)]);
        }
        let trainer = builder.build();
        let actual = trainer.do_train(&word_counts).unwrap();
        let expected = reference::train(&trainer, &word_counts).unwrap();
        assert_eq!(
            actual, expected,
            "case {case}, counts {word_counts:?}, trainer {trainer:?}"
        );
    }
}

#[test]
fn existing_token_aliases_match_the_previous_merge_loop() {
    let word_counts = counts(&[("aaba", 1)]);
    let trainer = BpeTrainer::builder()
        .show_progress(false)
        .min_frequency(1)
        .vocab_size(64)
        .continuing_subword_prefix("a".into())
        .special_tokens(vec![AddedToken::from("ab", false)])
        .build();
    assert_matches_reference(&trainer, &word_counts);
}

#[test]
fn whole_word_merge_respects_strict_max_token_length() {
    let word_counts = counts(&[("baba", 1)]);
    let trainer = BpeTrainer::builder()
        .show_progress(false)
        .min_frequency(1)
        .vocab_size(64)
        .continuing_subword_prefix("b".into())
        .max_token_length(Some(3))
        .build();
    assert_matches_reference(&trainer, &word_counts);
}

#[test]
fn whole_word_merge_respects_strict_left_neighbor_max_token_length() {
    let word_counts = counts(&[("ba", 2), ("baba", 1)]);
    let trainer = BpeTrainer::builder()
        .show_progress(false)
        .min_frequency(1)
        .vocab_size(64)
        .continuing_subword_prefix("b".into())
        .special_tokens(vec![AddedToken::from("bb", false)])
        .max_token_length(Some(4))
        .build();
    assert_matches_reference(&trainer, &word_counts);
}

#[test]
fn empty_input_and_full_width_counts_are_supported() {
    let empty = AHashMap::new();
    let empty_trainer = BpeTrainer::builder()
        .show_progress(false)
        .special_tokens(vec![AddedToken::from("<unk>", true)])
        .initial_alphabet(HashSet::from(['z']))
        .build();
    assert_matches_reference(&empty_trainer, &empty);
    assert!(empty_trainer.do_train(&empty).unwrap().1.is_empty());

    let wide = counts(&[("ab", u64::MAX)]);
    let trainer = BpeTrainer::builder()
        .show_progress(false)
        .min_frequency(u64::MAX)
        .build();
    let (vocab, merges, _) = trainer.do_train(&wide).unwrap();
    assert!(vocab.contains_key("ab"));
    assert_eq!(merges, [("a".into(), "b".into())]);
}

#[test]
fn repeated_u64_max_weight_returns_a_count_overflow_error() {
    let mut word = Word::new();
    for id in [0, 1, 0, 1] {
        word.add(id, 1);
    }
    let corpus = Corpus::from_words(vec![(word, u64::MAX)]);
    let error = match PairIndex::build(&corpus, 1, &None) {
        Err(error) => error,
        Ok(_) => panic!("pair count overflow should return an error"),
    };
    assert_eq!(error.to_string(), "BPE pair count exceeds u64");
}

fn next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}
