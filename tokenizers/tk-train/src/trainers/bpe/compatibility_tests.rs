use super::*;

fn check(trainer: &BpeTrainer, words: &AHashMap<CompactString, u64>, label: &str) {
    let reference: reference::BpeTrainer =
        serde_json::from_value(serde_json::to_value(trainer).unwrap()).unwrap();
    let expected = reference.do_train(words).unwrap();
    let mut trainer = trainer.clone();
    if trainer.limit_alphabet.is_some() {
        // The upstream frequency-tie cutoff itself is nondeterministic. Feed
        // both engines the same retained alphabet to test the merge algorithm.
        trainer.initial_alphabet = expected
            .0
            .keys()
            .filter_map(|token| {
                let mut chars = token.chars();
                let first = chars.next()?;
                chars.next().is_none().then_some(first)
            })
            .collect();
    }
    for workers in [1, 2, 4] {
        let actual = trainer
            .do_train_with_workers(words, workers)
            .unwrap_or_else(|error| panic!("{label}, workers={workers}: {error}"));
        assert_eq!(actual, expected, "{label}, workers={workers}");
    }
}

#[test]
fn yttm_matches_original_runs_ties_affixes_and_special_id_reuse() {
    let words = [
        ("aaaaab", 9),
        ("abcabc", 4),
        ("ababaa", 5),
        ("中中文文", 3),
        ("abc", 4),
        ("", 1),
    ]
    .into_iter()
    .map(|(word, count)| (word.into(), count))
    .collect();
    for prefix in [None, Some("##"), Some("a")] {
        for suffix in [None, Some("</w>"), Some("b")] {
            for specials in [
                vec![],
                vec![
                    AddedToken::from("ab", true),
                    AddedToken::from("abc", true),
                    AddedToken::from("aaaa", true),
                ],
            ] {
                let mut trainer = BpeTrainer::builder()
                    .vocab_size(40)
                    .show_progress(false)
                    .min_frequency(1)
                    .special_tokens(specials)
                    .build();
                trainer.continuing_subword_prefix = prefix.map(str::to_owned);
                trainer.end_of_word_suffix = suffix.map(str::to_owned);
                check(
                    &trainer,
                    &words,
                    &format!("prefix={prefix:?}, suffix={suffix:?}"),
                );
            }
        }
    }
}

#[test]
fn yttm_matches_original_randomized() {
    let mut random = 0x59e8354a_u64;
    let mut next = || {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        random
    };
    for case in 0..400 {
        let mut words = AHashMap::new();
        for _ in 0..(next() % 14 + 1) {
            let word: String = (0..next() % 30)
                .map(|_| ['a', 'b', 'c', '中', '文'][(next() % 5) as usize])
                .collect();
            words.insert(CompactString::from(word), next() % 8 + 1);
        }
        let mut trainer = BpeTrainer::builder()
            .vocab_size((next() % 60) as usize)
            .show_progress(false)
            .min_frequency(next() % 4)
            .build();
        trainer.continuing_subword_prefix = (case % 3 == 0).then(|| "##".into());
        trainer.end_of_word_suffix = (case % 4 == 0).then(|| "</w>".into());
        check(&trainer, &words, &format!("case={case}"));
    }
}

#[test]
fn yttm_matches_original_length_limits() {
    let words = [
        ("aaaabbbbbbcaaaa", 3),
        ("abcabcabc", 7),
        ("abcdefg", 1),
        ("中文中文中文", 5),
    ]
    .into_iter()
    .map(|(word, count)| (word.into(), count))
    .collect();
    for max in [0, 1, 2, 3, 4, 8, 16] {
        let trainer = BpeTrainer::builder()
            .vocab_size(100)
            .show_progress(false)
            .max_token_length(Some(max))
            .build();
        check(&trainer, &words, &format!("max={max}"));
    }
}

#[test]
fn yttm_feed_matches_original_callback_counts_and_propagates_errors() {
    let inputs = ["ab ab", "中 文 中", "", "aabaa", "ab", "中文"];
    let process = |s: &str| Ok(s.split_whitespace().map(str::to_owned).collect());
    let mut oracle = reference::BpeTrainer::builder()
        .show_progress(false)
        .build();
    oracle.feed(inputs.into_iter(), process).unwrap();
    // Obtain the original feed's table without exposing its private state.
    let original: BpeTrainer =
        serde_json::from_value(serde_json::to_value(&oracle).unwrap()).unwrap();
    for workers in [1, 2, 4, 8] {
        let words = feed::count(inputs.into_iter(), process, workers).unwrap();
        assert_eq!(words, original.words);
        let error = feed::count(
            inputs.into_iter(),
            |_| Err("callback failure".into()),
            workers,
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "callback failure");
        assert!(
            feed::count(std::iter::empty::<String>(), process, workers)
                .unwrap()
                .is_empty()
        );
    }
}

#[test]
fn yttm_waits_when_first_rules_birth_can_outrank_independent_second_rule() {
    let words = [("abab", 100), ("cd", 50)]
        .into_iter()
        .map(|(word, count)| (word.into(), count))
        .collect();
    let trainer = BpeTrainer::builder()
        .vocab_size(7)
        .show_progress(false)
        .build();
    check(
        &trainer,
        &words,
        "new (ab, ab) must precede independent (c, d)",
    );
}

#[test]
fn yttm_matches_original_mixed_compatibility_options() {
    let mut random = 0x729fee68_u64;
    let mut next = || {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        random
    };
    for case in 0..400 {
        let mut words = AHashMap::new();
        for _ in 0..(next() % 12 + 1) {
            let word: String = (0..next() % 40)
                .map(|_| ['a', 'b', '#', '中', '文'][(next() % 5) as usize])
                .collect();
            words.insert(CompactString::from(word), next() % 10 + 1);
        }
        let mut trainer = BpeTrainer::builder()
            .vocab_size((next() % 100) as usize)
            .show_progress(false)
            .min_frequency(next() % 4)
            .special_tokens(vec![
                AddedToken::from("ab", true),
                AddedToken::from("aa", true),
                AddedToken::from("##", true),
            ])
            .build();
        trainer.continuing_subword_prefix =
            [None, Some("##"), Some("a"), Some("#")][case % 4].map(str::to_owned);
        trainer.end_of_word_suffix = [None, Some("</w>"), Some("b")][case % 3].map(str::to_owned);
        trainer.max_token_length = [None, Some(0), Some(2), Some(4), Some(8)][case % 5];
        if case % 7 == 0 {
            trainer.limit_alphabet = Some(3);
        }
        check(&trainer, &words, &format!("mixed case={case}"));
    }
}

#[test]
#[ignore = "set YTTM_REFERENCE_CORPUS to a UTF-8 corpus for a real-input differential check"]
fn yttm_matches_original_real_corpus() {
    let path = std::env::var("YTTM_REFERENCE_CORPUS").expect("YTTM_REFERENCE_CORPUS");
    let text = std::fs::read_to_string(path).unwrap();
    let words = feed::count(
        text.lines(),
        |line| Ok(line.split_whitespace().map(str::to_owned).collect()),
        4,
    )
    .unwrap();
    let trainer = BpeTrainer::builder()
        .vocab_size(12000)
        .show_progress(false)
        .build();
    check(&trainer, &words, "real corpus, whitespace pre-tokenization");
}
