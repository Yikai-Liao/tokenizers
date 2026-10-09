//! Reserved identities, active aliases, cohort semantics, and attempt restart.
use super::*;

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
        WordCountsView::from_map(&words),
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
    let (vocab, _, _) = train(&trainer, WordCountsView::from_map(&words), 2, None).unwrap();
    assert_eq!(vocab["##ab"], 0);
    check_with_workers(&trainer, &words, &[1, 4, 8]);
}
