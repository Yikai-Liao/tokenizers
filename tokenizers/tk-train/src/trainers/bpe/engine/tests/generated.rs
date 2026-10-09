//! Fixed-seed combinations exercise public results and complete rule traces.
use super::*;

#[test]
fn generated_corpora_keep_complete_models_and_traces() {
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
            let length = next() as usize % 32;
            let text: String = (0..length)
                .map(|_| alphabet[next() as usize % alphabet.len()])
                .collect();
            *words.entry(text.into()).or_default() += next() % 8;
        }
        let mut trainer = BpeTrainer::builder()
            .vocab_size(48)
            .min_frequency(case % 4)
            .show_progress(false)
            .build();
        trainer.max_token_length =
            [None, Some(0), Some(1), Some(2), Some(3), Some(9)][case as usize % 6];
        trainer.continuing_subword_prefix =
            [None, Some(""), Some("##"), Some("a")][case as usize % 4].map(str::to_owned);
        trainer.end_of_word_suffix =
            [None, Some(""), Some("</w>"), Some("a")][case as usize / 4 % 4].map(str::to_owned);
        trainer.special_tokens = ["ab", "aa", "##a", "a🙂"]
            .into_iter()
            .take(case as usize % 5)
            .map(|s| AddedToken::from(s, true))
            .collect();
        check_with_workers(&trainer, &words, &[1, 4, 8]);
    }
}

#[test]
fn long_repeats_non_bmp_and_wide_ids_preserve_models() {
    let words = counts(&[
        (&"a".repeat(4097), 3),
        (&"abab🙂猫".repeat(1024), 2),
        ("e\u{301}e\u{301}", 7),
        ("", 0),
    ]);
    for target in [0, 1, 64, 65535, 65536, 100000] {
        let trainer = BpeTrainer::builder()
            .vocab_size(target)
            .min_frequency(2)
            .show_progress(false)
            .build();
        check_with_workers(&trainer, &words, &[1, 4, 8, 16]);
    }
}

#[test]
fn actual_high_token_ids_keep_pair_order_and_separator_distinct() {
    let special: Vec<_> = std::iter::once("a".to_owned())
        .chain((1..65536).map(|i| format!("<reserved{i}>")))
        .map(|s| AddedToken::from(s, true))
        .collect();
    let trainer = BpeTrainer::builder()
        .vocab_size(65550)
        .min_frequency(1)
        .special_tokens(special)
        .show_progress(false)
        .build();
    let words = counts(&[("abab", 3), ("bbaa", 2)]);
    check_with_workers(&trainer, &words, &[1, 4]);
    let (vocab, _, _) = trainer.do_train(&words).unwrap();
    assert_eq!(vocab["a"], 0);
    assert_eq!(vocab["<reserved65535>"], 65535);
    assert_eq!(vocab["b"], 65536);
}
