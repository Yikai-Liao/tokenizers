//! Integration tests compile tk-train without cfg(test), using production chunk sizes.
#![cfg(feature = "bpe")]
use ahash::AHashMap;
use compact_str::CompactString;
use tk_train::{BpeTrainer, Trainer};

#[test]
fn production_chunks_match_across_worker_counts() {
    let words: AHashMap<CompactString, u64> = [
        ("界".repeat(16_385).into(), 3),
        ("ab".repeat(8_193).into(), 2),
        ("한국어字🙂".repeat(9).into(), 1),
        ("aba".into(), 7),
        ("bbbbbbbaba".into(), 6),
    ]
    .into();
    let pools: Vec<_> = [1, 4]
        .into_iter()
        .map(|workers| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap()
        })
        .collect();
    for prefix in [None, Some("ab".to_owned())] {
        for limit in [None, Some(16)] {
            let mut trainer = BpeTrainer::builder()
                .show_progress(false)
                .vocab_size(128)
                .min_frequency(1)
                .max_token_length(limit)
                .build();
            trainer.continuing_subword_prefix = prefix.clone();
            let expected = pools[0].install(|| trainer.do_train(&words).unwrap());
            for pool in &pools {
                assert_eq!(pool.install(|| trainer.do_train(&words).unwrap()), expected);
            }
            if prefix.is_none() && limit.is_some() {
                assert!(expected.0.keys().all(|token| token.chars().count() <= 16));
            }
        }
    }
    let batch: Vec<_> = (0..4097).map(|i| format!("词{i}🙂")).collect();
    let mut expected = None;
    for pool in &pools {
        let mut trainer = BpeTrainer::default();
        pool.install(|| trainer.feed(["batch"; 2].into_iter(), |_| Ok(batch.clone())))
            .unwrap();
        let counts = serde_json::to_value(&trainer).unwrap()["words"].clone();
        assert_eq!(counts.as_object().unwrap().len(), 4097);
        assert!(counts.as_object().unwrap().values().all(|count| count == 2));
        assert_eq!(*expected.get_or_insert_with(|| counts.clone()), counts);
    }
}
