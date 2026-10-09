//! Proof-oriented engine tests; fixtures and oracle comparison helpers are shared.
use super::*;
use ahash::AHashMap;
use compact_str::CompactString;
use tk_encode::models::bpe::Pair;
fn counts(items: &[(&str, u64)]) -> AHashMap<CompactString, u64> {
    items
        .iter()
        .map(|&(word, count)| (word.into(), count))
        .collect()
}
pub(super) fn check(trainer: &BpeTrainer, words: &AHashMap<CompactString, u64>) {
    check_with_workers(trainer, words, &[1, 4, 8]);
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
            WordCountsView::from_map(words),
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

mod generated;
mod identity_reuse;
mod public_contract;
mod semantic_parity;

pub(super) static EXPECTED_WORKERS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
pub(super) static OBSERVED_TASKS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
pub(super) fn observe_worker() {
    use std::sync::atomic::Ordering;
    let workers = EXPECTED_WORKERS.load(Ordering::Relaxed);
    if workers != 0 {
        assert_eq!(rayon::current_num_threads(), workers);
        OBSERVED_TASKS.fetch_add(1, Ordering::Relaxed);
    }
}
