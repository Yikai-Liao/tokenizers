//! Proof-oriented engine tests; fixtures and oracle comparison helpers are shared.
use super::*;
use tk_encode::models::bpe::Pair;
fn counts(items: &[(&str, u64)]) -> AHashMap<CompactString, u64> {
    items
        .iter()
        .map(|&(word, count)| (word.into(), count))
        .collect()
}
fn check(trainer: &BpeTrainer, words: &AHashMap<CompactString, u64>) {
    check_with_workers(trainer, words, &[1, 4]);
}
fn check_with_workers(
    trainer: &BpeTrainer,
    words: &AHashMap<CompactString, u64>,
    workers: &[usize],
) {
    check_with_workers_and_cache(
        trainer,
        words,
        workers,
        &[corpus::InitialCachePolicy::default()],
    );
}
pub(super) fn check_cache_modes(trainer: &BpeTrainer, words: &AHashMap<CompactString, u64>) {
    check_with_workers_and_cache(
        trainer,
        words,
        &[1, 4],
        &[
            corpus::InitialCachePolicy::default(),
            corpus::InitialCachePolicy::U32,
            corpus::InitialCachePolicy::SCANNER,
        ],
    );
}
fn check_with_workers_and_cache(
    trainer: &BpeTrainer,
    words: &AHashMap<CompactString, u64>,
    workers: &[usize],
    cache_policies: &[corpus::InitialCachePolicy],
) {
    let mut expected_trace = Vec::new();
    let expected = trainer
        .do_train_observed(words, |pair, count, id| {
            expected_trace.push((pair, count, id))
        })
        .unwrap();
    for &workers in workers {
        for &cache_policy in cache_policies {
            let mut trace = Vec::<(Pair, u64, u32)>::new();
            let actual = train_with_merge_options_and_cache(
                trainer,
                WordCountsView::from_map(words),
                workers,
                merge::MergeOptions::default(),
                cache_policy,
                Some(&mut |pair, count, id| trace.push((pair, count, id))),
                None,
            )
            .unwrap();
            assert_eq!(
                trace,
                expected_trace,
                "workers={workers}, prefix={:?}, suffix={:?}, limit={:?}",
                trainer.continuing_subword_prefix,
                trainer.end_of_word_suffix,
                trainer.max_token_length
            );
            assert_eq!(actual, expected, "workers={workers}");
        }
    }
}

fn next(rng: &mut u64) -> u64 {
    *rng = rng
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *rng >> 32
}

mod identity_reuse;
mod initial_corpus;
mod producer_fast_path;
mod public_contract;
mod routing_and_publication;
mod semantic_parity;
