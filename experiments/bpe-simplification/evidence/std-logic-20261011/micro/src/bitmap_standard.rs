use crate::WordCountsView;
use fixedbitset::FixedBitSet;
use rayon::prelude::*;
fn observed_characters(word_counts: WordCountsView<'_>, workers: usize) -> FixedBitSet {
    let words: Vec<_> = word_counts.keys().collect();
    let chunk = words.len().div_ceil(workers).max(1);
    words
        .par_chunks(chunk)
        .map(|chunk| {
            let mut present = FixedBitSet::with_capacity(0x110000);
            for word in chunk {
                for character in word.chars() {
                    present.insert(character as usize);
                }
            }
            present
        })
        .reduce_with(|mut left, right| {
            left.union_with(&right);
            left
        })
        .unwrap_or_else(|| FixedBitSet::with_capacity(0x110000))
}
pub(super) fn observe(words:WordCountsView, workers:usize)->FixedBitSet { observed_characters(words, workers) }
