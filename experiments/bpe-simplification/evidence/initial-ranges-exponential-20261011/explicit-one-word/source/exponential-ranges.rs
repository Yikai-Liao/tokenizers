pub struct CorpusPlan<'a> {
    pub starts: &'a [usize],
    pub length: usize,
    pub words: usize,
}
impl CorpusPlan<'_> {
    fn word_count(&self) -> usize { self.words }
    pub(super) fn initial_ranges(&self, slots: usize) -> Vec<std::ops::Range<usize>> {
        let mut ranges = Vec::new();
        let mut begin = 0;
        let n = self.word_count();
        while begin < n {
            let start = self.starts[begin];
            let one_word_end = begin + 1;
            let end = if one_word_end == n || self.length - start <= slots {
                n
            } else {
                // The first word is admitted even when oversized. Each later
                // boundary closes one additional word; an exact-budget end fits.
                let additional_words = match slice_group_by::exponential_search_by(
                    &self.starts[one_word_end + 1..],
                    |&next| (next - start).cmp(&slots),
                ) {
                    Ok(exact) => exact + 1,
                    Err(insertion) => insertion,
                };
                one_word_end + additional_words
            };
            ranges.push(begin..end);
            begin = end;
        }
        ranges
    }
}
