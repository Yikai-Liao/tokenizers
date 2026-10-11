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
            let end = if self.length - start <= slots {
                n
            } else {
                let fitted =
                    slice_group_by::exponential_search_by(&self.starts[begin + 1..], |&next| {
                        (next - start).cmp(&slots)
                    })
                    // An exact-budget boundary still fits; otherwise the insertion
                    // index counts all fitting boundaries. Always admit one word.
                    .map_or_else(|insertion| insertion, |exact| exact + 1);
                begin + fitted.max(1)
            };
            ranges.push(begin..end);
            begin = end;
        }
        ranges
    }
}
