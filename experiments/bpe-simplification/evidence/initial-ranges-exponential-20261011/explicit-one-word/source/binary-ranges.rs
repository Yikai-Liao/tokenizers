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
            // starts omits the final separator boundary. Admit the tail directly
            // when it fits; otherwise search resident boundaries without addition.
            let end = if self.length - start <= slots {
                n
            } else {
                let fitted =
                    self.starts[begin + 1..].partition_point(|&next| next - start <= slots);
                begin + fitted.max(1)
            };
            ranges.push(begin..end);
            begin = end;
        }
        ranges
    }
}
