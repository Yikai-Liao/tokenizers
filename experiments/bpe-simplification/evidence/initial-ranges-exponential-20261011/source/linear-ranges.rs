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
            let end = (begin + 2..=n)
                .take_while(|&end| {
                    self.starts.get(end).copied().unwrap_or(self.length) - start <= slots
                })
                .last()
                .unwrap_or(begin + 1);
            ranges.push(begin..end);
            begin = end;
        }
        ranges
    }
}
