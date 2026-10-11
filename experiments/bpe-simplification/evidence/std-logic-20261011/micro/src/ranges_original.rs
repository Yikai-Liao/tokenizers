pub(super) struct CorpusPlan { starts:Vec<usize>, length:usize }
impl CorpusPlan {
pub(super) fn new(starts:Vec<usize>,length:usize)->Self {Self{starts,length}}
fn word_count(&self)->usize{self.starts.len()}
pub(super) fn initial_ranges(&self, slots: usize) -> Vec<std::ops::Range<usize>> {
        let mut ranges = Vec::new();
        let mut begin = 0;
        while begin < self.word_count() {
            let mut end = begin + 1;
            while end < self.word_count()
                && self.starts.get(end + 1).copied().unwrap_or(self.length) - self.starts[begin]
                    <= slots
            {
                end += 1;
            }
            ranges.push(begin..end);
            begin = end;
        }
        ranges
    }
}
