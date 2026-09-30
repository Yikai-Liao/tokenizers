//! Immutable flat word metadata shared by sparse merge and initial grouping.
use super::*;

// Directory narrows a spatial lookup to at most 256 word boundaries. This
// restores cheap weight queries when different rules visit sparse positions.
pub(super) struct WeightLookup {
    bounds: Vec<u32>,
}
impl WeightLookup {
    pub(super) fn new<O: Offset, const INLINE: usize>(
        block: &Block<O, INLINE>,
        slots: usize,
    ) -> Self {
        debug_assert_eq!(block.base, 0);
        let mut bounds = Vec::with_capacity(slots.div_ceil(256) + 1);
        let mut pivot = 0;
        for bucket in 0..=slots.div_ceil(256) {
            let before = (bucket * 256).min(slots);
            while pivot < block.pivots.len() && (block.pivots[pivot] as usize) < before {
                pivot += 1;
            }
            bounds.push(u32::try_from(pivot).expect("flat pivot count fits u32"));
        }
        Self { bounds }
    }
    pub(super) fn bytes(&self) -> usize {
        self.bounds.capacity() * 4
    }
    pub(super) fn weight<O: Offset, const INLINE: usize>(
        &self,
        block: &Block<O, INLINE>,
        p: u32,
    ) -> u64 {
        let bucket = p as usize / 256;
        let start = self.bounds[bucket] as usize;
        let end = self.bounds[bucket + 1] as usize;
        let pivot = start + block.pivots[start..end].partition_point(|&q| q <= p);
        if pivot == 0 {
            block.previous_weight
        } else {
            block.weights[pivot - 1]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bucket_weights_match_full_search_before_on_and_between_pivots() {
        let mut block = Block::<u32, 2>::new(0, 17);
        block.pivots = vec![1, 2, 255, 256, 257, 511, 1023, 2048, 8191];
        block.weights = vec![2, 3, 5, 7, 11, 13, 19, 23, 29];
        let lookup = WeightLookup::new(&block, 10000);
        for p in 0..10000 {
            assert_eq!(lookup.weight(&block, p as u32), block.weight(p, None));
        }
        let empty = Block::<u32, 2>::new(0, 37);
        let lookup = WeightLookup::new(&empty, 10000);
        for p in [0, 255, 256, 9999] {
            assert_eq!(lookup.weight(&empty, p), 37);
        }
    }
}
