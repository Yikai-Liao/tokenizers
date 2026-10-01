//! Immutable flat word metadata shared by sparse merge and initial grouping.
use super::*;

// Directory narrows a spatial lookup to at most 256 word boundaries. This
// restores cheap weight queries when different rules visit sparse positions.
pub(super) struct WeightLookup {
    // Sorted weights need one range test and a search over weight changes.
    interval_one: Option<(usize, usize)>,
    bounds: Vec<u32>,
    // A set bit certifies that every position in this spatial bucket has
    // weight one. Mixed buckets retain the exact pivot lookup below.
    one_buckets: Vec<u64>,
}
impl WeightLookup {
    pub(super) fn new<O: Offset, const INLINE: usize>(
        block: &Block<O, INLINE>,
        slots: usize,
    ) -> Self {
        debug_assert_eq!(block.base, 0);
        if block.weight_intervals {
            let mut one_start = 0;
            let mut one_end = 0;
            if block.previous_weight == 1 {
                one_end = block.pivots.first().map_or(slots, |&p| p as usize);
            }
            for (i, (&pivot, &weight)) in block.pivots.iter().zip(&block.weights).enumerate() {
                if weight == 1 {
                    one_start = pivot as usize;
                    one_end = block.pivots.get(i + 1).map_or(slots, |&p| p as usize);
                    break;
                }
            }
            return Self {
                interval_one: Some((one_start, one_end.wrapping_sub(one_start))),
                bounds: Vec::new(),
                one_buckets: Vec::new(),
            };
        }
        let mut bounds = Vec::with_capacity(slots.div_ceil(256) + 1);
        let mut pivot = 0;
        for bucket in 0..=slots.div_ceil(256) {
            let before = (bucket * 256).min(slots);
            while pivot < block.pivots.len() && (block.pivots[pivot] as usize) < before {
                pivot += 1;
            }
            bounds.push(u32::try_from(pivot).expect("flat pivot count fits u32"));
        }
        let buckets = slots.div_ceil(256);
        let mut one_buckets = vec![u64::MAX; buckets.div_ceil(64)];
        if buckets % 64 != 0 {
            *one_buckets.last_mut().unwrap() = (1u64 << (buckets % 64)) - 1;
        }
        let mut begin = 0;
        let mut weight = block.previous_weight;
        for (&pivot, &next_weight) in block.pivots.iter().zip(&block.weights) {
            let end = (pivot as usize).min(slots);
            Self::exclude_non_one(&mut one_buckets, begin, end, weight);
            begin = end;
            weight = next_weight;
        }
        Self::exclude_non_one(&mut one_buckets, begin, slots, weight);
        Self {
            interval_one: None,
            bounds,
            one_buckets,
        }
    }
    fn exclude_non_one(bits: &mut [u64], begin: usize, end: usize, weight: u64) {
        if weight != 1 && begin < end {
            for bucket in begin / 256..=(end - 1) / 256 {
                bits[bucket / 64] &= !(1u64 << (bucket % 64));
            }
        }
    }
    pub(super) fn one_bucket_count(&self) -> usize {
        self.one_buckets
            .iter()
            .map(|bits| bits.count_ones() as usize)
            .sum()
    }
    pub(super) fn bucket_count(&self) -> usize {
        self.bounds.len().saturating_sub(1)
    }
    pub(super) fn one_bucket_bytes(&self) -> usize {
        self.one_buckets.capacity() * 8
    }
    pub(super) fn bytes(&self) -> usize {
        self.bounds.capacity() * 4 + self.one_bucket_bytes()
    }
    // Keep the common one-weight test inside each grouping/rewrite loop.
    // Without this hint the combined diagnostic paths exceeded the compiler's
    // inline budget and introduced a call at every initial posting position.
    #[inline(always)]
    pub(super) fn weight<O: Offset, const INLINE: usize>(
        &self,
        block: &Block<O, INLINE>,
        p: u32,
    ) -> u64 {
        if let Some((start, length)) = self.interval_one {
            if (p as usize).wrapping_sub(start) < length {
                return 1;
            }
            let pivot = block.pivots.partition_point(|&q| q <= p);
            return if pivot == 0 {
                block.previous_weight
            } else {
                block.weights[pivot - 1]
            };
        }
        let bucket = p as usize / 256;
        if self.one_buckets[bucket / 64] & (1u64 << (bucket % 64)) != 0 {
            return 1;
        }
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
    fn sorted_intervals_match_boundaries_zero_large_weights_and_full_u32_domain() {
        for (slots, previous, pivots, weights) in [
            (1000, 0, vec![1, 255, 512, 999], vec![u64::MAX, 7, 1, 0]),
            (1000, 0, vec![1, 255], vec![7, 0]),
            (1000, 1, vec![512], vec![0]),
            (u32::MAX as usize + 1, 0, vec![1], vec![1]),
            (u32::MAX as usize + 1, 1, vec![], vec![]),
        ] {
            let mut block = Block::<u32, 2>::new(0, previous);
            block.pivots = pivots;
            block.weights = weights;
            block.weight_intervals = true;
            let lookup = WeightLookup::new(&block, slots);
            assert_eq!(lookup.bytes(), 0);
            assert_eq!(lookup.one_bucket_bytes(), 0);
            for p in (0..slots.min(1000)).chain([slots - 1]) {
                assert_eq!(lookup.weight(&block, p as u32), block.weight(p, None));
            }
        }
    }

    #[test]
    fn one_bucket_certificate_matches_every_position_including_empty_intervals() {
        for (slots, previous, pivots, weights) in [
            (0, 1, vec![], vec![]),
            (1, 1, vec![], vec![]),
            (20000, 1, vec![], vec![]),
            (20000, 37, vec![], vec![]),
            (
                1000,
                1,
                vec![0, 0, 1, 255, 256, 256, 512, 999, 1000],
                vec![7, 1, 1, 0, u64::MAX, 1, 1, 1, 9],
            ),
            (
                20000,
                1,
                vec![256, 512, 16384, 16385, 19999],
                vec![2, 1, 0, 1, 1],
            ),
        ] {
            let mut block = Block::<u32, 2>::new(0, previous);
            block.pivots = pivots;
            block.weights = weights;
            let lookup = WeightLookup::new(&block, slots);
            let mut certified = 0;
            for bucket in 0..slots.div_ceil(256) {
                let all_one = (bucket * 256..((bucket + 1) * 256).min(slots))
                    .all(|p| block.weight(p, None) == 1);
                assert_eq!(
                    lookup.one_buckets[bucket / 64] & (1 << (bucket % 64)) != 0,
                    all_one
                );
                certified += usize::from(all_one);
            }
            assert_eq!(lookup.one_bucket_count(), certified);
            assert_eq!(lookup.bucket_count(), slots.div_ceil(256));
            for p in 0..slots {
                assert_eq!(lookup.weight(&block, p as u32), block.weight(p, None));
            }
        }
    }
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
