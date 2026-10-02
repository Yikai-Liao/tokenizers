//! Immutable flat word metadata shared by sparse merge and initial grouping.
use super::*;

// Link the next actual boundary, which may be many address blocks away.
// Empty blocks inherit weight and do not terminate a cached interval.
pub(super) fn link_intervals<O: Offset, const INLINE: usize>(blocks: &mut [Block<O, INLINE>]) {
    let mut next = usize::MAX;
    for block in blocks.iter_mut().rev() {
        block.next_weight_change = next;
        if let Some(&local) = block.pivots.first() {
            next = block.base + local as usize;
        }
    }
}

#[derive(Default)]
pub(super) struct Cursor {
    end: usize,
    weight: u64,
}
impl Cursor {
    // Callers visit sorted positions and reset the cursor for each posting list.
    #[inline]
    pub(super) fn weight<O: Offset, const INLINE: usize>(
        &mut self,
        position: usize,
        blocks: &[Block<O, INLINE>],
        bits: u8,
    ) -> u64 {
        if position < self.end {
            return self.weight;
        }
        let block = &blocks[position >> bits];
        let local = (position - block.base) as u32;
        let index = block.pivots.partition_point(|&p| p <= local);
        self.weight = if index == 0 {
            block.previous_weight
        } else {
            block.weights[index - 1]
        };
        self.end = block.pivots.get(index).map_or_else(
            || {
                if block.next_weight_change == 0 {
                    // Standalone/unlinked test blocks remain correct at their boundary.
                    block.base.saturating_add(1usize << bits)
                } else {
                    block.next_weight_change
                }
            },
            |&p| block.base + p as usize,
        );
        self.weight
    }
}

/// The shared index carries global fast ranges separately from block-local
/// fallback metadata, so crossing a physical block does not defeat the range.
pub(super) struct WeightLookups {
    blocks: Vec<WeightLookup>,
    one: Option<(usize, usize)>,
    ordered_intervals: bool,
}
impl WeightLookups {
    pub(super) fn new<O: Offset, const INLINE: usize>(
        blocks: &[Block<O, INLINE>],
        slots: usize,
        bits: u8,
    ) -> Self {
        let local: Vec<_> = blocks
            .iter()
            .map(|b| {
                WeightLookup::from_parts(
                    &b.pivots,
                    &b.weights,
                    b.previous_weight,
                    b.weight_intervals,
                    (slots - b.base).min(1usize << bits),
                )
            })
            .collect();
        let ordered_intervals = blocks.iter().all(|b| b.weight_intervals);
        let mut one: Option<(usize, usize)> = None;
        if ordered_intervals {
            for (b, l) in blocks.iter().zip(&local) {
                if let Some((start, len)) = l.interval_one.filter(|&(_, len)| len != 0) {
                    let start = b.base + start;
                    match one {
                        None => one = Some((start, len)),
                        Some((begin, size)) if begin + size == start => {
                            one = Some((begin, size + len))
                        }
                        // Never join disconnected ranges, even if a caller's
                        // declared ordering fails to yield a global single run.
                        _ => {
                            one = None;
                            break;
                        }
                    }
                }
            }
        }
        Self {
            blocks: local,
            one,
            ordered_intervals,
        }
    }
    pub(super) fn iter(&self) -> impl Iterator<Item = &WeightLookup> {
        self.blocks.iter()
    }
    #[inline]
    pub(super) fn weight<O: Offset, const INLINE: usize>(
        &self,
        p: usize,
        blocks: &[Block<O, INLINE>],
        bits: u8,
        cursor: &mut Cursor,
    ) -> u64 {
        if self
            .one
            .is_some_and(|(start, len)| p.wrapping_sub(start) < len)
        {
            return 1;
        }
        if self.ordered_intervals {
            return cursor.weight(p, blocks, bits);
        }
        let id = p >> bits;
        self.blocks[id].weight(&blocks[id], (p - blocks[id].base) as u32)
    }
}

pub(super) fn sum_records<O: Offset, const INLINE: usize>(
    records: &[u64],
    base: usize,
    blocks: &[Block<O, INLINE>],
    bits: u8,
    uniform: Option<u64>,
    lookups: Option<&WeightLookups>,
) -> Result<u64> {
    if let Some(weight) = uniform {
        return weight
            .checked_mul(records.len() as u64)
            .ok_or_else(|| "initial frequency exceeds u64".into());
    }
    let mut cursor = Cursor::default();
    Ok(records
        .iter()
        .map(|&record| {
            let p = base + record as u32 as usize;
            if let Some(lookups) = lookups {
                lookups.weight(p, blocks, bits, &mut cursor)
            } else {
                cursor.weight(p, blocks, bits)
            }
        })
        .sum())
}

// Directory narrows a spatial lookup to at most 256 word boundaries. This
// restores cheap weight queries when different rules visit sparse positions.
pub(in super::super) struct WeightLookup {
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
        Self::from_parts(
            &block.pivots,
            &block.weights,
            block.previous_weight,
            block.weight_intervals,
            slots,
        )
    }
    pub(in super::super) fn from_parts(
        pivots: &[u32],
        weights: &[u64],
        previous_weight: u64,
        weight_intervals: bool,
        slots: usize,
    ) -> Self {
        if weight_intervals {
            let mut one_start = 0;
            let mut one_end = 0;
            if previous_weight == 1 {
                one_end = pivots.first().map_or(slots, |&p| p as usize);
            }
            for (i, (&pivot, &weight)) in pivots.iter().zip(weights).enumerate() {
                if weight == 1 {
                    one_start = pivot as usize;
                    one_end = pivots.get(i + 1).map_or(slots, |&p| p as usize);
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
            while pivot < pivots.len() && (pivots[pivot] as usize) < before {
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
        let mut weight = previous_weight;
        for (&pivot, &next_weight) in pivots.iter().zip(weights) {
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
    pub(in super::super) fn one_bucket_count(&self) -> usize {
        self.one_buckets
            .iter()
            .map(|bits| bits.count_ones() as usize)
            .sum()
    }
    pub(in super::super) fn bucket_count(&self) -> usize {
        self.bounds.len().saturating_sub(1)
    }
    pub(in super::super) fn one_bucket_bytes(&self) -> usize {
        self.one_buckets.capacity() * 8
    }
    pub(in super::super) fn bytes(&self) -> usize {
        self.bounds.capacity() * 4 + self.one_bucket_bytes()
    }
    // Keep the common one-weight test inside each grouping/rewrite loop.
    // In the measured release build the combined diagnostic paths otherwise
    // left an out-of-line call at every initial posting position.
    #[inline(always)]
    pub(super) fn weight<O: Offset, const INLINE: usize>(
        &self,
        block: &Block<O, INLINE>,
        p: u32,
    ) -> u64 {
        self.weight_parts(&block.pivots, &block.weights, block.previous_weight, p)
    }
    #[inline(always)]
    pub(in super::super) fn weight_parts(
        &self,
        pivots: &[u32],
        weights: &[u64],
        previous_weight: u64,
        p: u32,
    ) -> u64 {
        if let Some((start, length)) = self.interval_one {
            if (p as usize).wrapping_sub(start) < length {
                return 1;
            }
            let pivot = pivots.partition_point(|&q| q <= p);
            return if pivot == 0 {
                previous_weight
            } else {
                weights[pivot - 1]
            };
        }
        let bucket = p as usize / 256;
        if self.one_buckets[bucket / 64] & (1u64 << (bucket % 64)) != 0 {
            return 1;
        }
        let start = self.bounds[bucket] as usize;
        let end = self.bounds[bucket + 1] as usize;
        let pivot = start + pivots[start..end].partition_point(|&q| q <= p);
        if pivot == 0 {
            previous_weight
        } else {
            weights[pivot - 1]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn default_index_joins_weight_one_across_physical_blocks() {
        let mut blocks: Vec<Block<u32, 2>> = (0..8)
            .map(|i| {
                Block::new(
                    i * 16,
                    if i == 0 {
                        7
                    } else if i <= 6 {
                        1
                    } else {
                        9
                    },
                )
            })
            .collect();
        blocks[0].pivots = vec![5];
        blocks[0].weights = vec![1];
        blocks[6].pivots = vec![3];
        blocks[6].weights = vec![9];
        for b in &mut blocks {
            b.weight_intervals = true;
        }
        link_intervals(&mut blocks);
        let lookup = WeightLookups::new(&blocks, 128, 4);
        assert_eq!(lookup.one, Some((5, 94)));
        let mut cursor = Cursor::default();
        for p in 0..128 {
            assert_eq!(
                lookup.weight(p, &blocks, 4, &mut cursor),
                blocks[p >> 4].weight(p, None)
            );
        }
        let records: Vec<_> = (0..128u64).collect();
        assert_eq!(
            sum_records(&records, 0, &blocks, 4, None, Some(&lookup)).unwrap(),
            (0..128)
                .map(|p| blocks[p >> 4].weight(p, None))
                .sum::<u64>()
        );
        for b in &mut blocks {
            b.weight_intervals = false;
        }
        let lookup = WeightLookups::new(&blocks, 128, 4);
        assert_eq!(lookup.one, None);
        let mut cursor = Cursor::default();
        for p in (0..128).rev() {
            assert_eq!(
                lookup.weight(p, &blocks, 4, &mut cursor),
                blocks[p >> 4].weight(p, None)
            );
        }
    }
    #[test]
    fn cursor_keeps_weights_across_empty_blocks_and_sparse_jumps() {
        let mut blocks: Vec<Block<u32, 2>> = (0..8)
            .map(|i| Block::new(i * 16, if i < 3 { 7 } else { 11 }))
            .collect();
        blocks[0].pivots = vec![3, 7];
        blocks[0].weights = vec![0, 7];
        blocks[3].pivots = vec![2, 2, 9];
        blocks[3].weights = vec![9, 11, 11];
        // Carry the final weight of each preceding block.
        blocks[1].previous_weight = 7;
        blocks[2].previous_weight = 7;
        blocks[3].previous_weight = 7;
        link_intervals(&mut blocks);
        for positions in [
            (0..128).collect::<Vec<_>>(),
            vec![0, 3, 6, 7, 31, 49, 50, 57, 127],
        ] {
            let mut cursor = Cursor::default();
            for p in positions {
                assert_eq!(cursor.weight(p, &blocks, 4), blocks[p >> 4].weight(p, None));
            }
        }
        let records: Vec<_> = (0..128u64).map(|p| (99 << 32) | p).collect();
        assert_eq!(
            sum_records(&records, 0, &blocks, 4, None, None).unwrap(),
            (0..128)
                .map(|p| blocks[p >> 4].weight(p, None))
                .sum::<u64>()
        );
    }
    #[test]
    fn record_sum_checks_uniform_overflow() {
        let blocks = [Block::<u32, 2>::new(0, 1)];
        assert!(sum_records(&[0, 1], 0, &blocks, 32, Some(u64::MAX), None).is_err());
    }
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
