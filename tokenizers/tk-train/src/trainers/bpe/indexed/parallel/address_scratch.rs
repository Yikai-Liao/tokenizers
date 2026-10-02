//! Temporary global coordinates retain u32 storage when a buffer shares high bits.
//! These buffers are not the resident posting representation.
//! The compile-time override scales high-bit transitions for small-memory
//! experiments only. Production uses 32 low bits and covers all u64 addresses.
const ADDRESS_BITS: u32 = match option_env!("TK_POSTING_SCRATCH_BITS") {
    Some(s) if s.as_bytes().len() == 2 && s.as_bytes()[0] == b'1' && s.as_bytes()[1] == b'6' => 16,
    _ => 32,
};
#[derive(Default)]
pub(super) struct HighParts<const BITS: u32 = ADDRESS_BITS> {
    base: u32,
    values: Vec<u32>,
}
impl<const BITS: u32> HighParts<BITS> {
    pub(super) fn push(&mut self, index: usize, position: usize) -> u32 {
        debug_assert!(position >> BITS <= u32::MAX as usize);
        let high = (position >> BITS) as u32;
        if index == 0 {
            self.base = high;
        }
        if high != self.base || !self.values.is_empty() {
            if self.values.is_empty() {
                self.values.resize(index, self.base);
            }
            self.values.push(high);
        }
        (position & ((1usize << BITS) - 1)) as u32
    }
    #[inline]
    pub(super) fn address(&self, index: usize, low: u32) -> usize {
        ((self.values.get(index).copied().unwrap_or(self.base) as usize) << BITS) | low as usize
    }
    pub(super) fn bytes(&self) -> usize {
        self.values.capacity() * 4
    }
}
#[derive(Default)]
pub(super) struct Positions<const BITS: u32 = ADDRESS_BITS> {
    low: Vec<u32>,
    high: HighParts<BITS>,
}
impl<const BITS: u32> Positions<BITS> {
    pub(super) fn push(&mut self, position: usize) {
        self.low.push(self.high.push(self.low.len(), position));
    }
    pub(super) fn bytes(&self) -> usize {
        self.low.capacity() * 4 + self.high.bytes()
    }
    pub(super) fn for_each(&self, mut apply: impl FnMut(usize)) {
        if self.high.values.is_empty() {
            let base = (self.high.base as usize) << BITS;
            for &low in &self.low {
                apply(base | low as usize);
            }
        } else {
            for (&low, &high) in self.low.iter().zip(&self.high.values) {
                apply(((high as usize) << BITS) | low as usize);
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scaled_scratch_really_materializes_upper_plane() {
        let values = [1, 65535, 65536, 131075, (1usize << 40) + 5];
        let mut p = Positions::<16>::default();
        for value in values {
            p.push(value);
        }
        assert_eq!(p.high.values, [0, 0, 1, 2, 1 << 24]);
        assert_eq!(p.low, [1, 65535, 0, 3, 5]);
        let mut result = Vec::new();
        p.for_each(|v| result.push(v));
        assert_eq!(result, values);
    }
    #[test]
    fn scratch_promotes_without_losing_earlier_or_returning_high_bits() {
        for values in [
            vec![0, 3, 17],
            vec![1usize << 40, (1usize << 40) + 3, usize::MAX, 7],
        ] {
            let mut p = Positions::<32>::default();
            let mut h = HighParts::<32>::default();
            let mut lows = Vec::new();
            for &v in &values {
                p.push(v);
                lows.push(h.push(lows.len(), v));
            }
            let mut got = Vec::new();
            p.for_each(|v| got.push(v));
            assert_eq!(got, values);
            assert_eq!(
                lows.iter()
                    .enumerate()
                    .map(|(i, &v)| h.address(i, v))
                    .collect::<Vec<_>>(),
                values
            );
        }
    }
}
