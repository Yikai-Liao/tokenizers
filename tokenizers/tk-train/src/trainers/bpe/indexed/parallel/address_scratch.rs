//! Temporary global coordinates retain u32 storage when a buffer shares high bits.
//! These buffers are not the resident posting representation.
#[derive(Default)]
pub(super) struct HighParts {
    base: u32,
    values: Vec<u32>,
}
impl HighParts {
    pub(super) fn push(&mut self, index: usize, position: usize) -> u32 {
        let high = (position >> 32) as u32;
        if index == 0 {
            self.base = high;
        }
        if high != self.base || !self.values.is_empty() {
            if self.values.is_empty() {
                self.values.resize(index, self.base);
            }
            self.values.push(high);
        }
        position as u32
    }
    #[inline]
    pub(super) fn address(&self, index: usize, low: u32) -> usize {
        ((self.values.get(index).copied().unwrap_or(self.base) as usize) << 32) | low as usize
    }
    pub(super) fn bytes(&self) -> usize {
        self.values.capacity() * 4
    }
}
#[derive(Default)]
pub(super) struct Positions {
    low: Vec<u32>,
    high: HighParts,
}
impl Positions {
    pub(super) fn push(&mut self, position: usize) {
        self.low.push(self.high.push(self.low.len(), position));
    }
    pub(super) fn bytes(&self) -> usize {
        self.low.capacity() * 4 + self.high.bytes()
    }
    pub(super) fn for_each(&self, mut apply: impl FnMut(usize)) {
        if self.high.values.is_empty() {
            let base = (self.high.base as usize) << 32;
            for &low in &self.low {
                apply(base | low as usize);
            }
        } else {
            for (&low, &high) in self.low.iter().zip(&self.high.values) {
                apply(((high as usize) << 32) | low as usize);
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scratch_promotes_without_losing_earlier_or_returning_high_bits() {
        for values in [
            vec![0, 3, 17],
            vec![1usize << 40, (1usize << 40) + 3, usize::MAX, 7],
        ] {
            let mut p = Positions::default();
            let mut h = HighParts::default();
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
