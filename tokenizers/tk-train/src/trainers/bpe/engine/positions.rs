//! Full-u64 delta streams, with independent restart blocks for parallel readers.
use std::ops::Range;
use tk_encode::Result;

const RESTART: usize = 128;
#[derive(Default)]
pub(super) struct Positions {
    bytes: Vec<u8>,
    blocks: Vec<Block>,
    count: usize,
    first: u64,
    last: u64,
}
#[derive(Clone, Copy)]
struct Block {
    first: u64,
    byte: usize,
}
impl Positions {
    pub(super) fn from_sorted(values: &[u64]) -> Result<Self> {
        let mut result = Self {
            bytes: Vec::with_capacity(values.len()),
            blocks: Vec::with_capacity(values.len().saturating_sub(1) / RESTART),
            ..Self::default()
        };
        for &position in values {
            result.push(position)?;
        }
        Ok(result)
    }
    pub(super) fn push(&mut self, position: u64) -> Result<()> {
        if self.count != 0 && position < self.last {
            return Err("BPE positions are not sorted".into());
        }
        if self.count == 0 {
            self.first = position;
        } else if self.count.is_multiple_of(RESTART) {
            self.blocks.push(Block {
                first: position,
                byte: self.bytes.len(),
            });
        } else {
            let mut delta = position - self.last;
            while delta >= 128 {
                self.bytes.push((delta as u8 & 0x7f) | 0x80);
                delta >>= 7;
            }
            self.bytes.push(delta as u8);
        }
        self.last = position;
        self.count += 1;
        Ok(())
    }
    pub(super) fn len(&self) -> usize {
        self.count
    }
    pub(super) fn is_empty(&self) -> bool {
        self.count == 0
    }
    pub(super) fn block_count(&self) -> usize {
        self.count.div_ceil(RESTART)
    }
    pub(super) fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        self.read_blocks(0..self.block_count())
    }
    pub(super) fn read_blocks(&self, range: Range<usize>) -> impl Iterator<Item = u64> + '_ {
        range.flat_map(|index| {
            let block = if index == 0 {
                Block {
                    first: self.first,
                    byte: 0,
                }
            } else {
                self.blocks[index - 1]
            };
            Cursor {
                bytes: &self.bytes[block.byte..],
                position: block.first,
                remaining: (self.count - index * RESTART).min(RESTART),
                first: true,
            }
        })
    }
    pub(super) fn lower_bound(&self, target: u64) -> usize {
        let block = self.blocks.partition_point(|b| b.first < target);
        if self.is_empty() {
            return 0;
        }
        block * RESTART
            + self
                .read_blocks(block..block + 1)
                .take_while(|&p| p < target)
                .count()
    }
    pub(super) fn from(&self, index: usize) -> impl Iterator<Item = u64> + '_ {
        self.read_blocks(index / RESTART..self.block_count())
            .skip(index % RESTART)
    }
}
struct Cursor<'a> {
    bytes: &'a [u8],
    position: u64,
    remaining: usize,
    first: bool,
}
impl Iterator for Cursor<'_> {
    type Item = u64;
    fn next(&mut self) -> Option<u64> {
        if self.remaining == 0 {
            return None;
        }
        if !self.first {
            let mut delta = 0u64;
            let mut shift = 0;
            loop {
                let byte = self.bytes[0];
                self.bytes = &self.bytes[1..];
                delta |= u64::from(byte & 0x7f) << shift;
                if byte < 128 {
                    break;
                }
                shift += 7;
            }
            // The private encoder checked monotonic full-width coordinates.
            // Each delta reconstructs an original value, including u64::MAX.
            self.position += delta;
        }
        self.first = false;
        self.remaining -= 1;
        Some(self.position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_u64_values_duplicates_and_restart_boundaries_roundtrip() {
        let mut values = vec![
            0,
            0,
            65535,
            65536,
            (1 << 32) - 1,
            1 << 32,
            (1 << 32) + 1,
            (1 << 63) - 1,
            1 << 63,
            u64::MAX - 1,
            u64::MAX,
        ];
        for length in [0, 1, 2, 127, 128, 129, 256, 257] {
            let repeated: Vec<_> = (0..length).map(|i| values[i % values.len()]).collect();
            let mut sorted = repeated;
            sorted.sort_unstable();
            let positions = Positions::from_sorted(&sorted).unwrap();
            assert_eq!(positions.iter().collect::<Vec<_>>(), sorted);
            for index in 0..=sorted.len() {
                assert_eq!(positions.from(index).collect::<Vec<_>>(), sorted[index..]);
            }
            for &target in &values {
                assert_eq!(
                    positions.lower_bound(target),
                    sorted.partition_point(|&p| p < target)
                );
            }
        }
        assert!(Positions::from_sorted(&[u64::MAX, 0]).is_err());
        values.resize(129, u64::MAX);
        values[128] = 0;
        assert!(Positions::from_sorted(&values).is_err());
    }
}
