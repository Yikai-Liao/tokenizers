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
    entry: usize,
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
        } else if self.count - self.blocks.last().map_or(0, |b| b.entry) == RESTART {
            self.blocks.push(Block {
                first: position,
                byte: self.bytes.len(),
                entry: self.count,
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
    // Owned sorted pieces can retain their restart boundaries. Copy bytes and
    // directory offsets, rather than decoding and re-encoding every coordinate.
    pub(super) fn append(&mut self, mut other: Self) -> Result<()> {
        if other.is_empty() {
            return Ok(());
        }
        if self.is_empty() {
            *self = other;
            return Ok(());
        }
        if other.first < self.last {
            return Err("BPE positions are not sorted".into());
        }
        let count = self
            .count
            .checked_add(other.count)
            .ok_or("BPE position count exceeds usize")?;
        let byte = self.bytes.len();
        let entry = self.count;
        self.blocks.push(Block {
            first: other.first,
            byte,
            entry,
        });
        self.blocks.extend(other.blocks.into_iter().map(|b| Block {
            first: b.first,
            byte: byte + b.byte,
            entry: entry + b.entry,
        }));
        self.bytes.append(&mut other.bytes);
        self.count = count;
        self.last = other.last;
        Ok(())
    }
    fn block(&self, index: usize) -> Block {
        if index == 0 {
            Block {
                first: self.first,
                byte: 0,
                entry: 0,
            }
        } else {
            self.blocks[index - 1]
        }
    }
    pub(super) fn len(&self) -> usize {
        self.count
    }
    pub(super) fn is_empty(&self) -> bool {
        self.count == 0
    }
    pub(super) fn block_count(&self) -> usize {
        self.blocks.len() + usize::from(!self.is_empty())
    }
    pub(super) fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        self.read_blocks(0..self.block_count())
    }
    pub(super) fn read_blocks(&self, range: Range<usize>) -> impl Iterator<Item = u64> + '_ {
        Cursor {
            positions: self,
            bytes: &[],
            position: 0,
            block_remaining: 0,
            blocks: range,
        }
    }
    pub(super) fn lower_bound(&self, target: u64) -> usize {
        let block = self.blocks.partition_point(|b| b.first < target);
        if self.is_empty() {
            return 0;
        }
        self.block(block).entry
            + self
                .read_blocks(block..block + 1)
                .take_while(|&p| p < target)
                .count()
    }
    pub(super) fn from(&self, index: usize) -> impl Iterator<Item = u64> + '_ {
        let block = self.blocks.partition_point(|b| b.entry <= index);
        self.read_blocks(block..self.block_count())
            .skip(index - self.block(block).entry)
    }
}
struct Cursor<'a> {
    positions: &'a Positions,
    bytes: &'a [u8],
    position: u64,
    block_remaining: usize,
    blocks: Range<usize>,
}
impl Iterator for Cursor<'_> {
    type Item = u64;
    fn next(&mut self) -> Option<u64> {
        if self.block_remaining == 0 {
            let index = self.blocks.next()?;
            let block = self.positions.block(index);
            self.block_remaining = self
                .positions
                .blocks
                .get(index)
                .map_or(self.positions.count, |b| b.entry)
                - block.entry;
            self.position = block.first;
            self.bytes = &self.positions.bytes[block.byte..];
        } else {
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
        self.block_remaining -= 1;
        Some(self.position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fragmented_streams_preserve_order_seeks_and_push() {
        let mut positions = Positions::default();
        let mut values = Vec::new();
        for length in std::iter::repeat_n(1, 260).chain([0, 7, 127, 2, 128, 129, 17]) {
            let begin = values.last().copied().unwrap_or(1u64 << 32);
            let fragment: Vec<_> = (0..length).map(|i| begin + (i / 3) as u64).collect();
            positions
                .append(Positions::from_sorted(&fragment).unwrap())
                .unwrap();
            values.extend(fragment);
        }
        for value in [1u64 << 63, u64::MAX - 1, u64::MAX] {
            positions.push(value).unwrap();
            values.push(value);
        }
        positions
            .append(Positions::from_sorted(&[u64::MAX]).unwrap())
            .unwrap();
        values.push(u64::MAX);
        assert_eq!(positions.iter().collect::<Vec<_>>(), values);
        for index in 0..=values.len() {
            assert_eq!(positions.from(index).collect::<Vec<_>>(), values[index..]);
        }
        for target in values
            .iter()
            .copied()
            .chain([0, (1 << 32) - 1, (1 << 63) + 1])
        {
            assert_eq!(
                positions.lower_bound(target),
                values.partition_point(|&p| p < target)
            );
        }
        let blocks: Vec<_> = (0..positions.block_count())
            .flat_map(|i| positions.read_blocks(i..i + 1))
            .collect();
        assert_eq!(blocks, values);
        assert!(
            positions
                .append(Positions::from_sorted(&[0]).unwrap())
                .is_err()
        );
        assert_eq!(positions.iter().collect::<Vec<_>>(), values);
    }
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
