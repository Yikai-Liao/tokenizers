//! Full-u64 delta streams, with independent restart blocks for parallel readers.
use bumpalo::Bump;
use std::ops::Range;
use std::sync::{Mutex, MutexGuard};
use tk_encode::Result;

const RESTART: usize = 128;
pub(super) struct Arena(Vec<Mutex<Bump>>);
impl Arena {
    pub(super) fn new(workers: usize) -> Self {
        Self((0..workers).map(|_| Mutex::new(Bump::new())).collect())
    }
    pub(super) fn lease(&self) -> Lease<'_> {
        Lease {
            cursor: self.0[rayon::current_thread_index().unwrap_or(0) % self.0.len()]
                .lock()
                .unwrap_or_else(|e| e.into_inner()),
        }
    }
}
pub(super) struct Lease<'arena> {
    cursor: MutexGuard<'arena, Bump>,
}
impl<'arena> Lease<'arena> {
    fn allocate(&self, length: usize) -> Result<&'arena mut [u8]> {
        let bytes = self
            .cursor
            .try_alloc_slice_fill_copy(length, 0u8)
            .map_err(|_| "BPE position arena allocation failed")?;
        // SAFETY: Each bump allocation is initialized and disjoint. The arena is never
        // reset, and its borrow outlives all slices, independently of this lease.
        Ok(unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr(), bytes.len()) })
    }
}
enum Bytes<'arena> {
    Inline([u8; 16], usize),
    Heap(Vec<u8>),
    Frozen(&'arena [u8]),
}
impl Default for Bytes<'_> {
    fn default() -> Self {
        Self::Inline([0; 16], 0)
    }
}
impl std::ops::Deref for Bytes<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Self::Inline(bytes, length) => &bytes[..*length],
            Self::Heap(bytes) => bytes,
            Self::Frozen(bytes) => bytes,
        }
    }
}
impl<'arena> Bytes<'arena> {
    fn extend(&mut self, extra: &[u8]) -> Result<()> {
        if extra.is_empty() {
            return Ok(());
        }
        let length = self
            .len()
            .checked_add(extra.len())
            .ok_or("BPE bytes exceed usize")?;
        if let Self::Inline(bytes, used) = self {
            if length <= bytes.len() {
                bytes[*used..length].copy_from_slice(extra);
                *used = length;
                return Ok(());
            }
        }
        if !matches!(self, Self::Heap(_)) {
            let mut bytes = Vec::with_capacity(length.max(32));
            bytes.extend_from_slice(self);
            *self = Self::Heap(bytes);
        }
        if let Self::Heap(bytes) = self {
            bytes.extend_from_slice(extra);
        }
        Ok(())
    }
    fn freeze(&mut self, lease: &Lease<'arena>) -> Result<()> {
        if let Self::Heap(bytes) = self {
            if bytes.len() <= 256 {
                let frozen = lease.allocate(bytes.len())?;
                frozen.copy_from_slice(bytes);
                *self = Self::Frozen(frozen);
            }
        }
        Ok(())
    }
}
#[derive(Default)]
pub(super) struct Positions<'arena> {
    bytes: Bytes<'arena>,
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
impl<'arena> Positions<'arena> {
    pub(super) fn from_sorted(values: &[u64], lease: &Lease<'arena>) -> Result<Self> {
        let mut result = Self {
            blocks: Vec::with_capacity(values.len().saturating_sub(1) / RESTART),
            ..Self::default()
        };
        for &position in values {
            result.push(position)?;
        }
        result.freeze(lease)?;
        Ok(result)
    }
    pub(super) fn freeze(&mut self, lease: &Lease<'arena>) -> Result<()> {
        self.bytes.freeze(lease)
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
            let mut bytes = [0u8; 10];
            let mut length = 0;
            while delta >= 128 {
                bytes[length] = (delta as u8 & 0x7f) | 0x80;
                length += 1;
                delta >>= 7;
            }
            bytes[length] = delta as u8;
            self.bytes.extend(&bytes[..length + 1])?;
        }
        self.last = position;
        self.count += 1;
        Ok(())
    }
    // Owned sorted pieces can retain their restart boundaries. Copy bytes and
    // directory offsets, rather than decoding and re-encoding every coordinate.
    pub(super) fn append(&mut self, other: Self) -> Result<()> {
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
        self.bytes.extend(&other.bytes)?;
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
            remaining: if range.is_empty() {
                0
            } else {
                self.blocks
                    .get(range.end - 1)
                    .map_or(self.count, |b| b.entry)
                    - self.block(range.start).entry
            },
            block_remaining: 0,
            next_block: range.start,
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
struct Cursor<'a, 'arena> {
    positions: &'a Positions<'arena>,
    bytes: &'a [u8],
    position: u64,
    remaining: usize,
    block_remaining: usize,
    next_block: usize,
}
impl Iterator for Cursor<'_, '_> {
    type Item = u64;
    fn next(&mut self) -> Option<u64> {
        if self.remaining == 0 {
            return None;
        }
        if self.block_remaining == 0 {
            let block = self.positions.block(self.next_block);
            self.block_remaining = self
                .positions
                .blocks
                .get(self.next_block)
                .map_or(self.positions.count, |b| b.entry)
                - block.entry;
            self.next_block += 1;
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
        self.remaining -= 1;
        self.block_remaining -= 1;
        Some(self.position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn released_leases_keep_disjoint_payloads_live_across_allocations() {
        use rayon::prelude::*;
        let arena = Arena::new(2);
        let mut saved = Vec::new();
        {
            let lease = arena.lease();
            for id in 0..256 {
                let values: Vec<_> = (0..128).map(|p| (id << 32) + p).collect();
                saved.push(Positions::from_sorted(&values, &lease).unwrap());
            }
        }
        saved.par_iter().enumerate().for_each(|(id, positions)| {
            let lease = arena.lease();
            let large: Vec<_> = (0..512).map(|p| (p * 1024) as u64).collect();
            let temporary = Positions::from_sorted(&large, &lease).unwrap();
            assert_eq!(temporary.iter().collect::<Vec<_>>(), large);
            assert!(
                positions
                    .iter()
                    .eq((0..128).map(|p| ((id as u64) << 32) + p))
            );
        });
    }
    #[test]
    fn fragmented_streams_preserve_order_seeks_and_push() {
        let arena = Arena::new(1);
        let lease = arena.lease();
        let mut positions = Positions::default();
        let mut values = Vec::new();
        for length in std::iter::repeat_n(1, 260).chain([0, 7, 127, 2, 128, 129, 17]) {
            let begin = values.last().copied().unwrap_or(1u64 << 32);
            let fragment: Vec<_> = (0..length).map(|i| begin + (i / 3) as u64).collect();
            positions
                .append(Positions::from_sorted(&fragment, &lease).unwrap())
                .unwrap();
            values.extend(fragment);
        }
        for value in [1u64 << 63, u64::MAX - 1, u64::MAX] {
            positions.push(value).unwrap();
            values.push(value);
        }
        positions
            .append(Positions::from_sorted(&[u64::MAX], &lease).unwrap())
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
                .append(Positions::from_sorted(&[0], &lease).unwrap())
                .is_err()
        );
        assert_eq!(positions.iter().collect::<Vec<_>>(), values);
    }
    #[test]
    fn full_u64_values_duplicates_and_restart_boundaries_roundtrip() {
        let arena = Arena::new(1);
        let lease = arena.lease();
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
            let positions = Positions::from_sorted(&sorted, &lease).unwrap();
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
        assert!(Positions::from_sorted(&[u64::MAX, 0], &lease).is_err());
        values.resize(129, u64::MAX);
        values[128] = 0;
        assert!(Positions::from_sorted(&values, &lease).is_err());
    }
}
