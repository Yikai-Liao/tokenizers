//! Appendable compressed leaves and immutable, nonrecursive roots.
//! A leaf stores an eight-byte last-value cache, a geometric restart directory,
//! then absolute u64 restart seeds and LEB128 gaps. Directory entries
//! hold offsets relative to the data area. Two positions share the first and
//! last-value caches; longer leaves encode their gaps in the data area.
use smallvec::SmallVec;
use std::num::NonZeroU64;
use tk_encode::Result;

#[path = "positions/codec.rs"]
mod codec;

use codec::{Cursor, RESTART};
const OFFSET_BYTES: usize = std::mem::size_of::<usize>();

pub(super) struct Positions(Repr);

enum Repr {
    Leaf(Leaf),
    Root { leaves: Box<[Leaf]>, len: u64 },
}

struct Leaf {
    bytes: SmallVec<[u8; 16]>,
    // Encoding len + 1 permits an empty leaf and supplies a layout niche.
    len: NonZeroU64,
}

impl Default for Positions {
    fn default() -> Self {
        Self(Repr::Leaf(Leaf {
            bytes: SmallVec::new(),
            len: NonZeroU64::MIN,
        }))
    }
}

impl Positions {
    pub(super) fn len(&self) -> usize {
        match &self.0 {
            Repr::Leaf(leaf) => leaf.len(),
            Repr::Root { len, .. } => *len as usize,
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append a nondecreasing full-u64 coordinate. Roots are immutable.
    #[inline(always)]
    pub(super) fn push(&mut self, position: u64) -> Result<()> {
        match &mut self.0 {
            Repr::Leaf(leaf) => leaf.push(position),
            Repr::Root { .. } => Err("BPE roots cannot be appended to".into()),
        }
    }

    /// Consume ordered leaves; the heap payloads move without re-encoding.
    /// Validate every input before filtering empties, including a sole root.
    pub(super) fn concat(fragments: impl IntoIterator<Item = Self>) -> Result<Self> {
        let mut leaves = SmallVec::<[Leaf; 1]>::new();
        let mut len = 0usize;
        let mut previous = None;
        for fragment in fragments {
            let Repr::Leaf(leaf) = fragment.0 else {
                return Err("BPE concat requires leaves".into());
            };
            if leaf.len() == 0 {
                continue;
            }
            if previous.is_some_and(|last| last > leaf.first()) {
                return Err("BPE positions are not sorted".into());
            }
            len = len
                .checked_add(leaf.len())
                .ok_or("BPE fragment count exceeds usize")?;
            previous = Some(leaf.last());
            leaves.push(leaf);
        }
        match leaves.len() {
            0 => Ok(Self::default()),
            1 => Ok(Self(Repr::Leaf(leaves.pop().expect("one leaf")))),
            _ => Ok(Self(Repr::Root {
                leaves: leaves.into_vec().into_boxed_slice(),
                len: len as u64,
            })),
        }
    }

    fn leaves(&self) -> &[Leaf] {
        match &self.0 {
            Repr::Leaf(leaf) => std::slice::from_ref(leaf),
            Repr::Root { leaves, .. } => leaves,
        }
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        read(self.leaves(), 0)
    }

    /// Cover the ordered sequence with nonoverlapping, independently consumable
    /// iterators. Their lengths may differ from the requested target.
    pub(super) fn chunks(
        &self,
        target_len: usize,
    ) -> impl Iterator<Item = impl Iterator<Item = u64> + Send + '_> + '_ {
        // Split large leaves at restart boundaries and group small leaves;
        // every chunk retains its already located starting leaf.
        let target = target_len.max(1);
        let mut remaining = self.leaves();
        let mut start = 0;
        std::iter::from_fn(move || {
            let (first, begin) = (remaining, start);
            let mut count = 0;
            while let Some(leaf) = remaining.first() {
                let take = target
                    .saturating_sub(count)
                    .max(1)
                    .div_ceil(RESTART)
                    .saturating_mul(RESTART)
                    .min(leaf.len() - start);
                start += take;
                count += take;
                if start == leaf.len() {
                    remaining = &remaining[1..];
                    start = 0;
                }
                if count >= target {
                    break;
                }
            }
            let end = first.len() - remaining.len() + usize::from(start != 0);
            (count != 0).then(|| read(&first[..end], begin).take(count))
        })
    }

    /// Locate the first qualifying leaf using its cached tail, then its restart
    /// directory. Equal values at a preceding leaf/block tail remain visible.
    pub(super) fn iter_from_value(&self, target: u64) -> impl Iterator<Item = u64> + '_ {
        let leaves = self.leaves();
        let index = leaves.partition_point(|leaf| leaf.len() == 0 || leaf.last() < target);
        let start = leaves
            .get(index)
            .map_or(0, |leaf| leaf.restart_before(target));
        // Search and skip share one cursor, including equal values across restarts.
        read(&leaves[index..], start).skip_while(move |&position| position < target)
    }

    #[cfg(test)]
    pub(super) fn from_sorted(values: &[u64]) -> Result<Self> {
        let mut positions = Self::default();
        for &value in values {
            positions.push(value)?;
        }
        Ok(positions)
    }
}

impl Leaf {
    fn len(&self) -> usize {
        (self.len.get() - 1) as usize
    }

    fn reserved_directory_bytes(len: usize) -> Result<usize> {
        let groups = len.div_ceil(RESTART);
        // One block needs no directory; later capacities grow by powers of two,
        // so appending each restart does not relocate the data area.
        if groups <= 1 {
            return Ok(0);
        }
        groups
            .checked_next_power_of_two()
            .and_then(|capacity| capacity.checked_mul(OFFSET_BYTES))
            .ok_or_else(|| "BPE position layout exceeds usize".into())
    }

    // A single block has no physical directory. Expose only valid entries,
    // including entry zero, rather than the reserved power-of-two capacity.
    fn directory(&self) -> &[[u8; OFFSET_BYTES]] {
        let groups = self.len().div_ceil(RESTART);
        if groups <= 1 {
            return &[];
        }
        self.bytes[8..8 + groups * OFFSET_BYTES].as_chunks().0
    }

    // Nonempty leaves store the tail cache before the reserved directory.
    fn data(&self) -> &[u8] {
        let data_start =
            8 + Self::reserved_directory_bytes(self.len()).expect("validated leaf layout");
        &self.bytes[data_start..]
    }

    fn first(&self) -> u64 {
        codec::read_absolute(self.data())
    }

    fn last(&self) -> u64 {
        codec::read_absolute(&self.bytes)
    }

    #[inline(always)]
    fn push(&mut self, position: u64) -> Result<()> {
        let len = self.len();
        let encoded_len = self
            .len
            .checked_add(1)
            .ok_or("BPE position count exceeds u64")?;
        let next_len = usize::try_from(encoded_len.get() - 1)?;
        if len.is_multiple_of(RESTART) {
            return self.push_restart(position, next_len, encoded_len);
        }
        let previous = self.last();
        let gap = position
            .checked_sub(previous)
            .ok_or("BPE positions are not sorted")?;
        let width = codec::gap_width(gap);
        if len == 1 {
            // The first value and tail cache fully represent two positions.
            self.bytes[..8].copy_from_slice(&position.to_le_bytes());
            self.len = encoded_len;
            return Ok(());
        }
        if len == 2 {
            let first_gap = previous - self.first();
            // Materialize both gaps only after their complete reserve succeeds.
            self.reserve(codec::gap_width(first_gap) + width)?;
            codec::append_gap(&mut self.bytes, first_gap);
        } else {
            self.reserve(width)?;
        }
        codec::append_gap(&mut self.bytes, gap);
        self.bytes[..8].copy_from_slice(&position.to_le_bytes());
        self.len = encoded_len;
        Ok(())
    }

    fn reserve(&mut self, additional: usize) -> Result<()> {
        let new_size = self
            .bytes
            .len()
            .checked_add(additional)
            .ok_or("BPE position layout exceeds usize")?;
        // Reserve before mutation. A 1.5x capacity step remains amortized linear
        // while retaining less slack in the immutable published leaves.
        if new_size > self.bytes.capacity() {
            let capacity = self.bytes.capacity();
            let capacity = capacity.saturating_add(capacity / 2).max(new_size);
            self.bytes
                .try_reserve_exact(capacity - self.bytes.len())
                .map_err(|_| "BPE positions allocation failed")?;
        }
        Ok(())
    }

    #[inline(never)]
    fn push_restart(
        &mut self,
        position: u64,
        next_len: usize,
        encoded_len: NonZeroU64,
    ) -> Result<()> {
        let len = self.len();
        if len != 0 && position < self.last() {
            return Err("BPE positions are not sorted".into());
        }
        let old_data_start = 8 + Self::reserved_directory_bytes(len)?;
        let new_data_start = 8 + Self::reserved_directory_bytes(next_len)?;
        let growth = new_data_start - old_data_start;
        let data_end = self.bytes.len();
        self.reserve(
            growth
                .checked_add(if len == 0 { 16 } else { 8 })
                .ok_or("BPE position layout exceeds usize")?,
        )?;
        if len == 0 {
            self.bytes.resize(8, 0);
        } else if growth != 0 {
            self.bytes.resize(data_end + growth, 0);
            // Entries are relative to the data area and survive this relocation.
            self.bytes
                .copy_within(old_data_start..data_end, new_data_start);
            self.bytes[old_data_start..new_data_start].fill(0);
        }
        if len != 0 && new_data_start > 8 {
            let entry = 8 + (len / RESTART) * OFFSET_BYTES;
            self.bytes[entry..entry + OFFSET_BYTES]
                .copy_from_slice(&(data_end - old_data_start).to_le_bytes());
        }
        self.bytes.extend_from_slice(&position.to_le_bytes());
        self.bytes[..8].copy_from_slice(&position.to_le_bytes());
        self.len = encoded_len;
        Ok(())
    }

    // Chunk boundaries and searches supply an aligned restart position.
    fn cursor(&self, start: usize) -> Cursor<'_> {
        if start == self.len() {
            return Cursor::new(&[], start, None);
        }
        let offset = if start == 0 {
            0
        } else {
            usize::from_le_bytes(self.directory()[start / RESTART])
        };
        debug_assert!(start.is_multiple_of(RESTART));
        let pending_tail = (self.len() == 2 && start == 0)
            .then(|| self.bytes[..8].try_into().expect("internal tail cache"));
        Cursor::new(&self.data()[offset..], start, pending_tail)
    }

    fn restart_before(&self, target: u64) -> usize {
        let directory = self.directory();
        if directory.is_empty() {
            return 0;
        }
        let data = self.data();
        // Strict comparison and fallback retain duplicates in the prior block.
        directory
            .partition_point(|entry| codec::read_absolute(&data[usize::from_le_bytes(*entry)..]) < target)
            .saturating_sub(1)
            * RESTART
    }
}

fn read(leaves: &[Leaf], mut start: usize) -> impl Iterator<Item = u64> + '_ {
    leaves.iter().flat_map(move |leaf| {
        let cursor = leaf.cursor(start);
        start = 0;
        cursor
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_search_and_chunks() {
        let mut cases = vec![
            vec![],
            vec![0],
            vec![0, 0],
            vec![u64::MAX],
            vec![0, u64::MAX],
            vec![u64::MAX, u64::MAX],
            vec![0, u64::MAX, u64::MAX],
            vec![0, 1, u64::MAX],
            vec![0, u32::MAX as u64, u32::MAX as u64 + 1, u64::MAX],
            (0..2049).map(|n| n / 3).collect(),
        ];
        cases.extend([127, 128, 129, 255, 256, 257].map(|len| (0..len).map(|n| n / 3).collect()));
        for values in cases {
            let positions = Positions::from_sorted(&values).unwrap();
            assert_eq!(positions.len(), values.len());
            assert_eq!(positions.iter().collect::<Vec<_>>(), values);
            let Repr::Leaf(leaf) = &positions.0 else {
                unreachable!()
            };
            let mut cursor = leaf.cursor(0);
            for &value in &values {
                assert_eq!(cursor.next(), Some(value));
            }
            assert_eq!(cursor.next(), None);
            assert_eq!(cursor.next(), None);
            assert_eq!(leaf.cursor(leaf.len()).next(), None);
            for target in [0, 1, 42, 127, 128, 256, u32::MAX as u64, u64::MAX] {
                let expected = values.partition_point(|&p| p < target);
                assert_eq!(
                    positions.iter_from_value(target).collect::<Vec<_>>(),
                    values[expected..]
                );
            }
            for target in [0, 1, 127, 128, 129, 511, usize::MAX] {
                let chunks: Vec<_> = positions.chunks(target).collect();
                assert_eq!(chunks.into_iter().flatten().collect::<Vec<_>>(), values);
            }
        }
        if cfg!(target_pointer_width = "64") {
            assert_eq!(std::mem::size_of::<Positions>(), 32);
        }
        // The two-element representation fits the existing sixteen inline bytes.
        for (count, bytes, spilled) in
            [(0, 0, false), (1, 16, false), (2, 16, false), (3, 18, true)]
        {
            let positions = Positions::from_sorted(&vec![7; count]).unwrap();
            let Repr::Leaf(leaf) = positions.0 else {
                unreachable!()
            };
            assert_eq!((leaf.bytes.len(), leaf.bytes.spilled()), (bytes, spilled));
        }
    }

    #[test]
    fn leaf_format_and_exhausted_cursor() {
        let positions = Positions::from_sorted(&[1, 128, 256]).unwrap();
        let Repr::Leaf(leaf) = &positions.0 else {
            unreachable!()
        };
        let mut expected = 256u64.to_le_bytes().to_vec();
        expected.extend_from_slice(&1u64.to_le_bytes());
        expected.extend_from_slice(&[127, 128, 1]);
        assert_eq!(leaf.bytes.as_slice(), expected);
        for start in [0, leaf.len()] {
            let mut cursor = leaf.cursor(start);
            while cursor.next().is_some() {}
            assert_eq!(cursor.next(), None);
            assert_eq!(cursor.next(), None);
        }
        let empty = Positions::default();
        assert_eq!(empty.iter().next(), None);
        if cfg!(target_pointer_width = "64") {
            assert_eq!(std::mem::size_of::<Cursor<'_>>(), 40);
        }
    }

    #[test]
    fn concat_preserves_boundaries_and_chunks() {
        let values: Vec<u64> = (0..1300).map(|n| n / 3).collect();
        for width in [1, 2, 3, 127, 128, 129, 700, 1300] {
            let leaves = values
                .chunks(width)
                .map(|part| Positions::from_sorted(part).unwrap());
            let positions = Positions::concat(leaves).unwrap();
            assert_eq!(positions.len(), values.len());
            assert_eq!(positions.iter().collect::<Vec<_>>(), values);
            for target in [0, 1, 42, 127, 128, 256, 1000] {
                assert_eq!(
                    positions.iter_from_value(target).collect::<Vec<_>>(),
                    values[values.partition_point(|&p| p < target)..]
                );
            }
            for target in [1, 129, 511, 10000] {
                let chunks: Vec<_> = positions.chunks(target).collect();
                assert_eq!(chunks.into_iter().flatten().collect::<Vec<_>>(), values);
            }
            if width < values.len() {
                assert!(Positions::concat([positions]).is_err());
            }
        }
        assert!(Positions::concat([]).unwrap().is_empty());
        let one = Positions::from_sorted(&[u64::MAX]).unwrap();
        let one = Positions::concat([Positions::default(), one, Positions::default()]).unwrap();
        assert!(matches!(one.0, Repr::Leaf(_)));
        assert_eq!(one.iter().collect::<Vec<_>>(), [u64::MAX]);
        let unordered = [
            Positions::from_sorted(&[2]).unwrap(),
            Positions::from_sorted(&[1]).unwrap(),
        ];
        assert!(Positions::concat(unordered).is_err());
        let mut root = Positions::concat([
            Positions::from_sorted(&[1]).unwrap(),
            Positions::from_sorted(&[1]).unwrap(),
        ])
        .unwrap();
        assert!(root.push(2).is_err());
        assert_eq!(root.iter().collect::<Vec<_>>(), [1, 1]);
    }

    #[test]
    fn append_errors_preserve_leaf() {
        for count in [1, 2, 127, 128, 129, 256, 257] {
            let values: Vec<u64> = (0..count).map(|n| 100 + n as u64).collect();
            let mut positions = Positions::from_sorted(&values).unwrap();
            assert!(positions.push(0).is_err());
            assert_eq!(positions.iter().collect::<Vec<_>>(), values);
            positions.push(u64::MAX).unwrap();
            assert_eq!(positions.iter().last(), Some(u64::MAX));
        }
        let mut leaf = Leaf {
            bytes: SmallVec::new(),
            len: NonZeroU64::MAX,
        };
        assert!(leaf.push(0).is_err());
        assert!(leaf.bytes.is_empty());
        let mut cursor = Cursor::new(&[0; 7], 0, None);
        assert!(std::panic::catch_unwind(move || cursor.next()).is_err());
    }
}
