//! Appendable compressed leaves and immutable, nonrecursive roots.
//! A leaf stores an eight-byte last-value cache, a geometric restart directory,
//! then absolute u64 restart seeds and LEB128 gaps. Directory entries
//! hold offsets relative to the data area. SmallVec alone chooses inline storage.
use smallvec::SmallVec;
use std::num::NonZeroU64;
use tk_encode::Result;

mod codec {
    //! Internal absolute-u64 / LEB128-gap format and its sequential decoder.
    use smallvec::SmallVec;

    pub(super) const RESTART: usize = 128;

    pub(super) fn gap_width(gap: u64) -> usize {
        // LEB128 carries seven value bits per byte. Round the significant-bit
        // count upward, reserving one byte even for a zero gap.
        (u64::BITS - gap.leading_zeros()).div_ceil(7).max(1) as usize
    }

    /// Leaf reserves the complete append before changing its logical contents.
    #[inline(always)]
    pub(super) fn append_gap(bytes: &mut SmallVec<[u8; 16]>, gap: u64) {
        let width = gap_width(gap);
        debug_assert!(bytes.capacity() - bytes.len() >= width);
        let start = bytes.len();
        for _ in 0..width {
            bytes.push(0);
        }
        let mut end = start;
        leb128fmt::encode_uint_slice::<u64, 64>(gap, bytes.as_mut_slice(), &mut end)
            .expect("internal reserved gap buffer");
        debug_assert_eq!(end, bytes.len());
    }

    pub(super) fn read_absolute(bytes: &[u8]) -> u64 {
        u64::from_le_bytes(bytes[..8].try_into().expect("internal u64 position"))
    }

    pub(super) struct Cursor<'a> {
        bytes: &'a [u8],
        position: u64,
        index: usize,
    }

    impl<'a> Cursor<'a> {
        pub(super) fn new(bytes: &'a [u8]) -> Self {
            Self {
                bytes,
                position: 0,
                index: 0,
            }
        }
    }

    impl Iterator for Cursor<'_> {
        type Item = u64;
        #[inline]
        fn next(&mut self) -> Option<u64> {
            if self.bytes.is_empty() {
                return None;
            }
            // Every RESTART-th item is an absolute u64 seed; all intervening
            // items are gaps relative to the previously decoded coordinate.
            if self.index.is_multiple_of(RESTART) {
                self.position = read_absolute(self.bytes);
                self.bytes = &self.bytes[8..];
            } else {
                let mut consumed = 0;
                self.position += leb128fmt::decode_uint_slice::<u64, 64>(self.bytes, &mut consumed)
                    .expect("internal LEB128 gap");
                self.bytes = &self.bytes[consumed..];
            }
            self.index += 1;
            Some(self.position)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn gap_width() {
            let mut gaps = vec![0, 1, 127, 128, u64::MAX];
            // Each seven-bit boundary increases LEB128 width by one byte;
            // exercise the final value before it and the first value after it.
            for bit in (7..64).step_by(7) {
                gaps.extend([(1u64 << bit) - 1, 1u64 << bit]);
            }
            for gap in gaps {
                let mut bytes = SmallVec::new();
                bytes.extend_from_slice(&0u64.to_le_bytes());
                bytes.reserve(super::gap_width(gap));
                append_gap(&mut bytes, gap);
                assert_eq!(bytes.len() - 8, super::gap_width(gap));
                assert_eq!(Cursor::new(&bytes).collect::<Vec<_>>(), [0, gap]);
            }
        }
    }
}

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

    /// Merge interleaved sorted streams, retaining every duplicate. A sole stream
    /// moves unchanged; otherwise only the reader frontier is buffered while
    /// coordinates append directly to one ordered leaf.
    pub(super) fn merge(fragments: impl IntoIterator<Item = Self>) -> Result<Self> {
        let fragments: SmallVec<[Self; 1]> = fragments.into_iter().collect();
        if fragments.len() == 1 {
            return Ok(fragments.into_iter().next().expect("one fragment"));
        }
        use itertools::Itertools;
        let mut positions = Self::default();
        for position in fragments.iter().map(Self::iter).kmerge() {
            positions.push(position)?;
        }
        Ok(positions)
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
                // Fill the remaining chunk budget with whole restart blocks:
                // ceil(max(target - count, 1) / RESTART) * RESTART. Clamp at
                // this leaf's tail; saturating arithmetic handles huge targets.
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
            // Include every fully consumed leaf, plus the current leaf only
            // when part of it was consumed. end is an exclusive leaf index.
            let end = first.len() - remaining.len() + usize::from(start != 0);
            (count != 0).then(|| read(&first[..end], begin).take(count))
        })
    }

    /// Locate the first qualifying leaf using its cached tail, then its restart
    /// directory. Equal values at a preceding leaf/block tail remain visible.
    pub(super) fn iter_from(&self, target: u64) -> impl Iterator<Item = u64> + '_ {
        let leaves = self.leaves();
        // Empty leaves have no readable tail; nonempty tails strictly below
        // target can be skipped. Equality must retain duplicates in that leaf.
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
        // Stored counts include the extra one that keeps the NonZero niche;
        // remove it to recover the logical item count validated during append.
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
        // Skip the eight-byte tail cache and expose one offset per live block;
        // unused reserved directory capacity is outside this slice.
        self.bytes[8..8 + groups * OFFSET_BYTES].as_chunks().0
    }

    // Nonempty leaves store the tail cache before the reserved directory.
    fn data(&self) -> &[u8] {
        // Payload begins after the tail cache and the full reserved directory,
        // including directory entries not yet logically populated.
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
        // Decode the proposed count before mutating bytes, so an item count
        // that fits u64 but exceeds resident usize bounds fails atomically.
        let next_len = usize::try_from(encoded_len.get() - 1)?;
        if len.is_multiple_of(RESTART) {
            self.push_restart(position, next_len)?;
        } else {
            let previous = self.last();
            let gap = position
                .checked_sub(previous)
                .ok_or("BPE positions are not sorted")?;
            let width = codec::gap_width(gap);
            self.reserve(width)?;
            codec::append_gap(&mut self.bytes, gap);
            self.bytes[..8].copy_from_slice(&position.to_le_bytes());
        }
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
        // A large append can exceed that geometric step, so grow to at least
        // new_size and reserve only the capacity missing from the current length.
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
    fn push_restart(&mut self, position: u64, next_len: usize) -> Result<()> {
        let len = self.len();
        // Empty leaves have no cached tail to compare. Every later absolute
        // restart seed must preserve the same nondecreasing order as gaps.
        if len != 0 && position < self.last() {
            return Err("BPE positions are not sorted".into());
        }
        // Directory growth shifts the data area's origin; this displacement
        // is separate from the bytes needed for the new absolute restart seed.
        let old_data_start = 8 + Self::reserved_directory_bytes(len)?;
        let new_data_start = 8 + Self::reserved_directory_bytes(next_len)?;
        let growth = new_data_start - old_data_start;
        let data_end = self.bytes.len();
        // The first item needs both an eight-byte tail cache and an eight-byte
        // seed (16 bytes); later restarts add only a seed (8), plus directory growth.
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
        // A later restart writes a directory entry only when a directory exists.
        // len / RESTART is its block ordinal; the stored offset excludes headers
        // and still points at the old payload end after directory relocation.
        if len != 0 && new_data_start > 8 {
            let entry = 8 + (len / RESTART) * OFFSET_BYTES;
            self.bytes[entry..entry + OFFSET_BYTES]
                .copy_from_slice(&(data_end - old_data_start).to_le_bytes());
        }
        self.bytes.extend_from_slice(&position.to_le_bytes());
        self.bytes[..8].copy_from_slice(&position.to_le_bytes());
        Ok(())
    }

    // Chunk boundaries and searches supply an aligned restart position.
    fn cursor(&self, start: usize) -> Cursor<'_> {
        if start == self.len() {
            return Cursor::new(&[]);
        }
        // The first block starts at data offset zero even without a directory.
        // Later aligned starts select their block's cached byte offset.
        let offset = if start == 0 {
            0
        } else {
            usize::from_le_bytes(self.directory()[start / RESTART])
        };
        debug_assert!(start.is_multiple_of(RESTART));
        // The slice begins at an absolute seed, so decoder indices are local.
        Cursor::new(&self.data()[offset..])
    }

    fn restart_before(&self, target: u64) -> usize {
        let directory = self.directory();
        if directory.is_empty() {
            return 0;
        }
        let data = self.data();
        // Strict comparison and fallback retain duplicates in the prior block.
        // Read seeds through relative byte offsets, then convert the preceding
        // block ordinal into an item index. Saturation keeps the first block at zero.
        directory
            .partition_point(|entry| {
                codec::read_absolute(&data[usize::from_le_bytes(*entry)..]) < target
            })
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
    fn leaf_roundtrip() {
        let cases = [
            vec![],
            vec![0],
            vec![0, 0],
            vec![0, 1, u64::MAX],
            vec![0, u32::MAX as u64, u32::MAX as u64 + 1, u64::MAX],
            (0..2049).map(|n| n / 3).collect(),
        ];
        for values in cases {
            let positions = Positions::from_sorted(&values).unwrap();
            assert_eq!(positions.len(), values.len());
            assert_eq!(positions.iter().collect::<Vec<_>>(), values);
            for target in [0, 1, 42, 127, 128, 256, u32::MAX as u64, u64::MAX] {
                let expected = values.partition_point(|&p| p < target);
                assert_eq!(
                    positions.iter_from(target).collect::<Vec<_>>(),
                    values[expected..]
                );
            }
            for target in [0, 1, 127, 128, 129, 511, usize::MAX] {
                let chunks: Vec<_> = positions.chunks(target).collect();
                assert_eq!(chunks.into_iter().flatten().collect::<Vec<_>>(), values);
            }
        }
    }

    #[test]
    fn concat_boundaries() {
        // Three copies of every coordinate force equal values across both leaf
        // and restart boundaries while keeping the full sequence sorted.
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
                    positions.iter_from(target).collect::<Vec<_>>(),
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
    fn merge_streams() {
        assert!(Positions::merge([]).unwrap().is_empty());
        let root = Positions::concat([
            Positions::from_sorted(&[0, 2]).unwrap(),
            Positions::from_sorted(&[u64::MAX]).unwrap(),
        ])
        .unwrap();
        let root = Positions::merge([root]).unwrap();
        assert!(matches!(root.0, Repr::Root { .. }));
        let positions = Positions::merge([
            root,
            Positions::default(),
            Positions::from_sorted(&[0, 1, u64::MAX]).unwrap(),
        ])
        .unwrap();
        assert_eq!(
            positions.iter().collect::<Vec<_>>(),
            [0, 0, 1, 2, u64::MAX, u64::MAX]
        );
    }

    #[test]
    fn append_errors() {
        for count in [1, 2, 127, 128, 129, 256, 257] {
            let values: Vec<u64> = (0..count).map(|n| 100 + n as u64).collect();
            let mut positions = Positions::from_sorted(&values).unwrap();
            assert!(positions.push(0).is_err());
            assert_eq!(positions.len(), count);
            assert_eq!(positions.iter().collect::<Vec<_>>(), values);
            positions.push(u64::MAX).unwrap();
            assert_eq!(positions.len(), count + 1);
            assert_eq!(positions.iter().last(), Some(u64::MAX));
        }
        let mut leaf = Leaf {
            bytes: SmallVec::new(),
            len: NonZeroU64::MAX,
        };
        assert!(leaf.push(0).is_err());
        assert!(leaf.bytes.is_empty());
        let mut cursor = Cursor::new(&[0; 7]);
        assert!(std::panic::catch_unwind(move || cursor.next()).is_err());
    }
}
