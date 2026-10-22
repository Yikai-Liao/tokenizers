//! Nondecreasing full-u64 cohorts stored as absolute seeds and LEB128 gaps.
use smallvec::SmallVec;
use std::num::NonZeroU64;
use tk_encode::Result;
mod codec {
    //! Internal absolute-u64 / LEB128-gap format and its sequential decoder.
    use smallvec::SmallVec;

    pub(super) const RESTART: usize = 128;

    pub(super) fn gap_width(gap: u64) -> usize {
        // Each completed seven-bit group adds a byte; setting bit zero
        // also reserves one byte for a zero gap.
        let last_bit = u64::BITS - 1 - (gap | 1).leading_zeros();
        (last_bit / 7 + 1) as usize
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

    pub(super) struct Cursor<'bytes> {
        bytes: &'bytes [u8],
        position: u64,
        index: usize,
    }

    impl<'bytes> Cursor<'bytes> {
        pub(super) fn new(bytes: &'bytes [u8]) -> Self {
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
pub(super) struct Positions(Repr);
enum Repr {
    Leaf(Leaf),
    Root { leaves: Box<[Leaf]>, len: u64 },
}
struct Leaf {
    bytes: SmallVec<[u8; 16]>,
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
    pub(super) fn from_unsorted(mut values: Vec<u64>) -> Result<Self> {
        values.sort_unstable();
        let mut positions = Self::default();
        for value in values {
            positions.push(value)?;
        }
        Ok(positions)
    }
    #[inline(always)]
    pub(super) fn push(&mut self, position: u64) -> Result<()> {
        match &mut self.0 {
            Repr::Leaf(leaf) => leaf.push(position),
            Repr::Root { .. } => Err("BPE roots cannot be appended to".into()),
        }
    }
    pub(super) fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        self.leaves().iter().flat_map(Leaf::cursor)
    }
}
impl Leaf {
    fn len(&self) -> usize {
        // Stored counts include the extra one that keeps the NonZero niche;
        // remove it to recover the logical item count validated during append.
        (self.len.get() - 1) as usize
    }
    fn last(&self) -> u64 {
        codec::read_absolute(&self.bytes)
    }
    #[inline(always)]
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
    #[inline(always)]
    fn push(&mut self, position: u64) -> Result<()> {
        let len = self.len();
        let encoded_len = self
            .len
            .checked_add(1)
            .ok_or("BPE position count exceeds u64")?;
        usize::try_from(encoded_len.get() - 1)?;
        if len != 0 && position < self.last() {
            return Err("BPE positions are not sorted".into());
        }
        if len.is_multiple_of(RESTART) {
            self.reserve(if len == 0 { 16 } else { 8 })?;
            if len == 0 {
                self.bytes.resize(8, 0);
            }
            self.bytes.extend_from_slice(&position.to_le_bytes());
        } else {
            let gap = position - self.last();
            self.reserve(codec::gap_width(gap))?;
            codec::append_gap(&mut self.bytes, gap);
        }
        self.bytes[..8].copy_from_slice(&position.to_le_bytes());
        self.len = encoded_len;
        Ok(())
    }
    fn cursor(&self) -> Cursor<'_> {
        Cursor::new(if self.len() == 0 {
            &[]
        } else {
            &self.bytes[8..]
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejected_appends_preserve_gap_and_restart_streams() {
        let mut empty = Positions::default();
        assert_eq!(empty.iter().next(), None);
        empty.push(0).unwrap();
        assert_eq!(empty.iter().collect::<Vec<_>>(), [0]);
        for length in [1, 128, 256] {
            let values: Vec<_> = (1..=length).collect();
            let mut positions = Positions::from_unsorted(values.clone()).unwrap();
            assert!(positions.push(length - 1).is_err());
            assert_eq!(positions.len(), length as usize);
            assert_eq!(positions.iter().collect::<Vec<_>>(), values);
            positions.push(length).unwrap();
            positions.push(u64::MAX).unwrap();
            assert_eq!(
                positions
                    .iter()
                    .filter(|&coordinate| coordinate >= length)
                    .collect::<Vec<_>>(),
                [length, length, u64::MAX]
            );
        }
    }
}

impl Positions {
    pub(super) fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Positions {
    pub(super) fn len(&self) -> usize {
        match &self.0 {
            Repr::Leaf(leaf) => leaf.len(),
            Repr::Root { len, .. } => *len as usize,
        }
    }
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
}

impl Leaf {
    fn first(&self) -> u64 {
        codec::read_absolute(&self.bytes[8..])
    }
}
#[cfg(test)]
mod root_tests {
    use super::*;
    #[test]
    fn ordered_fragments_move_without_losing_duplicates() {
        let make = |values| Positions::from_unsorted(values).unwrap();
        let root = Positions::concat([make(vec![0, 128]), make(vec![]), make(vec![128, u64::MAX])])
            .unwrap();
        assert_eq!(root.len(), 4);
        assert_eq!(root.iter().collect::<Vec<_>>(), [0, 128, 128, u64::MAX]);
        assert!(Positions::concat([root]).is_err());
        assert!(Positions::concat([make(vec![9]), make(vec![8])]).is_err());
        let mut root = Positions::concat([make(vec![1]), make(vec![2])]).unwrap();
        assert!(root.push(3).is_err());
        assert_eq!(root.iter().collect::<Vec<_>>(), [1, 2]);
    }
}
