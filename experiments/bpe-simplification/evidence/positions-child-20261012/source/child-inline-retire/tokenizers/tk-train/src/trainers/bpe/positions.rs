//! Appendable compressed leaves and immutable, nonrecursive roots.
//! A leaf stores an eight-byte last-value cache, a geometric restart directory,
//! then absolute u64 restart seeds and LEB128 gaps. Directory entries
//! hold offsets relative to the data area. SmallVec alone chooses inline storage.
use smallvec::SmallVec;
use std::num::NonZeroU64;
use tk_encode::Result;

const RESTART: usize = 128;
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
    #[inline]
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

    /// Partition by position count, splitting large leaves at restart boundaries
    /// and grouping small leaves. Each task holds its starting leaf directly.
    pub(super) fn tasks(&self, target_items: usize) -> impl Iterator<Item = ReadTask<'_>> {
        let target = target_items.max(1);
        let leaves = self.leaves();
        let (mut leaf, mut start) = (0, 0);
        std::iter::from_fn(move || {
            while leaf < leaves.len() && start == leaves[leaf].len() {
                leaf += 1;
                start = 0;
            }
            let (first, begin) = (leaf, start);
            let mut count = 0;
            while leaf < leaves.len() {
                let take = target
                    .saturating_sub(count)
                    .max(1)
                    .div_ceil(RESTART)
                    .saturating_mul(RESTART)
                    .min(leaves[leaf].len() - start);
                start += take;
                count += take;
                if count >= target {
                    break;
                }
                leaf += 1;
                start = 0;
            }
            (count != 0).then(|| ReadTask {
                leaves: &leaves[first..(leaf + 1).min(leaves.len())],
                start: begin,
                len: count,
            })
        })
    }

    /// Locate the first qualifying leaf using its cached tail, then its restart
    /// directory. Equal values at a preceding leaf/block tail remain visible.
    pub(super) fn iter_from_value(&self, target: u64) -> impl Iterator<Item = u64> + '_ {
        let leaves = self.leaves();
        let index = leaves.partition_point(|leaf| leaf.len() == 0 || leaf.last() < target);
        let start = leaves.get(index).map_or(0, |leaf| leaf.lower_bound(target));
        read(&leaves[index..], start)
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

    fn directory_bytes(len: usize) -> Result<usize> {
        let groups = len.div_ceil(RESTART);
        if groups <= 1 {
            return Ok(0);
        }
        groups
            .checked_next_power_of_two()
            .and_then(|capacity| capacity.checked_mul(OFFSET_BYTES))
            .ok_or_else(|| "BPE position layout exceeds usize".into())
    }

    fn head(&self) -> usize {
        8 + Self::directory_bytes(self.len()).expect("validated leaf layout")
    }

    fn first(&self) -> u64 {
        read_u64(&self.bytes[self.head()..])
    }

    fn last(&self) -> u64 {
        read_u64(&self.bytes)
    }

    #[inline]
    fn push(&mut self, position: u64) -> Result<()> {
        let len = self.len();
        let encoded_len = self
            .len
            .checked_add(1)
            .ok_or("BPE position count exceeds u64")?;
        let next_len = usize::try_from(encoded_len.get() - 1)?;
        let previous = if len == 0 { 0 } else { self.last() };
        let gap = position
            .checked_sub(previous)
            .ok_or("BPE positions are not sorted")?;
        let mut encoded = [0u8; 10];
        let width = if len.is_multiple_of(RESTART) {
            encoded[..8].copy_from_slice(&position.to_le_bytes());
            8
        } else {
            let mut delta = gap;
            let mut width = 0;
            while delta >= 128 {
                encoded[width] = (delta as u8 & 127) | 128;
                delta >>= 7;
                width += 1;
            }
            encoded[width] = delta as u8;
            width + 1
        };
        // Only a restart can grow the directory. Ordinary gaps simply append
        // bytes and update the fixed front cache; no footer is moved or rewritten.
        let (old_head, new_head) = if len.is_multiple_of(RESTART) {
            (
                8 + Self::directory_bytes(len)?,
                8 + Self::directory_bytes(next_len)?,
            )
        } else {
            (0, 0)
        };
        let growth = new_head - old_head;
        let data_end = self.bytes.len();
        let new_size = data_end
            .checked_add(growth)
            .and_then(|size| size.checked_add(width))
            .and_then(|size| size.checked_add(if len == 0 { 8 } else { 0 }))
            .ok_or("BPE position layout exceeds usize")?;
        // Reserve before mutation. A 1.5x capacity step remains amortized linear
        // while retaining less slack in the immutable published leaves.
        if new_size > self.bytes.capacity() {
            let capacity = self.bytes.capacity();
            let capacity = capacity.saturating_add(capacity / 2).max(new_size);
            self.bytes
                .try_reserve_exact(capacity - data_end)
                .map_err(|_| "BPE positions allocation failed")?;
        }
        if len == 0 {
            self.bytes.resize(8, 0);
        } else if growth != 0 {
            self.bytes.resize(data_end + growth, 0);
            self.bytes.copy_within(old_head..data_end, new_head);
            self.bytes[old_head..new_head].fill(0);
        }
        if len != 0 && len.is_multiple_of(RESTART) && new_head > 8 {
            let entry = 8 + (len / RESTART) * OFFSET_BYTES;
            self.bytes[entry..entry + OFFSET_BYTES]
                .copy_from_slice(&(data_end - old_head).to_le_bytes());
        }
        self.bytes.extend_from_slice(&encoded[..width]);
        self.bytes[..8].copy_from_slice(&position.to_le_bytes());
        self.len = encoded_len;
        Ok(())
    }

    fn cursor(&self, start: usize) -> Cursor<'_> {
        if start == self.len() {
            return Cursor {
                bytes: &[],
                position: 0,
                index: start,
                end: start,
            };
        }
        let head = self.head();
        let block = start / RESTART;
        let offset = if block == 0 {
            0
        } else {
            let entry = 8 + block * OFFSET_BYTES;
            usize::from_le_bytes(
                self.bytes[entry..entry + OFFSET_BYTES]
                    .try_into()
                    .expect("restart offset"),
            )
        };
        let mut cursor = Cursor {
            bytes: &self.bytes[head + offset..],
            position: 0,
            index: block * RESTART,
            end: self.len(),
        };
        for _ in cursor.index..start {
            cursor.next();
        }
        cursor
    }

    fn lower_bound(&self, target: u64) -> usize {
        let groups = self.len().div_ceil(RESTART);
        let block = if groups <= 1 {
            0
        } else {
            let directory = self.bytes[8..8 + groups * OFFSET_BYTES]
                .as_chunks::<OFFSET_BYTES>()
                .0;
            let head = self.head();
            directory
                .partition_point(|entry| {
                    let offset = usize::from_le_bytes(*entry);
                    read_u64(&self.bytes[head + offset..]) < target
                })
                .saturating_sub(1)
        };
        let start = block * RESTART;
        start
            + self
                .cursor(start)
                .take(RESTART)
                .take_while(|&p| p < target)
                .count()
    }
}

/// Borrowed reading work; callers see coordinates, never leaf storage or blocks.
pub(super) struct ReadTask<'a> {
    leaves: &'a [Leaf],
    start: usize,
    len: usize,
}

impl ReadTask<'_> {
    pub(super) fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        read(self.leaves, self.start).take(self.len)
    }
}

fn read(leaves: &[Leaf], start: usize) -> impl Iterator<Item = u64> + '_ {
    let mut leaves = leaves.iter();
    let mut cursor = leaves.next().map(|leaf| leaf.cursor(start));
    std::iter::from_fn(move || {
        loop {
            if let Some(value) = cursor.as_mut().and_then(Iterator::next) {
                return Some(value);
            }
            cursor = Some(leaves.next()?.cursor(0));
        }
    })
}

fn read_u64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes[..8].try_into().expect("internal u64 position"))
}

struct Cursor<'a> {
    bytes: &'a [u8],
    position: u64,
    index: usize,
    end: usize,
}

impl Iterator for Cursor<'_> {
    type Item = u64;
    fn next(&mut self) -> Option<u64> {
        if self.index == self.end {
            return None;
        }
        if self.index.is_multiple_of(RESTART) {
            let (seed, rest) = self
                .bytes
                .split_first_chunk::<8>()
                .expect("internal restart seed");
            self.position = u64::from_le_bytes(*seed);
            self.bytes = rest;
        } else {
            let mut delta = 0u64;
            let mut shift = 0;
            loop {
                let byte = self.bytes[0];
                self.bytes = &self.bytes[1..];
                delta |= u64::from(byte & 127) << shift;
                if byte < 128 {
                    break;
                }
                shift += 7;
            }
            self.position += delta;
        }
        self.index += 1;
        Some(self.position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_search_and_tasks() {
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
                    positions.iter_from_value(target).collect::<Vec<_>>(),
                    values[expected..]
                );
            }
            for target in [0, 1, 127, 128, 129, 511, usize::MAX] {
                let tasks: Vec<_> = positions.tasks(target).collect();
                assert_eq!(
                    tasks.iter().flat_map(ReadTask::iter).collect::<Vec<_>>(),
                    values
                );
                assert_eq!(
                    tasks.iter().map(|task| task.len).sum::<usize>(),
                    values.len()
                );
            }
        }
        if cfg!(target_pointer_width = "64") {
            assert_eq!(std::mem::size_of::<Positions>(), 32);
        }
        // Actual byte width determines SmallVec storage, without item-count forms.
        for (count, bytes, spilled) in [(0, 0, false), (1, 16, false), (2, 17, true), (3, 18, true)]
        {
            let positions = Positions::from_sorted(&vec![7; count]).unwrap();
            let Repr::Leaf(leaf) = positions.0 else {
                unreachable!()
            };
            assert_eq!((leaf.bytes.len(), leaf.bytes.spilled()), (bytes, spilled));
        }
    }

    #[test]
    fn concat_preserves_boundaries_and_tasks() {
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
                let tasks: Vec<_> = positions.tasks(target).collect();
                assert_eq!(
                    tasks.iter().flat_map(ReadTask::iter).collect::<Vec<_>>(),
                    values
                );
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
        let mut cursor = Cursor {
            bytes: &[0; 7],
            position: 0,
            index: 0,
            end: 1,
        };
        assert!(std::panic::catch_unwind(move || cursor.next()).is_err());
    }
}
