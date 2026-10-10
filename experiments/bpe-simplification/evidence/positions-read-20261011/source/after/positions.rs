//! Owned nondecreasing position sequences with inline pairs and restart/delta bytes.
//! Construction writes directly into the final Vec, then transfers ownership to Box.
//! Compressed layout: usize element count, a usize offset per block only when
//! there are multiple blocks, then the data area. Offsets are relative to that
//! area; a single block omits the directory. Each block starts with a little-endian
//! u64 coordinate, followed by unsigned LEB128 gaps for its remaining values.
use std::ops::Range;
use tk_encode::Result;

const RESTART: usize = 128;

/// Immutable full-u64 lists with inline pairs and owned restart/delta bytes.
/// Ownership and automatic Send/Sync come from Box, without raw allocation or borrowed storage.
#[derive(Default)]
pub(super) enum Positions {
    #[default]
    Empty,
    One(u64),
    Two(u64, u64),
    Compressed(Box<[u8]>),
}

impl Positions {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::One(_) => 1,
            Self::Two(_, _) => 2,
            Self::Compressed(bytes) => {
                usize::from_le_bytes(bytes[..std::mem::size_of::<usize>()].try_into().unwrap())
            }
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Construct from nondecreasing coordinates, preserving duplicates.
    pub(super) fn from_sorted(values: &[u64]) -> Result<Self> {
        Self::encode(values.len(), values.iter().copied())
    }

    /// Take ownership of ordered fragments and concatenate without sorting or deduplication.
    /// An empty input is empty; a sole nonempty fragment is returned unchanged.
    pub(super) fn concat(mut fragments: Vec<Self>) -> Result<Self> {
        fragments.retain(|fragment| !fragment.is_empty());
        match fragments.len() {
            0 => return Ok(Self::Empty),
            1 => return Ok(fragments.pop().unwrap()),
            _ => {}
        }
        let count = fragments.iter().try_fold(0usize, |count, fragment| {
            count
                .checked_add(fragment.len())
                .ok_or("BPE fragment count exceeds usize")
        })?;
        Self::encode(count, fragments.iter().flat_map(Self::iter))
    }

    // Only a slice and owned fragment lengths supply cardinality.
    // The iterator stays private; safe Vec writes require no trusted-iterator or
    // uninitialized-memory protocol. Tiny lists bypass encoding entirely.
    fn encode(count: usize, mut values: impl Iterator<Item = u64>) -> Result<Self> {
        if count == 0 {
            return Ok(Self::Empty);
        }
        if count <= 2 {
            let first = values
                .next()
                .expect("nonempty positions have a first value");
            if count == 1 {
                return Ok(Self::One(first));
            }
            let last = values.next().expect("two positions have a second value");
            if last < first {
                return Err("BPE positions are not sorted".into());
            }
            return Ok(Self::Two(first, last));
        }

        let groups = count.div_ceil(RESTART);
        let head = (1usize + if groups > 1 { groups } else { 0 })
            .checked_mul(std::mem::size_of::<usize>())
            .ok_or("BPE position layout exceeds usize")?;
        // Reserve the directory and the minimum stream width (one byte per gap,
        // eight per seed). Larger gaps may grow Vec; boxing may also reallocate.
        // This is one encoding pass, not a promise of one allocation.
        let capacity = groups
            .checked_mul(7)
            .and_then(|seeds| count.checked_add(seeds))
            .and_then(|stream| head.checked_add(stream))
            .ok_or("BPE position layout exceeds usize")?;
        let mut bytes = Vec::with_capacity(capacity);
        bytes.resize(head, 0);
        bytes[..std::mem::size_of::<usize>()].copy_from_slice(&count.to_le_bytes());
        let mut previous = 0;
        for (index, position) in values.enumerate() {
            // Restart seeds are written absolutely, but this subtraction still
            // checks nondecreasing order across the preceding block boundary.
            let gap = position
                .checked_sub(previous)
                .ok_or("BPE positions are not sorted")?;
            if index.is_multiple_of(RESTART) {
                if groups > 1 {
                    let entry = (1 + index / RESTART) * std::mem::size_of::<usize>();
                    let offset = bytes.len() - head;
                    bytes[entry..entry + std::mem::size_of::<usize>()]
                        .copy_from_slice(&offset.to_le_bytes());
                }
                bytes.extend_from_slice(&position.to_le_bytes());
            } else {
                let mut delta = gap;
                while delta >= 128 {
                    bytes.push((delta as u8 & 127) | 128);
                    delta >>= 7;
                }
                bytes.push(delta as u8);
            }
            previous = position;
        }
        Ok(Self::Compressed(bytes.into_boxed_slice()))
    }

    /// Return the compressed data area after the count and optional directory.
    /// Only Compressed has this storage; callers must establish that variant.
    fn compressed_bytes(&self) -> &[u8] {
        let Self::Compressed(bytes) = self else {
            unreachable!("inline positions have no compressed data area")
        };
        let groups = self.block_count();
        let prefix = (1 + if groups > 1 { groups } else { 0 }) * std::mem::size_of::<usize>();
        &bytes[prefix..]
    }

    fn offset(&self, block: usize) -> usize {
        if block == 0 {
            return 0;
        }
        let Self::Compressed(bytes) = self else {
            unreachable!()
        };
        let start = (1 + block) * std::mem::size_of::<usize>();
        usize::from_le_bytes(
            bytes[start..start + std::mem::size_of::<usize>()]
                .try_into()
                .unwrap(),
        )
    }

    fn block_count(&self) -> usize {
        self.len().div_ceil(RESTART)
    }

    fn block_ranges(&self, target_items: usize) -> impl Iterator<Item = Range<usize>> {
        let blocks = target_items.div_ceil(RESTART).max(1);
        (0..self.block_count())
            .step_by(blocks)
            .map(move |begin| begin..(begin + blocks).min(self.block_count()))
    }

    /// Borrow read-only fragments, rounding the target up to an independent restart.
    /// Empty lists yield no fragments; short lists yield one. No values are copied
    /// or decoded until a fragment is iterated.
    pub(super) fn chunks(&self, target_items: usize) -> impl Iterator<Item = Chunk<'_>> {
        self.block_ranges(target_items).map(move |blocks| Chunk {
            positions: self,
            blocks,
        })
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        self.read_blocks(0..self.block_count())
    }

    fn read_blocks(&self, range: Range<usize>) -> impl Iterator<Item = u64> + '_ {
        // Reject inverted ranges, then clamp before multiplying. If a range starts
        // past the directory, start == end and no directory pointer is formed.
        assert!(range.start <= range.end);
        let blocks = self.block_count();
        let start = (range.start.min(blocks) * RESTART).min(self.len());
        let end = (range.end.min(blocks) * RESTART).min(self.len());
        if !matches!(self, Self::Compressed(_)) {
            // Zero fills the fixed-size array, not a sentinel: zero is a valid
            // coordinate. The actual list length truncates the unused entries.
            let values = match self {
                Self::One(value) => [*value, 0],
                Self::Two(first, last) => [*first, *last],
                _ => [0, 0],
            };
            itertools::Either::Left(values.into_iter().take(end).skip(start))
        } else {
            let bytes = if start == end {
                &[]
            } else {
                &self.compressed_bytes()[self.offset(range.start)..]
            };
            itertools::Either::Right(Cursor {
                bytes,
                position: 0,
                index: start,
                end,
            })
        }
    }

    fn lower_bound(&self, target: u64) -> usize {
        let mut begin = 0;
        let mut end = self.block_count();
        while begin < end {
            let mid = (begin + end) / 2;
            if self.read_blocks(mid..mid + 1).next().unwrap() < target {
                begin = mid + 1;
            } else {
                end = mid;
            }
        }
        // The binary search finds the first restart seed at or above target.
        // Its predecessor can contain the first match near its tail, including
        // duplicates spanning the boundary, so scan that block before advancing.
        let block = begin.saturating_sub(1);
        block * RESTART
            + self
                .read_blocks(block..block + 1)
                .take_while(|&p| p < target)
                .count()
    }

    /// Iterates from a list index, rather than a corpus coordinate.
    fn iter_from(&self, index: usize) -> impl Iterator<Item = u64> + '_ {
        self.read_blocks(index / RESTART..self.block_count())
            .skip(index % RESTART)
    }

    /// Iterate all coordinates at or above the target, including repeated values.
    pub(super) fn iter_from_value(&self, target: u64) -> impl Iterator<Item = u64> + '_ {
        self.iter_from(self.lower_bound(target))
    }
}

/// A borrowed read-only fragment whose storage boundaries stay inside this module.
pub(super) struct Chunk<'a> {
    positions: &'a Positions,
    blocks: Range<usize>,
}

impl Chunk<'_> {
    pub(super) fn len(&self) -> usize {
        let count = self.positions.len();
        let start = (self.blocks.start * RESTART).min(count);
        let end = (self.blocks.end * RESTART).min(count);
        end - start
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        self.positions.read_blocks(self.blocks.clone())
    }
}

/// A reader's private decode state over a borrowed immutable position stream.
/// Absolute restart seeds bound replay; index/end delimit the requested blocks.
/// Bytes come only from the internal encoder, and the known element count ends
/// iteration. Its format guarantees complete seeds, bounded gaps and exact sums,
/// permitting direct indexing and addition; this does not validate external bytes.
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
            self.position = u64::from_le_bytes(self.bytes[..8].try_into().unwrap());
            self.bytes = &self.bytes[8..];
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
    fn chunks_cover_lists_without_losing_duplicates_at_fragment_boundaries() {
        for length in [0, 1, 2, 3, 127, 128, 129, 255, 256, 257, 1024] {
            // Equal coordinates span both restart and fragment boundaries.
            let values: Vec<_> = (0..length).map(|index| (index / 200) as u64).collect();
            let positions = Positions::from_sorted(&values).unwrap();
            for target in [0, 1, 127, 128, 129, 130, 255, 256, 257, usize::MAX] {
                let chunks: Vec<_> = positions.chunks(target).collect();
                for chunk in &chunks {
                    assert!(chunk.len() > 0);
                    assert_eq!(chunk.iter().count(), chunk.len());
                }
                assert!(
                    chunks
                        .iter()
                        .flat_map(Chunk::iter)
                        .eq(values.iter().copied())
                );
            }
        }

        // The requested target rounds up, so length > target can still be complete.
        for (length, target, expected) in [
            (129, 128, vec![128, 1]),
            (129, 129, vec![129]),
            (255, 129, vec![255]),
        ] {
            let positions = Positions::from_sorted(&vec![7; length]).unwrap();
            let lengths: Vec<_> = positions.chunks(target).map(|chunk| chunk.len()).collect();
            assert_eq!(lengths, expected);
        }
    }

    #[test]
    fn concat_preserves_duplicates_and_reuses_a_single_nonempty_fragment() {
        assert!(Positions::concat(Vec::new()).unwrap().is_empty());
        assert!(
            Positions::concat(vec![Positions::default(), Positions::default()])
                .unwrap()
                .is_empty()
        );
        let fragment = Positions::from_sorted(&[7, 7, u64::MAX]).unwrap();
        let Positions::Compressed(bytes) = &fragment else {
            panic!("expected compressed list")
        };
        let pointer = bytes.as_ptr();
        let reused =
            Positions::concat(vec![Positions::default(), fragment, Positions::default()]).unwrap();
        let Positions::Compressed(bytes) = &reused else {
            panic!("expected compressed list")
        };
        assert_eq!(pointer, bytes.as_ptr());
        assert!(reused.iter().eq([7, 7, u64::MAX]));

        let joined = Positions::concat(vec![
            Positions::from_sorted(&[7, 7]).unwrap(),
            Positions::default(),
            Positions::from_sorted(&[7, u64::MAX]).unwrap(),
        ])
        .unwrap();
        assert!(joined.iter().eq([7, 7, 7, u64::MAX]));
    }

    #[test]
    fn direct_encoding_retains_restart_directory_and_gap_format() {
        // Exercise a growing Vec with ten-byte gaps and a second restart.
        let mut values = vec![0; 128];
        values[1..].fill(u64::MAX);
        values.push(u64::MAX);
        let positions = Positions::from_sorted(&values).unwrap();
        let Positions::Compressed(bytes) = &positions else {
            panic!("expected compressed list")
        };
        let mut expected = Vec::new();
        expected.extend_from_slice(&129usize.to_le_bytes());
        expected.extend_from_slice(&0usize.to_le_bytes());
        expected.extend_from_slice(&144usize.to_le_bytes());
        expected.extend_from_slice(&0u64.to_le_bytes());
        expected.extend_from_slice(&[255, 255, 255, 255, 255, 255, 255, 255, 255, 1]);
        expected.extend_from_slice(&[0; 126]);
        expected.extend_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(&**bytes, expected);
        assert!(positions.iter().eq(values));
    }

    #[test]
    fn storage_roundtrips_seeks_and_supports_concurrent_readers() {
        // Freeze and seek lists across inline, chunk and full-u64 boundaries.
        let boundary = [0, u32::MAX as u64, 1 << 32, 1 << 63, u64::MAX];
        let saved: Vec<_> = [0, 1, 2, 3, 127, 128, 129, 257, 1024]
            .into_iter()
            .map(|length| {
                let mut values: Vec<_> =
                    (0..length).map(|i| boundary[i % boundary.len()]).collect();
                if length == 3 {
                    values = vec![0, u64::MAX, u64::MAX];
                }
                values.sort_unstable();
                let positions = Positions::from_sorted(&values).unwrap();
                let fragments = values
                    .chunks(17)
                    .map(|chunk| Positions::from_sorted(chunk).unwrap())
                    .collect();
                assert!(
                    Positions::concat(fragments)
                        .unwrap()
                        .iter()
                        .eq(values.iter().copied())
                );
                assert!(positions.iter().eq(values.iter().copied()));
                for index in (0..=length)
                    .step_by(if cfg!(miri) { 127 } else { 1 })
                    .chain([length])
                {
                    assert!(
                        positions
                            .iter_from(index)
                            .eq(values[index..].iter().copied())
                    );
                }
                for target in boundary
                    .into_iter()
                    .flat_map(|p| [p.saturating_sub(1), p, p.saturating_add(1)])
                {
                    assert_eq!(
                        positions.lower_bound(target),
                        values.partition_point(|&p| p < target)
                    );
                    assert!(
                        positions.iter_from_value(target).eq(values
                            .iter()
                            .copied()
                            .filter(|&position| position >= target))
                    );
                }
                assert!(
                    positions
                        .read_blocks(usize::MAX..usize::MAX)
                        .next()
                        .is_none()
                );
                (positions, values)
            })
            .collect();

        // Independently build lists while other threads read existing owned lists.
        // No Rayon collector is involved in Miri.
        std::thread::scope(|scope| {
            for _ in 0..2 {
                let saved = &saved;
                scope.spawn(move || {
                    let values: Vec<_> = (0..512).map(|i| (1 << 63) + i).collect();
                    let next = Positions::from_sorted(&values).unwrap();
                    assert!(next.iter().eq(values));
                    for (positions, expected) in saved {
                        assert!(positions.iter().eq(expected.iter().copied()));
                    }
                });
            }
        });
        for (positions, expected) in saved {
            assert!(positions.iter().eq(expected));
        }
    }

    #[test]
    // The inverted range is deliberately passed as malformed decoder input.
    #[allow(clippy::reversed_empty_ranges)]
    fn storage_rejects_inverted_ranges_and_unsorted_input() {
        assert!(Positions::from_sorted(&[u64::MAX, 0]).is_err());
        for (a, b) in [(&[u64::MAX][..], &[0][..]), (&[0, u64::MAX][..], &[1][..])] {
            let fragments = [a, b].map(|v| Positions::from_sorted(v).unwrap());
            assert!(Positions::concat(fragments.into()).is_err());
        }
        let mut descending_at_restart = vec![1; 129];
        descending_at_restart[128] = 0;
        assert!(Positions::from_sorted(&descending_at_restart).is_err());
        let values: Vec<_> = (0..129u64).collect();
        let positions = Positions::from_sorted(&values).unwrap();
        let call = std::panic::AssertUnwindSafe(|| positions.read_blocks(1000..0).next());
        assert!(std::panic::catch_unwind(call).is_err());
    }
}
