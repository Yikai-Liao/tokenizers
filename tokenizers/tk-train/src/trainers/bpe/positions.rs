//! Owned restart/delta occurrence lists with inline pairs and reusable codec scratch.
use std::{
    cell::{RefCell, RefMut},
    ops::Range,
};
use thread_local::ThreadLocal;
use tk_encode::Result;
const RESTART: usize = 128;

/// Reuses per-thread position encoder buffers for one training attempt.
/// Published lists own their bytes; this object retains only temporary scratch.
pub(super) struct Codec {
    scratch: ThreadLocal<RefCell<CodecScratch>>,
}
impl Codec {
    /// Reserves scratch capacity for the expected number of executing threads.
    pub(super) fn new(num_threads: usize) -> Self {
        Self {
            scratch: ThreadLocal::with_capacity(num_threads),
        }
    }
    pub(super) fn lease(&self) -> Lease<'_> {
        Lease {
            scratch: self.scratch.get_or_default().borrow_mut(),
        }
    }
}

/// One executing thread's reusable codec buffers, borrowed together.
/// Bytes and offsets are overwritten per stream and reused across freezes.
/// The RefCell guard grants exclusive access to both scratch buffers.
#[derive(Default)]
struct CodecScratch {
    bytes: Vec<u8>,
    offsets: Vec<usize>,
}

impl CodecScratch {
    // Each restart stores an absolute coordinate; the rest of its block stores
    // varint gaps. Reset both buffers together so offsets address this stream.
    fn encode_stream(&mut self, input: &Input<'_>) -> Result<()> {
        self.bytes.clear();
        self.offsets.clear();
        let mut previous = 0;
        for (i, position) in input.iter().enumerate() {
            let gap = position
                .checked_sub(previous)
                .ok_or("BPE positions are not sorted")?;
            if i.is_multiple_of(RESTART) {
                self.offsets.push(self.bytes.len());
                self.bytes.extend_from_slice(&position.to_le_bytes());
            } else {
                let mut delta = gap;
                while delta >= 128 {
                    self.bytes.push((delta as u8 & 127) | 128);
                    delta >>= 7;
                }
                self.bytes.push(delta as u8);
            }
            previous = position;
        }
        Ok(())
    }
}

/// Exclusive access to one thread's reusable position encoder buffers.
/// Frozen lists own their bytes and can outlive this guard and the Codec.
pub(super) struct Lease<'codec> {
    scratch: RefMut<'codec, CodecScratch>,
}

/// Mutable sorted task positions, narrowed to u32 until a full-u64 value appears.
/// Small buffers stay inline; promotion preserves every previously stored value.
pub(super) enum Builder {
    Narrow(smallvec::SmallVec<[u32; 2]>),
    Wide(smallvec::SmallVec<[u64; 2]>),
}

impl Default for Builder {
    fn default() -> Self {
        Self::Narrow(smallvec::SmallVec::new())
    }
}

impl Builder {
    pub(super) fn iter(
        &self,
    ) -> impl DoubleEndedIterator<Item = u64> + ExactSizeIterator + Clone + '_ {
        match self {
            Self::Narrow(values) => itertools::Either::Left(values.iter().map(|&p| u64::from(p))),
            Self::Wide(values) => itertools::Either::Right(values.iter().copied()),
        }
    }

    pub(super) fn len(&self) -> usize {
        self.iter().len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(super) fn push(&mut self, position: u64) -> Result<()> {
        if self.iter().next_back().is_some_and(|last| position < last) {
            return Err("BPE positions are not sorted".into());
        }
        self.push_ordered(position);
        Ok(())
    }

    // Snapshot scans visit disjoint ordered matches; callers retain that order.
    // Generic append/input paths use `push` and keep its validation.
    pub(super) fn push_ordered(&mut self, position: u64) {
        if let Self::Narrow(values) = self {
            if let Ok(position) = u32::try_from(position) {
                values.push(position);
                return;
            }
            *self = Self::Wide(values.iter().map(|&p| u64::from(p)).collect());
        }
        if let Self::Wide(values) = self {
            values.push(position);
        }
    }

    pub(super) fn append(&mut self, mut other: Self) -> Result<()> {
        if self.is_empty() {
            *self = other;
            return Ok(());
        }
        if let (Some(last), Some(first)) = (self.iter().next_back(), other.iter().next())
            && first < last
        {
            return Err("BPE positions are not sorted".into());
        }
        match (&mut *self, &mut other) {
            (Self::Narrow(a), Self::Narrow(b)) => a.append(b),
            (Self::Wide(a), Self::Wide(b)) => a.append(b),
            _ => {
                for position in other.iter() {
                    self.push(position)?;
                }
            }
        }
        Ok(())
    }
}

/// Borrowed storage views accepted by the encoder, with a trustworthy cardinality.
/// Restricting input to these concrete types keeps allocation bounds independent
/// of arbitrary safe ExactSizeIterator implementations.
pub(super) enum Input<'a> {
    Builder(&'a Builder),
    Slice(&'a [u64]),
    Fragments(&'a [Positions]),
}

impl Input<'_> {
    fn len(&self) -> Result<usize> {
        match self {
            Self::Builder(values) => Ok(values.len()),
            Self::Slice(values) => Ok(values.len()),
            Self::Fragments(values) => values.iter().try_fold(0usize, |n, p| {
                n.checked_add(p.len())
                    .ok_or_else(|| "BPE fragment count exceeds usize".into())
            }),
        }
    }

    fn bounds(&self) -> Option<(u64, u64)> {
        match self {
            Self::Builder(values) => Some((values.iter().next()?, values.iter().next_back()?)),
            Self::Slice(values) => Some((*values.first()?, *values.last()?)),
            Self::Fragments(values) => Some((
                values.iter().find_map(|p| p.iter().next())?,
                values
                    .iter()
                    .rev()
                    .filter(|p| !p.is_empty())
                    .find_map(|p| p.iter_from(p.len() - 1).next())?,
            )),
        }
    }

    fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        match self {
            Self::Builder(values) => {
                itertools::Either::Left(itertools::Either::Left(values.iter()))
            }
            Self::Slice(values) => {
                itertools::Either::Left(itertools::Either::Right(values.iter().copied()))
            }
            Self::Fragments(values) => {
                itertools::Either::Right(values.iter().flat_map(Positions::iter))
            }
        }
    }
}

/// Immutable full-u64 lists with inline pairs and owned restart/delta bytes.
/// The first word holds cardinality; the directory and stream retain their format.
/// Ownership and automatic Send/Sync come from Box, without raw allocation or borrowed storage.
#[derive(Default)]
pub(super) enum Positions {
    #[default]
    Empty,
    One(u64),
    Two(u64, u64),
    Compressed(Box<[u8]>),
}
fn prefix(count: usize) -> usize {
    let groups = count.div_ceil(RESTART);
    (1 + if groups > 1 { groups } else { 0 }) * std::mem::size_of::<usize>()
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
    pub(super) fn from_sorted(input: Input<'_>, lease: &mut Lease<'_>) -> Result<Self> {
        let count = input.len()?;
        if count == 0 {
            return Ok(Self::Empty);
        }
        let (first, last) = input.bounds().expect("nonempty input has endpoints");
        if last < first {
            return Err("BPE positions are not sorted".into());
        }
        if count == 1 {
            return Ok(Self::One(first));
        }
        if count == 2 {
            return Ok(Self::Two(first, last));
        }
        let scratch = &mut *lease.scratch;
        scratch.encode_stream(&input)?;
        let capacity = prefix(count)
            .checked_add(scratch.bytes.len())
            .ok_or("BPE position layout exceeds usize")?;
        let mut bytes = Vec::with_capacity(capacity);
        bytes.extend_from_slice(&count.to_le_bytes());
        if scratch.offsets.len() > 1 {
            for &offset in &scratch.offsets {
                bytes.extend_from_slice(&offset.to_le_bytes());
            }
        }
        bytes.extend_from_slice(&scratch.bytes);
        Ok(Self::Compressed(bytes.into_boxed_slice()))
    }
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Compressed(bytes) => &bytes[prefix(self.len())..],
            _ => &[],
        }
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
    pub(super) fn block_count(&self) -> usize {
        self.len().div_ceil(RESTART)
    }

    pub(super) fn block_ranges(&self, target_items: usize) -> impl Iterator<Item = Range<usize>> {
        let blocks = target_items.div_ceil(RESTART).max(1);
        (0..self.block_count())
            .step_by(blocks)
            .map(move |begin| begin..(begin + blocks).min(self.block_count()))
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        self.read_blocks(0..self.block_count())
    }

    pub(super) fn read_blocks(&self, range: Range<usize>) -> impl Iterator<Item = u64> + '_ {
        // Reject inverted ranges, then clamp before multiplying. If a range starts
        // past the directory, start == end and no directory pointer is formed.
        assert!(range.start <= range.end);
        let blocks = self.block_count();
        let start = (range.start.min(blocks) * RESTART).min(self.len());
        let end = (range.end.min(blocks) * RESTART).min(self.len());
        if !matches!(self, Self::Compressed(_)) {
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
                &self.bytes()[self.offset(range.start)..]
            };
            itertools::Either::Right(Cursor {
                bytes,
                position: 0,
                index: start,
                end,
            })
        }
    }

    pub(super) fn lower_bound(&self, target: u64) -> usize {
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
        let block = begin.saturating_sub(1);
        block * RESTART
            + self
                .read_blocks(block..block + 1)
                .take_while(|&p| p < target)
                .count()
    }

    /// Iterates from a list index, rather than a corpus coordinate.
    pub(super) fn iter_from(&self, index: usize) -> impl Iterator<Item = u64> + '_ {
        self.read_blocks(index / RESTART..self.block_count())
            .skip(index % RESTART)
    }
}
/// A reader's private decode state over a borrowed immutable position stream.
/// Absolute restart seeds bound replay; index/end delimit the requested blocks.
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
    fn storage_roundtrips_seeks_and_supports_concurrent_readers() {
        // Freeze and seek lists across inline, chunk and full-u64 boundaries.
        let codec = Codec::new(2);
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
                let mut builder = Builder::default();
                let mut fragments = vec![Positions::default()];
                for chunk in values.chunks(17) {
                    let mut piece = Builder::default();
                    for &p in chunk {
                        piece.push(p).unwrap();
                    }
                    fragments.push(
                        Positions::from_sorted(Input::Builder(&piece), &mut codec.lease()).unwrap(),
                    );
                    builder.append(piece).unwrap();
                }
                assert!(builder.iter().eq(values.iter().copied()));
                fragments.push(Positions::default());
                let positions =
                    Positions::from_sorted(Input::Fragments(&fragments), &mut codec.lease())
                        .unwrap();
                drop(fragments);
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
                }
                assert!(
                    positions
                        .block_ranges(129)
                        .flat_map(|r| positions.read_blocks(r))
                        .eq(values.iter().copied())
                );
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
                let codec = &codec;
                scope.spawn(move || {
                    let values: Vec<_> = (0..512).map(|i| (1 << 63) + i).collect();
                    let next =
                        Positions::from_sorted(Input::Slice(&values), &mut codec.lease()).unwrap();
                    assert!(next.iter().eq(values));
                    for (positions, expected) in saved {
                        assert!(positions.iter().eq(expected.iter().copied()));
                    }
                });
            }
        });
        drop(codec);
        for (positions, expected) in saved {
            assert!(positions.iter().eq(expected));
        }
    }

    #[test]
    // The inverted range is deliberately passed as malformed decoder input.
    #[allow(clippy::reversed_empty_ranges)]
    fn storage_rejects_inverted_ranges_and_unsorted_input() {
        let codec = Codec::new(2);
        assert!(Positions::from_sorted(Input::Slice(&[u64::MAX, 0]), &mut codec.lease()).is_err());
        for (a, b) in [(&[u64::MAX][..], &[0][..]), (&[0, u64::MAX][..], &[1][..])] {
            let fragments = [a, b]
                .map(|v| Positions::from_sorted(Input::Slice(v), &mut codec.lease()).unwrap());
            assert!(
                Positions::from_sorted(Input::Fragments(&fragments), &mut codec.lease()).is_err()
            );
        }
        let mut builder = Builder::default();
        builder.push(u64::MAX).unwrap();
        assert!(builder.push(0).is_err());
        let values: Vec<_> = (0..129u64).collect();
        let positions = Positions::from_sorted(Input::Slice(&values), &mut codec.lease()).unwrap();
        let call = std::panic::AssertUnwindSafe(|| positions.read_blocks(1000..0).next());
        assert!(std::panic::catch_unwind(call).is_err());
    }
}
