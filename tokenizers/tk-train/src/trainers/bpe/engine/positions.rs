//! Frozen occurrence lists with inline pairs and Arena or owned allocations.
use bumpalo::Bump;
use std::{
    alloc::{Layout, alloc, dealloc},
    marker::PhantomData,
    ops::Range,
    sync::{Mutex, MutexGuard},
};
use tk_encode::Result;
const RESTART: usize = 128;
const INLINE: usize = 1 << (usize::BITS - 1);
const PAIR: usize = 1 << (usize::BITS - 2);
const DELTA_MASK: usize = PAIR - 1;
const ARENA: usize = 1;

// Final allocation cursors are leased within sequential worker closures.
// Published storage survives leases and is never reset during the attempt.
pub(super) struct Arena {
    workers: Vec<Mutex<Worker>>,
    cutoff: usize,
}
impl Arena {
    pub(super) fn new(workers: usize, items: usize) -> Self {
        Self {
            workers: (0..workers).map(|_| Mutex::default()).collect(),
            cutoff: ((items as u128 / 256).isqrt() as usize).max(256),
        }
    }
    pub(super) fn lease(&self) -> Lease<'_> {
        // A worker cursor is non-reentrant. Drop the lease before starting any
        // nested Rayon work: another task on this worker may request the same lock.
        Lease {
            arena: self,
            cursor: self.workers[rayon::current_thread_index().unwrap_or(0) % self.workers.len()]
                .lock()
                .unwrap_or_else(|e| e.into_inner()),
        }
    }
}
#[derive(Default)]
struct Worker {
    bump: Bump,
    bytes: Vec<u8>,
    offsets: Vec<usize>,
}
pub(super) struct Lease<'arena> {
    arena: &'arena Arena,
    cursor: MutexGuard<'arena, Worker>,
}
// Mutable task buffers keep four-byte coordinates until a full-u64 value appears.
// Small buffers stay inline; promotion preserves every previously stored value.
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
        self.push_ordered(position)
    }
    // Snapshot scans visit disjoint ordered matches; callers retain that order.
    // Generic append/input paths use `push` and keep its validation.
    pub(super) fn push_ordered(&mut self, position: u64) -> Result<()> {
        if let Self::Narrow(values) = self {
            if let Ok(position) = u32::try_from(position) {
                values.push(position);
                return Ok(());
            }
            *self = Self::Wide(values.iter().map(|&p| u64::from(p)).collect());
        }
        if let Self::Wide(values) = self {
            values.push(position);
        }
        Ok(())
    }
    pub(super) fn append(&mut self, mut other: Self) -> Result<()> {
        if self.is_empty() {
            std::mem::swap(self, &mut other);
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
// Only concrete storage views reach the allocator. Unlike an arbitrary safe
// ExactSizeIterator implementation, these views have a trustworthy cardinality.
pub(super) enum Input<'a> {
    Builder(&'a Builder),
    Slice(&'a [u64]),
    Fragments(&'a [Positions<'a>]),
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
                    .find_map(|p| p.from(p.len() - 1).next())?,
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
// Inline lists store the first coordinate in payload and the gap in count bits.
// Allocated lists store their length and tag the pointer's low bit for Arena ownership.
#[derive(Default)]
pub(super) struct Positions<'arena> {
    count_and_flags: usize,
    payload: *mut u8,
    arena_lifetime: PhantomData<&'arena Arena>,
}
// SAFETY: lists own their final heap storage or borrow stable Arena
// storage. Moving a list does not move its allocation; Arena is Send + Sync.
unsafe impl Send for Positions<'_> {}
// SAFETY: mutation requires &mut self; borrowed readers own decoder state.
// All initialized payloads stay live for the list or its training Arena.
unsafe impl Sync for Positions<'_> {}
fn prefix(count: usize) -> usize {
    let groups = count.div_ceil(RESTART);
    (1 + if groups > 1 { groups } else { 0 }) * std::mem::size_of::<usize>()
}
fn layout(count: usize, bytes: usize) -> Result<Layout> {
    Layout::from_size_align(
        prefix(count)
            .checked_add(bytes)
            .ok_or("BPE position layout exceeds usize")?,
        8,
    )
    .map_err(|_| "BPE position allocation exceeds resident bounds".into())
}
impl<'arena> Positions<'arena> {
    #[inline]
    pub(super) fn len(&self) -> usize {
        if self.count_and_flags & INLINE == 0 {
            self.count_and_flags
        } else {
            1 + usize::from(self.count_and_flags & PAIR != 0)
        }
    }
    pub(super) fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn pointer(&self) -> *mut u8 {
        self.payload.map_addr(|a| a & !1)
    }
    pub(super) fn from_sorted(input: Input<'_>, lease: &mut Lease<'arena>) -> Result<Self> {
        Self::encode(input, lease, false)
    }
    pub(super) fn from_sorted_owned(input: Input<'_>, lease: &mut Lease<'arena>) -> Result<Self> {
        Self::encode(input, lease, true)
    }
    fn encode(input: Input<'_>, lease: &mut Lease<'arena>, owned: bool) -> Result<Self> {
        let count = input.len()?;
        let values = input.iter();
        if count == 0 {
            return Ok(Self::default());
        }
        if count >= INLINE {
            return Err("BPE position count exceeds resident bounds".into());
        }
        let (first, last) = input
            .bounds()
            .expect("nonempty trusted input has endpoints");
        let gap = last
            .checked_sub(first)
            .ok_or("BPE positions are not sorted")?;
        if count <= 2 && first <= usize::MAX as u64 && (count == 1 || gap <= DELTA_MASK as u64) {
            return Ok(Self {
                count_and_flags: INLINE | if count == 2 { PAIR | gap as usize } else { 0 },
                payload: std::ptr::without_provenance_mut(first as usize),
                arena_lifetime: PhantomData,
            });
        }
        let worker = &mut *lease.cursor;
        worker.bytes.clear();
        worker.offsets.clear();
        let mut previous = 0;
        for (i, position) in values.enumerate() {
            let gap = position
                .checked_sub(previous)
                .ok_or("BPE positions are not sorted")?;
            if i.is_multiple_of(RESTART) {
                worker.offsets.push(worker.bytes.len());
                worker.bytes.extend_from_slice(&position.to_le_bytes());
            } else {
                let mut delta = gap;
                while delta >= 128 {
                    worker.bytes.push((delta as u8 & 127) | 128);
                    delta >>= 7;
                }
                worker.bytes.push(delta as u8);
            }
            previous = position;
        }
        let allocation = layout(count, worker.bytes.len())?;
        let arena = !owned && allocation.size() <= lease.arena.cutoff;
        let pointer = if arena {
            worker
                .bump
                .try_alloc_layout(allocation)
                .map_err(|_| "BPE position arena allocation failed")?
                .as_ptr()
        } else {
            // SAFETY: layout is nonzero and validated; null is handled below.
            unsafe { alloc(allocation) }
        };
        if pointer.is_null() {
            return Err("BPE position allocation failed".into());
        }
        // SAFETY: final layout reserves an aligned length word, optional full
        // restart directory and stream. All bytes read later are initialized here.
        unsafe {
            pointer.cast::<usize>().write(worker.bytes.len());
            if worker.offsets.len() > 1 {
                std::ptr::copy_nonoverlapping(
                    worker.offsets.as_ptr(),
                    pointer.cast::<usize>().add(1),
                    worker.offsets.len(),
                );
            }
            std::ptr::copy_nonoverlapping(
                worker.bytes.as_ptr(),
                pointer.add(prefix(count)),
                worker.bytes.len(),
            );
        }
        // Arena publication borrows the attempt,
        // independently of this cursor lease; no reset or early free is exposed.
        Ok(Self {
            count_and_flags: count,
            payload: pointer.map_addr(|a| a | usize::from(arena)),
            arena_lifetime: PhantomData,
        })
    }
    fn bytes(&self) -> &[u8] {
        // SAFETY: only final-storage readers call this. The prefix is initialized
        // and its stream length/layout were validated at freeze publication.
        unsafe {
            std::slice::from_raw_parts(
                self.pointer().add(prefix(self.len())),
                self.pointer().cast::<usize>().read(),
            )
        }
    }
    fn offset(&self, block: usize) -> usize {
        if block == 0 {
            0
        } else {
            // SAFETY: multi-block final layouts initialize this directory entry.
            unsafe { self.pointer().cast::<usize>().add(1 + block).read() }
        }
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
        if self.is_empty() || self.count_and_flags & INLINE != 0 {
            let first = self.payload.addr() as u64;
            itertools::Either::Left(
                [first, first + (self.count_and_flags & DELTA_MASK) as u64]
                    .into_iter()
                    .take(end)
                    .skip(start),
            )
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
    pub(super) fn from(&self, index: usize) -> impl Iterator<Item = u64> + '_ {
        self.read_blocks(index / RESTART..self.block_count())
            .skip(index % RESTART)
    }
}
impl Drop for Positions<'_> {
    fn drop(&mut self) {
        if self.is_empty() || self.count_and_flags & INLINE != 0 {
            return;
        }
        // SAFETY: the tag distinguishes owned final allocation and
        // borrowed Arena. Each owned allocation is reclaimed once with its layout.
        unsafe {
            if self.payload.addr() & ARENA == 0 {
                let allocation = layout(self.len(), self.pointer().cast::<usize>().read())
                    .expect("published layout was validated");
                dealloc(self.pointer(), allocation);
            }
        }
    }
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
    fn storage_roundtrips_seeks_and_survives_concurrent_cursor_reuse() {
        assert_eq!(
            std::mem::size_of::<Positions<'_>>(),
            2 * std::mem::size_of::<usize>()
        );
        let arena = Arena::new(2, 0);
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
                        Positions::from_sorted_owned(Input::Builder(&piece), &mut arena.lease())
                            .unwrap(),
                    );
                    builder.append(piece).unwrap();
                }
                assert!(builder.iter().eq(values.iter().copied()));
                fragments.push(Positions::default());
                let positions =
                    Positions::from_sorted(Input::Fragments(&fragments), &mut arena.lease())
                        .unwrap();
                drop(fragments);
                assert!(positions.iter().eq(values.iter().copied()));
                for index in (0..=length)
                    .step_by(if cfg!(miri) { 127 } else { 1 })
                    .chain([length])
                {
                    assert!(positions.from(index).eq(values[index..].iter().copied()));
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
        // Both worker cursors allocate while published Arena and heap lists are
        // shared with other threads. No Rayon collector is involved in Miri.
        std::thread::scope(|scope| {
            for worker in 0..2 {
                let arena = &arena;
                let saved = &saved;
                scope.spawn(move || {
                    let mut lease = Lease {
                        arena,
                        cursor: arena.workers[worker].lock().unwrap(),
                    };
                    let values: Vec<_> = (0..512).map(|i| (1 << 63) + i).collect();
                    let next = Positions::from_sorted(Input::Slice(&values), &mut lease).unwrap();
                    assert!(next.iter().eq(values));
                    for (positions, expected) in saved {
                        assert!(positions.iter().eq(expected.iter().copied()));
                    }
                });
            }
        });
    }

    #[test]
    // The inverted range is deliberately passed as malformed decoder input.
    #[allow(clippy::reversed_empty_ranges)]
    fn unsafe_storage_rejects_inverted_ranges_and_unsorted_input() {
        let arena = Arena::new(1, 0);
        let mut lease = arena.lease();
        assert!(Positions::from_sorted(Input::Slice(&[u64::MAX, 0]), &mut lease).is_err());
        for (a, b) in [(&[u64::MAX][..], &[0][..]), (&[0, u64::MAX][..], &[1][..])] {
            let fragments =
                [a, b].map(|v| Positions::from_sorted_owned(Input::Slice(v), &mut lease).unwrap());
            assert!(Positions::from_sorted(Input::Fragments(&fragments), &mut lease).is_err());
        }
        let mut builder = Builder::default();
        builder.push(u64::MAX).unwrap();
        assert!(builder.push(0).is_err());
        let values: Vec<_> = (0..129u64).collect();
        let positions = Positions::from_sorted(Input::Slice(&values), &mut lease).unwrap();
        let call = std::panic::AssertUnwindSafe(|| positions.read_blocks(1000..0).next());
        assert!(std::panic::catch_unwind(call).is_err());
    }
}
