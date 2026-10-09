//! Main's 16-byte list descriptor: inline pairs, scoped Arena or owned payloads.
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
    workers: Vec<Mutex<Bump>>,
    cutoff: usize,
}
impl Arena {
    pub(super) fn new(workers: usize, items: usize) -> Self {
        Self {
            workers: (0..workers).map(|_| Mutex::new(Bump::new())).collect(),
            cutoff: ((items as u128 / 256).isqrt() as usize).max(256),
        }
    }
    pub(super) fn lease(&self) -> Lease<'_> {
        Lease {
            arena: self,
            cursor: self.workers[rayon::current_thread_index().unwrap_or(0) % self.workers.len()]
                .lock()
                .unwrap_or_else(|e| e.into_inner()),
            bytes: Vec::new(),
            offsets: Vec::new(),
        }
    }
}
pub(super) struct Lease<'arena> {
    arena: &'arena Arena,
    cursor: MutexGuard<'arena, Bump>,
    bytes: Vec<u8>,
    offsets: Vec<usize>,
}
// Mutable task buffers remain separate from the published 16-byte descriptor.
// Two inline coordinates avoid allocating the common tiny neighbor lists.
#[derive(Default)]
pub(super) struct Builder(smallvec::SmallVec<[u64; 2]>);
impl std::ops::Deref for Builder {
    type Target = [u64];
    fn deref(&self) -> &[u64] {
        &self.0
    }
}
impl Builder {
    pub(super) fn push(&mut self, position: u64) -> Result<()> {
        if self.last().is_some_and(|&last| position < last) {
            return Err("BPE positions are not sorted".into());
        }
        self.0.push(position);
        Ok(())
    }
    pub(super) fn append(&mut self, mut other: Self) -> Result<()> {
        if let (Some(&last), Some(&first)) = (self.last(), other.first())
            && first < last
        {
            return Err("BPE positions are not sorted".into());
        }
        self.0.append(&mut other.0);
        Ok(())
    }
}
// These fields and inline flags are taken from main's SortedPositions.
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
    pub(super) fn from_sorted(values: &[u64], lease: &mut Lease<'arena>) -> Result<Self> {
        let count = values.len();
        if count == 0 {
            return Ok(Self::default());
        }
        if count >= INLINE {
            return Err("BPE position count exceeds resident bounds".into());
        }
        let first = values[0];
        let gap = values[count - 1]
            .checked_sub(first)
            .ok_or("BPE positions are not sorted")?;
        if count <= 2 && first <= usize::MAX as u64 && (count == 1 || gap <= DELTA_MASK as u64) {
            return Ok(Self {
                count_and_flags: INLINE | if count == 2 { PAIR | gap as usize } else { 0 },
                payload: std::ptr::without_provenance_mut(first as usize),
                arena_lifetime: PhantomData,
            });
        }
        lease.bytes.clear();
        lease.offsets.clear();
        let mut previous = 0;
        for (i, &position) in values.iter().enumerate() {
            let gap = position
                .checked_sub(previous)
                .ok_or("BPE positions are not sorted")?;
            if i.is_multiple_of(RESTART) {
                lease.offsets.push(lease.bytes.len());
                lease.bytes.extend_from_slice(&position.to_le_bytes());
            } else {
                let mut delta = gap;
                while delta >= 128 {
                    lease.bytes.push((delta as u8 & 127) | 128);
                    delta >>= 7;
                }
                lease.bytes.push(delta as u8);
            }
            previous = position;
        }
        let allocation = layout(count, lease.bytes.len())?;
        let arena = allocation.size() <= lease.arena.cutoff;
        let pointer = if arena {
            lease
                .cursor
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
            pointer.cast::<usize>().write(lease.bytes.len());
            if lease.offsets.len() > 1 {
                std::ptr::copy_nonoverlapping(
                    lease.offsets.as_ptr(),
                    pointer.cast::<usize>().add(1),
                    lease.offsets.len(),
                );
            }
            std::ptr::copy_nonoverlapping(
                lease.bytes.as_ptr(),
                pointer.add(prefix(count)),
                lease.bytes.len(),
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
    pub(super) fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        self.read_blocks(0..self.block_count())
    }
    pub(super) fn read_blocks(&self, range: Range<usize>) -> impl Iterator<Item = u64> + '_ {
        let start = (range.start * RESTART).min(self.len());
        let end = (range.end * RESTART).min(self.len());
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
    fn compact_final_lists_survive_leases_and_parallel_allocations() {
        use rayon::prelude::*;
        assert_eq!(
            std::mem::size_of::<Positions<'_>>(),
            2 * std::mem::size_of::<usize>()
        );
        let arena = Arena::new(2, 0);
        let saved: Vec<_> = [0, 1, 2, 3, 127, 128, 129, 257, 1024]
            .into_iter()
            .map(|length| {
                let values: Vec<_> = (0..length).map(|i| (1u64 << 32) + (i / 3) as u64).collect();
                let positions = Positions::from_sorted(&values, &mut arena.lease()).unwrap();
                (positions, values)
            })
            .collect();
        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap()
            .install(|| {
                saved.par_iter().for_each(|(positions, values)| {
                    let next: Vec<_> = (0..128).map(|i| (1u64 << 63) + i).collect();
                    let temporary = Positions::from_sorted(&next, &mut arena.lease()).unwrap();
                    assert!(temporary.iter().eq(next));
                    assert!(positions.iter().eq(values.iter().copied()));
                });
            });
    }
    #[test]
    fn fragmented_streams_preserve_order_seeks_and_push() {
        let arena = Arena::new(1, 0);
        let mut lease = arena.lease();
        let mut builder = Builder::default();
        let mut values = Vec::new();
        for length in std::iter::repeat_n(1, 260).chain([0, 7, 127, 2, 128, 129, 17]) {
            let begin = values.last().copied().unwrap_or(1u64 << 32);
            let fragment: Vec<_> = (0..length).map(|i| begin + (i / 3) as u64).collect();
            let mut fragment_builder = Builder::default();
            for &position in &fragment {
                fragment_builder.push(position).unwrap();
            }
            builder.append(fragment_builder).unwrap();
            values.extend(fragment);
        }
        for value in [1u64 << 63, u64::MAX - 1, u64::MAX] {
            builder.push(value).unwrap();
            values.push(value);
        }
        builder.push(u64::MAX).unwrap();
        values.push(u64::MAX);
        let positions = Positions::from_sorted(&builder, &mut lease).unwrap();
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
        assert!(builder.push(0).is_err());
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
        let arena = Arena::new(1, 0);
        let mut lease = arena.lease();
        for length in [0, 1, 2, 127, 128, 129, 256, 257] {
            let repeated: Vec<_> = (0..length).map(|i| values[i % values.len()]).collect();
            let mut sorted = repeated;
            sorted.sort_unstable();
            let positions = Positions::from_sorted(&sorted, &mut lease).unwrap();
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
        assert!(Positions::from_sorted(&[u64::MAX, 0], &mut lease).is_err());
        values.resize(129, u64::MAX);
        values[128] = 0;
        assert!(Positions::from_sorted(&values, &mut lease).is_err());
    }
}
