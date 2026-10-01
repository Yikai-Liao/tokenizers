//! Eight inline payload bytes: two u32 or four u16 local offsets.
//! Raw-part reconstruction is private to this module.

use super::posting_arena;
use std::mem::{self, ManuallyDrop};
use std::slice;
use tk_encode::Result;

#[repr(C)]
union Payload<T: Copy, const INLINE: usize> {
    inline: [T; INLINE],
    heap: *mut T,
}

pub(super) type SmallPosting = PackedPosting<u32, 2>;

#[repr(C)]
pub(super) struct PackedPosting<T: Copy + Default, const INLINE: usize> {
    len: u32,
    // Zero tags inline mode; otherwise the full u32 capacity is preserved.
    // For align >= 2, the pointer low bit tags arena ownership.
    capacity: u32,
    payload: Payload<T, INLINE>,
}

// SAFETY: every allocation has one posting owner; an arena session outlives
// all posting owners and joins its dedicated worker pools. Mutation requires
// &mut self, and shared slices contain only initialized Copy values.
unsafe impl<T: Copy + Default + Send, const INLINE: usize> Send for PackedPosting<T, INLINE> {}
// SAFETY: immutable access never mutates the allocation or the union tag.
unsafe impl<T: Copy + Default + Sync, const INLINE: usize> Sync for PackedPosting<T, INLINE> {}

impl<T: Copy + Default, const INLINE: usize> Default for PackedPosting<T, INLINE> {
    fn default() -> Self {
        Self {
            len: 0,
            capacity: 0,
            payload: Payload {
                inline: [T::default(); INLINE],
            },
        }
    }
}

impl<T: Copy + Default, const INLINE: usize> PackedPosting<T, INLINE> {
    /// Reserve for a freshly counted key. A heap posting may initially be empty:
    /// its allocation is owned even while no position has been initialized.
    pub(super) fn with_capacity(count: u32) -> Result<Self> {
        Self::reserve(count, false)
    }

    fn reserve(count: u32, growth: bool) -> Result<Self> {
        if count as usize <= INLINE {
            return Ok(Self::default());
        }
        let mut posting = Self::default();
        let capacity = (count as usize).max(INLINE * 2);
        let checked = u32::try_from(capacity).map_err(|_| "posting capacity exceeds u32")?;
        if let Some(pointer) = posting_arena::allocate::<T>(capacity, growth)? {
            posting.payload = Payload {
                heap: pointer.as_ptr().map_addr(|addr| addr | 1),
            };
            posting.capacity = checked;
        } else {
            posting.install_heap(Vec::with_capacity(capacity), growth)?;
        }
        Ok(posting)
    }

    #[inline]
    fn is_arena(&self) -> bool {
        !self.is_inline() && mem::align_of::<T>() >= 2
            // SAFETY: capacity != 0 identifies the active pointer field.
            && unsafe { self.payload.heap.addr() & 1 != 0 }
    }

    #[inline]
    fn pointer(&self) -> *mut T {
        debug_assert!(!self.is_inline());
        // SAFETY: capacity != 0 identifies the active pointer field. map_addr
        // preserves provenance; align-1 pointers may legally have an odd address.
        let pointer = unsafe { self.payload.heap };
        if mem::align_of::<T>() >= 2 {
            pointer.map_addr(|addr| addr & !1)
        } else {
            pointer
        }
    }

    #[inline]
    pub(super) fn len(&self) -> usize {
        self.len as usize
    }

    #[inline]
    #[cfg(test)]
    pub(super) fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub(super) fn is_inline(&self) -> bool {
        self.capacity == 0
    }

    /// Physical heap slots. Inline slots are already part of the map entry.
    #[inline]
    pub(super) fn allocated_capacity(&self) -> usize {
        if self.is_inline() {
            0
        } else {
            self.capacity as usize
        }
    }

    #[inline]
    pub(super) fn as_slice(&self) -> &[T] {
        if self.is_inline() {
            debug_assert!(self.len as usize <= INLINE);
            // SAFETY: inline is the active union field and all elements were
            // initialized by Default. Only the first len are exposed.
            let inline = unsafe { &self.payload.inline };
            &inline[..self.len()]
        } else {
            debug_assert!(self.len <= self.capacity && self.capacity as usize >= INLINE * 2);
            // SAFETY: pointer refers to a Vec or session-owned arena allocation of capacity at
            // least len; its first len elements were initialized by push.
            unsafe { slice::from_raw_parts(self.pointer(), self.len()) }
        }
    }

    #[inline]
    pub(super) fn as_mut_slice(&mut self) -> &mut [T] {
        if self.is_inline() {
            debug_assert!(self.len as usize <= INLINE);
            let len = self.len();
            // SAFETY: inline is active and initialized; &mut self is unique.
            let inline = unsafe { &mut self.payload.inline };
            &mut inline[..len]
        } else {
            debug_assert!(self.len <= self.capacity && self.capacity as usize >= INLINE * 2);
            // SAFETY: the allocation is uniquely owned and first len items
            // initialized; &mut self excludes other aliases.
            unsafe { slice::from_raw_parts_mut(self.pointer(), self.len()) }
        }
    }

    /// Move the posting out and leave an empty inline posting behind.
    #[cfg(test)]
    pub(super) fn take(&mut self) -> Self {
        mem::take(self)
    }

    fn install_heap(&mut self, vec: Vec<T>, growth: bool) -> Result<()> {
        // Both callers either still hold inline data or used mem::take first.
        debug_assert!(self.is_inline());
        let len = u32::try_from(vec.len()).map_err(|_| "posting length exceeds u32")?;
        let capacity = u32::try_from(vec.capacity()).map_err(|_| "posting capacity exceeds u32")?;
        if (capacity as usize) < INLINE * 2 || len > capacity {
            return Err("invalid heap posting Vec".into());
        }
        // Conversion cannot fail beyond this point. Suppress Vec's destructor
        // only after checking both fields, then transfer its allocation.
        posting_arena::heap_allocation::<T>(vec.capacity(), growth);
        let mut vec = ManuallyDrop::new(vec);
        self.payload = Payload {
            heap: vec.as_mut_ptr(),
        };
        self.len = len;
        self.capacity = capacity;
        Ok(())
    }

    /// Fill a pre-reserved suffix backwards. The producer visits values in
    /// reverse order; its final physical suffix is forward ordered. Length is
    /// published once after every new element has been initialized. A producer
    /// panic leaves the old prefix valid and the allocation uniquely owned.
    #[inline]
    pub(super) fn append_reversed_reserved(
        &mut self,
        count: u32,
        mut next: impl FnMut() -> T,
    ) -> Result<()> {
        let start = self.len;
        let end = start
            .checked_add(count)
            .ok_or("posting length exceeds u32")?;
        if self.is_inline() {
            if end as usize > INLINE {
                return Err("inline posting bulk fill exceeds reserved capacity".into());
            }
            for i in (start as usize..end as usize).rev() {
                // SAFETY: inline is active, fully initialized by Default,
                // and the checked final end is within its fixed capacity.
                unsafe {
                    self.payload.inline[i] = next();
                }
            }
        } else {
            if end > self.capacity {
                return Err("heap posting bulk fill exceeds reserved capacity".into());
            }
            for i in (start as usize..end as usize).rev() {
                let value = next();
                // SAFETY: heap owns the allocation and end <= capacity. Each
                // new suffix slot is written exactly once. If next panics, len
                // is still start; Copy T has no destructor for written extras.
                unsafe {
                    self.pointer().add(i).write(value);
                }
            }
        }
        self.len = end;
        Ok(())
    }

    pub(super) fn push(&mut self, pos: T) -> Result<()> {
        let next_len = self
            .len
            .checked_add(1)
            .ok_or("posting length exceeds u32")?;
        if self.is_inline() {
            if (self.len as usize) < INLINE {
                // SAFETY: active inline array is initialized; index < INLINE.
                unsafe {
                    self.payload.inline[self.len as usize] = pos;
                }
                self.len = next_len;
                return Ok(());
            }
            let mut replacement = Self::reserve(next_len, true)?;
            replacement.copy_reserved_prefix(self.as_slice());
            replacement.push(pos)?;
            *self = replacement;
            return Ok(());
        }
        if self.len < self.capacity {
            // SAFETY: the owned allocation has len < capacity, leaving one spare
            // uninitialized slot and &mut self is its unique owner.
            unsafe {
                self.pointer().add(self.len as usize).write(pos);
            }
            self.len = next_len;
            return Ok(());
        }
        let requested = (self.capacity as usize)
            .saturating_mul(2)
            .min(u32::MAX as usize)
            .max(next_len as usize);
        if self.is_arena() || posting_arena::eligible::<T>(requested) {
            let mut replacement = Self::reserve(
                u32::try_from(requested).map_err(|_| "posting capacity exceeds u32")?,
                true,
            )?;
            replacement.copy_reserved_prefix(self.as_slice());
            replacement.push(pos)?;
            *self = replacement;
            return Ok(());
        }
        // The destination is empty before a Vec can reallocate or unwind.
        // The old raw pointer is owned only by the reconstructed Vec below.
        let old = ManuallyDrop::new(mem::take(self));
        // SAFETY: old is heap mode, and ptr/len/cap are the exact raw parts
        // recorded when its unique Vec allocation was last installed.
        let mut vec =
            unsafe { Vec::from_raw_parts(old.pointer(), old.len as usize, old.capacity as usize) };
        posting_arena::retirement::<T>(old.capacity as usize, false);
        vec.reserve_exact(requested - vec.len());
        vec.push(pos);
        self.install_heap(vec, true)
    }
    fn copy_reserved_prefix(&mut self, prefix: &[T]) {
        debug_assert!(!self.is_inline() && prefix.len() <= self.capacity as usize);
        // SAFETY: destination is a fresh nonoverlapping allocation with enough
        // capacity, and only the initialized Copy prefix is published.
        unsafe {
            self.pointer()
                .copy_from_nonoverlapping(prefix.as_ptr(), prefix.len());
        }
        self.len = prefix.len() as u32;
    }
}

impl<T: Copy + Default, const INLINE: usize> Drop for PackedPosting<T, INLINE> {
    fn drop(&mut self) {
        if self.capacity != 0 {
            let arena = self.is_arena();
            posting_arena::retirement::<T>(self.capacity as usize, arena);
            if arena {
                return;
            }
            // SAFETY: heap mode uniquely owns a Vec allocation whose exact
            // raw pointer, initialized length, and capacity are stored here.
            unsafe {
                drop(Vec::from_raw_parts(
                    self.pointer(),
                    self.len as usize,
                    self.capacity as usize,
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_growth_migration_origin_and_fallback() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let init = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let session = posting_arena::Session::new(&pool, Some(&init));
        posting_arena::configure(&pool, Some(&init), 32);
        let posting = init.install(|| {
            let mut posting = SmallPosting::default();
            for value in 0..8 {
                posting.push(value).unwrap();
            }
            assert!(posting.is_arena());
            posting
        });
        let mut posting = pool.install(move || {
            let mut posting = posting;
            for value in 8..17 {
                posting.push(value).unwrap();
            }
            assert!(!posting.is_arena());
            assert_eq!(posting.as_slice(), (0..17).collect::<Vec<_>>());
            posting
        });
        // The policy changes only here as a test of origin independence. A real
        // training computes one immutable threshold before allocating postings.
        posting_arena::configure(&pool, Some(&init), 256);
        pool.install(move || {
            for value in 17..33 {
                posting.push(value).unwrap();
            }
            assert!(posting.is_arena());
            posting.as_mut_slice().reverse();
            assert_eq!(posting.as_slice()[0], 32);
            let mut bytes = PackedPosting::<u8, 8>::default();
            for value in 0..64 {
                bytes.push(value).unwrap();
            }
            assert!(!bytes.is_arena());
            assert_eq!(bytes.as_slice(), (0..64).collect::<Vec<_>>());
            assert!(PackedPosting::<(), 2>::with_capacity(3).is_err());
            drop(posting);
        });
        let stats = session.finish();
        assert!(stats.arena_buffers >= 3 && stats.heap_buffers >= 1);
        assert_eq!(stats.arena_requested_bytes, stats.arena_retired_bytes);
        assert_eq!(stats.heap_requested_bytes, stats.heap_freed_bytes);
        assert_eq!(stats.heap_buffers, stats.heap_frees);
        assert!(stats.grows > 0 && stats.backing_bytes >= stats.arena_requested_bytes);
    }

    #[test]
    fn reserved_reverse_fill_preserves_prefix_inline_heap_and_unwind() {
        let mut inline = SmallPosting::with_capacity(2).unwrap();
        let mut next = [9_u32, 3].into_iter();
        inline
            .append_reversed_reserved(2, || next.next().unwrap())
            .unwrap();
        assert_eq!(inline.as_slice(), &[3, 9]);
        assert!(inline.append_reversed_reserved(1, || 17).is_err());
        assert_eq!(inline.as_slice(), &[3, 9]);
        inline
            .append_reversed_reserved(0, || panic!("empty fill called producer"))
            .unwrap();
        let mut heap = SmallPosting::with_capacity(8).unwrap();
        heap.push(1).unwrap();
        let old_capacity = heap.allocated_capacity();
        let mut reversed = [7_u32, 5, 3].into_iter();
        heap.append_reversed_reserved(3, || reversed.next().unwrap())
            .unwrap();
        assert_eq!(heap.as_slice(), &[1, 3, 5, 7]);
        assert_eq!(heap.allocated_capacity(), old_capacity);
        let mut calls = 0;
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                heap.append_reversed_reserved(3, || {
                    calls += 1;
                    if calls == 2 {
                        panic!("interrupted producer");
                    }
                    19
                })
                .unwrap();
            }))
            .is_err()
        );
        assert_eq!(heap.as_slice(), &[1, 3, 5, 7]);
        let mut reversed = [13_u32, 11, 9].into_iter();
        heap.append_reversed_reserved(3, || reversed.next().unwrap())
            .unwrap();
        assert_eq!(heap.as_slice(), &[1, 3, 5, 7, 9, 11, 13]);
        let mut short = PackedPosting::<u16, 4>::with_capacity(4).unwrap();
        let mut reversed = [u16::MAX, 19, 7, 1].into_iter();
        short
            .append_reversed_reserved(4, || reversed.next().unwrap())
            .unwrap();
        assert_eq!(short.as_slice(), &[1, 7, 19, u16::MAX]);
    }

    #[test]
    fn u16_offsets_inline_four_heap_growth_and_full_range() {
        let mut posting = PackedPosting::<u16, 4>::default();
        assert_eq!(mem::size_of_val(&posting), 16);
        for value in [0, 1, u16::MAX, 37] {
            posting.push(value).unwrap();
        }
        assert!(posting.is_inline());
        posting.push(65534).unwrap();
        assert!(!posting.is_inline());
        assert_eq!(posting.as_slice(), &[0, 1, 65535, 37, 65534]);
        for value in 0..1000 {
            posting.push(value).unwrap();
        }
        let moved = posting.take();
        assert_eq!(moved.len(), 1005);
        assert!(posting.is_empty());
    }

    #[test]
    fn counted_reservation_empty_heap_fill_growth_and_take() {
        for count in [0_u32, 1, 2, 3, 17] {
            let mut posting = SmallPosting::with_capacity(count).unwrap();
            assert!(posting.is_empty());
            assert!(posting.as_slice().is_empty());
            assert!(posting.as_mut_slice().is_empty());
            assert_eq!(posting.is_inline(), count <= 2);
            if count > 2 {
                assert!(posting.allocated_capacity() >= count as usize);
            }
            for pos in 0..count {
                posting.push(pos).unwrap();
            }
            assert_eq!(
                posting.as_slice(),
                (0..count).collect::<Vec<_>>().as_slice()
            );
            if count > 2 {
                let reserved = posting.allocated_capacity();
                for pos in count..reserved as u32 {
                    posting.push(pos).unwrap();
                }
                assert_eq!(posting.allocated_capacity(), reserved);
                posting.push(u32::MAX).unwrap();
                assert_eq!(posting.len(), reserved + 1);
                assert_eq!(posting.as_slice()[reserved], u32::MAX);
            }
            let moved = posting.take();
            assert!(posting.is_inline());
            assert!(posting.is_empty());
            assert!(!moved.as_slice().is_empty() || count == 0);
            drop(moved);
        }
        // Empty heap mode must still free the reserved Vec on drop.
        drop(SmallPosting::with_capacity(17).unwrap());
    }

    #[test]
    fn position_payload_uses_all_u32_bits_across_inline_heap_and_take() {
        let values = [0, u32::MAX, 1_u32 << 31];
        let mut posting = SmallPosting::default();
        posting.push(values[0]).unwrap();
        posting.push(values[1]).unwrap();
        assert_eq!(posting.as_slice(), &values[..2]);
        posting.push(values[2]).unwrap();
        let moved = posting.take();
        assert_eq!(moved.as_slice(), values);
        assert!(posting.is_inline() && posting.is_empty());
    }

    #[test]
    fn layout_inline_growth_move_and_drop() {
        #[cfg(target_pointer_width = "64")]
        {
            assert_eq!(mem::size_of::<SmallPosting>(), 16);
            assert_eq!(mem::align_of::<SmallPosting>(), 8);
        }
        let mut posting = SmallPosting::default();
        assert_eq!(posting.len(), 0);
        assert!(posting.as_slice().is_empty());
        assert_eq!(posting.allocated_capacity(), 0);
        for n in 0..2 {
            posting.push(n).unwrap();
            assert!(posting.is_inline());
            assert_eq!(posting.as_slice(), (0..=n).collect::<Vec<_>>().as_slice());
        }
        posting.push(2).unwrap();
        assert!(!posting.is_inline());
        assert_eq!(posting.as_slice(), [0, 1, 2]);
        let mut moved = posting.take();
        assert_eq!(posting.len(), 0);
        for n in 3..1000 {
            moved.push(n).unwrap();
        }
        assert_eq!(moved.as_slice(), (0..1000).collect::<Vec<_>>().as_slice());
        moved.as_mut_slice().reverse();
        assert_eq!(moved.as_slice()[0], 999);
        drop(moved);
        posting.push(77).unwrap();
        assert_eq!(posting.as_slice(), [77]);
        let one = posting.take();
        posting.push(88).unwrap();
        assert_eq!(one.as_slice(), [77]);
        assert_eq!(posting.as_slice(), [88]);
        drop(one);
        posting.push(99).unwrap();
        let two = posting.take();
        assert_eq!(two.as_slice(), [88, 99]);
        assert_eq!(posting.len(), 0);
        drop(two);
    }

    #[test]
    fn grown_posting_can_move_between_threads() {
        let mut posting = SmallPosting::default();
        for n in 0..128 {
            posting.push(n).unwrap();
        }
        let posting = std::thread::spawn(move || {
            let mut posting = posting;
            for n in 128..512 {
                posting.push(n).unwrap();
            }
            assert_eq!(posting.as_slice()[511], 511);
            posting
        })
        .join()
        .unwrap();
        std::thread::scope(|scope| {
            scope
                .spawn(|| assert_eq!(posting.as_slice()[0], 0))
                .join()
                .unwrap();
        });
        drop(posting);
    }
}
