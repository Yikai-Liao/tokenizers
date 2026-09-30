//! Eight inline payload bytes: two u32 or four u16 local offsets.
//! Raw-part reconstruction is private to this module.

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
    // Zero tags inline mode. Heap mode records Vec's actual allocation capacity.
    capacity: u32,
    payload: Payload<T, INLINE>,
}

// SAFETY: every heap allocation has one SmallPosting owner; mutation requires
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
        if count as usize <= INLINE {
            return Ok(Self::default());
        }
        let mut posting = Self::default();
        posting.install_heap(Vec::with_capacity((count as usize).max(INLINE * 2)))?;
        Ok(posting)
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
            // SAFETY: heap came from a Vec<T> allocation of capacity at
            // least len; its first len elements were initialized by push.
            unsafe { slice::from_raw_parts(self.payload.heap, self.len()) }
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
            unsafe { slice::from_raw_parts_mut(self.payload.heap, self.len()) }
        }
    }

    /// Move the posting out and leave an empty inline posting behind.
    #[cfg(test)]
    pub(super) fn take(&mut self) -> Self {
        mem::take(self)
    }

    fn install_heap(&mut self, vec: Vec<T>) -> Result<()> {
        // Both callers either still hold inline data or used mem::take first.
        debug_assert!(self.is_inline());
        let len = u32::try_from(vec.len()).map_err(|_| "posting length exceeds u32")?;
        let capacity = u32::try_from(vec.capacity()).map_err(|_| "posting capacity exceeds u32")?;
        if (capacity as usize) < INLINE * 2 || len > capacity {
            return Err("invalid heap posting Vec".into());
        }
        // Conversion cannot fail beyond this point. Suppress Vec's destructor
        // only after checking both fields, then transfer its allocation.
        let mut vec = ManuallyDrop::new(vec);
        self.payload = Payload {
            heap: vec.as_mut_ptr(),
        };
        self.len = len;
        self.capacity = capacity;
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
            let mut vec = Vec::with_capacity(INLINE * 2);
            vec.extend_from_slice(self.as_slice());
            vec.push(pos);
            return self.install_heap(vec);
        }
        if self.len < self.capacity {
            // SAFETY: ptr is Vec allocation; len < capacity leaves one spare
            // uninitialized slot and &mut self is its unique owner.
            unsafe {
                self.payload.heap.add(self.len as usize).write(pos);
            }
            self.len = next_len;
            return Ok(());
        }
        let requested = (self.capacity as usize)
            .saturating_mul(2)
            .min(u32::MAX as usize)
            .max(next_len as usize);
        // The destination is empty before a Vec can reallocate or unwind.
        // The old raw pointer is owned only by the reconstructed Vec below.
        let old = ManuallyDrop::new(mem::take(self));
        // SAFETY: old is heap mode, and ptr/len/cap are the exact raw parts
        // recorded when its unique Vec allocation was last installed.
        let mut vec = unsafe {
            Vec::from_raw_parts(old.payload.heap, old.len as usize, old.capacity as usize)
        };
        vec.reserve_exact(requested - vec.len());
        vec.push(pos);
        self.install_heap(vec)
    }
}

impl<T: Copy + Default, const INLINE: usize> Drop for PackedPosting<T, INLINE> {
    fn drop(&mut self) {
        if self.capacity != 0 {
            // SAFETY: heap mode uniquely owns a Vec allocation whose exact
            // raw pointer, initialized length, and capacity are stored here.
            unsafe {
                drop(Vec::from_raw_parts(
                    self.payload.heap,
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
