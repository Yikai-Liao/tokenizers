//! Split address planes: raw low u32 plus 0..4 bytes of upper address per item.
//! There is no segment directory. A list dispatches once on its upper width.
//! Length and capacity are machine-sized; narrow birth links are job-local only.
use super::*;
use std::alloc::{Layout, alloc, dealloc};

#[derive(Default)]
pub(super) struct BlockPosting {
    len: usize,
    // For len <= 2 these two words hold full-width inline addresses.
    capacity: usize,
    // Otherwise capacity is an allocation size, and payload a 16-aligned pointer.
    // Pointer low bits: width (0..4) and arena origin (8).
    payload: usize,
}
// Unique ownership; arena sessions outlive postings moved between workers.
unsafe impl Send for BlockPosting {}
unsafe impl Sync for BlockPosting {}
fn width(value: u64) -> usize {
    (64 - value.leading_zeros()).saturating_sub(32).div_ceil(8) as usize
}
fn canonical(p: usize, bits: u8) -> u64 {
    ((p as u64 >> bits) << 32) | (p as u64 & ((1u64 << bits) - 1))
}
fn address(p: u64, bits: u8) -> usize {
    (((p >> 32) << bits) | (p & u32::MAX as u64)) as usize
}
fn layout(capacity: usize, high: usize) -> Result<Layout> {
    Layout::from_size_align(
        capacity
            .checked_mul(4 + high)
            .ok_or("posting size overflow")?,
        16,
    )
    .map_err(|_| "posting allocation layout overflow".into())
}
impl BlockPosting {
    pub(super) fn with_capacity(_count: usize) -> Result<Self> {
        Ok(Self::default())
    }
    pub(super) fn len(&self) -> usize {
        self.len
    }
    fn high(&self) -> usize {
        self.payload & 7
    }
    fn ptr(&self) -> *mut u8 {
        (self.payload & !15) as *mut u8
    }
    pub(super) fn allocated_capacity(&self) -> usize {
        if self.len <= 2 { 0 } else { self.capacity }
    }
    pub(super) fn directory_bytes(&self) -> usize {
        if self.len <= 2 {
            0
        } else {
            self.capacity * self.high()
        }
    }
    pub(super) fn run_count(&self) -> usize {
        usize::from(self.len != 0)
    }
    fn get(&self, i: usize) -> u64 {
        debug_assert!(i < self.len);
        if self.len <= 2 {
            return if self.len == 1 || i == 1 {
                self.payload as u64
            } else {
                self.capacity as u64
            };
        }
        unsafe {
            let low = self.ptr().cast::<u32>().add(i).read() as u64;
            let p = self.ptr().add(4 * self.capacity);
            let high = match self.high() {
                0 => 0,
                1 => p.add(i).read() as u32,
                2 => p.cast::<u16>().add(i).read() as u32,
                3 => {
                    p.cast::<u16>().add(i).read() as u32
                        | ((p.add(2 * self.capacity + i).read() as u32) << 16)
                }
                4 => p.cast::<u32>().add(i).read(),
                _ => unreachable!(),
            };
            low | ((high as u64) << 32)
        }
    }
    unsafe fn put(&mut self, i: usize, value: u64) {
        // SAFETY: reserve established aligned planes with capacity > i.
        unsafe {
            self.ptr().cast::<u32>().add(i).write(value as u32);
            let high = (value >> 32) as u32;
            let p = self.ptr().add(4 * self.capacity);
            match self.high() {
                0 => debug_assert_eq!(high, 0),
                1 => p.add(i).write(high as u8),
                2 => p.cast::<u16>().add(i).write(high as u16),
                3 => {
                    p.cast::<u16>().add(i).write(high as u16);
                    p.add(2 * self.capacity + i).write((high >> 16) as u8);
                }
                4 => p.cast::<u32>().add(i).write(high),
                _ => unreachable!(),
            }
        }
    }
    // The temporary may contain uninitialized elements until its builder finishes.
    // Drop only releases primitive storage, so callback panic/error does not read
    // those elements, and the caller's old list is unchanged until installation.
    fn allocate(capacity: usize, high: usize, len: usize, growth: bool) -> Result<Self> {
        debug_assert!(len > 2 && capacity >= len);
        let allocation = layout(capacity, high)?;
        let (ptr, arena) =
            if let Some(p) = super::super::posting_arena::allocate_layout(allocation, growth)? {
                (p.as_ptr(), true)
            } else {
                let p = unsafe { alloc(allocation) };
                if p.is_null() {
                    return Err("posting allocation failed".into());
                }
                super::super::posting_arena::heap_allocation::<u8>(allocation.size(), growth);
                (p, false)
            };
        Ok(Self {
            len,
            capacity,
            payload: ptr as usize | high | if arena { 8 } else { 0 },
        })
    }
    fn copy_prefix(&self, result: &mut Self) {
        if self.len > 2 && self.high() == result.high() {
            unsafe {
                std::ptr::copy_nonoverlapping(self.ptr(), result.ptr(), self.len * 4);
                let source = self.ptr().add(self.capacity * 4);
                let target = result.ptr().add(result.capacity * 4);
                if self.high() == 3 {
                    std::ptr::copy_nonoverlapping(source, target, self.len * 2);
                    std::ptr::copy_nonoverlapping(
                        source.add(self.capacity * 2),
                        target.add(result.capacity * 2),
                        self.len,
                    );
                } else {
                    std::ptr::copy_nonoverlapping(source, target, self.len * self.high());
                }
            }
        } else {
            for i in 0..self.len {
                unsafe {
                    result.put(i, self.get(i));
                }
            }
        }
    }
    // Dispatch once for the entire fill, rather than inspecting the format at
    // each birth-chain node. Producer returns canonical addresses in reverse.
    fn fill_reverse(&mut self, start: usize, end: usize, last: u64, mut next: impl FnMut() -> u64) {
        unsafe {
            self.put(end - 1, last);
            let low = self.ptr().cast::<u32>();
            let high = self.ptr().add(self.capacity * 4);
            macro_rules! fill {
                ($put_high:expr) => {
                    for i in (start..end - 1).rev() {
                        let p = next();
                        low.add(i).write(p as u32);
                        ($put_high)(i, (p >> 32) as u32);
                    }
                };
            }
            match self.high() {
                0 => fill!(|_: usize, _: u32| {}),
                1 => fill!(|i, h| high.add(i).write(h as u8)),
                2 => fill!(|i, h| high.cast::<u16>().add(i).write(h as u16)),
                3 => fill!(|i, h| {
                    high.cast::<u16>().add(i).write(h as u16);
                    high.add(self.capacity * 2 + i).write((h >> 16) as u8);
                }),
                4 => fill!(|i, h| high.cast::<u32>().add(i).write(h)),
                _ => unreachable!(),
            }
        }
    }
    pub(super) fn from_reversed(
        count: usize,
        bits: u8,
        next: impl FnMut() -> usize,
    ) -> Result<Self> {
        let mut result = Self::default();
        result.append_reversed_reserved_at(count, bits, next)?;
        Ok(result)
    }
    pub(super) fn append_reversed_reserved_at(
        &mut self,
        count: usize,
        bits: u8,
        mut next: impl FnMut() -> usize,
    ) -> Result<()> {
        self.extend_reverse(count, || canonical(next(), bits))
    }
    fn extend_reverse(&mut self, count: usize, mut next: impl FnMut() -> u64) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let last = next();
        let start = self.len;
        let end = start.checked_add(count).ok_or("posting length overflow")?;
        if end == 1 {
            self.payload = last as usize;
            self.len = 1;
            return Ok(());
        }
        if end == 2 {
            let first = if start == 0 {
                next()
            } else {
                self.payload as u64
            };
            self.capacity = first as usize;
            self.payload = last as usize;
            self.len = 2;
            return Ok(());
        }
        let old_capacity = self.allocated_capacity();
        let old_high = if start <= 2 {
            if start == 0 {
                0
            } else {
                width(self.get(start - 1))
            }
        } else {
            self.high()
        };
        let high = width(last).max(old_high);
        if old_capacity >= end && old_high >= high {
            self.fill_reverse(start, end, last, next);
            self.len = end;
        } else {
            let capacity = if old_capacity >= end {
                old_capacity
            } else if old_capacity == 0 {
                end
            } else {
                end.max(old_capacity.saturating_mul(2))
            };
            let mut result = Self::allocate(capacity, high, end, old_capacity != 0)?;
            self.copy_prefix(&mut result);
            result.fill_reverse(start, end, last, next);
            *self = result;
        }
        Ok(())
    }
    pub(super) fn append(&mut self, other: Self) -> Result<()> {
        if self.len == 0 {
            *self = other;
            return Ok(());
        }
        let mut i = other.len;
        self.extend_reverse(other.len, || {
            i -= 1;
            other.get(i)
        })
    }
    pub(super) fn iter(&self, bits: u8) -> impl Iterator<Item = usize> + '_ {
        (0..self.len).map(move |i| address(self.get(i), bits))
    }
    #[inline]
    pub(super) fn try_for_each_range(
        &self,
        bits: u8,
        begin: usize,
        end: usize,
        mut f: impl FnMut(usize) -> Result<()>,
    ) -> Result<()> {
        assert!(begin <= end && end <= self.len);
        if self.len <= 2 {
            for i in begin..end {
                f(address(self.get(i), bits))?;
            }
            return Ok(());
        }
        unsafe {
            let lows = std::slice::from_raw_parts(self.ptr().cast::<u32>(), self.len);
            let high = self.ptr().add(4 * self.capacity);
            macro_rules! scan {
                ($h:expr) => {
                    for i in begin..end {
                        let upper: u32 = ($h)(i);
                        f(((upper as usize) << bits) | lows[i] as usize)?;
                    }
                };
            }
            match self.high() {
                0 => {
                    for &low in &lows[begin..end] {
                        f(low as usize)?;
                    }
                }
                1 => scan!(|i| high.add(i).read() as u32),
                2 => scan!(|i| high.cast::<u16>().add(i).read() as u32),
                3 => scan!(|i| high.cast::<u16>().add(i).read() as u32
                    | ((high.add(2 * self.capacity + i).read() as u32) << 16)),
                4 => scan!(|i| high.cast::<u32>().add(i).read()),
                _ => unreachable!(),
            }
        }
        Ok(())
    }
    #[cfg(test)]
    pub(super) fn as_slice(&self) -> Vec<u32> {
        self.iter(32).map(|p| p as u32).collect()
    }
}
impl Drop for BlockPosting {
    fn drop(&mut self) {
        if self.len > 2 {
            let allocation = layout(self.capacity, self.high()).expect("validated posting layout");
            let arena = self.payload & 8 != 0;
            super::super::posting_arena::retirement::<u8>(allocation.size(), arena);
            if !arena {
                unsafe {
                    dealloc(self.ptr(), allocation);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn make(v: &[u64]) -> BlockPosting {
        let mut i = v.len();
        BlockPosting::from_reversed(v.len(), 32, || {
            i -= 1;
            v[i] as usize
        })
        .unwrap()
    }
    #[test]
    fn full_range_widths_slices_and_growth() {
        let mut values = vec![0, 1, u32::MAX as u64];
        for bit in 32..64 {
            values.extend([(1u64 << bit) - 1, 1u64 << bit, (1u64 << bit) + 1]);
        }
        values.push(u64::MAX);
        values.sort_unstable();
        values.dedup();
        for n in 0..=values.len() {
            let mut p = make(&values[..n]);
            p.append(make(&values[n..])).unwrap();
            assert_eq!(p.iter(32).map(|p| p as u64).collect::<Vec<_>>(), values);
            for (begin, end) in [
                (0, values.len()),
                (1, 2),
                (3, 9),
                (values.len() - 1, values.len()),
            ] {
                let mut actual = Vec::new();
                p.try_for_each_range(32, begin, end, |v| {
                    actual.push(v as u64);
                    Ok(())
                })
                .unwrap();
                assert_eq!(actual, values[begin..end]);
            }
        }
    }
    #[test]
    fn every_plane_width_and_odd_capacity_decode() {
        for high in [0, 1, 255, 256, 65535, 65536, 16777215, 16777216, u32::MAX] {
            for len in [1, 2, 3, 7, 128, 129] {
                let values: Vec<_> = (0..len)
                    .map(|i| ((high as u64) << 32) | (i * 17) as u64)
                    .collect();
                let p = make(&values);
                if p.allocated_capacity() != 0 {
                    assert_eq!(p.high(), width(*values.last().unwrap()));
                }
                let mut decoded = Vec::new();
                p.try_for_each_range(32, 0, len, |v| {
                    decoded.push(v as u64);
                    Ok(())
                })
                .unwrap();
                assert_eq!(decoded, values);
                assert_eq!(p.iter(32).map(|v| v as u64).collect::<Vec<_>>(), values);
            }
        }
    }
    #[test]
    fn arena_and_heap_layout_accounting() {
        let session = super::super::super::posting_arena::LocalSession::new();
        session.configure(256);
        for high in [0, 1, 256, 65536, 16777216] {
            for len in [3, 129] {
                let values: Vec<_> = (0..len).map(|i| ((high as u64) << 32) | i as u64).collect();
                let mut p = make(&values);
                p.append(make(&values)).unwrap();
            }
        }
        let counters = session.finish();
        assert!(counters.arena_buffers >= 5);
        assert!(counters.heap_buffers >= 5);
        assert_eq!(counters.arena_requested_bytes, counters.arena_retired_bytes);
        assert_eq!(counters.heap_requested_bytes, counters.heap_freed_bytes);
    }
    #[test]
    fn two_full_width_addresses_are_inline_and_survive_growth() {
        for values in [[0, u64::MAX], [1u64 << 40, (1u64 << 63) + 7]] {
            let mut p = make(&values);
            assert_eq!(p.allocated_capacity(), 0);
            assert_eq!(p.directory_bytes(), 0);
            p.append(make(&[u64::MAX])).unwrap();
            assert_eq!(
                p.iter(32).map(|v| v as u64).collect::<Vec<_>>(),
                [values[0], values[1], u64::MAX]
            );
        }
    }
    #[test]
    fn reverse_producer_panic_leaves_old_inline_or_heap_list_valid() {
        for original in [vec![1], vec![1, 2], vec![1, 2, 3]] {
            let mut p = make(&original);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut calls = 0;
                p.append_reversed_reserved_at(3, 32, || {
                    calls += 1;
                    if calls == 2 {
                        panic!("producer probe");
                    }
                    99
                })
                .unwrap();
            }));
            assert!(result.is_err());
            assert_eq!(p.iter(32).map(|v| v as u64).collect::<Vec<_>>(), original);
        }
    }
    #[test]
    fn widening_address_does_not_double_unused_capacity() {
        let mut p = make(&[0, 1, 2]);
        p.append(make(&[3])).unwrap();
        assert_eq!(p.capacity, 6);
        p.append(make(&[1u64 << 40])).unwrap();
        assert_eq!(p.capacity, 6);
        assert_eq!(
            p.iter(32).map(|v| v as u64).collect::<Vec<_>>(),
            [0, 1, 2, 3, 1u64 << 40]
        );
    }
    #[test]
    fn no_u32_count_limit_and_overflow_is_checked() {
        assert_eq!(std::mem::size_of::<BlockPosting>(), 24);
        assert_eq!(
            layout(u32::MAX as usize + 1, 4).unwrap().size(),
            1usize << 35
        );
        assert!(layout(usize::MAX, 4).is_err());
    }
    #[test]
    fn forced_geometry_and_amortized_append() {
        let values: Vec<_> = (0..4097).map(|i| i * 65537usize).collect();
        let mut p = BlockPosting::default();
        for &v in &values {
            p.append_reversed_reserved_at(1, 16, || v).unwrap();
        }
        assert_eq!(p.iter(16).collect::<Vec<_>>(), values);
        assert!(p.capacity < 2 * values.len());
        let mut actual = Vec::new();
        p.try_for_each_range(16, 127, 258, |v| {
            actual.push(v);
            Ok(())
        })
        .unwrap();
        assert_eq!(actual, values[127..258]);
    }
}
