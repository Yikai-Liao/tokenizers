//! D1 posting stream with a full U64 seed every 128 positions. Only gaps are
//! variable-length encoded: translating a list never changes its storage size.
//! Restart offsets, lengths, and allocation capacities use machine-sized words.
use super::*;
use std::alloc::{Layout, alloc, dealloc};

const GROUP: usize = 128;
const INLINE: usize = 1 << (usize::BITS - 1);
const PAIR: usize = 1 << (usize::BITS - 2);
const DELTA_MASK: usize = PAIR - 1;
const ARENA: usize = 1;
const MULTI: usize = 2;
const RESERVED: usize = 4;

#[derive(Default)]
pub(super) struct BlockPosting {
    // Heap counts need at least one stream byte per item, so Layout's isize
    // bound leaves the top bit free. Inline pairs store a gap, not an address.
    meta: usize,
    payload: usize,
}
unsafe impl Send for BlockPosting {}
unsafe impl Sync for BlockPosting {}

fn canonical(p: usize, bits: u8) -> u64 {
    ((p as u64 >> bits) << 32) | (p as u64 & ((1u64 << bits) - 1))
}
fn address(p: u64, bits: u8) -> usize {
    (((p >> 32) << bits) | (p & u32::MAX as u64)) as usize
}
fn prefix_bytes(groups: usize, reserved: bool) -> Option<usize> {
    if groups > 1 {
        groups.checked_mul(8)?.checked_add(24)
    } else {
        Some(if reserved { 16 } else { 8 })
    }
}
fn layout(groups: usize, stream_capacity: usize, reserved: bool) -> Result<Layout> {
    Layout::from_size_align(
        prefix_bytes(groups, reserved)
            .and_then(|head| head.checked_add(stream_capacity))
            .ok_or("posting allocation overflow")?,
        8,
    )
    .map_err(|_| "posting allocation layout overflow".into())
}
fn varint_bytes(value: u64) -> usize {
    (64 - value.leading_zeros()).max(1).div_ceil(7) as usize
}
unsafe fn write_varint(mut target: *mut u8, mut value: u64) -> *mut u8 {
    unsafe {
        while value >= 128 {
            target.write((value as u8 & 127) | 128);
            target = target.add(1);
            value >>= 7;
        }
        target.write(value as u8);
        target.add(1)
    }
}
unsafe fn read_varint(source: &mut *const u8) -> u64 {
    let mut value = 0u64;
    let mut shift = 0;
    unsafe {
        loop {
            let byte = (*source).read();
            *source = (*source).add(1);
            value |= ((byte & 127) as u64) << shift;
            if byte < 128 {
                return value;
            }
            shift += 7;
            debug_assert!(shift <= 63);
        }
    }
}

// Producers are replayable cursors over immutable input. A clone must yield
// the same values from its current position. Walking backward exposes each gap
// after one lookahead; seeds and varints can be placed backward without a group
// reversal buffer or another scan of the group's values.
#[inline]
fn reverse_codes(
    start: usize,
    end: usize,
    previous: u64,
    mut next: impl FnMut() -> u64,
    mut visit: impl FnMut(usize, u64) -> Result<()>,
) -> Result<()> {
    if start == end {
        return Ok(());
    }
    let mut value = next();
    for index in (start..end).rev() {
        let lower = if index > start { next() } else { previous };
        let code = if index % GROUP == 0 {
            value
        } else {
            value
                .checked_sub(lower)
                .ok_or("posting positions are not sorted")?
        };
        visit(index, code)?;
        value = lower;
    }
    Ok(())
}
fn suffix_bytes(
    start: usize,
    end: usize,
    previous: u64,
    next: impl FnMut() -> u64,
) -> Result<usize> {
    let mut bytes = 0usize;
    reverse_codes(start, end, previous, next, |index, code| {
        let size = if index % GROUP == 0 {
            8
        } else {
            varint_bytes(code)
        };
        bytes = bytes
            .checked_add(size)
            .ok_or("posting stream size overflow")?;
        Ok(())
    })?;
    Ok(bytes)
}

impl BlockPosting {
    pub(super) fn with_capacity(_count: usize) -> Result<Self> {
        Ok(Self::default())
    }
    fn is_inline(&self) -> bool {
        self.meta == 0 || self.meta & INLINE != 0
    }
    pub(super) fn len(&self) -> usize {
        if self.meta & INLINE == 0 {
            self.meta
        } else if self.meta & PAIR == 0 {
            1
        } else {
            2
        }
    }
    fn allocation_ptr(&self) -> *mut u8 {
        (self.payload & !7) as *mut u8
    }
    fn multi(&self) -> bool {
        self.payload & MULTI != 0
    }
    fn reserved(&self) -> bool {
        self.payload & RESERVED != 0
    }
    fn stream_len(&self) -> usize {
        unsafe { self.allocation_ptr().cast::<usize>().read() }
    }
    fn stream_capacity(&self) -> usize {
        unsafe {
            if self.multi() || self.reserved() {
                self.allocation_ptr().cast::<usize>().add(1).read()
            } else {
                self.stream_len()
            }
        }
    }
    fn group_capacity(&self) -> usize {
        if self.multi() {
            unsafe { self.allocation_ptr().cast::<usize>().add(2).read() }
        } else {
            1
        }
    }
    fn data_ptr(&self) -> *mut u8 {
        unsafe {
            self.allocation_ptr()
                .add(prefix_bytes(self.group_capacity(), self.reserved()).unwrap())
        }
    }
    fn group_offset(&self, group: usize) -> usize {
        if self.multi() {
            unsafe { self.allocation_ptr().cast::<usize>().add(3 + group).read() }
        } else {
            debug_assert_eq!(group, 0);
            0
        }
    }
    unsafe fn set_group_offset(&mut self, group: usize, offset: usize) {
        if self.multi() {
            unsafe {
                self.allocation_ptr()
                    .cast::<usize>()
                    .add(3 + group)
                    .write(offset);
            }
        } else {
            debug_assert_eq!((group, offset), (0, 0));
        }
    }
    pub(super) fn payload_bytes(&self) -> usize {
        if self.is_inline() {
            0
        } else {
            self.stream_capacity()
        }
    }
    pub(super) fn directory_bytes(&self) -> usize {
        if self.is_inline() {
            0
        } else {
            prefix_bytes(self.group_capacity(), self.reserved()).unwrap()
        }
    }
    pub(super) fn run_count(&self) -> usize {
        self.len().div_ceil(GROUP)
    }

    fn allocate(
        len: usize,
        groups: usize,
        bytes: usize,
        used: usize,
        reserved: bool,
        growth: bool,
    ) -> Result<Self> {
        let allocation = layout(groups, bytes, reserved)?;
        if len >= INLINE || len > allocation.size() {
            return Err("posting count exceeds allocation bounds".into());
        }
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
        unsafe {
            ptr.cast::<usize>().write(used);
            if groups > 1 || reserved {
                ptr.cast::<usize>().add(1).write(bytes);
            }
            if groups > 1 {
                ptr.cast::<usize>().add(2).write(groups);
            }
        }
        Ok(Self {
            meta: len,
            payload: ptr as usize
                | usize::from(arena) * ARENA
                | usize::from(groups > 1) * MULTI
                | usize::from(reserved) * RESERVED,
        })
    }
    // All writes are beyond the published stream's end, or into an unpublished
    // allocation. A producer panic leaves the previous posting readable.
    fn fill_suffix(
        &mut self,
        start: usize,
        end: usize,
        previous: u64,
        next: impl FnMut() -> u64,
        used: usize,
    ) -> Result<()> {
        let mut cursor = used;
        let old_end = if start == 0 { 0 } else { self.stream_len() };
        let data = self.data_ptr();
        reverse_codes(start, end, previous, next, |index, code| {
            let restart = index % GROUP == 0;
            let size = if restart { 8 } else { varint_bytes(code) };
            cursor = cursor
                .checked_sub(size)
                .filter(|&offset| offset >= old_end)
                .ok_or("posting producer changed during replay")?;
            unsafe {
                let target = data.add(cursor);
                if restart {
                    self.set_group_offset(index / GROUP, cursor);
                    target.cast::<u64>().write_unaligned(code);
                } else {
                    let end = write_varint(target, code);
                    debug_assert_eq!(end, target.add(size));
                }
            }
            Ok(())
        })?;
        if cursor != old_end {
            return Err("posting producer changed during replay".into());
        }
        Ok(())
    }
    fn build_exact(count: usize, mut next: impl FnMut() -> u64 + Clone) -> Result<Self> {
        if count == 0 {
            return Ok(Self::default());
        }
        if count >= INLINE {
            return Err("posting count exceeds allocation bounds".into());
        }
        if count == 1 {
            return Ok(Self {
                meta: INLINE,
                payload: next() as usize,
            });
        }
        if count == 2 {
            let last = next();
            let first = next();
            let gap = last
                .checked_sub(first)
                .ok_or("posting positions are not sorted")?;
            if gap <= DELTA_MASK as u64 {
                return Ok(Self {
                    meta: INLINE | PAIR | gap as usize,
                    payload: first as usize,
                });
            }
            let bytes = 8 + varint_bytes(gap);
            let result = Self::allocate(2, 1, bytes, bytes, false, false)?;
            unsafe {
                let data = result.data_ptr();
                data.cast::<u64>().write_unaligned(first);
                write_varint(data.add(8), gap);
            }
            return Ok(result);
        }
        let used = suffix_bytes(0, count, 0, next.clone())?;
        let mut result = Self::allocate(count, count.div_ceil(GROUP), used, used, false, false)?;
        result.fill_suffix(0, count, 0, next, used)?;
        Ok(result)
    }
    pub(super) fn from_reversed(
        count: usize,
        bits: u8,
        mut next: impl FnMut() -> usize + Clone,
    ) -> Result<Self> {
        Self::build_exact(count, move || canonical(next(), bits))
    }
    pub(super) fn append_reversed_reserved_at(
        &mut self,
        count: usize,
        bits: u8,
        mut next: impl FnMut() -> usize + Clone,
    ) -> Result<()> {
        self.extend_reverse(count, move || canonical(next(), bits))
    }
    fn extend_reverse(
        &mut self,
        count: usize,
        mut next: impl FnMut() -> u64 + Clone,
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let start = self.len();
        let end = start
            .checked_add(count)
            .filter(|&n| n < INLINE)
            .ok_or("posting count overflow")?;
        if self.is_inline() {
            let source = &*self;
            let mut i = end;
            let result = Self::build_exact(end, move || {
                i -= 1;
                if i >= start { next() } else { source.get(i) }
            })?;
            *self = result;
            return Ok(());
        }
        let previous = self.get(start - 1);
        let extra = suffix_bytes(start, end, previous, next.clone())?;
        let old_used = self.stream_len();
        let used = old_used
            .checked_add(extra)
            .ok_or("posting stream size overflow")?;
        let need_groups = end.div_ceil(GROUP);
        let old_groups = self.group_capacity();
        let old_bytes = self.stream_capacity();
        if old_groups >= need_groups && old_bytes >= used && (self.multi() || self.reserved()) {
            self.fill_suffix(start, end, previous, next, used)?;
            unsafe {
                self.allocation_ptr().cast::<usize>().write(used);
            }
            self.meta = end;
            return Ok(());
        }
        let groups = if old_groups >= need_groups {
            old_groups
        } else {
            need_groups.max(old_groups.saturating_mul(2))
        };
        let bytes = if old_bytes >= used {
            old_bytes
        } else {
            used.max(old_bytes.saturating_mul(2))
        };
        let mut result = Self::allocate(end, groups, bytes, old_used, true, true)?;
        unsafe {
            std::ptr::copy_nonoverlapping(self.data_ptr(), result.data_ptr(), old_used);
            for group in 0..start.div_ceil(GROUP) {
                result.set_group_offset(group, self.group_offset(group));
            }
        }
        result.fill_suffix(start, end, previous, next, used)?;
        unsafe {
            result.allocation_ptr().cast::<usize>().write(used);
        }
        *self = result;
        Ok(())
    }
    pub(super) fn append(&mut self, other: Self) -> Result<()> {
        if self.len() == 0 {
            *self = other;
            return Ok(());
        }
        let mut i = other.len();
        let source = &other;
        let mut cache = [0usize; GROUP];
        let mut cached_begin = usize::MAX;
        self.extend_reverse(i, move || {
            i -= 1;
            if source.is_inline() {
                return source.get(i);
            }
            let begin = i / GROUP * GROUP;
            if begin != cached_begin {
                let end = (begin + GROUP).min(source.len());
                source
                    .decode_into(32, begin, &mut cache[..end - begin])
                    .unwrap();
                cached_begin = begin;
            }
            cache[i - begin] as u64
        })
    }
    fn get(&self, i: usize) -> u64 {
        debug_assert!(i < self.len());
        if self.is_inline() {
            return self.payload as u64
                + if self.meta & PAIR != 0 && i == 1 {
                    (self.meta & DELTA_MASK) as u64
                } else {
                    0
                };
        }
        let group = i / GROUP;
        unsafe {
            let mut p = self.data_ptr().add(self.group_offset(group)) as *const u8;
            let mut value = p.cast::<u64>().read_unaligned();
            p = p.add(8);
            for _ in 0..i % GROUP {
                value += read_varint(&mut p);
            }
            value
        }
    }
    pub(super) fn decoder(&self, bits: u8, begin: usize, end: usize) -> Decoder<'_> {
        assert!(begin <= end && end <= self.len());
        let mut decoder = Decoder {
            posting: self,
            bits,
            index: begin,
            end,
            cursor: std::ptr::null(),
            value: 0,
        };
        // Prepare the value immediately before begin. Further bounded batches
        // preserve this cursor, so prefix replay happens only once per task.
        if begin < end && !self.is_inline() && begin % GROUP != 0 {
            unsafe {
                decoder.cursor = self.data_ptr().add(self.group_offset(begin / GROUP));
                decoder.value = decoder.cursor.cast::<u64>().read_unaligned();
                decoder.cursor = decoder.cursor.add(8);
                for _ in 1..begin % GROUP {
                    decoder.value += read_varint(&mut decoder.cursor);
                }
            }
        }
        decoder
    }
    pub(super) fn iter(&self, bits: u8) -> impl Iterator<Item = usize> + '_ {
        self.decoder(bits, 0, self.len())
    }
    pub(super) fn decode_into(&self, bits: u8, begin: usize, output: &mut [usize]) -> Result<()> {
        assert!(begin <= self.len() && output.len() <= self.len() - begin);
        let count = self.decoder(bits, begin, begin + output.len()).fill(output);
        debug_assert_eq!(count, output.len());
        Ok(())
    }
    #[inline]
    pub(super) fn try_for_each_range(
        &self,
        bits: u8,
        begin: usize,
        end: usize,
        mut f: impl FnMut(usize) -> Result<()>,
    ) -> Result<()> {
        for p in self.decoder(bits, begin, end) {
            f(p)?;
        }
        Ok(())
    }
    #[cfg(test)]
    pub(super) fn as_slice(&self) -> Vec<u32> {
        self.iter(32).map(|p| p as u32).collect()
    }
}
// Each worker constructs its own cursor over the immutable posting. No decoder
// state is stored in the posting or shared between tasks.
pub(super) struct Decoder<'a> {
    posting: &'a BlockPosting,
    bits: u8,
    index: usize,
    end: usize,
    cursor: *const u8,
    value: u64,
}
impl Decoder<'_> {
    #[inline]
    pub(super) fn fill(&mut self, output: &mut [usize]) -> usize {
        let count = output.len().min(self.end - self.index);
        for slot in &mut output[..count] {
            *slot = self.next().unwrap();
        }
        count
    }
}
impl Iterator for Decoder<'_> {
    type Item = usize;
    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.index == self.end {
            return None;
        }
        let value = if self.posting.is_inline() {
            self.posting.get(self.index)
        } else {
            unsafe {
                if self.index % GROUP == 0 {
                    self.cursor = self
                        .posting
                        .data_ptr()
                        .add(self.posting.group_offset(self.index / GROUP));
                    self.value = self.cursor.cast::<u64>().read_unaligned();
                    self.cursor = self.cursor.add(8);
                } else {
                    self.value += read_varint(&mut self.cursor);
                }
            }
            self.value
        };
        self.index += 1;
        Some(address(value, self.bits))
    }
}

impl Drop for BlockPosting {
    fn drop(&mut self) {
        if !self.is_inline() {
            let allocation = layout(
                self.group_capacity(),
                self.stream_capacity(),
                self.reserved(),
            )
            .expect("validated posting layout");
            let arena = self.payload & ARENA != 0;
            super::super::posting_arena::retirement::<u8>(allocation.size(), arena);
            if !arena {
                unsafe {
                    dealloc(self.allocation_ptr(), allocation);
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
        BlockPosting::build_exact(v.len(), move || {
            i -= 1;
            v[i]
        })
        .unwrap()
    }
    fn check(p: &BlockPosting, values: &[u64]) {
        assert_eq!(p.len(), values.len());
        assert_eq!(p.iter(32).map(|v| v as u64).collect::<Vec<_>>(), values);
        for i in 0..values.len() {
            assert_eq!(p.get(i), values[i]);
        }
        for batch in [1, 17, 128, 129, 509] {
            let mut decoded = vec![0usize; values.len()];
            for begin in (0..values.len()).step_by(batch) {
                let end = (begin + batch).min(values.len());
                p.decode_into(32, begin, &mut decoded[begin..end]).unwrap();
            }
            assert_eq!(
                decoded.iter().map(|&v| v as u64).collect::<Vec<_>>(),
                values
            );
        }
        p.decode_into(32, values.len(), &mut []).unwrap();
    }
    #[test]
    fn complete_address_domain_and_every_varint_boundary() {
        let mut values = vec![0, 0, 1];
        for bit in 1..64 {
            values.extend([(1u64 << bit) - 1, 1u64 << bit]);
        }
        values.push(u64::MAX);
        values.sort_unstable();
        for split in 0..=values.len() {
            let mut p = make(&values[..split]);
            p.append(make(&values[split..])).unwrap();
            check(&p, &values);
        }
    }
    #[test]
    fn group_boundaries_and_partial_appends() {
        for n in [0, 1, 2, 3, 7, 127, 128, 129, 255, 256, 257, 509] {
            let values: Vec<_> = (0..n).map(|i| (1u64 << 48) + i as u64 * 65537).collect();
            check(&make(&values), &values);
            for split in [0, 1, 2, 3, 126, 127, 128, 129, n / 2, n] {
                if split > n {
                    continue;
                }
                let mut p = make(&values[..split]);
                p.append(make(&values[split..])).unwrap();
                check(&p, &values);
            }
        }
    }
    #[test]
    fn translating_equal_gaps_never_changes_storage_size() {
        for n in [1, 2, 3, 8, 32, 127, 128, 129, 1024] {
            for gap in [0, 1, 127, 128, 16384, (1u64 << 32) + 7] {
                let relative: Vec<_> = (0..n).map(|i| i as u64 * gap).collect();
                let span = relative.last().copied().unwrap_or(0);
                let reference = make(&relative);
                let bytes = reference.payload_bytes() + reference.directory_bytes();
                for base in [1u64 << 40, 1u64 << 56, u64::MAX - span] {
                    let shifted: Vec<_> = relative.iter().map(|&p| p + base).collect();
                    let p = make(&shifted);
                    assert_eq!(
                        p.payload_bytes() + p.directory_bytes(),
                        bytes,
                        "n={n} gap={gap} base={base}"
                    );
                    check(&p, &shifted);
                }
            }
        }
    }
    #[test]
    fn inline_pairs_depend_on_gap_not_absolute_address() {
        assert_eq!(std::mem::size_of::<BlockPosting>(), 16);
        for gap in [0, 1, DELTA_MASK as u64, DELTA_MASK as u64 + 1, u64::MAX] {
            for first in [0, u64::MAX - gap] {
                let mut p = make(&[first, first + gap]);
                assert_eq!(p.is_inline(), gap <= DELTA_MASK as u64);
                check(&p, &[first, first + gap]);
                p.append(make(&[first + gap])).unwrap();
                check(&p, &[first, first + gap, first + gap]);
            }
        }
    }
    #[test]
    fn repeated_append_is_amortized_in_bytes_and_directory_capacity() {
        let session = super::super::super::posting_arena::LocalSession::new();
        session.configure(256);
        {
            let values: Vec<_> = (0..4097).map(|i| (1u64 << 48) + i * 17).collect();
            let mut p = BlockPosting::default();
            for &v in &values {
                p.extend_reverse(1, move || v).unwrap();
            }
            check(&p, &values);
            assert!(p.stream_capacity() < p.stream_len() * 2);
            assert!(p.group_capacity() < p.len().div_ceil(GROUP) * 2);
        }
        let counters = session.finish();
        assert!(counters.grows <= 32, "{}", counters.grows);
        assert_eq!(counters.arena_requested_bytes, counters.arena_retired_bytes);
        assert_eq!(counters.heap_requested_bytes, counters.heap_freed_bytes);
    }
    #[test]
    fn producer_panic_keeps_original_inline_heap_and_reserved_stream() {
        for n in [0, 1, 2, 3, 17, 127, 128, 129, 257] {
            let values: Vec<_> = (0..n).map(|i| i as u64).collect();
            let mut p = make(&values);
            for _ in 0..2 {
                let before: Vec<_> = p.iter(32).map(|p| p as u64).collect();
                let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut calls = 0;
                    p.extend_reverse(3, move || {
                        calls += 1;
                        if calls == 2 {
                            panic!("producer interrupted");
                        }
                        1000
                    })
                    .unwrap();
                }));
                assert!(failed.is_err());
                check(&p, &before);
                p.append(make(&[1000])).unwrap();
            }
        }
    }
    #[test]
    fn fill_pass_panic_does_not_publish_new_stream_or_offsets() {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        let mut p = make(&(0..130).collect::<Vec<u64>>());
        p.append(make(&(130..258).collect::<Vec<_>>())).unwrap();
        p.append(make(&(258..270).collect::<Vec<_>>())).unwrap();
        assert_eq!(p.group_capacity(), 4);
        assert!(p.stream_capacity() >= 498);
        let before: Vec<_> = p.iter(32).map(|x| x as u64).collect();
        let visits = Arc::new(AtomicUsize::new(0));
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut i = 469u64;
            p.extend_reverse(200, move || {
                if visits.fetch_add(1, Ordering::SeqCst) == 290 {
                    panic!("second pass interrupted");
                }
                let value = i;
                i -= 1;
                value
            })
            .unwrap();
        }));
        assert!(failed.is_err());
        check(&p, &before);
        p.append(make(&(270..470).collect::<Vec<_>>())).unwrap();
        check(&p, &(0..470).collect::<Vec<_>>());
    }
    #[test]
    fn changed_replay_size_cannot_overwrite_or_publish_uninitialized_bytes() {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        for wider_on_replay in [false, true] {
            let mut p = make(&(0..270).collect::<Vec<u64>>());
            p.append(make(&[270])).unwrap();
            let before: Vec<_> = p.iter(32).map(|x| x as u64).collect();
            let visits = Arc::new(AtomicUsize::new(0));
            let mut i = 20u64;
            let result = p.extend_reverse(20, move || {
                let replay = visits.fetch_add(1, Ordering::SeqCst) >= 20;
                let gap = if replay == wider_on_replay {
                    1u64 << 40
                } else {
                    1
                };
                let value = 270 + i * gap;
                i -= 1;
                value
            });
            assert!(result.is_err());
            check(&p, &before);
            p.append(make(&(271..291).collect::<Vec<_>>())).unwrap();
            check(&p, &(0..291).collect::<Vec<_>>());
        }
    }
    #[test]
    fn arena_and_heap_account_for_complete_allocations() {
        let session = super::super::super::posting_arena::LocalSession::new();
        session.configure(256);
        for n in [3, 32, 129, 1024] {
            let values: Vec<_> = (0..n).map(|i| (1u64 << 48) + i as u64 * 127).collect();
            let p = make(&values);
            let allocation = layout(p.group_capacity(), p.stream_capacity(), p.reserved()).unwrap();
            assert_eq!(allocation.size(), p.payload_bytes() + p.directory_bytes());
        }
        let counters = session.finish();
        assert!(counters.arena_buffers > 0 && counters.heap_buffers > 0);
        assert_eq!(counters.arena_requested_bytes, counters.arena_retired_bytes);
        assert_eq!(counters.heap_requested_bytes, counters.heap_freed_bytes);
    }
    #[test]
    fn wide_counts_and_offsets_have_no_u32_limit() {
        let count = u32::MAX as usize + 1;
        assert!(layout(count.div_ceil(GROUP), count * 2, false).is_ok());
        assert!(layout(usize::MAX, usize::MAX, false).is_err());
        assert!(layout(1, isize::MAX as usize, false).is_err());
        let p = std::mem::ManuallyDrop::new(BlockPosting {
            meta: count,
            payload: 0,
        });
        assert_eq!(p.len(), count);
    }
    #[test]
    fn one_long_posting_can_be_split_across_workers_with_private_cursors() {
        let values: Vec<_> = (0..8193u64)
            .map(|i| (1u64 << 60) + i * ((1u64 << 32) + 7))
            .collect();
        let posting = make(&values);
        let ranges: Vec<_> = (0..values.len())
            .step_by(509)
            .map(|begin| (begin, (begin + 509).min(values.len())))
            .collect();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let decoded = pool.install(|| {
            ranges
                .par_iter()
                .enumerate()
                .map(|(job, &(begin, end))| {
                    let mut decoder = posting.decoder(32, begin, end);
                    let mut buffer = [0usize; 129];
                    let batch = [17, 128, 129][job % 3];
                    let mut values = Vec::new();
                    loop {
                        let count = decoder.fill(&mut buffer[..batch]);
                        if count == 0 {
                            break;
                        }
                        values.extend_from_slice(&buffer[..count]);
                    }
                    values
                })
                .collect::<Vec<_>>()
        });
        assert_eq!(
            decoded
                .into_iter()
                .flatten()
                .map(|v| v as u64)
                .collect::<Vec<_>>(),
            values
        );
    }
    #[test]
    fn forced_geometry_restores_original_positions() {
        let values: Vec<_> = (0..4097).map(|i| i * 65537usize).collect();
        let mut p = BlockPosting::default();
        for chunk in values.chunks(73) {
            let mut i = chunk.len();
            p.append_reversed_reserved_at(chunk.len(), 16, move || {
                i -= 1;
                chunk[i]
            })
            .unwrap();
        }
        assert_eq!(p.iter(16).collect::<Vec<_>>(), values);
        let mut decoded = vec![0; 257];
        p.decode_into(16, 127, &mut decoded).unwrap();
        assert_eq!(decoded, values[127..384]);
    }
}
