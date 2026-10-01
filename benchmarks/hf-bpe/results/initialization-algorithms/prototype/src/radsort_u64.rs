//! Rust translation of Clausecker's BSD-2-Clause `radixsort_permuted.c`.
//! Reference f69e816c3cd79d312cd67aea5b9cf1c338c1b371, July 2026 paper:
//! https://arxiv.org/abs/2607.05302 ; source/license in ../vendor.
//! Sort only the HIGH 32 bits; retain LOW 32 bits in stable incoming order.
//! Fixed 512-element blocks: 2 MiB scratch + 9 bytes per input block.
//! This is an experimental standalone port, not production Trainer code.
use std::ptr;
const RADIX: usize = 256;
const BLOCK: usize = 512;
const SCRATCH: usize = 2 * RADIX;

#[derive(Clone, Copy)]
struct Partial { index: usize, length: usize }
#[derive(Clone, Copy)]
struct Bucket { next: *mut u64, end: *mut u64 }
struct Sorter<'a> {
    records: &'a mut [u64],
    scratch: Vec<u64>,
    perm: Vec<u32>,
    perm2: Vec<u32>,
    usage: Vec<u8>,
    partials: [Partial; RADIX],
    fill: usize,
}
impl<'a> Sorter<'a> {
    fn new(records: &'a mut [u64]) -> Self {
        let full = records.len() / BLOCK;
        let blocks = full + SCRATCH;
        assert!(blocks <= u32::MAX as usize);
        let fill = RADIX + full + 1;
        let mut perm = vec![0; blocks];
        for (i, p) in perm.iter_mut().enumerate() {
            *p = if i < RADIX { i } else if i < fill - 1 {
                i + RADIX
            } else { i - (fill - 1) + RADIX } as u32;
        }
        let mut scratch = vec![0; SCRATCH * BLOCK];
        let tail = records.len() % BLOCK;
        scratch[RADIX * BLOCK..RADIX * BLOCK + tail]
            .copy_from_slice(&records[full * BLOCK..]);
        let mut partials = [Partial { index: blocks, length: 0 }; RADIX];
        partials[0] = Partial { index: fill - 1, length: tail };
        Self { records, scratch, perm, perm2: vec![0; blocks],
            usage: vec![0; blocks], partials, fill }
    }
    fn block(&mut self, physical: usize) -> *mut u64 {
        assert!(physical < self.perm.len());
        // SAFETY: physical IDs below SCRATCH identify full scratch blocks.
        // Other IDs identify exactly floor(n/BLOCK) full input blocks.
        // Vec allocations are never resized while these pointers are in use.
        unsafe {
            if physical < SCRATCH { self.scratch.as_mut_ptr().add(physical * BLOCK) }
            else { self.records.as_mut_ptr().add((physical - SCRATCH) * BLOCK) }
        }
    }
    fn length(&self, logical: usize, partial: &mut usize) -> usize {
        if *partial < RADIX && self.partials[*partial].index == logical {
            let length = self.partials[*partial].length;
            *partial += 1; length
        } else { BLOCK }
    }
    fn step(&mut self, shift: u32) {
        let mut buckets = [Bucket { next: ptr::null_mut(), end: ptr::null_mut() }; RADIX];
        let mut counts = [1_usize; RADIX];
        for (i, bucket) in buckets.iter_mut().enumerate() {
            let out = self.block(self.perm[i] as usize);
            // SAFETY: block() returns exactly BLOCK valid elements.
            *bucket = Bucket { next: out, end: unsafe { out.add(BLOCK) } };
            self.usage[i] = i as u8;
        }
        let mut output = RADIX;
        let mut partial = 0;
        for input in RADIX..self.fill {
            let source = self.block(self.perm[input] as usize);
            let length = self.length(input, &mut partial);
            for j in 0..length {
                // SAFETY: source is a valid block, length <= BLOCK. The stable
                // logical input traversal consumes each element exactly once.
                let value = unsafe { source.add(j).read() };
                let b = ((value >> shift) & 255) as usize;
                // SAFETY: bucket.next points at an unused cell in its current
                // output block. It advances at most to end before reallocation.
                // Input blocks are recycled only after consumption: the RADIX
                // block head start establishes output <= input. If equality
                // occurs, allocation follows the last consumed source element;
                // the newly allocated block is written on a subsequent element.
                // This is the head-start proof in paper Lemma 1 and reference C.
                unsafe {
                    buckets[b].next.write(value);
                    buckets[b].next = buckets[b].next.add(1);
                }
                if buckets[b].next == buckets[b].end {
                    debug_assert!(output <= input);
                    let out = self.block(self.perm[output] as usize);
                    buckets[b] = Bucket { next: out, end: unsafe { out.add(BLOCK) } };
                    self.usage[output] = b as u8;
                    counts[b] += 1;
                    output += 1;
                }
            }
        }
        self.fill = output;
        let mut starts = [RADIX; RADIX];
        for b in 1..RADIX { starts[b] = starts[b - 1] + counts[b - 1]; }
        for i in 0..self.fill {
            let b = self.usage[i] as usize;
            let j = starts[b]; starts[b] += 1;
            self.perm2[j] = self.perm[i];
            let base = self.block(self.perm[i] as usize);
            // SAFETY: next/end belong to the same BLOCK-sized allocation.
            if unsafe { base.add(BLOCK) } == buckets[b].end {
                self.partials[b] = Partial {
                    index: j, length: unsafe { buckets[b].next.offset_from(base) as usize }
                };
            }
        }
        assert!(self.fill + RADIX <= self.perm.len());
        self.perm2[..RADIX].copy_from_slice(&self.perm[self.fill..self.fill + RADIX]);
        self.perm2[self.fill + RADIX..].copy_from_slice(&self.perm[self.fill + RADIX..]);
        self.fill += RADIX;
        std::mem::swap(&mut self.perm, &mut self.perm2);
    }
    fn compact(&mut self) {
        for i in 0..self.perm.len() { self.perm2[self.perm[i] as usize] = i as u32; }
        let mut start = 0;
        let mut partial = 0;
        let mut free_logical = 0;
        let mut free_physical = self.perm[0] as usize;
        for destination in SCRATCH..self.perm.len() {
            let input = destination - RADIX;
            let mut output = self.perm2[destination] as usize;
            let source = self.perm[input] as usize;
            if output > input && output < self.fill {
                debug_assert!(free_logical < RADIX || free_logical >= self.fill);
                let from = self.block(destination); let to = self.block(free_physical);
                // SAFETY: both physical IDs are valid full blocks. Free block
                // receives live output which would otherwise be overwritten.
                unsafe { ptr::copy(from, to, BLOCK); }
                self.perm[free_logical] = destination as u32;
                self.perm[output] = free_physical as u32;
                self.perm2[free_physical] = output as u32;
                self.perm2[destination] = free_logical as u32;
                free_logical = output;
                output = self.perm2[destination] as usize;
            }
            let length = self.length(input, &mut partial);
            let from = self.block(source);
            assert!(start + length <= self.records.len());
            // SAFETY: length valid source elements are moved into the next
            // logical output interval. ptr::copy permits overlap as memmove.
            unsafe { ptr::copy(from, self.records.as_mut_ptr().add(start), length); }
            self.perm[input] = destination as u32;
            self.perm[output] = source as u32;
            self.perm2[source] = output as u32;
            self.perm2[destination] = input as u32;
            start += length;
            if input != output { free_physical = source; free_logical = output; }
        }
        for input in self.perm.len() - RADIX..self.fill {
            let source = self.perm[input] as usize;
            assert!(source < SCRATCH);
            let length = self.length(input, &mut partial);
            let from = self.block(source);
            assert!(start + length <= self.records.len());
            unsafe { ptr::copy(from, self.records.as_mut_ptr().add(start), length); }
            start += length;
        }
        assert_eq!(start, self.records.len());
    }
}
/// Returns simultaneous allocated sorting scratch bytes. Mutates records in
/// place; output is physically contiguous and stable within each high32 key.
pub fn sort(records: &mut [u64]) -> usize {
    if records.len() < 2 { return 0; }
    let mut sorter = Sorter::new(records);
    let bytes = sorter.scratch.capacity() * 8 + sorter.perm.capacity() * 4
        + sorter.perm2.capacity() * 4 + sorter.usage.capacity();
    for shift in [32, 40, 48, 56] { sorter.step(shift); }
    sorter.compact(); bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stable_high32_at_block_boundaries_and_all_key_bits() {
        let mut seed = 371_u64;
        for n in [0, 1, 2, 255, 256, 511, 512, 513, 1023, 1024, 1025, 8192, 131072, 131073] {
            for distribution in 0..5 {
                let mut records = Vec::with_capacity(n);
                for p in 0..n {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let key = match distribution {
                        0 => u32::MAX,
                        1 => (p % 2) as u32,
                        2 => match p % 4 { 0 => 0, 1 => u32::MAX, 2 => 1 << 31, _ => 1 },
                        3 => (seed >> 32) as u32 % 97,
                        _ => (seed >> 32) as u32,
                    };
                    records.push((u64::from(key) << 32) | p as u64);
                }
                let mut expected = records.clone();
                expected.sort_by_key(|r| r >> 32);
                sort(&mut records);
                assert_eq!(records, expected, "n={n}, distribution={distribution}");
            }
        }
    }
}
