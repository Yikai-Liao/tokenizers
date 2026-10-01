// SPDX-License-Identifier: BSD-2-Clause
// Copyright (c) 2025, 2026 Robert Clausecker <clausecker@zib.de>
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

//! Rust translation of Clausecker's BSD-2-Clause `radixsort_permuted.c`.
//! Reference f69e816c3cd79d312cd67aea5b9cf1c338c1b371, July 2026 paper:
//! https://arxiv.org/abs/2607.05302 ; full upstream license retained above.
//! Sort the key half; retain the payload half in stable incoming order.
//! u64 records use 512-element blocks and 2 MiB scratch; u128 records
//! use 128-element blocks and 1 MiB scratch. Metadata adds 9 bytes per block.
//! Stable initial pair grouping with bounded block scratch.
use std::{marker::PhantomData, ptr};
const RADIX: usize = 256;
const BLOCK: usize = 512;
const SCRATCH: usize = 2 * RADIX;

trait Record: Copy + Default {
    const BLOCK: usize;
    fn key(self) -> u64;
}
impl Record for u64 {
    const BLOCK: usize = BLOCK;
    fn key(self) -> u64 {
        self >> 32
    }
}
impl Record for u128 {
    const BLOCK: usize = 128;
    fn key(self) -> u64 {
        (self >> 64) as u64
    }
}

#[derive(Clone, Copy)]
struct Partial {
    index: usize,
    length: usize,
}
#[derive(Clone, Copy)]
struct Bucket<T> {
    next: *mut T,
    end: *mut T,
}
struct Sorter<'a, T: Record> {
    records: *mut T,
    length: usize,
    _borrow: PhantomData<&'a mut [T]>,
    scratch: Vec<T>,
    scratch_base: *mut T,
    perm: Vec<u32>,
    perm2: Vec<u32>,
    usage: Vec<u8>,
    partials: [Partial; RADIX],
    fill: usize,
}
impl<'a, T: Record> Sorter<'a, T> {
    fn new(records: &'a mut [T]) -> Self {
        let full = records.len() / T::BLOCK;
        let blocks = full + SCRATCH;
        assert!(blocks <= u32::MAX as usize);
        let fill = RADIX + full + 1;
        let mut perm = vec![0; blocks];
        for (i, p) in perm.iter_mut().enumerate() {
            *p = if i < RADIX {
                i
            } else if i < fill - 1 {
                i + RADIX
            } else {
                i - (fill - 1) + RADIX
            } as u32;
        }
        let mut scratch = vec![T::default(); SCRATCH * T::BLOCK];
        let tail = records.len() % T::BLOCK;
        scratch[RADIX * T::BLOCK..RADIX * T::BLOCK + tail]
            .copy_from_slice(&records[full * T::BLOCK..]);
        let mut partials = [Partial {
            index: blocks,
            length: 0,
        }; RADIX];
        partials[0] = Partial {
            index: fill - 1,
            length: tail,
        };
        let length = records.len();
        let records = records.as_mut_ptr();
        let scratch_base = scratch.as_mut_ptr();
        Self {
            records,
            length,
            _borrow: PhantomData,
            scratch,
            scratch_base,
            perm,
            perm2: vec![0; blocks],
            usage: vec![0; blocks],
            partials,
            fill,
        }
    }
    fn block(&mut self, physical: usize) -> *mut T {
        assert!(physical < self.perm.len());
        // SAFETY: physical IDs below SCRATCH identify full scratch blocks.
        // Other IDs identify exactly floor(n/T::BLOCK) full input blocks.
        // Vec allocations are never resized while these pointers are in use.
        unsafe {
            if physical < SCRATCH {
                self.scratch_base.add(physical * T::BLOCK)
            } else {
                self.records.add((physical - SCRATCH) * T::BLOCK)
            }
        }
    }
    fn length(&self, logical: usize, partial: &mut usize) -> usize {
        if *partial < RADIX && self.partials[*partial].index == logical {
            let length = self.partials[*partial].length;
            *partial += 1;
            length
        } else {
            T::BLOCK
        }
    }
    #[cfg(debug_assertions)]
    fn validate(&self) {
        let mut seen = vec![false; self.perm.len()];
        for &physical in &self.perm {
            assert!(!seen[physical as usize]);
            seen[physical as usize] = true;
        }
        assert!(self.fill <= self.perm.len());
        assert!(self.partials.windows(2).all(|p| p[0].index < p[1].index));
        assert!(
            self.partials
                .iter()
                .all(|p| p.index >= RADIX && p.index < self.fill && p.length < T::BLOCK)
        );
        let mut partial = 0;
        let total: usize = (RADIX..self.fill)
            .map(|i| self.length(i, &mut partial))
            .sum();
        assert_eq!(total, self.length);
    }
    fn step(&mut self, shift: u32) {
        let mut buckets = [Bucket {
            next: ptr::null_mut(),
            end: ptr::null_mut(),
        }; RADIX];
        let mut counts = [1_usize; RADIX];
        for (i, bucket) in buckets.iter_mut().enumerate() {
            let out = self.block(self.perm[i] as usize);
            // SAFETY: block() returns exactly T::BLOCK valid elements.
            *bucket = Bucket {
                next: out,
                end: unsafe { out.add(T::BLOCK) },
            };
            self.usage[i] = i as u8;
        }
        let mut output = RADIX;
        let mut partial = 0;
        for input in RADIX..self.fill {
            let source = self.block(self.perm[input] as usize);
            let length = self.length(input, &mut partial);
            for j in 0..length {
                // SAFETY: source is a valid block, length <= T::BLOCK. The stable
                // logical input traversal consumes each element exactly once.
                let value = unsafe { source.add(j).read() };
                let b = ((value.key() >> shift) & 255) as usize;
                // SAFETY: bucket.next points at an unused cell in its current
                // output block. It advances at most to end before reallocation.
                // Input blocks are recycled only after consumption: the RADIX
                // block head start establishes output <= input. If equality
                // occurs, allocation follows the last consumed source element;
                // the newly allocated block is written on a subsequent element.
                // More precisely, after J consumed values the newly reserved
                // logical block is R + sum(floor(count[b]/B)) - 1 <=
                // R + floor(J/B) - 1 <= current input. Equality requires
                // J to end a full input block, so no unread value is overwritten.
                // This adapts paper Lemma 1 to the reference C allocation timing.
                unsafe {
                    buckets[b].next.write(value);
                    buckets[b].next = buckets[b].next.add(1);
                }
                if buckets[b].next == buckets[b].end {
                    debug_assert!(output <= input);
                    let out = self.block(self.perm[output] as usize);
                    buckets[b] = Bucket {
                        next: out,
                        end: unsafe { out.add(T::BLOCK) },
                    };
                    self.usage[output] = b as u8;
                    counts[b] += 1;
                    output += 1;
                }
            }
        }
        self.fill = output;
        let mut starts = [RADIX; RADIX];
        for b in 1..RADIX {
            starts[b] = starts[b - 1] + counts[b - 1];
        }
        for i in 0..self.fill {
            let b = self.usage[i] as usize;
            let j = starts[b];
            starts[b] += 1;
            self.perm2[j] = self.perm[i];
            let base = self.block(self.perm[i] as usize);
            // SAFETY: next/end belong to the same T::BLOCK-sized allocation.
            if unsafe { base.add(T::BLOCK) } == buckets[b].end {
                self.partials[b] = Partial {
                    index: j,
                    length: unsafe { buckets[b].next.offset_from(base) as usize },
                };
            }
        }
        assert!(self.fill + RADIX <= self.perm.len());
        self.perm2[..RADIX].copy_from_slice(&self.perm[self.fill..self.fill + RADIX]);
        self.perm2[self.fill + RADIX..].copy_from_slice(&self.perm[self.fill + RADIX..]);
        self.fill += RADIX;
        std::mem::swap(&mut self.perm, &mut self.perm2);
        #[cfg(debug_assertions)]
        self.validate();
    }
    fn compact(&mut self) {
        for i in 0..self.perm.len() {
            self.perm2[self.perm[i] as usize] = i as u32;
        }
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
                let from = self.block(destination);
                let to = self.block(free_physical);
                // SAFETY: both physical IDs are valid full blocks. Free block
                // receives live output which would otherwise be overwritten.
                unsafe {
                    ptr::copy(from, to, T::BLOCK);
                }
                self.perm[free_logical] = destination as u32;
                self.perm[output] = free_physical as u32;
                self.perm2[free_physical] = output as u32;
                self.perm2[destination] = free_logical as u32;
                free_logical = output;
                output = self.perm2[destination] as usize;
            }
            let length = self.length(input, &mut partial);
            let from = self.block(source);
            assert!(start + length <= self.length);
            // Completed partials can only shorten preceding output: start <=
            // (destination - SCRATCH) * T::BLOCK, and start + length never exceeds
            // this destination block's end. Any still-live destination block was
            // evacuated above; earlier physical destinations are already consumed.
            // SAFETY: length valid source elements are moved into the next
            // logical output interval. ptr::copy permits overlap as memmove.
            unsafe {
                ptr::copy(from, self.records.add(start), length);
            }
            self.perm[input] = destination as u32;
            self.perm[output] = source as u32;
            self.perm2[source] = output as u32;
            self.perm2[destination] = input as u32;
            start += length;
            if input != output {
                free_physical = source;
                free_logical = output;
            }
        }
        for input in self.perm.len() - RADIX..self.fill {
            let source = self.perm[input] as usize;
            assert!(source < SCRATCH);
            let length = self.length(input, &mut partial);
            let from = self.block(source);
            assert!(start + length <= self.length);
            unsafe {
                ptr::copy(from, self.records.add(start), length);
            }
            start += length;
        }
        assert_eq!(start, self.length);
    }
}
/// Returns simultaneous allocated sorting scratch bytes. Mutates records in
/// place; output is physically contiguous and stable within each high32 key.
pub(super) fn sort(records: &mut [u64]) -> usize {
    sort_digits(records, 0xffff_ffff)
}

/// Sort the full canonical u64 key in the high half of each u128 record.
/// The low half preserves the incoming local address. Constant key bytes
/// need no scatter pass. Scratch is 1 MiB plus 9 bytes per 128 input records; small
/// tiles use a simpler scatter when its allocation is smaller.
pub(in super::super) fn sort_wide(records: &mut [u128]) -> usize {
    let Some(&first) = records.first() else {
        return 0;
    };
    let varying = records
        .iter()
        .fold(0, |bits, &r| bits | (r.key() ^ first.key()));
    if records.len() * 16
        <= SCRATCH * <u128 as Record>::BLOCK * 16
            + (records.len() / <u128 as Record>::BLOCK + SCRATCH) * 9
    {
        sort_classic(records, varying)
    } else {
        sort_digits(records, varying)
    }
}
/// Bounded compact-key records use the smaller scatter implementation when
/// it avoids the fixed scratch allocation. Constant key digits are skipped.
pub(in super::super) fn sort_compact(records: &mut [u64]) -> usize {
    let Some(&first) = records.first() else {
        return 0;
    };
    let varying = records
        .iter()
        .fold(0, |bits, &r| bits | (r.key() ^ first.key()));
    if records.len() * 8 <= SCRATCH * BLOCK * 8 + (records.len() / BLOCK + SCRATCH) * 9 {
        sort_classic(records, varying)
    } else {
        sort_digits(records, varying)
    }
}
fn sort_classic<T: Record>(records: &mut [T], varying: u64) -> usize {
    if records.len() < 2 || varying == 0 {
        return 0;
    }
    let mut scratch = vec![T::default(); records.len()];
    let bytes = scratch.capacity() * std::mem::size_of::<T>();
    let mut flipped = false;
    {
        let mut input = &mut records[..];
        let mut output = scratch.as_mut_slice();
        for shift in (0..64).step_by(8) {
            if (varying >> shift) & 255 == 0 {
                continue;
            }
            let mut counts = [0usize; RADIX];
            for &r in input.iter() {
                counts[((r.key() >> shift) & 255) as usize] += 1;
            }
            let mut offsets = [0usize; RADIX];
            let mut sum = 0;
            for (count, offset) in counts.iter().zip(offsets.iter_mut()) {
                *offset = sum;
                sum += count;
            }
            for &r in input.iter() {
                let b = ((r.key() >> shift) & 255) as usize;
                output[offsets[b]] = r;
                offsets[b] += 1;
            }
            std::mem::swap(&mut input, &mut output);
            flipped = !flipped;
        }
    }
    if flipped {
        records.copy_from_slice(&scratch);
    }
    bytes
}
fn sort_digits<T: Record>(records: &mut [T], varying: u64) -> usize {
    if records.len() < 2 || varying == 0 {
        return 0;
    }
    let mut sorter = Sorter::new(records);
    let bytes = sorter.scratch.capacity() * std::mem::size_of::<T>()
        + sorter.perm.capacity() * 4
        + sorter.perm2.capacity() * 4
        + sorter.usage.capacity();
    for shift in (0..64).step_by(8) {
        if (varying >> shift) & 255 != 0 {
            sorter.step(shift);
        }
    }
    sorter.compact();
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stable_full64_keys_and_local_u32_addresses() {
        let mut seed = 371_u64;
        for n in [0, 1, 2, 511, 512, 513, 1025, 8192, 262145] {
            for distribution in 0..5 {
                let mut records = Vec::with_capacity(n);
                for p in 0..n {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let key = match distribution {
                        0 => u64::MAX,
                        1 => ((p % 3) as u64) << 32,
                        2 => seed % 97 | ((seed % 19) << 48),
                        3 => seed,
                        _ => [0, u64::MAX, 1 << 63, 1 << 32][p % 4],
                    };
                    let payload = if p % 2 == 0 {
                        u32::MAX - p as u32
                    } else {
                        p as u32
                    };
                    records.push((u128::from(key) << 64) | u128::from(payload));
                }
                let mut expected = records.clone();
                expected.sort_by_key(|r| r >> 64);
                sort_wide(&mut records);
                assert_eq!(records, expected, "n={n}, distribution={distribution}");
            }
        }
    }

    #[test]
    fn stable_high32_at_block_boundaries_and_all_key_bits() {
        let mut seed = 371_u64;
        for n in [
            0, 1, 2, 255, 256, 511, 512, 513, 1023, 1024, 1025, 8192, 131072, 131073, 300001,
        ] {
            for distribution in 0..6 {
                let mut records = Vec::with_capacity(n);
                for p in 0..n {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let key = match distribution {
                        0 => u32::MAX,
                        1 => (p % 2) as u32,
                        2 => match p % 4 {
                            0 => 0,
                            1 => u32::MAX,
                            2 => 1 << 31,
                            _ => 1,
                        },
                        3 => (seed >> 32) as u32 % 97,
                        _ => (seed >> 32) as u32,
                    };
                    let payload = if distribution == 5 {
                        seed as u32
                    } else {
                        p as u32
                    };
                    records.push((u64::from(key) << 32) | u64::from(payload));
                }
                let mut expected = records.clone();
                expected.sort_by_key(|r| r >> 32);
                let mut compact = records.clone();
                sort_compact(&mut compact);
                assert_eq!(
                    compact, expected,
                    "compact n={n}, distribution={distribution}"
                );
                sort(&mut records);
                assert_eq!(records, expected, "n={n}, distribution={distribution}");
            }
        }
    }
}
