//! Half-word skippable text described by Bille, Gørtz and Prezza (2017), §3.1.
//! Algorithm implemented independently for this engine's joined parallel phases.
//! Initial IDs select the same 16/24/32-bit planes as endpoint storage. The
//! 16/24-bit variants borrow a blank slot for a full merge ID; 32 bits stores
//! IDs directly. Physical coordinates, separators, and postings are unchanged.
use super::{CorpusPlan, PairMatch, SlotStorage, WORD_SEPARATOR_ID};
use std::sync::atomic::{AtomicU64, Ordering};
use tk_encode::Result;

pub(in super::super) struct PrezzaSlots<S: SlotStorage> {
    cells: S,
    live: Vec<AtomicU64>,
    skips: Vec<AtomicU64>,
}
impl<S: SlotStorage> PrezzaSlots<S> {
    #[inline]
    fn is_live(&self, position: usize) -> bool {
        self.live[position / 64].load(Ordering::Relaxed) & (1 << (position % 64)) != 0
    }
    #[inline]
    fn cell(&self, position: usize) -> u32 {
        self.cells.load(position) & Self::MASK
    }
    const MASK: u32 = u32::MAX >> (32 - S::CELL_BITS);
    fn bitmap(length: usize) -> Vec<AtomicU64> {
        (0..length.div_ceil(64))
            .map(|block| {
                let remaining = length - block * 64;
                AtomicU64::new(if remaining >= 64 {
                    u64::MAX
                } else {
                    (1 << remaining) - 1
                })
            })
            .collect()
    }
    #[cfg(test)]
    fn from_ids(ids: &[u32]) -> Self {
        Self {
            cells: S::from_ids(ids),
            live: Self::bitmap(ids.len()),
            skips: (0..ids.len().div_ceil(64))
                .map(|_| AtomicU64::new(0))
                .collect(),
        }
    }
}
impl<S: SlotStorage> SlotStorage for PrezzaSlots<S> {
    const NAVIGATES: bool = true;
    const LAYOUT: &'static str = match S::CELL_BITS {
        16 => "prezza16",
        24 => "prezza24",
        _ => "prezza32",
    };
    const CELL_BITS: u8 = S::CELL_BITS;
    #[cfg(test)]
    fn from_ids(ids: &[u32]) -> Self {
        Self::from_ids(ids)
    }
    fn allocation_bytes(&self) -> usize {
        self.cells.allocation_bytes() + (self.live.capacity() + self.skips.capacity()) * 8
    }
    fn from_prepared(
        prepared: &CorpusPlan<'_>,
        workers: usize,
        work: &crate::progress::WorkProgress,
    ) -> Result<Self> {
        let cells = S::from_prepared(prepared, workers, work)?;
        let live = Self::bitmap(cells.len());
        let skips = (0..live.len()).map(|_| AtomicU64::new(0)).collect();
        Ok(Self { cells, live, skips })
    }
    #[inline]
    fn len(&self) -> usize {
        self.cells.len()
    }
    #[inline]
    fn load(&self, position: usize) -> u32 {
        if !self.is_live(position) {
            return WORD_SEPARATOR_ID;
        }
        let first = self.cell(position);
        if S::CELL_BITS < 32 && position + 1 < self.len() && !self.is_live(position + 1) {
            (first << S::CELL_BITS) | self.cell(position + 1)
        } else if first == Self::MASK {
            WORD_SEPARATOR_ID
        } else {
            first
        }
    }
    #[inline]
    fn next(&self, position: usize) -> usize {
        let block = position / 64;
        let offset = position % 64;
        let bits = self.live[block].load(Ordering::Relaxed);
        let following = if offset == 63 {
            0
        } else {
            bits & (u64::MAX << (offset + 1))
        };
        if following != 0 {
            return block * 64 + following.trailing_zeros() as usize;
        }
        if block + 1 >= self.live.len() {
            return self.len();
        }
        let bits = self.live[block + 1].load(Ordering::Relaxed);
        if bits != 0 {
            return (block + 1) * 64 + bits.trailing_zeros() as usize;
        }
        if block + 2 == self.live.len() {
            return self.len();
        }
        // The first empty block of a long gap stores its distance in slots.
        (position + self.skips[block + 1].load(Ordering::Relaxed) as usize + 1).min(self.len())
    }
    #[inline]
    fn previous(&self, position: usize) -> usize {
        let block = position / 64;
        let offset = position % 64;
        let bits = self.live[block].load(Ordering::Relaxed) & ((1_u64 << offset) - 1);
        if bits != 0 {
            return block * 64 + (63 - bits.leading_zeros()) as usize;
        }
        if block == 0 {
            return 0;
        }
        let bits = self.live[block - 1].load(Ordering::Relaxed);
        if bits != 0 {
            return (block - 1) * 64 + (63 - bits.leading_zeros()) as usize;
        }
        position - self.skips[block - 1].load(Ordering::Relaxed) as usize - 1
    }
    unsafe fn store(&self, _position: usize, _id: u32) {
        unreachable!("half-word text must be changed by complete merges")
    }
    #[inline]
    unsafe fn merge(&self, matched: PairMatch, replacement: u32) {
        let left = matched.left_start as usize;
        let removed = matched.right_start as usize;
        let after = matched.next_start as usize;
        // Different writers may clear bits in the same word. RMW prevents lost
        // updates. Half words and skip cells lie within disjoint matched spans.
        self.live[removed / 64].fetch_and(!(1 << (removed % 64)), Ordering::Relaxed);
        // SAFETY: both cells are in this writer's disjoint matched span.
        // Reuse the endpoint planes' joined-write contract and separator codec.
        unsafe {
            if S::CELL_BITS == 32 {
                self.cells.store(left, replacement);
            } else {
                let high = replacement >> S::CELL_BITS;
                let low = replacement & Self::MASK;
                self.cells.store(
                    left,
                    if high == Self::MASK {
                        WORD_SEPARATOR_ID
                    } else {
                        high
                    },
                );
                self.cells.store(
                    left + 1,
                    if low == Self::MASK {
                        WORD_SEPARATOR_ID
                    } else {
                        low
                    },
                );
            }
        }
        let first_block = left / 64;
        let after_block = after / 64;
        if after_block > first_block + 1 {
            let distance = (after - left - 1) as u64;
            self.skips[first_block + 1].store(distance, Ordering::Relaxed);
            self.skips[after_block - 1].store(distance, Ordering::Relaxed);
        }
    }
    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn prefetch_pointer(&self, position: usize) -> *const i8 {
        self.cells.prefetch_pointer(position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn check<S: SlotStorage>(plane: &PrezzaSlots<S>, oracle: &[(usize, u32)]) {
        for (index, &(position, id)) in oracle.iter().enumerate() {
            assert_eq!(plane.load(position), id, "ID at {position}");
            assert_eq!(
                plane.next(position),
                oracle.get(index + 1).map_or(plane.len(), |p| p.0)
            );
            if index > 0 {
                assert_eq!(plane.previous(position), oracle[index - 1].0);
            }
        }
        for position in 0..plane.len() {
            if !oracle.iter().any(|p| p.0 == position) {
                assert_eq!(plane.load(position), WORD_SEPARATOR_ID);
            }
        }
    }
    #[test]
    fn all_cell_widths_preserve_initial_and_merged_full_width_ids() {
        fn exercise<S: SlotStorage>(initial: [u32; 3]) {
            for id in [
                65_534,
                65_535,
                65_536,
                0x00ff_ffff,
                0x0100_0000,
                0xfffe_ffff,
                u32::MAX - 1,
            ] {
                let plane = PrezzaSlots::<S>::from_ids(&[
                    WORD_SEPARATOR_ID,
                    initial[0],
                    initial[1],
                    initial[2],
                    WORD_SEPARATOR_ID,
                ]);
                check(
                    &plane,
                    &[
                        (0, WORD_SEPARATOR_ID),
                        (1, initial[0]),
                        (2, initial[1]),
                        (3, initial[2]),
                        (4, WORD_SEPARATOR_ID),
                    ],
                );
                // SAFETY: exclusive test access; separators are never removed.
                unsafe {
                    plane.merge(
                        PairMatch {
                            left_start: 1,
                            right_start: 2,
                            next_start: 3,
                            merged_span: 2,
                        },
                        id,
                    );
                }
                check(
                    &plane,
                    &[
                        (0, WORD_SEPARATOR_ID),
                        (1, id),
                        (3, initial[2]),
                        (4, WORD_SEPARATOR_ID),
                    ],
                );
                // SAFETY: exclusive test access to the three-slot merged span.
                unsafe {
                    plane.merge(
                        PairMatch {
                            left_start: 1,
                            right_start: 3,
                            next_start: 4,
                            merged_span: 3,
                        },
                        id,
                    );
                }
                check(
                    &plane,
                    &[(0, WORD_SEPARATOR_ID), (1, id), (4, WORD_SEPARATOR_ID)],
                );
            }
        }
        exercise::<super::super::slots::U16Slots>([0, 1, 65_534]);
        exercise::<super::super::slots::PackedU24Slots>([65_535, 65_536, 0x00ff_fffe]);
        exercise::<super::super::slots::U32Slots>([0x00ff_ffff, 0x0100_0000, u32::MAX - 1]);
    }
    #[test]
    fn halfword_ids_and_long_gaps_match_an_independent_live_list() {
        for length in [2, 63, 64, 65, 127, 128, 129, 513] {
            for seed in 1..=8_u64 {
                let plane =
                    PrezzaSlots::<super::super::slots::U16Slots>::from_ids(&vec![7; length]);
                let mut oracle: Vec<_> = (0..length).map(|p| (p, 7)).collect();
                let mut state = seed;
                while oracle.len() > 1 {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let index = (state as usize) % (oracle.len() - 1);
                    let left = oracle[index].0;
                    let right = oracle[index + 1].0;
                    let after = oracle.get(index + 2).map_or(length, |p| p.0);
                    let id = 65_534 + (state as u32 % 200_000);
                    // SAFETY: this test owns the plane exclusively.
                    unsafe {
                        plane.merge(
                            PairMatch {
                                left_start: left as u64,
                                right_start: right as u64,
                                next_start: after as u64,
                                merged_span: (after - left) as u64,
                            },
                            id,
                        );
                    }
                    oracle[index].1 = id;
                    oracle.remove(index + 1);
                    check(&plane, &oracle);
                }
            }
        }
    }
    #[test]
    fn adjacent_parallel_merges_preserve_shared_bitmap_words() {
        let plane = PrezzaSlots::<super::super::slots::U16Slots>::from_ids(&vec![1; 512]);
        std::thread::scope(|scope| {
            for lane in 0..8 {
                let plane = &plane;
                scope.spawn(move || {
                    for left in (lane * 2..512).step_by(16) {
                        // SAFETY: every writer owns a distinct two-slot span.
                        unsafe {
                            plane.merge(
                                PairMatch {
                                    left_start: left as u64,
                                    right_start: (left + 1) as u64,
                                    next_start: (left + 2) as u64,
                                    merged_span: 2,
                                },
                                100_000 + left as u32,
                            );
                        }
                    }
                });
            }
        });
        check(
            &plane,
            &(0..512)
                .step_by(2)
                .map(|p| (p, 100_000 + p as u32))
                .collect::<Vec<_>>(),
        );
    }
}
