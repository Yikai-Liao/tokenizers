//! Prefix each chunk's owner counts, then fill disjoint final owner streams.
use super::*;
use std::mem::{ManuallyDrop, MaybeUninit};

pub(super) fn direct<C: Slot>(corpus: &[C], workers: usize) -> Vec<Vec<u64>> {
    let chunk = corpus.len().div_ceil(workers).max(1);
    let counts: Vec<Vec<usize>> = corpus
        .par_chunks(chunk)
        .enumerate()
        .map(|(c, slots)| {
            let base = c * chunk;
            let mut counts = vec![0; workers];
            for (i, slot) in slots.iter().enumerate() {
                let p = base + i;
                if p + 1 == corpus.len() {
                    break;
                }
                let (a, b) = (slot.token(), corpus[p + 1].token());
                if a != NONE && b != NONE {
                    counts[owner(key(a, b), workers)] += 1;
                }
            }
            counts
        })
        .collect();
    let mut routed: Vec<Vec<MaybeUninit<u64>>> = (0..workers)
        .map(|o| {
            let count = counts.iter().map(|c| c[o]).sum();
            let mut records = Vec::with_capacity(count);
            records.resize_with(count, MaybeUninit::uninit);
            records
        })
        .collect();
    // Split each owner's final allocation in increasing physical chunk order.
    // The type system keeps all emitted ranges mutually exclusive across jobs.
    let mut destinations: Vec<Vec<&mut [MaybeUninit<u64>]>> = (0..counts.len())
        .map(|_| Vec::with_capacity(workers))
        .collect();
    for (o, records) in routed.iter_mut().enumerate() {
        let mut rest = records.as_mut_slice();
        for (c, destinations) in destinations.iter_mut().enumerate() {
            let (part, next) = rest.split_at_mut(counts[c][o]);
            destinations.push(part);
            rest = next;
        }
        assert!(rest.is_empty());
    }
    corpus
        .par_chunks(chunk)
        .enumerate()
        .zip(destinations.into_par_iter())
        .for_each(|((c, slots), mut destinations)| {
            let base = c * chunk;
            let mut next = vec![0; workers];
            for (i, slot) in slots.iter().enumerate() {
                let p = base + i;
                if p + 1 == corpus.len() {
                    break;
                }
                let (a, b) = (slot.token(), corpus[p + 1].token());
                if a != NONE && b != NONE {
                    debug_assert!(a <= u16::MAX as u32 && b <= u16::MAX as u32);
                    let o = owner(key(a, b), workers);
                    let code = (a << 16) | b;
                    destinations[o][next[o]].write((u64::from(code) << 32) | p as u64);
                    next[o] += 1;
                }
            }
            for (o, destination) in destinations.iter().enumerate() {
                assert_eq!(next[o], destination.len());
            }
        });
    routed
        .into_iter()
        .map(|records| {
            // SAFETY: count and emit use identical valid-edge/owner predicates on
            // the unchanged corpus. Disjoint safe slices cover each allocation;
            // every job checks its write count, and the parallel join completed.
            // MaybeUninit<u64> has u64's layout; transfer allocation ownership once.
            let mut records = ManuallyDrop::new(records);
            unsafe {
                Vec::from_raw_parts(
                    records.as_mut_ptr().cast::<u64>(),
                    records.len(),
                    records.capacity(),
                )
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn direct_owner_streams_match_physical_scan_before_floor_filtering() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let cases = [
            Vec::new(),
            vec![NONE; 19],
            vec![65535; 73],
            (0..173)
                .map(|p| [NONE, 0, 65535, 32768, 1, 2, 91][(p * p + 3 * p) % 7])
                .collect(),
        ];
        for raw in cases {
            for workers in [1, 2, 3, 4, 7] {
                let mut expected = vec![Vec::new(); workers];
                for p in 0..raw.len().saturating_sub(1) {
                    let (a, b) = (raw[p], raw[p + 1]);
                    if a != NONE && b != NONE {
                        expected[owner(key(a, b), workers)]
                            .push((u64::from((a << 16) | b) << 32) | p as u64);
                    }
                }
                assert_eq!(pool.install(|| direct(&raw, workers)), expected);
                let atomic: Vec<_> = raw.iter().copied().map(AtomicU32::new).collect();
                assert_eq!(pool.install(|| direct(&atomic, workers)), expected);
            }
        }
    }
}
