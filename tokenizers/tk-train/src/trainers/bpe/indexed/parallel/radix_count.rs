//! Group the initial pair stream before installing retained owner entries.
//! The caller checks flat u32 positions and the complete initial u16 ID domain;
//! the final corpus, pair keys and postings keep their canonical full types.
use super::weight_lookup::WeightLookup;
use super::*;

#[derive(Default)]
pub(super) struct Metrics {
    pub(super) route_ms: f64,
    pub(super) count_ms: f64,
    pub(super) compact_ms: f64,
    pub(super) sort_ms: f64,
    pub(super) group_ms: f64,
    pub(super) install_ms: f64,
    pub(super) route_bytes: usize,
    pub(super) peak_route_bytes: usize,
    pub(super) scratch_bytes: usize,
    pub(super) group_bytes: usize,
    pub(super) pruned: usize,
}
struct Group {
    code: u32,
    start: u32,
    end: u32,
    frequency: u64,
}
fn canonical(code: u32) -> u64 {
    key(code >> 16, code & 0xffff)
}

// Only the upper pair-code half is sorted. Stable scatters retain the incoming
// spatial order within each key, so final posting positions remain increasing.
fn sort(records: &mut Vec<u64>) -> usize {
    if records.len() < 2 {
        return 0;
    }
    let mut scratch = vec![0_u64; records.len()];
    let bytes = scratch.capacity() * 8;
    for shift in [32, 40, 48, 56] {
        let mut counts = [0_usize; 256];
        for &record in records.iter() {
            counts[((record >> shift) & 255) as usize] += 1;
        }
        let mut next = [0_usize; 256];
        let mut prefix = 0;
        for (count, offset) in counts.iter().zip(next.iter_mut()) {
            *offset = prefix;
            prefix += count;
        }
        for &record in records.iter() {
            let bucket = ((record >> shift) & 255) as usize;
            scratch[next[bucket]] = record;
            next[bucket] += 1;
        }
        std::mem::swap(records, &mut scratch);
    }
    // All sorting scratch is freed before any final posting allocation starts.
    bytes
}

pub(super) fn initialize<C: Slot, O: Offset, const INLINE: usize>(
    corpus: &[C],
    block: &Block<O, INLINE>,
    uniform: Option<u64>,
    lookup: Option<&WeightLookup>,
    workers: usize,
    floor: u64,
    owners: &mut [Owner],
) -> Result<Metrics> {
    let begin = Instant::now();
    let chunk = corpus.len().div_ceil(workers).max(1);
    let mut routes: Vec<Vec<Vec<u64>>> = corpus
        .par_chunks(chunk)
        .enumerate()
        .map(|(c, slots)| {
            let base = c * chunk;
            let mut counts = vec![0_usize; workers];
            // Two sequential scans give exact buffers instead of geometric growth
            // of the larger 8-byte records. Owner assignment matches the old path.
            for (i, slot) in slots.iter().enumerate() {
                let p = base + i;
                if p + 1 == corpus.len() {
                    break;
                }
                let a = slot.token();
                let b = corpus[p + 1].token();
                if a != NONE && b != NONE {
                    counts[owner(key(a, b), workers)] += 1;
                }
            }
            let mut routed: Vec<Vec<u64>> = counts.into_iter().map(Vec::with_capacity).collect();
            for (i, slot) in slots.iter().enumerate() {
                let p = base + i;
                if p + 1 == corpus.len() {
                    break;
                }
                let a = slot.token();
                let b = corpus[p + 1].token();
                if a != NONE && b != NONE {
                    debug_assert!(a <= u16::MAX as u32 && b <= u16::MAX as u32);
                    let code = (a << 16) | b;
                    routed[owner(key(a, b), workers)].push((u64::from(code) << 32) | p as u64);
                }
            }
            routed
        })
        .collect();
    let mut old_bytes: usize = routes
        .iter()
        .flat_map(|r| r.iter())
        .map(|r| r.capacity() * 8)
        .sum();
    let mut peak = old_bytes;
    let compact_begin = Instant::now();
    let mut routed = Vec::with_capacity(workers);
    let mut final_bytes = 0;
    // Compact owners one at a time, releasing every old buffer as it is moved.
    // Avoid retaining all original routes plus all final owner streams at once.
    for o in 0..workers {
        let len = routes.iter().map(|r| r[o].len()).sum();
        let mut records = Vec::with_capacity(len);
        let bytes = records.capacity() * 8;
        peak = peak.max(old_bytes + final_bytes + bytes);
        for route in &mut routes {
            let mut part = std::mem::take(&mut route[o]);
            let part_bytes = part.capacity() * 8;
            records.append(&mut part);
            old_bytes -= part_bytes;
        }
        final_bytes += records.capacity() * 8;
        routed.push(records);
    }
    drop(routes);
    let compact_ms = compact_begin.elapsed().as_secs_f64() * 1000.0;
    let route_ms = begin.elapsed().as_secs_f64() * 1000.0;
    let count_begin = Instant::now();
    let sorting = Instant::now();
    let scratch_bytes = routed.par_iter_mut().map(sort).sum();
    let sort_ms = sorting.elapsed().as_secs_f64() * 1000.0;
    peak = peak.max(final_bytes + scratch_bytes); // capacity bound, not sampled RSS
    let grouping = Instant::now();
    let grouped: Vec<_> = routed
        .par_iter()
        .map(|records| -> Result<_> {
            let mut groups = Vec::new();
            let mut pruned = 0;
            let mut start = 0;
            while start < records.len() {
                let code = (records[start] >> 32) as u32;
                let mut end = start + 1;
                while end < records.len() && (records[end] >> 32) as u32 == code {
                    end += 1;
                }
                let frequency = if let Some(weight) = uniform {
                    weight
                        .checked_mul((end - start) as u64)
                        .ok_or("initial frequency exceeds u64")?
                } else {
                    let lookup = lookup.ok_or("radix count needs flat word metadata")?;
                    records[start..end]
                        .iter()
                        .map(|&r| lookup.weight(block, r as u32))
                        .sum()
                };
                if frequency >= floor {
                    groups.push(Group {
                        code,
                        start: u32::try_from(start).map_err(|_| "owner route exceeds u32")?,
                        end: u32::try_from(end).map_err(|_| "owner route exceeds u32")?,
                        frequency,
                    });
                } else {
                    pruned += 1;
                }
                start = end;
            }
            Ok((groups, pruned))
        })
        .collect::<Result<Vec<_>>>()?;
    let group_ms = grouping.elapsed().as_secs_f64() * 1000.0;
    let group_bytes = grouped
        .iter()
        .map(|g| g.0.capacity() * std::mem::size_of::<Group>())
        .sum();
    let pruned = grouped.iter().map(|g| g.1).sum();
    let installing = Instant::now();
    owners
        .par_iter_mut()
        .zip(routed.into_par_iter())
        .zip(grouped.into_par_iter())
        .map(|((ledger, records), (groups, _))| -> Result<()> {
            ledger.entries = AHashMap::with_capacity(groups.len());
            for group in groups {
                let count = group.end - group.start;
                let mut positions = SmallPosting::with_capacity(count)?;
                let source = &records[group.start as usize..group.end as usize];
                let mut next = source.len();
                positions.append_reversed_reserved(count, || {
                    next -= 1;
                    source[next] as u32
                })?;
                debug_assert_eq!(next, 0);
                debug_assert!(positions.as_slice().windows(2).all(|w| w[0] < w[1]));
                ledger.entries.insert(
                    canonical(group.code),
                    Entry {
                        frequency: group.frequency,
                        blocks: positions,
                    },
                );
            }
            Ok(())
        })
        .collect::<Result<Vec<_>>>()?;
    let install_ms = installing.elapsed().as_secs_f64() * 1000.0;
    Ok(Metrics {
        route_ms,
        compact_ms,
        sort_ms,
        group_ms,
        install_ms,
        count_ms: count_begin.elapsed().as_secs_f64() * 1000.0,
        route_bytes: final_bytes,
        peak_route_bytes: peak,
        scratch_bytes,
        group_bytes,
        pruned,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn radix_codes_use_all_bits_and_preserve_order_within_a_key() {
        let mut seed = 371_u64;
        let mut records = Vec::new();
        for p in 0..8192_u32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let code = match p % 5 {
                0 => u32::MAX,
                1 => 0,
                2 => 1 << 31,
                _ => (seed >> 32) as u32 % 97,
            };
            records.push((u64::from(code) << 32) | u64::from(p));
        }
        let mut expected = records.clone();
        expected.sort_by_key(|r| r >> 32);
        sort(&mut records);
        assert_eq!(records, expected);
        assert_eq!(canonical(u32::MAX), key(65535, 65535));
        let mut empty = Vec::new();
        assert_eq!(sort(&mut empty), 0);
        let mut singleton = vec![u64::MAX];
        sort(&mut singleton);
        assert_eq!(singleton, vec![u64::MAX]);
    }
}
