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
    // Largest wave's allocation capacity; neither value is measured RSS.
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
    initialize_with_owner_width(
        corpus,
        block,
        uniform,
        lookup,
        workers,
        floor,
        owners,
        workers.min(2),
    )
}

// Keep a bounded number of sorting buffers alive. Finish each wave's postings
// before starting the next wave, so consumed owner records are released early.
fn initialize_with_owner_width<C: Slot, O: Offset, const INLINE: usize>(
    corpus: &[C],
    block: &Block<O, INLINE>,
    uniform: Option<u64>,
    lookup: Option<&WeightLookup>,
    workers: usize,
    floor: u64,
    owners: &mut [Owner],
    owner_width: usize,
) -> Result<Metrics> {
    assert!(owner_width > 0);
    debug_assert_eq!(owners.len(), workers);
    let owner_width = owner_width.min(workers);
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
    let mut remaining_record_bytes = final_bytes;
    let mut scratch_bytes = 0;
    let mut group_bytes = 0;
    let mut sort_ms = 0.0;
    let mut group_ms = 0.0;
    let mut install_ms = 0.0;
    let mut pruned = 0;
    for (owner_wave, records_wave) in owners
        .chunks_mut(owner_width)
        .zip(routed.chunks_mut(owner_width))
    {
        let wave_record_bytes: usize = records_wave.iter().map(|r| r.capacity() * 8).sum();
        let sorting = Instant::now();
        let wave_scratch_bytes = records_wave.par_iter_mut().map(sort).sum();
        sort_ms += sorting.elapsed().as_secs_f64() * 1000.0;
        scratch_bytes = scratch_bytes.max(wave_scratch_bytes);
        peak = peak.max(remaining_record_bytes + wave_scratch_bytes);
        let grouping = Instant::now();
        let grouped: Vec<_> = records_wave
            .par_iter()
            .map(|records| -> Result<_> {
                let mut groups = Vec::new();
                let mut discarded = 0;
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
                        discarded += 1;
                    }
                    start = end;
                }
                Ok((groups, discarded))
            })
            .collect::<Result<Vec<_>>>()?;
        group_ms += grouping.elapsed().as_secs_f64() * 1000.0;
        let wave_group_bytes = grouped
            .iter()
            .map(|g| g.0.capacity() * std::mem::size_of::<Group>())
            .sum();
        group_bytes = group_bytes.max(wave_group_bytes);
        pruned += grouped.iter().map(|g| g.1).sum::<usize>();
        let installing = Instant::now();
        owner_wave
            .par_iter_mut()
            .zip(records_wave.par_iter_mut())
            .zip(grouped.into_par_iter())
            .map(|((ledger, records), (groups, _))| -> Result<()> {
                let records = std::mem::take(records);
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
                    debug_assert!(positions.as_slice().windows(2).all(|w| w[0] < w[1]));
                    ledger.entries.insert(
                        canonical(group.code),
                        Entry {
                            frequency: group.frequency,
                            blocks: positions,
                        },
                    );
                }
                // Drop this owner's pair stream before the next wave allocates scratch.
                Ok(())
            })
            .collect::<Result<Vec<_>>>()?;
        install_ms += installing.elapsed().as_secs_f64() * 1000.0;
        remaining_record_bytes -= wave_record_bytes;
    }
    debug_assert_eq!(remaining_record_bytes, 0);
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

    fn check_wave_oracle<C: Slot>(raw: &[u32], block: &Block<u32, 2>) {
        let corpus: Vec<C> = raw.iter().copied().map(C::encode).collect();
        for uniform in [None, Some(3)] {
            let lookup = uniform
                .is_none()
                .then(|| WeightLookup::new(block, corpus.len()));
            for workers in [1, 4] {
                for floor in [1, 7] {
                    let mut expected = std::collections::BTreeMap::<u64, (u64, Vec<u32>)>::new();
                    let mut routed_counts = vec![0_usize; workers];
                    for p in 0..raw.len().saturating_sub(1) {
                        let (a, b) = (raw[p], raw[p + 1]);
                        if a == NONE || b == NONE {
                            continue;
                        }
                        let k = key(a, b);
                        routed_counts[owner(k, workers)] += 1;
                        let entry = expected.entry(k).or_default();
                        entry.0 += block.weight(p, uniform);
                        entry.1.push(p as u32);
                    }
                    let discarded = expected.values().filter(|e| e.0 < floor).count();
                    expected.retain(|_, e| e.0 >= floor);
                    for width in [1, 2, 3, 4] {
                        let mut owners: Vec<_> = (0..workers).map(|_| Owner::default()).collect();
                        let metrics = initialize_with_owner_width(
                            &corpus,
                            block,
                            uniform,
                            lookup.as_ref(),
                            workers,
                            floor,
                            &mut owners,
                            width,
                        )
                        .unwrap();
                        let actual: std::collections::BTreeMap<_, _> = owners
                            .iter()
                            .flat_map(|o| {
                                o.entries
                                    .iter()
                                    .map(|(&k, e)| (k, (e.frequency, e.blocks.as_slice().to_vec())))
                            })
                            .collect();
                        assert_eq!(
                            actual, expected,
                            "workers={workers}, width={width}, floor={floor}"
                        );
                        assert_eq!(metrics.pruned, discarded);
                        assert_eq!(metrics.route_bytes, routed_counts.iter().sum::<usize>() * 8);
                        let scratch_peak = routed_counts
                            .chunks(width.min(workers))
                            .map(|wave| wave.iter().filter(|&&n| n >= 2).sum::<usize>() * 8)
                            .max()
                            .unwrap();
                        assert_eq!(metrics.scratch_bytes, scratch_peak);
                    }
                }
            }
        }
    }

    #[test]
    fn owner_waves_preserve_weighted_frequencies_and_ordered_positions() {
        let mut raw = Vec::new();
        let mut block = Block::<u32, 2>::new(0, 1);
        let mut seed = 917_u64;
        for word in 0..96 {
            block.pivots.push(raw.len() as u32);
            block.weights.push(1 + word % 5);
            let length = 1 + (word * 7) % 23;
            for p in 0..length {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let mut id = [0, 1, 65535, 2000, 32768, 42][(seed >> 32) as usize % 6];
                if word % 7 == 0 {
                    id = 0;
                }
                if word % 9 == 0 && p == length / 2 {
                    id = 3000 + word as u32;
                }
                raw.push(id);
            }
            raw.push(NONE);
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| {
                check_wave_oracle::<u32>(&raw, &block);
                check_wave_oracle::<AtomicU32>(&raw, &block);
                check_wave_oracle::<u32>(&[], &Block::new(0, 1));
                check_wave_oracle::<u32>(&[NONE], &Block::new(0, 1));
            });
    }

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
