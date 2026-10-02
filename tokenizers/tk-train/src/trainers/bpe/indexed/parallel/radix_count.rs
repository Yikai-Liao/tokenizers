//! Group the initial pair stream before installing retained owner entries.
//! The caller checks flat u32 positions and the complete initial u16 ID domain;
//! the final corpus, pair keys and postings keep their canonical full types.
use super::weight_lookup::WeightLookup;
use super::*;

#[path = "block_radix.rs"]
pub(super) mod block_radix;
#[path = "owner_route.rs"]
mod owner_route;

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
fn sort_classic(records: &mut Vec<u64>) -> usize {
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

// Prefer the simpler full-buffer scatter when its scratch is smaller than
// the fixed block implementation. The comparison follows allocation sizes.
#[cfg(test)]
fn sorting_scratch(length: usize) -> usize {
    if length < 2 {
        return 0;
    }
    (length * 8).min(2 * 256 * 512 * 8 + (length / 512 + 512) * 9)
}
fn sort(records: &mut Vec<u64>) -> usize {
    if records.len() * 8 <= 2 * 256 * 512 * 8 + (records.len() / 512 + 512) * 9 {
        sort_classic(records)
    } else {
        block_radix::sort(records)
    }
}

pub(super) fn cohorts<C: Slot>(
    corpus: &[C],
    pivots: &[u32],
    weights: &[u64],
    workers: usize,
    sorted_weights: bool,
) -> Result<(AHashMap<Pair, i64>, OctonaryHeap<super::super::Candidate>)> {
    let mut block = Block::<u32, 2>::new(0, 0);
    if sorted_weights {
        block.weight_intervals = true;
        for (&pivot, &weight) in pivots.iter().zip(weights) {
            if block.weights.last() != Some(&weight) {
                block.pivots.push(pivot);
                block.weights.push(weight);
            }
        }
    } else {
        block.pivots = pivots.to_vec();
        block.weights = weights.to_vec();
    }
    let uniform = weights
        .first()
        .copied()
        .filter(|&w| weights.iter().all(|&n| n == w));
    let lookup = uniform
        .is_none()
        .then(|| WeightLookup::new(&block, corpus.len()));
    let routed = owner_route::direct(corpus, workers);
    let grouped: Vec<_> = routed
        .into_par_iter()
        .map(|mut records| -> Result<_> {
            sort(&mut records);
            let mut rows = Vec::new();
            let mut start = 0;
            while start < records.len() {
                let code = (records[start] >> 32) as u32;
                let mut end = start + 1;
                while end < records.len() && (records[end] >> 32) as u32 == code {
                    end += 1;
                }
                let count = if let Some(weight) = uniform {
                    weight
                        .checked_mul((end - start) as u64)
                        .ok_or("cohort frequency exceeds u64")?
                } else {
                    let lookup = lookup.as_ref().unwrap();
                    records[start..end]
                        .iter()
                        .map(|&r| lookup.weight(&block, r as u32))
                        .sum()
                };
                let signed = i64::try_from(count).map_err(|_| "cohort frequency exceeds i64")?;
                let mut positions = SmallPosting::default();
                if count > 0 {
                    let size =
                        u32::try_from(end - start).map_err(|_| "cohort posting exceeds u32")?;
                    positions = SmallPosting::with_capacity(size)?;
                    let mut cursor = end;
                    positions.append_reversed_reserved(size, || {
                        cursor -= 1;
                        records[cursor] as u32
                    })?;
                }
                rows.push(((code >> 16, code & 0xffff), signed, positions));
                start = end;
            }
            Ok(rows)
        })
        .collect::<Result<Vec<_>>>()?;
    let pairs = grouped.iter().map(Vec::len).sum();
    let mut counts = AHashMap::with_capacity(pairs);
    let mut queued = Vec::with_capacity(pairs);
    for rows in grouped {
        for (pair, count, positions) in rows {
            counts.insert(pair, count);
            if count > 0 {
                queued.push(super::super::Candidate {
                    pair,
                    count: count as u64,
                    positions,
                });
            }
        }
    }
    Ok((counts, queued.into_iter().collect()))
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
    initialize_with_widths(
        corpus,
        block,
        uniform,
        lookup,
        workers,
        floor,
        owners,
        owner_width,
        workers,
    )
}

// Keep a bounded number of sorting buffers alive. Finish each wave's postings
// before starting the next wave, so consumed owner records are released early.
fn initialize_with_widths<C: Slot, O: Offset, const INLINE: usize>(
    corpus: &[C],
    block: &Block<O, INLINE>,
    uniform: Option<u64>,
    lookup: Option<&WeightLookup>,
    workers: usize,
    floor: u64,
    owners: &mut [Owner],
    owner_width: usize,
    sort_width: usize,
) -> Result<Metrics> {
    initialize_core(
        corpus,
        workers,
        floor,
        owners,
        owner_width,
        sort_width,
        32,
        0,
        |p| {
            uniform.unwrap_or_else(|| {
                lookup.map_or_else(|| block.weight(p, None), |l| l.weight(block, p as u32))
            })
        },
    )
}

pub(super) fn initialize_segmented<C: Slot, O: Offset, const INLINE: usize>(
    corpus: &[C],
    blocks: &[Block<O, INLINE>],
    uniform: Option<u64>,
    lookups: Option<&[WeightLookup]>,
    workers: usize,
    floor: u64,
    owners: &mut [Owner],
    bits: u8,
) -> Result<Metrics> {
    initialize_segmented_waves(
        corpus,
        blocks,
        uniform,
        lookups,
        workers,
        floor,
        owners,
        bits,
        1 << 28,
    )
}

fn initialize_segmented_waves<C: Slot, O: Offset, const INLINE: usize>(
    corpus: &[C],
    blocks: &[Block<O, INLINE>],
    uniform: Option<u64>,
    lookups: Option<&[WeightLookup]>,
    workers: usize,
    floor: u64,
    owners: &mut [Owner],
    bits: u8,
    wave_slots: usize,
) -> Result<Metrics> {
    assert!(wave_slots > 1 && wave_slots <= (1usize << 32));
    let mut total = Metrics::default();
    for base in (0..corpus.len()).step_by(wave_slots) {
        // Include a one-slot read halo, so a word crossing a wave is counted once.
        let end = corpus
            .len()
            .min(base.saturating_add(wave_slots).saturating_add(1));
        let mut wave = (0..workers).map(|_| Owner::default()).collect::<Vec<_>>();
        let m = if blocks.len() == 1 {
            initialize_core(
                &corpus[base..end],
                workers,
                if corpus.len() <= wave_slots { floor } else { 1 },
                &mut wave,
                workers.min(2),
                workers,
                bits,
                base,
                |p| {
                    uniform.unwrap_or_else(|| {
                        lookups.map_or_else(
                            || blocks[0].weight(p, None),
                            |l| l[0].weight(&blocks[0], p as u32),
                        )
                    })
                },
            )?
        } else {
            initialize_core(
                &corpus[base..end],
                workers,
                if corpus.len() <= wave_slots { floor } else { 1 },
                &mut wave,
                workers.min(2),
                workers,
                bits,
                base,
                |p| {
                    uniform.unwrap_or_else(|| {
                        let b = p >> bits;
                        lookups.map_or_else(
                            || blocks[b].weight(p, None),
                            |l| l[b].weight(&blocks[b], (p - blocks[b].base) as u32),
                        )
                    })
                },
            )?
        };
        total.route_ms += m.route_ms;
        total.count_ms += m.count_ms;
        total.sort_ms += m.sort_ms;
        total.group_ms += m.group_ms;
        total.install_ms += m.install_ms;
        total.route_bytes = total.route_bytes.max(m.route_bytes);
        total.peak_route_bytes = total.peak_route_bytes.max(m.peak_route_bytes);
        total.scratch_bytes = total.scratch_bytes.max(m.scratch_bytes);
        total.group_bytes = total.group_bytes.max(m.group_bytes);
        let install = Instant::now();
        owners
            .par_iter_mut()
            .zip(wave.into_par_iter())
            .map(|(ledger, wave)| -> Result<()> {
                if ledger.entries.is_empty() {
                    ledger.entries = wave.entries;
                    return Ok(());
                }
                for (k, e) in wave.entries {
                    match ledger.entries.entry(k) {
                        std::collections::hash_map::Entry::Vacant(v) => {
                            v.insert(e);
                        }
                        std::collections::hash_map::Entry::Occupied(mut v) => {
                            let old = v.get_mut();
                            old.frequency = old
                                .frequency
                                .checked_add(e.frequency)
                                .ok_or("initial frequency exceeds u64")?;
                            old.blocks.append(e.blocks)?;
                        }
                    }
                }
                Ok(())
            })
            .collect::<Result<Vec<_>>>()?;
        let ms = install.elapsed().as_secs_f64() * 1000.0;
        total.count_ms += ms;
        total.install_ms += ms;
    }
    for ledger in owners {
        let before = ledger.entries.len();
        ledger.entries.retain(|_, e| e.frequency >= floor);
        total.pruned += before - ledger.entries.len();
    }
    Ok(total)
}

fn initialize_core<C: Slot>(
    corpus: &[C],
    workers: usize,
    floor: u64,
    owners: &mut [Owner],
    owner_width: usize,
    sort_width: usize,
    bits: u8,
    base: usize,
    weight: impl Fn(usize) -> u64 + Sync,
) -> Result<Metrics> {
    assert!(owner_width > 0 && sort_width > 0);
    debug_assert_eq!(owners.len(), workers);
    let owner_width = owner_width.min(workers);
    let begin = Instant::now();
    let mut routed = owner_route::direct(corpus, workers);
    let final_bytes: usize = routed.iter().map(|r| r.capacity() * 8).sum();
    let mut peak = final_bytes;
    let compact_ms = 0.0;
    let route_ms = begin.elapsed().as_secs_f64() * 1000.0;
    let count_begin = Instant::now();
    let mut remaining_record_bytes = final_bytes;
    let mut scratch_bytes = 0;
    let mut group_bytes = 0;
    let mut sort_ms = 0.0;
    let mut group_ms = 0.0;
    let mut install_ms = 0.0;
    let mut pruned = 0;
    // Block scratch is small enough for every owner to sort concurrently.
    // Bound installation independently: final postings overlap remaining records.
    for records_wave in routed.chunks_mut(sort_width.min(workers)) {
        let sorting = Instant::now();
        let wave_scratch_bytes = records_wave.par_iter_mut().map(sort).sum();
        sort_ms += sorting.elapsed().as_secs_f64() * 1000.0;
        scratch_bytes = scratch_bytes.max(wave_scratch_bytes);
        peak = peak.max(remaining_record_bytes + wave_scratch_bytes);
    }
    for (owner_wave, records_wave) in owners
        .chunks_mut(owner_width)
        .zip(routed.chunks_mut(owner_width))
    {
        let wave_record_bytes: usize = records_wave.iter().map(|r| r.capacity() * 8).sum();
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
                    let frequency = records[start..end]
                        .iter()
                        .map(|&r| weight(base + r as u32 as usize))
                        .sum::<u64>();
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
                    let source = &records[group.start as usize..group.end as usize];
                    let mut next = source.len();
                    let first = (base + source[0] as u32 as usize) >> bits;
                    let last = (base + source[source.len() - 1] as u32 as usize) >> bits;
                    let next = move || {
                        next -= 1;
                        base + source[next] as u32 as usize
                    };
                    let positions = if first == last {
                        BlockPosting::from_reversed_in_block(count, first as u32, bits, next)?
                    } else {
                        BlockPosting::from_reversed(count, bits, next)?
                    };
                    debug_assert!(
                        positions
                            .iter(bits)
                            .collect::<Vec<_>>()
                            .windows(2)
                            .all(|w| w[0] < w[1])
                    );
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

    #[test]
    fn segmented_waves_preserve_cross_wave_edges_and_global_frequency_floor() {
        let raw = (0..515)
            .map(|i| if i % 23 == 0 { NONE } else { (i % 7) as u32 })
            .collect::<Vec<_>>();
        let mut expected = std::collections::BTreeMap::<u64, Vec<usize>>::new();
        for p in 0..raw.len() - 1 {
            if raw[p] != NONE && raw[p + 1] != NONE {
                expected.entry(key(raw[p], raw[p + 1])).or_default().push(p);
            }
        }
        for bits in [4, 32] {
            let blocks = (0..raw.len().div_ceil(1usize << bits))
                .map(|b| Block::<u32, 2>::new(b << bits, 1))
                .collect::<Vec<_>>();
            for wave in [17, 64, 1024] {
                let mut owners = (0..4).map(|_| Owner::default()).collect::<Vec<_>>();
                initialize_segmented_waves(
                    &raw,
                    &blocks,
                    Some(3),
                    None,
                    4,
                    10,
                    &mut owners,
                    bits,
                    wave,
                )
                .unwrap();
                let got = owners
                    .iter()
                    .flat_map(|o| o.entries.iter())
                    .map(|(&k, e)| (k, (e.frequency, e.blocks.iter(bits).collect::<Vec<_>>())))
                    .collect::<std::collections::BTreeMap<_, _>>();
                assert_eq!(got.len(), expected.len());
                for (k, p) in &expected {
                    assert_eq!(&got[k], &(p.len() as u64 * 3, p.clone()));
                }
            }
        }
    }

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
                            .chunks(workers)
                            .map(|wave| wave.iter().map(|&n| sorting_scratch(n)).sum::<usize>())
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
