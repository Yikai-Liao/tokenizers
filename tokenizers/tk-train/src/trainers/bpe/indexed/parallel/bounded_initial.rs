//! Stable full-key grouping in spatial tiles; transient occurrence storage is
//! bounded independently of a physical block's size. Final postings remain.
use super::radix_count::block_radix;
use super::*;
const TILE_RECORDS: usize = 1 << 18;
// The table's actual cardinality decides whether to pay for grouping. This
// keeps the small-table spatial scanner while bounding large-table sort tiles.
const SORT_MIN_KEYS: usize = 1 << 16;
#[derive(Default)]
pub(super) struct Metrics {
    pub(super) tiles: usize,
    pub(super) groups: usize,
    pub(super) hash_edges: usize,
    pub(super) buffer_bound_bytes: usize,
}
pub(super) struct Initialized {
    pub(super) routes: Vec<Vec<(u64, u64)>>,
    pub(super) metrics: Metrics,
}
pub(super) fn initialize<C: Slot, O: Offset, const INLINE: usize>(
    corpus: &[C],
    block: &mut Block<O, INLINE>,
    end: usize,
    uniform: Option<u64>,
    workers: usize,
) -> Result<Initialized> {
    initialize_with_threshold(corpus, block, end, uniform, workers, SORT_MIN_KEYS)
}
pub(super) fn initialize_with_threshold<C: Slot, O: Offset, const INLINE: usize>(
    corpus: &[C],
    block: &mut Block<O, INLINE>,
    end: usize,
    uniform: Option<u64>,
    workers: usize,
    sort_min_keys: usize,
) -> Result<Initialized> {
    let mut metrics = Metrics::default();
    let mut extra_weights = AHashMap::<u64, i64>::new();
    let mut records = Vec::new();
    let mut word = 0;
    for start in (block.base..end).step_by(TILE_RECORDS) {
        let grouped = block.postings.len() >= sort_min_keys;
        records.clear();
        if grouped && records.capacity() == 0 {
            records.reserve_exact(TILE_RECORDS.min(end - start));
        }
        for p in start..end.min(start + TILE_RECORDS) {
            let a = corpus[p].token();
            let b = corpus[p + 1].token();
            if a == NONE || b == NONE {
                continue;
            }
            let local = (p - block.base) as u32;
            let weight = if let Some(weight) = uniform {
                weight
            } else {
                while word < block.pivots.len() && block.pivots[word] <= local {
                    word += 1;
                }
                if word == 0 {
                    block.previous_weight
                } else {
                    block.weights[word - 1]
                }
            };
            let k = key(a, b);
            // Zero-weight edges still generate postings and global summaries.
            if uniform.is_none() && weight != 1 {
                let delta =
                    i64::try_from(weight).map_err(|_| "indexed BPE weight exceeds i64::MAX")? - 1;
                let extra = extra_weights.entry(k).or_default();
                *extra = extra
                    .checked_add(delta)
                    .ok_or("initial weight correction exceeds i64")?;
            }
            if grouped {
                records.push((u128::from(k) << 64) | u128::from(local));
            } else {
                metrics.hash_edges += 1;
                block
                    .postings
                    .entry(k)
                    .or_default()
                    .push(O::encode(local as usize))?;
            }
        }
        if !grouped {
            continue;
        }
        let scratch = block_radix::sort_wide(&mut records);
        metrics.tiles += 1;
        metrics.buffer_bound_bytes = metrics
            .buffer_bound_bytes
            .max(records.capacity() * 16 + scratch);
        let mut begin = 0;
        while begin < records.len() {
            let k = (records[begin] >> 64) as u64;
            let mut after = begin + 1;
            while after < records.len() && (records[after] >> 64) as u64 == k {
                after += 1;
            }
            metrics.groups += 1;
            let positions = block.postings.entry(k).or_default();
            // Tiles visit ascending physical intervals; stable grouping keeps
            // equal keys in ascending local-address order within each tile.
            for &record in &records[begin..after] {
                positions.push(O::encode(record as u32 as usize))?;
            }
            begin = after;
        }
    }
    // Sorting storage is no longer needed when block-key summaries are built.
    drop(records);
    let mut routed: Vec<Vec<(u64, u64)>> = (0..workers).map(|_| Vec::new()).collect();
    for (&k, positions) in &block.postings {
        let weight = if let Some(weight) = uniform {
            (positions.len() as u64)
                .checked_mul(weight)
                .ok_or("initial frequency exceeds u64")?
        } else {
            let frequency = (positions.len() as i64)
                .checked_add(extra_weights.get(&k).copied().unwrap_or(0))
                .ok_or("initial frequency exceeds i64")?;
            u64::try_from(frequency).map_err(|_| "negative initial frequency")?
        };
        routed[owner(k, workers)].push((k, weight));
    }
    Ok(Initialized {
        routes: routed,
        metrics,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn adaptive_scan_switches_only_after_the_table_grows() {
        let end = TILE_RECORDS * 2 + 513;
        for large in [false, true] {
            let corpus: Vec<u32> = (0..=end)
                .map(|i| {
                    if large && i % 131072 < 70000 {
                        100_000 + (i % 131072) as u32
                    } else {
                        [7, 7, 7, 9, NONE][i % 5]
                    }
                })
                .collect();
            let mut block = Block::<u32, 2>::new(0, 1);
            let got = initialize(&corpus, &mut block, end, Some(1), 4).unwrap();
            let mut expected = AHashMap::<u64, Vec<u32>>::new();
            for p in 0..end {
                if corpus[p] != NONE && corpus[p + 1] != NONE {
                    expected
                        .entry(key(corpus[p], corpus[p + 1]))
                        .or_default()
                        .push(p as u32);
                }
            }
            for (k, positions) in &expected {
                assert_eq!(block.postings[k].as_slice(), positions);
            }
            assert_eq!(block.postings.len(), expected.len());
            let frequencies: AHashMap<_, _> = got.routes.into_iter().flatten().collect();
            assert_eq!(
                frequencies,
                expected.iter().map(|(&k, v)| (k, v.len() as u64)).collect()
            );
            assert!(got.metrics.hash_edges > 0);
            if large {
                assert_eq!(got.metrics.tiles, 2);
                assert!(got.metrics.groups > 0);
                assert!(got.metrics.buffer_bound_bytes > 0);
            } else {
                assert_eq!(
                    got.metrics.hash_edges,
                    expected.values().map(|v| v.len()).sum::<usize>()
                );
                assert_eq!(got.metrics.tiles, 0);
                assert_eq!(got.metrics.groups, 0);
                assert_eq!(got.metrics.buffer_bound_bytes, 0);
            }
        }
    }

    #[test]
    fn tile_boundaries_full_keys_zero_weights_and_sorted_postings() {
        let base = 37;
        let end = base + TILE_RECORDS * 2 + 513;
        let corpus: Vec<u32> = (0..=end)
            .map(|i| match i % 11 {
                0 => NONE,
                1..=5 => 100_003,
                6 => u32::MAX - 1,
                _ => 7,
            })
            .collect();
        for uniform in [None, Some(0), Some(1), Some(7)] {
            let mut block = Block::<u32, 2>::new(base, 1);
            block.pivots = vec![91, TILE_RECORDS as u32 - 1, TILE_RECORDS as u32 + 2];
            block.weights = vec![0, (u32::MAX as u64) + 3, 1];
            let mut expected = std::collections::BTreeMap::<u64, (Vec<u32>, u64)>::new();
            for p in base..end {
                if corpus[p] == NONE || corpus[p + 1] == NONE {
                    continue;
                }
                let entry = expected.entry(key(corpus[p], corpus[p + 1])).or_default();
                entry.0.push((p - base) as u32);
                entry.1 += block.weight(p, uniform);
            }
            let got = initialize_with_threshold(&corpus, &mut block, end, uniform, 4, 0).unwrap();
            let counts: std::collections::BTreeMap<_, _> =
                got.routes.into_iter().flatten().collect();
            assert_eq!(
                counts,
                expected.iter().map(|(&k, (_, f))| (k, *f)).collect()
            );
            assert_eq!(block.postings.len(), expected.len());
            for (k, (positions, _)) in expected {
                assert_eq!(block.postings[&k].as_slice(), positions);
            }
            assert_eq!(got.metrics.tiles, 3);
            let maximum = TILE_RECORDS * 16 + 512 * 128 * 16 + (TILE_RECORDS / 128 + 512) * 9;
            assert!(got.metrics.buffer_bound_bytes <= maximum);
        }
    }
}
