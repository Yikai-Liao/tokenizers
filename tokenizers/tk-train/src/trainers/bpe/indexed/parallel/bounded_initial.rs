//! Stable full-key grouping in spatial tiles; transient occurrence storage is
//! bounded independently of a physical block's size. Final postings remain.
use super::radix_count::block_radix;
use super::*;
const TILE_RECORDS: usize = 1 << 18;
#[derive(Default)]
pub(super) struct Metrics {
    pub(super) tiles: usize,
    pub(super) groups: usize,
    pub(super) buffer_bound_bytes: usize,
}
pub(super) struct Initialized {
    pub(super) routes: Vec<Vec<(u64, u64)>>,
    pub(super) metrics: Metrics,
}
trait InitialRecord: Copy {
    fn encode(key: u64, local: u32) -> Self;
    fn key(self) -> u64;
    fn local(self) -> usize;
    fn sort(records: &mut Vec<Self>) -> usize;
}
impl InitialRecord for u128 {
    fn encode(key: u64, local: u32) -> Self {
        (u128::from(key) << 64) | u128::from(local)
    }
    fn key(self) -> u64 {
        (self >> 64) as u64
    }
    fn local(self) -> usize {
        self as u32 as usize
    }
    fn sort(records: &mut Vec<Self>) -> usize {
        block_radix::sort_wide(records)
    }
}
impl InitialRecord for u64 {
    fn encode(key: u64, local: u32) -> Self {
        debug_assert!(key >> 48 == 0 && key as u32 <= u16::MAX as u32);
        let code = ((key >> 32) << 16) | (key & 0xffff);
        (code << 32) | u64::from(local)
    }
    fn key(self) -> u64 {
        let code = self >> 32;
        ((code >> 16) << 32) | (code & 0xffff)
    }
    fn local(self) -> usize {
        self as u32 as usize
    }
    fn sort(records: &mut Vec<Self>) -> usize {
        block_radix::sort_compact(records)
    }
}
// Compact records are selected only after checking the complete initial ID
// domain. Canonical keys and global addresses retain their original widths.
pub(super) fn initialize<C: Slot, O: Offset, const INLINE: usize>(
    corpus: &[C],
    block: &mut Block<O, INLINE>,
    end: usize,
    uniform: Option<u64>,
    workers: usize,
    identities: usize,
) -> Result<Initialized> {
    if identities <= u16::MAX as usize + 1 {
        initialize_typed::<C, O, INLINE, u64>(corpus, block, end, uniform, workers)
    } else {
        initialize_wide(corpus, block, end, uniform, workers)
    }
}
pub(super) fn initialize_wide<C: Slot, O: Offset, const INLINE: usize>(
    corpus: &[C],
    block: &mut Block<O, INLINE>,
    end: usize,
    uniform: Option<u64>,
    workers: usize,
) -> Result<Initialized> {
    initialize_typed::<C, O, INLINE, u128>(corpus, block, end, uniform, workers)
}
fn initialize_typed<C: Slot, O: Offset, const INLINE: usize, R: InitialRecord>(
    corpus: &[C],
    block: &mut Block<O, INLINE>,
    end: usize,
    uniform: Option<u64>,
    workers: usize,
) -> Result<Initialized> {
    let mut metrics = Metrics::default();
    let mut extra_weights = AHashMap::<u64, i64>::new();
    let mut records = Vec::with_capacity(TILE_RECORDS.min(end.saturating_sub(block.base)));
    let mut word = 0;
    for start in (block.base..end).step_by(TILE_RECORDS) {
        records.clear();
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
            records.push(R::encode(k, local));
        }
        let scratch = R::sort(&mut records);
        metrics.tiles += 1;
        metrics.buffer_bound_bytes = metrics
            .buffer_bound_bytes
            .max(records.capacity() * std::mem::size_of::<R>() + scratch);
        let mut begin = 0;
        while begin < records.len() {
            let k = records[begin].key();
            let mut after = begin + 1;
            while after < records.len() && records[after].key() == k {
                after += 1;
            }
            metrics.groups += 1;
            let positions = block.postings.entry(k).or_default();
            // Tiles visit ascending physical intervals; stable grouping keeps
            // equal keys in ascending local-address order within each tile.
            for &record in &records[begin..after] {
                positions.push(O::encode(record.local()))?;
            }
            begin = after;
        }
    }
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
    fn compact_records_keep_both_ids_and_the_full_local_address() {
        for left in [0, 1, 255, 256, u16::MAX as u32] {
            for right in [0, 1, 255, 256, u16::MAX as u32] {
                for local in [0, 1, 1 << 31, u32::MAX] {
                    let canonical = key(left, right);
                    let record = <u64 as InitialRecord>::encode(canonical, local);
                    assert_eq!(record.key(), canonical);
                    assert_eq!(record.local(), local as usize);
                }
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
            let got = initialize(&corpus, &mut block, end, uniform, 4, u32::MAX as usize).unwrap();
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
