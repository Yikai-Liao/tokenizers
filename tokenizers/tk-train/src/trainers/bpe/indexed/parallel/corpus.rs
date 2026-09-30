//! Build ordered corpus regions directly in their final allocation.
use super::*;

#[derive(Default)]
pub(super) struct Timings {
    pub measure_ms: f64,
    pub allocate_ms: f64,
    pub fill_ms: f64,
}

pub(super) struct Prepared<C: Slot, O: Offset, const INLINE: usize> {
    pub slots: Vec<C>,
    pub lengths: Vec<usize>,
    pub blocks: Vec<Block<O, INLINE>>,
    pub uniform: Option<u64>,
    pub symbols: usize,
    pub edges: usize,
    pub timings: Timings,
}

struct Region<'a> {
    words: &'a [(&'a CompactString, u64)],
    slots: usize,
    symbols: usize,
    edges: usize,
    weighted_edges: i64,
}

struct Filled {
    starts: Vec<(usize, u64)>,
    active: Vec<u64>,
}

fn retained(word: &str, ids: &AHashMap<CompactString, u32>, unfiltered: bool) -> usize {
    if unfiltered {
        word.chars().count()
    } else {
        word.chars()
            .filter(|c| {
                let mut utf8 = [0; 4];
                ids.contains_key(c.encode_utf8(&mut utf8) as &str)
            })
            .count()
    }
}

fn fill<C: Slot>(
    slots: &mut [C],
    base: usize,
    regions: &[Region<'_>],
    ids: &AHashMap<CompactString, u32>,
    identities: usize,
) -> Vec<Filled> {
    if regions.len() > 1 {
        let middle = regions.len() / 2;
        let cut = regions[..middle].iter().map(|r| r.slots).sum();
        let (left, right) = slots.split_at_mut(cut);
        let (mut a, mut b) = rayon::join(
            || fill(left, base, &regions[..middle], ids, identities),
            || fill(right, base + cut, &regions[middle..], ids, identities),
        );
        a.append(&mut b);
        return a;
    }
    let region = &regions[0];
    let mut result = Filled {
        starts: Vec::with_capacity(region.words.len()),
        active: vec![0; identities.div_ceil(64)],
    };
    let mut position = 0;
    for &(word, weight) in region.words {
        result.starts.push((base + position, weight));
        for c in word.chars() {
            let mut utf8 = [0; 4];
            if let Some(&id) = ids.get(c.encode_utf8(&mut utf8) as &str) {
                slots[position].set(id);
                position += 1;
                let id = id as usize;
                result.active[id / 64] |= 1_u64 << (id % 64);
            }
        }
        slots[position].set(NONE);
        position += 1;
    }
    assert_eq!(position, slots.len());
    vec![result]
}

pub(super) fn build<C: Slot, O: Offset, const INLINE: usize>(
    wc: &AHashMap<CompactString, u64>,
    ids: &AHashMap<CompactString, u32>,
    identities: usize,
    unfiltered: bool,
    bits: u8,
    workers: usize,
) -> Result<Prepared<C, O, INLINE>> {
    let measure = Instant::now();
    // Capture the original map traversal once; region scheduling cannot change
    // word order, canonical IDs, or left-to-right AA boundaries.
    let words: Vec<_> = wc.iter().map(|(word, &weight)| (word, weight)).collect();
    let chunk = words
        .len()
        .div_ceil(workers.saturating_mul(8).max(1))
        .max(1);
    let regions: Vec<Region<'_>> = words
        .par_chunks(chunk)
        .map(|words| -> Result<_> {
            let mut region = Region {
                words,
                slots: 0,
                symbols: 0,
                edges: 0,
                weighted_edges: 0,
            };
            for &(word, weight) in words {
                let signed =
                    i64::try_from(weight).map_err(|_| "indexed BPE weight exceeds i64::MAX")?;
                let count = retained(word, ids, unfiltered);
                region.slots = region
                    .slots
                    .checked_add(count)
                    .and_then(|n| n.checked_add(1))
                    .ok_or("corpus size exceeds usize")?;
                region.symbols += count;
                let edges = count.saturating_sub(1);
                region.edges += edges;
                region.weighted_edges = region
                    .weighted_edges
                    .checked_add(
                        signed
                            .checked_mul(
                                i64::try_from(edges).map_err(|_| "word edge count exceeds i64")?,
                            )
                            .ok_or("weighted pair counts exceed i64::MAX")?,
                    )
                    .ok_or("weighted pair counts exceed i64::MAX")?;
            }
            Ok(region)
        })
        .collect::<Result<_>>()?;
    let mut capacity = 1_usize;
    let mut weighted_edges = 0_i64;
    let mut symbols = 0;
    let mut edges = 0;
    for r in &regions {
        capacity = capacity
            .checked_add(r.slots)
            .ok_or("corpus size exceeds usize")?;
        weighted_edges = weighted_edges
            .checked_add(r.weighted_edges)
            .ok_or("weighted pair counts exceed i64::MAX")?;
        symbols += r.symbols;
        edges += r.edges;
    }
    let block_size = 1_usize
        .checked_shl(bits as u32)
        .ok_or("posting blocks require 64-bit usize")?;
    if capacity.div_ceil(block_size) > u32::MAX as usize {
        return Err("posting block directory exceeds u32 blocks".into());
    }
    let uniform = words
        .first()
        .map(|w| w.1)
        .filter(|&weight| words.iter().all(|w| w.1 == weight));
    let measure_ms = measure.elapsed().as_secs_f64() * 1000.0;

    let allocate = Instant::now();
    // Initialize the final allocation once. There is no per-worker corpus copy.
    let mut slots = Vec::with_capacity(capacity);
    slots.resize_with(capacity, C::default);
    slots[0].set(NONE);
    let allocate_ms = allocate.elapsed().as_secs_f64() * 1000.0;

    let filling = Instant::now();
    let outputs = if regions.is_empty() {
        Vec::new()
    } else {
        fill(&mut slots[1..], 1, &regions, ids, identities)
    };
    let mut lengths = vec![0; identities];
    let mut blocks = vec![Block::new(0, 0)];
    // Region order is stable. Rebuild only small boundary metadata after all
    // disjoint writes have joined; a long word keeps its weight across blocks.
    let starts = outputs.iter().flat_map(|o| o.starts.iter()).copied();
    let mut starts = starts.peekable();
    while let Some((start, weight)) = starts.next() {
        while blocks.len() <= start >> bits {
            blocks.push(Block::new(blocks.len() << bits, weight));
        }
        if uniform.is_none() {
            let block = &mut blocks[start >> bits];
            block.pivots.push((start - block.base) as u32);
            block.weights.push(weight);
        }
        let end = starts.peek().map_or(capacity - 1, |p| p.0 - 1);
        while blocks.len() <= end >> bits {
            blocks.push(Block::new(blocks.len() << bits, weight));
        }
    }
    for output in outputs {
        for (word, mut active) in output.active.into_iter().enumerate() {
            while active != 0 {
                lengths[word * 64 + active.trailing_zeros() as usize] = 1;
                active &= active - 1;
            }
        }
    }
    Ok(Prepared {
        slots,
        lengths,
        blocks,
        uniform,
        symbols,
        edges,
        timings: Timings {
            measure_ms,
            allocate_ms,
            fill_ms: filling.elapsed().as_secs_f64() * 1000.0,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check<C: Slot>(pool: &rayon::ThreadPool, unfiltered: bool) {
        let words: AHashMap<CompactString, u64> = [
            ("", 3),
            ("aaa中🙂".repeat(37).as_str(), 7),
            ("中中bb", 11),
            ("🙂", 13),
        ]
        .into_iter()
        .map(|(word, weight)| (CompactString::from(word), weight))
        .collect();
        let mut ids: AHashMap<CompactString, u32> = [("a", 2), ("b", 3), ("中", 91), ("🙂", 127)]
            .into_iter()
            .map(|(s, id)| (s.into(), id))
            .collect();
        if !unfiltered {
            ids.remove("中");
            ids.remove("🙂");
        }
        let mut expected = vec![NONE];
        let mut weights = vec![0];
        let mut lengths = vec![0; 129];
        let mut edges = 0;
        let mut symbols = 0;
        for (word, &weight) in &words {
            let start = expected.len();
            for c in word.chars() {
                let mut utf8 = [0; 4];
                if let Some(&id) = ids.get(c.encode_utf8(&mut utf8) as &str) {
                    expected.push(id);
                    weights.push(weight);
                    lengths[id as usize] = 1;
                }
            }
            let retained = expected.len() - start;
            symbols += retained;
            edges += retained.saturating_sub(1);
            expected.push(NONE);
            weights.push(weight);
        }
        // Small address blocks exercise long words and separators crossing
        // blocks without allocating a multi-gigabyte corpus.
        let got = pool
            .install(|| {
                build::<C, u16, 4>(&words, &ids, 129, unfiltered, 4, pool.current_num_threads())
            })
            .unwrap();
        assert_eq!(
            got.slots.iter().map(Slot::token).collect::<Vec<_>>(),
            expected
        );
        assert_eq!(got.lengths, lengths);
        assert_eq!(got.symbols, symbols);
        assert_eq!(got.edges, edges);
        assert_eq!(got.slots.capacity(), got.slots.len());
        for (p, &weight) in weights.iter().enumerate().skip(1) {
            assert_eq!(got.blocks[p >> 4].weight(p, got.uniform), weight);
        }
        let empty = pool
            .install(|| build::<C, u16, 4>(&AHashMap::new(), &ids, 129, unfiltered, 4, 4))
            .unwrap();
        assert_eq!(empty.slots.len(), 1);
        assert_eq!(empty.slots[0].token(), NONE);
        assert_eq!(empty.lengths, vec![0; 129]);
    }

    #[test]
    fn ordered_regions_preserve_filtered_slots_weights_and_reserved_id_activation() {
        for workers in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            for unfiltered in [false, true] {
                check::<u32>(&pool, unfiltered);
                check::<u16>(&pool, unfiltered);
                check::<AtomicU32>(&pool, unfiltered);
                check::<AtomicU16>(&pool, unfiltered);
            }
        }
    }
}
