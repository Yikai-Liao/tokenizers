//! Build ordered corpus regions directly in their final allocation.
use super::*;
use std::mem::{ManuallyDrop, MaybeUninit};

#[derive(Default)]
pub(super) struct Timings {
    pub measure_ms: f64,
    /// Nested inside measure_ms.
    pub sort_ms: f64,
    pub sort_buffer_bytes: usize,
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
    pub weighted_edges: u64,
    pub timings: Timings,
    pub character_table_bytes: usize,
    pub word_reference_bytes: usize,
    pub temporary_weight_bytes: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Order {
    Original,
    WeightSorted,
    WeightStable,
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

fn retained(word: &str, character_ids: &[u32], unfiltered: bool) -> usize {
    if unfiltered {
        word.chars().count()
    } else {
        word.chars()
            .filter(|&c| character_ids[c as usize] != NONE)
            .count()
    }
}

fn fill<C: Slot>(
    slots: &mut [MaybeUninit<C>],
    base: usize,
    regions: &[Region<'_>],
    character_ids: &[u32],
    identities: usize,
    order: Order,
) -> Vec<Filled> {
    if regions.len() > 1 {
        let middle = regions.len() / 2;
        let cut = regions[..middle].iter().map(|r| r.slots).sum();
        let (left, right) = slots.split_at_mut(cut);
        let (mut a, mut b) = rayon::join(
            || {
                fill(
                    left,
                    base,
                    &regions[..middle],
                    character_ids,
                    identities,
                    order,
                )
            },
            || {
                fill(
                    right,
                    base + cut,
                    &regions[middle..],
                    character_ids,
                    identities,
                    order,
                )
            },
        );
        a.append(&mut b);
        return a;
    }
    let region = &regions[0];
    let mut result = Filled {
        starts: if order == Order::Original {
            Vec::with_capacity(region.words.len())
        } else {
            Vec::new()
        },
        active: vec![0; identities.div_ceil(64)],
    };
    let mut position = 0;
    for &(word, weight) in region.words {
        if order == Order::Original || result.starts.last().is_none_or(|p| p.1 != weight) {
            result.starts.push((base + position, weight));
        }
        for c in word.chars() {
            let id = character_ids[c as usize];
            if id != NONE {
                slots[position].write(C::encode(id));
                position += 1;
                let id = id as usize;
                result.active[id / 64] |= 1_u64 << (id % 64);
            }
        }
        slots[position].write(C::encode(NONE));
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
    order: Order,
) -> Result<Prepared<C, O, INLINE>> {
    let measure = Instant::now();
    let character_ids = alphabet::character_ids(ids);
    let character_table_bytes = character_ids.capacity() * std::mem::size_of::<u32>();
    // Canonical IDs are assigned before construction. Reordering complete words
    // preserves pair counts and the left-to-right AA boundaries within each word.
    let mut words: Vec<_> = wc.iter().map(|(word, &weight)| (word, weight)).collect();
    let word_reference_bytes = words.capacity() * std::mem::size_of::<(&CompactString, u64)>();
    let sorting = Instant::now();
    if order == Order::WeightStable {
        // Preserve the physical order of equal-weight words. Stable sorting
        // borrows a temporary buffer which is released before slot allocation.
        words.par_sort_by(|a, b| b.1.cmp(&a.1));
    } else if order == Order::WeightSorted {
        words.par_sort_unstable_by(|a, b| b.1.cmp(&a.1));
    }
    let sort_ms = sorting.elapsed().as_secs_f64() * 1000.0;
    // Rayon 1.12's stable merge sort allocates len * size_of::<T>() scratch
    // above its insertion-sort threshold. This is allocated capacity, not RSS.
    let sort_buffer_bytes = if order == Order::WeightStable && words.len() > 20 {
        words.len() * std::mem::size_of::<(&CompactString, u64)>()
    } else {
        0
    };
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
                let count = retained(word, &character_ids, unfiltered);
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
    // Reserve final storage without a serial zeroing pass. Disjoint workers
    // initialize every slot directly; there is no per-worker corpus copy.
    let mut slots = Vec::with_capacity(capacity);
    slots.resize_with(capacity, MaybeUninit::uninit);
    slots[0].write(C::encode(NONE));
    let allocate_ms = allocate.elapsed().as_secs_f64() * 1000.0;

    let filling = Instant::now();
    let outputs = if regions.is_empty() {
        Vec::new()
    } else {
        fill(
            &mut slots[1..],
            1,
            &regions,
            &character_ids,
            identities,
            order,
        )
    };
    // SAFETY: slot zero was initialized above; fill covers exactly every other
    // slot through disjoint regions, checks each region's exact length, and its
    // joins complete before this conversion. MaybeUninit<C> has C's layout;
    // the allocation/length/capacity are unchanged and ownership transfers once.
    let mut initialized = ManuallyDrop::new(slots);
    let slots = unsafe {
        Vec::from_raw_parts(
            initialized.as_mut_ptr().cast::<C>(),
            initialized.len(),
            initialized.capacity(),
        )
    };
    let mut lengths = vec![0; identities];
    let mut blocks = vec![Block::new(0, 0)];
    // Region order is stable. Rebuild only small boundary metadata after all
    // disjoint writes have joined; a long word keeps its weight across blocks.
    let starts = outputs.iter().flat_map(|o| o.starts.iter()).copied();
    let temporary_weight_bytes = outputs.iter().map(|o| o.starts.capacity() * 16).sum();
    let mut starts = starts.peekable();
    let mut previous_weight = None;
    while let Some((start, weight)) = starts.next() {
        while blocks.len() <= start >> bits {
            blocks.push(Block::new(blocks.len() << bits, weight));
        }
        if uniform.is_none() && (order == Order::Original || previous_weight != Some(weight)) {
            let block = &mut blocks[start >> bits];
            block.pivots.push((start - block.base) as u32);
            block.weights.push(weight);
        }
        previous_weight = Some(weight);
        let end = starts.peek().map_or(capacity - 1, |p| p.0 - 1);
        while blocks.len() <= end >> bits {
            blocks.push(Block::new(blocks.len() << bits, weight));
        }
    }
    for block in &mut blocks {
        block.weight_intervals = order != Order::Original;
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
        weighted_edges: weighted_edges as u64,
        character_table_bytes,
        word_reference_bytes,
        temporary_weight_bytes,
        timings: Timings {
            measure_ms,
            sort_ms,
            sort_buffer_bytes,
            allocate_ms,
            fill_ms: filling.elapsed().as_secs_f64() * 1000.0,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check<C: Slot>(pool: &rayon::ThreadPool, unfiltered: bool, order: Order) {
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
        let mut ordered: Vec<_> = words.iter().map(|(w, &n)| (w, n)).collect();
        if order == Order::WeightStable {
            ordered.par_sort_by(|a, b| b.1.cmp(&a.1));
        } else if order == Order::WeightSorted {
            ordered.par_sort_unstable_by(|a, b| b.1.cmp(&a.1));
        }
        for (word, weight) in ordered {
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
                build::<C, u16, 4>(
                    &words,
                    &ids,
                    129,
                    unfiltered,
                    4,
                    pool.current_num_threads(),
                    order,
                )
            })
            .unwrap();
        assert_eq!(
            got.slots.iter().map(Slot::token).collect::<Vec<_>>(),
            expected
        );
        assert_eq!(got.lengths, lengths);
        assert_eq!(got.symbols, symbols);
        assert_eq!(got.edges, edges);
        let expected_weighted: u64 = expected
            .windows(2)
            .zip(&weights)
            .filter(|(pair, _)| pair[0] != NONE && pair[1] != NONE)
            .map(|(_, weight)| *weight)
            .sum();
        assert_eq!(got.weighted_edges, expected_weighted);
        assert_eq!(got.slots.capacity(), got.slots.len());
        for (p, &weight) in weights.iter().enumerate().skip(1) {
            assert_eq!(got.blocks[p >> 4].weight(p, got.uniform), weight);
        }
        let empty = pool
            .install(|| build::<C, u16, 4>(&AHashMap::new(), &ids, 129, unfiltered, 4, 4, order))
            .unwrap();
        assert_eq!(empty.slots.len(), 1);
        assert_eq!(empty.slots[0].token(), NONE);
        assert_eq!(empty.lengths, vec![0; 129]);
    }

    #[test]
    fn equal_weight_words_coalesce_across_regions_and_address_blocks() {
        let words: AHashMap<CompactString, u64> = (0..3000)
            .map(|i| (format!("a{i:04}b").into(), [0, 1, 7][i / 1000]))
            .collect();
        let alphabet: Vec<char> = "0123456789ab".chars().collect();
        let ids: AHashMap<CompactString, u32> = alphabet
            .iter()
            .enumerate()
            .map(|(id, c)| (c.to_string().into(), id as u32))
            .collect();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for bits in [4, 16] {
            let got = pool
                .install(|| {
                    build::<u32, u16, 4>(
                        &words,
                        &ids,
                        alphabet.len(),
                        true,
                        bits,
                        4,
                        Order::WeightSorted,
                    )
                })
                .unwrap();
            assert_eq!(got.blocks.iter().map(|b| b.pivots.len()).sum::<usize>(), 3);
            assert!(got.temporary_weight_bytes < words.len() * 16 / 10);
            let mut start = 1;
            for end in 1..got.slots.len() {
                if got.slots[end].token() != NONE {
                    continue;
                }
                let word: String = got.slots[start..end]
                    .iter()
                    .map(|c| alphabet[c.token() as usize])
                    .collect();
                let weight = words[word.as_str()];
                for p in start..=end {
                    assert_eq!(got.blocks[p >> bits].weight(p, got.uniform), weight);
                }
                start = end + 1;
            }
        }
    }

    #[test]
    fn ordered_regions_preserve_filtered_slots_weights_and_reserved_id_activation() {
        for workers in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            for order in [Order::Original, Order::WeightSorted, Order::WeightStable] {
                for unfiltered in [false, true] {
                    check::<u32>(&pool, unfiltered, order);
                    check::<u16>(&pool, unfiltered, order);
                    check::<AtomicU32>(&pool, unfiltered, order);
                    check::<AtomicU16>(&pool, unfiltered, order);
                }
            }
        }
    }
}
