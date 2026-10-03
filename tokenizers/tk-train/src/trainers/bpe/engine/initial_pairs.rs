//! Initial pair counting and position construction from a read-only corpus.
//!
//! Twelve-byte records cache the complete pair key and a wave-local coordinate.
//! Stable grouping reads the cached key and retains incoming spatial order.
//! Adding the wave base restores the full-width global coordinate.
//! A wave uses the original bound of 2^28 physical slots. Its boundary does not
//! end a word or discard an edge: scanning reads the next corpus slot.
//! Each key is filtered after its complete frequency is known. Multiwave counts
//! accumulate before filtering; a single wave can filter before encoding.
use super::{
    WORD_SEPARATOR_ID,
    corpus::InitialCorpus,
    execution::Execution,
    pair_index::{PairState, pair_key, shard_for},
};
use crate::progress::TrainingProgress;
use ahash::AHashMap;
use radix::KeyedValue;
use rayon::prelude::*;
use std::{mem::MaybeUninit, ops::Range, sync::atomic::Ordering};
use tk_collections::{AllocationArena, SortedPositions, radix};
use tk_encode::Result;

pub(super) struct InitialPairTable<'a> {
    pub(super) shards: Vec<AHashMap<u64, PairState<'a>>>,
    pub(super) weighted_mass: u128,
    pub(super) maximum_word_weight: u64,
}
struct InitialGroup {
    key: u64,
    begin: u32,
    end: u32,
    frequency: u64,
}
struct RecordJob<'a> {
    range: Range<usize>,
    buffers: Vec<&'a mut [MaybeUninit<KeyedValue>]>,
}
#[inline]
fn global_position(base: usize, record: KeyedValue) -> u64 {
    // The producer bounds the offset by its wave. Both parts fit a resident
    // corpus coordinate; retaining the base keeps positions above u32 intact.
    base as u64 + u64::from(record.value())
}
pub(super) fn build_initial_pairs<'a>(
    corpus: InitialCorpus<'_>,
    minimum_frequency: u64,
    execution: &Execution,
    arena: &'a AllocationArena,
    progress: &TrainingProgress,
) -> Result<InitialPairTable<'a>> {
    // PERF: Preserve complete-key filtering and the original large wave bound.
    // Smaller waves would encode partial counts and retire more arena buffers.
    build_in_waves(
        corpus,
        minimum_frequency,
        execution,
        arena,
        progress,
        1 << 28,
    )
}

pub(super) fn build_in_waves<'a>(
    corpus: InitialCorpus<'_>,
    minimum_frequency: u64,
    execution: &Execution,
    arena: &'a AllocationArena,
    progress: &TrainingProgress,
    records_per_wave: usize,
) -> Result<InitialPairTable<'a>> {
    assert!(records_per_wave > 1 && records_per_wave <= radix::MAX_RECORDS);
    let workers = execution.workers();
    let mut shards: Vec<_> = (0..workers)
        .map(|_| AHashMap::<u64, PairState<'a>>::new())
        .collect();
    let mut weighted_mass = 0_u128;
    let maximum_word_weight = corpus
        .word_weights
        .values()
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let edge = |position: usize| {
        let left = corpus.token_ids[position].load(Ordering::Relaxed);
        let right = corpus.token_ids[position + 1].load(Ordering::Relaxed);
        (left != WORD_SEPARATOR_ID && right != WORD_SEPARATOR_ID).then(|| pair_key((left, right)))
    };
    let uniform_weight =
        (corpus.word_weights.values().len() == 1).then(|| corpus.word_weights.values()[0]);
    let single_wave = corpus.token_ids.len() <= records_per_wave;
    let wave_floor = if single_wave {
        minimum_frequency
    } else {
        minimum_frequency.min(1)
    };
    for base in (0..corpus.token_ids.len().saturating_sub(1)).step_by(records_per_wave) {
        let end = (base + records_per_wave).min(corpus.token_ids.len() - 1);
        let mut wave_tables: Vec<_> = (0..workers)
            .map(|_| AHashMap::<u64, PairState<'a>>::new())
            .collect();
        let chunk = (end - base).div_ceil(workers * 8).max(1);
        let ranges: Vec<_> = (base..end)
            .step_by(chunk)
            .map(|start| start..(start + chunk).min(end))
            .collect();
        let route_work = progress.stage("Route initial pairs", (end - base) * 2);
        let sizes: Vec<Vec<usize>> = ranges
            .par_iter()
            .map(|range| {
                let mut sizes = vec![0; workers];
                for position in range.clone() {
                    if let Some(key) = edge(position) {
                        sizes[shard_for(key, workers)] += 1;
                    }
                }
                route_work.complete(range.len());
                sizes
            })
            .collect();
        let shard_sizes: Vec<usize> = (0..workers)
            .map(|shard| sizes.iter().map(|job| job[shard]).sum())
            .collect();
        let records: usize = shard_sizes.iter().sum();
        // PERF: Each owner owns its record allocation. Installation drops it
        // before later owners encode their positions, so complete raw records
        // do not remain live alongside every final compressed list.
        let mut record_buffers: Vec<_> = shard_sizes
            .iter()
            .map(|&count| {
                let mut records = Vec::<MaybeUninit<KeyedValue>>::with_capacity(count);
                // SAFETY: MaybeUninit admits unwritten elements. The counted job
                // slices cover this owner, and every producer fills its whole slice.
                unsafe {
                    records.set_len(count);
                }
                records
            })
            .collect();
        let mut jobs: Vec<_> = ranges
            .into_iter()
            .map(|range| RecordJob {
                range,
                buffers: Vec::with_capacity(workers),
            })
            .collect();
        for (shard, buffer) in record_buffers.iter_mut().enumerate() {
            let mut remaining = buffer.as_mut_slice();
            for (job, counts) in jobs.iter_mut().zip(&sizes) {
                let (part, next) = remaining.split_at_mut(counts[shard]);
                remaining = next;
                job.buffers.push(part);
            }
        }
        jobs.into_par_iter().for_each(|mut job| {
            let mut used = vec![0; workers];
            for position in job.range.clone() {
                if let Some(key) = edge(position) {
                    let shard = shard_for(key, workers);
                    job.buffers[shard][used[shard]]
                        .write(KeyedValue::new(key, (position - base) as u32));
                    used[shard] += 1;
                }
            }
            route_work.complete(job.range.len());
        });
        // Each emitted offset is below records_per_wave <= 2^28. The u32
        // payload is local to this wave; the global coordinate is never narrowed.
        // SAFETY: All producer jobs joined after initializing every counted
        // element. MaybeUninit<KeyedValue> and KeyedValue have the same layout
        // and allocation size.
        let mut record_buffers: Vec<Vec<KeyedValue>> = record_buffers
            .into_iter()
            .map(|records| {
                let mut records = std::mem::ManuallyDrop::new(records);
                unsafe {
                    Vec::from_raw_parts(
                        records.as_mut_ptr().cast::<KeyedValue>(),
                        records.len(),
                        records.capacity(),
                    )
                }
            })
            .collect();
        let sort_work = progress.stage("Sort initial pairs", records);
        record_buffers.par_iter_mut().for_each(|records| {
            radix::sort_by_key(records);
            sort_work.complete(records.len());
        });
        let group_work = progress.stage("Build initial positions", records * 2);
        // PERF: Sorting uses every worker. At most two owners install lists at
        // once: compressed output overlaps raw records that have not yet freed.
        let install_width = workers.min(2);
        for (tables, buffers) in wave_tables
            .chunks_mut(install_width)
            .zip(record_buffers.chunks_mut(install_width))
        {
            let grouped: Vec<_> = buffers
                .par_iter()
                .map(|records| -> Result<_> {
                    let mut mass = 0_u128;
                    let mut begin = 0;
                    let mut groups = Vec::new();
                    let mut completed = 0;
                    while begin < records.len() {
                        let key = records[begin].key();
                        let mut end = begin + 1;
                        while end < records.len() && records[end].key() == key {
                            end += 1;
                        }
                        // PERF: The original uniform-weight path counts the run
                        // without coordinate or interval queries.
                        let frequency = if let Some(weight) = uniform_weight {
                            weight
                                .checked_mul((end - begin) as u64)
                                .ok_or("BPE initial pair frequency exceeds u64")?
                        } else {
                            let mut frequency = 0_u64;
                            for (count, weight) in corpus
                                .word_weights
                                .runs_for_sorted(&records[begin..end], |&record| {
                                    global_position(base, record)
                                })
                            {
                                let weight =
                                    *weight.expect("every initial edge belongs to a word interval");
                                frequency = weight
                                    .checked_mul(count as u64)
                                    .and_then(|part| frequency.checked_add(part))
                                    .ok_or("BPE initial pair frequency exceeds u64")?;
                            }
                            frequency
                        };
                        // Resident edges times u64 weights fit u128. Per-key counts
                        // remain checked u64 values, without a global u64 mass cap.
                        mass += u128::from(frequency);
                        if frequency >= wave_floor {
                            groups.push(InitialGroup {
                                key,
                                begin: begin as u32,
                                end: end as u32,
                                frequency,
                            });
                        }
                        completed += end - begin;
                        if completed >= 1 << 16 {
                            group_work.complete(completed);
                            completed = 0;
                        }
                        begin = end;
                    }
                    group_work.complete(completed);
                    Ok((groups, mass))
                })
                .collect::<Result<_>>()?;
            let masses = tables
                .par_iter_mut()
                .zip(buffers.par_iter_mut())
                .zip(grouped.into_par_iter())
                .map(|((table, buffer), (groups, mass))| -> Result<u128> {
                    let records = std::mem::take(buffer);
                    let worker = execution.current_worker();
                    let lease = arena.lease(worker);
                    let mut scratch = execution.encoding(worker);
                    table.reserve(groups.len());
                    for InitialGroup {
                        key,
                        begin,
                        end,
                        frequency,
                    } in groups
                    {
                        let positions = records[begin as usize..end as usize]
                            .iter()
                            .map(|&record| global_position(base, record));
                        table.insert(
                            key,
                            PairState {
                                ledger_count_bits: frequency,
                                // A complete zero count has no candidate
                                // payload. A partial zero count may share
                                // its key with a positive wave elsewhere.
                                positions: if single_wave && frequency == 0 {
                                    SortedPositions::new()
                                } else {
                                    SortedPositions::from_sorted_iter(
                                        positions,
                                        &mut scratch,
                                        &lease,
                                    )?
                                },
                            },
                        );
                    }
                    // Group scanning and list installation each account
                    // for one pass. Publication remains visible work after
                    // the complete frequencies have been counted.
                    group_work.complete(records.len());
                    // This owner's raw stream frees before the next owner wave.
                    Ok(mass)
                })
                .collect::<Result<Vec<_>>>()?;
            weighted_mass += masses.into_iter().sum::<u128>();
        }
        // Each wave owns exact lists before publication. Repeated keys append
        // those compressed lists with the original measure/replay lifecycle.
        // Moving an empty owner's table avoids duplicating its map allocation.
        let publish_work = progress.stage(
            "Publish initial pairs",
            wave_tables.iter().map(|table| table.len()).sum(),
        );
        shards
            .par_iter_mut()
            .zip(wave_tables.into_par_iter())
            .map(|(table, wave)| -> Result<()> {
                let keys = wave.len();
                if table.is_empty() {
                    *table = wave;
                } else {
                    let worker = execution.current_worker();
                    let lease = arena.lease(worker);
                    let mut scratch = execution.encoding(worker);
                    for (key, state) in wave {
                        use std::collections::hash_map::Entry;
                        match table.entry(key) {
                            Entry::Vacant(entry) => {
                                entry.insert(state);
                            }
                            Entry::Occupied(mut entry) => {
                                let old = entry.get_mut();
                                old.ledger_count_bits = old
                                    .ledger_count_bits
                                    .checked_add(state.ledger_count_bits)
                                    .ok_or("BPE initial pair frequency exceeds u64")?;
                                old.positions
                                    .append(state.positions, &mut scratch, &lease)?;
                            }
                        }
                    }
                }
                publish_work.complete(keys);
                Ok(())
            })
            .collect::<Result<Vec<_>>>()?;
    }
    if minimum_frequency != 0 {
        for shard in &mut shards {
            shard.retain(|_, state| state.ledger_count_bits >= minimum_frequency);
        }
    }
    Ok(InitialPairTable {
        shards,
        weighted_mass,
        maximum_word_weight,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wave_offsets_restore_positions_above_u32() {
        let base = (1_usize << 32) + 17;
        let mut records = [
            KeyedValue::new(u64::MAX, 0),
            KeyedValue::new(u64::MAX, (1 << 28) - 1),
        ];
        radix::sort_by_key(&mut records);
        let arena = AllocationArena::new(1, 2);
        let lease = arena.lease(0);
        let mut scratch = tk_collections::PositionEncodingScratch::default();
        let positions = SortedPositions::from_sorted_iter(
            records
                .into_iter()
                .map(|record| global_position(base, record)),
            &mut scratch,
            &lease,
        )
        .unwrap();
        assert_eq!(
            positions.iter().collect::<Vec<_>>(),
            [(1_u64 << 32) + 17, (1_u64 << 32) + (1 << 28) + 16]
        );
    }
}
