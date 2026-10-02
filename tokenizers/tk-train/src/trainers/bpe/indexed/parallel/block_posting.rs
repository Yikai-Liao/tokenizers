//! One owner, contiguous local u32 offsets, and runs sharing an address block.
//! A run is two u32 values, never an address per occurrence. At bits=32 every
//! u64 address is representable, including u64::MAX. bits=16 is the scaled
//! experiment: offsets stay u32 so it isolates the cost of segmentation.
use super::*;

#[derive(Clone, Copy, Default)]
struct Run {
    block: u32,
    end: u32,
}
#[derive(Default)]
pub(super) struct BlockPosting {
    positions: SmallPosting,
    runs: PackedPosting<Run, 1>,
}
impl BlockPosting {
    pub(super) fn with_capacity(count: u32) -> Result<Self> {
        Ok(Self {
            positions: SmallPosting::with_capacity(count)?,
            runs: Default::default(),
        })
    }
    pub(super) fn len(&self) -> usize {
        self.positions.len()
    }
    pub(super) fn allocated_capacity(&self) -> usize {
        self.positions.allocated_capacity()
    }
    pub(super) fn directory_bytes(&self) -> usize {
        self.runs.allocated_capacity() * 8
    }
    pub(super) fn run_count(&self) -> usize {
        self.runs.len()
    }
    pub(super) fn reserve_runs(&mut self, count: u32) -> Result<()> {
        debug_assert_eq!(self.runs.len(), 0);
        self.runs = PackedPosting::with_capacity(count)?;
        Ok(())
    }
    pub(super) fn segments(&self, bits: u8) -> impl Iterator<Item = (usize, &[u32])> {
        let mut start = 0;
        self.runs.as_slice().iter().map(move |run| {
            let end = run.end as usize;
            let values = &self.positions.as_slice()[start..end];
            start = end;
            ((run.block as usize) << bits, values)
        })
    }
    pub(super) fn segments_range(
        &self,
        bits: u8,
        begin: usize,
        end: usize,
    ) -> impl Iterator<Item = (usize, &[u32])> {
        let first = self
            .runs
            .as_slice()
            .partition_point(|r| r.end as usize <= begin);
        let mut start = begin;
        self.runs.as_slice()[first..]
            .iter()
            .scan((), move |_, run| {
                if start >= end {
                    return None;
                }
                let stop = (run.end as usize).min(end);
                let slice = &self.positions.as_slice()[start..stop];
                start = stop;
                Some(((run.block as usize) << bits, slice))
            })
    }
    pub(super) fn from_reversed_in_block(
        count: u32,
        block: u32,
        bits: u8,
        mut next: impl FnMut() -> usize,
    ) -> Result<Self> {
        let mut result = Self::with_capacity(count)?;
        result.positions.append_reversed_reserved(count, || {
            let p = next();
            debug_assert_eq!(p >> bits, block as usize);
            (p & ((1usize << bits) - 1)) as u32
        })?;
        if count != 0 {
            result.runs.push(Run { block, end: count })?;
        }
        Ok(result)
    }
    pub(super) fn from_reversed(
        count: u32,
        bits: u8,
        mut next: impl FnMut() -> usize + Clone,
    ) -> Result<Self> {
        let mut counting = next.clone();
        let mut previous = None;
        let mut runs = 0u32;
        for _ in 0..count {
            let block =
                u32::try_from(counting() >> bits).map_err(|_| "block directory exceeds u32")?;
            if previous != Some(block) {
                runs += 1;
                previous = Some(block);
            }
        }
        let mut result = Self::with_capacity(count)?;
        result.reserve_runs(runs)?;
        result.runs.append_reversed_reserved(runs, Run::default)?;
        let directory = result.runs.as_mut_slice();
        let mut ri = runs as usize;
        let mut ordinal = count;
        previous = None;
        result.positions.append_reversed_reserved(count, || {
            let p = next();
            let block = (p >> bits) as u32;
            if previous != Some(block) {
                ri -= 1;
                directory[ri] = Run {
                    block,
                    end: ordinal,
                };
                previous = Some(block);
            }
            ordinal -= 1;
            (p & ((1usize << bits) - 1)) as u32
        })?;
        Ok(result)
    }
    pub(super) fn iter(&self, bits: u8) -> impl Iterator<Item = usize> {
        self.segments(bits)
            .flat_map(|(base, values)| values.iter().map(move |&p| base | p as usize))
    }
    pub(super) fn append(&mut self, other: Self) -> Result<()> {
        let prefix = u32::try_from(self.len()).map_err(|_| "posting length exceeds u32")?;
        let _ = prefix
            .checked_add(other.len() as u32)
            .ok_or("posting length exceeds u32")?;
        for &p in other.positions.as_slice() {
            self.positions.push(p)?;
        }
        for run in other.runs.as_slice() {
            let next = Run {
                block: run.block,
                end: prefix + run.end,
            };
            match self.runs.as_mut_slice().last_mut() {
                Some(last) if last.block == next.block => last.end = next.end,
                _ => self.runs.push(next)?,
            }
        }
        Ok(())
    }
    pub(super) fn push_address(&mut self, position: usize, bits: u8) -> Result<()> {
        let block = u32::try_from(position >> bits).map_err(|_| "block directory exceeds u32")?;
        let local = (position & ((1usize << bits) - 1)) as u32;
        self.positions.push(local)?;
        let end = self.positions.len() as u32;
        match self.runs.as_mut_slice().last_mut() {
            Some(last) if last.block == block => last.end = end,
            _ => self.runs.push(Run { block, end })?,
        }
        Ok(())
    }
    pub(super) fn append_reversed_reserved_at(
        &mut self,
        count: u32,
        bits: u8,
        mut next: impl FnMut() -> usize,
    ) -> Result<()> {
        let start = self.len();
        let mut reverse = PackedPosting::<Run, 1>::default();
        let mut index = start + count as usize;
        // The source is already sorted backwards. Capture each run's end while
        // writing the final low offsets directly into their reserved allocation.
        let mut error = None;
        self.positions.append_reversed_reserved(count, || {
            let p = next();
            let block = match u32::try_from(p >> bits) {
                Ok(v) => v,
                Err(_) => {
                    error = Some("block directory exceeds u32");
                    0
                }
            };
            if reverse.as_slice().last().is_none_or(|r| r.block != block) {
                if reverse
                    .push(Run {
                        block,
                        end: index as u32,
                    })
                    .is_err()
                {
                    error = Some("could not allocate posting run directory");
                }
            }
            index -= 1;
            (p & ((1usize << bits) - 1)) as u32
        })?;
        if let Some(error) = error {
            return Err(error.into());
        }
        for &run in reverse.as_slice().iter().rev() {
            match self.runs.as_mut_slice().last_mut() {
                Some(last) if last.block == run.block => last.end = run.end,
                _ => self.runs.push(run)?,
            }
        }
        Ok(())
    }
    // Single-block compatibility for existing focused initialization tests.
    #[cfg(test)]
    pub(super) fn push(&mut self, p: u32) -> Result<()> {
        self.push_address(p as usize, 32)
    }
    #[cfg(test)]
    pub(super) fn as_slice(&self) -> &[u32] {
        self.positions.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_u64_address_domain_and_scaled_geometry() {
        for bits in [16, 32] {
            let max = if bits == 32 {
                u64::MAX
            } else {
                (1u64 << 48) - 1
            };
            let values = [
                0,
                1,
                (1u64 << bits) - 1,
                1u64 << bits,
                (1u64 << bits) + 1,
                max - 1,
                max,
            ];
            let mut p = BlockPosting::with_capacity(values.len() as u32).unwrap();
            let mut i = values.len();
            p.append_reversed_reserved_at(values.len() as u32, bits, || {
                i -= 1;
                values[i] as usize
            })
            .unwrap();
            assert_eq!(p.iter(bits).map(|p| p as u64).collect::<Vec<_>>(), values);
            assert_eq!(p.run_count(), 3);
            assert_eq!(std::mem::size_of::<Run>(), 8);
        }
    }
    #[test]
    fn randomized_full_domain_roundtrip_and_wave_append() {
        let mut seed = 17u64;
        let mut values = vec![0, u64::MAX];
        for _ in 0..2048 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            values.push(seed);
        }
        values.sort_unstable();
        values.dedup();
        let make = |v: &[u64]| {
            let mut i = v.len();
            BlockPosting::from_reversed(v.len() as u32, 32, move || {
                i -= 1;
                v[i] as usize
            })
            .unwrap()
        };
        let mut p = make(&values[..500]);
        p.append(make(&values[500..])).unwrap();
        assert_eq!(p.iter(32).map(|p| p as u64).collect::<Vec<_>>(), values);
        for (begin, end) in [(0, values.len()), (3, 7), (498, 502), (700, values.len())] {
            let got = p
                .segments_range(32, begin, end)
                .flat_map(|(base, v)| v.iter().map(move |&p| (base | p as usize) as u64))
                .collect::<Vec<_>>();
            assert_eq!(got, values[begin..end]);
        }
    }
    #[test]
    fn append_fragments_coalesces_only_adjacent_equal_blocks() {
        let mut p = BlockPosting::with_capacity(8).unwrap();
        for values in [
            vec![1, 17, 65535],
            vec![65536, 65541],
            vec![65550, 131072, 262145],
        ] {
            let mut i = values.len();
            p.append_reversed_reserved_at(values.len() as u32, 16, || {
                i -= 1;
                values[i]
            })
            .unwrap();
        }
        assert_eq!(
            p.iter(16).collect::<Vec<_>>(),
            [1, 17, 65535, 65536, 65541, 65550, 131072, 262145]
        );
        assert_eq!(p.run_count(), 4);
    }
}
