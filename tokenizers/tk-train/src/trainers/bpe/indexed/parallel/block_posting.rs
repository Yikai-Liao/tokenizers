//! Local u32 offsets and a compact directory of (block ID, count) runs.
//! Both directory integers use 15 data bits per u16, with one continuation bit.
//! This keeps arena alignment, supports every u32 block ID, and avoids per-run
//! objects. All corpus geometries use this exact representation and builder.
use super::*;
use smallvec::SmallVec;

#[derive(Clone, Copy, Default)]
struct Run {
    block: u32,
    count: u32,
}
#[derive(Default)]
pub(super) struct BlockPosting {
    positions: SmallPosting,
    runs: PackedPosting<u16, 4>,
}
fn units(mut value: u32) -> usize {
    let mut n = 1;
    while value >= 0x8000 {
        value >>= 15;
        n += 1;
    }
    n
}
fn emit(out: &mut PackedPosting<u16, 4>, mut value: u32) -> Result<()> {
    while value >= 0x8000 {
        out.push((value as u16 & 0x7fff) | 0x8000)?;
        value >>= 15;
    }
    out.push(value as u16)
}
fn read(input: &[u16], cursor: &mut usize) -> u32 {
    let first = input[*cursor];
    *cursor += 1;
    if first < 0x8000 {
        return first as u32;
    }
    let second = input[*cursor];
    *cursor += 1;
    let mut value = (first as u32 & 0x7fff) | ((second as u32 & 0x7fff) << 15);
    if second >= 0x8000 {
        value |= (input[*cursor] as u32) << 30;
        *cursor += 1;
    }
    value
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
        self.runs.allocated_capacity() * 2
    }
    pub(super) fn run_count(&self) -> usize {
        let mut cursor = 0;
        let mut count = 0;
        while cursor < self.runs.len() {
            read(self.runs.as_slice(), &mut cursor);
            read(self.runs.as_slice(), &mut cursor);
            count += 1;
        }
        count
    }
    pub(super) fn segments(&self, bits: u8) -> impl Iterator<Item = (usize, &[u32])> {
        let mut cursor = 0;
        let mut start = 0;
        std::iter::from_fn(move || {
            if start == self.len() {
                return None;
            }
            let block = read(self.runs.as_slice(), &mut cursor);
            let count = read(self.runs.as_slice(), &mut cursor) as usize;
            let end = start + count;
            let positions = &self.positions.as_slice()[start..end];
            start = end;
            Some(((block as usize) << bits, positions))
        })
    }
    pub(super) fn segments_range(
        &self,
        bits: u8,
        begin: usize,
        end: usize,
    ) -> impl Iterator<Item = (usize, &[u32])> {
        let mut segments = self.segments(bits);
        let mut cursor = 0;
        std::iter::from_fn(move || {
            loop {
                if cursor >= end {
                    return None;
                }
                let (base, values) = segments.next()?;
                let start = cursor;
                cursor += values.len();
                if cursor <= begin {
                    continue;
                }
                return Some((
                    base,
                    &values[begin.saturating_sub(start)..(end - start).min(values.len())],
                ));
            }
        })
    }
    pub(super) fn iter(&self, bits: u8) -> impl Iterator<Item = usize> {
        self.segments(bits)
            .flat_map(|(base, values)| values.iter().map(move |&p| base | p as usize))
    }
    pub(super) fn from_reversed(count: u32, bits: u8, next: impl FnMut() -> usize) -> Result<Self> {
        let mut result = Self::with_capacity(count)?;
        result.append_reversed_reserved_at(count, bits, next)?;
        Ok(result)
    }
    pub(super) fn append_reversed_reserved_at(
        &mut self,
        count: u32,
        bits: u8,
        mut next: impl FnMut() -> usize,
    ) -> Result<()> {
        // Collect only per-run metadata while writing final offsets exactly once.
        // This temporary is per key, never per corpus or per owner wave.
        let mut reverse = SmallVec::<[Run; 4]>::new();
        let mut overflow = false;
        self.positions.append_reversed_reserved(count, || {
            let p = next();
            let high = p >> bits;
            if high > u32::MAX as usize {
                overflow = true;
            }
            let block = high as u32;
            match reverse.last_mut() {
                Some(run) if run.block == block => run.count += 1,
                _ => reverse.push(Run { block, count: 1 }),
            }
            (p & ((1usize << bits) - 1)) as u32
        })?;
        if overflow {
            return Err("block directory exceeds u32".into());
        }
        if self.runs.len() == 0 {
            let size = reverse
                .iter()
                .map(|r| units(r.block) + units(r.count))
                .sum::<usize>();
            self.runs = PackedPosting::with_capacity(
                u32::try_from(size).map_err(|_| "posting directory exceeds u32")?,
            )?;
        }
        for run in reverse.iter().rev() {
            emit(&mut self.runs, run.block)?;
            emit(&mut self.runs, run.count)?;
        }
        Ok(())
    }
    pub(super) fn append(&mut self, other: Self) -> Result<()> {
        let _ = (self.len() as u32)
            .checked_add(other.len() as u32)
            .ok_or("posting length exceeds u32")?;
        for &p in other.positions.as_slice() {
            self.positions.push(p)?;
        }
        // Absolute block IDs permit concatenation without scanning older runs.
        for &unit in other.runs.as_slice() {
            self.runs.push(unit)?;
        }
        Ok(())
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
    fn directory_words_cover_full_u32_values_and_keep_arena_alignment() {
        let values = [0, 1, 32767, 32768, (1 << 30) - 1, 1 << 30, u32::MAX];
        let mut encoded = PackedPosting::<u16, 4>::default();
        for v in values {
            emit(&mut encoded, v).unwrap();
        }
        let mut cursor = 0;
        for v in values {
            assert_eq!(read(encoded.as_slice(), &mut cursor), v);
        }
        assert_eq!(cursor, encoded.len());
        assert_eq!(std::mem::align_of::<u16>(), 2);
    }
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
    fn append_fragments_preserves_adjacent_runs_without_rewriting_prefixes() {
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
        assert_eq!(p.run_count(), 5);
    }
}
