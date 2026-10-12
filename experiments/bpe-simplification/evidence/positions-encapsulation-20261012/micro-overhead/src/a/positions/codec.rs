//! Internal absolute-u64 / LEB128-gap format and its sequential decoder.
use smallvec::SmallVec;

pub(super) const RESTART: usize = 128;

pub(super) fn gap_width(gap: u64) -> usize {
    (u64::BITS - gap.leading_zeros()).div_ceil(7).max(1) as usize
}

/// Leaf reserves the complete append before changing its logical contents.
#[inline]
pub(super) fn append_gap(bytes: &mut SmallVec<[u8; 16]>, mut gap: u64) {
    debug_assert!(bytes.capacity() - bytes.len() >= gap_width(gap));
    while gap >= 128 {
        bytes.push(gap as u8 | 128);
        gap >>= 7;
    }
    bytes.push(gap as u8);
}

pub(super) fn read_absolute(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes[..8].try_into().expect("internal u64 position"))
}

pub(super) struct Cursor<'a> {
    bytes: &'a [u8],
    position: u64,
    index: usize,
}

impl<'a> Cursor<'a> {
    pub(super) fn new(bytes: &'a [u8], start: usize) -> Self {
        Self {
            bytes,
            position: 0,
            index: start,
        }
    }
}

impl Iterator for Cursor<'_> {
    type Item = u64;
    #[inline]
    fn next(&mut self) -> Option<u64> {
        if self.bytes.is_empty() {
            return None;
        }
        if self.index.is_multiple_of(RESTART) {
            self.position = read_absolute(self.bytes);
            self.bytes = &self.bytes[8..];
        } else {
            let mut delta = 0u64;
            let mut shift = 0;
            loop {
                let byte = self.bytes[0];
                self.bytes = &self.bytes[1..];
                delta |= u64::from(byte & 127) << shift;
                if byte < 128 {
                    break;
                }
                shift += 7;
            }
            self.position += delta;
        }
        self.index += 1;
        Some(self.position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gap_width_matches_appended_bytes() {
        let mut gaps = vec![0, 1, 127, 128, u64::MAX];
        for bit in (7..64).step_by(7) {
            gaps.extend([(1u64 << bit) - 1, 1u64 << bit]);
        }
        for gap in gaps {
            let mut bytes = SmallVec::new();
            bytes.extend_from_slice(&0u64.to_le_bytes());
            bytes.reserve(gap_width(gap));
            append_gap(&mut bytes, gap);
            assert_eq!(bytes.len() - 8, gap_width(gap));
            assert_eq!(Cursor::new(&bytes, 0).collect::<Vec<_>>(), [0, gap]);
        }
    }
}
