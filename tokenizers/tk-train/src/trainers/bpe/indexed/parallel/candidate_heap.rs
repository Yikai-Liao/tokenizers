//! Range-checked compact candidates; wide candidates preserve the general path.
use super::{Candidate, OctonaryHeap, key};

#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub(super) struct PackedCandidate(u64);
impl PackedCandidate {
    fn encode(candidate: Candidate) -> Option<Self> {
        let frequency = u32::try_from(candidate.frequency).ok()?;
        let a = u16::try_from(candidate.key >> 32).ok()?;
        let b = u16::try_from(candidate.key as u32).ok()?;
        let code = (u32::from(a) << 16) | u32::from(b);
        Some(Self((u64::from(frequency) << 32) | u64::from(!code)))
    }
    fn decode(self) -> Candidate {
        let code = !(self.0 as u32);
        Candidate {
            frequency: self.0 >> 32,
            key: key(code >> 16, code & 0xffff),
        }
    }
}

pub(super) enum CandidateHeap {
    Packed(OctonaryHeap<PackedCandidate>),
    Wide(OctonaryHeap<Candidate>),
}
impl Default for CandidateHeap {
    fn default() -> Self {
        Self::Wide(OctonaryHeap::new())
    }
}
impl CandidateHeap {
    pub(super) fn new(candidates: impl Iterator<Item = Candidate>, packed: bool) -> Self {
        if packed {
            // Initial edge mass and the complete ID domain were checked before
            // constructing this heap. Build the narrow heap directly: no wide copy.
            Self::Packed(
                candidates
                    .map(|c| PackedCandidate::encode(c).expect("checked initial candidate range"))
                    .collect(),
            )
        } else {
            Self::Wide(candidates.collect())
        }
    }
    pub(super) fn peek(&self) -> Option<Candidate> {
        match self {
            Self::Packed(h) => h.peek().copied().map(PackedCandidate::decode),
            Self::Wide(h) => h.peek().copied(),
        }
    }
    pub(super) fn pop(&mut self) -> Option<Candidate> {
        match self {
            Self::Packed(h) => h.pop().map(PackedCandidate::decode),
            Self::Wide(h) => h.pop(),
        }
    }
    pub(super) fn push(&mut self, candidate: Candidate) {
        match self {
            Self::Packed(h) => {
                if let Some(packed) = PackedCandidate::encode(candidate) {
                    h.push(packed);
                } else {
                    // Defensive promotion also preserves semantics if a future
                    // caller adds IDs or frequencies beyond the proven domain.
                    let old = std::mem::take(h);
                    *self = Self::Wide(old.into_iter().map(PackedCandidate::decode).collect());
                    self.push(candidate);
                }
            }
            Self::Wide(h) => h.push(candidate),
        }
    }
    pub(super) fn capacity_bytes(&self) -> usize {
        match self {
            Self::Packed(h) => h.capacity() * 8,
            Self::Wide(h) => h.capacity() * 16,
        }
    }
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Packed(h) => h.len(),
            Self::Wide(h) => h.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packing_preserves_full_priority_and_arbitrary_id_bits() {
        let mut seed = 391_u64;
        let mut candidates = Vec::new();
        for i in 0..8192 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let a = [0, 1, 32768, 65535, seed as u16 as u32][i % 5];
            let b = [65535, 32768, 1, 0, (seed >> 16) as u16 as u32][i % 5];
            let frequency = [0, 1, u32::MAX as u64, (seed >> 32) % 97][i % 4];
            candidates.push(Candidate {
                key: key(a, b),
                frequency,
            });
        }
        let mut wide = CandidateHeap::new(candidates.iter().copied(), false);
        let mut packed = CandidateHeap::new(candidates.iter().copied(), true);
        assert_eq!(packed.capacity_bytes() * 2, wide.capacity_bytes());
        while let Some(expected) = wide.pop() {
            let actual = packed.pop().unwrap();
            assert_eq!(
                (actual.key, actual.frequency),
                (expected.key, expected.frequency)
            );
        }
        assert!(packed.pop().is_none());
    }
    #[test]
    fn out_of_range_insert_promotes_without_losing_old_candidates() {
        for extra in [
            Candidate {
                key: key(1, 1),
                frequency: u32::MAX as u64 + 1,
            },
            Candidate {
                key: key(65536, 1),
                frequency: 9,
            },
            Candidate {
                key: key(1, 65536),
                frequency: 9,
            },
        ] {
            let mut candidates = vec![
                Candidate {
                    key: key(1, 2),
                    frequency: 8,
                },
                Candidate {
                    key: key(2, 1),
                    frequency: 8,
                },
            ];
            let mut packed = CandidateHeap::new(candidates.iter().copied(), true);
            packed.push(extra);
            candidates.push(extra);
            assert!(matches!(packed, CandidateHeap::Wide(_)));
            let mut wide = CandidateHeap::new(candidates.into_iter(), false);
            while let Some(expected) = wide.pop() {
                let actual = packed.pop().unwrap();
                assert_eq!(
                    (actual.key, actual.frequency),
                    (expected.key, expected.frequency)
                );
            }
        }
    }
}
