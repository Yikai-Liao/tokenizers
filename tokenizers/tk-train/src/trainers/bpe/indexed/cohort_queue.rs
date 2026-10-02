//! Compact priorities keep the historical posting owner in each candidate.
//! Before any identity reuse, each pair has one queue cohort. Promotion at that
//! boundary therefore cannot reorder equal-priority historical cohorts.
use super::*;
struct Packed {
    priority: u64,
    positions: SmallPosting,
}
impl Packed {
    fn encode(c: Candidate) -> Self {
        debug_assert!(c.count <= u32::MAX as u64 && c.pair.0 < 65536 && c.pair.1 < 65536);
        let code = (c.pair.0 << 16) | c.pair.1;
        Self {
            priority: (c.count << 32) | u64::from(!code),
            positions: c.positions,
        }
    }
    fn decode(self) -> Candidate {
        let code = !(self.priority as u32);
        Candidate {
            pair: (code >> 16, code & 65535),
            count: self.priority >> 32,
            positions: self.positions,
        }
    }
}
impl Eq for Packed {}
impl PartialEq for Packed {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}
impl Ord for Packed {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.cmp(&other.priority)
    }
}
impl PartialOrd for Packed {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
pub(super) struct Queue {
    wide: OctonaryHeap<Candidate>,
    packed: Option<OctonaryHeap<Packed>>,
}
impl From<OctonaryHeap<Candidate>> for Queue {
    fn from(wide: OctonaryHeap<Candidate>) -> Self {
        Self { wide, packed: None }
    }
}
impl Queue {
    pub(super) fn pack(&mut self) {
        self.packed = Some(
            std::mem::take(&mut self.wide)
                .into_iter()
                .map(Packed::encode)
                .collect(),
        );
    }
    pub(super) fn promote(&mut self) {
        if let Some(heap) = self.packed.take() {
            self.wide = heap.into_iter().map(Packed::decode).collect();
        }
    }
    pub(super) fn pop(&mut self) -> Option<Candidate> {
        self.packed
            .as_mut()
            .map_or_else(|| self.wide.pop(), |h| h.pop().map(Packed::decode))
    }
    pub(super) fn push(&mut self, c: Candidate) {
        if let Some(heap) = &mut self.packed {
            heap.push(Packed::encode(c));
        } else {
            self.wide.push(c);
        }
    }
    pub(super) fn bytes(&self) -> usize {
        self.wide.capacity() * std::mem::size_of::<Candidate>()
            + self
                .packed
                .as_ref()
                .map_or(0, |h| h.capacity() * std::mem::size_of::<Packed>())
    }
    pub(super) fn packed(&self) -> bool {
        self.packed.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compact_priorities_and_promotion_preserve_order_and_posting_owners() {
        let make = || {
            (0..8192_u32).map(|i| Candidate {
                pair: (i, 65535 - i),
                count: [0, 1, u32::MAX as u64, (i % 97) as u64][i as usize % 4],
                positions: SmallPosting::default(),
            })
        };
        let mut wide = Queue::from(make().collect::<OctonaryHeap<_>>());
        let mut packed = Queue::from(make().collect::<OctonaryHeap<_>>());
        packed.pack();
        assert_eq!(std::mem::size_of::<Packed>(), 24);
        assert_eq!(std::mem::size_of::<Candidate>(), 32);
        for step in 0..8192 {
            if step == 1024 {
                packed.promote();
            }
            let a = wide.pop().unwrap();
            let b = packed.pop().unwrap();
            assert_eq!((a.pair, a.count), (b.pair, b.count));
        }
        assert!(packed.pop().is_none());
    }
}
