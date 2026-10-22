//! Task-local checked neighbor changes and reusable ID directories.
use super::{PairChange, Positions};
use std::sync::Mutex;
use tk_encode::Result;
use tk_encode::models::bpe::Pair;
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum EdgeSide {
    Left,
    Right,
}
pub(super) struct NeighborChanges<'scratch> {
    pair: Pair,
    merged_id: u32,
    scratch: &'scratch mut NeighborScratch,
    changes: [Vec<PairChange<Positions>>; 2],
}
#[derive(Default)]
pub(super) struct NeighborScratch {
    indices: [Vec<u32>; 2],
}
impl<'scratch> NeighborChanges<'scratch> {
    pub(super) fn new(
        pair: Pair,
        merged_id: u32,
        domain: usize,
        scratch: &'scratch mut NeighborScratch,
    ) -> Self {
        for values in &mut scratch.indices {
            values.resize(domain, u32::MAX);
        }
        Self {
            pair,
            merged_id,
            scratch,
            changes: std::array::from_fn(|_| Vec::new()),
        }
    }

    fn group(&mut self, neighbor: u32, side: EdgeSide) -> &mut PairChange<Positions> {
        let left = side == EdgeSide::Left;
        let changes = &mut self.changes[side as usize];
        let slot = &mut self.scratch.indices[side as usize][neighbor as usize];
        // Scratch indices survive earlier tasks. Reuse a slot only when it
        // names a current group whose neighbor occupies this side's pair field.
        if changes
            .get(*slot as usize)
            .is_none_or(|change| (if left { change.born.0 } else { change.born.1 }) != neighbor)
        {
            let id = self.merged_id;
            let pair = self.pair;
            let index = changes.len();
            changes.push(PairChange {
                removed: if left {
                    (neighbor, pair.0)
                } else {
                    (pair.1, neighbor)
                },
                born: if left { (neighbor, id) } else { (id, neighbor) },
                removed_weight: 0,
                born_weight: 0,
                positions: Positions::default(),
                birth_group: usize::from(!left && neighbor != id),
            });
            *slot =
                u32::try_from(index).expect("each neighbor directory is bounded by real token IDs");
        }
        &mut changes[*slot as usize]
    }

    pub(super) fn record(
        &mut self,
        side: EdgeSide,
        removed: u32,
        born: u32,
        position: usize,
        weight: u64,
        admit_birth: bool,
    ) -> Result<()> {
        let group = self.group(removed, side);
        super::super::add(&mut group.removed_weight, weight)?;
        if admit_birth {
            let group = if removed == born {
                group
            } else {
                self.group(born, side)
            };
            super::super::add(&mut group.born_weight, weight)?;
            group.positions.push(position as u64)?;
        }
        Ok(())
    }

    pub(super) fn finish(self) -> Vec<PairChange<Positions>> {
        // Removal-only events need a nonzero weight; zero-weight births still
        // carry historical positions and must reach the index.
        self.changes
            .into_iter()
            .flatten()
            .filter(|change| change.removed_weight != 0 || !change.positions.is_empty())
            .collect()
    }
}

#[derive(Default)]
pub(super) struct NeighborScratchPool(Mutex<Vec<NeighborScratch>>);

impl NeighborScratchPool {
    pub(super) fn with<T>(&self, scan: impl FnOnce(&mut NeighborScratch) -> T) -> T {
        let mut scratch = self
            .0
            .lock()
            .expect("neighbor scratch lock only guards pool bookkeeping")
            .pop()
            .unwrap_or_default();
        let result = scan(&mut scratch);
        self.0
            .lock()
            .expect("neighbor scratch lock only guards pool bookkeeping")
            .push(scratch);
        result
    }
}
