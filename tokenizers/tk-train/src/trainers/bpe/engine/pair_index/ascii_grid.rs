//! A small ASCII pair tier, following tk-encode's top_index/top_values split.
//! Cells hold stable indices into owned states; removal takes a state out of its
//! cell. Other keys keep the existing sparse ownership and checked count path.
use super::{PairState, pair_key};
use ahash::AHashMap;
use std::sync::Arc;

const EMPTY: u16 = u16::MAX;
const SIDE: usize = 128;

pub(super) struct AsciiIds {
    ids: [u32; SIDE],
    ordinals: Box<[u8]>,
}
impl AsciiIds {
    pub(super) fn new(ids: &[u32; SIDE]) -> Option<Self> {
        let maximum = ids.iter().copied().filter(|&id| id != u32::MAX).max()?;
        // Reserved IDs can be far apart. This is a storage admission bound, not
        // an ID restriction: a large range retains the original sparse table.
        if maximum >= 1 << 20 {
            return None;
        }
        let mut ordinals = vec![u8::MAX; maximum as usize + 1];
        for (ordinal, &id) in ids.iter().enumerate() {
            if id != u32::MAX {
                ordinals[id as usize] = ordinal as u8;
            }
        }
        Some(Self {
            ids: *ids,
            ordinals: ordinals.into_boxed_slice(),
        })
    }
    #[inline]
    fn cell(&self, key: u64) -> Option<usize> {
        let left = *self.ordinals.get((key >> 32) as usize)?;
        let right = *self.ordinals.get(key as u32 as usize)?;
        (left != u8::MAX && right != u8::MAX).then_some(left as usize * SIDE + right as usize)
    }
}

pub(super) struct PairStates<'arena> {
    ascii: Option<Arc<AsciiIds>>,
    grid: Box<[u16]>,
    dense: Vec<Option<PairState<'arena>>>,
    sparse: AHashMap<u64, PairState<'arena>>,
}
impl<'arena> PairStates<'arena> {
    pub(super) fn new(
        mut sparse: AHashMap<u64, PairState<'arena>>,
        mut ascii: Option<Arc<AsciiIds>>,
    ) -> Self {
        let mut grid = Vec::new();
        let mut dense = Vec::new();
        if let Some(ascii) = &ascii {
            grid.resize(SIDE * SIDE, EMPTY);
            for (left, &a) in ascii
                .ids
                .iter()
                .enumerate()
                .filter(|&(_, &id)| id != u32::MAX)
            {
                for (right, &b) in ascii
                    .ids
                    .iter()
                    .enumerate()
                    .filter(|&(_, &id)| id != u32::MAX)
                {
                    if let Some(state) = sparse.remove(&pair_key((a, b))) {
                        grid[left * SIDE + right] = dense.len() as u16;
                        dense.push(Some(state));
                    }
                }
            }
        }
        if dense.is_empty() {
            ascii = None;
            grid.clear();
        }
        Self {
            ascii,
            grid: grid.into_boxed_slice(),
            dense,
            sparse,
        }
    }
    #[inline]
    fn cell(&self, key: u64) -> Option<usize> {
        self.ascii.as_ref()?.cell(key)
    }
    #[inline]
    pub(super) fn get(&self, key: &u64) -> Option<&PairState<'arena>> {
        if let Some(cell) = self.cell(*key) {
            let index = self.grid[cell];
            if index == EMPTY {
                None
            } else {
                self.dense[index as usize].as_ref()
            }
        } else {
            self.sparse.get(key)
        }
    }
    #[inline]
    pub(super) fn get_mut(&mut self, key: &u64) -> Option<&mut PairState<'arena>> {
        if let Some(cell) = self.cell(*key) {
            let index = self.grid[cell];
            if index == EMPTY {
                None
            } else {
                self.dense[index as usize].as_mut()
            }
        } else {
            self.sparse.get_mut(key)
        }
    }
    pub(super) fn remove(&mut self, key: &u64) -> Option<PairState<'arena>> {
        if let Some(cell) = self.cell(*key) {
            let index = self.grid[cell];
            if index == EMPTY {
                None
            } else {
                self.dense[index as usize].take()
            }
        } else {
            self.sparse.remove(key)
        }
    }
    pub(super) fn contains_key(&self, key: &u64) -> bool {
        self.get(key).is_some()
    }
    pub(super) fn insert(&mut self, key: u64, state: PairState<'arena>) {
        if let Some(cell) = self.cell(key) {
            let index = self.grid[cell];
            if index == EMPTY {
                // One permanent index per ASCII cell: there are only 128² of
                // them, even if a previously removed key is inserted again.
                self.grid[cell] = self.dense.len() as u16;
                self.dense.push(Some(state));
            } else {
                self.dense[index as usize] = Some(state);
            }
        } else {
            self.sparse.insert(key, state);
        }
    }
}
