use super::{Result, StorageError, position_storage::PositionStorage};
use std::num::NonZeroU32;

#[derive(Clone, Copy)]
struct PositionNode(NonZeroU32);
/// One chain within a bounded position buffer. Clear its owner only after all
/// chains have been consumed. Chain links are local indices, not coordinates.
#[derive(Clone, Copy, Default)]
pub(in super::super) struct PositionChain {
    head: Option<PositionNode>,
    tail: Option<PositionNode>,
    length: u32,
    checkpoint: Option<PositionNode>,
}
impl PositionChain {
    /// Return the number of linked positions.
    #[inline]
    pub(in super::super) fn len(&self) -> usize {
        self.length as usize
    }
    /// Return whether there are no linked positions.
    #[inline]
    pub(in super::super) fn is_empty(&self) -> bool {
        self.length == 0
    }
}
/// Bounded reverse chains share compact full-width positions and local links.
#[derive(Default)]
pub(in super::super) struct PositionChains {
    positions: PositionStorage<Option<PositionNode>>,
    checkpoints: ahash::AHashMap<u32, Option<PositionNode>>,
}
impl PositionChains {
    pub(in super::super) const MAX_NODES: usize = 1 << 27;
    /// Construct an empty bounded buffer.
    pub(in super::super) fn new() -> Self {
        Self::default()
    }
    /// Append a coordinate to a chain. Node links remain within this buffer.
    ///
    /// # Errors
    /// Returns an error when the bounded buffer is full. The caller can consume
    /// the completed chunk and continue the same logical chain in a new chunk.
    // PERF: Merge preparation appends at most two nodes per rewritten token.
    // Inlining combines the node budget and coordinate-plane work.
    #[inline]
    pub(in super::super) fn push(
        &mut self,
        chain: &mut PositionChain,
        position: u64,
    ) -> Result<()> {
        if self.positions.len() == Self::MAX_NODES {
            return Err(StorageError("position chain chunk is full"));
        }
        let node = PositionNode(
            NonZeroU32::new(self.positions.len() as u32 + 1)
                .expect("bounded node indices plus one are nonzero"),
        );
        if chain.is_empty() {
            chain.tail = Some(node);
        }
        self.positions.push(position, chain.head);
        chain.head = Some(node);
        chain.length += 1;
        if chain.length.is_multiple_of(Self::BLOCK_SIZE as u32) {
            self.checkpoints.insert(node.0.get(), chain.checkpoint);
            chain.checkpoint = Some(node);
        }
        Ok(())
    }
    pub(in super::super) const BLOCK_SIZE: usize = 4096;

    /// Producer checkpoints make every reverse encoding run bounded. Splitting
    /// visits only block metadata, never the whole linked position payload.
    pub(in super::super) fn split_reverse(&self, chain: PositionChain) -> Vec<PositionChain> {
        let mut output = Vec::with_capacity(chain.len().div_ceil(Self::BLOCK_SIZE));
        let mut head = chain.head;
        let mut remaining = chain.len();
        let partial = remaining % Self::BLOCK_SIZE;
        if partial != 0 {
            output.push(PositionChain {
                head,
                tail: None,
                length: partial as u32,
                checkpoint: None,
            });
            head = chain.checkpoint;
            remaining -= partial;
        }
        while remaining != 0 {
            let node = head.expect("producer retained the full block boundary");
            output.push(PositionChain {
                head,
                tail: None,
                length: Self::BLOCK_SIZE as u32,
                checkpoint: None,
            });
            head = self.checkpoints[&node.0.get()];
            remaining -= Self::BLOCK_SIZE;
        }
        output
    }
    /// Visit a chain from its most recently appended coordinate.
    #[inline]
    pub(in super::super) fn reversed(
        &self,
        chain: PositionChain,
    ) -> impl ExactSizeIterator<Item = u64> + Clone + '_ {
        ChainIter {
            chains: self,
            head: chain.head,
            remaining: chain.len(),
        }
    }
    /// Return the first appended coordinate in a chain.
    #[inline]
    pub(in super::super) fn first(&self, chain: PositionChain) -> Option<u64> {
        chain
            .tail
            .map(|node| self.positions.get(node.0.get() as usize - 1).0)
    }
    /// Return the last appended coordinate in a chain.
    #[inline]
    pub(in super::super) fn last(&self, chain: PositionChain) -> Option<u64> {
        chain
            .head
            .map(|node| self.positions.get(node.0.get() as usize - 1).0)
    }
}
#[derive(Clone)]
struct ChainIter<'a> {
    chains: &'a PositionChains,
    head: Option<PositionNode>,
    remaining: usize,
}
impl Iterator for ChainIter<'_> {
    type Item = u64;
    #[inline]
    fn next(&mut self) -> Option<u64> {
        if self.remaining == 0 {
            return None;
        }
        let node = self.head?;
        let index = node.0.get() as usize - 1;
        let (position, next) = self.chains.positions.get(index);
        self.head = *next;
        self.remaining -= 1;
        Some(position)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}
impl ExactSizeIterator for ChainIter<'_> {}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn interleaved_chains_keep_full_coordinates() {
        let mut chains = PositionChains::new();
        let mut a = PositionChain::default();
        let mut b = PositionChain::default();
        for (left, right) in [(0, 1 << 32), (1 << 63, u64::MAX)] {
            chains.push(&mut a, left).unwrap();
            chains.push(&mut b, right).unwrap();
        }
        assert_eq!(chains.reversed(a).collect::<Vec<_>>(), [1 << 63, 0]);
        assert_eq!(chains.reversed(b).collect::<Vec<_>>(), [u64::MAX, 1 << 32]);
    }
    #[test]
    fn producer_checkpoints_split_long_chains_and_copied_branches() {
        let mut chains = PositionChains::new();
        let mut a = PositionChain::default();
        for position in 0..6001 {
            chains.push(&mut a, position).unwrap();
        }
        let snapshot = a;
        let mut branch = a;
        for position in 6001..20_001 {
            chains.push(&mut a, position).unwrap();
            chains.push(&mut branch, position + (1 << 32)).unwrap();
        }
        for chain in [snapshot, a, branch] {
            let runs = chains.split_reverse(chain);
            assert!(
                runs.iter()
                    .all(|run| run.len() <= PositionChains::BLOCK_SIZE)
            );
            assert_eq!(
                runs.iter()
                    .flat_map(|&run| chains.reversed(run))
                    .collect::<Vec<_>>(),
                chains.reversed(chain).collect::<Vec<_>>()
            );
        }
    }
}
