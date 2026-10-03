use crate::{Result, StorageError, position_storage::PositionStorage};
use std::num::NonZeroU32;

const MAX_NODES: usize = 1 << 27;

#[derive(Clone, Copy)]
struct PositionNode(NonZeroU32);
/// One chain within a bounded position buffer. Clear its owner only after all
/// chains have been consumed. Chain links are local indices, not coordinates.
#[derive(Clone, Copy, Default)]
pub struct PositionChain {
    head: Option<PositionNode>,
    tail: Option<PositionNode>,
    length: u32,
}
impl PositionChain {
    /// Return the number of linked positions.
    #[inline]
    pub fn len(&self) -> usize {
        self.length as usize
    }
    /// Return whether there are no linked positions.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
}
/// Bounded reverse chains share compact full-width positions and local links.
#[derive(Default)]
pub struct PositionChains {
    positions: PositionStorage<Option<PositionNode>>,
}
impl PositionChains {
    /// Construct an empty bounded buffer.
    pub fn new() -> Self {
        Self::default()
    }
    /// Append a coordinate to a chain. Node links remain within this buffer.
    ///
    /// # Errors
    /// Returns an error when the bounded buffer is full. The caller can consume
    /// the completed chunk and continue the same logical chain in a new chunk.
    // PERF: Merge preparation appends at most two nodes per rewritten token.
    // Cross-crate inlining combines the node budget and coordinate-plane work.
    #[inline]
    pub fn push(&mut self, chain: &mut PositionChain, position: u64) -> Result<()> {
        if self.positions.len() == MAX_NODES {
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
        Ok(())
    }
    /// Visit a chain from its most recently appended coordinate.
    #[inline]
    pub fn reversed(
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
    pub fn first(&self, chain: PositionChain) -> Option<u64> {
        chain
            .tail
            .map(|node| self.positions.get(node.0.get() as usize - 1).0)
    }
    /// Return the last appended coordinate in a chain.
    #[inline]
    pub fn last(&self, chain: PositionChain) -> Option<u64> {
        chain
            .head
            .map(|node| self.positions.get(node.0.get() as usize - 1).0)
    }
    /// Clear nodes after consuming all chains, while retaining allocation capacity.
    pub fn clear(&mut self) {
        self.positions.clear();
    }
    /// Return the number of nodes across all chains.
    #[inline]
    pub fn len(&self) -> usize {
        self.positions.len()
    }
    /// Return whether there are no nodes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.positions.len() == 0
    }
    /// Return the remaining node budget for this bounded chunk.
    #[inline]
    pub fn remaining_nodes(&self) -> usize {
        MAX_NODES - self.len()
    }
    /// Return the allocated bytes of the coordinate and link planes.
    pub fn capacity_bytes(&self) -> usize {
        self.positions.capacity_bytes()
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
}
