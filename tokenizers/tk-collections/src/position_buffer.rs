use crate::position_storage::PositionStorage;

/// Full-width coordinates with one shared high half until positions cross it.
#[derive(Default)]
pub struct PositionBuffer {
    storage: PositionStorage<()>,
}
impl PositionBuffer {
    pub fn new() -> Self {
        Self::default()
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.storage.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    #[inline]
    pub fn push(&mut self, position: u64) {
        self.storage.push(position, ());
    }
    #[inline]
    pub fn get(&self, index: usize) -> u64 {
        self.storage.get(index).0
    }
    #[inline]
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = u64> + ExactSizeIterator + '_ {
        self.storage.positions()
    }
    pub fn clear(&mut self) {
        self.storage.clear();
    }
    pub fn capacity_bytes(&self) -> usize {
        self.storage.capacity_bytes()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn promotion_and_reuse_preserve_full_positions() {
        let mut positions = PositionBuffer::new();
        for values in [
            vec![0, 7, u32::MAX as u64],
            vec![1 << 32, (1 << 32) + 1, 0, 1 << 63, u64::MAX],
            vec![u64::MAX, u64::MAX],
        ] {
            positions.clear();
            for &value in &values {
                positions.push(value);
            }
            assert_eq!(positions.iter().collect::<Vec<_>>(), values);
            assert_eq!(
                positions.iter().rev().collect::<Vec<_>>(),
                values.into_iter().rev().collect::<Vec<_>>()
            );
        }
    }
}
