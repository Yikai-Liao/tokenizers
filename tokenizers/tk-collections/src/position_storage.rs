/// Private coordinate storage shared by sequential buffers and linked nodes.
/// The payload stays beside its low coordinate; a high plane appears only when
/// coordinates no longer share one high half.
#[derive(Default)]
pub(crate) struct PositionStorage<T> {
    items: Vec<(u32, T)>,
    high: Vec<u32>,
    shared_high: u32,
}
impl<T> PositionStorage<T> {
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.items.len()
    }
    // PERF: Node traversal needs the coordinate and its link together. Keeping
    // both in one item avoids a second scattered allocation access; buffers use
    // a zero-sized payload. Both forms share the same full-width promotion.
    #[inline]
    pub(crate) fn push(&mut self, position: u64, payload: T) {
        let high = (position >> 32) as u32;
        if self.items.is_empty() {
            self.shared_high = high;
        }
        if high != self.shared_high || !self.high.is_empty() {
            if self.high.is_empty() {
                self.high.resize(self.items.len(), self.shared_high);
            }
            self.high.push(high);
        }
        self.items.push((position as u32, payload));
    }
    #[inline]
    pub(crate) fn get(&self, index: usize) -> (u64, &T) {
        let item = &self.items[index];
        let position = (u64::from(self.high.get(index).copied().unwrap_or(self.shared_high)) << 32)
            | u64::from(item.0);
        (position, &item.1)
    }
    pub(crate) fn positions(&self) -> PositionIter<'_, T> {
        if self.high.is_empty() {
            PositionIter::Shared(self.items.iter(), u64::from(self.shared_high) << 32)
        } else {
            PositionIter::Split(self.items.iter().zip(self.high.iter()))
        }
    }
    pub(crate) fn clear(&mut self) {
        self.items.clear();
        self.high.clear();
    }
    pub(crate) fn capacity_bytes(&self) -> usize {
        self.items.capacity() * std::mem::size_of::<(u32, T)>() + self.high.capacity() * 4
    }
}

pub(crate) enum PositionIter<'a, T> {
    Shared(std::slice::Iter<'a, (u32, T)>, u64),
    Split(std::iter::Zip<std::slice::Iter<'a, (u32, T)>, std::slice::Iter<'a, u32>>),
}
impl<T> Iterator for PositionIter<'_, T> {
    type Item = u64;
    #[inline]
    fn next(&mut self) -> Option<u64> {
        match self {
            Self::Shared(items, high) => items.next().map(|item| *high | u64::from(item.0)),
            Self::Split(items) => items
                .next()
                .map(|(item, high)| (u64::from(*high) << 32) | u64::from(item.0)),
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
    #[inline]
    fn fold<B, F: FnMut(B, u64) -> B>(self, init: B, mut f: F) -> B {
        match self {
            Self::Shared(items, high) => {
                items.fold(init, |acc, item| f(acc, high | u64::from(item.0)))
            }
            Self::Split(items) => items.fold(init, |acc, (item, high)| {
                f(acc, (u64::from(*high) << 32) | u64::from(item.0))
            }),
        }
    }
}
impl<T> DoubleEndedIterator for PositionIter<'_, T> {
    #[inline]
    fn next_back(&mut self) -> Option<u64> {
        match self {
            Self::Shared(items, high) => items.next_back().map(|item| *high | u64::from(item.0)),
            Self::Split(items) => items
                .next_back()
                .map(|(item, high)| (u64::from(*high) << 32) | u64::from(item.0)),
        }
    }
    #[inline]
    fn rfold<B, F: FnMut(B, u64) -> B>(self, init: B, mut f: F) -> B {
        match self {
            Self::Shared(items, high) => {
                items.rfold(init, |acc, item| f(acc, high | u64::from(item.0)))
            }
            Self::Split(items) => items.rfold(init, |acc, (item, high)| {
                f(acc, (u64::from(*high) << 32) | u64::from(item.0))
            }),
        }
    }
}
impl<T> ExactSizeIterator for PositionIter<'_, T> {
    fn len(&self) -> usize {
        match self {
            Self::Shared(items, _) => items.len(),
            Self::Split(items) => items.len(),
        }
    }
}
