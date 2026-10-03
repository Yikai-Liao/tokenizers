use ahash::AHashMap;

const NO_ENTRY: u32 = u32::MAX;
// Bound the dense u32 directory to 256 KiB; this does not narrow the ID domain.
const DENSE_IDS: usize = (256 << 10) / std::mem::size_of::<u32>();
/// Accumulates values by ID and drains only touched IDs.
/// A bounded dense directory avoids hashing; values occupy touched storage only.
pub struct IdAccumulator<T> {
    storage: Storage<T>,
}
enum Storage<T> {
    Dense {
        indices: Vec<u32>,
        entries: Vec<(u32, T)>,
    },
    Sparse(AHashMap<u32, T>),
}
impl<T: Default> IdAccumulator<T> {
    /// Choose a directory for the given ID domain.
    pub fn new(domain: usize) -> Self {
        Self {
            storage: if domain <= DENSE_IDS {
                Storage::Dense {
                    indices: vec![NO_ENTRY; domain],
                    entries: Vec::new(),
                }
            } else {
                Storage::Sparse(AHashMap::new())
            },
        }
    }
    /// Extend the ID domain after draining the previous chunk.
    pub fn ensure_domain(&mut self, domain: usize) {
        match &mut self.storage {
            Storage::Dense { indices, .. } if domain <= DENSE_IDS => {
                indices.resize(domain, NO_ENTRY)
            }
            Storage::Dense { .. } => self.storage = Storage::Sparse(AHashMap::new()),
            Storage::Sparse(_) => {}
        }
    }
    /// Return a touched value, creating its default value on the first visit.
    ///
    /// # Panics
    /// Panics when an ID lies outside a dense domain supplied to the constructor.
    // PERF: Touched-ID lookup runs for each adjacent boundary. Inlining lets
    // callers specialize the accumulator value without a per-boundary call.
    // The sparse insertion branch makes ordinary inline heuristics keep this
    // out of line even for dense token loops; keep the directory lookup local.
    #[inline(always)]
    pub fn touch(&mut self, id: u32) -> &mut T {
        match &mut self.storage {
            Storage::Dense { indices, entries } => {
                let index = &mut indices[id as usize];
                if *index == NO_ENTRY {
                    // Distinct touched IDs cannot exceed the bounded dense domain.
                    *index = entries.len() as u32;
                    entries.push((id, T::default()));
                }
                &mut entries[*index as usize].1
            }
            Storage::Sparse(values) => values.entry(id).or_default(),
        }
    }
    /// Look up a value without marking an ID as touched.
    pub fn get(&self, id: u32) -> Option<&T> {
        match &self.storage {
            Storage::Dense { indices, entries } => indices
                .get(id as usize)
                .filter(|&&index| index != NO_ENTRY)
                .map(|&index| &entries[index as usize].1),
            Storage::Sparse(values) => values.get(&id),
        }
    }
    /// Move out touched values, retaining allocations. Dropping the iterator
    /// removes its unconsumed values as well.
    pub fn drain(&mut self) -> impl Iterator<Item = (u32, T)> + '_ {
        match &mut self.storage {
            Storage::Dense { indices, entries } => IdDrain::Dense {
                entries: entries.drain(..),
                indices,
            },
            Storage::Sparse(values) => IdDrain::Sparse(values.drain()),
        }
    }
}
enum IdDrain<'a, T> {
    Dense {
        entries: std::vec::Drain<'a, (u32, T)>,
        indices: &'a mut [u32],
    },
    Sparse(std::collections::hash_map::Drain<'a, u32, T>),
}
impl<T> Iterator for IdDrain<'_, T> {
    type Item = (u32, T);
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Dense { entries, indices } => entries.next().map(|(id, value)| {
                indices[id as usize] = NO_ENTRY;
                (id, value)
            }),
            Self::Sparse(entries) => entries.next(),
        }
    }
}
impl<T> Drop for IdDrain<'_, T> {
    fn drop(&mut self) {
        if let Self::Dense { entries, indices } = self {
            for (id, _) in entries {
                indices[id as usize] = NO_ENTRY;
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn touched_values_do_not_survive_drain_in_either_domain() {
        for domain in [4, 100_000] {
            let mut counts = IdAccumulator::<usize>::new(domain);
            *counts.touch(1) += 3;
            *counts.touch(1) += 7;
            *counts.touch(3) += 11;
            let mut result: Vec<_> = counts.drain().collect();
            result.sort_unstable();
            assert_eq!(result, [(1, 10), (3, 11)]);
            assert_eq!(counts.get(1), None);
            assert_eq!(*counts.touch(1), 0);
            *counts.touch(3) = 99;
            drop(counts.drain().take(1));
            assert_eq!(counts.get(1), None);
            assert_eq!(counts.get(3), None);
            counts.ensure_domain(100_000);
            assert_eq!(*counts.touch(99_999), 0);
        }
    }
}
