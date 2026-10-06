use ahash::AHashMap;

const NO_ENTRY: u32 = u32::MAX;
// Bound the dense u32 directory to 256 KiB; this does not narrow the ID domain.
const DENSE_IDS: usize = (256 << 10) / std::mem::size_of::<u32>();
/// Reusable bounded ID-to-entry storage, independent of accumulated values.
/// Returning a directory releases every value and its backing allocation.
#[derive(Default)]
pub(in super::super) struct IdDirectory {
    indices: Vec<u32>,
    drain_pending: bool,
}
impl IdDirectory {
    fn ensure_domain(&mut self, domain: usize) {
        // Grow geometrically without allowing Vec's doubling to exceed the
        // bounded dense domain near its upper limit.
        if domain > self.indices.capacity() {
            self.indices
                .reserve_exact(domain.next_power_of_two() - self.indices.len());
        }
        self.indices.resize(domain, NO_ENTRY);
    }
}
/// Accumulates values by ID and drains only touched IDs.
/// A bounded dense directory avoids hashing; values occupy touched storage only.
pub(in super::super) struct IdAccumulator<T> {
    storage: Storage<T>,
}
enum Storage<T> {
    Dense {
        directory: IdDirectory,
        entries: Vec<(u32, T)>,
    },
    Sparse(AHashMap<u32, T>),
}
impl<T: Default> IdAccumulator<T> {
    /// Reuse a previously returned directory for a new value type and domain.
    /// Large domains use sparse storage and release the bounded directory.
    pub(in super::super) fn with_directory(domain: usize, mut directory: IdDirectory) -> Self {
        Self {
            storage: if domain <= DENSE_IDS {
                directory.ensure_domain(domain);
                Storage::Dense {
                    directory,
                    entries: Vec::new(),
                }
            } else {
                Storage::Sparse(AHashMap::new())
            },
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
    pub(in super::super) fn touch(&mut self, id: u32) -> &mut T {
        match &mut self.storage {
            Storage::Dense { directory, entries } => {
                let index = &mut directory.indices[id as usize];
                if *index == NO_ENTRY {
                    let value = T::default();
                    entries.push((id, value));
                    // Distinct touched IDs cannot exceed the bounded dense domain.
                    // Publish only after construction, so a caught default-value
                    // panic cannot leave a stale index in a reusable directory.
                    *index = (entries.len() - 1) as u32;
                }
                &mut entries[*index as usize].1
            }
            Storage::Sparse(values) => values.entry(id).or_default(),
        }
    }
    /// Look up a value without marking an ID as touched.
    #[cfg(test)]
    fn get(&self, id: u32) -> Option<&T> {
        match &self.storage {
            Storage::Dense { directory, entries } => directory
                .indices
                .get(id as usize)
                .filter(|&&index| index != NO_ENTRY)
                .map(|&index| &entries[index as usize].1),
            Storage::Sparse(values) => values.get(&id),
        }
    }
    /// Move out touched values, retaining allocations. Dropping the iterator
    /// removes its unconsumed values as well.
    pub(in super::super) fn drain(&mut self) -> impl Iterator<Item = (u32, T)> + '_ {
        match &mut self.storage {
            Storage::Dense { directory, entries } => {
                let prior_pending = directory.drain_pending;
                directory.drain_pending = true;
                IdDrain::Dense {
                    entries: entries.drain(..),
                    indices: &mut directory.indices,
                    pending: &mut directory.drain_pending,
                    prior_pending,
                }
            }
            Storage::Sparse(values) => IdDrain::Sparse(values.drain()),
        }
    }
}
impl<T> IdAccumulator<T> {
    /// Release values and return only reusable ID lookup storage.
    /// Unconsumed touched IDs are reset before the directory changes owners.
    pub(in super::super) fn into_directory(self) -> IdDirectory {
        match self.storage {
            Storage::Dense {
                mut directory,
                entries,
            } => {
                if directory.drain_pending {
                    // A forgotten Vec::Drain has removed its entries from the
                    // vector without running our touched-ID cleanup. Recover
                    // only this exceptional transfer with a complete reset.
                    directory.indices.fill(NO_ENTRY);
                    directory.drain_pending = false;
                } else {
                    for (id, _) in entries {
                        directory.indices[id as usize] = NO_ENTRY;
                    }
                }
                directory
            }
            Storage::Sparse(_) => IdDirectory::default(),
        }
    }
}
enum IdDrain<'a, T> {
    Dense {
        entries: std::vec::Drain<'a, (u32, T)>,
        indices: &'a mut [u32],
        pending: &'a mut bool,
        prior_pending: bool,
    },
    Sparse(std::collections::hash_map::Drain<'a, u32, T>),
}
impl<T> Iterator for IdDrain<'_, T> {
    type Item = (u32, T);
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Dense {
                entries, indices, ..
            } => entries.next().map(|(id, value)| {
                indices[id as usize] = NO_ENTRY;
                (id, value)
            }),
            Self::Sparse(entries) => entries.next(),
        }
    }
}
impl<T> Drop for IdDrain<'_, T> {
    fn drop(&mut self) {
        if let Self::Dense {
            entries,
            indices,
            pending,
            prior_pending,
        } = self
        {
            for (id, _) in entries {
                indices[id as usize] = NO_ENTRY;
            }
            // Completing this drain cannot clean IDs leaked by an earlier
            // forgotten drain. Preserve that state until ownership transfer.
            **pending = *prior_pending;
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn touched_values_do_not_survive_drain_in_either_domain() {
        for domain in [4, 100_000] {
            let mut counts = IdAccumulator::<usize>::with_directory(domain, IdDirectory::default());
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
            let mut counts =
                IdAccumulator::<usize>::with_directory(100_000, counts.into_directory());
            assert_eq!(*counts.touch(99_999), 0);
        }
    }
    #[test]
    fn recycled_directory_drops_values_and_resets_unconsumed_ids() {
        use std::{cell::Cell, rc::Rc};
        let drops = Rc::new(Cell::new(0));
        #[derive(Default)]
        struct Value(Option<Rc<Cell<usize>>>);
        impl Drop for Value {
            fn drop(&mut self) {
                if let Some(drops) = &self.0 {
                    drops.set(drops.get() + 1);
                }
            }
        }
        let mut values = IdAccumulator::<Value>::with_directory(4, IdDirectory::default());
        values.touch(1).0 = Some(drops.clone());
        values.touch(3).0 = Some(drops.clone());
        let mut drain = values.drain();
        drop(drain.next());
        drop(drain);
        values.touch(2).0 = Some(drops.clone());
        let directory = values.into_directory();
        assert_eq!(drops.get(), 3);
        let mut counts = IdAccumulator::<usize>::with_directory(8, directory);
        for id in [1, 2, 3, 7] {
            assert_eq!(*counts.touch(id), 0);
        }
        *counts.touch(7) = 19;
        let directory = counts.into_directory();
        let mut large = IdAccumulator::<usize>::with_directory(100_000, directory);
        assert_eq!(*large.touch(99_999), 0);
        let mut small = IdAccumulator::<usize>::with_directory(2, large.into_directory());
        assert_eq!(*small.touch(1), 0);
    }
    #[test]
    fn directory_growth_stays_within_the_dense_budget() {
        let mut directory = IdDirectory::default();
        for domain in [8_010, 16_021, 32_043, 64_087, DENSE_IDS] {
            let mut counts = IdAccumulator::<u64>::with_directory(domain, directory);
            *counts.touch((domain - 1) as u32) = 7;
            directory = counts.into_directory();
            assert!(directory.indices.capacity() <= DENSE_IDS);
            assert!(directory.indices.iter().all(|&index| index == NO_ENTRY));
        }
    }
    #[test]
    fn caught_value_construction_panic_leaves_a_reusable_directory() {
        struct Panics;
        impl Default for Panics {
            fn default() -> Self {
                panic!("value construction failed")
            }
        }
        let mut values = IdAccumulator::<Panics>::with_directory(4, IdDirectory::default());
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = values.touch(2);
            }))
            .is_err()
        );
        let mut counts = IdAccumulator::<u64>::with_directory(4, values.into_directory());
        assert_eq!(counts.get(2), None);
        assert_eq!(*counts.touch(2), 0);
    }
    #[test]
    fn forgotten_drain_does_not_transfer_dirty_ids_to_another_accumulator() {
        for later_drains in 0..3 {
            let mut counts = IdAccumulator::<u64>::with_directory(2, IdDirectory::default());
            *counts.touch(1) = 7;
            std::mem::forget(counts.drain());
            for _ in 0..later_drains {
                drop(counts.drain());
            }
            let mut counts = IdAccumulator::<usize>::with_directory(2, counts.into_directory());
            *counts.touch(0) = 99;
            assert_eq!(counts.get(1), None);
            assert_eq!(*counts.touch(1), 0);
        }
    }
}
