use super::{Result, StorageError};
use bumpalo::Bump;
use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::{Mutex, MutexGuard};

/// Owns small allocations until the complete algorithm scope ends.
/// Leases borrow this owner; releasing a lease never frees published storage.
pub(in super::super) struct AllocationArena {
    workers: Vec<Mutex<Bump>>,
    cutoff: usize,
}
impl AllocationArena {
    pub(in super::super) fn new(workers: usize, physical_items: usize) -> Self {
        Self {
            workers: (0..workers).map(|_| Mutex::new(Bump::new())).collect(),
            // A fixed decision for the scope; large buffers remain individually owned.
            cutoff: ((physical_items as u128 / 256).isqrt() as usize).max(256),
        }
    }
    pub(in super::super) fn lease(&self, worker: usize) -> AllocationLease<'_> {
        AllocationLease {
            arena: self,
            cursor: self.workers[worker]
                .lock()
                .unwrap_or_else(|e| e.into_inner()),
        }
    }
}
/// Exclusive access to one allocation cursor. Lists borrow the arena, not this lock.
pub(in super::super) struct AllocationLease<'a> {
    arena: &'a AllocationArena,
    cursor: MutexGuard<'a, Bump>,
}
impl AllocationLease<'_> {
    pub(in super::super) fn allocate(&self, layout: Layout) -> Result<Option<NonNull<u8>>> {
        if layout.size() > self.arena.cutoff {
            return Ok(None);
        }
        self.cursor
            .try_alloc_layout(layout)
            .map(Some)
            .map_err(|_| StorageError("position arena allocation failed"))
    }
}
