//! Per-training posting arenas. Threshold selection happens once, after the
//! physical edge count is known; it never changes while postings are alive.
//! Dedicated initialization and merge pools stay alive until all postings drop.
use bumpalo::Bump;
use rayon::ThreadPool;
use serde::Serialize;
use std::alloc::Layout;
use std::cell::RefCell;
use std::ptr::NonNull;
use tk_encode::Result;

#[derive(Clone, Copy, Debug)]
pub(super) enum Policy {
    System,
    Fixed(usize),
    /// Experimental gentle-growth rule, not a fitted universal posting law.
    /// max(256 B, 256 B * sqrt(E_u / 16 Mi edges)), rounded down to bytes.
    Auto,
}
impl Policy {
    pub(super) fn cutoff(self, physical_edges: usize) -> usize {
        match self {
            Self::System => 0,
            Self::Fixed(bytes) => bytes,
            Self::Auto => ((physical_edges as u128 / 256).isqrt() as usize).max(256),
        }
    }
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::Fixed(_) => "fixed",
            Self::Auto => "auto",
        }
    }
}

#[derive(Default, Debug, Clone, Serialize)]
pub(in crate::trainers::bpe) struct Counters {
    pub arena_buffers: usize,
    pub arena_requested_bytes: usize,
    pub arena_retired_bytes: usize,
    pub heap_buffers: usize,
    pub heap_requested_bytes: usize,
    pub heap_frees: usize,
    pub heap_freed_bytes: usize,
    pub grows: usize,
    pub backing_bytes: usize,
    pub backing_bytes_including_metadata: usize,
    pub chunks: usize,
}
impl Counters {
    fn add(&mut self, rhs: Self) {
        self.arena_buffers += rhs.arena_buffers;
        self.arena_requested_bytes += rhs.arena_requested_bytes;
        self.arena_retired_bytes += rhs.arena_retired_bytes;
        self.heap_buffers += rhs.heap_buffers;
        self.heap_requested_bytes += rhs.heap_requested_bytes;
        self.heap_frees += rhs.heap_frees;
        self.heap_freed_bytes += rhs.heap_freed_bytes;
        self.grows += rhs.grows;
        self.backing_bytes += rhs.backing_bytes;
        self.backing_bytes_including_metadata += rhs.backing_bytes_including_metadata;
        self.chunks += rhs.chunks;
    }
}
#[derive(Default)]
struct WorkerArena {
    bump: Bump,
    cutoff: usize,
    counters: Counters,
}
thread_local! {
    static ACTIVE: RefCell<Option<WorkerArena>> = const { RefCell::new(None) };
}

/// A training call owns fresh dedicated pools. No posting may escape the
/// train_typed call: only owned vocab, merges and scalar stats are returned.
/// Drop also runs after a failed/unwound train once Rayon jobs have joined.
pub(super) struct Session<'a> {
    pool: &'a ThreadPool,
    initialization_pool: Option<&'a ThreadPool>,
    finished: bool,
}
impl<'a> Session<'a> {
    pub(super) fn new(pool: &'a ThreadPool, initialization_pool: Option<&'a ThreadPool>) -> Self {
        let session = Self {
            pool,
            initialization_pool,
            finished: false,
        };
        let start = || {
            ACTIVE.with(|cell| {
                let mut active = cell.borrow_mut();
                assert!(
                    active.is_none(),
                    "posting arena requires a fresh dedicated pool"
                );
                *active = Some(WorkerArena::default());
            })
        };
        pool.broadcast(|_| start());
        if let Some(pool) = initialization_pool {
            pool.broadcast(|_| start());
        }
        session
    }
    pub(super) fn finish(mut self) -> Counters {
        let mut counters = Counters::default();
        for stats in self.pool.broadcast(|_| finish_worker()) {
            counters.add(stats);
        }
        if let Some(pool) = self.initialization_pool {
            for stats in pool.broadcast(|_| finish_worker()) {
                counters.add(stats);
            }
        }
        self.finished = true;
        debug_assert_eq!(counters.arena_requested_bytes, counters.arena_retired_bytes);
        debug_assert_eq!(counters.heap_requested_bytes, counters.heap_freed_bytes);
        debug_assert_eq!(counters.heap_buffers, counters.heap_frees);
        counters
    }
}
impl Drop for Session<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.pool.broadcast(|_| {
                finish_worker();
            });
            if let Some(pool) = self.initialization_pool {
                pool.broadcast(|_| {
                    finish_worker();
                });
            }
        }
    }
}
fn finish_worker() -> Counters {
    ACTIVE.with(|cell| {
        let Some(mut arena) = cell.borrow_mut().take() else {
            return Counters::default();
        };
        arena.counters.backing_bytes = arena.bump.allocated_bytes();
        arena.counters.backing_bytes_including_metadata =
            arena.bump.allocated_bytes_including_metadata();
        arena.counters.chunks = arena.bump.iter_allocated_chunks().count();
        // All posting owners are already destroyed; dropping bump is now safe.
        arena.counters
    })
}

pub(super) fn configure(
    pool: &ThreadPool,
    initialization_pool: Option<&ThreadPool>,
    cutoff: usize,
) {
    let configure = || {
        ACTIVE.with(|cell| {
            cell.borrow_mut()
                .as_mut()
                .expect("training arena session missing")
                .cutoff = cutoff;
        })
    };
    pool.broadcast(|_| configure());
    if let Some(pool) = initialization_pool {
        pool.broadcast(|_| configure());
    }
}

pub(super) fn eligible<T>(capacity: usize) -> bool {
    // Low pointer bit carries allocator origin. Align-1 types and ZSTs stay Vec.
    if std::mem::align_of::<T>() < 2 || std::mem::size_of::<T>() == 0 {
        return false;
    }
    let Some(bytes) = capacity.checked_mul(std::mem::size_of::<T>()) else {
        return false;
    };
    ACTIVE.with(|cell| cell.borrow().as_ref().is_some_and(|a| bytes <= a.cutoff))
}
pub(super) fn allocate<T>(capacity: usize, growth: bool) -> Result<Option<NonNull<T>>> {
    if std::mem::align_of::<T>() < 2 || std::mem::size_of::<T>() == 0 {
        return Ok(None);
    }
    let layout = Layout::array::<T>(capacity).map_err(|_| "posting allocation layout overflow")?;
    ACTIVE.with(|cell| {
        let mut active = cell.borrow_mut();
        let Some(arena) = active.as_mut() else {
            return Ok(None);
        };
        if layout.size() > arena.cutoff {
            return Ok(None);
        }
        let pointer = arena
            .bump
            .try_alloc_layout(layout)
            .map_err(|_| "posting arena allocation failed")?;
        arena.counters.arena_buffers += 1;
        arena.counters.arena_requested_bytes += layout.size();
        arena.counters.grows += usize::from(growth);
        Ok(Some(pointer.cast::<T>()))
    })
}
pub(super) fn heap_allocation<T>(capacity: usize, growth: bool) {
    ACTIVE.with(|cell| {
        if let Some(arena) = cell.borrow_mut().as_mut() {
            arena.counters.heap_buffers += 1;
            arena.counters.heap_requested_bytes += capacity * std::mem::size_of::<T>();
            arena.counters.grows += usize::from(growth);
        }
    });
}
pub(super) fn retirement<T>(capacity: usize, is_arena: bool) {
    ACTIVE.with(|cell| {
        if let Some(arena) = cell.borrow_mut().as_mut() {
            let bytes = capacity * std::mem::size_of::<T>();
            if is_arena {
                arena.counters.arena_retired_bytes += bytes;
            } else {
                arena.counters.heap_frees += 1;
                arena.counters.heap_freed_bytes += bytes;
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn session_cleans_up_after_error_and_worker_unwind() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let init = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        for panic in [false, true] {
            let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _session = Session::new(&pool, Some(&init));
                configure(&pool, Some(&init), 256);
                pool.install(|| -> Result<()> {
                    let mut posting = super::super::small_posting::SmallPosting::with_capacity(16)?;
                    posting.push(7)?;
                    assert_eq!(posting.as_slice(), &[7]);
                    if panic {
                        panic!("arena unwind probe");
                    }
                    Err("arena error probe".into())
                })
            }));
            assert_eq!(failed.is_err(), panic);
            for vacant in pool.broadcast(|_| ACTIVE.with(|a| a.borrow().is_none())) {
                assert!(vacant);
            }
            assert!(init.install(|| ACTIVE.with(|a| a.borrow().is_none())));
        }
        let session = Session::new(&pool, Some(&init));
        assert_eq!(session.finish().arena_buffers, 0);
    }

    #[test]
    fn threshold_floor_growth_and_full_address_domain() {
        assert_eq!(Policy::Auto.cutoff(0), 256);
        assert_eq!(Policy::Auto.cutoff(16 << 20), 256);
        assert_eq!(Policy::Auto.cutoff(64 << 20), 512);
        assert_eq!(Policy::Auto.cutoff(256 << 20), 1024);
        assert_eq!(
            Policy::Auto.cutoff(usize::MAX),
            ((usize::MAX as u128 / 256).isqrt() as usize).max(256)
        );
        assert_eq!(Policy::System.cutoff(usize::MAX), 0);
        assert_eq!(Policy::Fixed(usize::MAX).cutoff(0), usize::MAX);
    }
}
