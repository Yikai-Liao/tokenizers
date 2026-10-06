//! One pool with reusable ID directories and encoding scratch.
use super::merge::MergeScratch;
use super::storage::{IdAccumulator, IdDirectory, PositionEncodingScratch};
use super::{merge::SelectedRuleIndex, pair_index::ShardRouter};
use std::sync::Mutex;
use tk_encode::Result;

pub(super) struct Execution {
    pub(super) pool: rayon::ThreadPool,
    encoding: Vec<Mutex<PositionEncodingScratch>>,
    directories: Vec<Mutex<[IdDirectory; 2]>>,
    selected: Mutex<SelectedRuleIndex>,
    router: ShardRouter,
}
impl Execution {
    pub(super) fn new(workers: usize) -> Result<Self> {
        Ok(Self {
            router: ShardRouter::new(workers),
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()?,
            encoding: (0..workers)
                .map(|_| Mutex::new(PositionEncodingScratch::default()))
                .collect(),
            directories: (0..workers).map(|_| Mutex::default()).collect(),
            selected: Mutex::default(),
        })
    }
    pub(super) fn router(&self) -> ShardRouter {
        self.router
    }
    fn directories(&self) -> std::sync::MutexGuard<'_, [IdDirectory; 2]> {
        // The caller must finish sequential work before releasing this lease;
        // nested pool work could re-enter the same worker's directory lock.
        self.directories[self.current_worker()]
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }
    /// Run sequential task work and return only reusable lookup directories.
    /// Errors reset touched IDs as well. Unwinding drops values and unlocks the
    /// worker; the next task starts with empty directories. Never nest pool work.
    pub(super) fn with_merge_scratch<T>(
        &self,
        token_id_count: usize,
        work: impl FnOnce(&mut MergeScratch) -> Result<T>,
    ) -> Result<T> {
        let mut directories = self.directories();
        let mut scratch = MergeScratch::new(token_id_count, std::mem::take(&mut *directories));
        let result = work(&mut scratch);
        *directories = scratch.into_directories();
        result
    }
    /// Reuse one directory for owner-local accumulation within a joined phase.
    /// The closure is sequential and must not start nested pool work.
    pub(super) fn with_accumulator<V: Default, T>(
        &self,
        token_id_count: usize,
        directory_index: usize,
        work: impl FnOnce(&mut IdAccumulator<V>) -> Result<T>,
    ) -> Result<T> {
        let mut directories = self.directories();
        let mut values = IdAccumulator::with_directory(
            token_id_count,
            std::mem::take(&mut directories[directory_index]),
        );
        let result = work(&mut values);
        directories[directory_index] = values.into_directory();
        result
    }
    pub(super) fn selected_rules(&self) -> std::sync::MutexGuard<'_, SelectedRuleIndex> {
        self.selected
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }
    pub(super) fn release_scratch(&self) {
        // Every pool phase has joined. Output construction no longer needs
        // directories or codec buffers, so release them before duplicating
        // vocabulary strings for the public model.
        for directory in &self.directories {
            *directory.lock().unwrap_or_else(|error| error.into_inner()) = Default::default();
        }
        for encoding in &self.encoding {
            *encoding.lock().unwrap_or_else(|error| error.into_inner()) = Default::default();
        }
        *self
            .selected
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Default::default();
    }
    pub(super) fn workers(&self) -> usize {
        self.encoding.len()
    }
    pub(super) fn current_worker(&self) -> usize {
        // Allocation cursors follow executing workers, as the original TLS
        // storage did. Obtain the index inside this pool's task. Encoding with
        // either lease held must remain sequential, without nested pool work.
        self.pool
            .current_thread_index()
            .expect("BPE allocation runs inside its training pool")
    }
    pub(super) fn encoding(
        &self,
        worker_id: usize,
    ) -> std::sync::MutexGuard<'_, PositionEncodingScratch> {
        self.encoding[worker_id]
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn worker_directories_are_clean_after_errors_and_unwinding() {
        let execution = Execution::new(1).unwrap();
        execution.pool.install(|| {
            let result = execution.with_accumulator::<u64, ()>(4, 0, |values| {
                *values.touch(2) = 99;
                Err("interrupted task".into())
            });
            assert!(result.is_err());
            execution
                .with_accumulator::<u64, _>(4, 0, |values| {
                    assert_eq!(*values.touch(2), 0);
                    Ok(())
                })
                .unwrap();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = execution.with_accumulator::<u64, ()>(4, 0, |values| {
                    *values.touch(2) = 99;
                    panic!("task unwound");
                });
            }));
            assert!(result.is_err());
            execution
                .with_accumulator::<u64, _>(4, 0, |values| {
                    assert_eq!(*values.touch(2), 0);
                    Ok(())
                })
                .unwrap();
        });
    }
}
