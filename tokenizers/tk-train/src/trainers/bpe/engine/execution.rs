//! One pool with reusable ID directories and encoding scratch.
use super::merge::SelectedRules;
use std::sync::Mutex;
use tk_collections::{IdDirectory, PositionEncodingScratch};
use tk_encode::Result;

pub(super) struct Execution {
    pub(super) pool: rayon::ThreadPool,
    encoding: Vec<Mutex<PositionEncodingScratch>>,
    directories: Vec<Mutex<[IdDirectory; 2]>>,
    selected: Mutex<SelectedRules>,
}
impl Execution {
    pub(super) fn new(workers: usize) -> Result<Self> {
        Ok(Self {
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
    pub(super) fn directories(&self) -> std::sync::MutexGuard<'_, [IdDirectory; 2]> {
        // The caller must finish sequential work before releasing this lease;
        // nested pool work could re-enter the same worker's directory lock.
        self.directories[self.current_worker()]
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }
    pub(super) fn selected_rules(&self) -> std::sync::MutexGuard<'_, SelectedRules> {
        self.selected
            .lock()
            .unwrap_or_else(|error| error.into_inner())
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
        shard: usize,
    ) -> std::sync::MutexGuard<'_, PositionEncodingScratch> {
        self.encoding[shard]
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }
}
