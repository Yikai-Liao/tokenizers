//! One training pool, task-local merge scratch, and reusable encoding scratch.
use super::merge::MergeScratch;
use std::sync::Mutex;
use tk_collections::PositionEncodingScratch;
use tk_encode::Result;

pub(super) struct Execution {
    pub(super) pool: rayon::ThreadPool,
    encoding: Vec<Mutex<PositionEncodingScratch>>,
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
        })
    }
    pub(super) fn merge_scratch(&self) -> MergeScratch {
        MergeScratch::new(self.workers())
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
