// Diagnostic-only counters. Normal binaries do not contain this module.
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
static DIRECTORY_STATES: AtomicU64 = AtomicU64::new(0);
static DIRECTORY_RESIZES: AtomicU64 = AtomicU64::new(0);
static DIRECTORY_FILLED: AtomicU64 = AtomicU64::new(0);
static DIRECTORY_ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static DIRECTORY_REALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static DIRECTORY_OLD_LENGTH_ON_GROWTH: AtomicU64 = AtomicU64::new(0);
static DIRECTORY_LIVE_CAPACITY: AtomicU64 = AtomicU64::new(0);
static DIRECTORY_PEAK_CAPACITY: AtomicU64 = AtomicU64::new(0);
static DIRECTORY_MAX_DOMAIN: AtomicU64 = AtomicU64::new(0);
static DIRECTORY_LIVE_STATES: AtomicU64 = AtomicU64::new(0);
static DIRECTORY_PEAK_STATES: AtomicU64 = AtomicU64::new(0);
static DIRECTORY_THREADS: thread_local::ThreadLocal<()> = thread_local::ThreadLocal::new();

impl Default for Directories {
    fn default() -> Self {
        DIRECTORY_THREADS.get_or_default();
        DIRECTORY_STATES.fetch_add(1, Relaxed);
        let live = DIRECTORY_LIVE_STATES.fetch_add(1, Relaxed) + 1;
        DIRECTORY_PEAK_STATES.fetch_max(live, Relaxed);
        Self { indices: Default::default() }
    }
}
impl Drop for Directories {
    fn drop(&mut self) {
        let capacity = self.indices.iter().map(|v| v.capacity() as u64 * 4).sum();
        DIRECTORY_LIVE_CAPACITY.fetch_sub(capacity, Relaxed);
        DIRECTORY_LIVE_STATES.fetch_sub(1, Relaxed);
    }
}
pub(super) fn bench_report_directories() {
    eprintln!("BPE_DIRECTORIES {}", serde_json::json!({
        "states_created": DIRECTORY_STATES.load(Relaxed),
        "threads_observed": DIRECTORY_THREADS.iter().count(),
        "resize_calls": DIRECTORY_RESIZES.load(Relaxed),
        "sentinel_elements": DIRECTORY_FILLED.load(Relaxed),
        "sentinel_logical_bytes": DIRECTORY_FILLED.load(Relaxed) * 4,
        "vector_initial_allocations": DIRECTORY_ALLOCATIONS.load(Relaxed),
        "vector_capacity_growths": DIRECTORY_REALLOCATIONS.load(Relaxed),
        "old_elements_at_capacity_growth": DIRECTORY_OLD_LENGTH_ON_GROWTH.load(Relaxed),
        "peak_live_vector_capacity_bytes": DIRECTORY_PEAK_CAPACITY.load(Relaxed),
        "live_vector_capacity_bytes_at_report": DIRECTORY_LIVE_CAPACITY.load(Relaxed),
        "max_domain": DIRECTORY_MAX_DOMAIN.load(Relaxed),
        "peak_live_states": DIRECTORY_PEAK_STATES.load(Relaxed),
        "live_states_at_report": DIRECTORY_LIVE_STATES.load(Relaxed),
        "state_header_bytes": std::mem::size_of::<Directories>(),
        "scratch_owner_header_bytes": std::mem::size_of::<thread_local::ThreadLocal<std::cell::RefCell<Directories>>>()
    }));
}
