// Heavy diagnostic-only counters; absent from the normal performance binaries.
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
pub(super) static BENCH_MERGING: AtomicBool = AtomicBool::new(false);
pub(super) static BENCH_APPEND: AtomicU64 = AtomicU64::new(0);
pub(super) static BENCH_TOKEN_READS: AtomicU64 = AtomicU64::new(0);
pub(super) static BENCH_WEIGHT_LOOKUPS: AtomicU64 = AtomicU64::new(0);
static BENCH_ACCEPTED: AtomicU64 = AtomicU64::new(0);
static BENCH_JOBS: AtomicU64 = AtomicU64::new(0);
static BENCH_CROSS_CUTS: AtomicU64 = AtomicU64::new(0);
static M_PRODUCERS: AtomicU64 = AtomicU64::new(0);
static M_STARTS: AtomicU64 = AtomicU64::new(0);
static M_EVENTS: AtomicU64 = AtomicU64::new(0);
static M_KEYS: AtomicU64 = AtomicU64::new(0);
static M_MAX_KEYS: AtomicU64 = AtomicU64::new(0);
static M_MAX_SCALAR_BYTES: AtomicU64 = AtomicU64::new(0);
static M_SUM_CPU_NS: AtomicU64 = AtomicU64::new(0);
static M_MAX_CPU_NS: AtomicU64 = AtomicU64::new(0);
static M_MAX_WALL_NS: AtomicU64 = AtomicU64::new(0);
static M_PRUNED_PRODUCERS: AtomicU64 = AtomicU64::new(0);
static M_PRUNED_STARTS: AtomicU64 = AtomicU64::new(0);

fn bench_jobs(jobs: &[Job]) {
    BENCH_ACCEPTED.fetch_add(jobs.iter().map(|j| j.starts.len() as u64).sum(), Relaxed);
    BENCH_JOBS.fetch_add(jobs.len() as u64, Relaxed);
}
pub(super) fn bench_report_work() {
    eprintln!("BPE_WINDOW_WORK {}", serde_json::json!({
        "accepted_starts":BENCH_ACCEPTED.load(Relaxed),"jobs":BENCH_JOBS.load(Relaxed),
        "positions_append_calls_during_merges":BENCH_APPEND.load(Relaxed),
        "corpus_token_reads_during_merges":BENCH_TOKEN_READS.load(Relaxed),
        "weight_region_lookups_during_merges":BENCH_WEIGHT_LOOKUPS.load(Relaxed),
        "cross_cohort_owned_final_cuts":BENCH_CROSS_CUTS.load(Relaxed),
    }));
    eprintln!("BPE_PRODUCER_DIAGNOSTIC {}", serde_json::json!({
        "coherent_producers":M_PRODUCERS.load(Relaxed),"accepted_starts_scanned":M_STARTS.load(Relaxed),
        "coherent_family_tallies":M_EVENTS.load(Relaxed),"coherent_family_keys":M_KEYS.load(Relaxed),
        "max_family_keys_per_producer":M_MAX_KEYS.load(Relaxed),
        "max_single_producer_scalar_capacity_bytes":M_MAX_SCALAR_BYTES.load(Relaxed),
        "reducer_sum_cpu_ns":M_SUM_CPU_NS.load(Relaxed),"reducer_max_cpu_ns":M_MAX_CPU_NS.load(Relaxed),
        "reducer_max_wall_ns":M_MAX_WALL_NS.load(Relaxed),
        "pruned_producers":M_PRUNED_PRODUCERS.load(Relaxed),"pruned_accepted_starts":M_PRUNED_STARTS.load(Relaxed),
    }));
}
#[allow(clippy::too_many_arguments)]
fn bench_window_report(
    births: &[Vec<Change<Positions>>], boundary: u32, cohorts: &[usize], replay: &[bool],
    guard: Option<(u64,u32)>, blocked: Option<(u64,Pair)>, reason: &str,
    high_water: usize, retained: usize, floor: u64,
) -> Result<()> {
    let mut counts = AHashMap::<Pair,u64>::new();
    for change in births.iter().flatten() { add(counts.entry(change.born).or_default(),change.born_weight)?; }
    let best=counts.iter().filter(|(_,c)|**c>=floor).map(|(p,c)|(*c,*p))
        .max_by(|a,b|a.0.cmp(&b.0).then_with(||b.1.cmp(&a.1)));
    eprintln!("BPE_WINDOW_DIAGNOSTIC {}",serde_json::json!({
        "boundary":boundary,"cohorts":cohorts.iter().max().map_or(0,|c|c+1),"rules":cohorts.len(),
        "exact_m":EXACT_M,"stop_reason":reason,"guard":guard,"blocked_priority":blocked,
        "actual_best_final_child":best,"pruned_outputs":replay.iter().filter(|r|!**r).count(),
        "retained_checkpoint_bytes":retained,"attempt_high_water_bytes":high_water,
        "final_complete_keys":counts.len(),"final_live_keys":counts.values().filter(|c|**c>=floor).count(),
    }));
    Ok(())
}
