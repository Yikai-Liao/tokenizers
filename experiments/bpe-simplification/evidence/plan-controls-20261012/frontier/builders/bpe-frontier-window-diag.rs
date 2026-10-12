// Heavy diagnostic counters and retrospective analysis; absent from production.
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
pub(super) static BENCH_MERGING: AtomicBool = AtomicBool::new(false);
pub(super) static BENCH_APPEND: AtomicU64 = AtomicU64::new(0);
pub(super) static BENCH_TOKEN_READS: AtomicU64 = AtomicU64::new(0);
pub(super) static BENCH_WEIGHT_LOOKUPS: AtomicU64 = AtomicU64::new(0);
static BENCH_DECODED: AtomicU64 = AtomicU64::new(0);
static BENCH_ACCEPTED: AtomicU64 = AtomicU64::new(0);
static BENCH_JOBS: AtomicU64 = AtomicU64::new(0);
static BENCH_CROSS_CUTS: AtomicU64 = AtomicU64::new(0);
static BENCH_RIGHT_PREDICTIONS: AtomicU64 = AtomicU64::new(0);
fn bench_jobs(jobs: &[Job]) {
    BENCH_ACCEPTED.fetch_add(jobs.iter().map(|j| j.starts.len() as u64).sum(), Relaxed);
    BENCH_JOBS.fetch_add(jobs.len() as u64, Relaxed);
}
pub(super) fn bench_report_work() {
    eprintln!("BPE_WINDOW_WORK {}", serde_json::json!({
        "historical_coordinates_decoded": BENCH_DECODED.load(Relaxed),
        "accepted_starts": BENCH_ACCEPTED.load(Relaxed), "jobs": BENCH_JOBS.load(Relaxed),
        "positions_append_calls_during_merges": BENCH_APPEND.load(Relaxed),
        "corpus_token_reads_during_merges": BENCH_TOKEN_READS.load(Relaxed),
        "weight_region_lookups_during_merges": BENCH_WEIGHT_LOOKUPS.load(Relaxed),
        "cross_cohort_owned_final_cuts": BENCH_CROSS_CUTS.load(Relaxed),
        "selected_right_prediction_lookups": BENCH_RIGHT_PREDICTIONS.load(Relaxed),
    }));
}

// Exact coherent family counts from accepted starts before this cohort writes.
// This extra scan/hash map measures the M opportunity, not M's proposed cost.
fn bench_coherent(snapshot: &FreshSnapshot<'_>, jobs: &[Job], boundary: u32) -> Result<Vec<(u32, u64, u64)>> {
    BENCH_MERGING.store(false, Relaxed);
    let corpus = snapshot.corpus;
    let mut families = AHashMap::<(u32, bool, u32), u64>::new();
    let mut producers = AHashMap::<u32, (u64, u64)>::new();
    for job in jobs {
        let span = corpus.id_span(job.replacement);
        producers.entry(job.replacement).or_default().1 += job.starts.len() as u64;
        for &p in &job.starts {
            let weight = corpus.weight_region(p).0;
            let prior = corpus.token(p - 1);
            if prior != WORD_SEPARATOR_ID && prior < boundary {
                let before = p - corpus.id_span(prior);
                let merging = match &snapshot.matches {
                    FreshMatches::SelfPair(starts) => p >= span && starts.binary_search(&((p-span) as u64)).is_ok(),
                    FreshMatches::Ordinary(selected) => before != 0 && selected.contains_key(&(corpus.token(before-1), prior)),
                };
                if !merging && corpus.id_span(prior) + span < snapshot.limit {
                    add(families.entry((job.replacement, true, prior)).or_default(), weight)?;
                }
            }
            let next = corpus.token(p + span);
            if next != WORD_SEPARATOR_ID && next < boundary {
                let replacement = match &snapshot.matches {
                    FreshMatches::SelfPair(starts) => starts.binary_search(&((p+span) as u64)).is_ok().then_some(job.replacement),
                    FreshMatches::Ordinary(selected) => selected.get(&(next, corpus.token(p+span+corpus.id_span(next)))).copied(),
                };
                let born = replacement.unwrap_or(next);
                if span + corpus.id_span(born) < snapshot.limit {
                    add(families.entry((job.replacement, false, born)).or_default(), weight)?;
                }
            }
        }
    }
    for ((producer, _, _), mass) in families {
        let maximum = &mut producers.get_mut(&producer).expect("observed producer").0;
        *maximum = (*maximum).max(mass);
    }
    BENCH_MERGING.store(true, Relaxed);
    Ok(producers.into_iter().map(|(id, (m, a))| (id, m, a)).collect())
}

fn bench_window_report(
    births: &[Vec<Change<Positions>>], job_producers: &[u32], coherent: &[(u32, u64, u64)],
    boundary: u32, cohorts: &[usize], guard: Option<(u64,u32)>, blocked: Option<(u64,Pair)>,
    reason: &str, high_water: usize, retained: usize, floor: u64,
) -> Result<()> {
    let mut counts = AHashMap::<Pair, u64>::new();
    let rare: AHashSet<_> = coherent.iter().filter(|(_,m,_)| *m < floor).map(|(id,_,_)| *id).collect();
    let (mut rare_coordinates, mut rare_bytes) = (0u64, 0u64);
    for (changes, producer) in births.iter().zip(job_producers) {
        for change in changes {
            add(counts.entry(change.born).or_default(), change.born_weight)?;
            if rare.contains(producer) {
                rare_coordinates += change.positions.len() as u64;
                rare_bytes += change.positions.bench_storage().0;
            }
        }
    }
    let best = counts.iter().filter(|(_,c)| **c >= floor).map(|(p,c)| (*c,*p)).max_by(|a,b| a.0.cmp(&b.0).then_with(||b.1.cmp(&a.1)));
    let guard_false_positive = reason == "guard" && blocked.is_some_and(|old| best.is_none_or(|child| old.0 > child.0 || (old.0 == child.0 && old.1 < child.1)));
    eprintln!("BPE_WINDOW_DIAGNOSTIC {}", serde_json::json!({
        "boundary":boundary,"cohorts":cohorts.iter().max().map_or(0,|x|x+1),"rules":cohorts.len(),
        "stop_reason":reason,"guard":guard,"blocked_priority":blocked,"actual_best_final_child":best,
        "guard_false_positive_at_drain":guard_false_positive,
        "coherent_producers":coherent.len(),"coherent_rare_producers":rare.len(),
        "coherent_rare_accepted_starts":coherent.iter().filter(|(id,_,_)|rare.contains(id)).map(|(_,_,a)|a).sum::<u64>(),
        "coherent_rare_final_coordinates":rare_coordinates,"coherent_rare_final_encoded_bytes":rare_bytes,
        "retained_checkpoint_bytes":retained,"attempt_high_water_bytes":high_water,
        "final_complete_keys":counts.len(),"final_live_keys":counts.values().filter(|c|**c>=floor).count(),
    }));
    Ok(())
}
