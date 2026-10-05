//! Diagnostic-only coarse task spans and complete-rank fast-path coverage.
//!
//! Disabled mode allocates no collector, records no clock, and clones no metadata.
//! Busy wall and thread CPU use each worker's interval union, so an inner fast
//! codec span is never added again to its enclosing preparation span. Capacity
//! gaps describe assigned task coverage, not OS/hypervisor preemption or true CPU
//! idleness. All clocks/locks are per coarse task or parallel phase, never per key.
use super::merge::JobDiagnostics;
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
    time::Instant,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Stage {
    Prep,
    FastCodec,
    CommitOwner,
    RouteBirthGroup,
}
impl Stage {
    fn name(self) -> &'static str {
        match self {
            Self::Prep => "prep",
            Self::FastCodec => "fast_codec",
            Self::CommitOwner => "commit_owner",
            Self::RouteBirthGroup => "route_birth_group",
        }
    }
}
#[derive(Clone, Serialize)]
struct TaskSpan {
    stage: Stage,
    job: usize,
    worker: usize,
    start: f64,
    end: f64,
    cpu_start: Option<f64>,
    cpu_end: Option<f64>,
    dispatch_to_start: Option<f64>,
}
#[derive(Clone, Copy, Serialize)]
struct Window {
    start: f64,
    end: f64,
}
#[derive(Default)]
struct RoundRecords {
    spans: Vec<TaskSpan>,
    windows: Vec<Window>,
    phases: BTreeMap<&'static str, f64>,
    publication_calls: u64,
    cold_published_keys: u64,
    cold_published_positions: u64,
    fast_publication_keys: u64,
    fast_publication_positions: u64,
}
struct RoundState {
    start: Instant,
    selected_inputs: Vec<usize>,
    workers: usize,
    records: Mutex<RoundRecords>,
}
#[derive(Default, Serialize)]
struct StageTotals {
    spans: u64,
    inclusive_wall_seconds: f64,
    inclusive_thread_cpu_seconds: f64,
    cpu_missing_spans: u64,
}
#[derive(Default, Serialize)]
struct Totals {
    rounds: u64,
    rounds_with_parallel_windows: u64,
    parallel_span_wall_seconds: f64,
    worker_capacity_wall_seconds: f64,
    worker_busy_wall_union_seconds: f64,
    worker_thread_cpu_union_seconds: f64,
    cpu_missing_union_segments: u64,
    task_assignment_capacity_gap_seconds: f64,
    sum_round_max_worker_busy_wall_seconds: f64,
    dispatch_samples: u64,
    dispatch_missing_spans: u64,
    dispatch_to_start_seconds_sum: f64,
    dispatch_to_start_seconds_max: f64,
    task_tail_lag_samples: u64,
    task_tail_lag_seconds_sum: f64,
    task_tail_lag_seconds_max: f64,
    phase_head_gap_seconds: f64,
    phase_join_gap_seconds: f64,
    spans_outside_parallel_windows: u64,
    invalid_worker_spans: u64,
    invalid_rank_tasks: u64,
    invalid_fast_rank_records: u64,
    selected_ranks: u64,
    selected_input_positions: u64,
    task_metadata_ranks: u64,
    ranks_without_task_metadata: u64,
    prep_jobs: u64,
    prep_task_ranges: u64,
    task_range_input_positions: u64,
    matched_positions: u64,
    single_actual_worker_ranks: u64,
    single_producer_ranks: u64,
    single_worker_multiple_producer_ranks: u64,
    full_single_range_ranks: u64,
    eligible_full_single_chunk_ranks: u64,
    fast_ranks: u64,
    fast_input_positions: u64,
    fast_matched_positions: u64,
    fast_born_records: u64,
    fast_published_keys: u64,
    fast_published_positions: u64,
    fast_pruned_keys: u64,
    publication_calls: u64,
    cold_published_keys: u64,
    cold_published_positions: u64,
    fast_publication_keys: u64,
    fast_publication_positions: u64,
    single_actual_worker_input_positions: u64,
    single_producer_input_positions: u64,
    eligible_input_positions: u64,
    phases_inclusive_wall_seconds: BTreeMap<&'static str, f64>,
    worker_mask_producer_joint_histogram: BTreeMap<u64, BTreeMap<usize, u64>>,
    eligible_zero_birth_ranks: u64,
    whole_prep_jobs_fast: u64,
    mixed_prep_jobs_fast: u64,
    fallback_multiple_producer_ranks: u64,
    fallback_partial_range_ranks: u64,
    fallback_multiple_chunk_ranks: u64,
    fallback_node_budget_ranks: u64,
    eligible_not_taken_ranks: u64,
    unexpected_fast_ineligible_ranks: u64,
    actual_worker_bitmask_histogram: BTreeMap<u64, u64>,
    distinct_job_producer_histogram: BTreeMap<usize, u64>,
    stages: BTreeMap<&'static str, StageTotals>,
}
#[derive(Default)]
struct RankInfo {
    jobs: BTreeSet<usize>,
    worker_mask: u64,
    ranges: usize,
    full_ranges: usize,
    input: usize,
    matched: usize,
    single_chunk: bool,
    node_budget_fits: bool,
    fast: bool,
    born_records: usize,
}
#[derive(Serialize)]
struct RankDetail {
    rank: usize,
    selected_input_positions: usize,
    actual_worker_bitmask: u64,
    distinct_job_producers: usize,
    task_ranges: usize,
    task_range_input_positions: usize,
    matched_positions: usize,
    full_single_range: bool,
    whole_job_single_chunk: bool,
    node_budget_fits: bool,
    eligible: bool,
    fast_taken: bool,
}
#[derive(Serialize)]
struct RoundTimeline {
    selected_inputs: Vec<usize>,
    parallel_windows: Vec<Window>,
    task_spans: Vec<TaskSpan>,
    ranks: Vec<RankDetail>,
}
struct AttemptState {
    workers: usize,
    detailed: bool,
    totals: Mutex<Totals>,
    timelines: Mutex<Vec<RoundTimeline>>,
    completed: Mutex<Option<bool>>,
}
pub(super) struct AttemptDiagnostics {
    state: Option<Arc<AttemptState>>,
}
impl AttemptDiagnostics {
    pub(super) fn new(enabled: bool, workers: usize) -> Self {
        Self {
            state: enabled.then(|| {
                Arc::new(AttemptState {
                    workers,
                    detailed: std::env::var("TK_SINGLE_DIAG_TIMELINE")
                        .is_ok_and(|value| value == "1"),
                    totals: Mutex::new(Totals::default()),
                    timelines: Mutex::new(Vec::new()),
                    completed: Mutex::new(None),
                })
            }),
        }
    }
    /// Set true only for the attempt that completes the model. An abandoned
    /// Fresh attempt before reusable restart must set false and stays separately
    /// labeled; no selected-rule denominator is silently combined across attempts.
    pub(super) fn finish(&self, completed: bool) {
        if let Some(state) = &self.state {
            *state
                .completed
                .lock()
                .expect("single-producer outcome lock poisoned") = Some(completed);
        }
    }
    /// `selected_inputs[rank]` is the selected candidate's physical position
    /// count, not its weighted frequency. Include AA/reuse rounds in this vector;
    /// absent task metadata remains explicitly unclassified in the denominator.
    pub(super) fn begin_round(&self, selected_inputs: &[usize]) -> RoundDiagnostics {
        let state = self.state.as_ref().map(|attempt| {
            Arc::new(RoundState {
                start: Instant::now(),
                selected_inputs: selected_inputs.to_vec(),
                workers: attempt.workers,
                records: Mutex::new(RoundRecords::default()),
            })
        });
        RoundDiagnostics {
            attempt: self.state.clone(),
            state,
        }
    }
}
impl Drop for AttemptDiagnostics {
    fn drop(&mut self) {
        let Some(state) = &self.state else { return };
        let totals = state
            .totals
            .lock()
            .expect("single-producer totals lock poisoned");
        let fraction = (totals.worker_capacity_wall_seconds > 0.0).then(|| {
            totals.task_assignment_capacity_gap_seconds / totals.worker_capacity_wall_seconds
        });
        let cpu_wall_ratio = (totals.worker_busy_wall_union_seconds > 0.0
            && totals.cpu_missing_union_segments == 0)
            .then(|| {
                totals.worker_thread_cpu_union_seconds / totals.worker_busy_wall_union_seconds
            });
        let timeline = state
            .timelines
            .lock()
            .expect("single-producer timeline lock poisoned");
        let completed = *state
            .completed
            .lock()
            .expect("single-producer outcome lock poisoned");
        let published_keys = totals.cold_published_keys + totals.fast_publication_keys;
        let published_positions =
            totals.cold_published_positions + totals.fast_publication_positions;
        eprintln!(
            "SINGLE_PRODUCER_DIAG {}",
            serde_json::json!({
                "diagnostic_only": true,
                "attempt_completed": completed,
                "final_model_diagnostic": completed == Some(true),
                "workers": state.workers,
                "all_published_keys": published_keys,
                "all_published_positions": published_positions,
                "fast_published_key_fraction": (published_keys > 0).then(|| totals.fast_publication_keys as f64 / published_keys as f64),
                "fast_published_position_fraction": (published_positions > 0).then(|| totals.fast_publication_positions as f64 / published_positions as f64),
                "fast_metadata_matches_owner_publication": totals.fast_published_keys == totals.fast_publication_keys && totals.fast_published_positions == totals.fast_publication_positions,
                "route_phase_label": "mixed serial dispatch plus parallel route birth grouping; not pure serial",
                "gap_meaning": "task assignment or synchronization coverage gap; not system preemption or proven CPU idle",
                "worker_mask_meaning": "actual preparation worker(s) from per-job task metadata",
                "dispatch_meaning": "caller-provided dispatch/batch-ready marker to task start; missing markers are not zero",
                "stage_totals_inclusive_not_additive": true,
                "capacity_gap_scope": "instrumented parallel windows; not complete training wall",
                "task_assignment_capacity_gap_fraction": fraction,
                "observed_task_thread_cpu_to_busy_wall_ratio": cpu_wall_ratio,
                "totals": &*totals,
                "timeline_enabled": state.detailed,
                "timeline": if state.detailed { Some(&*timeline) } else { None },
            })
        );
    }
}
pub(super) struct RoundDiagnostics {
    attempt: Option<Arc<AttemptState>>,
    state: Option<Arc<RoundState>>,
}
impl RoundDiagnostics {
    pub(super) fn mark(&self) -> Option<Instant> {
        self.state.as_ref().map(|_| Instant::now())
    }
    /// Mixed/inclusive coordinator phase (such as events.route), deliberately
    /// excluded from task busy-wall accounting unless actual task hooks exist.
    pub(super) fn phase(&self, name: &'static str) -> PhaseGuard {
        PhaseGuard {
            state: self.state.clone(),
            name,
            start: self.mark(),
        }
    }
    /// Owner calls this once after publication, including the no-cold early
    /// return. Do not also add per-key publications: the denominator is reduced
    /// exactly once here; preparation metadata stays separate for consistency.
    pub(super) fn record_publication(
        &self,
        cold_keys: usize,
        cold_positions: usize,
        fast_keys: usize,
        fast_positions: usize,
    ) {
        if let Some(state) = &self.state {
            let mut records = state
                .records
                .lock()
                .expect("single-producer round lock poisoned");
            records.publication_calls += 1;
            records.cold_published_keys += cold_keys as u64;
            records.cold_published_positions += cold_positions as u64;
            records.fast_publication_keys += fast_keys as u64;
            records.fast_publication_positions += fast_positions as u64;
        }
    }
    /// Wrap the entire parallel call, including dispatch and join. Multiple
    /// sequential calls can have separate windows; overlapping windows are united.
    pub(super) fn parallel_span(&self) -> ParallelSpan {
        ParallelSpan {
            state: self.state.clone(),
            start: self.mark(),
        }
    }
    /// Use only synchronous coarse scopes without nested pool work. FastCodec may
    /// be lexically inside Prep on the same worker; unions remove that overlap.
    pub(super) fn task(
        &self,
        stage: Stage,
        job: usize,
        worker: usize,
        dispatched: Option<Instant>,
    ) -> SpanGuard {
        let start = self.mark();
        SpanGuard {
            state: self.state.clone(),
            stage,
            job,
            worker,
            start,
            cpu_start: start.and_then(|_| thread_cpu()),
            dispatch_to_start: start.and_then(|start| {
                dispatched.and_then(|dispatch| {
                    start
                        .checked_duration_since(dispatch)
                        .map(|elapsed| elapsed.as_secs_f64())
                })
            }),
        }
    }
    /// Call once after all worker spans/windows have joined, before PreparedMerges
    /// is consumed by apply. This records no training mutation and takes no key lock.
    pub(super) fn finish(self, jobs: &[JobDiagnostics]) {
        let (Some(attempt), Some(state)) = (&self.attempt, &self.state) else {
            return;
        };
        let records = state
            .records
            .lock()
            .expect("single-producer round lock poisoned");
        let windows = union_windows(&records.windows);
        let mut totals = attempt
            .totals
            .lock()
            .expect("single-producer totals lock poisoned");
        totals.rounds += 1;
        totals.publication_calls += records.publication_calls;
        totals.cold_published_keys += records.cold_published_keys;
        totals.cold_published_positions += records.cold_published_positions;
        totals.fast_publication_keys += records.fast_publication_keys;
        totals.fast_publication_positions += records.fast_publication_positions;
        for (&name, &wall) in &records.phases {
            *totals
                .phases_inclusive_wall_seconds
                .entry(name)
                .or_default() += wall;
        }
        let wall: f64 = windows.iter().map(|window| window.end - window.start).sum();
        if !windows.is_empty() {
            totals.rounds_with_parallel_windows += 1;
        }
        totals.parallel_span_wall_seconds += wall;
        totals.worker_capacity_wall_seconds += state.workers as f64 * wall;
        let mut worker_busy = vec![0.0; state.workers];
        let mut round_busy = 0.0;
        for worker in 0..state.workers {
            let mut spans: Vec<_> = records
                .spans
                .iter()
                .filter(|span| span.worker == worker)
                .collect();
            spans.sort_by(|a, b| {
                a.start
                    .total_cmp(&b.start)
                    .then_with(|| b.end.total_cmp(&a.end))
            });
            let merged = union_worker_spans(&spans);
            for span in merged {
                let covered: f64 = windows
                    .iter()
                    .map(|window| {
                        (span.end.min(window.end) - span.start.max(window.start)).max(0.0)
                    })
                    .sum();
                worker_busy[worker] += covered;
                // CPU boundaries cannot be interpolated across clipped spans.
                // Proper hooks keep each task within a phase window; otherwise
                // mark CPU unavailable instead of estimating a fabricated value.
                let whole = windows
                    .iter()
                    .any(|window| span.start >= window.start && span.end <= window.end);
                if whole {
                    if let (Some(start), Some(end)) = (span.cpu_start, span.cpu_end) {
                        totals.worker_thread_cpu_union_seconds += (end - start).max(0.0);
                    } else {
                        totals.cpu_missing_union_segments += 1;
                    }
                } else if covered > 0.0 {
                    totals.cpu_missing_union_segments += 1;
                }
            }
            round_busy += worker_busy[worker];
        }
        totals.worker_busy_wall_union_seconds += round_busy;
        totals.task_assignment_capacity_gap_seconds +=
            (state.workers as f64 * wall - round_busy).max(0.0);
        totals.sum_round_max_worker_busy_wall_seconds +=
            worker_busy.into_iter().fold(0.0, f64::max);
        for span in &records.spans {
            let stage = totals.stages.entry(span.stage.name()).or_default();
            stage.spans += 1;
            stage.inclusive_wall_seconds += span.end - span.start;
            if let (Some(start), Some(end)) = (span.cpu_start, span.cpu_end) {
                stage.inclusive_thread_cpu_seconds += (end - start).max(0.0);
            } else {
                stage.cpu_missing_spans += 1;
            }
            if span.worker >= state.workers {
                totals.invalid_worker_spans += 1;
            }
            if !windows
                .iter()
                .any(|window| span.start >= window.start && span.end <= window.end)
            {
                totals.spans_outside_parallel_windows += 1;
            }
            if span.stage != Stage::FastCodec {
                if let Some(delay) = span.dispatch_to_start {
                    totals.dispatch_samples += 1;
                    totals.dispatch_to_start_seconds_sum += delay;
                    totals.dispatch_to_start_seconds_max =
                        totals.dispatch_to_start_seconds_max.max(delay);
                } else {
                    totals.dispatch_missing_spans += 1;
                }
            }
        }
        for window in &windows {
            let tasks: Vec<_> = records
                .spans
                .iter()
                .filter(|span| {
                    span.stage != Stage::FastCodec
                        && span.start >= window.start
                        && span.end <= window.end
                })
                .collect();
            if tasks.is_empty() {
                continue;
            }
            let first = tasks
                .iter()
                .map(|span| span.start)
                .fold(f64::INFINITY, f64::min);
            let last = tasks.iter().map(|span| span.end).fold(0.0, f64::max);
            totals.phase_head_gap_seconds += (first - window.start).max(0.0);
            totals.phase_join_gap_seconds += (window.end - last).max(0.0);
            for task in tasks {
                let tail = (last - task.end).max(0.0);
                totals.task_tail_lag_samples += 1;
                totals.task_tail_lag_seconds_sum += tail;
                totals.task_tail_lag_seconds_max = totals.task_tail_lag_seconds_max.max(tail);
            }
        }
        let ranks = accumulate_rank_metadata(&state.selected_inputs, jobs, &mut totals);
        drop(totals);
        if attempt.detailed {
            attempt
                .timelines
                .lock()
                .expect("single-producer timeline lock poisoned")
                .push(RoundTimeline {
                    selected_inputs: state.selected_inputs.clone(),
                    parallel_windows: records.windows.clone(),
                    task_spans: records.spans.clone(),
                    ranks,
                });
        }
    }
}
pub(super) struct PhaseGuard {
    state: Option<Arc<RoundState>>,
    name: &'static str,
    start: Option<Instant>,
}
impl Drop for PhaseGuard {
    fn drop(&mut self) {
        if let (Some(state), Some(start)) = (&self.state, self.start) {
            let wall = start.elapsed().as_secs_f64();
            *state
                .records
                .lock()
                .expect("single-producer round lock poisoned")
                .phases
                .entry(self.name)
                .or_default() += wall;
        }
    }
}
pub(super) struct ParallelSpan {
    state: Option<Arc<RoundState>>,
    start: Option<Instant>,
}
impl Drop for ParallelSpan {
    fn drop(&mut self) {
        if let (Some(state), Some(start)) = (&self.state, self.start) {
            let end = Instant::now();
            state
                .records
                .lock()
                .expect("single-producer round lock poisoned")
                .windows
                .push(Window {
                    start: start.duration_since(state.start).as_secs_f64(),
                    end: end.duration_since(state.start).as_secs_f64(),
                });
        }
    }
}
pub(super) struct SpanGuard {
    state: Option<Arc<RoundState>>,
    stage: Stage,
    job: usize,
    worker: usize,
    start: Option<Instant>,
    cpu_start: Option<f64>,
    dispatch_to_start: Option<f64>,
}
impl Drop for SpanGuard {
    fn drop(&mut self) {
        if let (Some(state), Some(start)) = (&self.state, self.start) {
            let cpu_end = thread_cpu();
            let end = Instant::now();
            state
                .records
                .lock()
                .expect("single-producer round lock poisoned")
                .spans
                .push(TaskSpan {
                    stage: self.stage,
                    job: self.job,
                    worker: self.worker,
                    start: start.duration_since(state.start).as_secs_f64(),
                    end: end.duration_since(state.start).as_secs_f64(),
                    cpu_start: self.cpu_start,
                    cpu_end,
                    dispatch_to_start: self.dispatch_to_start,
                });
        }
    }
}
fn union_windows(windows: &[Window]) -> Vec<Window> {
    let mut ordered = windows.to_vec();
    ordered.sort_by(|a, b| a.start.total_cmp(&b.start));
    let mut result: Vec<Window> = Vec::new();
    for window in ordered {
        if let Some(last) = result.last_mut() {
            if window.start <= last.end {
                last.end = last.end.max(window.end);
                continue;
            }
        }
        result.push(window);
    }
    result
}
struct WorkerUnion {
    start: f64,
    end: f64,
    cpu_start: Option<f64>,
    cpu_end: Option<f64>,
}
fn union_worker_spans(spans: &[&TaskSpan]) -> Vec<WorkerUnion> {
    let mut result: Vec<WorkerUnion> = Vec::new();
    for span in spans {
        if let Some(last) = result.last_mut() {
            if span.start <= last.end {
                if span.end > last.end {
                    last.end = span.end;
                    last.cpu_end = span.cpu_end;
                }
                continue;
            }
        }
        result.push(WorkerUnion {
            start: span.start,
            end: span.end,
            cpu_start: span.cpu_start,
            cpu_end: span.cpu_end,
        });
    }
    result
}
fn accumulate_rank_metadata(
    inputs: &[usize],
    jobs: &[JobDiagnostics],
    totals: &mut Totals,
) -> Vec<RankDetail> {
    let mut ranks: Vec<RankInfo> = (0..inputs.len()).map(|_| RankInfo::default()).collect();
    totals.selected_ranks += inputs.len() as u64;
    totals.selected_input_positions += inputs.iter().map(|&count| count as u64).sum::<u64>();
    totals.prep_jobs += jobs.len() as u64;
    for (job_index, job) in jobs.iter().enumerate() {
        let fast_set: BTreeSet<_> = job.fast_ranks.iter().map(|fast| fast.rank).collect();
        let task_set: BTreeSet<_> = job.tasks.iter().map(|task| task.rank).collect();
        if !fast_set.is_empty() {
            if task_set.iter().all(|rank| fast_set.contains(rank)) {
                totals.whole_prep_jobs_fast += 1;
            } else {
                totals.mixed_prep_jobs_fast += 1;
            }
        }
        for task in &job.tasks {
            let Some(rank) = ranks.get_mut(task.rank) else {
                totals.invalid_rank_tasks += 1;
                continue;
            };
            rank.jobs.insert(job_index);
            if job.worker < 64 {
                rank.worker_mask |= 1u64 << job.worker;
            }
            rank.ranges += 1;
            rank.full_ranges += usize::from(task.full);
            rank.input += task.end.saturating_sub(task.begin);
            rank.matched += task.matched_positions;
            rank.single_chunk = job.chunks == 1;
            rank.node_budget_fits = job.node_budget_fits;
            totals.prep_task_ranges += 1;
            totals.task_range_input_positions += task.end.saturating_sub(task.begin) as u64;
            totals.matched_positions += task.matched_positions as u64;
            if fast_set.contains(&task.rank) {
                totals.fast_matched_positions += task.matched_positions as u64;
            }
        }
        for fast in &job.fast_ranks {
            let Some(rank) = ranks.get_mut(fast.rank) else {
                totals.invalid_fast_rank_records += 1;
                continue;
            };
            rank.fast = true;
            rank.born_records += fast.born_records;
            totals.fast_input_positions += fast.input_positions as u64;
            totals.fast_born_records += fast.born_records as u64;
            totals.fast_published_keys += fast.encoded_keys as u64;
            totals.fast_published_positions += fast.encoded_positions as u64;
            totals.fast_pruned_keys += fast.pruned_keys as u64;
        }
    }
    let mut detail = Vec::with_capacity(inputs.len());
    for (rank_index, rank) in ranks.into_iter().enumerate() {
        let producers = rank.jobs.len();
        let full = producers == 1
            && rank.ranges == 1
            && rank.full_ranges == 1
            && rank.input == inputs[rank_index];
        let eligible = full && rank.single_chunk && rank.node_budget_fits;
        *totals
            .actual_worker_bitmask_histogram
            .entry(rank.worker_mask)
            .or_default() += 1;
        *totals
            .distinct_job_producer_histogram
            .entry(producers)
            .or_default() += 1;
        *totals
            .worker_mask_producer_joint_histogram
            .entry(rank.worker_mask)
            .or_default()
            .entry(producers)
            .or_default() += 1;
        if producers == 0 {
            totals.ranks_without_task_metadata += 1;
        } else {
            totals.task_metadata_ranks += 1;
            if rank.worker_mask.count_ones() == 1 {
                totals.single_actual_worker_ranks += 1;
                totals.single_actual_worker_input_positions += inputs[rank_index] as u64;
                if producers > 1 {
                    totals.single_worker_multiple_producer_ranks += 1;
                }
            }
            if producers == 1 {
                totals.single_producer_ranks += 1;
                totals.single_producer_input_positions += inputs[rank_index] as u64;
            }
            if full {
                totals.full_single_range_ranks += 1;
            }
            if eligible {
                totals.eligible_full_single_chunk_ranks += 1;
                totals.eligible_input_positions += inputs[rank_index] as u64;
            }
            if producers > 1 {
                totals.fallback_multiple_producer_ranks += 1;
            } else if !full {
                totals.fallback_partial_range_ranks += 1;
            } else if !rank.single_chunk {
                totals.fallback_multiple_chunk_ranks += 1;
            } else if !rank.node_budget_fits {
                totals.fallback_node_budget_ranks += 1;
            } else if !rank.fast {
                totals.eligible_not_taken_ranks += 1;
            }
        }
        if rank.fast {
            totals.fast_ranks += 1;
            if !eligible {
                totals.unexpected_fast_ineligible_ranks += 1;
            }
            if eligible && rank.born_records == 0 {
                totals.eligible_zero_birth_ranks += 1;
            }
        }
        detail.push(RankDetail {
            rank: rank_index,
            selected_input_positions: inputs[rank_index],
            actual_worker_bitmask: rank.worker_mask,
            distinct_job_producers: producers,
            task_ranges: rank.ranges,
            task_range_input_positions: rank.input,
            matched_positions: rank.matched,
            full_single_range: full,
            whole_job_single_chunk: rank.single_chunk,
            node_budget_fits: rank.node_budget_fits,
            eligible,
            fast_taken: rank.fast,
        });
    }
    detail
}
#[cfg(target_os = "linux")]
fn thread_cpu() -> Option<f64> {
    use std::ffi::{c_int, c_long};
    #[repr(C)]
    struct Timespec {
        seconds: c_long,
        nanoseconds: c_long,
    }
    unsafe extern "C" {
        fn clock_gettime(clock: c_int, result: *mut Timespec) -> c_int;
    }
    let mut value = Timespec {
        seconds: 0,
        nanoseconds: 0,
    };
    // SAFETY: On Linux clock ID 3 is CLOCK_THREAD_CPUTIME_ID. The C-compatible,
    // aligned Timespec uses the host C long ABI and is writable for the full call.
    // This FFI runs only in enabled diagnostic coarse scopes, never production
    // timing. Synchronous scopes stay on their executing Rayon worker thread.
    let status = unsafe { clock_gettime(3, &mut value) };
    (status == 0).then_some(value.seconds as f64 + value.nanoseconds as f64 * 1e-9)
}
#[cfg(not(target_os = "linux"))]
fn thread_cpu() -> Option<f64> {
    None
}

#[cfg(test)]
mod tests {
    use super::super::merge::{FastRankDiagnostic, TaskDiagnostic};
    use super::*;
    #[test]
    fn nested_fast_codec_is_not_double_counted_in_worker_cpu_or_wall() {
        let outer = TaskSpan {
            stage: Stage::Prep,
            job: 0,
            worker: 0,
            start: 1.0,
            end: 5.0,
            cpu_start: Some(10.0),
            cpu_end: Some(13.0),
            dispatch_to_start: None,
        };
        let inner = TaskSpan {
            stage: Stage::FastCodec,
            job: 0,
            worker: 0,
            start: 2.0,
            end: 4.0,
            cpu_start: Some(10.5),
            cpu_end: Some(12.0),
            dispatch_to_start: None,
        };
        let merged = union_worker_spans(&[&outer, &inner]);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].end - merged[0].start, 4.0);
        assert_eq!(
            merged[0].cpu_end.unwrap() - merged[0].cpu_start.unwrap(),
            3.0
        );
    }
    #[test]
    fn parallel_window_union_preserves_sequential_phase_gap() {
        let merged = union_windows(&[
            Window {
                start: 1.0,
                end: 3.0,
            },
            Window {
                start: 2.0,
                end: 4.0,
            },
            Window {
                start: 6.0,
                end: 7.0,
            },
        ]);
        assert_eq!(merged.len(), 2);
        assert_eq!(
            merged
                .iter()
                .map(|window| window.end - window.start)
                .sum::<f64>(),
            4.0
        );
    }
    #[test]
    fn same_worker_two_jobs_are_two_producers_and_cannot_be_fast_eligible() {
        let jobs = vec![
            JobDiagnostics {
                worker: 0,
                tasks: vec![TaskDiagnostic {
                    rank: 0,
                    begin: 0,
                    end: 5,
                    full: false,
                    matched_positions: 4,
                }],
                chunks: 1,
                node_budget_fits: true,
                fast_ranks: Vec::<FastRankDiagnostic>::new(),
            },
            JobDiagnostics {
                worker: 0,
                tasks: vec![TaskDiagnostic {
                    rank: 0,
                    begin: 5,
                    end: 10,
                    full: false,
                    matched_positions: 3,
                }],
                chunks: 1,
                node_budget_fits: true,
                fast_ranks: Vec::<FastRankDiagnostic>::new(),
            },
        ];
        let mut totals = Totals::default();
        let detail = accumulate_rank_metadata(&[10], &jobs, &mut totals);
        assert_eq!(totals.single_actual_worker_ranks, 1);
        assert_eq!(totals.single_worker_multiple_producer_ranks, 1);
        assert_eq!(totals.single_producer_ranks, 0);
        assert_eq!(totals.fallback_multiple_producer_ranks, 1);
        assert_eq!(totals.matched_positions, 7);
        assert!(!detail[0].eligible);
    }

    #[test]
    fn actual_single_chunk_does_not_override_allocation_budget_fallback() {
        let jobs = [JobDiagnostics {
            worker: 1,
            tasks: vec![TaskDiagnostic {
                rank: 0,
                begin: 0,
                end: 8,
                full: true,
                matched_positions: 1,
            }],
            chunks: 1,
            node_budget_fits: false,
            fast_ranks: Vec::new(),
        }];
        let mut totals = Totals::default();
        let detail = accumulate_rank_metadata(&[8], &jobs, &mut totals);
        assert_eq!(totals.single_producer_ranks, 1);
        assert_eq!(totals.single_actual_worker_ranks, 1);
        assert_eq!(totals.full_single_range_ranks, 1);
        assert_eq!(totals.fallback_node_budget_ranks, 1);
        assert_eq!(totals.eligible_full_single_chunk_ranks, 0);
        assert!(!detail[0].eligible);
    }
}
