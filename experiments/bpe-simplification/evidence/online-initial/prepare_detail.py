from pathlib import Path
OUT=Path('/root/code/tokenizers-simplification-results/online-initial')
REL=Path('tokenizers/tk-train/src/trainers/bpe/engine')
MODULE=r'''//! Host-only preparation attribution. Worker durations overlap; thread CPU is additive.
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
const NAMES: [&str; 12] = ["selected_index", "aa_prepass", "task_partition", "ordinary_joined", "aa_joined", "reuse_joined", "worker_total", "worker_setup", "scan_collect", "finish_encode", "postjoin_gather", "candidate_release"];
const COUNTERS: [&str; 9] = ["rules", "candidate_positions", "ordinary_jobs", "ordinary_tasks", "complete_tasks", "partial_tasks", "scan_positions", "matched_positions", "neighbor_groups"];
static WALL: [AtomicU64; 12] = [const { AtomicU64::new(0) }; 12];
static CPU: [AtomicU64; 12] = [const { AtomicU64::new(0) }; 12];
static CALLS: [AtomicU64; 12] = [const { AtomicU64::new(0) }; 12];
static COUNTS: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];
#[repr(C)]
struct Timespec { sec: std::os::raw::c_long, nano: std::os::raw::c_long }
unsafe extern "C" { fn clock_gettime(clock: i32, time: *mut Timespec) -> i32; }
fn cpu_ns(clock: i32) -> u64 {
    let mut time = Timespec { sec: 0, nano: 0 };
    assert_eq!(unsafe { clock_gettime(clock, &mut time) }, 0);
    time.sec as u64 * 1_000_000_000 + time.nano as u64
}
pub(super) struct Stage { index: usize, wall: Instant, cpu: u64, clock: i32 }
pub(super) fn stage(index: usize) -> Stage {
    let clock = if (6..=9).contains(&index) { 3 } else { 2 }; // Linux THREAD / PROCESS CPU
    Stage { index, wall: Instant::now(), cpu: cpu_ns(clock), clock }
}
impl Drop for Stage {
    fn drop(&mut self) {
        let cpu = cpu_ns(self.clock) - self.cpu;
        WALL[self.index].fetch_add(self.wall.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CPU[self.index].fetch_add(cpu, Ordering::Relaxed);
        CALLS[self.index].fetch_add(1, Ordering::Relaxed);
    }
}
pub(super) fn count(index: usize, value: usize) { COUNTS[index].fetch_add(value as u64, Ordering::Relaxed); }
pub(super) struct Report;
impl Drop for Report {
    fn drop(&mut self) {
        let rows: Vec<_> = NAMES.iter().enumerate().map(|(i,name)| format!(
            r#"{{"name":"{}","wall_seconds":{},"cpu_seconds":{},"calls":{},"cpu_clock":"{}"}}"#,
            name,WALL[i].load(Ordering::Relaxed) as f64 / 1e9,CPU[i].load(Ordering::Relaxed) as f64 / 1e9,CALLS[i].load(Ordering::Relaxed),if (6..=9).contains(&i) { "thread" } else { "process" }
        )).collect();
        let counters: Vec<_> = COUNTERS.iter().enumerate().map(|(i,name)| format!(r#""{}":{}"#,name,COUNTS[i].load(Ordering::Relaxed))).collect();
        eprintln!("BPE_PREPARE_DETAIL {{\"stages\":[{}],\"counters\":{{{}}}}}",rows.join(","),counters.join(","));
    }
}
'''

def change(work,path,old,new,count=1):
    p=work/REL/path;s=p.read_text();assert s.count(old)==count,(p,old[:90],s.count(old),count);p.write_text(s.replace(old,new))

def common(work):
    (work/REL/'prepare_detail.rs').write_text(MODULE)
    change(work,'mod.rs','mod phase_timing;','mod phase_timing;\nmod prepare_detail;')
    change(work,'mod.rs','let _phase_report = phase_timing::Report;','let _phase_report = phase_timing::Report;\n    let _prepare_report = prepare_detail::Report;')

def main():
    w=Path('/root/code/tokenizers-workspaces/bpe-prepare-detail-main-20261010');common(w)
    change(w,'merge/prepare/mod.rs','use ahash::AHashMap;','use ahash::AHashMap;\nuse super::super::prepare_detail;')
    change(w,'merge/prepare/mod.rs','    let floor = floor.max(1);','    let _mode = prepare_detail::stage(if policy == IdentityPolicy::AllowActiveReuse { 5 } else if rules[0].pair.0 == rules[0].pair.1 { 4 } else { 3 });\n    prepare_detail::count(0,rules.len());\n    prepare_detail::count(1,candidates.iter().map(|c|c.positions.len()).sum());\n    let floor = floor.max(1);')
    change(w,'merge/prepare/mod.rs','    let mut jobs = Vec::new();\n    let mut chunks = Vec::new();','    let _gather = prepare_detail::stage(10);\n    let mut jobs = Vec::new();\n    let mut chunks = Vec::new();')
    change(w,'merge/prepare/mod.rs','        Self {\n            corpus,\n            scratch,','        let _setup = prepare_detail::stage(7);\n        Self {\n            corpus,\n            scratch,')
    change(w,'merge/prepare/mod.rs','        debug_assert!(\n            self.chunks.is_empty(),','        let _finish = prepare_detail::stage(9);\n        prepare_detail::count(8,self.scratch.left.touched_len()+self.scratch.right.touched_len());\n        debug_assert!(\n            self.chunks.is_empty(),')
    change(w,'merge/prepare/mod.rs','    fn finish(self) -> (WritePlan, Vec<EventChunk>) {\n        self.scratch.flush_rule','    fn finish(self) -> (WritePlan, Vec<EventChunk>) {\n        let _finish = prepare_detail::stage(9);\n        prepare_detail::count(8,self.scratch.left.touched_len()+self.scratch.right.touched_len());\n        self.scratch.flush_rule')
    p='merge/prepare/ordinary.rs'
    change(w,p,'    let jobs = position_jobs(','    let _partition = prepare_detail::stage(2);\n    let jobs = position_jobs(')
    change(w,p,'    let mut selected = execution.selected_rules();','    prepare_detail::count(2,jobs.len());\n    prepare_detail::count(3,jobs.iter().map(Vec::len).sum());\n    for task in jobs.iter().flatten() { prepare_detail::count(if task.whole_rule {4}else{5},1); }\n    drop(_partition);\n    let _selected = prepare_detail::stage(0);\n    let mut selected = execution.selected_rules();')
    change(w,p,'    selected.reset(rules, token_id_count);','    selected.reset(rules, token_id_count);\n    drop(_selected);')
    change(w,p,'        .map(|tasks| -> Result<_> {\n            execution.with_merge_scratch','        .map(|tasks| -> Result<_> {\n            let _worker = prepare_detail::stage(6);\n            execution.with_merge_scratch')
    change(w,p,'    let mut cursor = positions.cursor(task.begin..task.end);','    let _scan = prepare_detail::stage(8);\n    let mut visited=0;\n    let mut matched_count=0;\n    let mut cursor = positions.cursor(task.begin..task.end);')
    change(w,p,'        let position = ring[head];','        visited+=1;\n        let position = ring[head];')
    change(w,p,'        if let Some(matched) = plan.matcher.get(position) {','        if let Some(matched) = plan.matcher.get(position) {\n            matched_count+=1;')
    change(w,p,'    Ok(())\n}','    prepare_detail::count(6,visited);\n    prepare_detail::count(7,matched_count);\n    Ok(())\n}')
    p='merge/prepare/aa.rs'
    change(w,p,'    let chunk = candidate','    let _prepass = prepare_detail::stage(1);\n    let chunk = candidate')
    change(w,p,'    chosen\n        .into_par_iter()','    drop(_prepass);\n    chosen\n        .into_par_iter()')
    change(w,p,'        .map(|(index, positions)| -> Result<_> {','        .map(|(index, positions)| -> Result<_> {\n            let _worker = prepare_detail::stage(6);')
    change(w,p,'                for (offset, position) in positions.positions().enumerate() {','                let _scan = prepare_detail::stage(8);\n                prepare_detail::count(6,positions.len());\n                prepare_detail::count(7,positions.len());\n                for (offset, position) in positions.positions().enumerate() {')
    change(w,p,'                plan.positions = positions;','                drop(_scan);\n                plan.positions = positions;')
    change(w,'mod.rs','{ let _phase = phase_timing::stage(5); batch.candidates.clear() }','{ let _phase = phase_timing::stage(5); let _release = prepare_detail::stage(11); batch.candidates.clear() }')

def candidate():
    w=Path('/root/code/tokenizers-workspaces/bpe-prepare-detail-candidate-20261010');common(w)
    p='merge.rs';change(w,p,'use ahash::{AHashMap, AHashSet};','use ahash::{AHashMap, AHashSet};\nuse super::prepare_detail;')
    change(w,p,'        if self.reuse {\n            return self.prepare_cohort','        let _mode=prepare_detail::stage(if self.reuse {5}else if self.rules[0].pair.0==self.rules[0].pair.1 {4}else{3});\n        prepare_detail::count(0,self.rules.len());\n        prepare_detail::count(1,self.rules.iter().map(|r|r.candidate.positions.len()).sum());\n        if self.reuse {\n            return self.prepare_cohort')
    change(w,p,'        let selected: AHashMap<_, _> = self','        let _selected=prepare_detail::stage(0);\n        let selected: AHashMap<_, _> = self')
    change(w,p,'        let aa = self.rules[0].pair.0 == self.rules[0].pair.1;','        drop(_selected);\n        let aa = self.rules[0].pair.0 == self.rules[0].pair.1;\n        let _prepass=prepare_detail::stage(1);')
    change(w,p,'        let mut tasks = Vec::new();','        drop(_prepass);\n        let _partition=prepare_detail::stage(2);\n        let mut tasks = Vec::new();')
    change(w,p,'        let jobs = tasks\n            .into_par_iter()','        if !aa {\n            prepare_detail::count(2,tasks.len());\n            prepare_detail::count(3,tasks.len());\n            for (_,_,p) in &tasks { prepare_detail::count(if p.complete() {4}else{5},1); }\n        }\n        drop(_partition);\n        let jobs = tasks\n            .into_par_iter()')
    change(w,p,'                    directories.reset(corpus.id_count());','                    let _worker=prepare_detail::stage(6);\n                    let _setup=prepare_detail::stage(7);\n                    directories.reset(corpus.id_count());')
    change(w,p,'                    let mut weights = None;','                    let mut weights = None;\n                    drop(_setup);\n                    let _scan=prepare_detail::stage(8);\n                    let mut visited=0;\n                    let mut matched_count=0;')
    change(w,p,'                    for p in positions.prefetched(corpus) {','                    for p in positions.prefetched(corpus) {\n                        visited+=1;')
    change(w,p,'                        let (weight, end) = match weights {','                        matched_count+=1;\n                        let (weight, end) = match weights {')
    change(w,p,'                    Ok(Job {\n                        writes,','                    prepare_detail::count(6,visited);\n                    prepare_detail::count(7,matched_count);\n                    drop(_scan);\n                    Ok(Job {\n                        writes,')
    change(w,p,'        // Complete ordinary producers prune and encode before owner routing.','        let _finish=prepare_detail::stage(9);\n        prepare_detail::count(8,self.changes.iter().map(Vec::len).sum());\n        // Complete ordinary producers prune and encode before owner routing.')
    # Consume the same Batch immediately before returning, exposing the existing automatic release.
    change(w,p,'        Ok(Prepared { jobs })\n    }\n    fn prepare_cohort','        { let _release=prepare_detail::stage(11); drop(self); }\n        Ok(Prepared { jobs })\n    }\n    fn prepare_cohort')
if __name__=='__main__':
    import shutil
    for name,source in [('main','bpe-initial-main-20261010'),('candidate','bpe-online-full-phase-20261010')]:
        work=Path('/root/code/tokenizers-workspaces')/('bpe-prepare-detail-'+name+'-20261010')
        origin=Path('/root/code/tokenizers-workspaces')/source
        for p in (origin/REL).rglob('*'):
            if p.is_file(): shutil.copy2(p,work/REL/p.relative_to(origin/REL))
    main();candidate();print('Installed preparation attribution in isolated worktrees')
