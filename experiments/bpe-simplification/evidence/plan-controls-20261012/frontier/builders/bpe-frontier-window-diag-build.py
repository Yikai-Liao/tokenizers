from pathlib import Path
import ast, shutil, subprocess, os, json, hashlib, sys

root=Path('/root/code/tokenizers-workspaces/bpe-frontier-window-results-20261012')
work=Path('/tmp/bpe-selected-rules-measurement-20261011')
rel=Path('tokenizers/tk-train/src/trainers/bpe')
target=Path('/root/code/tokenizers-build/bpe-affix-core-build-20261011')
env=os.environ|{'PATH':'/root/.cargo/bin:'+os.environ['PATH'],'CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'2'}
prior=ast.parse(Path('/tmp/bpe-complete-publisher-diag-build.py').read_text())
strings={node.targets[0].id:ast.literal_eval(node.value) for node in prior.body if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str)}
directory=strings['directory'].replace('use std::sync::atomic::{AtomicU64, Ordering::Relaxed};','').replace('producer reducer','final replay').replace('{ "producer" }','{ "replay" }')
cpu=strings['producer_counters'];cpu=cpu[cpu.index('// Linux diagnostic'):cpu.index('pub(super) fn bench_report_producer')]
index_counters=strings['index_counters']+'''
static BENCH_COORDINATES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BENCH_ENCODED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BENCH_CAPACITY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BENCH_REJECTED_KEYS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BENCH_ADMITTED_KEYS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BENCH_HEAP_PUSHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(super) fn bench_report_encoding() {
    use std::sync::atomic::Ordering::Relaxed;
    eprintln!("BPE_ENCODING_DIAGNOSTIC {}",serde_json::json!({
        "birth_fragment_coordinates":BENCH_COORDINATES.load(Relaxed),
        "birth_fragment_encoded_bytes":BENCH_ENCODED.load(Relaxed),
        "sum_birth_fragment_heap_capacity_bytes":BENCH_CAPACITY.load(Relaxed),
        "complete_rejected_keys":BENCH_REJECTED_KEYS.load(Relaxed),
        "complete_admitted_keys":BENCH_ADMITTED_KEYS.load(Relaxed),
        "merge_heap_pushes":BENCH_HEAP_PUSHES.load(Relaxed),
    }));
}
'''

def replace(text,needle,new,count=1):
    assert text.count(needle)==count,(needle,text.count(needle),count)
    return text.replace(needle,new)

prepare_only='--prepare-only' in sys.argv
for arm in ([x for x in sys.argv[1:] if x!='--prepare-only'] or ['eager','single','frontier']):
    source=root/'source'/arm;snapshot=root/'source'/f'{arm}-diagnostic'
    assert not snapshot.exists(),snapshot
    shutil.copytree(source/rel,work/rel,dirs_exist_ok=True)
    runner=work/'experiments/bpe-simplification/runner'
    shutil.copytree(source/'experiments/bpe-simplification/runner',runner,dirs_exist_ok=True)
    manifest=work/'tokenizers/tk-train/Cargo.toml'
    shutil.copyfile(source/'tokenizers/tk-train/Cargo.toml',manifest)
    manifest.write_text(replace(manifest.read_text(),'[dependencies]','[dependencies]\nthread_local = "1.1.10"'))
    merge=work/rel/'merge.rs';text=merge.read_text()
    text=replace(text,'#[derive(Default)]\nstruct Directories {','struct Directories {\n    bench_site: usize,')
    old='''    fn resize(&mut self, domain: usize) {
        for values in &mut self.indices {
            values.resize(domain, u32::MAX);
        }
    }'''
    # Reuse precisely the previously verified directory capacity accounting.
    assignments=[n for n in ast.walk(prior) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='new']
    new=ast.literal_eval(assignments[0].value)
    text=replace(text,old,new)
    text=replace(text,'.map_init(Directories::default, |directories, job| {','.map_init(|| Directories::bench_new(1), |directories, job| {')
    text=replace(text,'let mut positions = coordinates.map(move |coordinate| corpus.resident(coordinate));','let mut positions = coordinates.map(move |coordinate| { if matches!(self, Source::Task(_)) { BENCH_DECODED.fetch_add(1, Relaxed); } corpus.resident(coordinate) });')
    text=replace(text,'            for coordinate in rule.candidate.positions.iter() {','            for coordinate in rule.candidate.positions.iter() {\n                BENCH_DECODED.fetch_add(1, Relaxed);')
    text=replace(text,'            let replacement = match &self.matches {','            BENCH_RIGHT_PREDICTIONS.fetch_add(1,Relaxed);\n            let replacement = match &self.matches {')
    needle='        BENCH_PREPARE_NS.fetch_add(bench_prepare_started.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);'
    text=replace(text,needle,needle+'\n        bench_jobs(&jobs);',2)
    text=replace(text,'        let boundary = u32::try_from(vocabulary.len())','        let mut bench_reason = "cohort_cap";\n        let mut bench_coherent_records = Vec::new();\n        let boundary = u32::try_from(vocabulary.len())')
    text=replace(text,'            if retained > *high_water {\n                break;','            if retained > *high_water {\n                bench_reason = "memory";\n                break;')
    text=replace(text,'            if cap == 0 {\n                break;','            if cap == 0 {\n                bench_reason = "rule_cap";\n                break;')
    # Only the second best() loop is in the window kernel.
    split=text.index('    fn merge_window(');head=text[:split];tail=text[split:]
    tail=replace(tail,'                let Some(priority) = index.best() else {\n                    break;','                let Some(priority) = index.best() else {\n                    bench_reason = "empty_old_heap";\n                    break;')
    tail=replace(tail,'                    drain = true;\n                    break;','                    bench_reason = if pair.0 >= boundary || pair.1 >= boundary { "new_input" } else { "guard" };\n                    drain = true;\n                    break;',2)
    # Reserved branch has its own cause, after the combined guard branch.
    tail=tail.replace('                    bench_reason = if pair.0 >= boundary || pair.1 >= boundary { "new_input" } else { "guard" };\n                    drain = true;\n                    break;\n                }\n                debug_assert_eq!', '                    bench_reason = "identity";\n                    drain = true;\n                    break;\n                }\n                debug_assert_eq!')
    # Collect coherent M inside prepare_fresh before Batch and snapshot retire.
    text=head+tail
    needle='        Ok(jobs)\n    }\n\n    fn prepare_cohort'
    text=replace(text,needle,'        if let Some(boundary) = boundary {\n            let values = bench_coherent(&snapshot, &jobs, boundary)?;\n            BENCH_COHERENT.lock().unwrap().extend(values);\n        }\n'+needle)
    split=text.index('    fn merge_window(')
    text=text[:split]+replace(text[split:],'        bench_jobs(&jobs);\n        let bench_apply_started','        bench_jobs(&jobs);\n        bench_coherent_records.extend(std::mem::take(&mut *BENCH_COHERENT.lock().unwrap()));\n        let bench_apply_started')
    # Final replay task order matches the flattened Job order.
    text=replace(text,'        let births = jobs\n','        if vocabulary.len() >= self.vocab_size { bench_reason = "target"; }\n        else if records.len() == 256 { bench_reason = "rule_cap"; }\n        let bench_blocked = index.best().map(|p|(p.count,p.pair()));\n        let bench_job_producers: Vec<_> = jobs.iter().map(|j|j.replacement).collect();\n        let births = jobs\n')
    text=replace(text,'        let bench_final_commit_started = std::time::Instant::now();','        bench_window_report(&births,&bench_job_producers,&bench_coherent_records,boundary,&output_cohorts,guard,bench_blocked,bench_reason,*high_water,retained,self.min_frequency.max(1))?;\n        let bench_final_commit_started = std::time::Instant::now();')
    text=replace(text,'                group.positions.push((p - prior_span) as u64)?;','                if prior >= boundary && cohort != cohorts[(prior-boundary) as usize] { BENCH_CROSS_CUTS.fetch_add(1,Relaxed); }\n                group.positions.push((p - prior_span) as u64)?;')
    text=replace(text,'            group.positions.push(p as u64)?;','            if next >= boundary && cohort != cohorts[(next-boundary) as usize] { BENCH_CROSS_CUTS.fetch_add(1,Relaxed); }\n            group.positions.push(p as u64)?;')
    text+='\nstatic BENCH_COHERENT: std::sync::Mutex<Vec<(u32,u64,u64)>> = std::sync::Mutex::new(Vec::new());\n'+directory+cpu+Path('/tmp/bpe-frontier-window-diag.rs').read_text()
    # The helper imports Relaxed once; directory uses that shared binding.
    merge.write_text(text)
    positions=work/rel/'positions.rs';text=positions.read_text()
    text=replace(text,'    pub(super) fn push(&mut self, position: u64) -> Result<()> {','    pub(super) fn push(&mut self, position: u64) -> Result<()> {\n        if super::merge::BENCH_MERGING.load(std::sync::atomic::Ordering::Relaxed) { super::merge::BENCH_APPEND.fetch_add(1,std::sync::atomic::Ordering::Relaxed); }')
    needle='    pub(super) fn iter(&self)'
    assert text.count(needle)==2
    text=text.replace(needle,'''    pub(super) fn bench_storage(&self) -> (u64, u64) {
        let encoded = self.leaves().iter().map(|v| v.bytes.len() as u64).sum();
        let heap = self.leaves().iter().filter(|v| v.bytes.spilled()).map(|v| v.bytes.capacity() as u64).sum::<u64>()
            + match &self.0 { Repr::Root { leaves, .. } => leaves.len() as u64 * std::mem::size_of::<Leaf>() as u64, _ => 0 };
        (encoded, heap)
    }

'''+needle,1)
    positions.write_text(text)
    corpus=work/rel/'corpus.rs';text=corpus.read_text()
    for signature,counter in [('    pub(super) fn token(&self, position: usize) -> u32 {','BENCH_TOKEN_READS'),('    pub(super) fn weight_region(&self, position: usize) -> (u64, usize) {','BENCH_WEIGHT_LOOKUPS')]:
        text=replace(text,signature,signature+f'\n        if super::merge::BENCH_MERGING.load(std::sync::atomic::Ordering::Relaxed) {{ super::merge::{counter}.fetch_add(1,std::sync::atomic::Ordering::Relaxed); }}')
    corpus.write_text(text)
    index=work/rel/'index.rs';text=index.read_text()
    text=replace(text,'        for change in changes.into_iter().flatten() {','        let (mut bench_l,mut bench_h,mut bench_e)=(0u64,0u64,0u64);\n        for change in changes.into_iter().flatten() {')
    text=replace(text,'            let index = events.len();','            bench_l += u64::from(removed.is_some()); bench_h += u64::from(born.is_some()); bench_e += u64::from(removed.is_some() && removed == born);\n            BENCH_COORDINATES.fetch_add(change.positions.len() as u64,std::sync::atomic::Ordering::Relaxed);\n            let (encoded,capacity)=change.positions.bench_storage();\n            BENCH_ENCODED.fetch_add(encoded,std::sync::atomic::Ordering::Relaxed);\n            BENCH_CAPACITY.fetch_add(capacity,std::sync::atomic::Ordering::Relaxed);\n            let index = events.len();')
    text=replace(text,'        events\n    }','        BENCH_L.fetch_add(bench_l,std::sync::atomic::Ordering::Relaxed);\n        BENCH_H.fetch_add(bench_h,std::sync::atomic::Ordering::Relaxed);\n        BENCH_E.fetch_add(bench_e,std::sync::atomic::Ordering::Relaxed);\n        events\n    }')
    text=replace(text,'        fragments.sort_by_key','        BENCH_SORT_FRAGMENTS.fetch_add(fragments.len() as u64,std::sync::atomic::Ordering::Relaxed);\n        fragments.sort_by_key')
    text=replace(text,'                streams.for_each(drop);','                BENCH_REJECTED_KEYS.fetch_add(1,std::sync::atomic::Ordering::Relaxed);\n                streams.for_each(drop);')
    text=replace(text,'            let positions = if reuse {','            BENCH_ADMITTED_KEYS.fetch_add(1,std::sync::atomic::Ordering::Relaxed);\n            let positions = if reuse {')
    text=replace(text,'        for candidate in births.into_iter().flatten() {','        for candidate in births.into_iter().flatten() {\n            BENCH_HEAP_PUSHES.fetch_add(1,std::sync::atomic::Ordering::Relaxed);')
    text=replace(text,'.map(|shard| shard.commit(&events, reuse, floor))','''.map(|shard| {
                let started=super::merge::bench_thread_cpu();
                let result=shard.commit(&events,reuse,floor);
                let cpu=super::merge::bench_thread_cpu()-started;
                BENCH_OWNER_SUM_CPU.fetch_add(cpu,std::sync::atomic::Ordering::Relaxed);
                BENCH_OWNER_MAX_CPU.fetch_max(cpu,std::sync::atomic::Ordering::Relaxed);
                result
            })''')
    index.write_text(text+index_counters)
    main=runner/'src/main.rs';main.write_text('mod allocator;\n'+main.read_text())
    shutil.copyfile('/tmp/bpe-publisher-allocator.rs',runner/'src/allocator.rs')
    core=runner/'src/core.rs';text=core.read_text()
    text=replace(text,'    let train_seconds = begin.elapsed().as_secs_f64();','    crate::allocator::report();\n    let train_seconds = begin.elapsed().as_secs_f64();')
    start=text.index('pub fn run');text=text[:start]+text[start:].replace('    let cpu = cpu_seconds()?;','    crate::allocator::start();\n    let cpu = cpu_seconds()?;',1);core.write_text(text)
    mod=work/rel/'mod.rs';text=mod.read_text()
    text=replace(text,'            let mut window_high_water = 0;','            merge::BENCH_MERGING.store(true,std::sync::atomic::Ordering::Relaxed);\n            let mut window_high_water = 0;')
    text=replace(text,'                        return Ok(AttemptOutcome::RestartForReuse);','                        merge::BENCH_MERGING.store(false,std::sync::atomic::Ordering::Relaxed);\n                        return Ok(AttemptOutcome::RestartForReuse);')
    text=replace(text,'            merge::bench_report_phases();','            merge::BENCH_MERGING.store(false,std::sync::atomic::Ordering::Relaxed);\n            merge::bench_report_phases();\n            merge::bench_report_work();\n            merge::bench_report_directories();\n            index::bench_report_actions();\n            index::bench_report_encoding();')
    mod.write_text(text)
    if prepare_only:
        print(f'{arm}: diagnostic sources prepared; no compilation',flush=True)
        break
    with (root/f'{arm}-diagnostic-resolve.log').open('w') as log:subprocess.run(['cargo','metadata','--offline','--format-version','1'],cwd=runner,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    shutil.copytree(work/rel,snapshot/rel);shutil.copytree(runner,snapshot/'experiments/bpe-simplification/runner');shutil.copyfile(manifest,snapshot/'tokenizers/tk-train/Cargo.toml')
    command=['cargo','build','--release','--locked','--offline','--target-dir',str(target)]
    with (root/f'{arm}-diagnostic-build.log').open('w') as log:subprocess.run(command,cwd=runner,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    binary=root/f'{arm}-diagnostic-bin';shutil.copyfile(target/'release/bpe-bench-runner',binary);binary.chmod(0o755)
    meta=dict(arm=arm+'-diagnostic',command=command,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),diagnostic_only=True)
    (root/f'{arm}-diagnostic-build.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)
