from pathlib import Path
import shutil, subprocess, os, json, hashlib, sys

root = Path('/root/code/tokenizers-workspaces/bpe-complete-publisher-results-20261012')
work = Path('/tmp/bpe-selected-rules-measurement-20261011')
rel = Path('tokenizers/tk-train/src/trainers/bpe')
target = Path('/root/code/tokenizers-build/bpe-affix-core-build-20261011')
env = os.environ | {'PATH': '/root/.cargo/bin:' + os.environ['PATH'], 'CARGO_INCREMENTAL': '0', 'CARGO_BUILD_JOBS': '2'}

directory = '''
// Diagnostic-only accounting. Site 0 is preparation; site 1 is the producer reducer.
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
struct DirectoryStats {
    states: AtomicU64, resizes: AtomicU64, filled: AtomicU64,
    allocations: AtomicU64, growths: AtomicU64, old_on_growth: AtomicU64,
    live_capacity: AtomicU64, peak_capacity: AtomicU64, max_domain: AtomicU64,
    live_states: AtomicU64, peak_states: AtomicU64,
    threads: thread_local::ThreadLocal<()>,
}
impl DirectoryStats {
    const fn new() -> Self { Self {
        states: AtomicU64::new(0), resizes: AtomicU64::new(0), filled: AtomicU64::new(0),
        allocations: AtomicU64::new(0), growths: AtomicU64::new(0), old_on_growth: AtomicU64::new(0),
        live_capacity: AtomicU64::new(0), peak_capacity: AtomicU64::new(0), max_domain: AtomicU64::new(0),
        live_states: AtomicU64::new(0), peak_states: AtomicU64::new(0), threads: thread_local::ThreadLocal::new(),
    }}
}
static DIRECTORY_SITES: [DirectoryStats; 2] = [DirectoryStats::new(), DirectoryStats::new()];
impl Directories {
    fn bench_new(site: usize) -> Self {
        let stats = &DIRECTORY_SITES[site];
        stats.threads.get_or_default();
        stats.states.fetch_add(1, Relaxed);
        let live = stats.live_states.fetch_add(1, Relaxed) + 1;
        stats.peak_states.fetch_max(live, Relaxed);
        Self { indices: Default::default(), bench_site: site }
    }
}
impl Default for Directories { fn default() -> Self { Self::bench_new(0) } }
impl Drop for Directories {
    fn drop(&mut self) {
        let stats = &DIRECTORY_SITES[self.bench_site];
        stats.live_capacity.fetch_sub(self.indices.iter().map(|v| v.capacity() as u64 * 4).sum(), Relaxed);
        stats.live_states.fetch_sub(1, Relaxed);
    }
}
pub(super) fn bench_report_directories() {
    for (site, stats) in DIRECTORY_SITES.iter().enumerate() {
        eprintln!("BPE_DIRECTORIES {}", serde_json::json!({
            "site": if site == 0 { "prepare" } else { "producer" },
            "states_created": stats.states.load(Relaxed), "threads_observed": stats.threads.iter().count(),
            "resize_calls": stats.resizes.load(Relaxed), "sentinel_elements": stats.filled.load(Relaxed),
            "sentinel_logical_bytes": stats.filled.load(Relaxed) * 4,
            "vector_initial_allocations": stats.allocations.load(Relaxed),
            "vector_capacity_growths": stats.growths.load(Relaxed),
            "old_elements_at_capacity_growth": stats.old_on_growth.load(Relaxed),
            "peak_live_vector_capacity_bytes": stats.peak_capacity.load(Relaxed),
            "live_vector_capacity_bytes_at_report": stats.live_capacity.load(Relaxed),
            "max_domain": stats.max_domain.load(Relaxed),
            "peak_live_states": stats.peak_states.load(Relaxed), "live_states_at_report": stats.live_states.load(Relaxed),
            "diagnostic_header_bytes": std::mem::size_of::<Directories>(), "production_header_bytes": 48,
        }));
    }
}
'''

producer_counters = '''
static PRODUCER_G: AtomicU64 = AtomicU64::new(0);
static PRODUCER_H: AtomicU64 = AtomicU64::new(0);
static PRODUCER_K: AtomicU64 = AtomicU64::new(0);
static PRODUCER_K_LIVE: AtomicU64 = AtomicU64::new(0);
static PRODUCER_COUNT: AtomicU64 = AtomicU64::new(0);
static PRODUCER_MAX_WALL_NS: AtomicU64 = AtomicU64::new(0);
static PRODUCER_SUM_CPU_NS: AtomicU64 = AtomicU64::new(0);
static PRODUCER_MAX_CPU_NS: AtomicU64 = AtomicU64::new(0);
static PRODUCER_MAX_G: AtomicU64 = AtomicU64::new(0);
static PRODUCER_MAX_H: AtomicU64 = AtomicU64::new(0);
static PRODUCER_MAX_K: AtomicU64 = AtomicU64::new(0);
static PRODUCER_SPILLS: AtomicU64 = AtomicU64::new(0);
static PRODUCER_MAX_STAGING: AtomicU64 = AtomicU64::new(0);
static PRODUCER_STAGING_SUM: AtomicU64 = AtomicU64::new(0);
static PRODUCER_INPUT_JOB_BYTES: AtomicU64 = AtomicU64::new(0);
// Linux diagnostic binary only: clock_gettime writes the C timespec layout.
#[repr(C)] struct BenchTimespec { seconds: std::os::raw::c_long, nanos: std::os::raw::c_long }
unsafe extern "C" { fn clock_gettime(clock: std::os::raw::c_int, time: *mut BenchTimespec) -> std::os::raw::c_int; }
pub(super) fn bench_thread_cpu() -> u64 {
    let mut value = BenchTimespec { seconds: 0, nanos: 0 };
    // CLOCK_THREAD_CPUTIME_ID is 3 on this Linux benchmark host; value is writable.
    assert_eq!(unsafe { clock_gettime(3, &mut value) }, 0);
    value.seconds as u64 * 1_000_000_000 + value.nanos as u64
}
pub(super) fn bench_report_producer() {
    eprintln!("BPE_PRODUCER_DIAGNOSTIC {}", serde_json::json!({
        "producers": PRODUCER_COUNT.load(Relaxed), "changes_G": PRODUCER_G.load(Relaxed),
        "birth_fragments_H": PRODUCER_H.load(Relaxed), "complete_keys_K": PRODUCER_K.load(Relaxed),
        "live_keys_K": PRODUCER_K_LIVE.load(Relaxed), "max_G": PRODUCER_MAX_G.load(Relaxed),
        "max_H": PRODUCER_MAX_H.load(Relaxed), "max_K": PRODUCER_MAX_K.load(Relaxed),
        "max_wall_ns": PRODUCER_MAX_WALL_NS.load(Relaxed), "sum_cpu_ns": PRODUCER_SUM_CPU_NS.load(Relaxed),
        "max_cpu_ns": PRODUCER_MAX_CPU_NS.load(Relaxed), "spilled_leaf_groups": PRODUCER_SPILLS.load(Relaxed),
        "max_single_producer_staging_checkpoint_bytes": PRODUCER_MAX_STAGING.load(Relaxed),
        "sum_producer_staging_checkpoints_bytes": PRODUCER_STAGING_SUM.load(Relaxed),
        "sum_input_job_vector_capacity_bytes": PRODUCER_INPUT_JOB_BYTES.load(Relaxed),
    }));
}
'''

index_counters = '''
static BENCH_L: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BENCH_H: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BENCH_E: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BENCH_K: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BENCH_SORT_FRAGMENTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BENCH_OWNER_SUM_CPU: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BENCH_OWNER_MAX_CPU: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(super) fn bench_report_actions() {
    use std::sync::atomic::Ordering::Relaxed;
    let (l,h,e,k) = (BENCH_L.load(Relaxed), BENCH_H.load(Relaxed), BENCH_E.load(Relaxed), BENCH_K.load(Relaxed));
    eprintln!("BPE_PUBLISHER_DIAGNOSTIC {}", serde_json::json!({
        "loss_hashes_L": l, "fragment_hashes_H": h, "coalesced_E": e, "complete_birth_hashes_K": k,
        "owner_hashes": l+h+k, "owner_actions": l+h-e+k,
        "sorted_fragments": BENCH_SORT_FRAGMENTS.load(Relaxed),
        "owner_sum_cpu_ns": BENCH_OWNER_SUM_CPU.load(Relaxed), "owner_max_cpu_ns": BENCH_OWNER_MAX_CPU.load(Relaxed),
    }));
}
'''

for arm in (sys.argv[1:] or ['baseline', 'producer']):
    snapshot = root/'source'/f'{arm}-diagnostic'
    assert not snapshot.exists(), snapshot
    source = root/'source'/arm
    shutil.copytree(source/rel, work/rel, dirs_exist_ok=True)
    runner = work/'experiments/bpe-simplification/runner'
    shutil.copytree(source/'experiments/bpe-simplification/runner', runner, dirs_exist_ok=True)
    manifest = work/'tokenizers/tk-train/Cargo.toml'
    shutil.copyfile(source/'tokenizers/tk-train/Cargo.toml', manifest)
    manifest.write_text(manifest.read_text().replace('[dependencies]', '[dependencies]\nthread_local = "1.1.10"'))
    merge = work/rel/'merge.rs'
    text = merge.read_text().replace('#[derive(Default)]\nstruct Directories {', 'struct Directories {\n    bench_site: usize,')
    old = '''    fn resize(&mut self, domain: usize) {
        for values in &mut self.indices {
            values.resize(domain, u32::MAX);
        }
    }'''
    new = '''    fn resize(&mut self, domain: usize) {
        let stats = &DIRECTORY_SITES[self.bench_site];
        stats.resizes.fetch_add(1, Relaxed); stats.max_domain.fetch_max(domain as u64, Relaxed);
        for values in &mut self.indices {
            let (old_length, old_capacity) = (values.len(), values.capacity());
            assert!(domain >= old_length);
            values.resize(domain, u32::MAX);
            stats.filled.fetch_add((domain-old_length) as u64, Relaxed);
            let capacity = values.capacity();
            if capacity != old_capacity {
                if old_capacity == 0 { stats.allocations.fetch_add(1, Relaxed); }
                else { stats.growths.fetch_add(1, Relaxed); stats.old_on_growth.fetch_add(old_length as u64, Relaxed); }
                let delta = (capacity-old_capacity) as u64 * 4;
                let live = stats.live_capacity.fetch_add(delta, Relaxed) + delta;
                stats.peak_capacity.fetch_max(live, Relaxed);
            }
        }
    }'''
    assert text.count(old) == 1
    text = text.replace(old, new)
    if arm == 'producer':
        text = text.replace('.map_init(Directories::default, |directories, (replacement, jobs)| {', '.map_init(|| Directories::bench_new(1), |directories, (replacement, jobs)| {\n            let bench_started = std::time::Instant::now();\n            let bench_cpu = bench_thread_cpu();\n            let mut bench_g = 0u64; let mut bench_h = 0u64;\n            PRODUCER_INPUT_JOB_BYTES.fetch_add(jobs.capacity() as u64 * std::mem::size_of::<Vec<Change<Positions>>>() as u64 + jobs.iter().map(|v| v.capacity() as u64 * std::mem::size_of::<Change<Positions>>() as u64).sum::<u64>(), Relaxed);')
        text = text.replace('            for mut change in jobs.into_iter().flatten() {', '            for mut change in jobs.into_iter().flatten() {\n                bench_g += 1;')
        text = text.replace('                if !change.positions.is_empty() {', '                if !change.positions.is_empty() {\n                    bench_h += 1;', 1)
        needle = '            let mut births = Vec::new();\n            for (side, groups)'
        replacement = '''            let bench_k = groups.iter().map(|v| v.len() as u64).sum::<u64>();
            let mut bench_staging = groups.iter().map(|v| v.capacity() as u64 * std::mem::size_of::<Birth>() as u64).sum::<u64>() + losses.capacity() as u64 * std::mem::size_of::<Change<Positions>>() as u64;
            for group in groups.iter().flatten() {
                if group.leaves.spilled() {
                    PRODUCER_SPILLS.fetch_add(1, Relaxed);
                    bench_staging += group.leaves.capacity() as u64 * std::mem::size_of::<Positions>() as u64;
                }
                bench_staging += group.leaves.iter().map(|p| p.bench_storage().1).sum::<u64>();
            }
            let mut births = Vec::new();
            for (side, groups)'''
        assert text.count(needle) == 1
        text = text.replace(needle, replacement)
        text = text.replace('            Ok((losses, births))', '''            PRODUCER_COUNT.fetch_add(1, Relaxed);
            PRODUCER_G.fetch_add(bench_g, Relaxed); PRODUCER_H.fetch_add(bench_h, Relaxed);
            PRODUCER_K.fetch_add(bench_k, Relaxed); PRODUCER_K_LIVE.fetch_add(births.len() as u64, Relaxed);
            PRODUCER_MAX_G.fetch_max(bench_g, Relaxed); PRODUCER_MAX_H.fetch_max(bench_h, Relaxed); PRODUCER_MAX_K.fetch_max(bench_k, Relaxed);
            PRODUCER_MAX_STAGING.fetch_max(bench_staging, Relaxed); PRODUCER_STAGING_SUM.fetch_add(bench_staging, Relaxed);
            let cpu = bench_thread_cpu() - bench_cpu;
            PRODUCER_SUM_CPU_NS.fetch_add(cpu, Relaxed); PRODUCER_MAX_CPU_NS.fetch_max(cpu, Relaxed);
            PRODUCER_MAX_WALL_NS.fetch_max(bench_started.elapsed().as_nanos() as u64, Relaxed);
            Ok((losses, births))''')
    merge.write_text(text + '\n' + directory + producer_counters)
    positions = work/rel/'positions.rs'
    text = positions.read_text()
    needle = '    pub(super) fn iter(&self)'
    assert text.count(needle) == 2
    text = text.replace(needle, '''    pub(super) fn bench_storage(&self) -> (u64, u64) {
        let encoded = self.leaves().iter().map(|v| v.bytes.len() as u64).sum();
        let heap = self.leaves().iter().filter(|v| v.bytes.spilled()).map(|v| v.bytes.capacity() as u64).sum::<u64>()
            + match &self.0 { Repr::Root { leaves, .. } => leaves.len() as u64 * std::mem::size_of::<Leaf>() as u64, _ => 0 };
        (encoded, heap)
    }

''' + needle, 1)
    positions.write_text(text)
    index = work/rel/'index.rs'
    text = index.read_text()
    needle = '        for change in changes.into_iter().flatten() {'
    assert text.count(needle) == 1
    text = text.replace(needle, '        let (mut bench_l, mut bench_h, mut bench_e) = (0u64, 0u64, 0u64);\n' + needle)
    needle = '            let index = events.len();'
    text = text.replace(needle, '            bench_l += u64::from(removed.is_some()); bench_h += u64::from(born.is_some());\n            bench_e += u64::from(removed.is_some() && removed == born);\n' + needle)
    text = text.replace('        events\n    }', '        BENCH_L.fetch_add(bench_l, std::sync::atomic::Ordering::Relaxed);\n        BENCH_H.fetch_add(bench_h, std::sync::atomic::Ordering::Relaxed);\n        BENCH_E.fetch_add(bench_e, std::sync::atomic::Ordering::Relaxed);\n        events\n    }')
    text = text.replace('        for candidate in births {', '        BENCH_K.fetch_add(births.len() as u64, std::sync::atomic::Ordering::Relaxed);\n        for candidate in births {')
    text = text.replace('        fragments.sort_by_key', '        BENCH_SORT_FRAGMENTS.fetch_add(fragments.len() as u64, std::sync::atomic::Ordering::Relaxed);\n        fragments.sort_by_key')
    # Baseline owner CPU surrounds shard.commit; producer arm covers its complete owner action.
    text = text.replace('.map(|shard| shard.commit(&events, reuse, floor))', '''.map(|shard| {
                let started = super::merge::bench_thread_cpu();
                let result = shard.commit(&events, reuse, floor);
                let cpu = super::merge::bench_thread_cpu() - started;
                BENCH_OWNER_SUM_CPU.fetch_add(cpu, std::sync::atomic::Ordering::Relaxed);
                BENCH_OWNER_MAX_CPU.fetch_max(cpu, std::sync::atomic::Ordering::Relaxed);
                result
            })''')
    text = text.replace('            .map(|(shard, births)| {', '            .map(|(shard, births)| {\n                let started = super::merge::bench_thread_cpu();')
    text = text.replace('                Ok(births)', '                let cpu = super::merge::bench_thread_cpu() - started;\n                BENCH_OWNER_SUM_CPU.fetch_add(cpu, std::sync::atomic::Ordering::Relaxed);\n                BENCH_OWNER_MAX_CPU.fetch_max(cpu, std::sync::atomic::Ordering::Relaxed);\n                Ok(births)')
    index.write_text(text + '\n' + index_counters)
    main = runner/'src/main.rs'
    main.write_text('mod allocator;\n' + main.read_text())
    shutil.copyfile('/tmp/bpe-publisher-allocator.rs', runner/'src/allocator.rs')
    core = runner/'src/core.rs'
    text = core.read_text()
    text = text.replace('    let train_seconds = begin.elapsed().as_secs_f64();', '    crate::allocator::report();\n    let train_seconds = begin.elapsed().as_secs_f64();')
    start = text.index('pub fn run')
    text = text[:start] + text[start:].replace('    let cpu = cpu_seconds()?;', '    crate::allocator::start();\n    let cpu = cpu_seconds()?;', 1)
    core.write_text(text)
    mod = work/rel/'mod.rs'
    mod.write_text(mod.read_text().replace('            index::bench_report_publisher();', '            index::bench_report_publisher();\n            merge::bench_report_directories();\n            merge::bench_report_producer();\n            index::bench_report_actions();'))
    with (root/f'{arm}-diagnostic-resolve.log').open('w') as log:
        subprocess.run(['cargo', 'metadata', '--offline', '--format-version', '1'], cwd=runner, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    shutil.copytree(work/rel, snapshot/rel)
    shutil.copytree(runner, snapshot/'experiments/bpe-simplification/runner')
    shutil.copyfile(manifest, snapshot/'tokenizers/tk-train/Cargo.toml')
    command = ['cargo', 'build', '--release', '--locked', '--offline', '--target-dir', str(target)]
    with (root/f'{arm}-diagnostic-build.log').open('w') as log:
        subprocess.run(command, cwd=runner, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    binary = root/f'{arm}-diagnostic-bin'
    shutil.copyfile(target/'release/bpe-bench-runner', binary)
    binary.chmod(0o755)
    meta = dict(arm=arm+'-diagnostic', command=command, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    (root/f'{arm}-diagnostic-build.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(meta), flush=True)
