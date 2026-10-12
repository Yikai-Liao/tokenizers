from pathlib import Path
import subprocess, shutil, hashlib, os, json, sys, re

root = Path('/root/code/tokenizers-workspaces/bpe-complete-publisher-results-20261012')
root.mkdir(exist_ok=True)
work = Path('/tmp/bpe-selected-rules-measurement-20261011')
rel = Path('tokenizers/tk-train/src/trainers/bpe')
baseline = Path('/tmp/bpe-prefetch-right-20261012')
candidate = Path('/tmp/bpe-complete-publisher-20261012')
frozen = Path('/root/code/tokenizers-workspaces/bpe-prefetch-right-results-20261012/source/baseline')
target = Path('/root/code/tokenizers-build/bpe-affix-core-build-20261011')
env = os.environ | {'PATH': '/root/.cargo/bin:' + os.environ['PATH'], 'CARGO_INCREMENTAL': '0', 'CARGO_BUILD_JOBS': '2'}

def timed_commit(body, complete):
    route = '        let events = self.route_changes(changes);'
    if complete:
        pattern = r'        let \(losses, births\) = super::merge::complete_births\([\s\S]*?;'
        match = re.search(pattern, body)
        assert match
        call = match.group()
        body = body.replace(call, '        let producer_started = std::time::Instant::now();\n' + call + '\n        BENCH_PRODUCER_NS.fetch_add(producer_started.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);\n        BENCH_PRODUCER_JOINS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);')
        route = '        let events = self.route_changes(losses);'
        body = body.replace(route, '        let route_started = std::time::Instant::now();\n' + route)
        needle = '        let floor = self.floor;'
        assert body.count(needle) == 1
        body = body.replace(needle, '        BENCH_ROUTE_NS.fetch_add(route_started.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);\n' + needle)
    else:
        assert body.count(route) == 1
        body = body.replace(route, '        let route_started = std::time::Instant::now();\n' + route + '\n        BENCH_ROUTE_NS.fetch_add(route_started.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);')
    needle = '        let births = self\n'
    assert body.count(needle) == 1
    body = body.replace(needle, '        let owners_started = std::time::Instant::now();\n' + needle)
    needle = '            .collect::<Result<Vec<_>>>()?;'
    assert body.count(needle) == 1
    body = body.replace(needle, needle + '\n        BENCH_OWNERS_NS.fetch_add(owners_started.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);')
    needle = '        for candidate in births.into_iter().flatten() {'
    assert body.count(needle) == 1
    body = body.replace(needle, '        let publish_started = std::time::Instant::now();\n' + needle)
    body = body.replace('        Ok(())', '        BENCH_PUBLISH_NS.fetch_add(publish_started.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);\n        Ok(())')
    return body

for arm in (sys.argv[1:] or ['baseline', 'producer']):
    assert arm in ('baseline', 'producer')
    snapshot = root/'source'/arm
    assert not snapshot.exists(), ('frozen snapshot already exists', snapshot)
    source = baseline if arm == 'baseline' else candidate
    runner = work/'experiments/bpe-simplification/runner'
    shutil.copytree(frozen/'experiments/bpe-simplification/runner', runner, dirs_exist_ok=True)
    shutil.copyfile(source/'tokenizers/tk-train/Cargo.toml', work/'tokenizers/tk-train/Cargo.toml')
    for path in (source/rel).glob('*.rs'):
        shutil.copyfile(path, work/rel/path.name)
    shutil.copyfile(frozen/rel/'mod.rs', work/rel/'mod.rs')
    shutil.copyfile(frozen/rel/'corpus.rs', work/rel/'corpus.rs')
    merge = work/rel/'merge.rs'
    if arm == 'baseline':
        shutil.copyfile(frozen/rel/'merge.rs', merge)
    else:
        text = merge.read_text()
        needle = '        let jobs = batch.prepare(corpus, self.max_token_length.unwrap_or(usize::MAX))?;'
        assert text.count(needle) == 1
        text = text.replace(needle, '        let prepare_reuse = batch.reuse;\n        let prepare_started = std::time::Instant::now();\n' + needle + '\n        BENCH_PREPARE_NS.fetch_add(prepare_started.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);\n        if prepare_reuse { &BENCH_REUSE_BATCHES } else { &BENCH_FRESH_BATCHES }.fetch_add(1, std::sync::atomic::Ordering::Relaxed);')
        old = (frozen/rel/'merge.rs').read_text()
        text += '\n' + old[old.index('static BENCH_PREPARE_NS:'):]
        merge.write_text(text)
    text = merge.read_text()
    matches = re.findall(r'        let changes(?:: Vec<_>)? = jobs', text)
    assert len(matches) == 1
    needle = matches[0]
    text = text.replace(needle, '        let apply_started = std::time::Instant::now();\n' + needle)
    start = text.index('        let changes')
    end = text.index('            .collect();', start) + len('            .collect();')
    text = text[:end] + '\n        BENCH_APPLY_NS.fetch_add(apply_started.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);\n        let commit_started = std::time::Instant::now();' + text[end:]
    needle = '        Ok(MergeOutcome::Committed(records))'
    text = text.replace(needle, '        BENCH_COMMIT_NS.fetch_add(commit_started.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);\n' + needle)
    text += '''
static BENCH_APPLY_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BENCH_COMMIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(super) fn bench_report_apply_commit() {
    eprintln!("BPE_APPLY_SECONDS {:.9}", BENCH_APPLY_NS.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e9);
    eprintln!("BPE_COMMIT_SECONDS {:.9}", BENCH_COMMIT_NS.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e9);
}
'''
    merge.write_text(text)
    index = work/rel/'index.rs'
    text = index.read_text()
    for name in (['commit', 'commit_complete'] if arm == 'producer' else ['commit']):
        start = text.index(f'    pub(super) fn {name}(')
        end = text.index('\n    fn route_changes(', start) if name == 'commit_complete' or arm == 'baseline' else text.index('\n    /// Fresh producers', start)
        text = text[:start] + timed_commit(text[start:end], name == 'commit_complete') + text[end:]
    text += '''
static BENCH_ROUTE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BENCH_OWNERS_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BENCH_PUBLISH_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BENCH_PRODUCER_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BENCH_PRODUCER_JOINS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(super) fn bench_report_publisher() {
    eprintln!("BPE_ROUTE_SECONDS {:.9}", BENCH_ROUTE_NS.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e9);
    eprintln!("BPE_OWNERS_SECONDS {:.9}", BENCH_OWNERS_NS.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e9);
    eprintln!("BPE_PUBLISH_SECONDS {:.9}", BENCH_PUBLISH_NS.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e9);
    eprintln!("BPE_PRODUCER_SECONDS {:.9}", BENCH_PRODUCER_NS.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e9);
    eprintln!("BPE_PRODUCER_JOINS {}", BENCH_PRODUCER_JOINS.swap(0, std::sync::atomic::Ordering::Relaxed));
}
'''
    index.write_text(text)
    mod = work/rel/'mod.rs'
    text = mod.read_text()
    needle = '            merge::bench_report_prepare();'
    assert text.count(needle) == 1
    mod.write_text(text.replace(needle, needle + '\n            merge::bench_report_apply_commit();\n            index::bench_report_publisher();'))
    shutil.copyfile(frozen/'experiments/bpe-simplification/runner/Cargo.lock', runner/'Cargo.lock')
    shutil.copytree(work/rel, snapshot/rel)
    shutil.copytree(runner, snapshot/'experiments/bpe-simplification/runner')
    shutil.copyfile(work/'tokenizers/tk-train/Cargo.toml', snapshot/'tokenizers/tk-train/Cargo.toml')
    command = ['cargo', 'build', '--release', '--locked', '--offline', '--target-dir', str(target)]
    with (root/f'{arm}-build.log').open('w') as log:
        subprocess.run(command, cwd=runner, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    binary = root/f'{arm}-bin'
    shutil.copyfile(target/'release/bpe-bench-runner', binary)
    binary.chmod(0o755)
    metadata = dict(arm=arm, command=command, base_commit='890ea1f58e65f8d8503f7cd3fc97ad1651eaf284', binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), rustc=subprocess.check_output(['rustc', '-Vv'], env=env, text=True), environment={key: env[key] for key in ('CARGO_INCREMENTAL', 'CARGO_BUILD_JOBS')})
    (root/f'{arm}-build.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata), flush=True)
