from pathlib import Path
import subprocess, shutil, hashlib, os, json, sys

root = Path('/root/code/tokenizers-workspaces/bpe-frontier-window-results-20261012')
root.mkdir(exist_ok=True)
work = Path('/tmp/bpe-selected-rules-measurement-20261011')
rel = Path('tokenizers/tk-train/src/trainers/bpe')
candidate = Path('/tmp/bpe-frontier-window-20261012')
frozen = Path('/root/code/tokenizers-workspaces/bpe-prefetch-right-results-20261012/source/baseline')
target = Path('/root/code/tokenizers-build/bpe-affix-core-build-20261011')
env = os.environ | {'PATH': '/root/.cargo/bin:' + os.environ['PATH'], 'CARGO_INCREMENTAL': '0', 'CARGO_BUILD_JOBS': '2'}
phases = ['PREPARE', 'APPLY', 'COMMIT', 'OLD_COMMIT', 'FINAL_COMMIT', 'REPLAY']

def add_timer(text, needle, phase, end=None):
    assert text.count(needle) == 1, (phase, needle, text.count(needle))
    start = text.index(needle)
    stop = start + len(needle) if end is None else text.index(end, start) + len(end)
    return (text[:start] + f'        let bench_{phase.lower()}_started = std::time::Instant::now();\n'
            + text[start:stop] + f'\n        BENCH_{phase}_NS.fetch_add(bench_{phase.lower()}_started.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);'
            + text[stop:])

for arm in (sys.argv[1:] or ['eager', 'single', 'frontier']):
    assert arm in ('eager', 'single', 'frontier')
    snapshot = root/'source'/arm
    assert not snapshot.exists(), ('frozen snapshot already exists', snapshot)
    runner = work/'experiments/bpe-simplification/runner'
    shutil.copytree(frozen/'experiments/bpe-simplification/runner', runner, dirs_exist_ok=True)
    shutil.copyfile(candidate/'tokenizers/tk-train/Cargo.toml', work/'tokenizers/tk-train/Cargo.toml')
    for path in (candidate/rel).glob('*.rs'):
        shutil.copyfile(path, work/rel/path.name)
    for name in ['mod.rs', 'corpus.rs']:
        shutil.copyfile(frozen/rel/name, work/rel/name)
    mod = work/rel/'mod.rs'
    text = mod.read_text()
    needle = '            while vocabulary.len() < self.vocab_size {'
    assert text.count(needle) == 1
    text = text.replace(needle, '            let mut window_high_water = 0;\n' + needle)
    text = text.replace('self.merge_next(&mut vocabulary, &mut corpus, &mut index)?', 'self.merge_next(&mut vocabulary, &mut corpus, &mut index, &mut window_high_water)?')
    text = text.replace('            merge::bench_report_prepare();', '            merge::bench_report_phases();\n            index::bench_report_publisher();')
    mod.write_text(text)
    merge = work/rel/'merge.rs'
    text = merge.read_text()
    assert text.count('pub(super) const WINDOW_COHORTS: usize = 4;') == 1
    text = text.replace('pub(super) const WINDOW_COHORTS: usize = 4;', f'pub(super) const WINDOW_COHORTS: usize = '+str({'eager':0,'single':1,'frontier':4}[arm])+';')
    # Timers run once per joined phase; no per-position counters in normal builds.
    text = add_timer(text, '        let jobs = batch.prepare(corpus, self.max_token_length.unwrap_or(usize::MAX))?;', 'PREPARE')
    text = text.replace('        let bench_prepare_started', '        if reuse { &BENCH_REUSE_BATCHES } else { &BENCH_FRESH_BATCHES }.fetch_add(1, std::sync::atomic::Ordering::Relaxed);\n        let bench_prepare_started', 1)
    text = add_timer(text, '        let changes = jobs', 'APPLY', '            .collect();')
    text = add_timer(text, '        index.commit(changes)?;', 'COMMIT')
    text = add_timer(text, '            let jobs = batch.prepare_fresh(corpus, limit, Some(boundary))?;', 'PREPARE')
    text = text.replace('            let jobs = batch.prepare_fresh', '            BENCH_FRESH_BATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);\n            let jobs = batch.prepare_fresh')
    text = add_timer(text, '            let (jobs, losses): (Vec<_>, Vec<_>) = jobs', 'APPLY', '                .unzip();')
    text = add_timer(text, '            index.commit(losses)?;', 'OLD_COMMIT')
    text = add_timer(text, '        let jobs: Vec<_> = staged.into_iter().flatten().collect();', 'REPLAY', '            .collect::<Result<Vec<_>>>()?;')
    text = add_timer(text, '        index.commit(births)?;', 'FINAL_COMMIT')
    for phase in phases:
        text += f'\nstatic BENCH_{phase}_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);'
    for name in ['FRESH_BATCHES', 'REUSE_BATCHES']:
        text += f'\nstatic BENCH_{name}: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);'
    text += '\npub(super) fn bench_report_phases() {\n'
    for phase in phases:
        text += f'    eprintln!("BPE_{phase}_SECONDS {{:.9}}", BENCH_{phase}_NS.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e9);\n'
    for name in ['FRESH_BATCHES', 'REUSE_BATCHES']:
        text += f'    eprintln!("BPE_{name} {{}}", BENCH_{name}.swap(0, std::sync::atomic::Ordering::Relaxed));\n'
    text += '}\n'
    merge.write_text(text)
    index = work/rel/'index.rs'
    text = index.read_text()
    start = text.index('    pub(super) fn commit(')
    stop = text.index('\n    fn route_changes(', start)
    body = text[start:stop]
    body = add_timer(body, '        let events = self.route_changes(changes);', 'ROUTE')
    body = add_timer(body, '        let births = self\n', 'OWNERS', '            .collect::<Result<Vec<_>>>()?;')
    body = add_timer(body, '        for candidate in births.into_iter().flatten() {', 'PUBLISH', '\n        }')
    text = text[:start] + body + text[stop:]
    for phase in ['ROUTE', 'OWNERS', 'PUBLISH']:
        text += f'\nstatic BENCH_{phase}_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);'
    text += '\npub(super) fn bench_report_publisher() {\n'
    for phase in ['ROUTE', 'OWNERS', 'PUBLISH']:
        text += f'    eprintln!("BPE_{phase}_SECONDS {{:.9}}", BENCH_{phase}_NS.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e9);\n'
    text += '}\n'
    index.write_text(text)
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
