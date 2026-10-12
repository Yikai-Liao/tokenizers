from pathlib import Path
import shutil, subprocess, os, json, hashlib, sys

root = Path('/root/code/tokenizers-workspaces/bpe-attempt-scratch-results-20261012')
work = Path('/tmp/bpe-selected-rules-measurement-20261011')
rel = Path('tokenizers/tk-train/src/trainers/bpe')
target = Path('/root/code/tokenizers-build/bpe-affix-core-build-20261011')
env = os.environ | {'PATH': '/root/.cargo/bin:' + os.environ['PATH'], 'CARGO_INCREMENTAL': '0', 'CARGO_BUILD_JOBS': '2'}

for arm in (sys.argv[1:] or ['baseline', 'scratch']):
    snapshot = root/'source'/f'{arm}-diagnostic'
    assert not snapshot.exists()
    source = root/'source'/arm
    shutil.copytree(source/rel, work/rel, dirs_exist_ok=True)
    runner = work/'experiments/bpe-simplification/runner'
    shutil.copytree(source/'experiments/bpe-simplification/runner', runner, dirs_exist_ok=True)
    manifest = work/'tokenizers/tk-train/Cargo.toml'
    shutil.copyfile(source/'tokenizers/tk-train/Cargo.toml', manifest)
    text = manifest.read_text()
    if 'thread_local = ' not in text:
        text = text.replace('[dependencies]', '[dependencies]\nthread_local = "1.1.10"')
    manifest.write_text(text)
    merge = work/rel/'merge.rs'
    text = merge.read_text()
    needle = '#[derive(Default)]\n'
    begin = text.index(needle, text.index('struct Neighbors'))
    assert text[begin:].split('{', 1)[0].strip().endswith('struct Directories')
    text = text[:begin] + text[begin:].replace(needle, '', 1)
    needle = '''    fn resize(&mut self, domain: usize) {
        for values in &mut self.indices {
            values.resize(domain, u32::MAX);
        }
    }'''
    assert text.count(needle) == 1
    text = text.replace(needle, '''    fn resize(&mut self, domain: usize) {
        DIRECTORY_RESIZES.fetch_add(1, Relaxed);
        DIRECTORY_MAX_DOMAIN.fetch_max(domain as u64, Relaxed);
        for values in &mut self.indices {
            let old_length = values.len();
            let old_capacity = values.capacity();
            assert!(domain >= old_length, "attempt ID domain must not shrink");
            values.resize(domain, u32::MAX);
            DIRECTORY_FILLED.fetch_add((domain - old_length) as u64, Relaxed);
            let capacity = values.capacity();
            if capacity != old_capacity {
                if old_capacity == 0 { DIRECTORY_ALLOCATIONS.fetch_add(1, Relaxed); }
                else {
                    DIRECTORY_REALLOCATIONS.fetch_add(1, Relaxed);
                    DIRECTORY_OLD_LENGTH_ON_GROWTH.fetch_add(old_length as u64, Relaxed);
                }
                let live = DIRECTORY_LIVE_CAPACITY.fetch_add((capacity - old_capacity) as u64 * 4, Relaxed) + (capacity - old_capacity) as u64 * 4;
                DIRECTORY_PEAK_CAPACITY.fetch_max(live, Relaxed);
            }
        }
    }''')
    text += '\n' + Path('/tmp/bpe-attempt-scratch-directory-diag.rs').read_text()
    merge.write_text(text)
    main = runner/'src/main.rs'
    main.write_text('mod allocator;\n' + main.read_text())
    shutil.copyfile('/tmp/bpe-child-allocator.rs', runner/'src/allocator.rs')
    core = runner/'src/core.rs'
    text = core.read_text()
    needle = '    let train_seconds = begin.elapsed().as_secs_f64();'
    assert text.count(needle) == 1
    text = text.replace(needle, '    crate::allocator::report();\n' + needle)
    needle = '    let cpu = cpu_seconds()?;'
    # Only the training CPU sample, not the separate prepare-mode sample.
    assert text.count(needle) == 2
    start = text.index('pub fn run')
    text = text[:start] + text[start:].replace(needle, '    crate::allocator::ACTIVE.store(true, std::sync::atomic::Ordering::Relaxed);\n' + needle, 1)
    core.write_text(text)
    mod = work/rel/'mod.rs'
    text = mod.read_text()
    needle = '            merge::bench_report_prepare();'
    assert text.count(needle) == 1
    text = text.replace(needle, needle + '\n            merge::bench_report_directories();')
    mod.write_text(text)
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
