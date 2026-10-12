from pathlib import Path
import subprocess, shutil, hashlib, os, json, sys, re

root = Path('/root/code/tokenizers-workspaces/bpe-attempt-scratch-results-20261012')
root.mkdir(exist_ok=True)
work = Path('/tmp/bpe-selected-rules-measurement-20261011')
rel = Path('tokenizers/tk-train/src/trainers/bpe')
baseline = Path('/tmp/bpe-prefetch-right-20261012')
candidate = Path('/tmp/bpe-attempt-scratch-20261012')
frozen = Path('/root/code/tokenizers-workspaces/bpe-prefetch-right-results-20261012/source/baseline')
target = Path('/root/code/tokenizers-build/bpe-affix-core-build-20261011')
env = os.environ | {'PATH': '/root/.cargo/bin:' + os.environ['PATH'], 'CARGO_INCREMENTAL': '0', 'CARGO_BUILD_JOBS': '2'}

for arm in (sys.argv[1:] or ['baseline', 'scratch']):
    assert arm in ('baseline', 'scratch')
    snapshot = root/'source'/arm
    assert not snapshot.exists(), ('frozen snapshot already exists', snapshot)
    source = baseline if arm == 'baseline' else candidate
    shutil.copytree(frozen/'experiments/bpe-simplification/runner', work/'experiments/bpe-simplification/runner', dirs_exist_ok=True)
    shutil.copyfile(source/'tokenizers/tk-train/Cargo.toml', work/'tokenizers/tk-train/Cargo.toml')
    for path in (source/rel).glob('*.rs'):
        shutil.copyfile(path, work/rel/path.name)
    shutil.copyfile(frozen/rel/'mod.rs', work/rel/'mod.rs')
    shutil.copyfile(frozen/rel/'corpus.rs', work/rel/'corpus.rs')
    if arm == 'baseline':
        shutil.copyfile(frozen/rel/'merge.rs', work/rel/'merge.rs')
    else:
        mod = work/rel/'mod.rs'
        text = mod.read_text()
        needle = '            corpus = plan.materialize();'
        assert text.count(needle) == 1
        text = text.replace(needle, needle + '\n            let directories = merge::Scratch::default();')
        needle = 'self.merge_next(&mut vocabulary, &mut corpus, &mut index)?'
        assert text.count(needle) == 1
        text = text.replace(needle, 'self.merge_next(&mut vocabulary, &mut corpus, &mut index, &directories)?')
        needle = '            index.retire();'
        assert text.count(needle) == 2
        text = text.replace(needle, '            drop(directories);\n' + needle, 1)
        mod.write_text(text)
        merge = work/rel/'merge.rs'
        text = merge.read_text()
        pattern = r'        let jobs = batch\.prepare\([\s\S]*?\)\?;'
        matches = re.findall(pattern, text)
        assert len(matches) == 1
        text = text.replace(matches[0], '        let prepare_reuse = batch.reuse;\n        let prepare_started = std::time::Instant::now();\n' + matches[0] + '\n        BENCH_PREPARE_NS.fetch_add(prepare_started.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);\n        if prepare_reuse { &BENCH_REUSE_BATCHES } else { &BENCH_FRESH_BATCHES }.fetch_add(1, std::sync::atomic::Ordering::Relaxed);')
        baseline_merge = (frozen/rel/'merge.rs').read_text()
        text += '\n' + baseline_merge[baseline_merge.index('static BENCH_PREPARE_NS:'):]
        merge.write_text(text)
    runner = work/'experiments/bpe-simplification/runner'
    shutil.copyfile(frozen/'experiments/bpe-simplification/runner/Cargo.lock', runner/'Cargo.lock')
    with (root/f'{arm}-resolve.log').open('w') as log:
        subprocess.run(['cargo', 'metadata', '--offline', '--format-version', '1'], cwd=runner, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
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
