from pathlib import Path
import subprocess, shutil, json, hashlib, os

root = Path('/root/code/tokenizers-workspaces/bpe-affix-core-results-20261011')
root.mkdir(exist_ok=True)
measurement = Path('/tmp/bpe-selected-rules-measurement-20261011')
base = Path('/tmp/bpe-test-compaction-20261011')
relative = Path('tokenizers/tk-train/src/trainers/bpe')
# Keep identical diagnostic timers in mod.rs; all other modules use current source.
for source in (base / relative).glob('*.rs'):
    if source.name != 'mod.rs':
        shutil.copy2(source, measurement / relative / source.name)
shutil.copy2('/root/code/tokenizers-workspaces/bpe-selected-rules-results-20261011/inputs.json', root / 'inputs.json')
arms = {'baseline': base, 'cache': Path('/tmp/bpe-affix-cache-20261011'), 'split': Path('/tmp/bpe-affix-split-20261011')}
for arm, source in arms.items():
    # Refresh mtime so Cargo cannot reuse an arm built from a newer source file.
    shutil.copyfile(source / relative / 'vocabulary.rs', measurement / relative / 'vocabulary.rs')
    snapshot = root / 'source' / arm
    shutil.copytree(measurement / relative, snapshot / relative, dirs_exist_ok=True)
    shutil.copytree(measurement / 'experiments/bpe-simplification/runner', snapshot / 'experiments/bpe-simplification/runner', dirs_exist_ok=True)
    for name in ['Cargo.lock', 'Cargo.toml']:
        shutil.copy2(measurement / 'tokenizers' / name, snapshot / 'tokenizers' / name)
    cmd = ['cargo', 'build', '--release', '--locked', '--offline', '--target-dir', '/root/code/tokenizers-build/bpe-affix-core-build-20261011']
    (root / (arm + '-build-command.json')).write_text(json.dumps(cmd))
    environment = os.environ | {'PATH': '/root/.cargo/bin:' + os.environ['PATH'], 'CARGO_INCREMENTAL': '0'}
    with (root / (arm + '-build.log')).open('w') as log:
        completed = subprocess.run(cmd, cwd=measurement / 'experiments/bpe-simplification/runner', env=environment, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError((arm, (root / (arm + '-build.log')).read_text()))
    binary = root / (arm + '-bin')
    shutil.copy2('/root/code/tokenizers-build/bpe-affix-core-build-20261011/release/bpe-bench-runner', binary)
    print(arm, 'built', hashlib.sha256(binary.read_bytes()).hexdigest(), flush=True)
