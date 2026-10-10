"""Freeze the selected production variant with the common ten phase timers."""
from pathlib import Path
import sys, json, os, shutil, subprocess
from run_full_phases_1g import OUT, ROOT, busy, sha

arm = sys.argv[1]
assert arm in ('ordered', 'lookup', 'combined')
assert not busy(), busy()
repo = Path('/root/code/tokenizers-workspaces/bpe-simplification')
clean = repo.parent / ('bpe-small-' + arm + '-20261010')
work = repo.parent / ('bpe-selected-full-' + arm + '-20261010')
rel = Path('tokenizers/tk-train/src/trainers/bpe/engine')
assert not work.exists()
subprocess.run(['git', 'worktree', 'add', '--detach', str(work), '7e77262b'], cwd=repo, check=True)
for p in (clean / rel).rglob('*.rs'):
    shutil.copy2(p, work / rel / p.relative_to(clean / rel))
instrumented = repo.parent / 'bpe-online-full-phase-20261010'
for name in ('mod.rs', 'phase_timing.rs'):
    shutil.copy2(instrumented / rel / name, work / rel / name)
runner = work / 'experiments/bpe-simplification/runner'
env = {k:v for k,v in os.environ.items() if not k.startswith(('BPE_', 'RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS'))}
env['PATH'] = '/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin'
env['CARGO_TARGET_DIR'] = str(ROOT / 'target')
command = ['cargo', 'build', '--release', '--locked', '--offline', '--manifest-path', str(runner / 'Cargo.toml'), '-j', '2']
log = OUT / 'validation' / ('build-selected-full-' + arm + '.log')
with log.open('w') as f:
    subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
binary = OUT / 'bin' / ('bpe-bench-selected-full-' + arm)
shutil.copy2(ROOT / 'target/release/bpe-bench-runner', binary)
patch = OUT / ('selected-full-' + arm + '.patch')
patch.write_bytes(subprocess.check_output(['git', 'diff', '--binary', '--', str(rel)], cwd=work))
with patch.open('ab') as f:
    r = subprocess.run(['git', 'diff', '--no-index', '--binary', '/dev/null', str(rel / 'phase_timing.rs')], cwd=work, capture_output=True)
    assert r.returncode == 1
    f.write(r.stdout)
manifest = json.loads((OUT / 'full-phase-manifest.json').read_text())
manifest['arms'] = [a for a in manifest['arms'] if a['arm'] != 'candidate']
for a in manifest['arms']:
    assert sha(a['binary']) == a['binary_sha256']
manifest['arms'].append(dict(arm='candidate', selected_variant=arm, base_commit='7e77262b', production_worktree=str(clean), production_patch=str(OUT / ('small-' + arm + '.patch')), production_patch_sha256=sha(OUT / ('small-' + arm + '.patch')), worktree=str(work), binary=str(binary), binary_sha256=sha(binary), patch=str(patch), patch_sha256=sha(patch), sources_sha256={str(p.relative_to(work)):sha(p) for p in sorted((work / rel).rglob('*.rs'))}, runner_lock_sha256=sha(runner / 'Cargo.lock'), runner_sources_sha256={str(p.relative_to(work)):sha(p) for p in sorted((runner / 'src').rglob('*.rs'))}))
manifest.update(command=command, log=str(log), purpose='selected variant: ByteLevel / Whitespace contrast')
(OUT / 'selected-full-manifest.json').write_text(json.dumps(manifest, indent=2))
print('Frozen selected full-phase binary', arm, sha(binary), flush=True)
