"""Build only after the user-directed diagnostic stop has completed."""
from pathlib import Path
import os, json, subprocess, shutil, time
import run_experiment as experiment

OUT = experiment.OUT
WORK = Path('/root/code/tokenizers-workspaces/bpe-online-full-phase-20261010')
REL = Path('tokenizers/tk-train/src/trainers/bpe/engine')
while Path('/proc/2500471').exists():
    time.sleep(2)
assert not experiment.busy(), experiment.busy()
env = {k:v for k,v in os.environ.items() if not k.startswith(('BPE_','RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS'))}
env['PATH'] = '/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin'
env['CARGO_TARGET_DIR'] = str(experiment.ROOT/'target')
command = ['cargo','build','--release','--locked','--offline','--manifest-path',
           str(WORK/'experiments/bpe-simplification/runner/Cargo.toml'),'-j','2']
print('Building full-phase prototype', flush=True)
with (OUT/'validation/build-full-phases.log').open('w') as log:
    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
binary = OUT/'bin/bpe-bench-full-candidate'
shutil.copy2(experiment.ROOT/'target/release/bpe-bench-runner', binary)
patch = OUT/'full-candidate.patch'
patch.write_bytes(subprocess.check_output(['git','diff','--binary'], cwd=WORK))
with patch.open('ab') as f:
    result = subprocess.run(['git','diff','--no-index','--binary','/dev/null',str(REL/'phase_timing.rs')],
                            cwd=WORK, capture_output=True)
    assert result.returncode == 1, result.stderr
    f.write(result.stdout)
old = json.loads((Path('/root/code/tokenizers-workspaces/bpe-simplification')/
                  'experiments/bpe-simplification/evidence/full-review-provenance.json').read_text())['arms']
arms = []
for name, source_name in (('main','main'),('baseline','final')):
    source = next(a for a in old if a['arm'] == source_name)
    assert experiment.sha(source['binary']) == source['binary_sha256']
    arms.append(dict(arm=name, **{k:v for k,v in source.items() if k != 'arm'}))
arms.append(dict(arm='candidate', base_commit='7e77262b', production_patch='candidate.patch',
    worktree=str(WORK), binary=str(binary), binary_sha256=experiment.sha(binary),
    patch=str(patch), patch_sha256=experiment.sha(patch),
    sources_sha256={str(p.relative_to(WORK)):experiment.sha(p) for p in sorted((WORK/REL).rglob('*.rs'))},
    runner_lock_sha256=experiment.sha(WORK/'experiments/bpe-simplification/runner/Cargo.lock'),
    runner_sources_sha256={str(p.relative_to(WORK)):experiment.sha(p) for p in sorted((WORK/'experiments/bpe-simplification/runner/src').rglob('*.rs'))}))
(OUT/'full-phase-manifest.json').write_text(json.dumps(dict(arms=arms, command=command,
    profile=dict(opt_level=3,lto='fat',codegen_units=1),workers=4,affinity=[0,1,2,3]),indent=2))
print('Frozen full-phase prototype '+experiment.sha(binary), flush=True)
