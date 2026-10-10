from pathlib import Path
import hashlib, json, os, shutil, subprocess

ROOT=Path('/root/code/tokenizers-simplification-results')
OUT=ROOT/'online-initial'
WORK=Path('/root/code/tokenizers-workspaces/bpe-online-candidate-20261010')
REL=Path('tokenizers/tk-train/src/trainers/bpe/engine')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
env={k:v for k,v in os.environ.items() if not k.startswith(('BPE_','RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS'))}
env['PATH']='/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin'
env['CARGO_TARGET_DIR']=str(ROOT/'target')
command=['cargo','build','--release','--locked','--offline','--manifest-path',
         str(WORK/'experiments/bpe-simplification/runner/Cargo.toml'),'-j','2']
with (OUT/'validation/build-candidate.log').open('w') as f:
    subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
binary=OUT/'bin/bpe-bench-candidate'
shutil.copy2(ROOT/'target/release/bpe-bench-runner',binary)
patch_path=OUT/'candidate.patch'
patch_path.write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=WORK))
assert not subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=WORK)
manifest=json.loads((OUT/'manifest.json').read_text())
manifest['builds'].append(dict(arm='candidate',worktree=str(WORK),command=command,
    binary=str(binary),binary_sha256=sha(binary),patch=str(patch_path),patch_sha256=sha(patch_path),
    sources_sha256={str(p.relative_to(WORK)):sha(p) for p in sorted((WORK/REL).rglob('*.rs'))},
    runner_lock_sha256=sha(WORK/'experiments/bpe-simplification/runner/Cargo.lock'),
    runner_sources_sha256={str(p.relative_to(WORK)):sha(p) for p in sorted((WORK/'experiments/bpe-simplification/runner/src').rglob('*.rs'))}))
manifest['candidate_lines']=json.loads((OUT/'validation/candidate-lines.json').read_text())
manifest['measurement_script_sha256']=sha(OUT/'run_experiment.py')
inputs=[x for x in json.loads((ROOT/'final-scaling/inputs.json').read_text()) if x['case'] in ('zh-256MiB','zh-512MiB')]
for info in inputs:
    assert sha(Path(info['prepared_path']))==info['prepared_sha256']
    assert sha(Path(info['path']))==info['source_sha256']
manifest['inputs']=inputs
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('frozen candidate',sha(binary),flush=True)
