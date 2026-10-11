"""Freeze independent candidates against the actual 303597a2 baseline."""
from pathlib import Path
import hashlib,json,shutil,subprocess
r=Path(__file__).resolve().parent
repo=Path('/home/lyk/code/tokenizers')
base='303597a24e013c644209df71b6d82fd9a649c8a2'
builds={'baseline_commit':base,'builds':{}}
inputs=json.loads((repo/'experiments/bpe-simplification/evidence/initialization-rounds-20261011/inputs.json').read_text())
(r/'inputs.json').write_text(json.dumps(inputs,indent=2))
for arm in ('baseline','directories','insert','combined'):
 w=Path('/tmp/bpe-neighbor-worktree-'+arm+'-20261011')
 subprocess.run(['git','worktree','add','--detach',str(w),base],cwd=repo,check=True)
 changed=[]
 if arm in ('directories','combined'):changed.append('merge')
 if arm in ('insert','combined'):changed.append('vocabulary')
 for name in changed:
  f=Path('tokenizers/tk-train/src/trainers/bpe')/(name+'.rs');shutil.copy2(repo/f,w/f)
 for f in ('tokenizers/tk-train/Cargo.lock','experiments/bpe-simplification/runner/Cargo.lock'):
  shutil.copy2(repo/f,w/f)
 # Identical runner instrumentation outside the production trainer. Fault counters
 # are read before/after the public train timer; no timers enter merge hot paths.
 protocol=Path('experiments/bpe-simplification/runner/src/protocol.rs')
 rich=repo/'experiments/bpe-simplification/evidence/feed-cost-20261011/production/source/drain'/protocol
 shutil.copy2(rich,w/protocol)
 core=w/'experiments/bpe-simplification/runner/src/core.rs'
 s=core.read_text().replace('    let begin = Instant::now();\n    let cpu = cpu_seconds()?;\n    let (vocab, merges, special)', '    let initial_faults = crate::protocol::faults()?;\n    let begin = Instant::now();\n    let cpu = cpu_seconds()?;\n    let (vocab, merges, special)')
 s=s.replace('    let process_hwm_kib_before_validation = hwm()?;', '    let final_faults = crate::protocol::faults()?;\n    let train_major_faults = final_faults.1 - initial_faults.1;\n    let process_hwm_kib_before_validation = hwm()?;')
 s=s.replace('"load_seconds":load_seconds,"train_seconds":train_seconds,','"load_seconds":load_seconds,"train_seconds":train_seconds,"train_major_faults":train_major_faults,')
 assert 'train_major_faults' in s;core.write_text(s)
 files=sorted(str(p.relative_to(w)) for p in (w/'tokenizers/tk-train/src/trainers/bpe').glob('*.rs') if p.name not in ('parity_trainer.rs','word.rs'))
 files+=['tokenizers/tk-train/Cargo.toml','tokenizers/tk-train/Cargo.lock']
 files+=sorted(str(p.relative_to(w)) for p in (w/'experiments/bpe-simplification/runner/src').rglob('*.rs'))
 files+=['experiments/bpe-simplification/runner/Cargo.toml','experiments/bpe-simplification/runner/Cargo.lock']
 hashes={}
 for f in files:
  dst=r/'source'/arm/f;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(w/f,dst)
  hashes[f]=hashlib.sha256(dst.read_bytes()).hexdigest()
 patch=subprocess.check_output(['git','diff','--','tokenizers/tk-train/src/trainers/bpe'],cwd=w)
 (r/(arm+'.diff')).write_bytes(patch)
 builds['builds'][arm]={'worktree':str(w),'binary':str(r/arm),'source_hashes':hashes,'source_commit':base,'candidate_files':changed,'patch_sha256':hashlib.sha256(patch).hexdigest(),'binary_sha256':None,'profile':'release opt-level=3 fat LTO codegen-units=1; public do_train; fixed caller-map seeds only, production default internal hashers; no phase timers'}
(r/'builds.json').write_text(json.dumps(builds,indent=2))
# Verify isolation: all files outside the selected candidates are identical.
baseline=builds['builds']['baseline']['source_hashes']
for arm,b in builds['builds'].items():
 allowed={f'tokenizers/tk-train/src/trainers/bpe/{name}.rs' for name in b['candidate_files']}
 for f,h in b['source_hashes'].items():assert f in allowed or baseline[f]==h,(arm,f)
print('four isolated arms frozen',flush=True)
