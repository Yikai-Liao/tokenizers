from pathlib import Path
import hashlib,json,shutil,subprocess
r=Path(__file__).resolve().parent
source=Path('/home/lyk/code/tokenizers/experiments/bpe-simplification/evidence/feed-cost-20261011/production/source/drain')
repo=Path('/home/lyk/code/tokenizers')
builds={'baseline_commit':'2f4803a936619aa575ba4f6cb49fc660057ef5e4','builds':{}}
for arm in ('baseline','final'):
 w=Path('/tmp/bpe-next-'+arm+'-20261011')
 for name in ('pipeline.rs','feed_stats.rs','protocol.rs'):
  f=Path('experiments/bpe-simplification/runner/src')/name
  shutil.copy2(source/f,w/f)
 f=Path('experiments/bpe-simplification/runner/Cargo.lock');shutil.copy2(repo/f,w/f)
 files=sorted(str(p.relative_to(w)) for p in (w/'tokenizers/tk-train/src/trainers/bpe').glob('*.rs') if p.name not in ('parity_trainer.rs','word.rs'))
 files+=['tokenizers/tk-train/Cargo.toml','tokenizers/tk-train/Cargo.lock']
 files+=sorted(str(p.relative_to(w)) for p in (w/'experiments/bpe-simplification/runner/src').rglob('*.rs'))
 files+=['experiments/bpe-simplification/runner/Cargo.toml','experiments/bpe-simplification/runner/Cargo.lock']
 hashes={}
 for f in files:
  dst=r/'source'/arm/f;dst.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(w/f,dst);hashes[f]=hashlib.sha256(dst.read_bytes()).hexdigest()
 builds['builds'][arm]={'worktree':str(w),'binary':str(r/arm),'source_hashes':hashes,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=w,text=True).strip(),'profile':'release opt-level=3 fat LTO codegen-units=1; same runner/lock; default RandomState, no seed/phase instrumentation','binary_sha256':None}
(r/'builds.json').write_text(json.dumps(builds,indent=2))
inputs=json.loads((repo/'experiments/bpe-simplification/evidence/feed-cost-20261011/production/inputs.json').read_text())[:4]
(r/'inputs.json').write_text(json.dumps(inputs,indent=2))
shutil.copy2(repo/'experiments/bpe-simplification/evidence/feed-cost-20261011/production/measure.py',r/'measure.py')
