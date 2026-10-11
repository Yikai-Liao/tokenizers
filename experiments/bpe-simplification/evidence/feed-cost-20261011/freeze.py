import hashlib,json,shutil,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
arm=sys.argv[1]; work=Path('/tmp/bpe-feed-cost-'+arm)
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
binpath=root/arm; shutil.copy2(work/'experiments/bpe-simplification/runner/target/release/bpe-bench-runner' if arm == 'final' else Path('/tmp/bpe-feed-release/release/bpe-bench-runner'),binpath)
files=['tokenizers/tk-train/src/trainers/bpe/feed.rs','tokenizers/tk-train/src/trainers/bpe/tests.rs','experiments/bpe-simplification/runner/src/pipeline.rs','experiments/bpe-simplification/runner/src/feed_stats.rs','experiments/bpe-simplification/runner/src/protocol.rs','experiments/bpe-simplification/runner/Cargo.toml','experiments/bpe-simplification/runner/Cargo.lock']
hashes={}
for f in files:
 dst=root/'source'/arm/f;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(work/f,dst);hashes[f]=digest(dst)
(root/(arm+'.patch')).write_bytes(subprocess.check_output(['git','diff'],cwd=work))
p=root/'builds.json';data=json.loads(p.read_text()) if p.exists() else {'baseline_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=work,text=True).strip(),'builds':{}}
data['builds'][arm]={'binary':str(binpath),'binary_sha256':digest(binpath),'source_hashes':hashes,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=work,text=True).strip(),'worktree':str(work),'profile':'release opt-level=3 lto=fat codegen-units=1; same runner dependencies'}
p.write_text(json.dumps(data,indent=2)); print(arm,data['builds'][arm]['binary_sha256'])
