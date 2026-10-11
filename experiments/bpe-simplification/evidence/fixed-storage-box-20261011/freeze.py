from pathlib import Path
import hashlib,json,shutil,subprocess
r=Path(__file__).resolve().parent;p=r/'builds.json';builds=json.loads(p.read_text())
for arm,b in builds['builds'].items():
    w=Path(b['worktree'])
    files=sorted(str(f.relative_to(w)) for f in (w/'tokenizers/tk-train/src/trainers/bpe').glob('*') if f.is_file())
    files+=['tokenizers/tk-train/Cargo.toml','tokenizers/tk-train/Cargo.lock']
    files+=sorted(str(f.relative_to(w)) for f in (w/'experiments/bpe-simplification/runner/src').rglob('*.rs'))
    files+=['experiments/bpe-simplification/runner/Cargo.toml','experiments/bpe-simplification/runner/Cargo.lock']
    hashes={}
    for f in files:
        dst=r/'source'/arm/f;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(w/f,dst)
        hashes[f]=hashlib.sha256(dst.read_bytes()).hexdigest()
    b['source_hashes']=hashes
    b['profile']='release opt-level=3 fat LTO codegen-units=1; runner default features (trainer no-default-features); normal production hashers; system allocator; no trainer instrumentation'
    patch=subprocess.check_output(['git','diff','--','tokenizers/tk-train/src/trainers/bpe'],cwd=w)
    (r/(arm+'.diff')).write_bytes(patch)
    b['patch_sha256']=hashlib.sha256(patch).hexdigest()
    runner=subprocess.check_output(['git','diff','--','experiments/bpe-simplification/runner'],cwd=w)
    (r/(arm+'-runner.diff')).write_bytes(runner)
base=builds['builds']['baseline']['source_hashes'];boxed=builds['builds']['boxed']['source_hashes']
changes=[f for f in base if base[f]!=boxed[f]]
assert set(changes)=={f'tokenizers/tk-train/src/trainers/bpe/{n}.rs' for n in ('corpus','vocabulary','merge','index')},changes
assert (r/'baseline-runner.diff').read_bytes()==(r/'boxed-runner.diff').read_bytes()
builds['production_differences']=changes
p.write_text(json.dumps(builds,indent=2))
print('frozen sources; four production differences; identical runner and locks')
