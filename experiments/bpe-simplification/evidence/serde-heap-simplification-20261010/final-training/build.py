from pathlib import Path
import hashlib,json,os,shutil,subprocess,time
work=Path('/root/code/tokenizers-workspaces/bpe-heap-experiment')
data=json.loads(Path('/tmp/bpe-serde-peekmut-builds.json').read_text());data['builds'].pop('candidate')
shutil.copy2('/tmp/bpe-serde-peekmut-baseline-bin','/tmp/bpe-cleanup-baseline-bin')
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=work,text=True).strip()
paths=[p for p in (work/'tokenizers/tk-train/src/trainers/bpe').glob('*.rs')]+[work/'experiments/bpe-simplification/runner/Cargo.lock',work/'experiments/bpe-simplification/runner/Cargo.toml',work/'tokenizers/tk-train/Cargo.toml']+list((work/'experiments/bpe-simplification/runner/src').rglob('*.rs'))
hashes={str(p.relative_to(work)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
command=['/root/.cargo/bin/cargo','build','--locked','--release','--manifest-path','experiments/bpe-simplification/runner/Cargo.toml']
env=os.environ.copy();env['PATH']='/root/.cargo/bin:'+env['PATH'];env['CARGO_INCREMENTAL']='0';env['CARGO_TARGET_DIR']='/tmp/bpe-global-release-target'
start=time.monotonic()
with open('/tmp/bpe-cleanup-candidate-build.log','w') as log: subprocess.run(command,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
binary=Path('/tmp/bpe-cleanup-candidate-bin');shutil.copy2('/tmp/bpe-global-release-target/release/bpe-bench-runner',binary)
data['builds']['candidate']=dict(commit=head,source_hashes=hashes,command=command,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),build_seconds=time.monotonic()-start)
data['builds']['candidate']['environment']={k:env.get(k) for k in ['RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','CARGO_INCREMENTAL','CARGO_TARGET_DIR']}
Path('/tmp/bpe-cleanup-builds.json').write_text(json.dumps(data,indent=2))
s=Path('/tmp/bpe-serde-peekmut-measure.py').read_text().replace('bpe-serde-peekmut','bpe-cleanup').replace('f4ed232493d0cabf5a8bb15c06afb4d1e7faa07b',head).replace('Compare pre-change and Serde/PeekMut production builds.','Compare pre-change d4e3fa45 and final simplified production builds.')
Path('/tmp/bpe-cleanup-measure.py').write_text(s)
print('Final binary ready',head,'seconds',round(time.monotonic()-start,1))
