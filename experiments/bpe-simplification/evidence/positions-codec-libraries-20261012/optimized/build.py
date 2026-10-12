from pathlib import Path
import subprocess,os,json,hashlib,shutil
r=Path(__file__).resolve().parent;old=Path('/tmp/bpe-positions-encapsulation-final-results-20261012');scratch=Path('/tmp/bpe-positions-encapsulation-measurement-20261012');rel=Path('tokenizers/tk-train/src/trainers/bpe');env=os.environ|{'PATH':'/root/.cargo/bin:'+os.environ['PATH'],'CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'1'}
arms={'fmt-zero-push': '/tmp/bpe-positions-codec-fmt-zero-push-20261012', 'fmt-tls-push': '/tmp/bpe-positions-codec-fmt-tls-push-20261012', 'gimli-inline': '/tmp/bpe-positions-codec-gimli-inline-20261012'}
arms={k:Path(v) for k,v in arms.items()}
for name,source in arms.items():
 # Entire bpe and runner are restored to the frozen normal single-inline A.
 shutil.rmtree(scratch/rel);shutil.copytree(old/'source/a'/rel,scratch/rel)
 shutil.copyfile(source/rel/'positions/codec.rs',scratch/rel/'positions/codec.rs')
 runner=scratch/'experiments/bpe-simplification/runner';shutil.rmtree(runner);shutil.copytree(old/'source/a/experiments/bpe-simplification/runner',runner)
 manifest=(Path('/root/code/tokenizers-workspaces/bpe-positions-encapsulation-20261012')/'tokenizers/tk-train/Cargo.toml').read_text();manifest=manifest.replace('features = ["union"]','features = ["union", "write"]')
 manifest=manifest.replace('[features]','leb128 = "=0.2.7"\nleb128fmt = { version = "=0.1.0", default-features = false }\n\n[features]')
 (scratch/'tokenizers/tk-train/Cargo.toml').write_text(manifest)
 # Resolve just the two new packages using the frozen lockfile, then freeze.
 resolve=['cargo','generate-lockfile','--offline'];
 # cargo build without locked retains every already-locked dependency.
 cmd=['cargo','build','--release','--offline','--target-dir','/tmp/bpe-positions-encapsulation-release-20261012']
 with (r/(name+'-build.log')).open('w') as log:subprocess.run(cmd,cwd=runner,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 binary=r/(name+'-bin');shutil.copyfile('/tmp/bpe-positions-encapsulation-release-20261012/release/bpe-bench-runner',binary);binary.chmod(0o755)
 snapshot=r/'source'/name;shutil.copytree(scratch/rel,snapshot/rel);shutil.copytree(runner,snapshot/'experiments/bpe-simplification/runner');shutil.copyfile(scratch/'tokenizers/tk-train/Cargo.toml',snapshot/'tokenizers/tk-train/Cargo.toml')
 (r/(name+'-build.json')).write_text(json.dumps(dict(production_commit='5d617efe' if name=='hand' else subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),benchmark_note='Same frozen coarse hooks, SmallVec write feature and both library dependencies enabled for all arms; normal allocator; no diagnostic counters',command=cmd,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),rustc=subprocess.check_output(['rustc','-Vv'],env=env,text=True),source_sha256={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.rglob('*') if p.is_file()}),indent=2)+'\n');print(name,'built',flush=True)
