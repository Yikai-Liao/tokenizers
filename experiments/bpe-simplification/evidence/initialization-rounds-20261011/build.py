import hashlib,json,os,shutil,subprocess
from pathlib import Path
r=Path(__file__).resolve().parent;p=r/'builds.json';builds=json.loads(p.read_text())
for arm,b in builds['builds'].items():
 for f,h in b['source_hashes'].items():assert hashlib.sha256((Path(b['worktree'])/f).read_bytes()).hexdigest()==h
 cmd=['cargo','build','--release','--locked','--offline','--manifest-path','experiments/bpe-simplification/runner/Cargo.toml']
 env=dict(os.environ,CARGO_TARGET_DIR='/tmp/bpe-feed-release');b['command']=cmd;b['environment']={'CARGO_TARGET_DIR':env['CARGO_TARGET_DIR'],'RUSTFLAGS':env.get('RUSTFLAGS','')}
 print('building',arm,flush=True)
 with (r/(arm+'-build.log')).open('w') as stream:subprocess.run(cmd,cwd=b['worktree'],env=env,stdout=stream,stderr=stream,check=True)
 shutil.copy2(Path(env['CARGO_TARGET_DIR'])/'release/bpe-bench-runner',b['binary']);b['binary_sha256']=hashlib.sha256(Path(b['binary']).read_bytes()).hexdigest();p.write_text(json.dumps(builds,indent=2));print('frozen',arm,b['binary_sha256'],flush=True)
common=[f for f in builds['builds']['baseline']['source_hashes'] if f.startswith('experiments/') or f.endswith(('Cargo.lock','Cargo.toml')) or f.endswith(('/feed.rs','/positions.rs'))]
for f in common:assert builds['builds']['baseline']['source_hashes'][f]==builds['builds']['final']['source_hashes'][f],f
print('common source hashes match',len(common),flush=True)
