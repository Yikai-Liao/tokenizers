from pathlib import Path
import os,json,subprocess,shutil
from run_full_phases_1g import OUT,ROOT,busy,sha
REL=Path('tokenizers/tk-train/src/trainers/bpe/engine')
assert not busy(),busy()
env={k:v for k,v in os.environ.items() if not k.startswith(('BPE_','RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS'))}
env['PATH']='/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin'
env['CARGO_TARGET_DIR']=str(ROOT/'target')
arms=[]
for arm,base in [('main','e4f787dc189d9be7192107490d652096cde7480e'),('candidate','7e77262b')]:
 work=Path('/root/code/tokenizers-workspaces')/('bpe-prepare-detail-'+arm+'-20261010')
 runner=work/'experiments/bpe-simplification/runner'
 if not runner.exists():
  shutil.copytree(Path('/root/code/tokenizers-workspaces/bpe-online-full-phase-20261010/experiments/bpe-simplification/runner'),runner,ignore=shutil.ignore_patterns('target','__pycache__'))
 command=['cargo','build','--release','--locked','--offline','--manifest-path',str(runner/'Cargo.toml'),'-j','2']
 print('Building detailed '+arm,flush=True)
 with (OUT/'validation'/('build-prepare-detail-'+arm+'.log')).open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 binary=OUT/'bin'/('bpe-bench-prepare-detail-'+arm)
 shutil.copy2(ROOT/'target/release/bpe-bench-runner',binary)
 patch=OUT/('prepare-detail-'+arm+'.patch')
 patch.write_bytes(subprocess.check_output(['git','diff','--binary','--',str(REL)],cwd=work))
 with patch.open('ab') as f:
  for p in sorted((work/REL).rglob('*.rs')):
   tracked=subprocess.run(['git','ls-files','--error-unmatch',str(p.relative_to(work))],cwd=work,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
   if not tracked:
    r=subprocess.run(['git','diff','--no-index','--binary','/dev/null',str(p.relative_to(work))],cwd=work,capture_output=True)
    assert r.returncode==1,r.stderr;f.write(r.stdout)
 arms.append(dict(arm=arm,base_commit=base,worktree=str(work),command=command,binary=str(binary),binary_sha256=sha(binary),patch=str(patch),patch_sha256=sha(patch),sources_sha256={str(p.relative_to(work)):sha(p) for p in sorted((work/REL).rglob('*.rs'))},runner_lock_sha256=sha(runner/'Cargo.lock'),runner_sources_sha256={str(p.relative_to(work)):sha(p) for p in sorted((runner/'src').rglob('*.rs'))}))
 print('Frozen detailed '+arm+' '+sha(binary),flush=True)
(OUT/'prepare-detail-manifest.json').write_text(json.dumps(dict(arms=arms,profile=dict(opt_level=3,lto='fat',codegen_units=1),workers=4,affinity=[0,1,2,3]),indent=2))
print('Detailed binaries frozen',flush=True)
