from pathlib import Path
import os,subprocess,json,shutil
from run_full_phases_1g import OUT,ROOT,busy,sha
from prepare_small_optimizations import ordered,lookup,REL,REPO
assert not busy(),busy()
env={k:v for k,v in os.environ.items() if not k.startswith(('BPE_','RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','MIRIFLAGS'))}
env['PATH']='/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin';env['CARGO_TARGET_DIR']=str(ROOT/'target')
validation=[]
def run(command,label,environment=env):
 print(label,flush=True)
 with (OUT/'validation'/(label+'.log')).open('w') as log:subprocess.run(command,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
 validation.append(dict(label=label,command=command,log=str(OUT/'validation'/(label+'.log'))))
for arm in ('ordered','lookup','combined'):
 clean=Path('/root/code/tokenizers-workspaces')/('bpe-small-'+arm+'-20261010')
 shutil.copy2(Path('/root/code/tokenizers-workspaces/bpe-online-candidate-20261010/tokenizers/tk-train/Cargo.lock'),clean/'tokenizers/tk-train/Cargo.lock')
 run(['cargo','test','--locked','--offline','--manifest-path',str(clean/'tokenizers/tk-train/Cargo.toml'),'--no-default-features','-j','2'],'small-'+arm+'-native')
clean=Path('/root/code/tokenizers-workspaces/bpe-small-combined-20261010')
run(['cargo','test','--locked','--offline','--manifest-path',str(clean/'tokenizers/tk-train/Cargo.toml'),'trainers::bpe','-j','2'],'small-combined-default')
run(['cargo','test','--locked','--offline','--manifest-path',str(clean/'tokenizers/tk-train/Cargo.toml'),'--no-default-features','--doc','-j','2'],'small-combined-doc')
run(['cargo','clippy','--locked','--offline','--manifest-path',str(clean/'tokenizers/tk-train/Cargo.toml'),'--no-default-features','--all-targets','-j','2','--','-D','warnings'],'small-combined-clippy')
miri=env.copy();miri['MIRIFLAGS']='-Zmiri-strict-provenance'
run(['cargo','+nightly','miri','test','--locked','--offline','--manifest-path',str(clean/'experiments/bpe-simplification/miri-codec/Cargo.toml')],'small-combined-miri',miri)
arms=[]
base=next(a for a in json.loads((OUT/'prepare-detail-manifest.json').read_text())['arms'] if a['arm']=='candidate')
arms.append(dict(**{k:v for k,v in base.items() if k!='arm'},arm='control'))
origin=Path('/root/code/tokenizers-workspaces/bpe-prepare-detail-candidate-20261010')
for arm in ('ordered','lookup','combined'):
 work=Path('/root/code/tokenizers-workspaces')/('bpe-small-detail-'+arm+'-20261010')
 assert not work.exists(),work
 subprocess.run(['git','worktree','add','--detach',str(work),'7e77262b'],cwd=REPO,check=True,stdout=subprocess.DEVNULL)
 for p in (origin/REL).rglob('*.rs'):shutil.copy2(p,work/REL/p.relative_to(origin/REL))
 if arm in ('ordered','combined'):ordered(work)
 if arm in ('lookup','combined'):
  lookup(work)
  p=work/REL/'merge.rs';s=p.read_text();assert s.count('        drop(_selected);')==1;s=s.replace('        drop(_selected);\n','');s=s.replace('        let aa = self.rules[0].pair.0 == self.rules[0].pair.1;','        drop(_selected);\n        let aa = self.rules[0].pair.0 == self.rules[0].pair.1;');p.write_text(s)
 runner=work/'experiments/bpe-simplification/runner'
 command=['cargo','build','--release','--locked','--offline','--manifest-path',str(runner/'Cargo.toml'),'-j','2']
 run(command,'build-small-detail-'+arm)
 binary=OUT/'bin'/('bpe-bench-small-detail-'+arm);shutil.copy2(ROOT/'target/release/bpe-bench-runner',binary)
 patch=OUT/('small-detail-'+arm+'.patch');patch.write_bytes(subprocess.check_output(['git','diff','--binary','--',str(REL)],cwd=work))
 with patch.open('ab') as f:
  for p in sorted((work/REL).rglob('*.rs')):
   if subprocess.run(['git','ls-files','--error-unmatch',str(p.relative_to(work))],cwd=work,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
    r=subprocess.run(['git','diff','--no-index','--binary','/dev/null',str(p.relative_to(work))],cwd=work,capture_output=True);assert r.returncode==1;f.write(r.stdout)
 arms.append(dict(arm=arm,base_commit='7e77262b',worktree=str(work),production_worktree='/root/code/tokenizers-workspaces/bpe-small-'+arm+'-20261010',command=command,binary=str(binary),binary_sha256=sha(binary),patch=str(patch),patch_sha256=sha(patch),production_patch=str(OUT/('small-'+arm+'.patch')),production_patch_sha256=sha(OUT/('small-'+arm+'.patch')),sources_sha256={str(p.relative_to(work)):sha(p) for p in sorted((work/REL).rglob('*.rs'))},runner_lock_sha256=sha(runner/'Cargo.lock'),runner_sources_sha256={str(p.relative_to(work)):sha(p) for p in sorted((runner/'src').rglob('*.rs'))}))
 print('Frozen '+arm+' '+sha(binary),flush=True)
(OUT/'small-ablation-manifest.json').write_text(json.dumps(dict(arms=arms,validation=validation,profile=dict(opt_level=3,lto='fat',codegen_units=1),workers=4,affinity=[0,1,2,3]),indent=2))
print('Small ablation binaries frozen',flush=True)
