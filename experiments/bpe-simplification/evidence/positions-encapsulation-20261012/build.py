from pathlib import Path
import subprocess,os,json,hashlib,shutil,difflib,sys
r=Path(__file__).resolve().parent
old=Path('/root/code/tokenizers-workspaces/bpe-positions-simd-results-20261012')
w=Path('/tmp/bpe-positions-encapsulation-measurement-20261012')
rel=Path('tokenizers/tk-train/src/trainers/bpe');a=Path('/root/code/tokenizers-workspaces/bpe-positions-encapsulation-20261012');b=Path('/tmp/bpe-positions-two-inline-20261012')
env=os.environ|{'PATH':'/root/.cargo/bin:'+os.environ['PATH'],'CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'1'}
# Reuse exactly the former baseline's normal runner and coarse timing hooks.
patch=''
for f in (old/'source/a'/rel).glob('*.rs'):
 original=subprocess.check_output(['git','show','3436bba7:'+str(rel/f.name)],cwd=a,text=True)
 patch+=''.join(difflib.unified_diff(original.splitlines(True),f.read_text().splitlines(True),fromfile='a/'+str(rel/f.name),tofile='b/'+str(rel/f.name)))
assert 'positions.rs' not in patch
(r/'benchmark-hooks.patch').write_text(patch)
for name,source in [('base',None),('a',a),('b',b)]:
 if sys.argv[1:] and name not in sys.argv[1:]:continue
 # Restore baseline first; mutations are confined to the task's scratch worktree.
 subprocess.run(['git','restore','--source=3436bba7','--',str(rel)],cwd=w,check=True)
 if (w/rel/'positions').exists():shutil.rmtree(w/rel/'positions')
 subprocess.run(['git','apply','-'],cwd=w,input=patch,text=True,check=True)
 if source:
  shutil.copyfile(source/rel/'positions.rs',w/rel/'positions.rs')
  shutil.copytree(source/rel/'positions',w/rel/'positions')
  shutil.copyfile(source/rel/'merge.rs',w/rel/'merge.rs')
  merge_patch=''
  for f in [old/'source/a'/rel/'merge.rs']:
   original=subprocess.check_output(['git','show','3436bba7:'+str(rel/f.name)],cwd=a,text=True)
   merge_patch+=''.join(difflib.unified_diff(original.splitlines(True),f.read_text().splitlines(True),fromfile='a/'+str(rel/f.name),tofile='b/'+str(rel/f.name)))
  subprocess.run(['git','apply','-'],cwd=w,input=merge_patch,text=True,check=True)
 runner=w/'experiments/bpe-simplification/runner'
 shutil.copytree(old/'source/a/experiments/bpe-simplification/runner',runner,dirs_exist_ok=True)
 shutil.copyfile(old/'source/a/tokenizers/tk-train/Cargo.toml',w/'tokenizers/tk-train/Cargo.toml')
 cmd=['cargo','build','--release','--locked','--offline','--target-dir','/tmp/bpe-positions-encapsulation-release-20261012']
 with (r/(name+'-build.log')).open('w') as log:subprocess.run(cmd,cwd=runner,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 binary=r/(name+'-bin');shutil.copyfile('/tmp/bpe-positions-encapsulation-release-20261012/release/bpe-bench-runner',binary);binary.chmod(0o755)
 snapshot=r/'source'/name;shutil.copytree(w/rel,snapshot/rel,dirs_exist_ok=True);shutil.copytree(runner,snapshot/'experiments/bpe-simplification/runner',dirs_exist_ok=True)
 metadata=dict(commit='3436bba7' if source is None else subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),command=cmd,environment={'CARGO_BUILD_JOBS':'1','CARGO_INCREMENTAL':'0'},rustc=subprocess.check_output(['rustc','-Vv'],env=env,text=True),source_sha256={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.rglob('*') if p.is_file()})
 (r/(name+'-build.json')).write_text(json.dumps(metadata,indent=2)+'\n');print(name,'built',flush=True)
