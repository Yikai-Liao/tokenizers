from pathlib import Path
import os,subprocess,time,json
root=Path('/tmp/bpe-affix-slot-20261011');out=Path(__file__).resolve().parent
env=dict(os.environ,PATH='/root/.cargo/bin:'+os.environ['PATH'],CARGO_TARGET_DIR='/tmp/bpe-selected-rules-checks-20261011',CARGO_INCREMENTAL='0')
checks=[]
for name,args in [('test-default',['test','--locked','--manifest-path','tokenizers/tk-train/Cargo.toml','trainers::bpe','--lib']),('test-nodefault',['test','--locked','--manifest-path','tokenizers/tk-train/Cargo.toml','--no-default-features','trainers::bpe','--lib']),('clippy',['clippy','--locked','--manifest-path','tokenizers/tk-train/Cargo.toml','--all-targets','--no-default-features','--','-D','warnings'])]:
 start=time.monotonic()
 with (out/(name+'.log')).open('w') as log:
  result=subprocess.run(['cargo',*args],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
 checks.append(dict(name=name,command=['cargo',*args],exit_code=result.returncode,seconds=time.monotonic()-start))
 (out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
 print(name,result.returncode,flush=True)
 if result.returncode not in ([0,101] if name.startswith('test') else [0]):raise SystemExit(result.returncode)
