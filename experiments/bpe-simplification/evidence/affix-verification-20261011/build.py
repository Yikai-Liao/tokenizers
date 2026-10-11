from pathlib import Path
import subprocess,os,shutil,json,time,hashlib
out=Path(__file__).resolve().parent
env=dict(os.environ,PATH='/root/.cargo/bin:'+os.environ['PATH'],CARGO_TARGET_DIR='/tmp/bpe-alphabet-build-20261011',CARGO_INCREMENTAL='0',CARGO_BUILD_JOBS='2')
records=[]
for arm in ['baseline','candidate']:
 root=Path('/tmp/bpe-alphabet-'+arm+'-20261011')
 cmd=['cargo','build','--offline','--locked','--release','--manifest-path','experiments/bpe-simplification/runner/Cargo.toml']
 start=time.monotonic()
 with (out/(arm+'-build.log')).open('w') as log:
  p=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
 record=dict(arm=arm,command=cmd,exit_code=p.returncode,seconds=time.monotonic()-start)
 records.append(record);(out/'builds.json').write_text(json.dumps(records,indent=2)+'\n')
 print(arm,'build',p.returncode,flush=True)
 if p.returncode: raise SystemExit(p.returncode)
 binary=out/(arm+'-bin');shutil.copyfile(Path(env['CARGO_TARGET_DIR'])/'release/bpe-bench-runner',binary);binary.chmod(0o755)
 record['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest();(out/'builds.json').write_text(json.dumps(records,indent=2)+'\n')
