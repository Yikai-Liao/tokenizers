from pathlib import Path
import subprocess,os,shutil,json,time,hashlib
out=Path(__file__).resolve().parent
root=Path('/tmp/bpe-flags-only-20261011')
env=dict(os.environ,PATH='/root/.cargo/bin:'+os.environ['PATH'],CARGO_TARGET_DIR='/tmp/bpe-alphabet-build-20261011',CARGO_INCREMENTAL='0',CARGO_BUILD_JOBS='2')
cmd=['cargo','build','--offline','--locked','--release','--manifest-path','experiments/bpe-simplification/runner/Cargo.toml']
start=time.monotonic()
with (out/'flags-only-build.log').open('w') as log:p=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
assert p.returncode==0
binary=out/'flags-only-bin';shutil.copyfile(Path(env['CARGO_TARGET_DIR'])/'release/bpe-bench-runner',binary);binary.chmod(0o755)
records=json.loads((out/'builds.json').read_text());records.append(dict(arm='flags-only',command=cmd,exit_code=0,seconds=time.monotonic()-start,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()));(out/'builds.json').write_text(json.dumps(records,indent=2)+'\n')
print('flags-only build complete',flush=True)
