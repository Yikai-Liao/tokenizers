from pathlib import Path
import subprocess,os,time,json,shutil,hashlib
out=Path(__file__).resolve().parent;root=Path('/tmp/bpe-affix-slot-measurement-20261011')
env=dict(os.environ,PATH='/root/.cargo/bin:'+os.environ['PATH'],CARGO_TARGET_DIR='/tmp/bpe-alphabet-build-20261011',CARGO_INCREMENTAL='0',CARGO_BUILD_JOBS='2')
command=['cargo','build','--offline','--locked','--release','--manifest-path','experiments/bpe-simplification/runner/Cargo.toml']
start=time.monotonic()
with (out/'build.log').open('w') as log:p=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
assert p.returncode==0,p.returncode
binary=out/'candidate-bin';shutil.copyfile(Path(env['CARGO_TARGET_DIR'])/'release/bpe-bench-runner',binary);binary.chmod(0o755)
(out/'build.json').write_text(json.dumps(dict(command=command,exit_code=p.returncode,seconds=time.monotonic()-start,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()),indent=2)+'\n')
