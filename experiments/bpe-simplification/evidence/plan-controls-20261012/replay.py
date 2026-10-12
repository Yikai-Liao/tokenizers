#!/usr/bin/env python3
"""Build one frozen benchmark arm over the recorded clean base and run one job."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess
p=argparse.ArgumentParser();p.add_argument('--checkout',type=Path,required=True);p.add_argument('--stage',choices=['scratch','publisher','frontier'],required=True);p.add_argument('--arm',required=True);p.add_argument('--job',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--target-dir',type=Path,required=True);a=p.parse_args()
e=Path(__file__).resolve().parent;env=json.loads((e/'environment.json').read_text());w=a.checkout.resolve()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=w,text=True).strip()==env['base_sha']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=w,text=True).strip(), 'Use a clean disposable checkout'
assert (e/a.stage/'source'/a.arm).is_dir(), 'Unknown frozen arm';job=json.loads(a.job.read_text());h=hashlib.sha256()
with a.input.open('rb') as f:
 for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
input_key = job['input_id'] if job['input_id'] in ['aa','reuse'] else job['input_id'].removesuffix('256') + '-' + job['pretokenizer']
assert h.hexdigest() == env['inputs'][input_key]['sha256'], 'Prepared input differs from the recorded case input'
for f in (e/a.stage/'source'/a.arm).rglob('*'):
 if f.is_file():
  dest=w/f.relative_to(e/a.stage/'source'/a.arm);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
runner=w/'experiments/bpe-simplification/runner'
subprocess.run(['cargo','build','--release','--locked','--offline','--target-dir',str(a.target_dir.resolve())],cwd=runner,env=os.environ|{'CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'2'},check=True)
a.output.mkdir(parents=True,exist_ok=True)
job.update(input=str(a.input.resolve()),output=str((a.output/'model.json').resolve()),build_id=a.arm,attempt_id=a.output.name)
jobpath=a.output/'job.json';jobpath.write_text(json.dumps(job,indent=2)+'\n')
with (a.output/'stdout.log').open('w') as out,(a.output/'stderr.log').open('w') as err:
 subprocess.run(['taskset','-c',env['affinity'],str(a.target_dir.resolve()/'release/bpe-bench-runner'),str(jobpath.resolve())],cwd=runner,stdout=out,stderr=err,check=True)
print(a.output.resolve())
