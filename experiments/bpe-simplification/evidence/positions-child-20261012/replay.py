"""Replay an archived run in a disposable checkout of environment.json base_sha.
python replay.py WORKSPACE SUITE/RUN --input ORIGINAL_PATH=LOCAL_PATH
This overlays the frozen timed sources and writes replay-model.json in WORKSPACE.
"""
from pathlib import Path
import argparse,json,shutil,subprocess,os
p=argparse.ArgumentParser();p.add_argument('workspace',type=Path);p.add_argument('run');p.add_argument('--input',action='append',default=[]);a=p.parse_args()
e=Path(__file__).resolve().parent;w=a.workspace.resolve();record=json.loads((e/a.run/'job.json').read_text());arm=record['build_id']
root=e/('final-diagnostic-source' if arm.endswith('-final-diagnostic') else 'source')
source_arm=arm.replace('-final-diagnostic','');snapshot=root/source_arm
if not snapshot.exists():raise SystemExit('This historical arm has no replay snapshot; use baseline or child-inline-fast, inline24 or thinvec.')
for f in snapshot.rglob('*'):
 if f.is_file():
  dest=w/f.relative_to(snapshot);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
inputs=dict(x.split('=',1) for x in a.input);record['input']=inputs.get(record['input'],record['input']);record['output']=str(w/'replay-model.json')
job=w/'replay-job.json';job.write_text(json.dumps(record,indent=2)+'\n');target=w/'replay-target'
subprocess.run(['cargo','build','--release','--locked','--target-dir',str(target)],cwd=w/'experiments/bpe-simplification/runner',env=os.environ|{'CARGO_INCREMENTAL':'0'},check=True)
subprocess.run(['taskset','-c','0-3',str(target/'release/bpe-bench-runner'),str(job)],check=True)
