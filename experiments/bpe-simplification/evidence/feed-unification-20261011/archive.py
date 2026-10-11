"""Archive compact, reproducible evidence; large inputs/models/count files stay outside Git."""
import json,shutil
from pathlib import Path
root=Path(__file__).resolve().parent
out=Path('/home/lyk/code/tokenizers/experiments/bpe-simplification/evidence/feed-unification-20261011');out.mkdir(exist_ok=True,parents=True)
for name in ['plan.md','freeze.py','measure.py','archive.py','write-report.py','builds.json','inputs.json','source-counts.json','final-production.patch','micro-summary.json','combined-summary.json','micro.csv','micro.stderr','micro-build.log','micro-lock.log','baseline-tests.log','local-tests.log','direct-tests.log','runner-tests.log','build-baseline.log','build-local.log','build-direct.log','measure.log','extra.log','validation.json','source-corpora.json','micro-manifest.json']:
 p=root/name
 if p.exists():shutil.copy2(p,out/name)
for log in root.glob('final-*.log'):
 shutil.copy2(log,out/log.name)
(out/'.gitattributes').write_text('*.patch -whitespace\n*.log -whitespace\n')
for arm in ['baseline','local','direct']:
 shutil.copy2(root/(arm+'.patch'),out/(arm+'.patch'))
 shutil.copytree(root/'source'/arm,out/'source'/arm,dirs_exist_ok=True)
(out/'micro').mkdir(exist_ok=True)
for name in ['Cargo.toml','Cargo.lock']:
 shutil.copy2(root/'micro'/name,out/'micro'/name)
shutil.copytree(root/'micro'/'src',out/'micro'/'src',dirs_exist_ok=True)
for directory in ['measurements','extra']:
 src=root/directory
 if not src.exists():continue
 dest=out/directory;dest.mkdir(exist_ok=True)
 for name in ['manifest.json','runs.json','summary.json']:
  if (src/name).exists():shutil.copy2(src/name,dest/name)
 records=json.loads((src/'runs.json').read_text())
 artifacts=[]
 for row in records:
  run=src/row['run'];artifacts.append({'run':row['run'],'job':json.loads((run/'job.json').read_text()),'stdout':(run/'stdout.log').read_text(),'stderr':(run/'stderr.log').read_text(),'returncode':row['returncode']})
 (dest/'sample-artifacts.json').write_text(json.dumps(artifacts,indent=2))
print(out)
