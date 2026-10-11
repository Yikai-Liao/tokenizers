from pathlib import Path
import hashlib,json,shutil,subprocess
r=Path(__file__).resolve().parent;repo=Path('/home/lyk/code/tokenizers')
dest=repo/'experiments/bpe-simplification/evidence/neighbor-directories-20261011';dest.mkdir(exist_ok=True)
def copy(src,target):target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
for name in ('plan.md','builds.json','inputs.json','checks.json','prepare.py','build.py','measure-core.py','check-final.py','archive.py','report.py','check-integration.py','baseline.diff','directories.diff','insert.diff','combined.diff'):
 copy(r/name,dest/name)
for name in ('default','default-final','no-default','clippy','format','whitespace','baseline-build','directories-build','insert-build','combined-build'):
 copy(r/(name+'.log'),dest/'checks'/(name+'.log'))
for name in ('default','no-default','clippy'):
 copy(Path('/tmp/bpe-interface-prefix-20261011')/('interfaces-'+name+'.log'),dest/'checks'/('baseline-'+name+'.log'))
# Baseline and combined trees plus independent patches reconstruct all four arms.
for arm in ('baseline','combined'):
 for src in (r/'source'/arm).rglob('*'):
  if src.is_file():copy(src,dest/'source'/arm/src.relative_to(r/'source'/arm))
for name in ('manifest.json','runs.json','summary.json'):copy(r/'core'/name,dest/'core'/name)
artifacts=[]
for run in json.loads((r/'core/runs.json').read_text()):
 src=r/'core'/run['run']
 artifacts.append(dict(run=run['run'],job=json.loads((src/'job.json').read_text()),stdout=(src/'stdout.log').read_text(),stderr=(src/'stderr.log').read_text()))
(dest/'core/sample-artifacts.json').write_text(json.dumps(artifacts,ensure_ascii=False,indent=2))
checks={}
for f,h in json.loads((r/'builds.json').read_text())['builds']['combined']['source_hashes'].items():
 if f.startswith('tokenizers/'):
  actual=hashlib.sha256((r/'source/combined'/f).read_bytes()).hexdigest();assert actual==h,f;checks[f]=actual
(dest/'measured-source-hashes.json').write_text(json.dumps(checks,indent=2))
for src in (r/'integration').rglob('*'):
 if src.is_file():copy(src,dest/'integration'/src.relative_to(r/'integration'))
(dest/'.gitattributes').write_text('*.log -whitespace\n*.diff -whitespace\n')
print('archived',len(artifacts),'attempts',flush=True)
