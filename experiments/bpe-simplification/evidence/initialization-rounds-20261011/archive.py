from pathlib import Path
import hashlib,json,shutil,subprocess
r=Path(__file__).resolve().parent;repo=Path('/home/lyk/code/tokenizers');dest=repo/'experiments/bpe-simplification/evidence/initialization-rounds-20261011';dest.mkdir(exist_ok=True)
def copy(src,tgt):tgt.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(src,tgt)
for name in ('plan.md','builds.json','inputs.json','checks.json','prepare-builds.py','build.py','measure.py','check-final.py','review.py','archive.py','report.py'):
 copy(r/name,dest/name)
for name in ('a1-test','a1-default-test','a2-test','a2-clippy','a3-test','a3-no-default','a3-clippy','a2-restart-cutoff','baseline-default-unfiltered','final-default-unfiltered','final-no-default','final-clippy','final-format','final-whitespace','baseline-build','final-build'):
 copy(r/(name+'.log'),dest/'checks'/(name+'.log'))
for folder in ('source','upstream'):
 for src in (r/folder).rglob('*'):
  if src.is_file():copy(src,dest/folder/src.relative_to(r/folder))
for name in ('manifest.json','runs.json','summary.json'):copy(r/'measurements'/name,dest/'sanity'/name)
artifacts=[]
for run in json.loads((r/'measurements/runs.json').read_text()):
 src=r/'measurements'/run['run'];artifacts.append(dict(run=run['run'],job=json.loads((src/'job.json').read_text()),stdout=(src/'stdout.log').read_text(),stderr=(src/'stderr.log').read_text()))
(dest/'sanity/sample-artifacts.json').write_text(json.dumps(artifacts,ensure_ascii=False,indent=2))
checks={}
for f,h in json.loads((r/'builds.json').read_text())['builds']['final']['source_hashes'].items():
 if f.startswith('tokenizers/'):
  actual=hashlib.sha256((repo/f).read_bytes()).hexdigest();checks[f]=actual;assert actual==h,f
(dest/'integrated-source-hashes.json').write_text(json.dumps(checks,indent=2))
(dest/'.gitattributes').write_text('*.log -whitespace\n*.diff -whitespace\nupstream/mod.rs -whitespace\n')
print('archived',len(artifacts),'attempts',dest)
