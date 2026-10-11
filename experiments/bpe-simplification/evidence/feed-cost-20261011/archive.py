from pathlib import Path
import hashlib,json,shutil,subprocess
r=Path(__file__).resolve().parent;repo=Path('/home/lyk/code/tokenizers'); dest=repo/'experiments/bpe-simplification/evidence/feed-cost-20261011';dest.mkdir(exist_ok=True)
def copy(src,tgt):
 tgt.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(src,tgt)
for name in ('builds.json','inputs.json','freeze.py','freeze-probe.py','measure.py','probe.py','archive.py','report.py','prepare-scales.py','scc-source-excerpts.txt','drain-validation.json'):
 copy(r/name,dest/name)
for name in ('drain-default-filtered.log','drain-no-default-filtered.log','drain-clippy.log','drain-format.log','drain-tests.log'):
 copy(r/name,dest/name)
for size in (128,384,512):copy(r/'corpora'/str(size)/'manifest.json',dest/'source-corpora'/f'zh{size}.json')
copy(repo.parent/'tokenizers-bpe-benchmarks-laptop/.bench/corpora/zh256/manifest.json',dest/'source-corpora/zh256.json')
for source in (r/'source').rglob('*'):
 if source.is_file():copy(source,dest/'source'/source.relative_to(r/'source'))
for source in r.glob('*.patch'):copy(source,dest/source.name)
for name in ('builds.json','inputs.json','measure.py'):copy(r/'production'/name,dest/'production'/name)
for arm,b in json.loads((r/'production/builds.json').read_text())['builds'].items():
 for file,h in b['source_hashes'].items():
  source=Path(b['worktree'])/file if arm in ('baseline','unified','collected') else r/'source'/arm/file
  if hashlib.sha256(source.read_bytes()).hexdigest()!=h:
   source=Path('/tmp/bpe-feed-unification-results/source')/('local' if arm=='unified' else arm)/file
  assert hashlib.sha256(source.read_bytes()).hexdigest()==h,(arm,file)
  copy(source,dest/'production/source'/arm/file)
for path,target in [(r/'measurements',dest/'controlled'),(r/'probes',dest/'probes'),(r/'probes-drain',dest/'probes-drain'),(r/'production/measurements',dest/'initial-attempt'),(r/'production/scaling',dest/'production/scaling')]:
 if not path.exists():continue
 for name in ('manifest.json','runs.json','summary.json'):
  if (path/name).exists():copy(path/name,target/name)
 artifacts=[]
 for run in json.loads((path/'runs.json').read_text()):
  folder=path/run['run'];a=dict(run=run['run'],returncode=run['returncode'])
  for file in ('job.json','stdout.log','stderr.log'):
   a[file]=json.loads((folder/file).read_text()) if file=='job.json' else (folder/file).read_text()
  artifacts.append(a)
 (target/'sample-artifacts.json').write_text(json.dumps(artifacts,ensure_ascii=False,indent=2))
p=r/'failed-training-retry';retry={file:json.loads((p/file).read_text()) if file.endswith('.json') else (p/file).read_text() for file in ('job.json','stdout.log','stderr.log','validation.json')};(dest/'retry-artifacts.json').write_text(json.dumps(retry,ensure_ascii=False,indent=2))
(dest/'.gitattributes').write_text('*.log -whitespace\n*.patch -whitespace\n')
(dest/'final-production.patch').write_bytes(subprocess.check_output(['git','diff'],cwd=repo))
print(dest)
