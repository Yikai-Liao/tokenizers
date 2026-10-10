from pathlib import Path
import gzip,hashlib,json,shutil,subprocess
work=Path('/root/code/tokenizers-workspaces/bpe-heap-experiment')
archive=work/'experiments/bpe-simplification/evidence/serde-heap-simplification-20261010';archive.mkdir(exist_ok=True)
def copy(src,dest):
 p=archive/dest;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,p)
def model_hash(model):return hashlib.sha256(json.dumps(model,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def measurements(source,label):
 source=Path(source); target=archive/label;target.mkdir(exist_ok=True);references={};checks=[]
 for p in source.rglob('*'):
  if not p.is_file(): continue
  if p.name!='model.json':copy(p,Path(label)/p.relative_to(source));continue
  job=json.loads((p.parent/'job.json').read_text());case=job['input_id'];model=json.loads(p.read_text())
  if case not in references:
   references[case]=model
   with gzip.open(target/f'{case}-reference-model.json.gz','wb') as f:f.write(p.read_bytes())
  assert model==references[case],p
  checks.append(dict(run=p.parent.name,model_sha256_raw=hashlib.sha256(p.read_bytes()).hexdigest(),model_sha256_canonical=model_hash(model),complete_model_equal=True))
 (target/'model-validation.json').write_text(json.dumps(checks,indent=2))
measurements('/tmp/bpe-serde-peekmut-measurements','initial-heap')
copy('/tmp/bpe-serde-peekmut-measure.py','initial-heap/measure.py');copy('/tmp/bpe-serde-peekmut-builds.json','initial-heap/builds.json')
for label in ['baseline','candidate']:copy(f'/tmp/bpe-serde-peekmut-{label}-build.log',f'initial-heap/{label}-build.log')
for src,name in [('bpe-serde-peekmut-serialization.rs','counter.rs'),('bpe-serde-peekmut-serialization-buffer.rs','buffer.rs'),('bpe-serde-peekmut-serialization-runs.json','counter-runs.json'),('bpe-serde-peekmut-serialization-buffer-runs.json','buffer-runs.json'),('bpe-serde-peekmut-serialization-build-command.json','counter-build-command.json'),('bpe-serde-peekmut-serialization-buffer-build-command.json','buffer-build-command.json'),('bpe-serde-peekmut-old-word-counts.rs','old-word-counts.rs'),('bpe-serde-peekmut-new-word-counts.rs','new-word-counts.rs')]:copy('/tmp/'+src,'serialization/'+name)
for src in ['dedup.rs','dedup-runs.json','dedup-build-command.json','positions.rs']:copy('/tmp/bpe-cleanup-'+src,'dedup/'+src)
for src,name in [('bpe-serde-peekmut-cohort-before-fix.log','injected-peekmut-failure.log'),('bpe-serde-peekmut-cohort-after-fix.log','conditional-intermediate-pass.log'),('bpe-cleanup-tests.log','default-tests.log'),('bpe-cleanup-no-default-tests.log','no-default-tests.log'),('bpe-cleanup-clippy.log','clippy.log')]:copy('/tmp/'+src,'validation/'+name)
for rev,label in [('d4e3fa45','baseline-d4e3fa45'),('f4ed2324','initial-f4ed2324'),('a31898fa','final-a31898fa')]:
 for path in subprocess.check_output(['git','ls-tree','-r','--name-only',rev,'tokenizers/tk-train/src/trainers/bpe'],cwd=work,text=True).splitlines():
  if path.endswith('.rs'):
   p=archive/'source'/label/Path(path).name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(subprocess.check_output(['git','show',rev+':'+path],cwd=work))
print(archive)
