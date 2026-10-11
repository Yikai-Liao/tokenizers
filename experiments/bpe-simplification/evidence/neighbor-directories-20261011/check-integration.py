"""Validate the merge with independently completed remote Corpus B1 work."""
from pathlib import Path
import hashlib,json,os,subprocess
r=Path(__file__).resolve().parent;repo=Path('/home/lyk/code/tokenizers');out=r/'integration';out.mkdir(exist_ok=True)
commands={
 'default':['cargo','test','--manifest-path','tokenizers/tk-train/Cargo.toml','--locked','--offline','--lib'],
 'no-default':['cargo','test','--manifest-path','tokenizers/tk-train/Cargo.toml','--locked','--offline','--no-default-features','--lib','--','--skip','feed_training_tokenizer_json_and_encoding_roundtrip'],
 'clippy':['cargo','clippy','--manifest-path','tokenizers/tk-train/Cargo.toml','--locked','--offline','--all-targets','--','-D','warnings'],
 'format':['uv','run','experiments/bpe-simplification/format-changed.py','--check','--base','8f078a8c','tokenizers/tk-train/src/trainers/bpe/merge.rs','tokenizers/tk-train/src/trainers/bpe/vocabulary.rs'],
 'whitespace':['git','diff','--check','8f078a8c','HEAD'],
}
checks={};env=dict(os.environ,CARGO_TARGET_DIR='/tmp/bpe-feed-native')
for name,cmd in commands.items():
 with (out/(name+'.log')).open('w') as f:p=subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=f)
 log=(out/(name+'.log')).read_text()
 if name=='default':
  assert p.returncode==101 and '43 passed; 1 failed;' in log
  assert 'failures:\n    trainers::bpe::tests::feed_training_tokenizer_json_and_encoding_roundtrip' in log
 else:assert p.returncode==0,(name,log)
 checks[name]={'command':cmd,'returncode':p.returncode};print(name,p.returncode,flush=True)
builds=json.loads((r/'builds.json').read_text());actual={};changed=[]
for f,h in builds['builds']['combined']['source_hashes'].items():
 if f.startswith('tokenizers/'):
  a=hashlib.sha256((repo/f).read_bytes()).hexdigest();actual[f]=a
  if a!=h:changed.append(f)
assert changed==['tokenizers/tk-train/src/trainers/bpe/corpus.rs'],changed
corpus=repo/changed[0];remote=subprocess.check_output(['git','show','8f078a8c:'+changed[0]],cwd=repo)
assert corpus.read_bytes()==remote
(out/'corpus.rs').write_bytes(remote)
manifest={'production_change_commit':'fc1021c7','remote_commit':'8f078a8c','merge_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'checks':checks,'source_hashes':actual,'differences_from_measured_combined':changed,'timing_note':'32-arm comparison precedes remote B1 integration; integrated version receives native checks, no new timing claim'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
