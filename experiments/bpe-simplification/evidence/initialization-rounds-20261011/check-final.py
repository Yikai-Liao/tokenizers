import json,subprocess
from pathlib import Path
r=Path(__file__).resolve().parent;repo=Path('/home/lyk/code/tokenizers')
commands={
 'final-default-unfiltered':['cargo','test','--manifest-path','tokenizers/tk-train/Cargo.toml','--locked','--offline','--lib'],
 'final-no-default':['cargo','test','--manifest-path','tokenizers/tk-train/Cargo.toml','--locked','--offline','--no-default-features','--lib','--','--skip','feed_training_tokenizer_json_and_encoding_roundtrip'],
 'final-clippy':['cargo','clippy','--manifest-path','tokenizers/tk-train/Cargo.toml','--locked','--offline','--all-targets','--','-D','warnings'],
 'final-format':['uv','run','experiments/bpe-simplification/format-changed.py','--check','--base','2f4803a9',*[f'tokenizers/tk-train/src/trainers/bpe/{f}.rs' for f in ('mod','vocabulary','corpus','merge','index','tests')]],
 'final-whitespace':['git','diff','--check','2f4803a9','HEAD'],
}
results={}
for name,cmd in commands.items():
 with (r/(name+'.log')).open('w') as stream:p=subprocess.run(cmd,cwd=repo,stdout=stream,stderr=stream)
 log=(r/(name+'.log')).read_text();results[name]={'command':cmd,'returncode':p.returncode}
 if name=='final-default-unfiltered':
  assert p.returncode==101 and '42 passed; 1 failed;' in log
  assert 'failures:\n    trainers::bpe::tests::feed_training_tokenizer_json_and_encoding_roundtrip' in log
 else:assert p.returncode==0,(name,log)
 print(name,p.returncode,flush=True)
(r/'checks.json').write_text(json.dumps(results,indent=2))
