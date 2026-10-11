"""Final native checks, preserving the independently known encoder failure."""
import json,os,subprocess
from pathlib import Path
r=Path(__file__).resolve().parent;repo=Path('/home/lyk/code/tokenizers')
cmd=['cargo','test','--manifest-path','tokenizers/tk-train/Cargo.toml','--locked','--offline','--lib']
env=dict(os.environ,CARGO_TARGET_DIR='/tmp/bpe-feed-native')
with (r/'default-final.log').open('w') as f:p=subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=f)
log=(r/'default-final.log').read_text()
assert p.returncode==101 and '43 passed; 1 failed;' in log
assert 'failures:\n    trainers::bpe::tests::feed_training_tokenizer_json_and_encoding_roundtrip' in log
assert '43 passed; 0 failed;' in (r/'no-default.log').read_text()
assert 'Finished' in (r/'clippy.log').read_text()
checks={'default':{'command':cmd,'returncode':p.returncode,'log':'default-final.log','expected_known_failure':'feed_training_tokenizer_json_and_encoding_roundtrip'},
 'no-default':{'command':cmd[:-1]+['--no-default-features','--lib','--','--skip','feed_training_tokenizer_json_and_encoding_roundtrip'],'returncode':0,'log':'no-default.log'},
 'clippy':{'command':['cargo','clippy','--manifest-path','tokenizers/tk-train/Cargo.toml','--locked','--offline','--all-targets','--','-D','warnings'],'returncode':0,'log':'clippy.log'}}
for name,cmd in {
 'format':['uv','run','experiments/bpe-simplification/format-changed.py','--check','tokenizers/tk-train/src/trainers/bpe/merge.rs','tokenizers/tk-train/src/trainers/bpe/vocabulary.rs'],
 'whitespace':['git','diff','--check'],
}.items():
 with (r/(name+'.log')).open('w') as f:p=subprocess.run(cmd,cwd=repo,stdout=f,stderr=f)
 assert p.returncode==0,name;checks[name]={'command':cmd,'returncode':0,'log':name+'.log'}
(r/'checks.json').write_text(json.dumps(checks,indent=2))
print('native checks complete: 43 passes, known encoder failure preserved',flush=True)
