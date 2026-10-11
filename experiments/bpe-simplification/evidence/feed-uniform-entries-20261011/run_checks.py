from pathlib import Path
import os,subprocess,json,time
e=Path(__file__).parent
base=['cargo','test','--offline','--locked','--manifest-path','tokenizers/tk-train/Cargo.toml','--lib','trainers::bpe']
checks=[('baseline','default',base),('baseline','no-default',base+['--no-default-features','--','--skip','feed_training_tokenizer_json_and_encoding_roundtrip']),('candidate','default',base),('candidate','no-default',base+['--no-default-features','--','--skip','feed_training_tokenizer_json_and_encoding_roundtrip']),('candidate','clippy',['cargo','clippy','--offline','--locked','--manifest-path','tokenizers/tk-train/Cargo.toml','--all-targets','--','-D','warnings']),('candidate','format',['uv','run','experiments/bpe-simplification/format-changed.py','--check','tokenizers/tk-train/src/trainers/bpe/word_counts.rs','tokenizers/tk-train/src/trainers/bpe/tests.rs'])]
results=[]
env=dict(os.environ,PATH='/root/.cargo/bin:'+os.environ['PATH'],CARGO_TARGET_DIR='/tmp/bpe-index-tests-20261011',CARGO_INCREMENTAL='0')
for arm,name,command in checks:
 log=e/(arm+'-'+name+'.log')
 begin=time.time()
 with log.open('w') as out: r=subprocess.run(command,cwd='/tmp/bpe-feed-uniform-'+arm+'-20261011',env=env,stdout=out,stderr=subprocess.STDOUT)
 results.append(dict(arm=arm,name=name,command=command,exit_code=r.returncode,seconds=time.time()-begin,log=log.name))
 (e/'validation.json').write_text(json.dumps(results,indent=2))
 print(results[-1],flush=True)
 if r.returncode and name!='default': raise SystemExit(r.returncode)
