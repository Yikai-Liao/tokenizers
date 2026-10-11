"""Run unfiltered feature matrices and retain the known failure in both arms."""
from pathlib import Path
import json, os, subprocess, time
r=Path(__file__).resolve().parent
env=dict(os.environ,CARGO_TARGET_DIR='/tmp/bpe-feed-native')
results=[]
for arm in ('baseline','boxed'):
    w=r/('worktree-'+arm)
    commands={
      'default': ['cargo','test','--manifest-path','tokenizers/tk-train/Cargo.toml','--lib','--offline','--locked'],
      'no-default': ['cargo','test','--manifest-path','tokenizers/tk-train/Cargo.toml','--lib','--no-default-features','--offline','--locked'],
      'clippy-default': ['cargo','clippy','--manifest-path','tokenizers/tk-train/Cargo.toml','--all-targets','--offline','--locked','--','-D','warnings'],
      'clippy-no-default': ['cargo','clippy','--manifest-path','tokenizers/tk-train/Cargo.toml','--all-targets','--no-default-features','--offline','--locked','--','-D','warnings'],
    }
    for name,cmd in commands.items():
        log=r/(arm+'-'+name+'.log');start=time.time()
        with log.open('w') as f: p=subprocess.run(cmd,cwd=w,env=env,stdout=f,stderr=f)
        s=log.read_text()
        row={'arm':arm,'check':name,'command':cmd,'cwd':str(w),'environment':{'CARGO_TARGET_DIR':env['CARGO_TARGET_DIR']},'exit_code':p.returncode,'seconds':time.time()-start}
        results.append(row);(r/'checks.json').write_text(json.dumps(results,indent=2))
        print(arm,name,p.returncode,flush=True)
        if name in ('default','no-default'):
            assert p.returncode==101 and '43 passed; 1 failed;' in s and 'failures:\n    trainers::bpe::tests::feed_training_tokenizer_json_and_encoding_roundtrip' in s,(arm,name,s[-5000:])
        else: assert p.returncode==0,(arm,name,s[-5000:])
cmd=['uv','run','experiments/bpe-simplification/format-changed.py','--check']+[f'tokenizers/tk-train/src/trainers/bpe/{n}.rs' for n in ('corpus','vocabulary','merge','index')]
subprocess.run(cmd,cwd=r/'worktree-boxed',check=True)
subprocess.run(['git','diff','--check'],cwd=r/'worktree-boxed',check=True)
