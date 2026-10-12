from pathlib import Path
import subprocess,shutil,os,json
r=Path(__file__).resolve().parent;old=Path('/tmp/bpe-positions-encapsulation-final-results-20261012');target=Path('/tmp/bpe-positions-codec-libraries-native-20261012');original=Path('/root/code/tokenizers-build/bpe-positions-simd-native-20261012')
assert not target.exists();shutil.copytree(original,target)
env=os.environ|{'PATH':'/root/.cargo/bin:'+os.environ['PATH'],'CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'1','RAYON_NUM_THREADS':'1'}
for arm in json.loads((r/'variants.json').read_text()):
 name=arm['name'];work=Path(arm['worktree']);manifest=work/'tokenizers/tk-train/Cargo.toml';shutil.copyfile(old/'native-Cargo.lock',manifest.parent/'Cargo.lock')
 commands=[['cargo','test','--offline','--manifest-path',str(manifest),'--lib','trainers::bpe','--target-dir',str(target),'--','--test-threads=1','--skip','feed_training_tokenizer_json_and_encoding_roundtrip'],['cargo','test','--locked','--offline','--manifest-path',str(manifest),'--no-default-features','--lib','trainers::bpe','--target-dir',str(target),'--','--test-threads=1','--skip','feed_training_tokenizer_json_and_encoding_roundtrip'],['cargo','clippy','--locked','--offline','--manifest-path',str(manifest),'--all-targets','--target-dir',str(target),'--','-D','warnings','-A','clippy::redundant_field_names']]
 for label,command in zip(['default','no-default','clippy'],commands):
  with (r/f'{name}-{label}.log').open('w') as output:subprocess.run(['taskset','-c','0-1']+command,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
  print(name,label,'passed',flush=True)
 shutil.copyfile(manifest.parent/'Cargo.lock',r/(name+'-native-Cargo.lock'))
(r/'native-checks.json').write_text(json.dumps(dict(candidates=[a['name'] for a in json.loads((r/'variants.json').read_text())],checks=['BPE lib tests default and no-default','clippy all-targets -D warnings with baseline redundant_field_names allow'],skipped_known_baseline_fixture='feed_training_tokenizer_json_and_encoding_roundtrip',affinity=[0,1]),indent=2)+'\n')
# Remove only this task's copied build cache, preserving the original cache.
assert target.name=='bpe-positions-codec-libraries-native-20261012';assert (target/'CACHEDIR.TAG').exists();assert set(p.name for p in target.iterdir()) <= {'CACHEDIR.TAG','debug','.rustc_info.json'}
shutil.rmtree(target)
