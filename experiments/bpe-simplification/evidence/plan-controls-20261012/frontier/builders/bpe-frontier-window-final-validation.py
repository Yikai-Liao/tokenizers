from pathlib import Path
import subprocess,os
w=Path('/tmp/bpe-frontier-window-20261012');p=w/'tokenizers/tk-train/src/trainers/bpe/merge.rs';original=p.read_text()
env=os.environ|{'PATH':'/root/.cargo/bin:'+os.environ['PATH'],'CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'2'}
common=['cargo','test','--locked','--offline','--manifest-path','tokenizers/tk-train/Cargo.toml','--target-dir','/tmp/bpe-selected-rules-checks-20261011']
def run(args,name):
 with Path('/tmp/'+name).open('w') as log:subprocess.run(args,cwd=w,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print(name+' passed',flush=True)
try:
 run(common+['--no-default-features','trainers::bpe::tests::frontier_window_boundaries','--','--exact'],'bpe-frontier-window-corpus-oracle-nodefault.log')
 for cohorts,name in [(1,'single'),(0,'eager')]:
  p.write_text(original.replace('pub(super) const WINDOW_COHORTS: usize = 4;',f'pub(super) const WINDOW_COHORTS: usize = {cohorts};'))
  run(common+['trainers::bpe::tests::frontier_window_boundaries','--','--exact'],f'bpe-frontier-{name}-corpus-oracle.log')
finally:p.write_text(original)
run(['cargo','clippy','--locked','--offline','--manifest-path','tokenizers/tk-train/Cargo.toml','--target-dir','/tmp/bpe-selected-rules-checks-20261011','--all-targets','--','-D','warnings','-A','clippy::redundant_field_names'],'bpe-frontier-window-final-clippy.log')
