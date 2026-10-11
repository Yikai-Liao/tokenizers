from pathlib import Path
import subprocess,shutil,hashlib,os,json,sys
r=Path('/root/code/tokenizers-workspaces/bpe-affix-initial-final-results-20261012');r.mkdir(exist_ok=True)
w=Path('/tmp/bpe-selected-rules-measurement-20261011');rel=Path('tokenizers/tk-train/src/trainers/bpe')
base=Path('/tmp/bpe-positions-read-20261012');candidate=Path('/tmp/bpe-affix-initial-20261012')
runner=Path('/root/code/tokenizers-workspaces/bpe-positions-child-results-20261012/source/baseline/experiments/bpe-simplification/runner')
shutil.copy2('/root/code/tokenizers-workspaces/bpe-affix-verification-results-20261011/inputs.json',r/'inputs.json')
for arm in (sys.argv[1:] or ['baseline','candidate']):
 source=base if arm=='baseline' else candidate
 shutil.copytree(runner,w/'experiments/bpe-simplification/runner',dirs_exist_ok=True)
 shutil.copyfile(source/'tokenizers/tk-train/Cargo.toml',w/'tokenizers/tk-train/Cargo.toml')
 for f in (source/rel).glob('*.rs'):shutil.copyfile(f,w/rel/f.name)
 if arm=='flags':shutil.copyfile(r/'production-vocabulary-flags.rs',w/rel/'vocabulary.rs')
 # Use the exact same train phase boundaries as the retained child comparison.
 shutil.copyfile('/root/code/tokenizers-workspaces/bpe-positions-child-results-20261012/source/child-inline-fast/tokenizers/tk-train/src/trainers/bpe/mod.rs',w/rel/'mod.rs')
 mod=w/rel/'mod.rs';s=mod.read_text()
 s=s.replace('        let (mut vocabulary, ids) = Vocabulary::initialize(self, words, alphabet, progress)?;','        let init_cpu = unsafe { bpe_init_cpu() };\n        let (mut vocabulary, ids) = Vocabulary::initialize(self, words, alphabet, progress)?;\n        eprintln!("BPE_VOCABULARY_CPU_SECONDS {:.9}", unsafe { bpe_init_cpu() } - init_cpu);',1)
 assert 'let init_cpu' in s
 s+='\nunsafe extern "C" { fn bpe_init_cpu() -> f64; }\n';mod.write_text(s)
 main=w/'experiments/bpe-simplification/runner/src/main.rs';main.write_text(main.read_text()+'\n#[unsafe(no_mangle)]\npub extern "C" fn bpe_init_cpu() -> f64 { protocol::cpu_seconds().unwrap() }\n')
 snapshot=r/'source'/arm
 shutil.copytree(w/rel,snapshot/rel,dirs_exist_ok=True);shutil.copytree(w/'experiments/bpe-simplification/runner',snapshot/'experiments/bpe-simplification/runner',dirs_exist_ok=True)
 shutil.copyfile(w/'tokenizers/tk-train/Cargo.toml',snapshot/'tokenizers/tk-train/Cargo.toml')
 cmd=['cargo','build','--release','--locked','--offline','--target-dir','/root/code/tokenizers-build/bpe-affix-core-build-20261011']
 with (r/f'{arm}-build.log').open('w') as log:subprocess.run(cmd,cwd=w/'experiments/bpe-simplification/runner',env=os.environ|{'PATH':'/root/.cargo/bin:'+os.environ['PATH'],'CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'2'},stdout=log,stderr=subprocess.STDOUT,check=True)
 binary=r/f'{arm}-bin';shutil.copyfile('/root/code/tokenizers-build/bpe-affix-core-build-20261011/release/bpe-bench-runner',binary)
 binary.chmod(0o755)
 metadata={'arm':arm,'command':cmd,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=base,text=True).strip(),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest()}
 (r/f'{arm}-build.json').write_text(json.dumps(metadata,indent=2)+'\n');print(metadata,flush=True)
