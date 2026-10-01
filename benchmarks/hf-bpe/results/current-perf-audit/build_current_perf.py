#!/usr/bin/env python3
"""Build the clean current candidate as an isolated optimized DWARF diagnostic binary."""
import difflib, hashlib, json, os, shutil, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1]
WORKTREE = Path('/root/code/tokenizers-worktrees/initial-owner-waves').resolve()
BUILD = BENCH / '.build/native-j-current-perf'
TARGET = BENCH / '.build/native-j-current-perf-target'
SOURCE = BUILD / 'source/tokenizers'
if subprocess.check_output(['git','-C',str(WORKTREE),'status','--porcelain'],text=True).strip():
    raise SystemExit('candidate worktree must be clean')
if BUILD.exists() or TARGET.exists():
    raise SystemExit('refusing to overwrite an existing diagnostic build')
shutil.copytree(WORKTREE/'tokenizers', SOURCE, ignore=shutil.ignore_patterns('target','.git','data'))
trainer = SOURCE/'tk-train/src/trainers/bpe/mod.rs'
text = trainer.read_text()
anchor = '        Ok((trained.vocab, trained.merges, trained.special_tokens))'
assert text.count(anchor) == 1
trainer.write_text(text.replace(anchor, '        eprintln!("{}", serde_json::json!({"bench_indexed_stats": trained.stats}));\n'+anchor))
runner = BUILD/'runner'
(runner/'src').mkdir(parents=True)
main = (BENCH/'src/main.rs').read_text()
anchor = '    let reader = BufReader::new(File::open(input)?);'
assert main.count(anchor) == 1
main = main.replace(anchor, '    let bench_workers = env::var("HF_BPE_BENCH_WORKERS").map(|v| v.parse::<usize>().expect("invalid workers")).unwrap_or(4);\n    assert!(matches!(bench_workers, 1 | 4));\n    tk_encode::parallelism::set_num_threads(bench_workers);\n    tk_encode::parallelism::set_parallelism(false);\n'+anchor)
anchor = '    let feed_ms = begin.elapsed().as_secs_f64() * 1000.0;'
assert main.count(anchor) == 1
main = main.replace(anchor, anchor+'\n    tk_encode::parallelism::set_parallelism(true);')
(runner/'src/main.rs').write_text(main)
shutil.copyfile(BENCH/'Cargo.lock',runner/'Cargo.lock')
(runner/'Cargo.toml').write_text(f'''[package]\nname = "hf-bpe-native-j-current-perf"\nversion = "0.1.0"\nedition = "2024"\n[features]\nindexed = []\n[dependencies]\ntk-train = {{ path = {json.dumps(str(SOURCE/'tk-train'))}, default-features = false }}\ntk-encode = {{ path = {json.dumps(str(SOURCE/'tk-encode'))}, default-features = false }}\nserde_json = "1"\nsha2 = "0.10"\nhf-front-end = {{ package = "tokenizers", version = "=0.23.2", default-features = false, features = ["fancy-regex"] }}\n''')
# Keep the diagnostic instrumentation reviewable and hash every overlay/runner input.
tracked = {
    'overlay/tokenizers/tk-train/src/trainers/bpe/mod.rs': SOURCE/'tk-train/src/trainers/bpe/mod.rs',
    'runner/src/main.rs': runner/'src/main.rs',
    'runner/Cargo.toml': runner/'Cargo.toml',
    'runner/Cargo.lock': runner/'Cargo.lock',
}
patch_parts = []
baseline_trainer = (WORKTREE/'tokenizers/tk-train/src/trainers/bpe/mod.rs').read_text().splitlines(keepends=True)
patch_parts.extend(difflib.unified_diff(baseline_trainer, trainer.read_text().splitlines(keepends=True), fromfile='candidate/tokenizers/tk-train/src/trainers/bpe/mod.rs', tofile='overlay/tokenizers/tk-train/src/trainers/bpe/mod.rs'))
baseline_main = (BENCH/'src/main.rs').read_text().splitlines(keepends=True)
patch_parts.extend(difflib.unified_diff(baseline_main, main.splitlines(keepends=True), fromfile='benchmarks/hf-bpe/src/main.rs', tofile='runner/src/main.rs'))
(HERE/'instrumentation.patch').write_text(''.join(patch_parts))
source_files_sha256 = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in tracked.items()}
instrumentation_patch_sha256 = hashlib.sha256((HERE/'instrumentation.patch').read_bytes()).hexdigest()
env = dict(os.environ, CARGO_TARGET_DIR=str(TARGET), CARGO_PROFILE_RELEASE_DEBUG='2', CARGO_PROFILE_RELEASE_STRIP='none')
cmd = [str(Path.home()/'.cargo/bin/cargo'),'build','--offline','--release','--manifest-path',str(runner/'Cargo.toml')]
with (HERE/'build.log').open('w') as log:
    subprocess.run(cmd,check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
binary = TARGET/'release/hf-bpe-native-j-current-perf'
meta = dict(source_worktree=str(WORKTREE),source_commit=subprocess.check_output(['git','-C',str(WORKTREE),'rev-parse','HEAD'],text=True).strip(),source_files_sha256=source_files_sha256,instrumentation_patch='instrumentation.patch',instrumentation_patch_sha256=instrumentation_patch_sha256,build_root=str(BUILD),target_dir=str(TARGET),binary=str(binary),build_command=cmd,build_env={k:env[k] for k in ('CARGO_TARGET_DIR','CARGO_PROFILE_RELEASE_DEBUG','CARGO_PROFILE_RELEASE_STRIP')},binary_sha256=subprocess.check_output(['sha256sum',str(binary)],text=True).split()[0],build_id=subprocess.check_output(['readelf','-n',str(binary)],text=True).split('Build ID: ')[1].splitlines()[0],rustc=subprocess.check_output([str(Path.home()/'.cargo/bin/rustc'),'--version','--verbose'],text=True),sections=subprocess.check_output(['readelf','-S',str(binary)],text=True))
(HERE/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
print(binary)
