from pathlib import Path
import subprocess,shutil,hashlib,os,json,sys
r=Path('/root/code/tokenizers-workspaces/bpe-prefetch-right-results-20261012');r.mkdir(exist_ok=True)
w=Path('/tmp/bpe-selected-rules-measurement-20261011');rel=Path('tokenizers/tk-train/src/trainers/bpe')
base=Path('/tmp/bpe-affix-initial-20261012');candidate=Path('/tmp/bpe-prefetch-right-20261012')
runner=Path('/root/code/tokenizers-workspaces/bpe-positions-child-results-20261012/source/baseline/experiments/bpe-simplification/runner')
for arm in (sys.argv[1:] or ['baseline','head-right']):
 source=base if arm=='baseline' else candidate
 shutil.copytree(runner,w/'experiments/bpe-simplification/runner',dirs_exist_ok=True)
 shutil.copyfile(source/'tokenizers/tk-train/Cargo.toml',w/'tokenizers/tk-train/Cargo.toml')
 for f in (source/rel).glob('*.rs'):shutil.copyfile(f,w/rel/f.name)
 old=Path('/root/code/tokenizers-workspaces/bpe-positions-child-results-20261012/source/child-inline-fast')/rel/'mod.rs'
 shutil.copyfile(old,w/rel/'mod.rs')
 mod=w/rel/'mod.rs';s=mod.read_text()
 needle='            corpus = plan.materialize();';assert needle in s
 s=s.replace(needle,needle+'\n            eprintln!("BPE_TOKEN_WIDTH {}", corpus.bench_token_width());')
 needle='            eprintln!("BPE_MERGE_SECONDS {:.9}", phase.elapsed().as_secs_f64());';assert needle in s
 s=s.replace(needle,needle+'\n            merge::bench_report_prepare();');mod.write_text(s)
 corpus=w/rel/'corpus.rs';corpus.write_text(corpus.read_text()+'\nimpl Corpus { pub(super) fn bench_token_width(&self) -> usize { match self.tokens { Slots::Narrow(_) => 16, Slots::Wide(_) => 32 } } }\n')
 merge=w/rel/'merge.rs';s=merge.read_text()
 needle='        let jobs = batch.prepare(corpus, self.max_token_length.unwrap_or(usize::MAX))?;';assert needle in s
 s=s.replace(needle,'        let prepare_reuse = batch.reuse;\n        let prepare_started = std::time::Instant::now();\n'+needle+'\n        BENCH_PREPARE_NS.fetch_add(prepare_started.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);\n        if prepare_reuse { &BENCH_REUSE_BATCHES } else { &BENCH_FRESH_BATCHES }.fetch_add(1, std::sync::atomic::Ordering::Relaxed);')
 s+='''
static BENCH_PREPARE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BENCH_FRESH_BATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BENCH_REUSE_BATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(super) fn bench_report_prepare() {
    eprintln!("BPE_PREPARE_SECONDS {:.9}", BENCH_PREPARE_NS.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e9);
    eprintln!("BPE_FRESH_BATCHES {}", BENCH_FRESH_BATCHES.swap(0, std::sync::atomic::Ordering::Relaxed));
    eprintln!("BPE_REUSE_BATCHES {}", BENCH_REUSE_BATCHES.swap(0, std::sync::atomic::Ordering::Relaxed));
}
''';merge.write_text(s)
 snapshot=r/'source'/arm
 shutil.copytree(w/rel,snapshot/rel,dirs_exist_ok=True);shutil.copytree(w/'experiments/bpe-simplification/runner',snapshot/'experiments/bpe-simplification/runner',dirs_exist_ok=True)
 shutil.copyfile(w/'tokenizers/tk-train/Cargo.toml',snapshot/'tokenizers/tk-train/Cargo.toml')
 cmd=['cargo','build','--release','--locked','--offline','--target-dir','/root/code/tokenizers-build/bpe-affix-core-build-20261011']
 with (r/f'{arm}-build.log').open('w') as log:subprocess.run(cmd,cwd=w/'experiments/bpe-simplification/runner',env=os.environ|{'PATH':'/root/.cargo/bin:'+os.environ['PATH'],'CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'2'},stdout=log,stderr=subprocess.STDOUT,check=True)
 binary=r/f'{arm}-bin';shutil.copyfile('/root/code/tokenizers-build/bpe-affix-core-build-20261011/release/bpe-bench-runner',binary);binary.chmod(0o755)
 metadata={'arm':arm,'command':cmd,'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=base,text=True).strip(),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest()}
 (r/f'{arm}-build.json').write_text(json.dumps(metadata,indent=2)+'\n');print(metadata,flush=True)
