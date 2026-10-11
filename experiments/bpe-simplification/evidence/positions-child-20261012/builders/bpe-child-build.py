from pathlib import Path
import os, subprocess, shutil, json, hashlib, sys
root=Path('/root/code/tokenizers-workspaces/bpe-positions-child-results-20261012');root.mkdir(exist_ok=True)
work=Path('/tmp/bpe-selected-rules-measurement-20261011')
rel=Path('tokenizers/tk-train/src/trainers/bpe')
for arm, source in [(sys.argv[1] if len(sys.argv)>1 else 'child',Path(sys.argv[2] if len(sys.argv)>2 else '/tmp/bpe-positions-child-20261012'))]:
    shutil.copytree(root/'source/baseline/experiments/bpe-simplification/runner',work/'experiments/bpe-simplification/runner',dirs_exist_ok=True)
    shutil.copyfile(source/'tokenizers/tk-train/Cargo.toml', work/'tokenizers/tk-train/Cargo.toml')
    for file in (source/rel).glob('*.rs'): shutil.copyfile(file, work/rel/file.name)
    p=work/rel/'mod.rs';s=p.read_text()
    retire = 'index.retire();' in s
    if retire: s=s.replace('index.retire();','drop(index);')
    s=s.replace('            let alphabet = vocabulary::select_alphabet(self, words, workers);','            let phase = std::time::Instant::now();\n            let alphabet = vocabulary::select_alphabet(self, words, workers);\n            eprintln!("BPE_SELECTION_SECONDS {:.9}", phase.elapsed().as_secs_f64());')
    s=s.replace('        let (mut vocabulary, ids) = Vocabulary::initialize(self, words, alphabet, progress)?;','        let phase = std::time::Instant::now();\n        let (mut vocabulary, ids) = Vocabulary::initialize(self, words, alphabet, progress)?;\n        eprintln!("BPE_VOCABULARY_SECONDS {:.9}", phase.elapsed().as_secs_f64());\n        let phase = std::time::Instant::now();')
    s=s.replace('        let plan = CorpusPlan::build(words, ids, self, reuse)?;','        let plan = CorpusPlan::build(words, ids, self, reuse)?;\n        eprintln!("BPE_PLAN_SECONDS {:.9}", phase.elapsed().as_secs_f64());')
    s=s.replace('        let mut index = PairIndex::build(&plan, self.min_frequency, workers, reuse, progress)?;','        let phase = std::time::Instant::now();\n        let mut index = PairIndex::build(&plan, self.min_frequency, workers, reuse, progress)?;\n        eprintln!("BPE_INDEX_SECONDS {:.9}", phase.elapsed().as_secs_f64());')
    s=s.replace('            corpus = plan.materialize();','            let phase = std::time::Instant::now();\n            corpus = plan.materialize();\n            eprintln!("BPE_MATERIALIZE_SECONDS {:.9}", phase.elapsed().as_secs_f64());\n            let phase = std::time::Instant::now();')
    s=s.replace('                        return Ok(AttemptOutcome::RestartForReuse);','                        eprintln!("BPE_MERGE_RESTART_SECONDS {:.9}", phase.elapsed().as_secs_f64());\n                        return Ok(AttemptOutcome::RestartForReuse);')
    s=s.replace('            drop(index);\n            drop(corpus);','            eprintln!("BPE_MERGE_SECONDS {:.9}", phase.elapsed().as_secs_f64());\n            let phase = std::time::Instant::now();\n            drop(index);\n            drop(corpus);\n            eprintln!("BPE_DROP_SECONDS {:.9}", phase.elapsed().as_secs_f64());')
    s=s.replace('            drop(index);\n            drop(plan);','            let phase = std::time::Instant::now();\n            drop(index);\n            drop(plan);\n            eprintln!("BPE_DROP_SECONDS {:.9}", phase.elapsed().as_secs_f64());')
    if retire: s=s.replace('drop(index);','index.retire();')
    p.write_text(s)
    snapshot=root/'source'/arm/rel; snapshot.mkdir(parents=True,exist_ok=True)
    for f in (work/rel).glob('*.rs'):shutil.copy2(f,snapshot/f.name)
    shutil.copytree(work/'experiments/bpe-simplification/runner',root/'source'/arm/'experiments/bpe-simplification/runner',dirs_exist_ok=True)
    shutil.copy2(source/'tokenizers/tk-train/Cargo.toml',root/'source'/arm/'tokenizers/tk-train/Cargo.toml')
    if arm == 'child-thinvec':
        with (root/f'{arm}-dependency-lock.log').open('w') as log:
            subprocess.run(['cargo','fetch','--offline'],cwd=work/'experiments/bpe-simplification/runner',env=os.environ|{'PATH':'/root/.cargo/bin:'+os.environ['PATH']},stdout=log,stderr=subprocess.STDOUT,check=True)
        shutil.copy2(work/'experiments/bpe-simplification/runner/Cargo.lock',root/'source'/arm/'experiments/bpe-simplification/runner/Cargo.lock')
    cmd=['cargo','build','--release','--locked','--offline','--target-dir','/root/code/tokenizers-build/bpe-affix-core-build-20261011']
    (root/f'{arm}-command.json').write_text(json.dumps(cmd))
    with (root/f'{arm}-build.log').open('w') as log:
        subprocess.run(cmd,cwd=work/'experiments/bpe-simplification/runner',env=os.environ|{'PATH':'/root/.cargo/bin:'+os.environ['PATH'],'CARGO_INCREMENTAL':'0'},stdout=log,stderr=subprocess.STDOUT,check=True)
    binary=root/f'{arm}-bin';shutil.copy2('/root/code/tokenizers-build/bpe-affix-core-build-20261011/release/bpe-bench-runner',binary)
    print(arm,hashlib.sha256(binary.read_bytes()).hexdigest(),flush=True)
