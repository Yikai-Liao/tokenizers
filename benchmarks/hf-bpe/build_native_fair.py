#!/usr/bin/env python3
"""Build a committed worktree through its original training API with one stats probe."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import os

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('worktree',type=Path)
    parser.add_argument('--label',required=True)
    args = parser.parse_args()
    worktree=args.worktree.resolve()
    if subprocess.check_output(['git','-C',str(worktree),'status','--porcelain'],text=True).strip():
        raise SystemExit('worktree must be committed and clean')
    build=ROOT/'.build'/f'native-{args.label}'
    source=build/'source/tokenizers'
    shutil.copytree(worktree/'tokenizers',source,dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('target','.git','data'))
    trainer=source/'tk-train/src/trainers/bpe/mod.rs'
    text=trainer.read_text()
    anchor='        Ok((trained.vocab, trained.merges, trained.special_tokens))'
    assert text.count(anchor)==1
    trainer.write_text(text.replace(anchor,
        '        eprintln!("{}", serde_json::json!({"bench_indexed_stats": trained.stats}));\n'+anchor))
    runner=build/'runner'
    (runner/'src').mkdir(parents=True,exist_ok=True)
    text=(ROOT/'src/main.rs').read_text()
    anchor='    let reader = BufReader::new(File::open(input)?);'
    assert text.count(anchor)==1
    text=text.replace(anchor,'    let bench_workers = env::var("HF_BPE_BENCH_WORKERS")\n'
                            '        .map(|value| value.parse::<usize>().expect("invalid benchmark worker count"))\n'
                            '        .unwrap_or(4);\n'
                            '    assert!(matches!(bench_workers, 1 | 4), "benchmark workers must be 1 or 4");\n'
                            '    tk_encode::parallelism::set_num_threads(bench_workers);\n'
                            '    tk_encode::parallelism::set_parallelism(false);\n'+anchor)
    anchor='    let feed_ms = begin.elapsed().as_secs_f64() * 1000.0;'
    assert text.count(anchor)==1
    text=text.replace(anchor,anchor+'\n    tk_encode::parallelism::set_parallelism(true);')
    (runner/'src/main.rs').write_text(text)
    shutil.copyfile(ROOT/'Cargo.lock',runner/'Cargo.lock')
    (runner/'Cargo.toml').write_text(f'''[package]
name = "hf-bpe-native-{args.label}"
version = "0.1.0"
edition = "2024"
[features]
indexed = []
[dependencies]
tk-train = {{ path = {json.dumps(str(source/'tk-train'))}, default-features = false }}
tk-encode = {{ path = {json.dumps(str(source/'tk-encode'))}, default-features = false }}
serde_json = "1"
sha2 = "0.10"
hf-front-end = {{ package = "tokenizers", version = "=0.23.2", default-features = false, features = ["fancy-regex"] }}
''')
    subprocess.run([str(Path.home()/'.cargo/bin/cargo'),'build','--offline','--release','--manifest-path',str(runner/'Cargo.toml')],
                   check=True,env=dict(os.environ,CARGO_TARGET_DIR=str(ROOT/'target')))
    print(ROOT/f'target/release/hf-bpe-native-{args.label}')

if __name__=='__main__':
    main()
