#!/usr/bin/env python3
"""Build the dynamic-arena candidate with diagnostic policy selection and fixed hash seeds."""
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
    parser.add_argument('--label', default='arena-dynamic-seeded')
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
    parallel = source/'tk-train/src/trainers/bpe/indexed/parallel.rs'
    text = parallel.read_text()
    anchor = '    train_with_policy(trainer, wc, config, super::posting_arena::Policy::Auto)'
    assert text.count(anchor) == 1
    replacement = """    let policy = match std::env::var("HF_BPE_ARENA_MODE").as_deref() {
        Ok("0") => super::posting_arena::Policy::System,
        Ok("256") => super::posting_arena::Policy::Fixed(256),
        Ok("auto") | Err(_) => super::posting_arena::Policy::Auto,
        _ => panic!("invalid diagnostic arena policy"),
    };
    train_with_policy(trainer, wc, config, policy)"""
    parallel.write_text(text.replace(anchor, replacement))
    # Seed feed maps before corpus construction. Feed is serial in the fair
    # runner, so iteration is reproducible without sorting inside train timing.
    text = trainer.read_text()
    start = text.index('    fn feed<I, S, F>')
    end = text.index('\n#[cfg(test)]', start)
    feed = text[start:end]
    assert feed.count('AHashMap::new()') == 2
    feed = feed.replace('AHashMap::new()',
        'AHashMap::with_hasher(ahash::RandomState::with_seeds(11, 13, 17, 19))')
    trainer.write_text(text[:start] + feed + text[end:])
    # Preserve every overlay change, including the existing scalar stats probe.
    import difflib
    patch = []
    for candidate in source.rglob('*.rs'):
        original = worktree/'tokenizers'/candidate.relative_to(source)
        if original.exists():
            patch.extend(difflib.unified_diff(original.read_text().splitlines(True),
                candidate.read_text().splitlines(True), fromfile=str(original), tofile=str(candidate)))
    (build/'instrumentation.patch').write_text(''.join(patch))
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
                   check=True,env=dict(os.environ,CARGO_TARGET_DIR=str(build/'target')))
    print(build/f'target/release/hf-bpe-native-{args.label}')

if __name__=='__main__':
    main()
