#!/usr/bin/env python3
"""Insert stage clocks into disposable copies; never patch source checkouts."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PR_HEAD = '6ac0de5359d9e0e1ed0608422575a360ef91b908'


def instrument(path):
    source = path.read_text()
    if '__hf_bench_clock' in source:
        raise ValueError('source already instrumented')
    anchor = '        let progress = self.setup_progress();'
    if source.count(anchor) != 1:
        raise ValueError('progress anchor is ambiguous')
    source = source.replace(anchor, anchor + '\n        let mut __hf_bench_clock = std::time::Instant::now();', 1)

    def emit(stage):
        return ('        eprintln!(r#"{{"bench_stage":"%s","ms":{}}}"#, '
                '__hf_bench_clock.elapsed().as_secs_f64() * 1000.0);\n'
                '        __hf_bench_clock = std::time::Instant::now();\n') % stage

    for number, stage in [(2, 'special_tokens'), (3, 'alphabet'), (4, 'tokenize_words'), (5, 'count_pairs')]:
        marker = f'        // {number}. '
        position = source.index(marker, source.index('__hf_bench_clock'))
        source = source[:position] + emit(stage) + source[position:]
    match = re.search(r'        self\.finalize_progress\(&progress, merges\.len\(\), "Compute merges"\);', source)
    if not match:
        raise ValueError('merge completion anchor missing')
    source = source[:match.end()] + '\n' + emit('merges') + source[match.end():]
    anchor = '        Ok((vocab, merges, self.special_tokens.clone()))'
    if source.count(anchor) != 1:
        raise ValueError('result anchor is ambiguous')
    # No reset after the final timer; keep compiler warnings focused on actual code.
    final = emit('finalize_model').replace('        __hf_bench_clock = std::time::Instant::now();\n', '')
    path.write_text(source.replace(anchor, final + anchor, 1))


def build(label, source, indexed, training_workers=None):
    root = ROOT / f'.build/profiled-{label}'
    copied = root / 'source/tokenizers'
    shutil.copytree(source / 'tokenizers', copied, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('target', '.git', 'data'))
    instrument(copied / 'tk-train/src/trainers/bpe/mod.rs')
    runner = root / 'runner'
    (runner / 'src').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / 'src/main.rs', runner / 'src/main.rs')
    if training_workers is not None:
        path = runner / 'src/main.rs'
        text = path.read_text()
        anchor = '    let reader = BufReader::new(File::open(input)?);'
        assert text.count(anchor) == 1
        text = text.replace(anchor, f'    tk_encode::parallelism::set_num_threads({training_workers});\n'
                            '    tk_encode::parallelism::set_parallelism(false);\n' + anchor)
        anchor = '    let feed_ms = begin.elapsed().as_secs_f64() * 1000.0;'
        assert text.count(anchor) == 1
        text = text.replace(anchor, anchor + '\n    tk_encode::parallelism::set_parallelism(true);')
        path.write_text(text)
    shutil.copyfile(ROOT / 'Cargo.lock', runner / 'Cargo.lock')
    default = 'default = ["indexed"]\n' if indexed else ''
    worker_dependency = (f'tk-encode = {{ path = {json.dumps(str(copied / "tk-encode"))}, default-features = false }}\n'
                         if training_workers is not None else '')
    runner.joinpath('Cargo.toml').write_text(f'''[package]
name = "hf-bpe-profiled-{label}"
version = "0.1.0"
edition = "2024"
[features]
{default}indexed = []
[dependencies]
tk-train = {{ path = {json.dumps(str(copied / 'tk-train'))}, default-features = false }}
{worker_dependency}serde_json = "1"
sha2 = "0.10"
hf-front-end = {{ package = "tokenizers", version = "=0.23.2", default-features = false, features = ["fancy-regex"] }}
''')
    cargo = os.environ.get('CARGO', str(Path.home() / '.cargo/bin/cargo'))
    subprocess.run([cargo, 'build', '--release', '--manifest-path', str(runner / 'Cargo.toml')], check=True,
                   env=dict(os.environ, CARGO_TARGET_DIR=str(ROOT / 'target')))
    return ROOT / f'target/release/hf-bpe-profiled-{label}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pr_checkout', type=Path)
    args = parser.parse_args()
    pr = args.pr_checkout.resolve()
    sha = subprocess.check_output(['git', '-C', str(pr), 'rev-parse', 'HEAD'], text=True).strip()
    if sha != PR_HEAD or subprocess.check_output(['git', '-C', str(pr), 'status', '--porcelain'], text=True).strip():
        raise SystemExit('PR checkout must be unmodified at pinned PR head')
    for label, source, indexed in [('current', REPO, True), ('pr', pr, False)]:
        print(build(label, source, indexed), flush=True)


if __name__ == '__main__':
    main()
