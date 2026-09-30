#!/usr/bin/env python3
"""Build an unmodified pinned PR checkout with the same benchmark source."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

PR_HEAD = '6ac0de5359d9e0e1ed0608422575a360ef91b908'
ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkout', type=Path)
    args = parser.parse_args()
    source = args.checkout.resolve()
    sha = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if sha != PR_HEAD:
        raise SystemExit(f'expected PR head {PR_HEAD}, got {sha}')
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip():
        raise SystemExit('PR checkout must be unmodified')
    build = ROOT / '.build/pr-runner'
    (build / 'src').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / 'src/main.rs', build / 'src/main.rs')
    shutil.copyfile(ROOT / 'Cargo.lock', build / 'Cargo.lock')
    build.joinpath('Cargo.toml').write_text(f'''[package]
name = "hf-bpe-pr-head-bench"
version = "0.1.0"
edition = "2024"
[features]
indexed = []
[dependencies]
tk-train = {{ path = {json.dumps(str(source / 'tokenizers/tk-train'))}, default-features = false }}
serde_json = "1"
sha2 = "0.10"
hf-front-end = {{ package = "tokenizers", version = "=0.23.2", default-features = false, features = ["fancy-regex"] }}
''')
    cargo = os.environ.get('CARGO', str(Path.home() / '.cargo/bin/cargo'))
    subprocess.run([cargo, 'build', '--release', '--manifest-path', str(build / 'Cargo.toml')], check=True,
                   env=dict(os.environ, CARGO_TARGET_DIR=str(ROOT / 'target')))
    print(ROOT / 'target/release/hf-bpe-pr-head-bench')


if __name__ == '__main__':
    main()
