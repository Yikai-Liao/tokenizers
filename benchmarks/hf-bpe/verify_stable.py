#!/usr/bin/env python3
"""Check shared weighted inputs against the pinned, unmodified HF stable source."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
STABLE = '88a4498ad4ea1a9487b0a9b0ff881383fd5a06a3'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkout', type=Path)
    args = parser.parse_args()
    source = args.checkout.resolve()
    sha = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if sha != STABLE:
        raise SystemExit(f'expected stable {STABLE}, got {sha}')
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip():
        raise SystemExit('stable checkout must be unmodified')
    build = ROOT / '.build/stable-verify'
    (build / 'src').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / 'verify_stable.rs', build / 'src/main.rs')
    build.joinpath('Cargo.toml').write_text(f'''[package]
name = "hf-bpe-stable-verify"
version = "0.1.0"
edition = "2024"
[dependencies]
tk-train = {{ path = {json.dumps(str(ROOT.parents[1] / 'tokenizers/tk-train'))}, default-features = false }}
tk-encode = {{ path = {json.dumps(str(ROOT.parents[1] / 'tokenizers/tk-encode'))}, default-features = false, features = ["bpe"] }}
tokenizers = {{ path = {json.dumps(str(source / 'tokenizers'))}, default-features = false, features = ["fancy-regex"] }}
serde_json = "1"
ahash = "0.8.12"
compact_str = "0.9"
tempfile = "3"
''')
    cargo = os.environ.get('CARGO', str(Path.home() / '.cargo/bin/cargo'))
    subprocess.run([cargo, 'run', '--release', '--manifest-path', str(build / 'Cargo.toml')], check=True,
                   env=dict(os.environ, CARGO_TARGET_DIR=str(ROOT / 'target'), TOKENIZERS_PARALLELISM='false'))


if __name__ == '__main__':
    main()
