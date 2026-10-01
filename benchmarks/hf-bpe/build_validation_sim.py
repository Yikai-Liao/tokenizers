#!/usr/bin/env python3
"""Compile a simulation using the committed trainer's real queue implementation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('worktree', type=Path)
    args = p.parse_args()
    worktree = args.worktree.resolve()
    assert not subprocess.check_output(['git', '-C', str(worktree), 'status', '--porcelain'], text=True).strip()
    source_commit = subprocess.check_output(['git', '-C', str(worktree), 'rev-parse', 'HEAD'], text=True).strip()
    source = worktree / 'tokenizers/tk-train/src/trainers/bpe/indexed/parallel.rs'
    text = source.read_text()
    build = ROOT / '.build/validation-window-sim'
    (build / 'src').mkdir(parents=True, exist_ok=True)
    snippets = text[text.index('fn key('):text.index('struct Block<')]
    # Only empty postings are needed. Keep the production record's 16-byte
    # posting payload and its 32-byte key/value bucket layout.
    prefix = '''use ahash::AHashMap;
use std::cmp::Ordering;
use dary_heap::OctonaryHeap;
use validation_window::{Frontier, SelectionMode, Window};
use candidate_heap::CandidateHeap;
#[derive(Default)]
struct SmallPosting { _storage: [u64; 2] }
type Pair = (u32, u32);
'''
    prefix += '#[path = "candidate_heap.rs"] mod candidate_heap;\n'
    prefix += '#[path = "validation_window.rs"] mod validation_window;\n'
    driver = ROOT / 'validation_window_sim.rs'
    (build / 'src/main.rs').write_text(prefix + snippets + driver.read_text())
    for name in ['candidate_heap.rs', 'validation_window.rs']:
        shutil.copyfile(source.parent / 'parallel' / name, build / 'src' / name)
    shutil.copyfile(ROOT / 'Cargo.lock', build / 'Cargo.lock')
    (build / 'Cargo.toml').write_text('''[package]
name = "validation-window-sim"
version = "0.1.0"
edition = "2024"
[dependencies]
ahash = "0.8.11"
dary_heap = "0.3.6"
rayon = "1.10"
serde_json = "1"
sha2 = "0.10"
''')
    subprocess.run([str(Path.home() / '.cargo/bin/cargo'), 'build', '--offline', '--release',
        '--manifest-path', str(build / 'Cargo.toml')], check=True,
        env=dict(os.environ, CARGO_TARGET_DIR=str(build / 'target')))
    binary = build / 'target/release/validation-window-sim'
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = dict(source_commit=source_commit, source_sha256=sha(source),
        window_sha256=sha(source.parent / 'parallel/validation_window.rs'),
        heap_sha256=sha(source.parent / 'parallel/candidate_heap.rs'),
        driver_sha256=sha(driver), generated_main_sha256=sha(build / 'src/main.rs'),
        binary_sha256=sha(binary), rustc=subprocess.check_output(
            [str(Path.home() / '.cargo/bin/rustc'), '-Vv'], text=True),
        empty_posting_stub_bytes=16, actual_owner_ledger_and_heap=True)
    (build / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(binary)


if __name__ == '__main__':
    main()
