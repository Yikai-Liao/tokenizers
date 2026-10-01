#!/usr/bin/env python3
"""Collect intrusive posting lifecycle diagnostics; timings are not rankings."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
WHITE = set('\t\n\v\f\r \u0085\u00a0\u1680\u2028\u2029\u202f\u205f\u3000') | set(map(chr, range(0x2000, 0x200b)))


def read_marker(prefix, key):
    path = prefix.with_suffix('.stderr')
    if path.exists():
        text = path.read_text()
    else:
        text = gzip.open(str(path) + '.gz', 'rt').read()
    return next(json.loads(line)[key] for line in text.splitlines() if key in line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/tmp/tokenizers-bpe-bench/data/text'))
    parser.add_argument('--language', choices=['en', 'zh', 'de', 'ja'], required=True)
    parser.add_argument('--size', type=int, choices=[1, 4, 16, 32, 512], required=True)
    parser.add_argument('--split', choices=['none', 'whitespace_split'], required=True)
    parser.add_argument('--rules', type=int, required=True)
    parser.add_argument('--held-out', action='store_true')
    args = parser.parse_args()
    case = f'{args.language}-{args.size}m-{args.split}-r{args.rules}'
    source = args.source / f'{args.language}-{args.size}m.txt'
    text = source.read_text()
    alphabet = set(text)
    if args.split == 'whitespace_split':
        alphabet -= WHITE
    directory = ROOT / '.build/posting-distribution-inputs' / case
    directory.mkdir(parents=True, exist_ok=True)
    corpus = directory / 'input.txt'
    # run_native_fair resolves the corpus path before locating its manifest.
    if corpus.is_symlink():
        corpus.unlink()
    if not corpus.exists():
        shutil.copyfile(source, corpus)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = dict(language=args.language, size_mib_label=args.size,
                    source=str(source.resolve()), sha256=digest, bytes=source.stat().st_size,
                    split=args.split, alphabet_size=len(alphabet), target_rules=args.rules,
                    target_vocab=args.rules + len(alphabet), held_out=args.held_out,
                    revision='b04c8d1ceb2f5cd4588862100d08de323dccfbaa')
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    output = ROOT / 'results/posting-distribution' / (case + '.jsonl')
    if not output.exists():
        command = [sys.executable, str(ROOT / 'run_native_fair.py'), '--case', case,
                   '--worktree', '/root/code/tokenizers-worktrees/prepare-rule-aggregate',
                   '--build-root', str(ROOT / '.build/native-j-posting-distribution'),
                   '--binary', str(ROOT / 'target/release/hf-bpe-native-j-posting-distribution'),
                   '--corpus', str(corpus), '--output', str(output), '--split', args.split,
                   '--vocab', str(manifest['target_vocab']), '--initialization-workers', '4',
                   '--merge-workers', '4', '--atomic-corpus']
        subprocess.run(command, cwd=ROOT, check=True)
    result = json.loads(output.read_text())
    probe = read_marker(output, 'bench_posting_probe')
    inventory = read_marker(output, 'bench_posting_inventory')
    terminal = probe['trace'][-1]
    allocated = sum(terminal['allocated_bytes'])
    retired = sum(terminal['retired_bytes'])
    assert allocated - retired == inventory['heap_payload_bytes']
    assert sum(terminal['allocated_count']) - sum(terminal['retired_count']) == inventory['heap_postings']
    assert probe['summary']['grows'] == 0
    edges = result['indexed_stats']['initial_edges']
    rewrites = probe['summary']['physical_rewrites']
    assert rewrites <= edges
    assert probe['summary']['birth_slots'] <= edges + 2*rewrites
    assert allocated <= 16*edges
    row = dict(input=manifest, actual_rules=result['actual_merges'],
               reached_target=result['actual_merges'] == args.rules,
               physical_symbols=result['indexed_stats']['initial_symbols'], physical_edges=edges,
               initial_pairs=result['indexed_stats']['initial_pairs'], unique_pieces=result['unique_words'],
               model_sha256=result['model_sha256'], inventory=inventory,
               probe_summary=probe['summary'], initialize=probe['trace'][0], terminal=terminal,
               live_length_quantiles=probe['live_length_quantiles'],
               live_capacity_quantiles=probe['live_capacity_quantiles'],
               heap_length_quantiles=probe['heap_length_quantiles'],
               weighted_frequency_quantiles=probe['live_weighted_frequency_quantiles'],
               exact_live_length_histogram=probe['exact_live_length_histogram'],
               survivor_age_counts=probe['survivor_age_counts'], survivor_age_bytes=probe['survivor_age_bytes'],
               physical_entry_bytes=probe['physical_entry_bytes'],
               diagnostic_only=True, invariant_checks='PASS')
    output.with_suffix('.summary.json').write_text(json.dumps(row, indent=2) + '\n')
    with gzip.open(output.with_suffix('.probe.json.gz'), 'wt') as stream:
        json.dump(probe, stream, separators=(',', ':'))
    stderr = output.with_suffix('.stderr')
    if stderr.exists():
        with gzip.open(str(stderr) + '.gz', 'wb') as stream:
            stream.write(stderr.read_bytes())
        stderr.unlink()
    print(f'{case}: rules={result["actual_merges"]}; E={edges}; alloc={allocated}; live={allocated-retired}; checked')


if __name__ == '__main__':
    main()
