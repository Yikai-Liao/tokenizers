#!/usr/bin/env python3
"""One monitored full-training call for the simple per-training arena policy."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
WORKTREE = Path('/root/code/tokenizers-worktrees/arena-dynamic-threshold')
BUILD = ROOT / '.build/native-arena-dynamic-seeded'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['0', '256', 'auto'], required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--vocab', type=int, required=True)
    parser.add_argument('--expected-model', help='required for arena; omitted only when recording a new system baseline')
    parser.add_argument('--split', choices=['none', 'whitespace_split'], default='none')
    args = parser.parse_args()
    if args.mode != '0' and not args.expected_model:
        parser.error('Arena calls require --expected-model from the system baseline or an archived oracle')
    output = ROOT / 'results/arena-dynamic' / (args.case + '.jsonl')
    output.parent.mkdir(parents=True, exist_ok=True)
    extra = {'HF_BPE_ARENA_MODE': args.mode}
    command = [sys.executable, str(ROOT / 'run_native_fair.py'), '--case', args.case,
               '--worktree', str(WORKTREE), '--build-root', str(BUILD),
               '--binary', str(BUILD / 'target/release/hf-bpe-native-arena-dynamic-seeded'),
               '--corpus', str(args.corpus), '--output', str(output), '--split', args.split,
               '--vocab', str(args.vocab), '--initialization-workers', '4',
               '--merge-workers', '4', '--atomic-corpus',
               '--require-stats', 'posting_arena_cutoff_bytes']
    control = dict(command=command, extra_environment=extra, count=1,
                   expected_model_sha256=args.expected_model,
                   word_order='fixed feed hash seeds (11, 13, 17, 19); serial feed; same diagnostic binary for every policy',
                   scope='full train API includes posting drop and final arena release; no concurrent CPU tasks')
    output.with_suffix('.control.json').write_text(json.dumps(control, indent=2) + '\n')
    subprocess.run(command, cwd=ROOT, env=dict(os.environ, **extra), check=True)
    row = json.loads(output.read_text())
    stats = row['indexed_stats']
    allocations = stats['posting_allocations']
    expected_cutoff = {'0': 0, '256': 256}.get(args.mode)
    if expected_cutoff is None:
        from math import isqrt
        expected_cutoff = max(256, isqrt(stats['initial_edges'] // 256))
    assert stats['posting_arena_cutoff_bytes'] == expected_cutoff
    assert allocations['arena_requested_bytes'] == allocations['arena_retired_bytes']
    assert allocations['heap_requested_bytes'] == allocations['heap_freed_bytes']
    assert allocations['heap_buffers'] == allocations['heap_frees']
    if args.expected_model:
        assert row['model_sha256'] == args.expected_model
    assert row['memory']['sampled_peak_process_swap_bytes'] == 0
    if args.mode == '0':
        assert allocations['arena_buffers'] == allocations['arena_requested_bytes'] == 0
    total_buffers = allocations['arena_buffers'] + allocations['heap_buffers']
    summary = dict(control=control, cutoff_bytes=expected_cutoff,
                   allocator_origin_gate='PASS', model_gate='PASS' if args.expected_model else 'SYSTEM BASELINE RECORDED', no_process_swap_gate='PASS',
                   allocations=allocations,
                   arena_buffer_fraction=allocations['arena_buffers'] / total_buffers if total_buffers else 0,
                   train_ms=row['train_ms'], initialize_ms=row['initialize_ms'], merge_ms=row['merge_ms'],
                   full_peak_rss_bytes=max(row['maxrss_kib'] * 1024,
                                           row['memory']['sampled_peak_rss_bytes']))
    output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f"{args.case}: model/lifetime/swap gates PASS; threshold {expected_cutoff} B")


if __name__ == '__main__':
    main()
