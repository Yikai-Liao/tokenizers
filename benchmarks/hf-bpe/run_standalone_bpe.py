#!/usr/bin/env python3
"""Validate a standalone baseline or dense-birth trainer through the public API."""
import argparse
import json
import os
from math import isqrt
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=['baseline', 'dense'], required=True)
    p.add_argument('--worktree', type=Path, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--case', required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--vocab', type=int, required=True)
    p.add_argument('--expected-model', required=True)
    args = p.parse_args()
    build = ROOT / '.build' / ('native-' + args.label)
    output = ROOT / 'results/fused-rewrite' / (args.case + '.jsonl')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        p.error('case already exists')
    command = [sys.executable, str(ROOT / 'run_native_fair.py'), '--case', args.case,
        '--worktree', str(args.worktree.resolve()), '--build-root', str(build),
        '--binary', str(build / 'target/release' / ('hf-bpe-native-' + args.label)),
        '--corpus', str(args.corpus.resolve()), '--output', str(output), '--split', 'none',
        '--vocab', str(args.vocab), '--initialization-workers', '4', '--merge-workers', '4',
        '--atomic-corpus', '--expected-layout', 'parallel_u32_flat32',
        '--require-stats', 'posting_arena_cutoff_bytes']
    control = dict(command=command, extra_environment=dict(HF_BPE_ARENA_MODE='auto'),
        expected_model=args.expected_model, count=1, diagnostic_only=False,
        word_order='serial feed; hash seeds (11,13,17,19)', variant=args.variant)
    output.with_suffix('.control.json').write_text(json.dumps(control, indent=2) + '\n')
    subprocess.run(command, cwd=ROOT, env=dict(os.environ, HF_BPE_ARENA_MODE='auto'), check=True)
    row = json.loads(output.read_text())
    s = row['indexed_stats']
    a = s['posting_allocations']
    assert row['model_sha256'] == args.expected_model
    assert s['workers'] == s['initialization_workers'] == 4
    assert s['atomic_corpus'] and s['layout'] == 'parallel_u32_flat32'
    assert s['posting_arena_cutoff_bytes'] == max(256, isqrt(s['initial_edges'] // 256))
    assert a['arena_requested_bytes'] == a['arena_retired_bytes']
    assert a['heap_requested_bytes'] == a['heap_freed_bytes']
    assert a['heap_buffers'] == a['heap_frees']
    assert row['memory']['sampled_peak_process_swap_bytes'] == 0
    summary = dict(mode='native-' + args.variant, diagnostic_only=False, bits=32,
        train_ms=row['train_ms'], initialize_ms=s['initialize_ms'], merge_ms=s['merge_ms'],
        full_peak_rss_bytes=max(row['maxrss_kib'] * 1024, row['memory']['sampled_peak_rss_bytes']),
        worker_task_occupancy=None, visits_balance_efficiency=None,
        model_lifetime_swap_gates='PASS')
    output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(args.case + ': standalone model/lifetime/swap gates PASS', flush=True)


if __name__ == '__main__':
    main()
