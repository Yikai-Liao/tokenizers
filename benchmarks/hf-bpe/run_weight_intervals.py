#!/usr/bin/env python3
"""Compare original bitmap metadata and sorted weight intervals through the public API."""
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
    p.add_argument('--mode', choices=['original', 'sorted', 'stable'], required=True)
    p.add_argument('--reference', action='store_true', help='unchanged unified heap trainer without weight-order dispatch')
    p.add_argument('--workers', type=int, choices=[1,4], default=4)
    p.add_argument('--worktree', type=Path, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--case', required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--vocab', type=int, required=True)
    p.add_argument('--expected-model', required=True)
    args = p.parse_args()
    build = ROOT / '.build' / ('native-' + args.label)
    output = ROOT / 'results/weight-intervals' / (args.case + '.jsonl')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        p.error('case already exists')
    command = [sys.executable, str(ROOT / 'run_native_fair.py'), '--case', args.case,
        '--worktree', str(args.worktree.resolve()), '--build-root', str(build),
        '--binary', str(build / 'target/release' / ('hf-bpe-native-' + args.label)),
        '--corpus', str(args.corpus.resolve()), '--output', str(output), '--split', 'none',
        '--vocab', str(args.vocab), '--initialization-workers', str(args.workers), '--merge-workers', str(args.workers),
        '--atomic-corpus', '--expected-layout', 'parallel_u32_flat32',
        '--require-stats', 'posting_arena_cutoff_bytes']
    control = dict(command=command, extra_environment=dict(HF_BPE_ARENA_MODE='auto', HF_BPE_QUEUE_MODE='auto', HF_BPE_WEIGHT_ORDER=args.mode),
        expected_model=args.expected_model, count=1, diagnostic_only=False,
        word_order='serial feed; hash seeds (11,13,17,19)', queue_mode='bulk4', weight_order=args.mode,
        workers=args.workers, same_binary_controls=not args.reference, unchanged_reference=args.reference)
    output.with_suffix('.control.json').write_text(json.dumps(control, indent=2) + '\n')
    subprocess.run(command, cwd=ROOT, env=dict(os.environ, HF_BPE_ARENA_MODE='auto', HF_BPE_QUEUE_MODE='auto', HF_BPE_WEIGHT_ORDER=args.mode), check=True)
    row = json.loads(output.read_text())
    s = row['indexed_stats']
    assert s['queue_selection_mode'] == 'bulk4'
    if not args.reference:
        assert s['corpus_weight_order'] == ('original' if args.mode == 'original' else 'weight_sorted')
    if args.mode != 'original' and not args.reference:
        assert s['weight_one_bucket_bytes'] == s['weight_lookup_bytes'] == 0
    a = s['posting_allocations']
    assert row['model_sha256'] == args.expected_model
    assert s['workers'] == s['initialization_workers'] == args.workers
    assert s['atomic_corpus'] and s['layout'] == 'parallel_u32_flat32'
    assert s['posting_arena_cutoff_bytes'] == max(256, isqrt(s['initial_edges'] // 256))
    assert a['arena_requested_bytes'] == a['arena_retired_bytes']
    assert a['heap_requested_bytes'] == a['heap_freed_bytes']
    assert a['heap_buffers'] == a['heap_frees']
    assert row['memory']['sampled_peak_process_swap_bytes'] == 0
    summary = dict(mode='unchanged' if args.reference else args.mode, workers=args.workers, diagnostic_only=False, bits=32,
        train_ms=row['train_ms'], initialize_ms=s['initialize_ms'], merge_ms=s['merge_ms'], sort_ms=s.get('corpus_sort_ms', 0),
        weight_interval_count=s.get('weight_interval_count'), weight_bytes=s['initial_weight_bytes'],
        lookup_bytes=s['weight_lookup_bytes'], temporary_weight_bytes=s.get('corpus_temporary_weight_bytes'),
        full_peak_rss_bytes=max(row['maxrss_kib'] * 1024, row['memory']['sampled_peak_rss_bytes']),
        worker_task_occupancy=None, visits_balance_efficiency=None,
        model_lifetime_swap_gates='PASS')
    output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(args.case + ': standalone model/lifetime/swap gates PASS', flush=True)


if __name__ == '__main__':
    main()
