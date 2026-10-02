#!/usr/bin/env python3
"""Run one isolated six-algorithm case with model, lifetime and swap gates."""
import argparse
import json
from math import isqrt
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FLAGS = {'boundaries', 'lazy-counts', 'lazy-births', 'positions', 'hot256', 'hot512', 'hot1024', 'final'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--algorithms', required=True)
    p.add_argument('--reference', action='store_true')
    p.add_argument('--workers', type=int, choices=[1, 4], default=4)
    p.add_argument('--worktree', type=Path, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--case', required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--vocab', type=int, required=True)
    p.add_argument('--expected-model', required=True)
    p.add_argument('--result-dir', type=Path, default=ROOT / 'results/algorithmic-ablation')
    args = p.parse_args()
    flags = set(args.algorithms.split(','))
    if flags == {'none'}:
        flags.clear()
    elif flags == {'all'}:
        flags = {'boundaries', 'lazy-counts', 'lazy-births', 'positions', 'hot256', 'final'}
    if not flags <= FLAGS or len(flags & {'hot256', 'hot512', 'hot1024'}) > 1:
        p.error('invalid algorithm combination')
    if args.reference and flags:
        p.error('unchanged reference cannot enable candidates')
    build = ROOT / '.build' / ('native-' + args.label)
    output = args.result_dir.resolve() / (args.case + '.jsonl')
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
    environment = dict(HF_BPE_ARENA_MODE='auto', HF_BPE_QUEUE_MODE='auto', HF_BPE_WEIGHT_ORDER='sorted',
        HF_BPE_ALGORITHMS=args.algorithms)
    control = dict(command=command, extra_environment=environment, expected_model=args.expected_model,
        count=1, diagnostic_only=False, workers=args.workers, algorithms=sorted(flags),
        unchanged_reference=args.reference, word_order='serial seeded feed (11,13,17,19)',
        scope='complete train includes candidate setup, scans, certification and reconstruction')
    output.with_suffix('.control.json').write_text(json.dumps(control, indent=2) + '\n')
    subprocess.run(command, cwd=ROOT, env=dict(os.environ, **environment), check=True)
    row = json.loads(output.read_text())
    s = row['indexed_stats']
    a = s['posting_allocations']
    assert row['model_sha256'] == args.expected_model
    assert s['queue_selection_mode'] == 'bulk4' and s['corpus_weight_order'] == 'weight_sorted'
    assert s['workers'] == s['initialization_workers'] == args.workers
    assert s['atomic_corpus'] and s['layout'] == 'parallel_u32_flat32'
    assert s['posting_arena_cutoff_bytes'] == max(256, isqrt(s['initial_edges'] // 256))
    assert a['arena_requested_bytes'] == a['arena_retired_bytes']
    assert a['heap_requested_bytes'] == a['heap_freed_bytes'] and a['heap_buffers'] == a['heap_frees']
    assert row['memory']['sampled_peak_process_swap_bytes'] == 0
    assert row['memory']['minimum_available_bytes'] >= 1024**3
    if not args.reference:
        for name, field in [('boundaries', 'permanent_boundaries'), ('lazy-counts', 'lazy_counts'),
                            ('lazy-births', 'lazy_births'), ('positions', 'positional_batches')]:
            assert s[field] == (name in flags), (name, s[field], flags)
        assert s['skipped_final_batches'] == 0 if 'final' not in flags else s['skipped_final_batches'] <= 1
        assert bool(s['initial_hot_characters']) == bool(flags & {'hot256', 'hot512', 'hot1024'})
    summary = dict(algorithms=sorted(flags), workers=args.workers, unchanged_reference=args.reference,
        train_ms=row['train_ms'], initialize_ms=s['initialize_ms'], merge_ms=s['merge_ms'],
        full_peak_rss_bytes=max(row['maxrss_kib'] * 1024, row['memory']['sampled_peak_rss_bytes']),
        gates='PASS')
    output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(args.case + ': model/lifetime/resource gates PASS', flush=True)


if __name__ == '__main__':
    main()
