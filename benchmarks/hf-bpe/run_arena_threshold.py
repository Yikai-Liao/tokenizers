#!/usr/bin/env python3
"""Measure one cutoff on the shared arena/heap diagnostic binary."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutoff', required=True, help='payload bytes, or all')
    parser.add_argument('--case', required=True)
    parser.add_argument('--corpus', type=Path, default=ROOT / '.build/gb-corpus/zh-512m.txt')
    parser.add_argument('--vocab', type=int, default=50000)
    parser.add_argument('--split', default='none')
    args = parser.parse_args()
    cutoff = sys.maxsize * 2 + 1 if args.cutoff == 'all' else int(args.cutoff)
    if cutoff < 0:
        raise SystemExit('negative cutoff')
    output = ROOT / 'results/arena-threshold' / (args.case + '.jsonl')
    command = [sys.executable, str(ROOT / 'run_native_fair.py'), '--case', args.case,
               '--worktree', '/root/code/tokenizers-worktrees/prepare-rule-aggregate',
               '--build-root', str(ROOT / '.build/native-j-arena-threshold'),
               '--binary', str(ROOT / 'target/release/hf-bpe-native-j-arena-threshold'),
               '--corpus', str(args.corpus), '--output', str(output), '--split', args.split,
               '--vocab', str(args.vocab), '--initialization-workers', '4',
               '--merge-workers', '4', '--atomic-corpus']
    output.parent.mkdir(parents=True, exist_ok=True)
    control = dict(command=command, extra_environment={'HF_BPE_ARENA_MAX_BYTES': str(cutoff)},
                   cutoff_bytes=cutoff, cutoff_label=args.cutoff,
                   count=1, scope='same binary cutoff sensitivity; no concurrent CPU tasks')
    output.with_suffix('.control.json').write_text(json.dumps(control, indent=2) + '\n')
    subprocess.run(command, cwd=ROOT,
                   env=dict(os.environ, HF_BPE_ARENA_MAX_BYTES=str(cutoff)), check=True)
    stderr = output.with_suffix('.stderr').read_text().splitlines()
    markers = [json.loads(line) for line in stderr if line.startswith('{')]
    arena = next(x['bench_arena_stats'] for x in markers if 'bench_arena_stats' in x)
    phase = next(x['bench_arena_phase'] for x in markers if 'bench_arena_phase' in x)
    inventory = next(x['bench_posting_inventory'] for x in markers if 'bench_posting_inventory' in x)
    assert arena['cutoff_bytes'] == phase['cutoff_bytes'] == cutoff
    totals = {key: sum(worker[key] for worker in arena['arenas']) for key in arena['arenas'][0]}
    assert totals['grows'] == 0
    assert totals['system_allocations'] == totals['system_frees']
    assert totals['system_requested_bytes'] == totals['system_freed_bytes']
    if cutoff == 0:
        assert totals['allocations'] == totals['requested_bytes'] == 0
    if args.cutoff == 'all':
        assert totals['system_allocations'] == totals['system_requested_bytes'] == 0
    output.with_suffix('.summary.json').write_text(json.dumps(
        dict(control=control, arena=arena, totals=totals, phase=phase, inventory=inventory,
             allocator_origin_checks='PASS'), indent=2) + '\n')
    print(f'{args.case}: cutoff {args.cutoff}, allocator-origin checks PASS')


if __name__ == '__main__':
    main()
