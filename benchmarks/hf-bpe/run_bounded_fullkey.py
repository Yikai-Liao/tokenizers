#!/usr/bin/env python3
"""Same-binary full training comparison of sparse hash and bounded full-key radix initialization."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm', choices=['hash', 'bounded'], required=True)
    parser.add_argument('--input-case', required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--block-bits', type=int, choices=[16,20,24], default=24)
    args = parser.parse_args()
    corpus = ROOT/'.build/posting-distribution-inputs'/args.input_case/'input.txt'
    manifest = json.loads(corpus.with_name('manifest.json').read_text())
    output = ROOT/'results/bounded-fullkey'/(args.case+'.jsonl')
    output.parent.mkdir(parents=True, exist_ok=True)
    extra = dict(HF_BPE_BLOCK_INIT_ALGO=args.algorithm, HF_BPE_BLOCK_PROXY_BITS=str(args.block_bits))
    command = [sys.executable, str(ROOT/'run_native_fair.py'), '--case', args.case,
        '--worktree', '/root/code/tokenizers-worktrees/initial-owner-waves',
        '--build-root', str(ROOT/'.build/native-j-bounded-fullkey'),
        '--binary', str(ROOT/'target/release/hf-bpe-native-j-bounded-fullkey'),
        '--corpus', str(corpus), '--output', str(output), '--split', manifest['split'],
        '--vocab', str(manifest['target_vocab']), '--initialization-workers','4',
        '--merge-workers','4','--atomic-corpus','--expected-layout',
        'parallel_u32_dict16' if args.block_bits == 16 else 'parallel_u32_dict32',
        '--require-stats','initial_bounded_tiles','--require-stats','initial_bounded_groups',
        '--require-stats','initial_bounded_sort_buffer_bound_bytes']
    control = dict(command=command, extra_environment=extra, algorithm=args.algorithm,
        block_bits=args.block_bits, local_offset_bytes=4, word_order='lexical (diagnostic only)',
        allocator='original system allocator', summary_wave='initialization concurrency (4)',
        tile_records=262144, radix_record_bytes=16, radix_block_records=128,
        count=1, scope='full training, small block span proxy; not a measurement above 2^32 corpus positions')
    output.with_suffix('.control.json').write_text(json.dumps(control,indent=2)+'\n')
    subprocess.run(command,cwd=ROOT,env=dict(os.environ,**extra),check=True)
    row = json.loads(output.read_text()); stats = row['indexed_stats']
    expected = json.loads((ROOT/'results/posting-distribution'/(args.input_case+'.jsonl')).read_text())
    assert row['model_sha256'] == expected['model_sha256']
    backend = 'spatial_block_bounded_radix64' if args.algorithm == 'bounded' else 'spatial_block_sparse_weights'
    assert stats['initial_count_backend'] == backend and stats['initial_blocks'] >= 1
    assert stats['initial_summary_waves'] == (stats['initial_blocks']+3)//4
    if args.algorithm == 'bounded':
        assert stats['initial_bounded_tiles'] > 0
        assert stats['initial_bounded_groups'] <= stats['initial_edges']
        assert stats['initial_bounded_sort_buffer_bound_bytes'] <= 4*(262144*16 + 512*128*16 +(262144//128+512)*9)
    else:
        assert stats['initial_bounded_tiles'] == stats['initial_bounded_groups'] == 0
        assert stats['initial_bounded_sort_buffer_bound_bytes'] == 0
    output.with_suffix('.summary.json').write_text(json.dumps(dict(control=control,model_gate='PASS',
        expected_model_sha256=expected['model_sha256'],posting_layout_gate='u32 selected before block span override'),indent=2)+'\n')
    print(f'{args.case}: full model gate PASS; {stats["initial_bounded_groups"]} group dictionary queries')

if __name__ == '__main__':
    main()
