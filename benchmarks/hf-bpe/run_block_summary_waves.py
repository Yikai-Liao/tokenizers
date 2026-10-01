#!/usr/bin/env python3
"""Compare complete vs bounded block-key summary waves at fixed physical word order."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['legacy','sparse'],required=True)
    parser.add_argument('--input-case',required=True)
    parser.add_argument('--case',required=True)
    parser.add_argument('--block-bits',type=int,choices=[16,20,24,32],default=20)
    parser.add_argument('--wave', choices=['all','4'], required=True)
    args=parser.parse_args()
    corpus=ROOT/'.build/posting-distribution-inputs'/args.input_case/'input.txt'
    manifest=json.loads(corpus.with_name('manifest.json').read_text())
    output=ROOT/'results/block-summary-waves'/(args.case+'.jsonl')
    output.parent.mkdir(parents=True,exist_ok=True)
    env=dict(HF_BPE_BLOCK_COUNT_MODE=args.mode,HF_BPE_BLOCK_PROXY_BITS=str(args.block_bits),HF_BPE_BLOCK_SUMMARY_WAVE=args.wave)
    command=[sys.executable,str(ROOT/'run_native_fair.py'),'--case',args.case,
        '--worktree','/root/code/tokenizers-worktrees/initial-owner-waves',
        '--build-root',str(ROOT/'.build/native-j-block-summary-waves'),
        '--binary',str(ROOT/'target/release/hf-bpe-native-j-block-summary-waves'),
        '--corpus',str(corpus),'--output',str(output),'--split',manifest['split'],
        '--vocab',str(manifest['target_vocab']),'--initialization-workers','4',
        '--merge-workers','4','--atomic-corpus','--expected-layout',
        'parallel_u32_dict16' if args.block_bits == 16 else 'parallel_u32_dict32']
    control=dict(command=command,extra_environment=env,mode=args.mode,block_bits=args.block_bits,
        wave=args.wave,local_offset_bytes=4,word_order="lexical (diagnostic only)",count=1,scope='same binary block initialization; fixed system allocator; small block span proxy, not large corpus measurement')
    output.with_suffix('.control.json').write_text(json.dumps(control,indent=2)+'\n')
    subprocess.run(command,cwd=ROOT,env=dict(os.environ,**env),check=True)
    probe=next(json.loads(line)['bench_block_count_probe'] for line in output.with_suffix('.stderr').read_text().splitlines()
        if line.startswith('{') and 'bench_block_count_probe' in line)
    row=json.loads(output.read_text());stats=row['indexed_stats']
    assert probe['mode']==args.mode and probe['block_bits']==args.block_bits
    assert probe['local_offset_bytes']==4
    assert probe['blocks']==stats['initial_blocks']>=1
    assert probe['physical_edges']==stats['initial_edges']
    assert probe['global_frequency_floor_after_reduction']
    expected_waves=1 if args.wave=='all' else (probe['blocks']+3)//4
    assert stats['initial_summary_waves']==expected_waves
    assert stats['peak_initial_summary_buffer_bytes']<=stats['initial_summary_buffer_bytes']
    expected=json.loads((ROOT/'results/posting-distribution'/(args.input_case+'.jsonl')).read_text())
    assert row['model_sha256']==expected['model_sha256']
    output.with_suffix('.summary.json').write_text(json.dumps(dict(control=control,probe=probe,
        model_gate='PASS',expected_model_sha256=expected['model_sha256']),indent=2)+'\n')
    print(f'{args.case}: u32 local offsets, dictionary path, model gate PASS')

if __name__=='__main__':
    main()
