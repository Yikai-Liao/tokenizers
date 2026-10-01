#!/usr/bin/env python3
"""One matched full training call or merge-window perf diagnostic."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
WORKTREE = Path('/root/code/tokenizers-worktrees/fused-aa-rewrite')

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(1 << 20): h.update(block)
    return h.hexdigest()

def validate(row, args):
    stats = row['indexed_stats']
    expected_mode = 'baseline' if args.mode.startswith('baseline') else 'fused'
    assert stats['bench_fused_mode'] == expected_mode
    assert stats['atomic_corpus'] == (args.mode != 'plain')
    assert stats['workers'] == stats['initialization_workers'] == 4
    assert stats['layout'] == ('parallel_u32_flat32' if args.bits == 32 else 'parallel_u32_dict16')
    assert row['model_sha256'] == args.expected_model
    a = stats['posting_allocations']
    assert a['arena_requested_bytes'] == a['arena_retired_bytes']
    assert a['heap_requested_bytes'] == a['heap_freed_bytes']
    assert a['heap_buffers'] == a['heap_frees']
    assert row['memory']['sampled_peak_process_swap_bytes'] == 0
    if expected_mode == 'fused':
        assert stats['peak_valid_start_bytes'] == 0
        assert stats['fused_task_visits'] == stats['posting_visits']
    if args.perf_phase != 'merge':
        assert args.perf and stats.get('bench_perf_phase') == args.perf_phase
    if 'bench_dense_commit_batches' in stats:
        dense = args.bits == 32 and args.mode in ['baseline-dense', 'atomic', 'plain']
        assert (stats['bench_dense_commit_batches'] > 0) == dense
    if args.mode == 'atomic-hash':
        assert 'bench_dense_commit_batches' in stats, 'atomic-hash needs --phase-controls build'
    return stats

def perf_call(args, binary, build, output, env):
    from run_native_fair import system
    import resource
    control = output.with_suffix('.perf.control.fifo')
    ack = output.with_suffix('.perf.ack.fifo')
    for p in [control, ack]: os.mkfifo(p)
    env = dict(env, HF_BPE_PERF_CONTROL=str(control), HF_BPE_PERF_ACK=str(ack))
    events = 'instructions:u,L1-dcache-loads:u,L1-dcache-load-misses:u,cache-references:u,cache-misses:u'
    perf_path = output.with_suffix('.perf.csv')
    command = ['perf','stat','-x',';','-o',str(perf_path),'-e',events,'--delay=-1',
               '--control',f'fifo:{control},{ack}','--',str(binary),str(args.corpus),args.split,'reference',str(args.vocab),'2']
    before = system()
    if before['available'] <= (1 << 30): raise SystemExit('MemAvailable below 1 GiB')
    start = time.monotonic()
    peak = swap = 0
    minimum = before['available']
    stdout = output.with_suffix('.stdout')
    stderr = output.with_suffix('.stderr')
    fault = None
    print(f'{args.case}: {args.perf_phase} perf diagnostic; {args.mode}', flush=True)
    try:
        with stdout.open('w') as out, stderr.open('w') as err:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err)
            while process.poll() is None:
                state = system(); minimum = min(minimum, state['available'])
                # perf is the parent; inspect its actual training child as well.
                pids = [process.pid]
                try: pids += [int(p) for p in Path(f'/proc/{process.pid}/task/{process.pid}/children').read_text().split()]
                except FileNotFoundError: pass
                rss_sum = 0
                for pid in pids:
                    try:
                        status = Path(f'/proc/{pid}/status').read_text()
                        values = {k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in status.splitlines() if line.startswith(('VmRSS:','VmSwap:')))}
                        rss_sum += values.get('VmRSS',0); swap = max(swap, values.get('VmSwap',0))
                    except FileNotFoundError: pass
                peak = max(peak, rss_sum)
                if minimum <= (1 << 30) or time.monotonic()-start > 300:
                    fault = 'memory limit or diagnostic timeout'; process.terminate(); break
                time.sleep(.5)
            code = process.wait()
        if fault or code: raise RuntimeError(f'perf diagnostic failed: {fault or code}')
    finally:
        for p in [control,ack]: p.unlink(missing_ok=True)
    row = json.loads(stdout.read_text())
    records = [json.loads(line) for line in stderr.read_text().splitlines() if line.startswith('{')]
    stats = [r['bench_indexed_stats'] for r in records if 'bench_indexed_stats' in r]
    assert len(stats) == 1
    row.update(indexed_stats=stats[0], initialize_ms=stats[0]['initialize_ms'], merge_ms=stats[0]['merge_ms'],
               maxrss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
               memory=dict(sampled_peak_rss_bytes=peak, sampled_peak_process_swap_bytes=swap, minimum_available_bytes=minimum),
               engine=args.case, command=command, diagnostic_only=True,
               worktree_commit=subprocess.check_output(['git','-C',str(args.worktree),'rev-parse','HEAD'],text=True).strip(),
               binary_sha256=sha256(binary), input_sha256=sha256(args.corpus),
               instrumentation_sha256=sha256(build/'instrumentation.patch'),
               perf_scope=args.perf_phase + '; enable/disable acknowledged by perf control FIFO')
    output.write_text(json.dumps(row,ensure_ascii=False)+'\n')
    return row

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['baseline','baseline-dense','atomic','atomic-hash','plain'],required=True)
    p.add_argument('--case',required=True)
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--vocab',type=int,required=True)
    p.add_argument('--split',choices=['none','whitespace_split'],default='none')
    p.add_argument('--bits',type=int,choices=[16,32],default=32)
    p.add_argument('--expected-model',required=True)
    p.add_argument('--label',default='fused-rewrite-v1')
    p.add_argument('--worktree', type=Path, default=WORKTREE)
    p.add_argument('--diagnostic',action='store_true')
    p.add_argument('--perf',action='store_true')
    p.add_argument('--perf-phase', choices=['merge','kernel','commit'], default='merge')
    p.add_argument('--trace', action='store_true')
    args = p.parse_args()
    args.corpus = args.corpus.resolve()
    args.worktree = args.worktree.resolve()
    if args.perf: args.diagnostic = True
    if subprocess.check_output(['git','-C',str(args.worktree),'status','--porcelain'],text=True).strip():
        raise SystemExit('worktree must be clean')
    build = ROOT/'.build'/f'native-{args.label}'
    binary = build/f'target/release/hf-bpe-native-{args.label}'
    output = ROOT/'results/fused-rewrite'/(args.case+'.jsonl')
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists(): raise SystemExit(f'output exists: {output}')
    extra = dict(HF_BPE_FUSED_MODE='baseline' if args.mode.startswith('baseline') else 'fused',
                 HF_BPE_DENSE_COMMIT='1' if args.mode in ['baseline-dense','atomic','plain'] else '0',
                 HF_BPE_PERF_PHASE=args.perf_phase, HF_BPE_FUSED_TRACE='1' if args.trace else '0',
                 HF_BPE_FUSED_ATOMIC='0' if args.mode == 'plain' else '1',
                 HF_BPE_FUSED_BLOCK_BITS=str(args.bits), HF_BPE_ARENA_MODE='auto',
                 HF_BPE_FUSED_DIAGNOSTICS='1' if args.diagnostic else '0')
    command = [sys.executable,str(ROOT/'run_native_fair.py'),'--case',args.case,'--worktree',str(args.worktree),
               '--build-root',str(build),'--binary',str(binary),'--corpus',str(args.corpus),'--output',str(output),
               '--split',args.split,'--vocab',str(args.vocab),'--initialization-workers','4','--merge-workers','4',
               '--expected-layout','parallel_u32_flat32' if args.bits == 32 else 'parallel_u32_dict16',
               '--require-stats','fused_partition_ms']
    if args.mode != 'plain': command.append('--atomic-corpus')
    control = dict(command=command,extra_environment=extra,expected_model=args.expected_model,
                   word_order='serial feed; hash seeds (11,13,17,19)', count=1, diagnostic_only=args.diagnostic)
    output.with_suffix('.control.json').write_text(json.dumps(control,indent=2)+'\n')
    env = dict(os.environ,**extra,TOKENIZERS_PARALLELISM='false',RAYON_NUM_THREADS='4',HF_BPE_BENCH_WORKERS='4')
    if args.perf: row = perf_call(args,binary,build,output,env)
    else:
        subprocess.run(command,cwd=ROOT,env=env,check=True)
        row = json.loads(output.read_text())
    stats = validate(row,args)
    summary = dict(train_ms=row['train_ms'],initialize_ms=stats['initialize_ms'],merge_ms=stats['merge_ms'],
                   full_peak_rss_bytes=max(row['maxrss_kib']*1024,row['memory']['sampled_peak_rss_bytes']),
                   model_lifetime_swap_gates='PASS',mode=args.mode,bits=args.bits,diagnostic_only=args.diagnostic,
                   partition_ms=stats['fused_partition_ms'],worker_task_occupancy=(stats['fused_worker_ms']/(4*stats['fused_parallel_ms']) if stats['fused_parallel_ms'] else None),
                   visits_balance_efficiency=(stats['fused_task_visits']/(4*stats['fused_task_max_visits_sum']) if stats['fused_task_max_visits_sum'] else None))
    output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(f'{args.case}: model/lifetime/swap gates PASS; {args.mode}; {args.bits}-bit blocks',flush=True)

if __name__ == '__main__': main()
