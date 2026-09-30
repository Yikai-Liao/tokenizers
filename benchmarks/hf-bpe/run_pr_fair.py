#!/usr/bin/env python3
"""Run one monitored PR benchmark call for the pinned fair comparison."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from run import sha256
from run_parallel_key import system


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vocab', type=int, default=50000)
    parser.add_argument('--min-frequency', type=int, default=2)
    parser.add_argument('--split', default='none')
    args = parser.parse_args()
    bench = Path(__file__).resolve().parent
    corpus = args.corpus.resolve()
    preflight_path = None
    preflight = None
    for candidate in sorted(corpus.parent.glob('preflight-*.json')):
        value = json.loads(candidate.read_text())
        if value['bytes'] == corpus.stat().st_size:
            preflight_path, preflight = candidate, value
            break
    if preflight is None:
        raise SystemExit(f'no preflight matches corpus size {corpus.stat().st_size}')
    symbols = preflight['slots_upper'] - preflight['lines_upper'] - 1
    estimate = (8 * symbols + 8 * preflight['edges_upper'] + 160 * preflight['pairs_upper']
                + preflight['bytes'] + 96 * preflight['lines_upper'] + (256 << 20))
    before = system()
    if estimate + (1 << 30) > before['available']:
        raise SystemExit(f'PR estimate {estimate} plus 1 GiB exceeds available {before["available"]}')
    if args.output.exists():
        raise SystemExit('output exists')
    binary = bench / 'target/release/hf-bpe-profiled-pr4'
    root = bench / '.build/profiled-pr4'
    checkout = bench / '.build/pr-head'
    head = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    metadata = dict(pr_head=head, input_path=str(corpus), input_sha256=sha256(corpus),
                    binary_path=str(binary), binary_sha256=sha256(binary),
                    preflight_path=str(preflight_path), preflight_sha256=sha256(preflight_path),
                    preflight=preflight, estimated_rss_peak_bytes=estimate,
                    symbol_virtual_capacity_upper_bytes=8 * preflight['bytes'],
                    corpus_manifest=json.loads((corpus.parent / 'manifest.json').read_text()),
                    system_before=before, workers=4, feed_workers=1,
                    memory_policy='stop only if MemAvailable <= 1 GiB; record process VmSwap and global paging',
                    parallel_merge_threshold=1000, heap='OctonaryHeap',
                    algorithm='WordArena Symbol{u32 ID,u32 length}; historical word cohorts; one rule per round; conditional parallel word scans',
                    source_sha256={str(p.relative_to(bench)): sha256(p) for p in [
                        root / 'source/tokenizers/tk-train/src/trainers/bpe/mod.rs',
                        root / 'source/tokenizers/tk-train/src/trainers/bpe/word.rs',
                        root / 'source/tokenizers/tk-encode/src/utils/parallelism.rs',
                        root / 'runner/src/main.rs', root / 'runner/Cargo.toml', root / 'runner/Cargo.lock',
                        bench / 'build_pr_compare.py', bench / 'build_profiled.py', Path(__file__),
                        bench / 'run_parallel_key.py']})
    env = dict(os.environ, TOKENIZERS_PARALLELISM='false', RAYON_NUM_THREADS='4',
               TOKENIZERS_TRAIN_PARALLEL_MIN='1000')
    metadata['env_overrides'] = {k: env[k] for k in ['TOKENIZERS_PARALLELISM', 'RAYON_NUM_THREADS',
                                                       'TOKENIZERS_TRAIN_PARALLEL_MIN']}
    metadata['runner_override'] = 'set_num_threads(4), set_parallelism(false) before feed; set_parallelism(true) after feed'
    metadata['parameters'] = dict(split=args.split, vocab_size=args.vocab, min_frequency=args.min_frequency)
    env_path = args.output.with_suffix('.environment.json')
    env_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    command = [str(binary), str(corpus), args.split, 'reference', str(args.vocab), str(args.min_frequency)]
    stdout = args.output.with_suffix('.stdout')
    stderr = args.output.with_suffix('.stderr')
    peak_swap = peak_rss = 0
    min_available = before['available']
    fault = None
    print(f'planned one PR4 call; estimated peak {estimate/(1<<30):.2f} GiB; available {before["available"]/(1<<30):.2f} GiB', flush=True)
    with stdout.open('w') as out, stderr.open('w') as err:
        process = subprocess.Popen(command, env=env, stdout=out, stderr=err)
        print(f'PR4 pid {process.pid}', flush=True)
        while process.poll() is None:
            now = system()
            min_available = min(min_available, now['available'])
            try:
                status = Path(f'/proc/{process.pid}/status').read_text()
                usage = {k: int(v.split()[0]) * 1024 for k, v in
                         (s.split(':', 1) for s in status.splitlines() if s.startswith(('VmRSS:', 'VmSwap:')))}
                peak_swap = max(peak_swap, usage.get('VmSwap', 0))
                peak_rss = max(peak_rss, usage.get('VmRSS', 0))
            except FileNotFoundError:
                pass
            if now['available'] <= (1 << 30):
                fault = 'available memory at/below 1 GiB'
                process.terminate()
                break
            time.sleep(0.5)
        returncode = process.wait()
    finish = system()
    memory = dict(system_before=before, system_after=finish, minimum_available_bytes=min_available,
                  sampled_peak_rss_bytes=peak_rss, sampled_peak_process_swap_bytes=peak_swap,
                  pswpin_delta=finish['pswpin']-before['pswpin'], pswpout_delta=finish['pswpout']-before['pswpout'])
    if fault or returncode:
        args.output.write_text(json.dumps(dict(engine='pr_head4', failure=fault or str(returncode), memory=memory)) + '\n')
        raise SystemExit(f'PR4 failed: {fault or returncode}; failure retained in {args.output}')
    row = json.loads(stdout.read_text())
    row.update(engine='pr_head4', memory=memory, command=command, provenance_sha256=sha256(env_path),
               model_matches_comparison=None,
               stage_ms={v['bench_stage']: v['ms'] for s in stderr.read_text().splitlines()
                         if s.startswith('{') and 'bench_stage' in (v := json.loads(s))})
    args.output.write_text(json.dumps(row, ensure_ascii=False) + '\n')
    if not row['actual_merges']:
        raise SystemExit('PR model has zero merges')
    print(f'PR4 train {row["train_ms"]/1000:.3f}s, peak RSS {row["maxrss_kib"]/1048576:.2f} GiB, '
          f'min available {min_available/(1<<30):.2f} GiB; full model digest {row["model_sha256"]}', flush=True)


if __name__ == '__main__':
    main()
