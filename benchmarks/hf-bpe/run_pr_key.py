#!/usr/bin/env python3
"""One PR call against the completed same-input parallel critical comparison."""
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
    parser.add_argument('comparison', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    bench = Path(__file__).resolve().parent
    baseline = json.loads(args.comparison.read_text().splitlines()[0])
    corpus = Path(baseline['input'])
    preflight = json.loads((corpus.parent/'preflight-512.json').read_text())
    # RSS bound: only initialized Symbol pages are touched. PR reserves its
    # Symbol Vec by UTF-8 bytes, so its virtual capacity is substantially larger.
    symbols = preflight['slots_upper']-preflight['lines_upper']-1
    estimate = (8*symbols + 8*preflight['edges_upper'] + 160*preflight['pairs_upper']
                + preflight['bytes'] + 96*preflight['lines_upper'] + (256 << 20))
    before = system()
    if estimate + (1 << 30) > before['available']:
        raise SystemExit(f'PR estimate {estimate} plus 1 GiB exceeds available {before["available"]}')
    if args.output.exists():
        raise SystemExit('output exists')
    binary = bench/'target/release/hf-bpe-profiled-pr4'
    root = bench/'.build/profiled-pr4'
    head = subprocess.check_output(['git','-C',str(bench/'.build/pr-head'),'rev-parse','HEAD'],text=True).strip()
    metadata = dict(pr_head=head, baseline_path=str(args.comparison.resolve()),
                    baseline_sha256=sha256(args.comparison), input_path=str(corpus), input_sha256=sha256(corpus),
                    binary_path=str(binary), binary_sha256=sha256(binary),
                    preflight=preflight, estimated_rss_peak_bytes=estimate,
                    symbol_virtual_capacity_upper_bytes=8*preflight['bytes'],
                    system_before=before, workers=4, feed_workers=1,
                    parallel_merge_threshold=1000, heap='OctonaryHeap',
                    memory_policy='stop only if MemAvailable <= 1 GiB; record sampled process VmSwap and global paging separately',
                    algorithm='WordArena Symbol{u32 ID,u32 length}; historical word cohorts; one rule per round; conditional parallel word scans',
                    source_sha256={str(p.relative_to(bench)):sha256(p) for p in [
                        root/'source/tokenizers/tk-train/src/trainers/bpe/mod.rs',
                        root/'source/tokenizers/tk-train/src/trainers/bpe/word.rs',
                        root/'source/tokenizers/tk-encode/src/utils/parallelism.rs',
                        root/'runner/src/main.rs',root/'runner/Cargo.toml',root/'runner/Cargo.lock',
                        bench/'build_pr_compare.py',bench/'build_profiled.py',Path(__file__),bench/'run_parallel_key.py']})
    env = dict(os.environ,TOKENIZERS_PARALLELISM='false',RAYON_NUM_THREADS='4',TOKENIZERS_TRAIN_PARALLEL_MIN='1000')
    metadata['env_overrides'] = {k:env[k] for k in ['TOKENIZERS_PARALLELISM','RAYON_NUM_THREADS','TOKENIZERS_TRAIN_PARALLEL_MIN']}
    metadata['runner_override'] = 'set_num_threads(4), set_parallelism(false) before feed; set_parallelism(true) after feed'
    path = args.output.with_suffix('.environment.json')
    path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    command = [str(binary),str(corpus),baseline['split'],'reference',str(baseline['vocab_size']),str(baseline['min_frequency'])]
    stdout = args.output.with_suffix('.stdout')
    stderr = args.output.with_suffix('.stderr')
    peak_swap = peak_rss = 0
    min_available = before['available']
    fault = None
    print(f'planned one PR4 call; estimated RSS {estimate/(1<<30):.2f} GiB; available {before["available"]/(1<<30):.2f} GiB',flush=True)
    with stdout.open('w') as out, stderr.open('w') as err:
        process = subprocess.Popen(command,env=env,stdout=out,stderr=err)
        print(f'PR4 pid {process.pid}',flush=True)
        while process.poll() is None:
            now = system()
            min_available = min(min_available,now['available'])
            try:
                status = Path(f'/proc/{process.pid}/status').read_text()
                usage = {k:int(v.split()[0])*1024 for k,v in
                         (s.split(':',1) for s in status.splitlines() if s.startswith(('VmRSS:','VmSwap:')))}
                peak_swap = max(peak_swap,usage.get('VmSwap',0))
                peak_rss = max(peak_rss,usage.get('VmRSS',0))
            except FileNotFoundError:
                pass
            if now['available'] <= (1 << 30):
                fault = 'available memory at/below 1 GiB'
                process.terminate()
                break
            time.sleep(0.5)
        returncode = process.wait()
    finish = system()
    memory = dict(system_before=before,system_after=finish,minimum_available_bytes=min_available,
                  sampled_peak_rss_bytes=peak_rss,sampled_peak_process_swap_bytes=peak_swap,
                  pswpin_delta=finish['pswpin']-before['pswpin'],pswpout_delta=finish['pswpout']-before['pswpout'])
    if fault or returncode:
        args.output.write_text(json.dumps(dict(engine='pr_head4',failure=fault or str(returncode),memory=memory))+'\n')
        raise SystemExit(f'PR4 failed: {fault or returncode}')
    row = json.loads(stdout.read_text())
    signature = lambda r: (r['model_sha256'],r['actual_vocab'],r['actual_merges'],r['unique_words'])
    row.update(engine='pr_head4',memory=memory,command=command,provenance_sha256=sha256(path),
               model_matches_comparison=signature(row)==signature(baseline),
               stage_ms={v['bench_stage']:v['ms'] for s in stderr.read_text().splitlines()
                         if s.startswith('{') and 'bench_stage' in (v:=json.loads(s))})
    args.output.write_text(json.dumps(row,ensure_ascii=False)+'\n')
    if not row['model_matches_comparison']:
        raise SystemExit('PR model digest differs from the parallel comparison')
    print(f'PR4 train {row["train_ms"]/1000:.3f}s, peak RSS {row["maxrss_kib"]/1048576:.2f} GiB, '
          f'min available {min_available/(1<<30):.2f} GiB; complete model matches',flush=True)


if __name__ == '__main__':
    main()
