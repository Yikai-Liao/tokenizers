#!/usr/bin/env python3
"""Three serial critical comparisons, with memory and paging observations."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from run import capture, sha256


def system():
    mem = {k: int(v.split()[0]) * 1024 for k, v in
           (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
    vm = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
    return dict(available=mem['MemAvailable'], swap_used=mem['SwapTotal']-mem['SwapFree'],
                pswpin=int(vm['pswpin']), pswpout=int(vm['pswpout']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=Path(__file__).parent/'target/release/hf-bpe-indexed-bench')
    parser.add_argument('--vocab', type=int, default=50000)
    args = parser.parse_args()
    root = args.corpus_dir.resolve()
    binary = args.binary.resolve()
    engines = ['narrow32par1', 'narrow32par4', 'atomic32par4']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit('output already exists')
    preflight = json.loads((root/'preflight-512.json').read_text())
    # Corpus and both initial route/posting arrays at at most twice length,
    # plus upper bounds for pair/word tables, lengths, weights and allocator.
    estimate = (2*preflight['slots_upper'] + 16*preflight['edges_upper']
                + 128*preflight['pairs_upper'] + preflight['bytes']
                + 96*preflight['lines_upper'] + (256 << 20))
    before = system()
    if estimate + (1 << 30) > before['available']:
        raise SystemExit(f'preflight estimate {estimate} + reserve exceeds available {before["available"]}')
    metadata = capture(root, {e:binary for e in engines}, [('zh',512,'none')])
    metadata.update(training_workers={e:1 if e.endswith('par1') else 4 for e in engines},
                    preflight=preflight, estimated_peak_bytes=estimate,
                    system_before=before, corpus_manifest=json.loads((root/'manifest.json').read_text()),
                    runner_sha256=sha256(Path(__file__)), runs=3, repeats=1,
                    interpretation='key comparison only; no full benchmark matrix',
                    memory_policy='stop only if MemAvailable <= 1 GiB; record sampled process VmSwap and global paging separately')
    path = args.output.with_suffix('.environment.json')
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    provenance = sha256(path)
    signatures = set()
    env = dict(os.environ, TOKENIZERS_PARALLELISM='false', RAYON_NUM_THREADS='1')
    print(f'planned 3 serial calls; estimated peak {estimate/(1<<30):.2f} GiB; available {before["available"]/(1<<30):.2f} GiB', flush=True)
    with args.output.open('w') as output:
        for engine in engines:
            command = [str(binary),str(root/'zh-512m.txt'),'none',engine,str(args.vocab),'2']
            start = system()
            min_available = start['available']
            peak_swap = peak_rss = 0
            fault = None
            stderr = args.output.with_suffix(f'.{engine}.stderr')
            stdout = args.output.with_suffix(f'.{engine}.stdout')
            with stdout.open('w') as out, stderr.open('w') as err:
                process = subprocess.Popen(command, env=env, stdout=out, stderr=err)
                print(f'{engine}: pid {process.pid}', flush=True)
                while process.poll() is None:
                    now = system()
                    min_available = min(min_available,now['available'])
                    try:
                        status = Path(f'/proc/{process.pid}/status').read_text()
                        usage = {k:int(v.split()[0])*1024 for k,v in
                                 (line.split(':',1) for line in status.splitlines() if line.startswith(('VmRSS:','VmSwap:')))}
                        peak_rss = max(peak_rss,usage.get('VmRSS',0))
                        peak_swap = max(peak_swap,usage.get('VmSwap',0))
                    except FileNotFoundError:
                        pass
                    if now['available'] <= (1 << 30):
                        fault = 'available memory at/below 1 GiB'
                        process.terminate()
                        break
                    time.sleep(0.5)
                returncode = process.wait()
            finish = system()
            memory = dict(system_before=start, system_after=finish, minimum_available_bytes=min_available,
                          sampled_peak_rss_bytes=peak_rss, sampled_peak_process_swap_bytes=peak_swap,
                          pswpin_delta=finish['pswpin']-start['pswpin'], pswpout_delta=finish['pswpout']-start['pswpout'])
            if fault or returncode:
                output.write(json.dumps(dict(engine=engine, failure=fault or f'exit {returncode}',memory=memory))+'\n')
                output.flush()
                raise SystemExit(f'{engine}: {fault or returncode}; record saved, reduce workload before retry')
            row = json.loads(stdout.read_text())
            row.update(engine=engine, memory=memory, provenance_sha256=provenance, command=command,
                       language='zh', size_mib=512, repeat=0)
            output.write(json.dumps(row,ensure_ascii=False)+'\n')
            output.flush()
            signatures.add((row['model_sha256'],row['actual_vocab'],row['actual_merges'],row['unique_words']))
            if len(signatures) != 1:
                raise SystemExit('complete model mismatch')
            if not row['actual_merges']:
                raise SystemExit('alphabet consumed the merge budget; no meaningful merge comparison')
            print(f'{engine}: train {row["train_ms"]/1000:.3f}s, RSS {row["maxrss_kib"]/1048576:.2f} GiB, '
                  f'min available {min_available/(1<<30):.2f} GiB, paging {memory["pswpin_delta"]}/{memory["pswpout_delta"]} pages',flush=True)
    print('3 complete model digests match',flush=True)


if __name__ == '__main__':
    main()
