#!/usr/bin/env python3
"""Run fixed-work sharded queue simulations with complete sequence hash gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent


def available():
    return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                    if x.startswith('MemAvailable:'))) * 1024


def measured(command, output):
    if output.exists():
        raise RuntimeError(f'case exists: {output}')
    if available() <= 1 << 30:
        raise RuntimeError('MemAvailable <= 1 GiB')
    min_available = available()
    peak_swap = 0
    with output.with_suffix('.stdout').open('w') as stdout, output.with_suffix('.stderr').open('w') as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        start = time.monotonic()
        while True:
            pid, status, usage = os.wait4(process.pid, os.WNOHANG)
            if pid:
                process.returncode = os.waitstatus_to_exitcode(status)
                break
            min_available = min(min_available, available())
            try:
                lines = Path(f'/proc/{process.pid}/status').read_text().splitlines()
                peak_swap = max(peak_swap, int(next(x.split()[1] for x in lines if x.startswith('VmSwap:'))) * 1024)
            except (FileNotFoundError, StopIteration):
                pass
            if min_available <= 1 << 30 or time.monotonic() - start > 300:
                process.kill()
                process.wait()
                raise RuntimeError('resource/timeout gate stopped case')
            time.sleep(0.05)
    assert process.returncode == 0, output.with_suffix('.stderr').read_text()
    row = json.loads(output.with_suffix('.stdout').read_text())
    row['memory'] = dict(maxrss_bytes=usage.ru_maxrss * 1024,
        sampled_peak_process_swap_bytes=peak_swap, min_available_bytes=min_available)
    row['process_usage'] = dict(user_seconds=usage.ru_utime, system_seconds=usage.ru_stime,
        voluntary_context_switches=usage.ru_nvcsw, involuntary_context_switches=usage.ru_nivcsw)
    row['command'] = command
    assert peak_swap == 0
    output.write_text(json.dumps(row, indent=2) + '\n')
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weak', action='store_true', help='256k keys per shard, stale75, three modes')
    args = p.parse_args()
    build = ROOT / '.build/validation-window-sim'
    binary = build / 'target/release/validation-window-sim'
    destination = ROOT / 'results/validation-window'
    destination.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((build / 'provenance.json').read_text())
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == provenance['binary_sha256']
    shutil.copyfile(build / 'provenance.json', destination / 'simulation-provenance.json')
    cases = []
    stale_values = [75] if args.weak else [0, 75]
    shard_values = [16, 32, 64] if args.weak else [4, 16, 32, 64]
    modes = ['serial', 'leader', 'bulk4'] if args.weak else ['serial', 'cached', 'leader', 'bulk4', 'bulk16']
    for stale in stale_values:
        for shards in shard_values:
            rows = []
            for mode in modes:
                total = shards * 262144 if args.weak else 1048576
                tag = f'{"weak" if args.weak else "fixed"}-stale{stale}-n{shards}-{mode}'
                command = [str(binary), str(shards), '4', mode, str(stale), str(total), '16000']
                row = measured(command, destination / (tag + '.json'))
                rows.append(row)
                cases.append(row)
                print(f'{tag}: pipeline {row["pipeline_ms"]:.1f} ms, select {row["select_ms"]:.1f} ms, '
                      f'truth {row["truth_checks"]}, probes {row["owner_probes"]}', flush=True)
            assert len({r['trace_sha256'] for r in rows}) == 1
            assert len({(r['selected'], r['batch_rounds']) for r in rows}) == 1
            assert all(r['selected'] == 16000 for r in rows)
    if not args.weak:
        for stale in stale_values:
            assert len({r['trace_sha256'] for r in cases if r['initial_stale_percent'] == stale}) == 1
    summary = dict(cases_completed=len(cases), exact_trace_gates='PASS',
        physical_workers=4, simulated_shards=shard_values,
        interpretation='queue/ledger simulation; no corpus rewrite; not a 64-core measurement',
        provenance=provenance, cases=cases)
    (destination / ('weak-simulation-summary.json' if args.weak else 'fixed-simulation-summary.json')).write_text(
        json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
