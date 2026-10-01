#!/usr/bin/env python3
"""Run one original efficient_bpe ebpe call with provenance and memory monitoring."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from run import sha256
from run_parallel_key import system


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repository.resolve()
    directory = args.directory.resolve()
    manifest = json.loads((directory / 'manifest.json').read_text())
    binary = repo / 'rust/target/release/ebpe'
    fixture = directory / 'zh-prefix.prepared.json'
    if args.output.exists():
        raise SystemExit('output already exists')
    before = system()
    if before['available'] <= 1 << 30:
        raise SystemExit('MemAvailable at/below 1 GiB')
    command = [str(binary), '--input', str(fixture), '--workers', '4',
               '--rules', str(manifest['efficient_max_rules']), '--min-frequency', '2']
    tracked = subprocess.check_output(['git', '-C', str(repo), 'ls-files', '-z'], text=True).split('\0')
    metadata = dict(command=command, input_manifest=manifest,
                    repository_commit=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
                    repository_clean=not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip(),
                    source_sha256={p: sha256(repo / p) for p in tracked
                                   if (repo / p).is_file() and (p.endswith('.rs') or Path(p).name in ('Cargo.toml', 'Cargo.lock'))},
                    binary_sha256=sha256(binary), fixture_sha256=sha256(fixture),
                    runner_sha256=sha256(Path(__file__)),
                    profile='repository release: debug=1, lto=thin, codegen-units=1',
                    env_overrides={'RAYON_NUM_THREADS': '4'}, system_before=before)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.environment.json').write_text(json.dumps(metadata, indent=2) + '\n')
    min_available, peak_rss, peak_swap = before['available'], 0, 0
    fault = None
    with args.output.with_suffix('.stdout').open('w') as out, args.output.with_suffix('.stderr').open('w') as err:
        begin = time.monotonic()
        process = subprocess.Popen(command, stdout=out, stderr=err,
                                   env=dict(os.environ, RAYON_NUM_THREADS='4'))
        while process.poll() is None:
            min_available = min(min_available, system()['available'])
            try:
                status = Path(f'/proc/{process.pid}/status').read_text()
                values = {k: int(v.split()[0])*1024 for k, v in
                          (line.split(':', 1) for line in status.splitlines()
                           if line.startswith(('VmRSS:', 'VmSwap:')))}
                peak_rss = max(peak_rss, values.get('VmRSS', 0))
                peak_swap = max(peak_swap, values.get('VmSwap', 0))
            except FileNotFoundError:
                pass
            if min_available <= 1 << 30:
                fault = 'MemAvailable at/below 1 GiB'
                process.terminate()
                break
            time.sleep(0.05)
        code = process.wait()
        wall = time.monotonic() - begin
    row = dict(process_wall_seconds=wall, minimum_available_bytes=min_available,
               sampled_peak_rss_bytes=peak_rss, sampled_peak_swap_bytes=peak_swap,
               system_after=system(), failure=fault or (str(code) if code else None))
    if not row['failure']:
        row['result'] = json.loads(args.output.with_suffix('.stdout').read_text())
    args.output.write_text(json.dumps(row) + '\n')
    print(json.dumps(row))
    if row['failure']:
        raise SystemExit(row['failure'])


if __name__ == '__main__':
    main()
