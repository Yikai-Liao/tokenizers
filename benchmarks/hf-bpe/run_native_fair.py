#!/usr/bin/env python3
"""Run one monitored call for a committed native fair-comparison worktree."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import time

from run import sha256
from run_parallel_key import system


def host_diagnostics():
    cpu_line = next(line for line in Path('/proc/stat').read_text().splitlines()
                    if line.startswith('cpu '))
    cpu = [int(value) for value in cpu_line.split()[1:]]
    load = [float(value) for value in Path('/proc/loadavg').read_text().split()[:3]]
    return dict(cpu_ticks=cpu, loadavg_1_5_15=load)


def children_usage():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return dict(user_seconds=usage.ru_utime, system_seconds=usage.ru_stime,
                minor_faults=usage.ru_minflt, major_faults=usage.ru_majflt,
                voluntary_context_switches=usage.ru_nvcsw,
                involuntary_context_switches=usage.ru_nivcsw,
                maxrss_kib=usage.ru_maxrss)


def diagnostic_delta(before, after):
    fields = ('user_seconds', 'system_seconds', 'minor_faults', 'major_faults',
              'voluntary_context_switches', 'involuntary_context_switches')
    cpu_ticks = [right-left for left, right in zip(before['host']['cpu_ticks'], after['host']['cpu_ticks'])]
    # guest/guest_nice are already included in user/nice, so count only
    # the first eight Linux counters in the total.
    total = sum(cpu_ticks[:8])
    idle = sum(cpu_ticks[3:5])
    steal = cpu_ticks[7]
    return dict(child_usage_before=before['children'], child_usage_after=after['children'],
                child_usage_delta={key: after['children'][key]-before['children'][key] for key in fields},
                child_wall_seconds=after['child_wall_seconds'],
                host_before=before['host'], host_after=after['host'],
                host_cpu_tick_delta=cpu_ticks,
                host_cpu_busy_fraction=((total-idle-steal)/total) if total else None,
                host_cpu_steal_fraction=(steal/total) if total else None,
                host_loadavg_delta=[right-left for left, right in
                                    zip(before['host']['loadavg_1_5_15'], after['host']['loadavg_1_5_15'])])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True, help='short output label, e.g. count1-parallel')
    parser.add_argument('--worktree', type=Path, required=True)
    parser.add_argument('--build-root', type=Path, required=True,
                        help='temporary instrumented build root, e.g. .build/native-count1')
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', default='reference')
    parser.add_argument('--split', default='none')
    parser.add_argument('--vocab', type=int, default=50000)
    parser.add_argument('--min-frequency', type=int, default=2)
    parser.add_argument('--initialization-workers', type=int, choices=[1, 4], required=True)
    parser.add_argument('--merge-workers', type=int, choices=[4], default=4,
                        help='the generated fair-comparison runner fixes training to four workers')
    parser.add_argument('--atomic-corpus', action='store_true')
    parser.add_argument('--require-stats', action='append', default=[],
                        help='require this numeric field in bench_indexed_stats; may be repeated')
    args = parser.parse_args()
    worktree = args.worktree.resolve()
    build_root = args.build_root.resolve()
    binary = args.binary.resolve()
    corpus = args.corpus.resolve()
    bench = Path(__file__).resolve().parent
    if args.output.exists():
        raise SystemExit(f'output exists: {args.output}')
    status = subprocess.check_output(['git', '-C', str(worktree), 'status', '--porcelain'], text=True).strip()
    if status:
        raise SystemExit(f'worktree is dirty: {worktree}')
    commit = subprocess.check_output(['git', '-C', str(worktree), 'rev-parse', 'HEAD'], text=True).strip()
    tracked = subprocess.check_output(['git', '-C', str(worktree), 'ls-files', '-z'], text=True).split('\0')
    locked = {}
    for relative in tracked:
        path = worktree / relative
        if path.is_file() and (path.suffix == '.rs' or path.name in ('Cargo.toml', 'Cargo.lock')):
            locked[relative] = sha256(path)
    manifest = corpus.parent / 'manifest.json'
    before = system()
    if before['available'] <= (1 << 30):
        raise SystemExit(f'MemAvailable {before["available"]} is at/below 1 GiB; not starting')
    env = dict(os.environ, TOKENIZERS_PARALLELISM='false', RAYON_NUM_THREADS=str(args.merge_workers))
    built_source = {}
    source_root = build_root / 'source/tokenizers'
    for path in source_root.rglob('*'):
        if path.is_file() and (path.suffix == '.rs' or path.name in ('Cargo.toml', 'Cargo.lock')):
            built_source[str(path.relative_to(build_root))] = sha256(path)
    for path in [build_root / 'runner/src/main.rs', build_root / 'runner/Cargo.toml',
                 build_root / 'runner/Cargo.lock']:
        if path.is_file():
            built_source[str(path.relative_to(build_root))] = sha256(path)
    metadata = dict(case=args.case, worktree=str(worktree), worktree_commit=commit,
                    worktree_clean=True, worktree_source_sha256=locked,
                    built_source_root=str(source_root), built_source_sha256=built_source,
                    build_script_sha256=sha256(bench / 'build_native_fair.py'),
                    runner_script_sha256=sha256(Path(__file__)), binary_path=str(binary),
                    binary_sha256=sha256(binary), input_path=str(corpus), input_sha256=sha256(corpus),
                    input_bytes=corpus.stat().st_size, corpus_manifest=json.loads(manifest.read_text()),
                    parameters=dict(split=args.split, backend=args.backend, vocab_size=args.vocab,
                                    min_frequency=args.min_frequency),
                    initialization_workers=args.initialization_workers, merge_workers=args.merge_workers,
                    expected_atomic_corpus=args.atomic_corpus,
                    required_indexed_stats=args.require_stats,
                    env_overrides={k: env[k] for k in ('TOKENIZERS_PARALLELISM', 'RAYON_NUM_THREADS')},
                    memory_policy='stop only if MemAvailable <= 1 GiB; record sampled process VmSwap and global paging',
                    system_before=before)
    env_path = args.output.with_suffix('.environment.json')
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    command = [str(binary), str(corpus), args.split, args.backend, str(args.vocab), str(args.min_frequency)]
    stdout, stderr = args.output.with_suffix('.stdout'), args.output.with_suffix('.stderr')
    min_available = before['available']
    peak_rss = peak_swap = 0
    fault = None
    diagnostic_before = dict(children=children_usage(), host=host_diagnostics())
    with stdout.open('w') as out, stderr.open('w') as err:
        child_begin = time.monotonic()
        process = subprocess.Popen(command, env=env, stdout=out, stderr=err)
        print(f'{args.case}: pid {process.pid}; MemAvailable {before["available"]/(1<<30):.2f} GiB', flush=True)
        while process.poll() is None:
            now = system()
            min_available = min(min_available, now['available'])
            try:
                status_text = Path(f'/proc/{process.pid}/status').read_text()
                usage = {k: int(v.split()[0]) * 1024 for k, v in
                         (s.split(':', 1) for s in status_text.splitlines()
                          if s.startswith(('VmRSS:', 'VmSwap:')))}
                peak_rss = max(peak_rss, usage.get('VmRSS', 0))
                peak_swap = max(peak_swap, usage.get('VmSwap', 0))
            except FileNotFoundError:
                pass
            if now['available'] <= (1 << 30):
                fault = 'available memory at/below 1 GiB'
                process.terminate()
                break
            time.sleep(0.5)
        returncode = process.wait()
        child_wall_seconds = time.monotonic() - child_begin
    diagnostic_after = dict(children=children_usage(), host=host_diagnostics(),
                             child_wall_seconds=child_wall_seconds)
    diagnostics = diagnostic_delta(diagnostic_before, diagnostic_after)
    finish = system()
    memory = dict(system_before=before, system_after=finish, minimum_available_bytes=min_available,
                  sampled_peak_rss_bytes=peak_rss, sampled_peak_process_swap_bytes=peak_swap,
                  pswpin_delta=finish['pswpin']-before['pswpin'], pswpout_delta=finish['pswpout']-before['pswpout'])
    if fault or returncode:
        args.output.write_text(json.dumps(dict(engine=args.case, failure=fault or str(returncode),
                                               memory=memory, diagnostics=diagnostics)) + '\n')
        raise SystemExit(f'{args.case} failed: {fault or returncode}; failure retained in {args.output}')
    row = json.loads(stdout.read_text())
    stats_rows = [v['bench_indexed_stats'] for s in stderr.read_text().splitlines()
                  if s.startswith('{') and 'bench_indexed_stats' in (v := json.loads(s))]
    if len(stats_rows) != 1:
        args.output.write_text(json.dumps(dict(engine=args.case, failure='indexed stats record count mismatch',
                                               observed_count=len(stats_rows), memory=memory,
                                               diagnostics=diagnostics)) + '\n')
        raise SystemExit(f'expected exactly one bench_indexed_stats record, found {len(stats_rows)}')
    stats = stats_rows[0]
    missing_stats = [key for key in args.require_stats
                     if key not in stats or not isinstance(stats[key], (int, float))]
    if missing_stats:
        args.output.write_text(json.dumps(dict(engine=args.case, failure='required indexed stats missing',
                                               missing=missing_stats, observed=stats, memory=memory,
                                               diagnostics=diagnostics),
                                          ensure_ascii=False) + '\n')
        raise SystemExit(f'{args.case}: required numeric indexed stats missing: {missing_stats}')
    expected = dict(workers=args.merge_workers, initialization_workers=args.initialization_workers,
                    atomic_corpus=args.atomic_corpus, layout='parallel_u32_flat32')
    mismatches = {key: (stats.get(key), value) for key, value in expected.items() if stats.get(key) != value}
    if mismatches:
        args.output.write_text(json.dumps(dict(engine=args.case, failure='indexed stats mismatch',
                                               observed=stats, expected=expected, memory=memory,
                                               diagnostics=diagnostics),
                                          ensure_ascii=False) + '\n')
        raise SystemExit(f'{args.case}: indexed stats mismatch: {mismatches}')
    if row.get('indexed_stats') is not None:
        args.output.write_text(json.dumps(dict(engine=args.case, failure='stdout indexed_stats was not null',
                                               observed=row.get('indexed_stats'), memory=memory,
                                               diagnostics=diagnostics),
                                          ensure_ascii=False) + '\n')
        raise SystemExit(f'{args.case}: expected reference runner stdout indexed_stats=null')
    row.update(engine=args.case, commit=commit, source_sha256=locked, command=command,
               indexed_stats=stats, initialize_ms=stats['initialize_ms'], merge_ms=stats['merge_ms'],
               memory=memory, diagnostics=diagnostics, provenance_sha256=sha256(env_path))
    args.output.write_text(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'{args.case}: init {stats["initialize_ms"]/1000:.3f}s + merge {stats["merge_ms"]/1000:.3f}s; '
          f'train {row["train_ms"]/1000:.3f}s, peak RSS {row["maxrss_kib"]/1048576:.2f} GiB, '
          f'min available {min_available/(1<<30):.2f} GiB, VmSwap {peak_swap/(1<<20):.1f} MiB, '
          f'model {row["model_sha256"]}', flush=True)


if __name__ == '__main__':
    main()
