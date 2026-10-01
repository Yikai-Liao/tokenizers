#!/usr/bin/env python3
"""Summarize completed fused training calls and separately scoped diagnostics."""
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results/fused-rewrite'


def counters(path):
    result = {}
    for columns in csv.reader(path.read_text().splitlines(), delimiter=';'):
        if len(columns) < 5 or columns[0].startswith('#'):
            continue
        name = columns[2]
        result[name] = dict(count=int(columns[0]), enabled_ns=int(columns[3]), running_percent=float(columns[4]))
    return result


def traces(path):
    batches = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.startswith('{'):
            continue
        record = json.loads(line)
        if 'bench_task' in record:
            t = record['bench_task']
            batches[t['phase'], t['batch']].append(t)
    phases = {}
    for (phase, _), tasks in batches.items():
        p = phases.setdefault(phase, dict(batches=0, tasks=0, visits=0,
            summed_task_ms=0.0, summed_last_end_ms=0.0, summed_max_visits=0))
        p['batches'] += 1
        p['tasks'] += len(tasks)
        p['visits'] += sum(t['visits'] for t in tasks)
        p['summed_task_ms'] += sum(t['end_ms'] - t['start_ms'] for t in tasks)
        p['summed_last_end_ms'] += max(t['end_ms'] for t in tasks)
        p['summed_max_visits'] += max(t['visits'] for t in tasks)
    for p in phases.values():
        p['task_span_occupancy_estimate'] = p['summed_task_ms'] / (4 * p['summed_last_end_ms'])
        p['visits_balance_efficiency'] = p['visits'] / (4 * p['summed_max_visits']) if p['summed_max_visits'] else None
    return phases


def main():
    calls = {}
    for path in sorted(RESULTS.glob('*.jsonl')):
        row = json.loads(path.read_text())
        stats = row['indexed_stats']
        summary = json.loads(path.with_suffix('.summary.json').read_text())
        provenance = row
        if 'binary_sha256' not in row:
            provenance = json.loads(path.with_suffix('.environment.json').read_text())
        call = dict(mode=summary['mode'], diagnostic_only=summary['diagnostic_only'],
            bits=summary['bits'], train_ms=row['train_ms'], initialize_ms=stats['initialize_ms'],
            merge_ms=stats['merge_ms'], peak_rss_bytes=summary['full_peak_rss_bytes'],
            source=provenance['worktree_commit'], binary=provenance['binary_sha256'],
            input=provenance['input_sha256'], model=row['model_sha256'],
            initial_edges=stats['initial_edges'], posting_visits=stats['posting_visits'],
            batches=stats['batch_rounds'], flat_group_visits=stats.get('flat_route_group_visits'),
            commit_ms=stats['commit_ms'], fused_rewrite_ms=stats.get('fused_rewrite_ms'),
            baseline_prepare_ms=stats['fused_prepare_ms'],
            baseline_delta_and_plan_and_rewrite_ms=stats['delta_ms'] + stats['plan_ms'] + stats['rewrite_ms'],
            partition_ms=stats.get('fused_partition_ms'),
            task_occupancy=summary['worker_task_occupancy'],
            visits_balance=summary['visits_balance_efficiency'], gates=summary['model_lifetime_swap_gates'])
        perf = path.with_suffix('.perf.csv')
        if perf.exists():
            call['perf'] = counters(perf)
            events = call['perf']
            def count(prefix):
                return next(e['count'] for name, e in events.items() if name.split(':')[0] == prefix)
            call['l1_miss_ratio'] = count('L1-dcache-load-misses') / count('L1-dcache-loads')
            call['generic_cache_miss_ratio'] = count('cache-misses') / count('cache-references')
            call['merge_worker_cpu_ms'] = stats['bench_merge_worker_cpu_ms']
            call['merge_worker_cpu_occupancy'] = sum(stats['bench_merge_worker_cpu_ms']) / (4 * stats['merge_ms'])
            call['task_traces'] = traces(path.with_suffix('.stderr'))
            call['perf_phase'] = stats.get('bench_perf_phase', 'merge')
            call['dense_commit_batches'] = stats.get('bench_dense_commit_batches')
        calls[path.stem] = call
    comparisons = {}
    for name, base in calls.items():
        if not name.endswith('-baseline') or base['diagnostic_only']:
            continue
        prefix = name[:-len('baseline')]
        for mode in ['atomic', 'plain', 'baseline-dense', 'dense']:
            candidate = calls.get(prefix + mode)
            if not candidate:
                continue
            keys = ['input', 'model', 'initial_edges', 'posting_visits', 'batches']
            if not base['mode'].startswith('native-'):
                keys.append('binary')
            for key in keys:
                assert candidate[key] == base[key], (name, key)
            comparisons[prefix + mode] = dict(
                train_ratio=candidate['train_ms'] / base['train_ms'],
                merge_ratio=candidate['merge_ms'] / base['merge_ms'],
                commit_ratio=candidate['commit_ms'] / base['commit_ms'],
                peak_rss_ratio=candidate['peak_rss_bytes'] / base['peak_rss_bytes'])
    result = dict(completed_calls=len(calls), diagnostic_calls=sum(c['diagnostic_only'] for c in calls.values()),
        calls=calls, comparisons=comparisons,
        notes=['Formal single calls are screens, not statistical confidence estimates.',
            'Baseline fused_prepare is nested inside delta; never add both.',
            'Trace occupancy uses last task timestamp, excludes phase tail, and includes instrumentation effects.',
            'Baseline task traces cover non-AA flat batches; fused traces include AA.',
            'Worker CPU occupancy includes extra work and does not by itself demonstrate fewer bubbles.',
            'Perf merge windows include trace logging; generic cache events are not all cache levels.'])
    (RESULTS / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(completed_calls=result['completed_calls'], diagnostic_calls=result['diagnostic_calls'], comparisons=comparisons), indent=2))


if __name__ == '__main__':
    main()
