#!/usr/bin/env python3
"""Validate saved queue experiments and compare complete operations."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results/validation-window'


def main():
    rows = []
    for path in sorted(RESULTS.glob('*.jsonl')):
        row = json.loads(path.read_text())
        env = json.loads(path.with_suffix('.environment.json').read_text())
        stats = row['indexed_stats']
        assert row['provenance_sha256'] == hashlib.sha256(path.with_suffix('.environment.json').read_bytes()).hexdigest()
        assert row['memory']['sampled_peak_process_swap_bytes'] == 0
        allocations = stats['posting_allocations']
        assert allocations['arena_requested_bytes'] == allocations['arena_retired_bytes']
        assert allocations['heap_requested_bytes'] == allocations['heap_freed_bytes']
        rows.append(dict(case=path.stem, workers=stats['workers'],
            actual_mode=stats.get('queue_selection_mode', 'unchanged-reference'),
            input_sha256=env['input_sha256'], binary_sha256=env['binary_sha256'],
            source_commit=env['worktree_commit'], model_sha256=row['model_sha256'],
            train_ms=row['train_ms'], initialize_ms=stats['initialize_ms'],
            merge_ms=stats['merge_ms'], select_ms=stats['select_ms'], commit_ms=stats['commit_ms'],
            worker_prefetch_ms=stats.get('queue_worker_prefetch_ms'),
            truth_checks=stats.get('queue_truth_checks'), owner_probes=stats.get('queue_owner_probes'),
            prefetched=stats.get('queue_prefetched'), unused_restored=stats.get('queue_unused_restored'),
            initial_edges=stats['initial_edges'], posting_visits=stats['posting_visits'],
            batch_rounds=stats['batch_rounds'],
            peak_rss_bytes=max(row['maxrss_kib'] * 1024, row['memory']['sampled_peak_rss_bytes'])))
    by_case = {r['case']: r for r in rows}
    comparisons = []
    for corpus in ['zh512m', 'en16m']:
        for workers in [1, 4]:
            for repeat in ['', '-r2']:
                baseline = f'{corpus}-reference-t{workers}{repeat}'
                candidate = f'{corpus}-unified-v3-t{workers}{repeat}'
                if baseline not in by_case or candidate not in by_case:
                    continue
                left, right = by_case[baseline], by_case[candidate]
                for field in ['input_sha256', 'model_sha256', 'initial_edges', 'posting_visits', 'batch_rounds']:
                    assert left[field] == right[field], (baseline, candidate, field)
                comparisons.append(dict(baseline=baseline, candidate=candidate,
                    train_change_percent=(right['train_ms'] / left['train_ms'] - 1) * 100,
                    merge_change_percent=(right['merge_ms'] / left['merge_ms'] - 1) * 100,
                    select_change_percent=(right['select_ms'] / left['select_ms'] - 1) * 100,
                    peak_change_bytes=right['peak_rss_bytes'] - left['peak_rss_bytes']))
    simulations = []
    for name in ['fixed-simulation-summary.json', 'weak-simulation-summary.json']:
        data = json.loads((RESULTS / name).read_text())
        assert data['exact_trace_gates'] == 'PASS'
        simulations.append(data)
    output = dict(native_calls_completed=len(rows), native_gates='PASS',
        simulation_calls_completed=sum(x['cases_completed'] for x in simulations),
        simulation_gates='PASS', native=rows, comparisons=comparisons,
        rejected_specialization_case='zh512m-auto-v2-t1',
        caution='worker prefetch is nested; simulations use four real workers; historical v2 single-worker dispatch is not retained')
    (RESULTS / 'analysis.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(dict(native_calls_completed=len(rows), simulation_calls_completed=output['simulation_calls_completed'],
        comparisons=comparisons), indent=2))


if __name__ == '__main__':
    main()
