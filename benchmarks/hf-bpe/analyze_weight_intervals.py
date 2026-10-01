#!/usr/bin/env python3
"""Validate weight-order experiments and report costs including sorting."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results/weight-intervals'


def main():
    rows = []
    for path in sorted(RESULTS.glob('*.jsonl')):
        row = json.loads(path.read_text())
        provenance = path.with_suffix('.environment.json')
        env = json.loads(provenance.read_text())
        control = json.loads(path.with_suffix('.control.json').read_text())
        s = row['indexed_stats']
        assert row['provenance_sha256'] == hashlib.sha256(provenance.read_bytes()).hexdigest()
        assert env['worktree_clean']
        assert row['memory']['sampled_peak_process_swap_bytes'] == 0
        assert row['model_sha256'] == control['expected_model']
        assert s['workers'] == s['initialization_workers'] == control['workers']
        assert s['queue_selection_mode'] == 'bulk4'
        if not control.get('unchanged_reference'):
            assert s['corpus_weight_order'] == ('original' if control['weight_order'] == 'original' else 'weight_sorted')
        a = s['posting_allocations']
        assert a['arena_requested_bytes'] == a['arena_retired_bytes']
        assert a['heap_requested_bytes'] == a['heap_freed_bytes']
        assert a['heap_buffers'] == a['heap_frees']
        if control['weight_order'] != 'original':
            assert s['weight_lookup_bytes'] == s['weight_one_bucket_bytes'] == 0
        rows.append(dict(case=path.stem, workers=s['workers'], mode='unchanged' if control.get('unchanged_reference') else control['weight_order'],
            input_sha256=env['input_sha256'], binary_sha256=env['binary_sha256'],
            source_commit=env['worktree_commit'], diagnostic_only=control.get('diagnostic_only',False),
            smoke_case=path.stem.endswith('-smoke'), model_sha256=row['model_sha256'],
            train_ms=row['train_ms'], feed_ms=row['feed_ms'], elapsed_ms=row['elapsed_ms'], initialize_ms=s['initialize_ms'], merge_ms=s['merge_ms'],
            sort_ms=s.get('corpus_sort_ms', 0), stable_sort=s.get('corpus_stable_weight_sort', False),
            sort_buffer_bytes=s.get('corpus_sort_buffer_bytes', 0), measure_ms=s['corpus_measure_ms'], fill_ms=s['corpus_fill_ms'],
            initial_group_count_ms=s['initial_group_count_ms'], fused_prepare_ms=s['fused_prepare_ms'],
            commit_ms=s['commit_ms'], select_ms=s['select_ms'], tokenize_ms=s['tokenize_ms'],
            weight_intervals=s.get('weight_interval_count'), weight_bytes=s['initial_weight_bytes'],
            lookup_bytes=s['weight_lookup_bytes'], initial_lookup_bytes=s['initial_weight_lookup_bytes'],
            bitmap_bytes=s['weight_one_bucket_bytes'],
            temporary_weight_bytes=s.get('corpus_temporary_weight_bytes'), word_reference_bytes=s.get('corpus_word_reference_bytes'),
            initial_slots=s['initial_slots'], initial_edges=s['initial_edges'], initial_symbols=s['initial_symbols'],
            posting_visits=s['posting_visits'], batch_rounds=s['batch_rounds'], merges=row['actual_merges'],
            peak_rss_bytes=max(row['maxrss_kib'] * 1024, row['memory']['sampled_peak_rss_bytes'])))
    by_case = {r['case']: r for r in rows}
    comparisons = []
    for corpus in ['zh512m', 'en16m', 'en1m']:
        for workers in [1, 4]:
            for suffix in ['', '-r2', '-smoke', '-stable', '-inline', '-inline-stable']:
                baseline_suffix = '-inline' if suffix == '-inline-stable' else suffix
                baseline = f'{corpus}-original-t{workers}{baseline_suffix}'
                candidate = f'{corpus}-sorted-t{workers}{suffix}'
                if baseline not in by_case or candidate not in by_case:
                    continue
                left, right = by_case[baseline], by_case[candidate]
                for field in ['input_sha256', 'model_sha256', 'binary_sha256', 'source_commit',
                              'initial_slots', 'initial_edges', 'initial_symbols', 'posting_visits', 'batch_rounds', 'merges']:
                    assert left[field] == right[field], (baseline, candidate, field)
                comparisons.append(dict(baseline=baseline, candidate=candidate,
                    sorting_algorithm='stable' if right['stable_sort'] else 'unstable',
                    train_change_percent=(right['train_ms'] / left['train_ms'] - 1) * 100,
                    initialize_change_percent=(right['initialize_ms'] / left['initialize_ms'] - 1) * 100,
                    merge_change_percent=(right['merge_ms'] / left['merge_ms'] - 1) * 100,
                    peak_change_bytes=right['peak_rss_bytes'] - left['peak_rss_bytes'],
                    weight_and_lookup_saved_bytes=left['weight_bytes']+left['lookup_bytes']-right['weight_bytes']-right['lookup_bytes']))
    unchanged_comparisons = []
    for corpus in ['zh512m', 'en16m']:
        for workers in [1, 4]:
            preferred = [f'{corpus}-unchanged-t{workers}-inline',
                         f'{corpus}-unchanged-t{workers}-clean',
                         f'{corpus}-unchanged-t{workers}']
            baseline = next((case for case in preferred if case in by_case and not by_case[case]['diagnostic_only']), None)
            if baseline is None:
                continue
            left = by_case[baseline]
            for suffix in ['', '-r2', '-stable', '-inline', '-inline-stable']:
                candidate = f'{corpus}-sorted-t{workers}{suffix}'
                if candidate not in by_case:
                    continue
                right = by_case[candidate]
                for field in ['input_sha256', 'model_sha256', 'initial_slots', 'initial_edges',
                              'initial_symbols', 'posting_visits', 'batch_rounds', 'merges']:
                    assert left[field] == right[field], (baseline, candidate, field)
                adjacent_pair = suffix == '-inline' and baseline.endswith('-inline')
                unchanged_comparisons.append(dict(baseline=baseline, candidate=candidate,
                    train_change_percent=(right['train_ms'] / left['train_ms'] - 1) * 100,
                    merge_change_percent=(right['merge_ms'] / left['merge_ms'] - 1) * 100,
                    elapsed_change_percent=(right['elapsed_ms'] / left['elapsed_ms'] - 1) * 100,
                    adjacent_pair=adjacent_pair,
                    caution=None if adjacent_pair else 'one unchanged reference per corpus; historical candidates were not adjacent pairs'))
    output = dict(calls_completed=len(rows), diagnostic_calls=sum(r['diagnostic_only'] for r in rows),
        smoke_calls=sum(r['smoke_case'] for r in rows),
        formal_calls=sum(not r['diagnostic_only'] and not r['smoke_case'] for r in rows),
        correctness_lifetime_swap_provenance_gates='PASS',
        rows=rows, comparisons=comparisons, unchanged_comparisons=unchanged_comparisons,
        caution='sort_ms is nested inside measure_ms/initialize_ms/train_ms; metadata capacity is not peak RSS; owner table hash layouts are random')
    (RESULTS / 'analysis.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(dict(calls_completed=len(rows), comparisons=comparisons, unchanged_comparisons=unchanged_comparisons), indent=2))


if __name__ == '__main__':
    main()
