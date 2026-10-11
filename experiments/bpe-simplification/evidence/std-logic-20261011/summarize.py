from pathlib import Path
import json, statistics, sys
root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'measurements'
runs = json.loads((root / 'core-runs.json').read_text())
keys = ('train_seconds', 'train_cpu_seconds', 'process_hwm_kib_before_validation')
summary = []
for case in dict.fromkeys(r['case'] for r in runs):
    rows = [r for r in runs if r['case'] == case and not r['warmup']]
    arms = list(dict.fromkeys(r['arm'] for r in rows))
    item = {'case': case, 'arms': {}, 'comparisons': {}}
    for arm in arms:
        values = [r for r in rows if r['arm'] == arm]
        item['arms'][arm] = {key: statistics.median(r['metrics'][key] for r in values) for key in keys}
    for reference in ['baseline', 'core', 'deque']:
        for candidate in arms:
            if reference == candidate:
                continue
            pairs = []
            for r in rows:
                if r['arm'] != candidate:
                    continue
                match = next((x for x in rows if x['arm'] == reference and x['block'] == r['block']), None)
                if match:
                    pairs.append({key: (r['metrics'][key] / match['metrics'][key] - 1) * 100 for key in keys})
            if pairs:
                item['comparisons'][candidate + '/' + reference] = {'formal_pairs': len(pairs), 'paired_median_delta_percent': {key: statistics.median(p[key] for p in pairs) for key in keys}, 'paired_delta_percent': pairs}
    summary.append(item)
(root / 'comparisons.json').write_text(json.dumps(summary, indent=2))
for item in summary:
    print(item['case'])
    for pair in ['core/baseline', 'deque/core', 'tree/core', 'bitmap/baseline', 'seed/baseline', 'ranges/baseline', 'final/baseline', 'arraydeque/deque']:
        if pair in item['comparisons']:
            c = item['comparisons'][pair]
            print(pair, c['formal_pairs'], {k: round(v, 2) for k, v in c['paired_median_delta_percent'].items()})
