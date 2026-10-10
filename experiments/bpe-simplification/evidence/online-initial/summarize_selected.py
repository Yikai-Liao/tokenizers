from pathlib import Path
import json
OUT = Path('/root/code/tokenizers-simplification-results/online-initial')
rows = [json.loads(l) for l in (OUT / 'runs.jsonl').read_text().splitlines()]
result = []
for case in (x['case'] for x in json.loads((OUT / 'input-pretokenizers.json').read_text())):
    for kind in ('selected-phases', 'phases-pipeline'):
        arms = {r['arm']:r for r in rows if r['case']==case and r['kind']==kind and r['valid']}
        if not {'main', 'baseline', 'candidate'} <= arms.keys():
            continue
        metrics = {}
        for arm, r in arms.items():
            metrics[arm] = dict(r['metrics'])
            if kind=='phases-pipeline':
                metrics[arm]['derived_feed_cpu_seconds'] = metrics[arm]['pipeline_cpu_seconds'] - metrics[arm]['train_cpu_seconds']
            for p in r['phases']:
                metrics[arm][p['name'] + '_wall_seconds'] = p['wall_seconds']
                metrics[arm][p['name'] + '_cpu_seconds'] = p['cpu_seconds']
        overhead = {arm:{k:100*(v/metrics['main'][k]-1) for k,v in m.items() if metrics['main'][k]} for arm,m in metrics.items() if arm != 'main'}
        result.append(dict(case=case, kind=kind, n_per_arm=1, metrics=metrics, relative_to_main_percent=overhead))
(OUT / 'selected-summary.json').write_text(json.dumps(result, indent=2))
for r in result:
    print(r['case'], r['kind'])
    for arm in ('main', 'baseline', 'candidate'):
        m=r['metrics'][arm]
        print(arm, {k:v for k,v in m.items() if k in ('train_seconds', 'train_cpu_seconds', 'feed_seconds', 'pipeline_seconds', 'process_hwm_kib_before_validation', 'prepare_wall_seconds', 'prepare_cpu_seconds', 'initial_index_wall_seconds', 'initial_index_cpu_seconds')})
