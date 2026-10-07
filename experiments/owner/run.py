#!/usr/bin/env python3
"""Run one shared, exact-model comparison with tokenizers-bpe-benchmarks."""
import argparse
import json
import os
from pathlib import Path

from bench.config import write
from bench.runs import run

p = argparse.ArgumentParser()
p.add_argument('--baseline', required=True)
for name in ['stable', 'epoch', 'bucket', 'blocks']:
    p.add_argument('--' + name)
p.add_argument('--arms', default='stable,epoch,bucket,blocks')
p.add_argument('--zh', required=True, help='prepared whitespace word manifest')
p.add_argument('--en', required=True, help='prepared whitespace word manifest')
p.add_argument('--workers', type=int, choices=[1, 4], default=4)
p.add_argument('--paired-blocks', type=int, default=3)
p.add_argument('--warmups', type=int, default=1)
p.add_argument('--out', required=True)
a = p.parse_args()
selected = a.arms.split(',')
if len(set(selected)) != len(selected) or set(selected) - {'stable', 'epoch', 'bucket', 'blocks'}:
    p.error('--arms must list distinct known candidates')
arms = {'baseline': {'build': str(Path(a.baseline).resolve()), 'environment': {}}}
for name in selected:
    record = getattr(a, name)
    if not record:
        p.error('--' + name + ' is required for the selected arm')
    arms[name] = {'build': str(Path(record).resolve()), 'environment': {}}
cpus = sorted(os.sched_getaffinity(0))[:a.workers]
if len(cpus) != a.workers:
    raise SystemExit('Not enough CPUs in supervisor affinity')
cfg = dict(
    schema_version=1, name='owner-zh-en256-whitespace-v50000', mode='core',
    arms=arms,
    cases=[dict(name=lang + '256-whitespace-v50000', input_manifest=str(Path(manifest).resolve()),
                pretokenizer='whitespace', trainer=dict(vocab_size=50000, min_frequency=2,
                prefix=None, suffix=None, max_token_length=None))
           for lang, manifest in [('zh', a.zh), ('en', a.en)]],
    execution=dict(workers=[a.workers], cpu_set=cpus, warmups_per_cell=a.warmups,
                   paired_blocks=a.paired_blocks, timeout_seconds=900, min_available_gib=2,
                   max_process_rss_gib=10, order='balanced-alternating'),
    comparison='exact-model',
)
out = Path(a.out).resolve()
out.mkdir(parents=True, exist_ok=True)
write(out / 'config.json', cfg)
summary = run(out / 'config.json', out / 'runs')
print(json.dumps({key: summary[key] for key in ['performance_conclusion_valid', 'attempts',
                 'complete_blocks', 'expected_blocks', 'correctness_failed']}))
if not summary['performance_conclusion_valid']:
    raise SystemExit('Shared comparison incomplete or exact model validation failed')
