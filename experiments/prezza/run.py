#!/usr/bin/env python3
"""Run exact-model ablations with tokenizers-bpe-benchmarks (PYTHONPATH)."""
import argparse
from pathlib import Path
from bench.config import write
from bench.inputs import prepared, text_manifest
from bench.runs import run

p = argparse.ArgumentParser()
p.add_argument('--baseline', required=True)
p.add_argument('--candidate', required=True)
p.add_argument('--out', required=True)
p.add_argument('--words')
p.add_argument('--smoke', action='store_true')
p.add_argument('--diagnostic', action='store_true')
p.add_argument('--workers', default='1,2,4,6')
p.add_argument('--blocks', type=int, default=3)
a = p.parse_args()
out = Path(a.out).resolve()
out.mkdir(parents=True, exist_ok=True)
workers = [int(w) for w in a.workers.split(',')]
import os
cpus = sorted(os.sched_getaffinity(0))[:max(workers)]
if len(cpus) < max(workers):
    raise SystemExit('Not enough CPUs in supervisor affinity')
base, candidate = str(Path(a.baseline).resolve()), str(Path(a.candidate).resolve())
extra = {'BPE_CORPUS_STATS': '1'} if a.diagnostic else {}
arms = {
    'baseline': {'build': candidate if a.diagnostic else base, 'environment': extra},
    'prezza': {'build': candidate, 'environment': {**extra, 'BPE_CORPUS_LAYOUT': 'prezza'}},
}
if not (a.smoke or a.diagnostic):
    arms['endpoint_control'] = {'build': candidate, 'environment': {}}
execution = dict(workers=workers, cpu_set=cpus, warmups_per_cell=0 if a.smoke or a.diagnostic else 1,
                 paired_blocks=a.blocks, timeout_seconds=900, min_available_gib=2,
                 max_process_rss_gib=10, order='balanced-alternating')
if a.smoke:
    text = out / 'fixture.txt'
    text.write_text(('aaaa aaab abab roses reddish café 中文 字符 12345 🙂\n') * 13)
    text_manifest(text, out / 'text.json')
    for tokenizer in ['whitespace', 'whitespace_split', 'none', 'bytelevel_regex']:
        words = out / ('words-' + tokenizer)
        prepared(text, tokenizer, base, words)
        for mode in ['core', 'pipeline']:
            cfg = dict(schema_version=1, name='prezza-smoke', mode=mode, arms=arms,
                       cases=[dict(name='fixture', input_manifest=str(words/'manifest.json' if mode=='core' else out/'text.json'),
                       pretokenizer=tokenizer, trainer=dict(vocab_size=300, min_frequency=2, max_token_length=12))],
                       execution=execution, comparison='exact-model')
            name = mode + '-' + tokenizer
            write(out/(name+'.json'), cfg)
            summary = run(out/(name+'.json'), out/name)
            if not summary['performance_conclusion_valid']:
                raise SystemExit('Exact model validation failed: '+name)
else:
    if not a.words:
        p.error('--words is required for full-size core experiments')
    cfg = dict(schema_version=1, name='prezza-zh512-whitespace'+('-diagnostic' if a.diagnostic else ''), mode='core', arms=arms,
               cases=[dict(name='zh512-whitespace-v'+str(v), input_manifest=str(Path(a.words).resolve()), pretokenizer='whitespace',
                           trainer=dict(vocab_size=v, min_frequency=2, prefix=None, suffix=None, max_token_length=None))
                      for v in [50000, 100000]], execution=execution, comparison='exact-model')
    write(out/'config.json', cfg)
    summary = run(out/'config.json', out/'runs')
    if not summary['performance_conclusion_valid']:
        raise SystemExit('Exact model validation failed')
