#!/usr/bin/env python3
"""Run the fixed four-case block prepare comparison, recording exact provenance."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results/block-fused-prepare'


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    cases = []
    for language, paired, vocab in [('en16m', 'en16m-none4', 30000), ('zh512m', 'zh512m-none4', 50000)]:
        metadata = json.loads((ROOT / 'results/affix-all-fast-v4' / (paired + '.environment.json')).read_text())
        for mode, label in [('plan', 'block-plan-baseline'), ('fragments', 'block-fragments-candidate')]:
            case = language + '-' + mode
            command = [sys.executable, str(ROOT / 'run_affix_analysis.py'), '--label', label,
                       '--case', case, '--corpus', metadata['corpus_path'], '--result-dir', str(RESULTS),
                       '--threads', '4', '--vocab', str(vocab), '--min-frequency', '2',
                       '--require-absent-affixes', '--baseline-case', paired,
                       '--baseline-result-dir', str(ROOT / 'results/affix-all-fast-v4'),
                       '--expected-corpus-sha', metadata['corpus_sha256']]
            if mode == 'fragments':
                command.append('--require-block-fused')
            cases.append({'case': case, 'mode': mode, 'command': command})
    plan = {'planned_calls': len(cases), 'cases': cases, 'completed': [], 'failures': []}
    plan_path = RESULTS / 'run-plan.json'
    if plan_path.exists() or any((RESULTS / (c['case'] + '.jsonl')).exists() for c in cases):
        raise SystemExit('refusing to overwrite prior block measurements; use a fresh result directory')
    plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    for case in cases:
        result = subprocess.run(case['command'])
        if result.returncode:
            plan['failures'].append({'case': case['case'], 'returncode': result.returncode})
            plan_path.write_text(json.dumps(plan, indent=2) + '\n')
            raise SystemExit(result.returncode)
        row = json.loads((RESULTS / (case['case'] + '.jsonl')).read_text())
        stats = row['indexed_stats']
        expected = {'workers': 4, 'initialization_workers': 4, 'atomic_corpus': True,
                    'layout': 'parallel_u32_dict16', 'corpus_slot_bytes': 4}
        actual = {name: stats.get(name) for name in expected}
        if actual != expected:
            raise SystemExit(f"incorrect block configuration for {case['case']}: {actual}")
        plan['completed'].append(case['case'])
        plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    rows = {case['case']: json.loads((RESULTS / (case['case'] + '.jsonl')).read_text()) for case in cases}
    summary = {'completed_calls': len(plan['completed']), 'rows': rows}
    for language in ('en16m', 'zh512m'):
        old = rows[language + '-plan']
        new = rows[language + '-fragments']
        summary[language] = {'train_speedup': old['train_ms'] / new['train_ms'],
                             'train_change_percent': (new['train_ms'] / old['train_ms'] - 1) * 100,
                             'merge_change_percent': (new['indexed_stats']['merge_ms'] / old['indexed_stats']['merge_ms'] - 1) * 100,
                             'rss_change_percent': (new['peak_rss_bytes'] / old['peak_rss_bytes'] - 1) * 100,
                             'model_equal': old['model_sha256'] == new['model_sha256']}
    (RESULTS / 'analysis.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
