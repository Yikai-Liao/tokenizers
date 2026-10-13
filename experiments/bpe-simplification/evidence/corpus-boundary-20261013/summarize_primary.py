"""Compare complete single-call training with explicit binary provenance."""
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent


def summarize(groups, baseline_hash, candidate_hash):
    pairs = defaultdict(dict)
    for group in groups:
        path = HERE / f'{group}-runs.json'
        if not path.exists():
            continue
        for record in json.loads(path.read_text()):
            arm = record['arm']
            if arm not in ('baseline', 'candidate'):
                continue
            expected = baseline_hash if arm == 'baseline' else candidate_hash
            assert record['binary_sha256'] == expected, (group, arm, 'binary')
            assert not record.get('iteration_model_sha256'), (group, 'repeated caller')
            key = (record['case'], record['workers'], group, record['repeat'])
            assert arm not in pairs[key]
            pairs[key][arm] = record
    cases = defaultdict(list)
    checks = []
    for key, pair in pairs.items():
        assert set(pair) == {'baseline', 'candidate'}, key
        old, new = pair['baseline'], pair['candidate']
        assert old['input_sha256'] == new['input_sha256'], key
        assert old['model_sha256'] == new['model_sha256'], key
        cases[key[:2]].append(pair)
        checks.append({'case':key[0], 'workers':key[1], 'group':key[2],
                       'repeat':key[3], 'model_sha256':old['model_sha256']})
    rows = []
    for (case, workers), samples in sorted(cases.items()):
        row = {'case':case, 'workers':workers, 'pairs':len(samples)}
        for metric in ['train_seconds', 'train_cpu_seconds']:
            values = {arm:[p[arm]['metrics'][metric] for p in samples]
                      for arm in ['baseline', 'candidate']}
            changes = [100 * (p['candidate']['metrics'][metric] /
                              p['baseline']['metrics'][metric] - 1) for p in samples]
            row[metric] = {'median':{a:median(v) for a,v in values.items()},
                           'paired_percent':changes, 'median_percent':median(changes),
                           'range_percent':[min(changes), max(changes)]}
        row['hwm_kib'] = {arm:[p[arm]['metrics']['process_hwm_kib_before_validation']
                               for p in samples] for arm in ['baseline', 'candidate']}
        rows.append(row)
    return {'groups':groups, 'baseline_binary_sha256':baseline_hash,
            'candidate_binary_sha256':candidate_hash, 'checks':checks, 'timing':rows}


if __name__ == '__main__':
    original = json.loads((HERE / 'final-manifest.json').read_text())['binaries']
    aligned = json.loads((HERE / 'aligned-manifest.json').read_text())
    base = original['final-baseline-production']
    result = {'boundary':'one complete public do_train call per fresh process; '
                         'input loading and model export timed separately',
              'original_candidate':summarize(
                  ['final-timing', 'targeted-repeat', 'isolate-step3',
                   'range-en-bytelevel', 'range-normal', 'range-fixtures',
                   'codegen-borrow'],
                  base, original['final-candidate-production']),
              'aligned_candidate_acceptance_series':summarize(
                  ['aligned-timing', 'aligned-confirm', 'aligned-targeted', 'aligned-steady'],
                  base, aligned['binary_sha256']),
              'aligned_candidate':summarize(['aligned-timing', 'aligned-confirm',
                                             'aligned-targeted', 'aligned-steady',
                                             'zh-birth-layout', 'zh-match-span'],
                                             base, aligned['binary_sha256']),
              'same_baseline_control':summarize(['aligned-noise',
                                                'aligned-target-noise',
                                                'aligned-steady-noise'], base, base)}
    (HERE / 'primary-summary.json').write_text(json.dumps(result, indent=2))
    for name in ['aligned_candidate', 'same_baseline_control']:
        for row in result[name]['timing']:
            print(name, row['case'], row['workers'], row['pairs'],
                  round(row['train_seconds']['median_percent'], 3),
                  round(row['train_cpu_seconds']['median_percent'], 3))
