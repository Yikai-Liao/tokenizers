"""Check the aligned implementation against the frozen production-grain trace."""
import json
from pathlib import Path

from summarize import telemetry

HERE = Path(__file__).resolve().parent


def records(group):
    return json.loads((HERE / f'{group}-runs.json').read_text())


def compare(old_group, new_group, trace=False):
    old = {(r['case'], r['workers']):r for r in records(old_group)
           if r['arm'] == 'baseline'}
    checks = []
    for new in records(new_group):
        if new['arm'] != 'candidate':
            continue
        key = (new['case'], new['workers'])
        reference = old[key]
        assert reference['input_sha256'] == new['input_sha256'], key
        assert reference['model_sha256'] == new['model_sha256'], key
        if trace:
            assert reference['trace_sha256'] == new['trace_sha256'], key
        checks.append({'case':key[0], 'workers':key[1],
                       'model_sha256':new['model_sha256'],
                       'trace_sha256':new['trace_sha256'] if trace else None,
                       'coverage':new['coverage']})
    return checks


if __name__ == '__main__':
    traces = compare('final-trace', 'aligned-trace', trace=True)
    assert len(traces) == 28
    steady = compare('aligned-steady-trace', 'aligned-steady-trace', trace=True)
    assert len(steady) == 2
    assert all(r['coverage']['multi_rule'] > 0 for r in steady)
    for case, condition in [('aa-large', lambda c:c['aa'] > 0),
                            ('ab-large', lambda c:c['multi_rule'] > 0),
                            ('reserved-large', lambda c:c['existing_fresh'] > 0),
                            ('alias-large', lambda c:c.get('restart', 0) > 0),
                            ('length-reuse-large', lambda c:c.get('restart', 0) > 0)]:
        assert all(condition(r['coverage']) for r in traces if r['case'] == case)
    result = {'trace_checks':traces + steady,
              'stage_checks':compare('final-stage', 'aligned-stage'),
              'telemetry':telemetry('aligned-stage')}
    (HERE / 'aligned-validation.json').write_text(json.dumps(result, indent=2))
    print('30 complete model/trace comparisons passed; targeted stage models match')
