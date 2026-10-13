"""Check complete model/trace equality and summarize paired, separate probes."""
from collections import defaultdict
import json
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent


def records(group):
    return json.loads((HERE / (group + '-runs.json')).read_text())


def paired(group, trace=False):
    pairs = defaultdict(dict)
    for record in records(group):
        pairs[(record['case'], record['workers'], record['repeat'])][record['arm']] = record
    checks = []
    for key, arms in pairs.items():
        assert set(arms) == {'baseline', 'candidate'}, (group, key, arms.keys())
        baseline, candidate = arms['baseline'], arms['candidate']
        assert baseline['model_sha256'] == candidate['model_sha256'], (group, key, 'model')
        if trace:
            assert baseline['trace_sha256'] == candidate['trace_sha256'], (group, key, 'trace')
        checks.append({'case':key[0], 'workers':key[1], 'repeat':key[2],
                       'model_sha256':baseline['model_sha256'],
                       'trace_sha256':baseline['trace_sha256'] if trace else None})
    return checks


def timing(group):
    grouped = defaultdict(lambda: defaultdict(dict))
    for record in records(group):
        grouped[(record['case'], record['workers'])][record['repeat']][record['arm']] = record
    summary = []
    for (case, workers), repeats in grouped.items():
        ratios, cpu_ratios, times, cpus, hwms = [], [], defaultdict(list), defaultdict(list), defaultdict(list)
        for arms in repeats.values():
            for arm, record in arms.items():
                times[arm].append(record['metrics']['train_seconds'])
                hwms[arm].append(record['metrics']['process_hwm_kib_before_validation'])
                cpus[arm].append(record['metrics']['train_cpu_seconds'])
            ratios.append(100 * (arms['candidate']['metrics']['train_seconds'] /
                                 arms['baseline']['metrics']['train_seconds'] - 1))
            cpu_ratios.append(100 * (arms['candidate']['metrics']['train_cpu_seconds'] /
                                     arms['baseline']['metrics']['train_cpu_seconds'] - 1))
        summary.append({'case':case, 'workers':workers, 'pairs':len(repeats),
                        'train_seconds_median':{arm:median(values) for arm, values in times.items()},
                        'paired_change_percent':ratios, 'paired_change_percent_median':median(ratios),
                        'train_cpu_seconds_median':{arm:median(values) for arm,values in cpus.items()},
                        'paired_cpu_change_percent':cpu_ratios,
                        'hwm_kib':dict(hwms)})
    return summary


def telemetry(group):
    result = []
    for record in records(group):
        stages, layouts, prepared, types = [], [], [], []
        for line in (Path(record['directory']) / 'stderr.log').read_text().splitlines():
            for prefix, target in [('BPE_STAGE ', stages), ('BPE_LAYOUT ', layouts),
                                   ('BPE_PREPARED ', prepared), ('BPE_TYPES ', types)]:
                if line.startswith(prefix):
                    target.append(json.loads(line[len(prefix):]))
        phase = defaultdict(list)
        for event in stages:
            phase[event['stage']].append(event)
        result.append({'case':record['case'], 'workers':record['workers'], 'arm':record['arm'],
                       'types':types, 'layouts':layouts,
                       'stage_totals':{name:sum(e['seconds'] for e in events)
                                       for name, events in phase.items()},
                       'stage_max_rss_kib':{name:max(int(e['rss'].split()[1]) for e in events)
                                            for name, events in phase.items()},
                       'max_prepared_capacity':{key:max((e[key] for e in prepared), default=0)
                                                for key in ['chunks_capacity','starts_capacity',
                                                            'events_capacity','birth_positions']}})
    return result


if __name__ == '__main__':
    result = {'trace_checks':paired('final-trace', trace=True),
              'stage_checks':paired('final-stage'),
              'timing_checks':paired('final-timing'),
              'timing':timing('final-timing'), 'telemetry':telemetry('final-stage')}
    for group in ['targeted-repeat','baseline-noise','repeated-timing','repeated-noise','repeated-confirm','memory-repeat','memory-noise']:
        if (HERE / (group + '-runs.json')).exists():
            result[group] = {'checks':paired(group), 'timing':timing(group)}
    (HERE / 'summary.json').write_text(json.dumps(result, indent=2))
    for row in result['timing']:
        print(row['case'], row['workers'], row['paired_change_percent'])
