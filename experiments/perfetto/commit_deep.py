"""Attribute owner-duration inequality separately from total owner CPU cost."""
import bisect
import collections
import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path('/root/code/tokenizers-perfetto-results')
SUB = ['commit.group', 'commit.counts', 'commit.completed', 'commit.aggregate', 'commit.encode_publish', 'commit.prefix']
CASES = ['en-whitespace', 'zh-whitespace', 'en-bytelevel', 'zh-bytelevel']


def extract(run):
    _, buffers = json.loads((run / 'events.json').read_text())
    rounds = collections.defaultdict(list)
    phases = {}
    routes = {}
    for tid, events in buffers:
        owners = sorted((e for e in events if e['name'] == 'commit.owner'), key=lambda e: e['ts'])
        starts = [o['ts'] for o in owners]
        for o in owners:
            o['sub'] = collections.Counter()
            o['tid'] = tid
        for e in events:
            if e['name'] == 'commit':
                phases[e['round']] = e
            if e['name'] == 'commit.route':
                routes[e['round']] = e
            if e['name'] not in SUB:
                continue
            i = bisect.bisect_right(starts, e['ts']) - 1
            if i >= 0 and e['ts'] + e['dur'] <= owners[i]['ts'] + owners[i]['dur']:
                owners[i]['sub'][e['name']] += e['dur']
        for o in owners:
            o['sub']['other'] = o['dur'] - sum(o['sub'].values())
            assert o['sub']['other'] >= 0
            rounds[o['round']].append(o)
    return rounds, phases, routes


def analyze_run(run):
    rounds, phases, routes = extract(run)
    result = json.loads((run / 'result.json').read_text())
    totals = collections.Counter()
    excess = collections.Counter()
    critical_workers = collections.Counter()
    critical_owners = collections.Counter()
    med = collections.defaultdict(list)
    errors = collections.Counter()
    weighted_heaviest = collections.Counter()
    critical_series = {}
    share_series = {}
    for round_id, owners in rounds.items():
        critical = max(owners, key=lambda o: o['dur'])
        critical_series[round_id] = critical['fields'][0]
        busy = sum(o['dur'] for o in owners)
        bound = max(critical['dur'], busy / 4)
        task_span = max(o['ts'] + o['dur'] for o in owners) - min(o['ts'] for o in owners)
        boundary = phases[round_id]['dur'] - task_span
        totals.update(busy=busy, critical=critical['dur'], grain_gap=4 * bound - busy,
                      boundary=4 * boundary, dispatch=4 * (task_span - bound))
        critical_workers[critical['fields'][4]] += critical['dur']
        critical_owners[critical['fields'][0]] += critical['dur']
        for name in SUB + ['other']:
            excess[name] += 4 * critical['sub'][name] - sum(o['sub'][name] for o in owners)
        share_series[round_id] = {o['fields'][0]: o['dur'] / busy for o in owners}
        for name, field in [('references', 1), ('completed_births', 3)]:
            values = [o['fields'][field] for o in owners]
            if sum(values):
                med[name].append(4 * max(values) / sum(values))
                weighted_heaviest[name] += busy * 4 * max(values) / sum(values)
                errors[name] += busy * sum(abs(v / sum(values) - o['dur'] / busy) for v, o in zip(values, owners)) / 2
                errors[name + '_weight'] += busy
        med['duration'].append(4 * critical['dur'] / busy)
        weighted_heaviest['duration'] += 4 * critical['dur']
    phase_time = sum(e['dur'] for e in phases.values())
    # Empty rounds have only phase boundary cost.
    totals['boundary'] += 4 * sum(p['dur'] for r, p in phases.items() if r not in rounds)
    assert abs(4 * phase_time - totals['busy'] - totals['boundary'] - totals['grain_gap'] - totals['dispatch']) < 1
    assert abs(sum(excess.values()) - totals['grain_gap']) < 1
    train = result['metrics']['train_seconds'] * 1e9
    out = dict(run=run.name, batches=len(phases),
               capacity={name: value / (4 * phase_time) for name, value in totals.items()},
               route_share_of_phase=sum(e['dur'] for e in routes.values()) / phase_time,
               owner_inequality_attribution={k: v / totals['grain_gap'] for k, v in excess.items()},
               critical_worker_time_share={k: v / totals['critical'] for k, v in critical_workers.items()},
               critical_owner_time_share={k: v / totals['critical'] for k, v in critical_owners.items()},
               median_heaviest_to_four_worker_ideal={k: statistics.median(v) for k, v in med.items()},
               time_weighted_mean_heaviest_to_four_worker_ideal={k: v / (totals['busy'] if k == 'duration' else errors[k + '_weight']) for k, v in weighted_heaviest.items()},
               within_batch_time_weighted_share_error={k: errors[k] / errors[k + '_weight'] for k in med if k != 'duration'},
               whole_training_fraction=phase_time / train,
               model_upper_bound_whole_training_savings=dict(
                   perfect_job_scheduling=totals['dispatch'] / 4 / train,
                   perfect_divisible_owner_work=totals['grain_gap'] / 4 / train,
                   zero_routing=sum(e['dur'] for e in routes.values()) / train))
    return out, critical_series, share_series


def pearson(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    xx = sum((x - mx) ** 2 for x in xs)
    yy = sum((y - my) ** 2 for y in ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(xx * yy)


def analyze(case):
    results = [analyze_run(run) for run in sorted((ROOT / 'runs').glob(f'{case}-w4-detail-r[123]'))]
    stability = []
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        common = results[a][1].keys() & results[b][1].keys()
        xs, ys = [], []
        for r in common:
            for owner in range(4):
                xs.append(results[a][2][r].get(owner, 0))
                ys.append(results[b][2][r].get(owner, 0))
        stability.append(dict(repeats=[a + 1, b + 1],
                              same_critical_owner_fraction=sum(results[a][1][r] == results[b][1][r] for r in common) / len(common),
                              within_batch_owner_duration_share_correlation=pearson(xs, ys)))
    return dict(case=case, repeats=[r[0] for r in results], stability=stability)


if __name__ == '__main__':
    result = [analyze(case) for case in CASES]
    (ROOT / 'commit-deep.json').write_text(json.dumps(result, indent=2))
    for r in result:
        print(json.dumps(r))
