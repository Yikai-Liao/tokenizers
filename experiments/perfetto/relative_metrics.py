"""Dimensionless metrics: ordinary prepare batches and all commit batches."""
import bisect
import collections
import csv
import json
import statistics
from pathlib import Path

ROOT = Path('/root/code/tokenizers-perfetto-results')
CASES = ['en-whitespace', 'zh-whitespace', 'en-bytelevel', 'zh-bytelevel']


def jobs_by_round(run):
    _, buffers = json.loads((run / 'events.json').read_text())
    rounds = collections.defaultdict(list)
    for _, events in buffers:
        jobs = sorted((e for e in events if e['name'] == 'prepare.job'), key=lambda e: e['ts'])
        starts = [j['ts'] for j in jobs]
        for j in jobs:
            j['matched'] = 0
        for e in events:
            if e['name'] != 'prepare.scan':
                continue
            i = bisect.bisect_right(starts, e['ts']) - 1
            if i >= 0 and e['ts'] + e['dur'] <= jobs[i]['ts'] + jobs[i]['dur']:
                jobs[i]['matched'] += e['fields'][2]
        for j in jobs:
            rounds[j['round']].append(j)
    return rounds


def share_distance(values, durations):
    # Total variation distance between the predicted and observed job shares.
    # 0 is perfect agreement; 1 means entirely disjoint shares.
    a, b = sum(values), sum(durations)
    return sum(abs(x / a - y / b) for x, y in zip(values, durations)) / 2


def phase_summary(rows):
    busy = sum(r['busy_ns'] for r in rows)
    capacity = sum(r['capacity_ns'] for r in rows)
    loss = capacity - busy
    boundary = sum(r['boundary_ns'] for r in rows)
    return dict(
        batches=len(rows),
        batches_without_jobs=sum(r['tasks'] == 0 for r in rows),
        mean_rules=statistics.mean(r['rules'] for r in rows),
        median_rules=statistics.median(r['rules'] for r in rows),
        median_jobs=statistics.median(r['tasks'] for r in rows),
        occupancy=busy / capacity,
        boundary_share_of_capacity=boundary / capacity,
        boundary_share_of_loss=boundary / loss,
        insufficient_share_of_loss=sum(r['insufficient_tasks_ns'] for r in rows) / loss,
        ready_share_of_loss=sum(r['ready_gap_ns'] for r in rows) / loss,
        # Indivisible-task bound, even with zero dispatch cost and ideal scheduling.
        fixed_task_granularity_occupancy_ceiling=busy / sum(4 * r['job_lower_bound_ns'] for r in rows),
        median_boundary_to_ideal_work=statistics.median(r['boundary_ns'] / r['busy_ns'] for r in rows if r['busy_ns']),
    )


def analyze(case):
    phase_rows = collections.defaultdict(list)
    for run in sorted((ROOT / 'runs').glob(f'{case}-w4-coarse-r[123]')):
        with (run / 'batches.csv').open() as f:
            for row in csv.DictReader(f):
                phase_rows[row['phase']].append({k: float(v) for k, v in row.items() if k != 'phase'})
    rep_stats = []
    for run in sorted((ROOT / 'runs').glob(f'{case}-w4-detail-r[123]')):
        rounds = jobs_by_round(run)
        original = json.loads((run / 'job-costs.json').read_text())
        assert sum(len(jobs) for jobs in rounds.values()) == original['jobs']
        assert sum(j['fields'][2] for jobs in rounds.values() for j in jobs) == original['raw']
        assert sum(j['matched'] for jobs in rounds.values() for j in jobs) == original['matched']
        sums = collections.defaultdict(float)
        med = collections.defaultdict(list)
        zero_jobs = jobs_count = zero_time = all_time = 0
        for jobs in rounds.values():
            raw = [j['fields'][2] for j in jobs]
            matched = [j['matched'] for j in jobs]
            times = [j['dur'] for j in jobs]
            t = sum(times)
            all_time += t
            jobs_count += len(jobs)
            zero_jobs += sum(v == 0 for v in matched)
            zero_time += sum(d for v, d in zip(matched, times) if v == 0)
            if not sum(matched) or len(jobs) < 2:
                continue
            sums['raw_matched_share_distance'] += t * share_distance(raw, matched)
            for name, values in [('raw', raw), ('matched', matched)]:
                sums[f'{name}_time_share_distance'] += t * share_distance(values, times)
                med[f'{name}_heaviest_to_four_worker_ideal'].append(4 * max(values) / sum(values))
            med['time_heaviest_to_four_worker_ideal'].append(4 * max(times) / t)
            sums['weight'] += t
        rep_stats.append(dict(
            run=run.name,
            **{k: v / sums['weight'] for k, v in sums.items() if k != 'weight'},
            **{f'median_{k}': statistics.median(v) for k, v in med.items()},
            zero_matched_job_share=zero_jobs / jobs_count,
            zero_matched_job_time_share=zero_time / all_time,
        ))
    return dict(case=case, phases={p: phase_summary(rows) for p, rows in phase_rows.items()}, detailed_repeats=rep_stats)


if __name__ == '__main__':
    result = [analyze(case) for case in CASES]
    (ROOT / 'relative-metrics.json').write_text(json.dumps(result, indent=2))
    for r in result:
        print(json.dumps(r))
