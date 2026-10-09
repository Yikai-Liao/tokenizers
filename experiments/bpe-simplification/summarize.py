"""Recompute workload-specific paired ratios and block bootstrap intervals."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics


def interval(values):
    rng = random.Random(20261009)
    samples = sorted(statistics.median(rng.choices(values, k=len(values)))
                     for _ in range(10000))
    return [samples[250], samples[9749]]


def summarize(path):
    groups = defaultdict(dict)
    excluded = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row['warmup']:
            continue
        if not row['valid']:
            excluded.append(row)
            continue
        groups[(row['case'], row['block'])][row['arm']] = row
    results = defaultdict(list)
    for (case, block), arms in sorted(groups.items()):
        candidate = arms.get('B', arms.get('A2'))
        baseline = arms.get('A')
        if candidate is None or baseline is None:
            continue
        results[case].append((block, baseline, candidate))
    report = dict(source=str(path), cases={}, invalid_samples=excluded)
    for case, pairs in results.items():
        metrics = ['train_seconds', 'train_cpu_seconds', 'process_hwm_kib_before_validation']
        if 'pipeline_seconds' in pairs[0][1]['metrics']:
            metrics += ['pipeline_seconds', 'pipeline_cpu_seconds', 'feed_seconds']
        result = dict(pairs=len(pairs), metrics={}, models_equal=all(a['model_equal'] and b['model_equal'] for _, a, b in pairs),
                      sample_limit='Exploratory bootstrap with few pairs; host is a KVM guest.')
        for metric in metrics:
            ratios = [b['metrics'][metric] / a['metrics'][metric] for _, a, b in pairs]
            result['metrics'][metric] = dict(
                paired_median_ratio=statistics.median(ratios), bootstrap_95=interval(ratios),
                baseline_median=statistics.median(a['metrics'][metric] for _, a, _ in pairs),
                candidate_median=statistics.median(b['metrics'][metric] for _, _, b in pairs),
                raw=[dict(block=block, baseline=a['metrics'][metric], candidate=b['metrics'][metric], ratio=ratio)
                     for (block, a, b), ratio in zip(pairs, ratios)],
            )
        result['rss_delta_mib_median'] = statistics.median(
            (b['metrics']['process_hwm_kib_before_validation'] - a['metrics']['process_hwm_kib_before_validation']) / 1024
            for _, a, b in pairs)
        report['cases'][case] = result
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = summarize(args.input)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2))
    for case, result in report['cases'].items():
        print(case, 'pairs=', result['pairs'], 'RSS delta MiB=', round(result['rss_delta_mib_median'], 2))
        for metric, values in result['metrics'].items():
            if metric in ('train_seconds', 'pipeline_seconds', 'process_hwm_kib_before_validation'):
                print(' ', metric, 'paired median=', round(values['paired_median_ratio'], 4),
                      '95%=', [round(v, 4) for v in values['bootstrap_95']])


if __name__ == '__main__':
    main()
