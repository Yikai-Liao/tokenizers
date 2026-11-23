"""Aggregate same-machine benchmark records and write the GitHub job summary."""
from pathlib import Path
import argparse, json, os, statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--expected-versions', default='')
    args = parser.parse_args()
    rows = []
    for path in args.artifacts.glob('bpe-*/results.jsonl'):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    expected_versions = [v for v in args.expected_versions.split(',') if v] or sorted({row['version'] for row in rows})
    errors = []
    for version in expected_versions:
        if not any(row['version'] == version for row in rows):
            errors.append(f'missing completed version {version}')
    summaries = []
    for version in expected_versions:
        for workers in (1, 4):
            for language in ('en', 'zh'):
                got = [r for r in rows if r['version'] == version and r['workers'] == workers and r['language'] == language and not r['warmup']]
                if len(got) != args.rounds:
                    errors.append(f'missing rounds for {version}/{language}/workers={workers}')
                    continue
                if any(r.get('model_equal_baseline') is not True or r['process']['exit_code'] != 0 for r in got):
                    errors.append(f'model or process failure for {version}/{language}/workers={workers}')
                summaries.append({
                    'version': version,
                    'ref': got[0].get('ref', version),
                    'workers': workers,
                    'language': language,
                    'feed_seconds': statistics.median(r['metrics']['feed_seconds'] for r in got),
                    'feed_cpu_seconds': statistics.median(r['metrics']['feed_cpu_seconds'] for r in got),
                    'train_seconds': statistics.median(r['metrics']['train_seconds'] for r in got),
                    'train_cpu_seconds': statistics.median(r['metrics']['train_cpu_seconds'] for r in got),
                    'peak_rss_mib': statistics.median(r['process']['maxrss_kib'] for r in got) / 1024,
                    'model_match': all(r.get('model_equal_baseline') is True for r in got),
                })
    baseline = {(row['language'], row['workers']): row for row in summaries if row['version'] == 'baseline'}
    for row in summaries:
        reference = baseline.get((row['language'], row['workers']))
        if reference is None:
            errors.append(f'missing baseline row for {row["language"]}/workers={row["workers"]}')
            continue
        row['feed_wall_ratio'] = row['feed_seconds'] / reference['feed_seconds']
        row['train_wall_ratio'] = row['train_seconds'] / reference['train_seconds']
        row['peak_rss_ratio'] = row['peak_rss_mib'] / reference['peak_rss_mib']
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {'complete': not errors, 'errors': errors, 'rounds': args.rounds, 'versions': expected_versions, 'rows': summaries}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    table = [
        '# BPE benchmark',
        f'{args.rounds} timed rounds after one warmup per version on each language/worker runner.',
        '',
        '| Version | Ref | Workers | Language | Feed wall s | Feed CPU s | Train wall s | Train CPU s | Peak RSS MiB | Feed wall × baseline | Train wall × baseline | RSS × baseline | Model match |',
        '| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |',
    ]
    for row in summaries:
        table.append('| {version} | {ref} | {workers} | {language} | {feed_seconds:.3f} | {feed_cpu_seconds:.3f} | {train_seconds:.3f} | {train_cpu_seconds:.3f} | {peak_rss_mib:.1f} | {feed_wall_ratio:.3f} | {train_wall_ratio:.3f} | {peak_rss_ratio:.3f} | {model_match} |'.format(**row))
    if errors:
        table += ['', '## Failures', *[f'- {error}' for error in errors]]
    text = '\n'.join(table) + '\n'
    (args.out / 'summary.md').write_text(text)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as stream:
            stream.write(text)
    (args.out / 'results.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    print(json.dumps({'complete': not errors, 'versions': expected_versions, 'rows': len(rows), 'errors': errors}))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
