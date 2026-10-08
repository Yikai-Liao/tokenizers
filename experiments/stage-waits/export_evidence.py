#!/usr/bin/env python3
"""Export a completed comparison, deduplicating only identical model values."""
import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

from bench.config import identity, read, write
from bench.runs import canonical_model


def export(comparison, destination):
    runs = comparison / 'runs'
    summary = read(runs / 'report/summary.json')
    if not summary['performance_conclusion_valid']:
        raise SystemExit('Comparison is incomplete or failed exact-model validation')
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(runs / 'report/summary.json', destination / 'summary.json')
    (destination / 'comparisons.csv').write_text((runs / 'report/comparisons.csv').read_text())
    references = {p.stem: read(p) for p in (runs / 'models').glob('*.json')}
    model_map = []
    for path in sorted((runs / 'attempts').glob('*/result.json')):
        row = read(path)
        if row['status'] != 'ok':
            raise SystemExit(f'Non-successful attempt: {row["attempt_id"]}')
        model_path = path.parent / 'model.json'
        model = canonical_model(model_path)
        if model != references[row['case']] or identity(model) != row['model_sha256']:
            raise SystemExit(f'Model evidence changed: {row["attempt_id"]}')
        model_map.append(dict(attempt_id=row['attempt_id'], slot=row['slot'],
                              raw_file_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                              canonical_sha256=identity(model),
                              reference=f"runs/models/{row['case']}.json"))
    write(destination / 'models.json', model_map)
    members = []
    for path in sorted(comparison.rglob('*')):
        if not path.is_file() or path.name == '.writer.lock':
            continue
        relative = path.relative_to(comparison)
        if path.name == 'model.json' and 'attempts' in relative.parts:
            continue
        members.append((path, str(relative)))
    # The unchanged harness and all runner source files make parsing reproducible.
    bench_root = comparison.parents[2]
    for folder in ['bench', 'runner']:
        for path in sorted((bench_root / folder).rglob('*')):
            if path.is_file() and (path.suffix == '.py' or path.suffix == '.rs'
                                  or path.name == 'Cargo.toml'):
                members.append((path, 'harness/' + str(path.relative_to(bench_root))))
    members.append((bench_root / '.bench/prezza/runner.lock', 'harness/runner.lock'))
    for name in ['zh256', 'en256', 'zh256-words', 'en256-words']:
        path = bench_root / '.bench/owner' / name / 'manifest.json'
        members.append((path, 'corpora/' + name + '/manifest.json'))
    archive = destination / 'raw-metadata-and-models.tar.gz'
    # mtime=0 and sorted paths avoid changing the archive on repeated exports.
    with archive.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode='w|') as tar:
            for path, name in sorted(members, key=lambda item: item[1]):
                info = tar.gettarinfo(str(path), arcname=name)
                info.mtime = info.uid = info.gid = 0
                info.uname = info.gname = ''
                with path.open('rb') as data:
                    tar.addfile(info, data)
    write(destination / 'archive.json', dict(
        path=archive.name, bytes=archive.stat().st_size,
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), members=len(members),
        attempts=len(model_map), model_values=len(references),
        model_deduplication='Sort vocabulary by spelling, preserving IDs and merge order; '
                            'each verified attempt refers to its identical case model.'))
    print(json.dumps(dict(attempts=len(model_map), archive_bytes=archive.stat().st_size)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    export(args.comparison.resolve(), args.out.resolve())
