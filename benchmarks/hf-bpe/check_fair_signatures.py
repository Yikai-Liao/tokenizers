#!/usr/bin/env python3
"""Require identical complete model signatures across PR and native fair runs."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', nargs='+', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if len(args.results) < 2:
        raise SystemExit('provide the PR run and every native run')
    signature = None
    rows = []
    for path in args.results:
        lines = path.read_text().splitlines()
        if len(lines) != 1:
            raise SystemExit(f'{path}: expected one completed run record, got {len(lines)}')
        row = json.loads(lines[0])
        fields = ('model_sha256', 'actual_vocab', 'actual_merges', 'unique_words')
        if any(field not in row for field in fields):
            raise SystemExit(f'{path}: incomplete or failed run record')
        value = tuple(row[field] for field in fields)
        if signature is None:
            signature = value
        elif signature != value:
            raise SystemExit(f'{path}: model signature mismatch: {value} != {signature}')
        rows.append((row.get('engine', path.name), value))
    record = dict(signature=dict(model_sha256=signature[0], actual_vocab=signature[1],
                                 actual_merges=signature[2], unique_words=signature[3]),
                  runs=[dict(name=name, signature=dict(model_sha256=value[0], actual_vocab=value[1],
                                                       actual_merges=value[2], unique_words=value[3]))
                        for name, value in rows], matches=True)
    if args.output:
        args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(f'complete model signature matches across {len(rows)} runs: {signature}')


if __name__ == '__main__':
    main()
