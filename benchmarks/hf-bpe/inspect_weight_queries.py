#!/usr/bin/env python3
"""Inspect actual native ELF weight-query call sites before timed runs."""
import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def inspect(binary):
    names = {}
    targets = set()
    for line in subprocess.check_output(['nm', '-C', str(binary)], text=True).splitlines():
        match = re.match(r'([0-9a-f]+) [tT] (.*)', line)
        if not match:
            continue
        address = int(match[1], 16)
        name = match[2]
        names[address] = name
        if name.startswith('<tk_train::trainers::bpe::indexed::parallel::weight_lookup::WeightLookup>::weight::'):
            targets.add(address)
    assert names, 'ELF function symbols are required'
    found = []
    if targets:
        process = subprocess.Popen(['objdump', '-d', str(binary)], stdout=subprocess.PIPE, text=True)
        current = 0
        for line in process.stdout:
            header = re.match(r'([0-9a-f]+) <', line)
            if header:
                current = int(header[1], 16)
            call = re.search(r'\bcall\s+([0-9a-f]+)\b', line)
            if call and int(call[1], 16) in targets:
                found.append(dict(address=line.strip().split(':')[0], caller_address=hex(current),
                                  caller=names.get(current, '')))
        assert process.wait() == 0
    return dict(binary=str(binary.resolve()), weight_symbols=[hex(a) for a in sorted(targets)],
                direct_callsites=len(found), radix_count_callsites=sum('radix_count' in r['caller'] for r in found),
                calls=found)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', nargs='+', default=['validation-window-v3', 'weight-intervals', 'weight-intervals-inline'])
    parser.add_argument('--output', type=Path, default=ROOT/'results/weight-intervals/weight-query-calls.json')
    args = parser.parse_args()
    results = {}
    for label in args.labels:
        binary = ROOT/f'.build/native-{label}/target/release/hf-bpe-native-{label}'
        results[label] = inspect(binary)
        row = results[label]
        print(f"{label}: {len(row['weight_symbols'])} weight symbols, {row['direct_callsites']} direct call sites, "
              f"{row['radix_count_callsites']} in initial grouping")
    args.output.write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
