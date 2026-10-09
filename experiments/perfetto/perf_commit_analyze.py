"""Join CPU-stack samples to monotonic per-thread commit component spans."""
import bisect
import collections
import gzip
import json
import re
import subprocess
from pathlib import Path
from commit_deep import extract
from run import ROOT

HEADER = re.compile(r'^(\d+)/(\d+)\s+(\d+)\.(\d+):\s+(\d+)\s+')
FRAME = re.compile(r'^\s+([0-9a-f]+)\s+(.+)\s+\((.*)\)$')


def analyze(run):
    _, buffers = json.loads((run / 'events.json').read_text())
    rounds, phases, routes = extract(run)
    owners = {o['tid']: [] for oo in rounds.values() for o in oo}
    critical = {r: max(oo, key=lambda o: o['dur']) for r, oo in rounds.items()}
    parts = {}
    for tid, events in buffers:
        owners[tid] = sorted((e for e in events if e['name'] == 'commit.owner'), key=lambda e: e['ts'])
        parts[tid] = sorted((e for e in events if e['name'] in ['commit.route', 'commit.group', 'commit.counts', 'commit.completed', 'commit.aggregate', 'commit.encode_publish', 'commit.prefix']), key=lambda e: e['ts'])
    phase_list = sorted(phases.values(), key=lambda e: e['ts'])
    indices = {tid: [e['ts'] for e in ev] for tid, ev in parts.items()}
    owner_indices = {tid: [e['ts'] for e in ev] for tid, ev in owners.items()}
    phase_starts = [e['ts'] for e in phase_list]

    def at(events, starts, timestamp):
        i = bisect.bisect_right(starts, timestamp) - 1
        return events[i] if i >= 0 and timestamp < events[i]['ts'] + events[i]['dur'] else None

    samples = []
    current = None
    script_path = run / 'perf-script.txt'
    script = script_path.open() if script_path.exists() else gzip.open(run / 'perf-script.txt.gz', 'rt')
    with script as f:
        for line in f:
            match = HEADER.match(line)
            if match:
                if current is not None:
                    samples.append(current)
                tid = int(match[2])
                timestamp = int(match[3]) * 1_000_000_000 + int(match[4].ljust(9, '0'))
                phase = at(phase_list, phase_starts, timestamp)
                if phase is None:
                    current = None
                    continue
                owner = at(owners.get(tid, []), owner_indices.get(tid, []), timestamp)
                part = at(parts.get(tid, []), indices.get(tid, []), timestamp)
                current = dict(tid=tid, timestamp=timestamp, round=phase['round'], period=int(match[5]),
                               component=part['name'] if part else ('commit.owner.other' if owner else 'commit.glue'),
                               critical=owner is not None and tid == critical[phase['round']]['tid'], frames=[])
            elif current is not None:
                frame = FRAME.match(line)
                if frame:
                    current['frames'].append((frame[2], frame[3]))
    if current is not None:
        samples.append(current)
    names = sorted({name for s in samples for name, _ in s['frames']})
    demangled = subprocess.run(['c++filt', '-s', 'rust'], input='\n'.join(names) + '\n', stdout=subprocess.PIPE, text=True, check=True).stdout.splitlines()
    translations = dict(zip(names, demangled))
    counters = collections.defaultdict(collections.Counter)
    leaves = collections.defaultdict(collections.Counter)
    critical_counts = collections.Counter()
    critical_categories = collections.defaultdict(collections.Counter)
    owner_categories = collections.Counter()
    critical_owner_categories = collections.Counter()
    for sample in samples:
        stack = [translations.get(n, n) for n, _ in sample['frames']]
        text = '\n'.join(stack)
        if 'rehash_in_place' in text:
            category = 'hash_rehash_in_place'
        elif any(word in text for word in ['prepare_resize', 'resize_inner']):
            category = 'hash_resize'
        elif 'reserve_rehash' in text:
            category = 'hash_rehash_or_resize'
        elif any(word in text for word in ['__libc_free', '_int_free', 'tcache', 'dealloc', 'drop_glue']) and sample['component'] == 'commit.counts':
            category = 'retire_or_free'
        elif 'dary_heap' in text and sample['component'] == 'commit.completed':
            category = 'priority_heap'
        elif any(word in text for word in ['grow_amortized', 'finish_grow', 'realloc', '_int_malloc', '__libc_malloc']):
            category = 'other_allocation_or_growth'
        elif 'clock_gettime' in text or 'bpe_perfetto' in text:
            category = 'trace_instrumentation'
        elif 'hashbrown' in text and sample['component'] == 'commit.completed':
            category = 'hash_insert_without_rehash'
        elif 'hashbrown' in text and 'remove' in text and sample['component'] == 'commit.counts':
            category = 'hash_remove_without_free'
        elif 'hashbrown' in text and sample['component'] == 'commit.counts':
            category = 'hash_lookup_update'
        elif 'rayon' in text and sample['component'] == 'commit.glue':
            category = 'rayon_glue'
        else:
            category = 'remaining_inline_or_other'
        counters[sample['component']][category] += 1
        if sample['critical']:
            critical_counts[sample['component']] += 1
            critical_categories[sample['component']][category] += 1
            critical_owner_categories[category] += 1
        if sample['component'] not in ['commit.route', 'commit.glue']:
            owner_categories[category] += 1
        if stack:
            leaves[sample['component']][stack[0]] += 1
    out = dict(run=run.name, samples=len(samples), frequency_hz=499,
               components={k: dict(v) for k, v in counters.items()},
               critical_owner_samples=dict(critical_counts),
               critical_owner_categories={k: dict(v) for k, v in critical_categories.items()},
               owner_categories=dict(owner_categories),
               sampled_inequality_excess_by_category={k: 4 * critical_owner_categories[k] - v for k, v in owner_categories.items()},
               most_common_leaf_symbols={k: v.most_common(8) for k, v in leaves.items()})
    (run / 'perf-commit-analysis.json').write_text(json.dumps(out, indent=2))
    return out


if __name__ == '__main__':
    for run in sorted((ROOT / 'runs').glob('*w4-perf-r*')):
        if ((run / 'perf-script.txt').exists() or (run / 'perf-script.txt.gz').exists()) and (run / 'result.json').exists():
            result = analyze(run)
            print(result['run'], result['samples'], result['components'])
