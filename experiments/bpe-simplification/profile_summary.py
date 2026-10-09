"""Attribute weighted cycle samples to the nearest recognizable engine phase."""
from collections import Counter
import json
from pathlib import Path
import re
import subprocess

ROOT = Path('/root/code/tokenizers-simplification-results/profiles')


def phase(frame):
    for name, markers in [
        ('merge_prepare', ['Batch>::prepare', 'Source>::heads', 'Neighbors>::',
                           'engine::merge::prepare', 'MergeScratch>']),
        ('apply', ['Prepared>::apply', 'PreparedMerges>::apply', 'Corpus>::apply']),
        ('owner_commit', ['PairIndex>::commit', 'PairShard>::reduce_', 'PairShard>::apply_',
                          'PairShard>::commit']),
        ('initial_pairs', ['PairIndex>::build', 'engine::initial_pairs']),
        ('materialize', ['CorpusPlan>::materialize', 'CorpusPlan>::fill_tokens']),
        ('vocabulary', ['engine::vocabulary']),
        ('plan', ['CorpusPlan>::build', 'CorpusPlan>::new']),
    ]:
        if any(marker in frame for marker in markers):
            return name
    return None


def summarize(label):
    root = ROOT / ('zh-core-4-' + label)
    raw = (root / 'stacks.raw').read_text()
    demangled = subprocess.run(['c++filt', '-s', 'rust'], input=raw, text=True,
                               capture_output=True, check=True).stdout
    demangled = re.sub(r'\[[0-9a-f]{6,}\]', '', demangled)
    (root / 'stacks.demangled.txt').write_text(demangled)
    cycles = Counter()
    samples = Counter()
    codec = Counter()
    symbol_cycles = Counter()
    bounds = {}
    for block in demangled.split('\n\n'):
        lines = block.splitlines()
        if not lines:
            continue
        head = re.search(r'(\d+\.\d+):\s+(\d+)\s+cycles:', lines[0])
        if not head:
            continue
        stamp, weight = float(head[1]), int(head[2])
        frames = lines[1:]
        selected = next((p for f in frames if (p := phase(f))), None)
        if selected is None:
            selected = 'engine_other' if any('trainers::bpe::engine' in f for f in frames) else 'outside_or_unknown'
        cycles[selected] += weight
        samples[selected] += 1
        bounds.setdefault(selected, [stamp, stamp])
        bounds[selected][1] = stamp
        symbol = frames[0].strip() if frames else 'unknown'
        symbol_cycles[symbol] += weight
        if 'Positions>::push' in symbol or 'positions::Cursor' in symbol:
            codec[selected] += weight
    total = sum(cycles.values())
    classified = total - cycles['outside_or_unknown']
    return dict(label=label, weighted_cycles=total, classified_engine_cycles=classified,
                phase_cycles=dict(cycles), phase_samples=dict(samples),
                phase_pct_of_classified={p: 100*w/classified for p, w in cycles.items() if p != 'outside_or_unknown'},
                codec_self_pct_of_classified_by_phase={p: 100*w/classified for p, w in codec.items()},
                phase_first_last_sample=bounds,
                note='Cycle-sample attribution by nearest matching stack frame; engine_other and outside/unknown retained. Not phase wall time. No inclusive sample double counting.')


if __name__ == '__main__':
    result = [summarize(label) for label in ['baseline', 'head-ring']]
    (ROOT / 'phase-summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
