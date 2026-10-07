#!/usr/bin/env python3
"""Plot paired training ratios from the exported benchmark summary."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--summary', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True, help='output path without suffix')
args = parser.parse_args()
summary = json.loads(args.summary.read_text())
if not summary['performance_conclusion_valid']:
    raise SystemExit('Only plot a complete, correct comparison')
arms = ['bucket', 'blocks', 'stable', 'epoch']
labels = ['New A: canonical buckets', 'New B: independent blocks',
          'Old A: stable key groups', 'Old B: immutable epochs']
colors = ['#167d8d', '#315fc4', '#af6f17', '#bc4545']
plt.rcParams['svg.fonttype'] = 'none'
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
for ax, lang, title in zip(axes, ['zh', 'en'], ['Chinese', 'English']):
    cell = next(c for c in summary['cells'] if c['case'].startswith(lang))
    ratios = cell['metrics']['train_seconds']['paired_ratios']
    for y, arm, color in zip(range(4), arms, colors):
        data = ratios[arm]
        ax.scatter(data['samples'], [y - .08, y, y + .08], color=color, s=25)
        ax.scatter([data['median']], [y], color=color, marker='D', s=45, zorder=3)
        ax.annotate(f"{data['median']:.3f}x", (data['median'], y),
                    xytext=(8, -15), textcoords='offset points', fontsize=9)
    ax.axvline(1, color='#666666', linestyle='--', linewidth=1)
    ax.set_xscale('log')
    ax.set_xlim(.75, 21)
    ax.set_ylim(3.4, -.4)
    ticks = [.8, 1, 2, 4, 8, 16]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_yticks(range(4), labels)
    ax.set_title(title)
    ax.set_xlabel('Candidate / Baseline training time (lower is better)')
    ax.grid(axis='x', alpha=.15)
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('256 MiB whitespace BPE, 50K vocabulary, 4 workers')
fig.text(.5, .015, 'Three paired blocks: circles = individual ratios; diamond = median. Logarithmic axis.',
         ha='center', fontsize=9)
fig.tight_layout(rect=(0, .045, 1, .94))
args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out.with_suffix('.svg'))
svg = args.out.with_suffix('.svg')
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
fig.savefig(args.out.with_suffix('.png'), dpi=160)
