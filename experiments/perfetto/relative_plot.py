"""Render dimensionless comparisons; all job shares are normalized per batch."""
import json
import statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/root/code/tokenizers-perfetto-results')
data = json.loads((ROOT / 'relative-metrics.json').read_text())
labels = ['EN / WS', 'ZH / WS', 'EN / BL', 'ZH / BL']
x = list(range(len(data)))
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

ax = axs[0, 0]
for shift, key, label, color in [
    (-.18, 'raw', 'Candidate positions', '#173f5f'),
    (.18, 'matched', 'Matched positions', '#3caea3'),
]:
    vals = [r['detailed_repeats'][0][f'median_{key}_heaviest_to_four_worker_ideal'] for r in data]
    ax.bar([i + shift for i in x], vals, width=.34, label=label, color=color)
ax.axhline(1, linestyle='--', color='#777', linewidth=1)
ax.set_ylabel('4 x largest job / batch total (median)')
ax.set_ylim(0, 1.65)
ax.set_title('Prepare: heaviest job relative to ideal worker share')
ax.legend(fontsize=8)

ax = axs[0, 1]
for shift, key, label, color in [
    (-.18, 'raw', 'Candidate share predicts time', '#173f5f'),
    (.18, 'matched', 'Matched share predicts time', '#3caea3'),
]:
    reps = [[100 * d[f'{key}_time_share_distance'] for d in r['detailed_repeats']] for r in data]
    vals = [statistics.mean(v) for v in reps]
    errors = [[m - min(v) for m, v in zip(vals, reps)], [max(v) - m for m, v in zip(vals, reps)]]
    ax.bar([i + shift for i in x], vals, width=.34, label=label, color=color, yerr=errors, capsize=3)
ax.set_ylabel('Within-batch share distance (%)')
ax.set_ylim(0, 24)
ax.set_title('Prepare: workload prediction error (lower is better)')
ax.legend(fontsize=8)

for ax, phase in zip(axs[1], ['prepare', 'commit']):
    bottom = [0.] * len(data)
    for key, label, color in [
        ('occupancy', 'Jobs occupied', '#3caea3'),
        ('boundary', 'Phase boundaries', '#173f5f'),
        ('insufficient', 'Too few unfinished jobs', '#f6d55c'),
        ('ready', 'Dispatch gap', '#ed553b'),
    ]:
        vals = []
        for r in data:
            p = r['phases'][phase]
            if key == 'occupancy':
                v = p['occupancy']
            else:
                v = (1 - p['occupancy']) * p[f'{key}_share_of_loss']
            vals.append(v * 100)
        ax.bar(x, vals, bottom=bottom, label=label, color=color)
        bottom = [a + b for a, b in zip(bottom, vals)]
    ax.set_ylim(0, 100)
    ax.set_ylabel('Share of total four-worker phase capacity (%)')
    ax.set_title(f'{phase.capitalize()}: same capacity denominator')
axs[1, 0].legend(fontsize=8, loc='upper left', bbox_to_anchor=(0, -.17), ncol=2)
for ax in axs.flat:
    ax.set_xticks(x, labels)
fig.suptitle('Relative parallelism metrics: 4 workers, all batches, 3 repeats')
fig.tight_layout(rect=(0, .05, 1, .96))
fig.savefig(ROOT / 'relative-parallelism.png', dpi=180)
