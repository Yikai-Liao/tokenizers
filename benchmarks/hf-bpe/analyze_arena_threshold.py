#!/usr/bin/env python3
"""Verify saved cutoff runs and render standalone figures (matplotlib needed)."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/arena-threshold'
MIB = 2**20
GIB = 2**30


def load(path):
    return json.loads(path.read_text())


def main():
    groups = ['zh512m', 'en16m-none', 'en16m-whitespace_split']
    rows = []
    signatures = {}
    binaries = set()
    for group in groups:
        for cutoff in ['0', '32', '256', 'all']:
            name = f'{group}-t{cutoff}'
            raw = load(OUT / f'{name}.jsonl')
            summary = load(OUT / f'{name}.summary.json')
            env = load(OUT / f'{name}.environment.json')
            binaries.add(env['binary_sha256'])
            trace = summary['phase']['trace']
            totals = summary['totals']
            stats = raw['indexed_stats']
            signature = dict(model=raw['model_sha256'], input=env['input_sha256'],
                             merges=raw['actual_merges'], vocab=raw['actual_vocab'],
                             unique_words=raw['unique_words'],
                             work={k: stats[k] for k in [
                                 'initial_symbols', 'initial_edges', 'initial_pairs',
                                 'initial_slots', 'initial_corpus_bytes', 'initial_posting_bytes',
                                 'batch_rounds', 'fused_batches', 'posting_visits', 'pruned_pairs']},
                             inventory={k: v for k, v in summary['inventory'].items()
                                        if k != 'inventory_ms'},
                             total_allocations=totals['allocations'] + totals['system_allocations'],
                             total_payload=totals['requested_bytes'] + totals['system_requested_bytes'])
            if group in signatures:
                assert signature == signatures[group], (name, 'work/model differs')
            else:
                signatures[group] = signature
            assert summary['allocator_origin_checks'] == 'PASS'
            assert totals['grows'] == 0
            assert raw['memory']['sampled_peak_process_swap_bytes'] == 0
            assert raw['memory']['minimum_available_bytes'] > GIB
            peak = max(raw['maxrss_kib'] * 1024,
                       *(x['memory']['VmHWM'] for x in trace))
            rows.append(dict(case=name, group=group, cutoff=cutoff,
                             initialize_s=raw['initialize_ms']/1000,
                             merge_s=raw['merge_ms']/1000,
                             train_s=raw['train_ms']/1000,
                             elapsed_s=raw['elapsed_ms']/1000,
                             peak_bytes=peak,
                             initialize_hwm=trace[0]['memory']['VmHWM'],
                             merge_terminal_rss=trace[-1]['memory']['VmRSS'],
                             phase_hwm_unchanged=all(x['memory']['VmHWM'] == trace[0]['memory']['VmHWM']
                                                     for x in trace),
                             arena_release_ms=summary['arena']['release_ms'],
                             inventory_ms=summary['inventory']['inventory_ms'],
                             minimum_available_bytes=raw['memory']['minimum_available_bytes'],
                             process_swap_bytes=raw['memory']['sampled_peak_process_swap_bytes'],
                             totals=totals))
    assert len(binaries) == 1
    for row in rows:
        base = next(x for x in rows if x['group'] == row['group'] and x['cutoff'] == '0')
        row['train_change_percent'] = 100*(row['train_s']/base['train_s']-1)
        row['peak_change_bytes'] = row['peak_bytes']-base['peak_bytes']
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for group in groups for t in ['0','32','256','all']
               for p in [OUT/f'{group}-t{t}.jsonl', OUT/f'{group}-t{t}.summary.json',
                         OUT/f'{group}-t{t}.environment.json']}
    (OUT/'bundle.summary.json').write_text(json.dumps(dict(
        binary_sha256=next(iter(binaries)), checks='PASS', count=12,
        scope='one run per configuration; diagnostic counters; no stable ranking',
        signatures=signatures, rows=rows, sources=sources), indent=2)+'\n')
    for row in rows:
        print(row['case'], f"train={row['train_s']:.6f}s peak={row['peak_bytes']/MIB:.3f}MiB",
              f"train_delta={row['train_change_percent']:+.2f}% peak_delta={row['peak_change_bytes']/MIB:+.3f}MiB")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#525252', '#3b82f6', '#14b8a6', '#dc2626']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), layout='constrained')
    chinese = [r for r in rows if r['group'] == 'zh512m']
    axes[0].bar(['heap', '32 B', '256 B', 'all'], [r['train_s'] for r in chinese], color=colors)
    axes[0].set(ylabel='Full train seconds', title='zh 512 MiB, 29,243 rules (one run each)')
    for i, r in enumerate(chinese):
        axes[0].text(i, r['train_s']+.15, f"{r['train_s']:.2f}", ha='center')
    for r, color in zip(chinese, colors):
        trace = load(OUT/f"{r['case']}.summary.json")['phase']['trace']
        axes[1].plot([x['rules'] for x in trace], [x['memory']['VmRSS']/GIB for x in trace],
                     label=r['cutoff'], color=color)
    axes[1].axhline(chinese[0]['initialize_hwm']/GIB, ls='--', color='black', label='initialization HWM')
    axes[1].set(xlabel='Merge rules completed', ylabel='Resident GiB', title='Merge fits below prior process peak')
    axes[1].legend(fontsize=8)
    for group, marker in [('en16m-none', 'o'), ('en16m-whitespace_split', 's')]:
        selected = [r for r in rows if r['group'] == group]
        axes[2].plot(range(4), [r['peak_change_bytes']/MIB for r in selected], marker=marker,
                     label=group.removeprefix('en16m-'))
    axes[2].axhline(0, color='black', linewidth=.8)
    axes[2].set(xticks=range(4), xticklabels=['heap', '32 B', '256 B', 'all'],
                ylabel='Peak change from heap, MiB', title='en 16 MiB: full arena can increase peak')
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    for ext in ['png','svg']:
        fig.savefig(OUT/f'arena-threshold-phase.{ext}', dpi=160)
    plt.close(fig)

    fits = load(ROOT/'posting-threshold-empirical-fits.json')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for ax, metric in zip(axes.flat, ['p50','p90','p99','max']):
        for language, color in zip(['en','de','zh','ja'], ['#3b82f6','#14b8a6','#dc2626','#9333ea']):
            cases = sorted([c for c in fits['cases'] if c['language'] == language
                            and c['split'] == 'none' and c['rules'] == 16000],
                           key=lambda c:c['x']['physical_edges'])
            narrow = [c for c in cases if c['size_mib'] <= 32]
            ax.loglog([c['x']['physical_edges'] for c in narrow], [c['y'][metric] for c in narrow],
                      'o-', color=color, label=language)
            if language == 'zh':
                held = next(c for c in cases if c['size_mib'] == 512)
                f = next(f['fit'] for f in fits['local_fits'] if f['language'] == 'zh'
                         and f['split'] == 'none' and f['rules'] == 16000 and f['metric'] == metric
                         and f['predictor'] == 'physical_edges' and f['model'] == 'power')
                x = held['x']['physical_edges']; pred = f['a']*(x/f['x_ref'])**f['gamma']
                ax.scatter([x],[held['y'][metric]], color=color, marker='s', s=50, label='zh 512 MiB actual')
                ax.scatter([x],[pred], color=color, marker='x', s=65, label='zh local-power extrapolation')
        ax.set(title=f'{metric}: surviving posting types, 16k rules', xlabel='Initial unique physical edges',
               ylabel='Historical physical positions')
        ax.grid(alpha=.2, which='both')
        ax.legend(fontsize=7)
    for ext in ['png','svg']:
        fig.savefig(OUT/f'posting-length-growth.{ext}', dpi=160)
    plt.close(fig)
    # Matplotlib SVG paths contain harmless trailing spaces; normalize exports.
    for path in OUT.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


if __name__ == '__main__':
    main()
