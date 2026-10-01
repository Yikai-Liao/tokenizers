#!/usr/bin/env python3
"""Check saved initialization experiments; archive exact overlay differences."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
WORKTREE = Path('/root/code/tokenizers-worktrees/initial-owner-waves')
GROUPS = ['initial-owner-waves', 'block-radix', 'block-radix-stages',
          'frontier', 'frontier-sort-cost', 'block-count', 'block-count-stable',
          'block-summary-waves']
GIB = 2**30


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows, provenance, signatures = [], {}, {}
    source_cache = {}
    for group in GROUPS:
        directory = ROOT / 'results' / group
        paths = [p for p in sorted(directory.glob('*.jsonl')) if not p.name.startswith('smoke')]
        assert paths, group
        binaries, sources, commits = set(), [], set()
        for path in paths:
            row = read(path)
            env = read(path.with_suffix('.environment.json'))
            summary = read(path.with_suffix('.summary.json'))
            assert env['worktree_clean']
            binaries.add(env['binary_sha256'])
            commits.add(env['worktree_commit'])
            sources.append(env['built_source_sha256'])
            assert row['memory']['sampled_peak_process_swap_bytes'] == 0
            assert row['memory']['minimum_available_bytes'] > GIB
            stats = row['indexed_stats']
            assert stats['workers'] == stats['initialization_workers'] == 4
            assert stats['atomic_corpus']
            sig = dict(model=row['model_sha256'], input=env['input_sha256'],
                       rules=row['actual_merges'], vocab=row['actual_vocab'], words=row['unique_words'],
                       work={k: stats[k] for k in ['initial_symbols', 'initial_slots', 'initial_edges',
                            'initial_pairs', 'batch_rounds', 'posting_visits', 'pruned_pairs']})
            workload = f"{sig['input']}:{row['split']}:{row['vocab_size']}"
            if workload in signatures:
                assert sig == signatures[workload], path
            else:
                signatures[workload] = sig
            trace = summary.get('phase', {}).get('trace', [])
            peak = max([row['maxrss_kib'] * 1024] + [t['memory']['VmHWM'] for t in trace])
            if 'allocator_origin_checks' in summary:
                assert summary['allocator_origin_checks'] == 'PASS'
                assert summary['totals']['grows'] == 0
                assert summary['inventory']['stored_positions'] == 207224101
            probe = summary.get('probe')
            if probe:
                assert probe['local_offset_bytes'] == 4
                assert probe['physical_edges'] == stats['initial_edges']
                assert probe['global_frequency_floor_after_reduction']
            rows.append(dict(group=group, case=path.stem, initialize_s=row['initialize_ms']/1000,
                merge_s=row['merge_ms']/1000, train_s=row['train_ms']/1000, elapsed_s=row['elapsed_ms']/1000,
                peak_bytes=peak, initialize_hwm=trace[0]['memory']['VmHWM'] if trace else None,
                radix_scratch_bytes=stats.get('initial_radix_scratch_bytes', 0),
                route_compact_ms=stats.get('initial_route_compact_ms', 0),
                heap_capacity_bytes=summary.get('inventory', {}).get('owner_heap_capacity_bytes'),
                summary_waves=stats.get('initial_summary_waves', 0),
                summary_capacity_sum=stats.get('initial_summary_buffer_bytes', 0),
                summary_capacity_peak=stats.get('peak_initial_summary_buffer_bytes', 0),
                probe=probe, input_sha256=sig['input'], model_sha256=sig['model']))
        assert len(binaries) == len(commits) == 1, group
        assert all(s == sources[0] for s in sources), group
        built = Path(env['built_source_root']).parents[1]
        for relative, expected in sources[0].items():
            assert sha(built / relative) == expected, (group, relative)
        assert sha(Path(env['binary_path'])) == next(iter(binaries))
        # Each overlay has the source worktree as its reproducible base. Keep only
        # actual differences, including runtime switches and diagnostics.
        patch = []
        commit = next(iter(commits))
        for relative in sorted(sources[0]):
            base_path = relative.removeprefix('source/') if relative.startswith('source/') else (
                'benchmarks/hf-bpe/' + relative.removeprefix('runner/'))
            base_hash = env['worktree_source_sha256'].get(base_path)
            if base_hash == sources[0][relative]:
                continue
            cache_key = (commit, base_path)
            if cache_key not in source_cache:
                result = subprocess.run(['git', 'show', f'{commit}:{base_path}'], cwd=WORKTREE,
                                        capture_output=True, check=False)
                source_cache[cache_key] = result.stdout.decode() if result.returncode == 0 else ''
            before = source_cache[cache_key]
            after = (built / relative).read_text()
            if before != after:
                patch.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                    fromfile='a/'+base_path if before else '/dev/null', tofile='b/'+base_path))
        overlay_patch = directory / 'instrumentation.patch'
        overlay_patch.write_text(''.join(patch))
        label = built.name.removeprefix('native-j-')
        if label == 'owner-waves':
            label = 'initial-owner-waves'
        log = ROOT / '.build' / (label + '.build.log')
        assert log.exists(), log
        shutil.copyfile(log, directory / 'build.log')
        provenance[group] = dict(source_commit=commit, binary_sha256=next(iter(binaries)),
            overlay_patch_sha256=sha(overlay_patch), build_log_sha256=sha(directory/'build.log'),
            source_file_count=len(sources[0]), source_hashes=sources[0], formal_calls=len(paths))
    # Only the lexical-order proxy promises identical physical address blocks.
    stable = [r for r in rows if r['group'] == 'block-count-stable']
    for language in ['en', 'zh']:
        for split in ['none', 'whitespace_split']:
            pair = [r for r in stable if r['case'].startswith(f'{language}32m-{split}-')]
            assert len(pair) == 2
            probes = [{k:v for k,v in r['probe'].items() if k not in
                ['mode', 'count_keys', 'count_capacity_bytes_sum_estimate']} for r in pair]
            assert probes[0] == probes[1], (language, split)
    heap_rows = [r for r in rows if r['group'] == 'frontier']
    for r in heap_rows:
        expected = 172641088 if '-auto-' in r['case'] else 345282176
        assert r['heap_capacity_bytes'] == expected
    wave_rows = [r for r in rows if r['group'] == 'block-summary-waves']
    assert len(wave_rows) == 2
    assert wave_rows[0]['probe'] == wave_rows[1]['probe']
    assert {r['summary_waves'] for r in wave_rows} == {1, 4}
    assert {r['summary_capacity_peak'] for r in wave_rows} == {50333208, 16777696}
    output = ROOT/'results/initialization-summary'
    output.mkdir(exist_ok=True)
    files = {str(p.relative_to(ROOT)):sha(p) for group in GROUPS
             for p in (ROOT/'results'/group).iterdir() if p.suffix in ['.json', '.jsonl']}
    (output/'bundle.summary.json').write_text(json.dumps(dict(checks='PASS', rows=rows,
        provenance=provenance, signatures=signatures, saved_data_hashes=files,
        scope='diagnostic runs; n=1 except two frontier configurations n=2; block proxy, not >2^32 allocation'),
        indent=2)+'\n')
    print(f'PASS: {len(rows)} formal calls, {len(signatures)} input/model signatures; exact source hashes')
    for r in stable:
        print(r['case'], f"init={r['initialize_s']:.3f}s train={r['train_s']:.3f}s peak={r['peak_bytes']/2**20:.2f}MiB")
    if '--plot' in sys.argv:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        selected = [next(r for r in rows if r['group']==g and r['case']==c) for g,c in [
            ('initial-owner-waves', 'zh512m-w4-all'),
            ('initial-owner-waves', 'zh512m-w2-all'),
            ('block-radix-stages', 'zh512m-block-s4-i2-t256'),
            ('frontier', 'zh512m-direct-auto-t256')]]
        labels = ['Classic, 4 install', 'Classic, 2 install', 'Block, sort4/install2', 'Direct + packed']
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout='constrained')
        axes[0].bar(labels, [r['peak_bytes']/GIB for r in selected], color=['#525252','#64748b','#0d9488','#2563eb'])
        axes[0].set(ylabel='Full process peak GiB', title='512 MiB zh: measured configurations')
        axes[0].tick_params(axis='x', labelrotation=18)
        for i,r in enumerate(selected):
            axes[0].text(i,r['peak_bytes']/GIB+.04,f"{r['peak_bytes']/GIB:.3f}",ha='center')
        axes[1].bar(labels, [r['radix_scratch_bytes']/2**20 for r in selected], color=['#525252','#64748b','#0d9488','#2563eb'])
        axes[1].set(ylabel='Concurrent sort scratch MiB', title='Scratch falls; records/corpus/postings remain')
        axes[1].tick_params(axis='x', labelrotation=18)
        for ax in axes:
            ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
        for ext in ['png','svg']:
            fig.savefig(output/f'initialization-memory.{ext}',dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout='constrained')
        labels, capacity, timing = [], [], []
        for language in ['en','zh']:
            for split in ['none','whitespace_split']:
                a=next(r for r in stable if r['case']==f'{language}32m-{split}-legacy-b24')
                b=next(r for r in stable if r['case']==f'{language}32m-{split}-sparse-b24')
                labels.append(f'{language} '+('none' if split=='none' else 'WS'))
                capacity.append(100*b['probe']['count_capacity_bytes_sum_estimate']/a['probe']['count_capacity_bytes_sum_estimate'])
                timing.append(100*(b['initialize_s']/a['initialize_s']-1))
        axes[0].bar(labels,capacity,color='#0d9488');axes[0].set(ylabel='Sparse / legacy capacity, %',title='Temporary frequency maps (sum, not peak RSS)')
        axes[1].bar(labels,timing,color=['#0d9488' if v<0 else '#dc2626' for v in timing])
        axes[1].axhline(0,color='black',lw=.8);axes[1].set(ylabel='Initialization time change, %',title='Fixed lexical order, 32 MiB; one call each')
        for ax in axes:
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        for ext in ['png','svg']:
            fig.savefig(output/f'block-count-capacity-time.{ext}',dpi=160)
        plt.close(fig)
        for p in output.glob('*.svg'):
            p.write_text('\n'.join(s.rstrip() for s in p.read_text().splitlines())+'\n')


if __name__ == '__main__':
    main()
