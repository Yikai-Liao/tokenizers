#!/usr/bin/env python3
"""Serial, shuffled, separate-process comparisons with model digest gates."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def capture(corpus, binaries, cases):
    source_files = subprocess.check_output(['rg', '--files', 'tokenizers', '-g', '*.rs', '-g', 'Cargo.toml'], cwd=REPO, text=True).splitlines()
    source_files += ['benchmarks/hf-bpe/src/main.rs', 'benchmarks/hf-bpe/Cargo.toml', 'benchmarks/hf-bpe/Cargo.lock']
    return {
        'date_utc': datetime.now(timezone.utc).isoformat(),
        'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'rustc': subprocess.check_output([str(Path.home() / '.cargo/bin/rustc'), '--version'], text=True).strip(),
        'uname': platform.uname()._asdict(), 'visible_cpus': os.cpu_count(),
        'cpu_model': next((s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')), None),
        'cpu_max': Path('/sys/fs/cgroup/cpu.max').read_text().strip() if Path('/sys/fs/cgroup/cpu.max').exists() else None,
        'memory_max': Path('/sys/fs/cgroup/memory.max').read_text().strip() if Path('/sys/fs/cgroup/memory.max').exists() else None,
        'env': {'TOKENIZERS_PARALLELISM': 'false', 'RAYON_NUM_THREADS': '1'},
        'training_workers': {engine: (4 if engine in ['parallel4', 'dict16', 'narrow16'] else 1)
                             for engine in binaries},
        'binaries': {label: {'path': str(path), 'sha256': sha256(path)} for label, path in binaries.items()},
        'sources': {s: sha256(REPO / s) for s in sorted(source_files)},
        'corpora': {f'{lang}-{size}m': {'path': str(corpus / f'{lang}-{size}m.txt'), 'sha256': sha256(corpus / f'{lang}-{size}m.txt')}
                    for lang, size in sorted({(lang, size) for lang, size, _ in cases})},
        'pr_head': '6ac0de5359d9e0e1ed0608422575a360ef91b908',
        'profiling_script_sha256': sha256(ROOT / 'build_profiled.py'),
        'profiled_trainers': {label: sha256(ROOT / f'.build/profiled-{label}/source/tokenizers/tk-train/src/trainers/bpe/mod.rs')
                              for label in ['current', 'pr'] if any(p.name == f'hf-bpe-profiled-{label}' for p in binaries.values())},
        'profiled_locks': {label: sha256(ROOT / f'.build/profiled-{label}/runner/Cargo.lock')
                           for label in ['current', 'pr'] if any(p.name == f'hf-bpe-profiled-{label}' for p in binaries.values())},
    }


def summarize(records):
    if any(r['engine'] == 'parallel1' for r in records):
        keys = ['initial_corpus_bytes', 'initial_pair_table_bytes', 'initial_posting_bytes',
                'initial_block_table_bytes', 'initial_directory_bytes', 'initial_heap_bytes']
        lines = ['# Parallel BPE bounded comparison', '',
                 'One process per cell; timing is preliminary. Initial layout is the source-layout estimate immediately after count/filter/heap, excluding HF input/vocabulary, allocator overhead and thread stacks. PR has no comparable initialization counter.', '',
                 '| Language | Split | Engine | Train s | Init s | Merge s | Initial layout MiB |',
                 '|---|---|---|---:|---:|---:|---:|']
        for r in sorted(records,key=lambda r:(r['language'],r['engine'])):
            st = r['indexed_stats']
            memory = f'{sum(st[k] for k in keys) / (1<<20):.2f}' if st else '—'
            lines.append(f'| {r["language"]} | {r["split"]} | {r["engine"]} | {r["train_ms"]/1000:.3f} | '
                         f'{r["initialize_ms"]/1000:.3f} | {r["merge_ms"]/1000:.3f} | {memory} |')
        return '\n'.join(lines)+'\n'
    groups = defaultdict(list)
    for row in records:
        groups[(row['language'], row['size_mib'], row['split'], row['engine'])].append(row)
    lines = ['# HF BPE sequential prototype benchmark', '',
             'Times are medians of the recorded separate processes in one shuffled serial run. Train includes initialization, indexing, merges and output. Total is feed + train.', '',
             '| Language | MiB | Split | Engine | n | Train s | Merge s | Total s | RSS MiB | Model SHA-256 |',
             '|---|---:|---|---|---:|---:|---:|---:|---:|---|']
    for (language, size, split, engine), rows in sorted(groups.items()):
        median = lambda key: statistics.median(r[key] for r in rows)
        lines.append(f'| {language} | {size} | {split} | {engine} | {len(rows)} | {median("train_ms") / 1000:.3f} | '
                     f'{median("merge_ms") / 1000:.3f} | {median("elapsed_ms") / 1000:.3f} | {median("maxrss_kib") / 1024:.1f} | `{rows[0]["model_sha256"]}` |')
    reference_cases = [case for case in sorted({k[:3] for k in groups})
                       if all((*case, engine) in groups for engine in ['reference', 'indexed', 'pr_head'])]
    if reference_cases:
        lines += ['', '| Language | MiB | Split | Reference/indexed train | PR/indexed train | Reference/indexed total | PR/indexed total |',
                  '|---|---:|---|---:|---:|---:|---:|']
    for case in reference_cases:
        def value(engine, metric):
            return statistics.median(r[metric] for r in groups[(*case, engine)])
        if all((*case, engine) in groups for engine in ['reference', 'indexed', 'pr_head']):
            lines.append(f'| {case[0]} | {case[1]} | {case[2]} | '
                         f'{value("reference", "train_ms") / value("indexed", "train_ms"):.2f}× | '
                         f'{value("pr_head", "train_ms") / value("indexed", "train_ms"):.2f}× | '
                         f'{value("reference", "elapsed_ms") / value("indexed", "elapsed_ms"):.2f}× | '
                         f'{value("pr_head", "elapsed_ms") / value("indexed", "elapsed_ms"):.2f}× |')
    if any(k[3] == 'fused' for k in groups):
        lines += ['', '| Language | MiB | Split | PR/indexed train | PR/fused train | PR/fused total |',
                  '|---|---:|---|---:|---:|---:|']
        for case in sorted({k[:3] for k in groups}):
            if all((*case, e) in groups for e in ['indexed', 'fused', 'pr_head']):
                value = lambda e, m: statistics.median(r[m] for r in groups[(*case, e)])
                lines.append(f'| {case[0]} | {case[1]} | {case[2]} | '
                             f'{value("pr_head", "train_ms") / value("indexed", "train_ms"):.2f}× | '
                             f'{value("pr_head", "train_ms") / value("fused", "train_ms"):.2f}× | '
                             f'{value("pr_head", "elapsed_ms") / value("fused", "elapsed_ms"):.2f}× |')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus', type=Path)
    parser.add_argument('--profile', choices=['parallel', 'focused', 'smoke', 'quick', 'representative'], default='smoke')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/runs.jsonl')
    parser.add_argument('--pr-binary', type=Path, default=ROOT / 'target/release/hf-bpe-profiled-pr')
    parser.add_argument('--binary', type=Path, default=ROOT / 'target/release/hf-bpe-profiled-current')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    corpus = args.corpus.resolve()
    binary = args.binary.resolve()
    binaries = {'reference': binary, 'indexed': binary, 'pr_head': args.pr_binary.resolve()}
    cases = list(itertools.product(['en', 'zh'], [1, 4], ['none', 'whitespace_split', 'bytelevel']))
    if args.profile == 'smoke':
        cases = [('en', 1, 'none'), ('en', 1, 'whitespace_split'),
                 ('zh', 1, 'whitespace_split'), ('zh', 1, 'bytelevel')]
    if args.profile == 'representative':
        cases += [(lang, 16, 'whitespace_split') for lang in ['en', 'zh']]
    if args.profile == 'focused':
        cases = [('zh', 4, 'none'), ('en', 4, 'whitespace_split')]
        binaries = {'indexed': binary, 'fused': binary, 'pr_head': args.pr_binary.resolve()}
    if args.profile == 'parallel':
        cases = [('zh', 4, 'none'), ('en', 4, 'whitespace_split')]
        binaries = {engine: binary for engine in ['indexed', 'parallel1', 'parallel4', 'dict16', 'narrow16']}
        binaries['pr_head'] = args.pr_binary.resolve()
    tasks = [(lang, size, split, engine, repeat) for lang, size, split in cases
             for engine in binaries for repeat in range(args.repeats)]
    print(f'planned: {len(cases)} cases × {len(binaries)} engines × {args.repeats} repeats = {len(tasks)} training calls', flush=True)
    random.Random(2348).shuffle(tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f'{args.output} already exists; use a new output path')
    environment = capture(corpus, binaries, cases)
    environment_path = args.output.with_suffix('.environment.json')
    environment_path.write_text(json.dumps(environment, ensure_ascii=False, indent=2) + '\n')
    provenance = sha256(environment_path)
    env = dict(os.environ, TOKENIZERS_PARALLELISM='false', RAYON_NUM_THREADS='1')
    records = []
    signatures = {}
    begin = time.monotonic()
    with args.output.open('w') as out:
        for number, (language, size, split, engine, repeat) in enumerate(tasks, 1):
            vocab = 8000 if size <= 4 else 16000
            backend = 'reference' if engine == 'pr_head' else engine
            command = [str(binaries[engine]), str(corpus / f'{language}-{size}m.txt'), split, backend, str(vocab), '2']
            process = subprocess.run(command, env=env, text=True, capture_output=True, timeout=240, check=True)
            row = json.loads(process.stdout.strip())
            stages = {item['bench_stage']: item['ms'] for line in process.stderr.splitlines()
                      if line.startswith('{') and 'bench_stage' in (item := json.loads(line))}
            row['stage_ms'] = stages
            row['merge_ms'] = row['indexed_stats']['merge_ms'] if row['indexed_stats'] else stages.get('merges')
            row['initialize_ms'] = row['indexed_stats']['initialize_ms'] if row['indexed_stats'] else sum(stages.get(stage, 0) for stage in ['special_tokens', 'alphabet', 'tokenize_words', 'count_pairs'])
            row.update(language=language, size_mib=size, engine=engine, repeat=repeat,
                       provenance_sha256=provenance, command=command, stderr=process.stderr)
            signature = (row['model_sha256'], row['actual_vocab'], row['actual_merges'], row['unique_words'])
            case = (language, size, split)
            if case in signatures and signatures[case] != signature:
                raise SystemExit(f'model mismatch {case}: {signature} != {signatures[case]}')
            signatures[case] = signature
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
            out.flush()
            records.append(row)
            if args.profile == 'parallel':
                stats = row['indexed_stats']
                keys = ['initial_corpus_bytes', 'initial_pair_table_bytes', 'initial_posting_bytes',
                        'initial_block_table_bytes', 'initial_directory_bytes', 'initial_heap_bytes']
                space = f'init-layout={sum(stats[k] for k in keys)/(1<<20):.2f}MiB' if stats else 'init-layout=unavailable'
            else:
                space = f'RSS={row["maxrss_kib"] / 1024:.1f}MiB'
            print(f'{number}/{len(tasks)} {language} {size} MiB {split} {engine}: train={row["train_ms"] / 1000:.3f}s '
                  f'total={row["elapsed_ms"] / 1000:.3f}s {space} elapsed={time.monotonic()-begin:.1f}s', flush=True)
    args.output.with_suffix('.summary.md').write_text(summarize(records))
    print(f'all {len(records)} models matched; wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
