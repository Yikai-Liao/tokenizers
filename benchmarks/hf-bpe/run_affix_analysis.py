#!/usr/bin/env python3
"""Run a monitored native affix case against an immutable build manifest."""
import argparse
import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXTENDED_DIAGNOSTIC_FIELDS = {
    'alias_guarded', 'alias_fallback', 'corpus_slot_bytes',
    'speculative_selected_merges', 'speculative_applied_merges',
    'speculative_initialize_ms', 'speculative_merge_ms', 'speculative_total_ms',
    'speculative_posting_allocations',
}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def system():
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith(('MemAvailable:', 'SwapFree:')):
            values[line.split(':')[0]] = int(line.split()[1]) * 1024
    vm = {}
    for line in Path('/proc/vmstat').read_text().splitlines():
        key, value = line.split()
        if key in ('pswpin', 'pswpout'):
            vm[key] = int(value)
    return {**values, **vm}

def host():
    cpu = next(line for line in Path('/proc/stat').read_text().splitlines() if line.startswith('cpu '))
    return {'cpu_ticks': [int(v) for v in cpu.split()[1:]], 'loadavg': Path('/proc/loadavg').read_text().split()[:3]}

def stop_child(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

def allocation_mismatches(counters, prefix):
    if not isinstance(counters, dict):
        return {prefix: ('missing', 'allocation counter object')}
    checks = (
        ('arena_requested_bytes', 'arena_retired_bytes'),
        ('heap_requested_bytes', 'heap_freed_bytes'),
        ('heap_buffers', 'heap_frees'),
    )
    mismatch = {}
    for requested, released in checks:
        left, right = counters.get(requested), counters.get(released)
        if left is None or right is None or left != right:
            mismatch[f'{prefix}.{requested}/{released}'] = (left, right)
    return mismatch

def normalized_slot_bytes(stats):
    explicit = stats.get('corpus_slot_bytes')
    if explicit in (2, 4):
        return explicit, 'corpus_slot_bytes'
    slots = stats.get('initial_slots') or 0
    corpus_bytes = stats.get('corpus_bytes') or 0
    if slots and corpus_bytes % slots == 0 and corpus_bytes // slots in (2, 4):
        return corpus_bytes // slots, 'corpus_bytes/initial_slots'
    # Old cohort rows encoded a per-slot width here; old ordinary rows encoded
    # total payload bytes. Prefer the corpus/slot ratio above whenever present.
    legacy = stats.get('initial_slot_bytes')
    if legacy in (2, 4):
        return legacy, 'legacy_initial_slot_bytes_per_slot'
    if slots and legacy and legacy % slots == 0 and legacy // slots in (2, 4):
        return legacy // slots, 'legacy_initial_slot_bytes_total/initial_slots'
    return None, 'unavailable'

def check_allocation_lifetimes(stats, require_speculative):
    mismatch = allocation_mismatches(stats.get('posting_allocations'), 'posting_allocations')
    speculative = stats.get('speculative_posting_allocations')
    if speculative is None and not require_speculative:
        return mismatch
    mismatch.update(allocation_mismatches(speculative, 'speculative_posting_allocations'))
    return mismatch

def expose_diagnostics(row, stats, require_extended):
    missing = sorted(EXTENDED_DIAGNOSTIC_FIELDS.difference(stats))
    if missing and require_extended:
        raise SystemExit(f'candidate runtime diagnostics missing: {missing}; outputs retained')
    slot_bytes, slot_source = normalized_slot_bytes(stats)
    diagnostics = {
        'alias_guarded': stats.get('alias_guarded'),
        'alias_fallback': stats.get('alias_fallback'),
        'corpus_slot_bytes': stats.get('corpus_slot_bytes'),
        'corpus_slot_bytes_normalized': slot_bytes,
        'corpus_slot_bytes_source': slot_source,
        'speculative': {key: value for key, value in stats.items() if key.startswith('speculative_')},
        'missing_extended_fields': missing,
    }
    row['runtime_diagnostics'] = diagnostics
    row['alias_guarded'] = diagnostics['alias_guarded']
    row['alias_fallback'] = diagnostics['alias_fallback']
    row['corpus_slot_bytes'] = diagnostics['corpus_slot_bytes']
    row['corpus_slot_bytes_normalized'] = slot_bytes
    row.update(diagnostics['speculative'])

def baseline_metadata(case, directory, corpus, corpus_sha, prefix, suffix, threads, vocab, min_frequency, thread_policy):
    if not case:
        return None
    env_path = directory/(case+'.environment.json')
    if not env_path.is_file():
        raise SystemExit(f'missing paired baseline environment before run: {env_path}')
    env = json.loads(env_path.read_text())
    old_path = env.get('corpus_path', env.get('corpus'))
    if not old_path or Path(old_path).resolve() != corpus:
        raise SystemExit(f'baseline corpus path differs before run: {old_path!r} != {corpus}')
    old_sha = env.get('corpus_sha256')
    if old_sha != corpus_sha:
        raise SystemExit(f'baseline corpus SHA differs before run: {old_sha} != {corpus_sha}')
    old_vocab = env.get('vocab_size')
    old_minfreq = env.get('min_frequency')
    if (old_vocab, old_minfreq) != (vocab, min_frequency):
        raise SystemExit(f'baseline vocab/min-frequency differ before run: {(old_vocab, old_minfreq)} != {(vocab, min_frequency)}')
    if env.get('prefix') != prefix or env.get('suffix') != suffix:
        raise SystemExit(f'baseline affix options differ before run: {(env.get("prefix"), env.get("suffix"))} != {(prefix, suffix)}')
    old_threads = env.get('threads_requested', env.get('requested_threads'))
    if old_threads is None:
        old_threads = int(env.get('env_overrides', {}).get('HF_BPE_BENCH_WORKERS', env.get('env_overrides', {}).get('RAYON_NUM_THREADS', '1')))
    if thread_policy == 'match' and old_threads != threads:
        raise SystemExit(f'baseline requested threads differ before run: {old_threads} != {threads}')
    if prefix is None and suffix is None and old_threads == 4:
        row_path = directory/(case+'.jsonl')
        if not row_path.is_file():
            raise SystemExit(f'missing paired baseline result for actual-worker validation: {row_path}')
        old_row = json.loads(row_path.read_text())
        old_stats = old_row.get('indexed_stats') or {}
        if (old_stats.get('workers'), old_stats.get('initialization_workers')) != (4, 4):
            raise SystemExit('baseline requests four threads but recorded actual stats are not workers/init=4')
    row_path = directory/(case+'.jsonl')
    baseline_row = json.loads(row_path.read_text()) if row_path.is_file() else {}
    model_sha = baseline_row.get('model_sha256')
    return {'case': case, 'environment_path': str(env_path.resolve()), 'result_path': str(row_path.resolve()) if row_path.is_file() else None,
            'model_sha256': model_sha, 'corpus_path': str(Path(old_path).resolve()),
            'corpus_sha256': old_sha, 'vocab_size': old_vocab, 'min_frequency': old_minfreq,
            'prefix': env.get('prefix'), 'suffix': env.get('suffix'), 'threads_requested': old_threads,
            'thread_policy': thread_policy}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', required=True)
    p.add_argument('--case', required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--result-dir', type=Path, default=ROOT/'results/affix-general-v1')
    p.add_argument('--prefix')
    p.add_argument('--suffix')
    p.add_argument('--oracle', action='store_true')
    p.add_argument('--baseline', action='store_true', help='record a frozen baseline without applying candidate-only stat gates')
    p.add_argument('--baseline-case', help='validate corpus and training configuration against an existing paired environment before running')
    p.add_argument('--baseline-result-dir', type=Path, default=ROOT/'results/affix-analysis')
    p.add_argument('--baseline-thread-policy', choices=['match','ignore'], default='match')
    p.add_argument('--cohort-disable', help='benchmark-only comma-separated cohort options to disable (or all)')
    p.add_argument('--expected-model-sha')
    p.add_argument('--expected-corpus-sha')
    p.add_argument('--require-absent-affixes', action='store_true')
    p.add_argument('--require-block-fused', action='store_true',
                   help='require observed non-AA block fusion in the frozen candidate')
    p.add_argument('--threads', type=int, choices=[1, 4], default=1)
    p.add_argument('--vocab', type=int, default=30000)
    p.add_argument('--min-frequency', type=int, default=2)
    p.add_argument('--max-token-length', type=int)
    p.add_argument('--timeout', type=int, default=0)
    p.add_argument('--require-parallel-rounds', action='store_true')
    p.add_argument('--require-extended-diagnostics', action='store_true',
                   help='fail before starting unless alias, slot-width, and speculative counters exist')
    p.add_argument('--allow-small-affix-init', action='store_true', help='allow 1 or requested initialization workers for tiny alias fixtures')
    p.add_argument('--allow-short-process', action='store_true', help='allow four-thread fixtures too short for OS thread sampling')
    a = p.parse_args()
    build = ROOT/'.build'/f'native-{a.label}'
    manifest_path = build/'build_manifest.json'
    if not manifest_path.is_file():
        raise SystemExit(f'missing immutable build manifest: {manifest_path}')
    build_manifest = json.loads(manifest_path.read_text())
    binary = build/'target/release'/f'hf-bpe-native-{a.label}'
    if not binary.is_file() or sha(binary) != build_manifest.get('binary_sha256'):
        raise SystemExit('binary is absent or does not match immutable build manifest')
    available_options = set(build_manifest.get('benchmark_cohort_switch_names', []))
    require_extended = a.require_extended_diagnostics or 'guarded_fast' in available_options
    if require_extended and not build_manifest.get('benchmark_extended_diagnostics', False):
        raise SystemExit('this build lacks required alias/speculative/corpus-slot diagnostics')
    disabled = [] if not a.cohort_disable else [x.strip() for x in a.cohort_disable.split(',') if x.strip()]
    if disabled:
        if not build_manifest.get('benchmark_cohort_switches_enabled'):
            raise SystemExit('this immutable build has no benchmark cohort switches')
        if ('all' in disabled and len(disabled) != 1) or any(x != 'all' and x not in available_options for x in disabled):
            raise SystemExit(f'invalid cohort disable list: {disabled}; available: {sorted(available_options)}')
    effective_disabled = sorted(available_options) if disabled == ['all'] else sorted(set(disabled))
    corpus = a.corpus.resolve()
    corpus_sha = sha(corpus)
    if a.expected_corpus_sha and corpus_sha != a.expected_corpus_sha:
        raise SystemExit(f'corpus SHA mismatch before run: got {corpus_sha}, expected {a.expected_corpus_sha}')
    baseline_pair = baseline_metadata(a.baseline_case, a.baseline_result_dir.resolve(), corpus, corpus_sha,
                                      a.prefix, a.suffix, a.threads, a.vocab, a.min_frequency,
                                      a.baseline_thread_policy)
    paired_model_sha = (baseline_pair or {}).get('model_sha256')
    if a.expected_model_sha and paired_model_sha and a.expected_model_sha != paired_model_sha:
        raise SystemExit(f'explicit expected model SHA differs from paired baseline row: {a.expected_model_sha} != {paired_model_sha}')
    expected_model_sha = a.expected_model_sha or paired_model_sha
    result_dir = a.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    result = result_dir/(a.case+'.jsonl')
    if any(result.with_suffix(suffix).exists() for suffix in ('.jsonl','.stdout','.stderr','.environment.json')):
        raise SystemExit(f'refusing to overwrite existing case artifacts: {a.case}')
    if a.require_absent_affixes:
        raw = corpus.read_bytes()
        for value in (a.prefix, a.suffix):
            if value is not None and value.encode('utf-8') in raw:
                raise SystemExit(f'affix marker occurs in input: {value!r}')

    before = system()
    if before.get('MemAvailable', 0) <= 1 << 30:
        raise SystemExit(f'MemAvailable <= 1 GiB; refusing to start ({before.get("MemAvailable")})')
    host_before = host()
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    env = dict(os.environ)
    for key in ('HF_BPE_PREFIX', 'HF_BPE_SUFFIX', 'HF_BPE_ORACLE', 'HF_BPE_MAX_TOKEN_LENGTH', 'HF_BPE_BENCH_WORKERS', 'HF_BPE_COHORT_DISABLE'):
        env.pop(key, None)
    env.update(TOKENIZERS_PARALLELISM='false', RAYON_NUM_THREADS=str(a.threads), HF_BPE_BENCH_WORKERS=str(a.threads))
    if a.prefix is not None:
        env['HF_BPE_PREFIX'] = a.prefix
    if a.suffix is not None:
        env['HF_BPE_SUFFIX'] = a.suffix
    if a.oracle:
        env['HF_BPE_ORACLE'] = '1'
    if a.max_token_length is not None:
        env['HF_BPE_MAX_TOKEN_LENGTH'] = str(a.max_token_length)
    if disabled:
        env['HF_BPE_COHORT_DISABLE'] = ','.join(disabled)
    command = [str(binary), str(corpus), 'none', 'reference', str(a.vocab), str(a.min_frequency)]
    metadata = {
        'case': a.case, 'label': a.label, 'command': command, 'worktree_commit': build_manifest['worktree_commit'],
        'build_manifest_path': str(manifest_path), 'build_manifest_sha256': sha(manifest_path),
        'binary_path': str(binary), 'binary_sha256': sha(binary),
        'build_script_sha256': build_manifest['build_script_sha256'],
        'run_script_sha256': sha(Path(__file__)), 'instrumentation_patch_sha256': build_manifest['instrumentation_patch_sha256'],
        'runner_source_sha256': build_manifest['runner_source_sha256'], 'runner_lock_sha256': build_manifest['runner_lock_sha256'],
        'corpus_path': str(corpus), 'corpus_bytes': corpus.stat().st_size, 'corpus_sha256': corpus_sha,
        'expected_corpus_sha256': a.expected_corpus_sha,
        'baseline_pair': baseline_pair,
        'prefix': a.prefix, 'suffix': a.suffix, 'oracle': a.oracle, 'baseline': a.baseline, 'max_token_length': a.max_token_length,
        'require_absent_affixes': a.require_absent_affixes, 'expected_model_sha256': expected_model_sha, 'threads_requested': a.threads,
        'require_block_fused': a.require_block_fused,
        'allow_short_process': a.allow_short_process,
        'vocab_size': a.vocab, 'min_frequency': a.min_frequency, 'env_overrides': {
            k: env[k] for k in ('TOKENIZERS_PARALLELISM', 'RAYON_NUM_THREADS', 'HF_BPE_BENCH_WORKERS',
                                'HF_BPE_PREFIX', 'HF_BPE_SUFFIX', 'HF_BPE_ORACLE', 'HF_BPE_MAX_TOKEN_LENGTH',
                                'HF_BPE_COHORT_DISABLE') if k in env
        }, 'mem_before': before, 'host_before': host_before,
        'cohort_disable': effective_disabled,
        'cohort_options_available': sorted(available_options),
        'cohort_option_defaults': build_manifest.get('benchmark_cohort_option_defaults', {}),
        'extended_diagnostics_expected': build_manifest.get('benchmark_extended_diagnostics', False),
        'posting_block_bits': build_manifest.get('benchmark_posting_block_bits', 32),
        'require_extended_diagnostics': require_extended,
    }
    input_manifest = corpus.with_suffix('.manifest.json')
    if not input_manifest.exists():
        input_manifest = corpus.parent/'manifest.json'
    metadata['input_manifest_path'] = str(input_manifest) if input_manifest.exists() else None
    metadata['input_manifest'] = json.loads(input_manifest.read_text()) if input_manifest.exists() else None
    env_path = result.with_suffix('.environment.json')
    env_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')

    peak_rss = peak_swap = max_threads = 0
    min_available = before['MemAvailable']
    fault = None
    start = time.monotonic()
    with result.with_suffix('.stdout').open('w') as stdout, result.with_suffix('.stderr').open('w') as stderr:
        proc = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
        try:
            while proc.poll() is None:
                current = system()
                min_available = min(min_available, current.get('MemAvailable', min_available))
                try:
                    status = Path(f'/proc/{proc.pid}/status').read_text().splitlines()
                    values = {line.split(':',1)[0]: int(line.split()[1])*1024 for line in status if line.startswith(('VmRSS:', 'VmSwap:'))}
                    peak_rss = max(peak_rss, values.get('VmRSS', 0))
                    peak_swap = max(peak_swap, values.get('VmSwap', 0))
                    max_threads = max(max_threads, len(list(Path(f'/proc/{proc.pid}/task').iterdir())))
                except FileNotFoundError:
                    pass
                if min_available <= 1 << 30:
                    fault = 'MemAvailable <= 1 GiB'
                    stop_child(proc)
                    break
                if a.timeout and time.monotonic()-start > a.timeout:
                    fault = 'timeout'
                    stop_child(proc)
                    break
                time.sleep(.25)
        except BaseException:
            stop_child(proc)
            raise
        returncode = proc.wait()
    wall_seconds = time.monotonic()-start
    after = system()
    host_after = host()
    child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    metadata.update(mem_after=after, host_after=host_after, wall_seconds=wall_seconds,
                    min_memavailable_bytes=min_available, peak_rss_bytes=peak_rss, peak_vmswap_bytes=peak_swap,
                    max_process_threads=max_threads, pswpin_delta=after.get('pswpin',0)-before.get('pswpin',0),
                    pswpout_delta=after.get('pswpout',0)-before.get('pswpout',0),
                    child_user_seconds=child_after.ru_utime-child_before.ru_utime,
                    child_system_seconds=child_after.ru_stime-child_before.ru_stime,
                    returncode=returncode, failure=fault)
    env_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    if returncode or fault:
        result.write_text(json.dumps({'failure': fault or returncode, 'wall_seconds': wall_seconds,
                                      'peak_rss_bytes': peak_rss, 'peak_vmswap_bytes': peak_swap})+'\n')
        raise SystemExit(f'{a.case} failed: {fault or returncode}; artifacts retained')
    row = json.loads(result.with_suffix('.stdout').read_text())
    if not a.oracle:
        stats = [json.loads(line)['bench_indexed_stats'] for line in result.with_suffix('.stderr').read_text().splitlines()
                 if line.startswith('{') and 'bench_indexed_stats' in line]
        if len(stats) != 1:
            raise SystemExit(f'expected one indexed stats row, got {len(stats)}')
        row['indexed_stats'] = stats[0]
        layout_bits = build_manifest.get('benchmark_posting_block_bits', 32)
        if layout_bits == 16 and not stats[0].get('layout', '').endswith('dict16'):
            raise SystemExit(f'expected 16-bit posting dictionary, got {stats[0].get("layout")}; outputs retained')
        if a.require_block_fused and stats[0].get('fused_block_batches', 0) <= 0:
            raise SystemExit('candidate never executed block fusion; outputs retained')
        if not a.baseline:
            stats = row['indexed_stats']
            affix = bool(a.prefix or a.suffix)
            if affix and a.threads == 4:
                disabled_set = set(effective_disabled)
                expected_workers = 1 if (
                    'parallel_apply' in disabled_set and 'guarded_fast' not in disabled_set
                ) else 4
                expected = {'workers': expected_workers}
                mismatch = {k: (stats.get(k), v) for k, v in expected.items() if stats.get(k) != v}
                init_workers = stats.get('initialization_workers')
                if init_workers not in (1, 4):
                    mismatch['initialization_workers'] = (init_workers, '1 or 4')
                min_threads = expected_workers + 1
                if a.threads == 4 and not a.allow_short_process and max_threads < min_threads:
                    mismatch['max_process_threads'] = (max_threads, f'>= {min_threads}')
                if mismatch: raise SystemExit(f'affix worker gate failed: {mismatch}; outputs retained')
            elif a.threads == 4:
                expected = {'workers': 4, 'initialization_workers': 4}
                mismatch = {k: (stats.get(k), v) for k, v in expected.items() if stats.get(k) != v}
                if max_threads < 5 and not a.allow_short_process: mismatch['max_process_threads'] = (max_threads, '>= 5')
                if mismatch: raise SystemExit(f'v2 no-affix four-thread gate failed: {mismatch}; outputs retained')
            mismatch = check_allocation_lifetimes(
                stats, build_manifest.get('benchmark_extended_diagnostics', False))
            if mismatch:
                raise SystemExit(f'postings allocation lifetime gate failed: {mismatch}; outputs retained')
        expose_diagnostics(row, stats, require_extended)
    else:
        row['indexed_stats'] = None
    if expected_model_sha and row['model_sha256'] != expected_model_sha:
        raise SystemExit(f'model SHA mismatch: got {row["model_sha256"]}, expected {expected_model_sha}; outputs retained')
    row.update(case=a.case, wall_seconds=wall_seconds, peak_rss_bytes=peak_rss,
               peak_vmswap_bytes=peak_swap, max_process_threads=max_threads,
               min_memavailable_bytes=min_available, env_metadata=str(env_path), environment_sha256=sha(env_path))
    result.write_text(json.dumps(row, ensure_ascii=False)+'\n')
    if peak_swap:
        raise SystemExit(f'VmSwap not zero: {peak_swap}; output retained')
    print(f'{a.case}: wall {wall_seconds:.3f}s train {row["train_ms"]:.1f}ms RSS {peak_rss/2**30:.2f}GiB swap 0 model {row["model_sha256"]}', flush=True)

if __name__ == '__main__':
    main()
