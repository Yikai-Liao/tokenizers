"""Compare raw partial births against uniformly compressed fragments."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

WORKERS = int(os.environ.get('BPE_MEASURE_WORKERS','4'))
ROOT = Path(os.environ.get('BPE_MEASURE_ROOT','/tmp/bpe-write-real-measurements'))
ROOT.mkdir(exist_ok=True)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

real = json.loads(Path('/root/code/tokenizers-simplification-results/online-initial/input-pretokenizers.json').read_text())
mode = 'clean'
binaries = dict(baseline='/tmp/bpe-birth-baseline', candidate='/tmp/bpe-write-only-clean')
cases = [item for item in real if item['language'] == 'zh']
records = []
manifest = dict(baseline_commit='3a298346', candidate_patch_sha256=digest('/tmp/bpe-write-only.patch'), mode=mode, binaries=binaries, binary_sha256={k:digest(v) for k,v in binaries.items()}, cases=cases,
    scope='public do_train; load and serialization excluded from train time; HWM includes input loading',
    design=f'Chinese focus: one warmup + three AB/BA alternating pairs per case. workers{WORKERS} affinity0-{WORKERS-1}; vocab50k/min2; no affixes',
    limitation='Shared VM; descriptive observations, no statistical speedup claim.')
(ROOT / f'{mode}-manifest.json').write_text(json.dumps(manifest, indent=2))
for item in cases:
    assert digest(item['prepared_path']) == item['prepared_sha256']
    reference = None
    repeats = 1 if mode != 'clean' else (3 if 'n' in item else 4)
    for block in range(repeats):
        for arm in (['baseline','candidate'] if block % 2 == 0 else ['candidate','baseline']):
            competitors = []
            for entry in Path('/proc').iterdir():
                if not entry.name.isdigit(): continue
                try:
                    comm = (entry / 'comm').read_text().strip()
                    exe = str((entry / 'exe').resolve())
                    if comm in ('cargo','rustc') or comm.startswith('bpe-bench') or exe in binaries.values() or exe in ('/tmp/bpe-birth-baseline','/tmp/bpe-birth-clean','/tmp/bpe-write-only-clean','/tmp/bpe-global-measurements/candidate','/tmp/bpe-heap-clean','/tmp/bpe-heap-diagnostic-baseline','/tmp/bpe-heap-diagnostic-candidate'):
                        competitors.append((int(entry.name),comm))
                except (FileNotFoundError,ProcessLookupError): pass
            if competitors: raise RuntimeError(f'Concurrent build/benchmark: {competitors}')
            out = ROOT / f'{mode}-{item["case"]}-b{block}-{arm}'
            out.mkdir(exist_ok=False)
            job = dict(protocol_version=1, attempt_id=out.name, build_id=arm, input_id=item['case'], mode='core', input=item['prepared_path'], output=str(out/'model.json'), workers=WORKERS, pretokenizer=item['pretokenizer'], trainer=dict(vocab_size=10000 if 'n' in item else 50000,min_frequency=1 if 'n' in item else 2,prefix=None,suffix=None,max_token_length=None))
            (out/'job.json').write_text(json.dumps(job,indent=2))
            command = ['taskset','-c',f'0-{WORKERS-1}',binaries[arm],str(out/'job.json')]
            env = {k:v for k,v in os.environ.items() if not k.startswith(('BPE_TRACE_','TK_WORD_COUNTS_CACHE','TK_WRITE_WORD_COUNTS_CACHE','BPE_HEAP_DIAGNOSTIC'))}
            if mode != 'clean':
                env['BPE_HEAP_DIAGNOSTIC'] = '1'
                env['BPE_HEAP_DIAGNOSTIC_STRIDE'] = '1' if 'n' in item else '32'
            swap = rss = 0
            start = time.monotonic()
            with (out/'stdout.log').open('w') as stdout, (out/'stderr.log').open('w') as stderr:
                process = subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env)
                while process.poll() is None:
                    try:
                        status = Path(f'/proc/{process.pid}/status').read_text().splitlines()
                        field = lambda name: next((int(s.split()[1]) for s in status if s.startswith(name)),0)
                        swap = max(swap, field('VmSwap:'))
                        rss = max(rss, field('VmRSS:'))
                    except (FileNotFoundError,ProcessLookupError): pass
                    time.sleep(.025)
            record = dict(case=item['case'],block=block,arm=arm,warmup=mode=='clean' and 'n' not in item and block==0,command=command,concurrent_builds_or_benchmarks=competitors,returncode=process.returncode,max_swap_kib=swap,sampled_max_rss_kib=rss,subprocess_wall=time.monotonic()-start)
            if process.returncode == 0:
                record.update(json.loads((out/'stdout.log').read_text()))
                actual = json.loads((out/'model.json').read_text())
                if reference is None: reference = actual
                record['model_equal'] = actual == reference
                record['model_sha256'] = digest(out/'model.json')
                record['valid'] = record['model_equal'] and swap == 0
                if mode != 'clean':
                    trace = [json.loads(s.removeprefix('BPE_HEAP_DIAGNOSTIC ')) for s in (out/'stderr.log').read_text().splitlines() if s.startswith('BPE_HEAP_DIAGNOSTIC ')]
                    (out/'trace.json').write_text(json.dumps(trace,indent=2))
                    record['trace'] = trace
            else: record['valid'] = False
            records.append(record)
            (ROOT/f'{mode}-runs.json').write_text(json.dumps(records,indent=2))
            print(json.dumps({k:record.get(k) for k in ('case','block','arm','valid','metrics')}),flush=True)
            if not record['valid']: raise RuntimeError(f'Invalid sample {out}; evidence retained')
summary = []
for item in cases:
    rows = [r for r in records if r['case']==item['case'] and not r['warmup']]
    row = dict(case=item['case'])
    for arm in binaries:
        arm_rows = [r for r in rows if r['arm']==arm]
        row[arm] = {k:statistics.median(r['metrics'][k] for r in arm_rows) for k in ('train_seconds','train_cpu_seconds','process_hwm_kib_before_validation')}
        if mode != 'clean':
            trace = arm_rows[0]['trace']
            row[arm].update(max_stale_owned_bytes=max(t['stale_owned_bytes'] for t in trace),max_stale_records=max(t['stale_records'] for t in trace),peak_live_owned_bytes=max(t['peak_live_owned_bytes'] for t in trace),max_queue_descriptor_capacity_bytes=max(t['queue_descriptor_capacity_bytes'] for t in trace),max_map_payload_capacity_bytes=max(t['map_payload_capacity_bytes'] for t in trace),trace_records=len(trace))
    row['paired_median_delta_percent'] = {k:statistics.median((next(r for r in rows if r['arm']=='candidate' and r['block']==block)['metrics'][k]/next(r for r in rows if r['arm']=='baseline' and r['block']==block)['metrics'][k]-1)*100 for block in sorted(set(r['block'] for r in rows))) for k in row['baseline'] if k in ('train_seconds','train_cpu_seconds','process_hwm_kib_before_validation')}
    summary.append(row)
(ROOT/f'{mode}-summary.json').write_text(json.dumps(summary,indent=2))
