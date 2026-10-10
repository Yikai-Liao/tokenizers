"""Compare scratch-backed Enum/Box storage with directly encoded owned positions."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

WORKERS = int(os.environ.get('BPE_MEASURE_WORKERS','4'))
ROOT = Path(os.environ.get('BPE_MEASURE_ROOT','/tmp/bpe-inline-capacity-measurements'))
ROOT.mkdir(parents=True,exist_ok=True)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

evidence = Path(__file__).resolve().parent
real = json.loads(Path(os.environ.get('BPE_MEASURE_INPUTS', str(evidence / 'inputs.json'))).read_text())
mode = os.environ.get('BPE_MEASURE_MODE','core')
arms = os.environ.get('BPE_MEASURE_ARMS','enum,direct,u642,u644').split(',')
binaries = {arm:('/tmp/bpe-box-enum-bin' if arm == 'enum' else f'/tmp/bpe-positions-value-{arm}-bin') for arm in arms}
cases = real
case_filter = os.environ.get('BPE_MEASURE_CASE')
if case_filter: cases = [item for item in cases if item['case'] == case_filter]
repeats = int(os.environ.get('BPE_MEASURE_ROUNDS','4'))
warmup = os.environ.get('BPE_MEASURE_WARMUP','1') == '1'
reverse_blocks = os.environ.get('BPE_MEASURE_ORDER','rotate') == 'reverse'
perf_enabled = os.environ.get('BPE_MEASURE_PERF','0') == '1'
records_path = ROOT / f'{mode}-runs.json'
records = json.loads(records_path.read_text()) if records_path.exists() else []
manifest = dict(baseline_commit='0416ae7a97bbc55f47f7969f4bdb04e8cfed1eff', box_commit='9f5b553dba61dba45a75021f4292997c16359dfb', builds=json.loads((evidence/'builds.json').read_text()), mode=mode, binaries=binaries, binary_sha256={k:digest(v) for k,v in binaries.items()},
    cases=cases, perf_enabled=perf_enabled, perf_counter_scope='whole subprocess, including input load and model serialization; training metrics still exclude both',
    scope='core: public do_train, load/serialization excluded; pipeline: public feed + train, validation excluded; HWM includes input and retained counts',
    warmup_enabled=warmup, order_method='reverse whole arm order in odd blocks' if reverse_blocks else 'rotate order by one arm in each block',
    design=f'{len(cases)} frozen input cases with per-case trainer settings; {repeats} blocks, first block warmup when enabled; workers{WORKERS} affinity0-{WORKERS-1}',
    limitation='Shared VM; descriptive observations, no statistical speedup claim.')
(ROOT / f'{mode}-manifest.json').write_text(json.dumps(manifest, indent=2))
for item in cases:
    assert digest(item['prepared_path']) == item['prepared_sha256']
    if mode=='pipeline': assert digest(item['path']) == item['source_sha256']
    previous = next((r for r in records if r["case"] == item["case"]), None)
    reference = json.loads((ROOT / f'{mode}-{item["case"]}-b{previous["block"]}-{previous["arm"]}' / "model.json").read_text()) if previous else None
    repeats = int(os.environ.get('BPE_MEASURE_ROUNDS','4'))
    for block in range(repeats):
        rotation = block % len(arms)
        order = (arms[::-1] if block % 2 else arms[:]) if reverse_blocks else arms[rotation:]+arms[:rotation]
        for arm in order:
            if any(r["case"] == item["case"] and r["block"] == block and r["arm"] == arm for r in records): continue
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
            job = dict(protocol_version=1, attempt_id=out.name, build_id=arm, input_id=item['case'], mode=mode, input=item['path'] if mode=='pipeline' else item['prepared_path'], output=str(out/'model.json'), workers=WORKERS, pretokenizer=item['pretokenizer'], trainer=item.get('trainer',dict(vocab_size=10000 if 'n' in item else 50000,min_frequency=1 if 'n' in item else 2,prefix=None,suffix=None,max_token_length=None)))
            (out/'job.json').write_text(json.dumps(job,indent=2))
            command = ['taskset','-c',f'0-{WORKERS-1}',binaries[arm],str(out/'job.json')]
            runner_command = command[:]
            if perf_enabled:
                command = ['perf','stat','-x',';','-o',str(out/'perf.csv'),'-e','{instructions,cycles},task-clock,context-switches,minor-faults,major-faults','--'] + command
            env = {k:v for k,v in os.environ.items() if not k.startswith(('BPE_TRACE_','TK_WORD_COUNTS_CACHE','TK_WRITE_WORD_COUNTS_CACHE','BPE_HEAP_DIAGNOSTIC'))}
            swap = rss = 0
            start = time.monotonic()
            with (out/'stdout.log').open('w') as stdout, (out/'stderr.log').open('w') as stderr:
                process = subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env)
                while process.poll() is None:
                    pids = [process.pid]
                    if perf_enabled:
                        try:
                            pids += [int(p) for p in Path(f'/proc/{process.pid}/task/{process.pid}/children').read_text().split()]
                        except (FileNotFoundError,ProcessLookupError): pass
                    for pid in pids:
                        try:
                            status = Path(f'/proc/{pid}/status').read_text().splitlines()
                            field = lambda name: next((int(s.split()[1]) for s in status if s.startswith(name)),0)
                            swap = max(swap, field('VmSwap:'))
                            rss = max(rss, field('VmRSS:'))
                        except (FileNotFoundError,ProcessLookupError): pass
                    time.sleep(.025)
            record = dict(case=item['case'],block=block,arm=arm,warmup=block==0 and repeats>1 and warmup,command=command,runner_command=runner_command,perf_enabled=perf_enabled,concurrent_builds_or_benchmarks=competitors,returncode=process.returncode,max_swap_kib=swap,sampled_max_rss_kib=rss,subprocess_wall=time.monotonic()-start)
            if process.returncode == 0:
                record.update(json.loads((out/'stdout.log').read_text()))
                actual = json.loads((out/'model.json').read_text())
                if reference is None: reference = actual
                record['model_equal'] = actual == reference
                record['model_sha256'] = digest(out/'model.json')
                record['valid'] = record['model_equal'] and swap == 0
            else: record['valid'] = False
            records.append(record)
            (ROOT/f'{mode}-runs.json').write_text(json.dumps(records,indent=2))
            print(json.dumps({k:record.get(k) for k in ('case','block','arm','valid','metrics')}),flush=True)
            if not record['valid']: raise RuntimeError(f'Invalid sample {out}; evidence retained')
summary = []
for item in cases:
    rows = [r for r in records if r['case']==item['case'] and not r['warmup']]
    if not rows: continue
    row = dict(case=item['case'])
    keys = ['train_seconds','train_cpu_seconds','process_hwm_kib_before_validation']
    if mode=='pipeline': keys += ['feed_seconds','feed_cpu_seconds','feed_hwm_kib','feed_rss_kib','train_rss_kib','pipeline_seconds','pipeline_cpu_seconds']
    for arm in binaries:
        arm_rows = [r for r in rows if r['arm']==arm]
        row[arm] = {k:statistics.median(r['metrics'][k] for r in arm_rows) for k in keys}
    ref = arms[0]
    row['reference'] = ref
    row['paired_median_delta_percent'] = {arm:{k:statistics.median((next(r for r in rows if r['arm']==arm and r['block']==block)['metrics'][k]/next(r for r in rows if r['arm']==ref and r['block']==block)['metrics'][k]-1)*100 for block in sorted(set(r['block'] for r in rows))) for k in keys} for arm in arms[1:]}
    summary.append(row)
(ROOT/f'{mode}-summary.json').write_text(json.dumps(summary,indent=2))
