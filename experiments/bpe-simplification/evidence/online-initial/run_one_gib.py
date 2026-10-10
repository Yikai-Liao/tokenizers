"""Controlled initial-index experiment. All jobs run serially on CPUs 0-3.

The four factorial arms share a frozen binary, 2^24-slot whole-word blocks,
owned temporary fragments, global frequency admission, and final encoding.
Only collection/freeze overlap and a four-block wave barrier differ.
"""
from pathlib import Path
import argparse, hashlib, json, os, statistics, subprocess, time

ROOT = Path('/root/code/tokenizers-simplification-results')
OUT = ROOT/'online-initial'
WORK = Path('/root/code/tokenizers-workspaces/bpe-online-experiment-20261010')
ARMS = ('baseline','raw-all','online-all','raw-wave','online-wave')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()

def busy(own=None):
    found = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == own: continue
        try:
            name = (p/'comm').read_text().strip()
            args = (p/'cmdline').read_bytes().split(b'\0')
            first = args[0].decode(errors='replace') if args else ''
            is_runner = first.startswith(str(ROOT/'bin')+'/') or first.startswith(str(OUT/'bin')+'/')
            is_runner |= first.endswith('/bpe-bench-runner') or name.startswith('bpe-bench')
            if name in ('rustc','cargo') or is_runner:
                found.append(dict(pid=int(p.name), command=b' '.join(args).decode(errors='replace')))
        except (FileNotFoundError, ProcessLookupError, PermissionError): pass
    return found

def inputs():
    return [json.loads((OUT/'one-gib/input.json').read_text())]

def measure(info, arm, kind, repeat, comparison_repeat=None):
    assert not busy(), busy()
    case = info['case']
    d = OUT/'runs'/kind/case/f'{repeat:02d}-{arm}'
    d.mkdir(parents=True, exist_ok=False)
    diagnostic = kind == 'diagnostic'
    if arm == 'main':
        binary = ROOT/'bin/baseline'
    elif arm == 'baseline':
        binary = OUT/'bin/bpe-bench-baseline-probe' if diagnostic else ROOT/'bin/review-final'
    elif arm == 'candidate': binary = OUT/'bin/bpe-bench-candidate'
    else: binary = OUT/'bin/bpe-bench-experimental'
    model = d/'model.json'
    job = dict(protocol_version=1, attempt_id=f'{kind}-{repeat}-{arm}', build_id=arm,
        input_id=case, mode='core', input=info['prepared_path'], output=str(model),
        workers=4, pretokenizer='bytelevel_regex', trainer=dict(vocab_size=50000,
        min_frequency=2, prefix=None, suffix=None, max_token_length=None))
    (d/'job.json').write_text(json.dumps(job, indent=2))
    env = {k:v for k,v in os.environ.items() if not k.startswith(('BPE_','TK_WORD_COUNTS_CACHE','TK_WRITE_WORD_COUNTS_CACHE'))}
    if arm not in ('main','baseline','codec-control','candidate'): env['BPE_INITIAL_EXPERIMENT'] = arm
    if diagnostic: env['BPE_INITIAL_DIAGNOSTIC'] = '1'
    cmd = ['taskset','-c','0-3',str(binary),str(d/'job.json')]
    samples, foreign = [], []
    failure = None
    start = time.monotonic()
    last_check = start
    with (d/'stdout.json').open('w') as out, (d/'stderr.log').open('w') as err:
        p = subprocess.Popen(cmd, stdout=out, stderr=err, env=env)
        while p.poll() is None:
            try:
                status = Path(f'/proc/{p.pid}/status').read_text().splitlines()
                fields = {l.split(':')[0]:int(l.split()[1]) for l in status
                          if l.startswith(('VmRSS:','VmHWM:','VmSwap:'))}
                available = next(int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines()
                                 if l.startswith('MemAvailable:'))
                samples.append(dict(seconds=time.monotonic()-start, available_kib=available, **fields))
                if fields.get('VmSwap',0) or available < 524288:
                    failure = 'swap-or-memory-guard'; p.terminate(); break
                if time.monotonic()-start > 900:
                    failure = 'own-process-timeout'; p.terminate(); break
                if time.monotonic()-last_check > 1:
                    others = busy(p.pid); last_check = time.monotonic()
                    if others:
                        foreign.extend(others); failure = 'foreign-heavy-process'; p.terminate(); break
            except (FileNotFoundError, ProcessLookupError): pass
            time.sleep(.02)
        p.wait()
    result = dict(case=case, arm=arm, kind=kind, repeat=repeat, comparison_repeat=comparison_repeat or repeat, command=cmd,
        environment={k:env[k] for k in ('BPE_INITIAL_EXPERIMENT','BPE_INITIAL_DIAGNOSTIC') if k in env},
        binary_sha256=sha(binary), returncode=p.returncode, failure=failure,
        elapsed_seconds=time.monotonic()-start, foreign_processes=foreign,
        max_swap_kib=max([x.get('VmSwap',0) for x in samples]+[0]),
        minimum_available_kib=min([x['available_kib'] for x in samples]+[99999999]))
    if p.returncode == 0:
        result.update(json.loads((d/'stdout.json').read_text()))
        reference = ROOT/'final-scaling/reference'/f'{case}.json'
        if arm == 'main':
            if not reference.exists():
                reference.write_bytes(model.read_bytes())
        result['model_equal'] = json.loads(model.read_text()) == json.loads(reference.read_text())
        result['model_sha256'] = sha(model)
        if result['model_equal']: model.unlink()
        log = (d/'stderr.log').read_text().splitlines()
        result['geometry'] = [json.loads(l.split(' ',1)[1]) for l in log if l.startswith('BPE_INITIAL_GEOMETRY ')]
        result['initial_phase'] = [json.loads(l.split(' ',1)[1]) for l in log if l.startswith('BPE_INITIAL_PHASE ')]
        assert not diagnostic or len(result['initial_phase']) == 1, 'fixture unexpectedly restarts'
        assert result['effective_affinity'] == [0,1,2,3]
        if arm in ARMS[1:] and diagnostic:
            assert result['geometry'][0]['block_slot_cap'] == 1 << 24
            assert result['geometry'][0]['max_block_slots'] <= 1 << 24
    if p.returncode != 0:
        log = (d/'stderr.log').read_text().splitlines()
        result['initial_phase'] = [json.loads(l.split(' ',1)[1]) for l in log if l.startswith('BPE_INITIAL_PHASE ')]
        result['geometry'] = [json.loads(l.split(' ',1)[1]) for l in log if l.startswith('BPE_INITIAL_GEOMETRY ')]
    result['valid'] = result.get('model_equal',False) and not failure and p.returncode == 0 and result['max_swap_kib'] == 0
    (d/'samples.json').write_text(json.dumps(samples))
    (d/'result.json').write_text(json.dumps(result, indent=2))
    with (OUT/'runs.jsonl').open('a') as f: f.write(json.dumps(result)+'\n')
    print(json.dumps({k:result.get(k) for k in ('case','arm','kind','repeat','valid','metrics','initial_phase','geometry')}), flush=True)
    if not result['valid']:
        assert result['failure'] == 'swap-or-memory-guard', d
        print('EXCLUDED memory-guard run: '+str(d), flush=True)
    return result

def run_diagnostics():
    # Rotating/reversed order avoids giving one arm all early or late host periods.
    for info in inputs():
        for repeat, order in enumerate((ARMS, tuple(reversed(ARMS))), 1):
            for arm in order: measure(info, arm, 'diagnostic', repeat)

def run_plain():
    orders = (('main','baseline','candidate'), ('candidate','main','baseline'), ('baseline','candidate','main'))
    for info in inputs():
        for repeat, order in enumerate(orders, 1):
            for arm in order:
                measure(info, arm, 'plain', repeat + (arm == 'main'), comparison_repeat=repeat)
            if repeat == 1: measure(info, 'online-wave', 'plain', 1)

def control():
    for info in inputs(): measure(info, 'codec-control','plain',1)

def summary():
    rows = [json.loads(l) for l in (OUT/'runs.jsonl').read_text().splitlines()]
    rows = [r for r in rows if r['valid']]
    groups = []
    for case in (x['case'] for x in inputs()):
        for kind in ('diagnostic','plain'):
            for arm in (*ARMS,'candidate','codec-control'):
                runs = [r for r in rows if (r['case'],r['kind'],r['arm']) == (case,kind,arm)]
                if not runs: continue
                item = dict(case=case, kind=kind, arm=arm, n=len(runs), metrics={})
                for key in runs[0]['metrics']:
                    vals = [r['metrics'][key] for r in runs]
                    item['metrics'][key] = dict(median=statistics.median(vals), minimum=min(vals), maximum=max(vals), raw=vals)
                if kind == 'diagnostic':
                    item['initial_phase'] = {}
                    for key in runs[0]['initial_phase'][0]:
                        vals = [r['initial_phase'][0][key] for r in runs]
                        item['initial_phase'][key] = dict(median=statistics.median(vals),minimum=min(vals),maximum=max(vals),raw=vals)
                groups.append(item)
    (OUT/'one-gib/summary.json').write_text(json.dumps(dict(runs=len(rows), groups=groups), indent=2))
    print(json.dumps(groups, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=('diagnostic','plain','main','summary'))
    a = p.parse_args()
    {'diagnostic':run_diagnostics, 'plain':run_plain, 'main':lambda:measure(inputs()[0], 'main', 'plain', 1), 'summary':summary}[a.action]()
