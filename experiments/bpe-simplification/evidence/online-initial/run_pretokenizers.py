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
    return json.loads((OUT/'input-pretokenizers.json').read_text())

def measure(info, arm, kind, repeat):
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
        input_id=case, mode=('pipeline' if kind == 'pipeline' else 'core'), input=(info['path'] if kind == 'pipeline' else info['prepared_path']), output=str(model),
        workers=4, pretokenizer=info['pretokenizer'], trainer=dict(vocab_size=50000,
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
    result = dict(case=case, arm=arm, kind=kind, repeat=repeat, command=cmd,
        environment={k:env[k] for k in ('BPE_INITIAL_EXPERIMENT','BPE_INITIAL_DIAGNOSTIC') if k in env},
        binary_sha256=sha(binary), returncode=p.returncode, failure=failure,
        elapsed_seconds=time.monotonic()-start, foreign_processes=foreign,
        max_swap_kib=max([x.get('VmSwap',0) for x in samples]+[0]),
        minimum_available_kib=min([x['available_kib'] for x in samples]+[99999999]))
    if p.returncode == 0:
        result.update(json.loads((d/'stdout.json').read_text()))
        reference = ROOT/'final-scaling/reference'/f'{case}.json'
        if arm == 'main' and not reference.exists():
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
    result['valid'] = result.get('model_equal',False) and not failure and p.returncode == 0 and result['max_swap_kib'] == 0
    (d/'samples.json').write_text(json.dumps(samples))
    (d/'result.json').write_text(json.dumps(result, indent=2))
    with (OUT/'runs.jsonl').open('a') as f: f.write(json.dumps(result)+'\n')
    print(json.dumps({k:result.get(k) for k in ('case','arm','kind','repeat','valid','metrics','initial_phase','geometry')}), flush=True)
    assert result['valid'], d
    return result

def prepare():
    assert not busy(), busy()
    repo = Path('/root/code/tokenizers-workspaces/bpe-simplification')
    manifest = json.loads((repo/'experiments/bpe-simplification/evidence/manifest-review-final.json').read_text())['inputs']
    result = []
    for language in ('en', 'zh'):
        raw = manifest[language+'-bytelevel-pipeline']
        assert sha(raw['path']) == raw['sha256']
        for pretokenizer in ('bytelevel_regex', 'whitespace'):
            case = language+('-256MiB' if pretokenizer == 'bytelevel_regex' else '-whitespace-256MiB')
            if pretokenizer == 'bytelevel_regex':
                prepared = manifest[language+'-bytelevel-core']
                assert sha(prepared['path']) == prepared['sha256']
                prepared_path = Path(prepared['path'])
                ref = ROOT/'final-scaling/reference'/f'{case}.json'
                old = ROOT/'reference'/f'{language}-core-w4-v50000.json'
                if not ref.exists(): ref.write_bytes(old.read_bytes())
                assert json.loads(ref.read_text()) == json.loads(old.read_text())
            else:
                d = OUT/'pretokenizers'/case
                d.mkdir(parents=True, exist_ok=False)
                prepared_path = d/'words.json'
                job = dict(protocol_version=1, attempt_id='prepare-'+case, build_id='main',
                    input_id=case, mode='prepare', input=raw['path'], output=str(prepared_path),
                    workers=4, pretokenizer=pretokenizer, trainer=dict(vocab_size=50000,
                    min_frequency=2, prefix=None, suffix=None, max_token_length=None))
                (d/'job.json').write_text(json.dumps(job, indent=2))
                assert not busy(), busy()
                with (d/'stdout.json').open('w') as stdout, (d/'stderr.log').open('w') as stderr:
                    subprocess.run(['taskset','-c','0-3',str(ROOT/'bin/baseline'),str(d/'job.json')],
                                   stdout=stdout, stderr=stderr, check=True)
            result.append(dict(case=case, language=language, requested_mib=256,
                pretokenizer=pretokenizer, path=raw['path'], bytes=raw['bytes'],
                source_sha256=raw['sha256'], prepared_path=str(prepared_path),
                prepared_sha256=sha(prepared_path), prepared_bytes=prepared_path.stat().st_size))
    (OUT/'input-pretokenizers.json').write_text(json.dumps(result, indent=2))
    print('Prepared pretokenizer matrix', flush=True)

def run_core():
    orders = (('main','baseline','candidate'), ('candidate','main','baseline'), ('baseline','candidate','main'))
    for info in inputs():
        # The frozen original plan already measures these exact core samples.
        if info['case'] == 'zh-256MiB': continue
        for repeat, order in enumerate(orders, 1):
            for arm in order: measure(info, arm, 'plain', repeat)

def run_pipeline():
    for info in inputs():
        for arm in ('main','baseline','candidate'):
            measure(info, arm, 'pipeline', 1)

def run_all():
    while Path('/proc/2492732').exists(): time.sleep(5)
    rows = [json.loads(l) for l in (OUT/'runs.jsonl').read_text().splitlines()]
    assert len(rows) == 61, 'prior frozen plans did not finish'
    print('Starting pretokenizer preparation', flush=True)
    prepare()
    print('Starting pretokenizer core comparisons', flush=True)
    run_core()
    print('Starting pretokenizer pipeline comparisons', flush=True)
    run_pipeline()
    print('Pretokenizer comparisons complete', flush=True)

def summary():
    rows = [json.loads(l) for l in (OUT/'runs.jsonl').read_text().splitlines()]
    assert all(r['valid'] for r in rows)
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
    (OUT/'summary.json').write_text(json.dumps(dict(runs=len(rows), groups=groups), indent=2))
    print(json.dumps(groups, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=('prepare','core','pipeline','all'))
    a = p.parse_args()
    {'prepare':prepare, 'core':run_core, 'pipeline':run_pipeline, 'all':run_all}[a.action]()
