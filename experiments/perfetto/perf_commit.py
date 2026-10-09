"""Collect sampled CPU stacks alongside existing spans, without changing BPE."""
import argparse
import json
import os
import subprocess
from pathlib import Path
from run import ROOT, DIAG, fingerprint, job


def record(case, rep):
    out = ROOT / 'runs' / f'{case}-w4-perf-r{rep}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        return
    request = job(case, 4, out)
    (out / 'job.json').write_text(json.dumps(request, indent=2))
    env = os.environ.copy()
    env['BPE_TRACE_LEVEL'] = '2'
    env['BPE_TRACE_PATH'] = str(out / 'events.json')
    with (out / 'perf.log').open('w') as err:
        completed = subprocess.run(
            ['perf', 'record', '--clockid', 'mono', '--timestamp', '-e', 'cpu-clock:u',
             '-F', '499', '--call-graph', 'dwarf,8192', '-o', str(out / 'perf.data'),
             '--', 'taskset', '-c', '0-3', str(DIAG), str(out / 'job.json')],
            env=env, stdout=subprocess.PIPE, stderr=err, text=True, check=True)
    (out / 'stdout.log').write_text(completed.stdout)
    result = json.loads(completed.stdout)
    result.update(case=case, workers=4, arm='perf', rep=rep,
                  model_sha256=fingerprint(out / 'model.json'))
    assert result['model_sha256'] == fingerprint(ROOT / 'reference' / f'{case}.json')
    (out / 'model.json').unlink()
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    print(case, rep, result['metrics']['train_seconds'], flush=True)
    with (out / 'perf-script.txt').open('w') as output:
        subprocess.run(['perf', 'script', '-i', str(out / 'perf.data'), '--ns', '--inline',
                        '-F', 'pid,tid,time,event,period,ip,sym,dso'],
                       stdout=output, stderr=(out / 'script.log').open('w'), check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', nargs='+', default=['en-bytelevel', 'zh-bytelevel'])
    parser.add_argument('--rep', type=int, default=1)
    args = parser.parse_args()
    for case in args.cases:
        record(case, args.rep)
