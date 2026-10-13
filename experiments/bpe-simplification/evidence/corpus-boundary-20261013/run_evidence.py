"""Run frozen core cases, keeping raw outputs and alternating paired order."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent


def run(binary, arm, case, workers, repeat, group):
    cases = json.loads((HERE / 'inputs.json').read_text())
    definition = cases[case]
    directory = HERE / group / f'{case}-w{workers}-r{repeat}-{arm}'
    directory.mkdir(parents=True, exist_ok=True)
    job = dict(protocol_version=1, attempt_id=directory.name, build_id=arm,
               input_id=case, mode='core', workers=workers, output=str(directory / 'model.json'),
               **{k:v for k,v in definition.items() if k in ['input','pretokenizer','trainer']})
    path = directory / 'job.json'
    path.write_text(json.dumps(job, indent=2))
    begin = time.monotonic()
    with (directory / 'stdout.log').open('w') as out, (directory / 'stderr.log').open('w') as err:
        result = subprocess.run([str(binary), str(path)], stdout=out, stderr=err)
    assert result.returncode == 0, (directory, result.returncode)
    output = json.loads((directory / 'stdout.log').read_text())
    model = directory / 'model.json'
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    with gzip.open(directory / 'model.json.gz', 'wb') as target:
        target.write(model.read_bytes())
    model.unlink()
    trace = [json.loads(line.removeprefix('BPE_DIAG ')) for line in
             (directory / 'stderr.log').read_text().splitlines() if line.startswith('BPE_DIAG ')]
    events = Counter(event['kind'] for event in trace)
    selects = [event for event in trace if event['kind'] == 'select']
    coverage = dict(events)
    coverage.update(aa=sum(event['pair'][0] == event['pair'][1] for event in selects),
                    ab=sum(event['pair'][0] != event['pair'][1] for event in selects),
                    existing_fresh=sum(event['existing'] and not event['reuse'] for event in selects),
                    multi_rule=sum(event['kind']=='batch' and event['rules']>1 for event in trace),
                    whole_words=sum(event.get('whole_words',False) for event in trace))
    record = dict(arm=arm, case=case, workers=workers, repeat=repeat,
                  elapsed_process_seconds=time.monotonic()-begin, metrics=output['metrics'],
                  model_sha256=model_hash, coverage=coverage,
                  trace_sha256=hashlib.sha256(json.dumps(trace).encode()).hexdigest(),
                  binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  directory=str(directory))
    (directory / 'result.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arms', nargs='+', required=True, help='name=binary')
    parser.add_argument('--cases', nargs='+', default=['aa','ab','reserved','alias','length-reuse'])
    parser.add_argument('--workers', nargs='+', type=int, default=[1,6])
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--group', required=True)
    args = parser.parse_args()
    arms = [(name,Path(binary)) for name,binary in (a.split('=',1) for a in args.arms)]
    results=[]
    for case in args.cases:
        for workers in args.workers:
            for repeat in range(args.rounds):
                for name,binary in (arms if repeat % 2 == 0 else arms[::-1]):
                    results.append(run(binary,name,case,workers,repeat,args.group))
    (HERE / (args.group + '-runs.json')).write_text(json.dumps(results,indent=2))
