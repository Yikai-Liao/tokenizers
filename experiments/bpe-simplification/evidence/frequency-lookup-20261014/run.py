"""Sequential native runs with explicit ambient workers and exact model checks."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def one(binary, case, definition, workers, directory, profile=False):
    directory.mkdir(parents=True, exist_ok=False)
    model = directory / 'model.json'
    job = dict(protocol_version=1, attempt_id=directory.name, build_id=binary.name,
               input_id=case, mode='core', workers=workers, output=str(model),
               **{k: definition[k] for k in ['input', 'pretokenizer', 'trainer']})
    job_path = directory / 'job.json'; job_path.write_text(json.dumps(job, indent=2))
    environment = {k:v for k,v in os.environ.items()
                   if not k.startswith(('TK_', 'TOKENIZERS_', 'RAYON_'))}
    environment.update(RAYON_NUM_THREADS=str(workers), TOKENIZERS_PARALLELISM='true')
    command = ['taskset', '-c', ','.join(str(i) for i in range(workers)), str(binary), str(job_path)]
    if profile:
        command = ['perf', 'record', '-F', '997', '-g', '--call-graph', 'dwarf,16384',
                   '-o', str(directory / 'perf.data'), '--', *command]
    command = ['/usr/bin/time', '-v', '-o', str(directory / 'time.txt'), *command]
    begin = time.monotonic()
    with (directory/'stdout.log').open('w') as out, (directory/'stderr.log').open('w') as err:
        status = subprocess.run(command, env=environment, stdout=out, stderr=err)
    record = dict(case=case, workers=workers, binary_sha256=sha(binary), input_sha256=definition['sha256'],
                  command=command, environment={k:environment[k] for k in ['RAYON_NUM_THREADS','TOKENIZERS_PARALLELISM']},
                  seconds=time.monotonic()-begin, returncode=status.returncode)
    if status.returncode:
        (directory/'result.json').write_text(json.dumps(record, indent=2))
        raise RuntimeError((directory, status.returncode))
    record['result'] = json.loads((directory/'stdout.log').read_text())
    assert record['result']['workers_effective'] == workers
    record['model_sha256'] = sha(model)
    with gzip.GzipFile(filename=str(directory/'model.json.gz'), mode='wb', mtime=0) as out:
        out.write(model.read_bytes())
    model.unlink()
    counters = Counter(); readers = []; reuse = Counter(); layouts = set()
    for line in (directory/'stderr.log').read_text().splitlines():
        if not line.startswith('LOOKUP_DIAG '): continue
        value = json.loads(line.removeprefix('LOOKUP_DIAG ')); kind = value.pop('kind')
        if kind == 'fresh':
            # Ignore the zero-directory temporary emitted by an earlier diagnostic build.
            if value['runs'] == 0: continue
            readers.append(value)
            for key, val in value.items():
                if key == 'jumps':
                    for i, n in enumerate(val): counters[f'jump_{i}'] += n
                elif key not in ['runs','first_index_sum','first_position','last_position']:
                    counters[key] += val
            counters['readers'] += 1
            counters['zero_readers'] += value['valid'] == 0
            counters['one_lookup_readers'] += value['first'] == 1 and value['expired'] == 0
        elif kind == 'reuse':
            prefix = 'whole_' if value.pop('whole') else 'history_'
            reuse.update({prefix + k:v for k,v in value.items()})
        elif kind == 'layout': layouts.add(tuple(sorted(value.items())))
        else: counters.update({kind+'_'+k:v for k,v in value.items()})
    record['diagnostic'] = dict(fresh=dict(counters), reuse=dict(reuse),
                              layouts=[dict(x) for x in sorted(layouts)])
    if readers:
        (directory/'readers.json.gz').write_bytes(gzip.compress(json.dumps(readers).encode(), mtime=0))
    if profile:
        with (directory/'perf-report.txt').open('w') as out:
            subprocess.run(['perf','report','--stdio','--no-children','-i',str(directory/'perf.data'),
                            '--sort','symbol','--percent-limit','0.05'], stdout=out, check=True)
    (directory/'result.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(dict(directory=str(directory), seconds=record['seconds'],
                          metrics=record['result'].get('metrics'), diagnostic=record['diagnostic'])), flush=True)
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--arms', nargs='+', required=True, help='label=binary')
    p.add_argument('--definitions', type=Path, default=HERE.parent/'corpus-boundary-20261013/inputs.json')
    p.add_argument('--cases', nargs='+', required=True)
    p.add_argument('--workers', nargs='+', type=int, default=[1])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--pairs', type=int, default=1); p.add_argument('--warmup', action='store_true')
    p.add_argument('--seed', type=int, default=20261014); p.add_argument('--profile', action='store_true')
    a = p.parse_args(); definitions = json.loads(a.definitions.read_text())
    a.out.mkdir(parents=True, exist_ok=False)
    for case in a.cases: assert sha(definitions[case]['input']) == definitions[case]['sha256'], case
    arms = [(name,Path(binary).resolve()) for name,binary in (x.split('=',1) for x in a.arms)]
    seed = random.Random(a.seed); records=[]; schedule=[]; models={}
    for case in a.cases:
        for workers in a.workers:
            for repeat in range(-int(a.warmup),a.pairs):
                order = list(arms); seed.shuffle(order)
                schedule.append(dict(case=case,workers=workers,repeat=repeat,order=[name for name,_ in order]))
                (a.out/'schedule.json').write_text(json.dumps(dict(seed=a.seed,schedule=schedule),indent=2))
                for arm,binary in order:
                    record=one(binary,case,definitions[case],workers,a.out/f'{case}-w{workers}-r{repeat}-{arm}',a.profile)
                    record.update(arm=arm,repeat=repeat,warmup=repeat<0)
                    # IDs, full configuration and ordered merges remain significant.
                    key=case
                    expected=models.setdefault(key,record['model_sha256'])
                    record['exact_model_match'] = expected == record['model_sha256']
                    records.append(record); (a.out/'runs.json').write_text(json.dumps(records,indent=2))
                    if not record['exact_model_match']:
                        (a.out/'failure.json').write_text(json.dumps(dict(
                            kind='exact-model-mismatch',case=case,workers=workers,repeat=repeat,
                            arm=arm,expected=expected,actual=record['model_sha256']),indent=2))
                        raise AssertionError((case,workers,repeat,arm,'exact model mismatch'))
