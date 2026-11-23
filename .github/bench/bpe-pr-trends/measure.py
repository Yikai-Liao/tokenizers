"""Run one version on one language/worker machine and return a measurement record."""
from pathlib import Path
import gzip, hashlib, json, os, shutil, subprocess, time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def run_one(*, version, ref, head, binary, corpus, expected_sha, language, workers,
            round_name, warmup, output_root, environment):
    dest = output_root / 'runs' / f'{version}-{language}-w{workers}-{round_name}'
    dest.mkdir(parents=True, exist_ok=True)
    output = dest / 'model.json'
    job = {
        'protocol_version': 1, 'attempt_id': dest.name, 'build_id': head,
        'input_id': f'{language}128-sha256:{expected_sha}', 'input': str(corpus),
        'output': str(output), 'workers': workers, 'pretokenizer': 'bytelevel_regex',
        'trainer': {'vocab_size': 100000, 'min_frequency': 2, 'prefix': None,
                    'suffix': None, 'max_token_length': None},
    }
    (dest / 'job.json').write_text(json.dumps(job, indent=2))
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < workers:
        raise RuntimeError(f'need {workers} CPUs, have {allowed}')
    cpus = allowed[:workers]
    time_file = dest / 'time.json'
    fmt = '{"wall_seconds":%e,"user_seconds":%U,"system_seconds":%S,"maxrss_kib":%M,"exit_code":%x}'
    command = ['/usr/bin/time', '-o', str(time_file), '-f', fmt, 'taskset', '-c', ','.join(map(str, cpus)), str(binary), str(dest / 'job.json')]
    run_env = dict(environment, RAYON_NUM_THREADS=str(workers), RAYON_RS_NUM_THREADS=str(workers))
    with (dest / 'result.json').open('w') as stdout, (dest / 'stderr.log').open('w') as stderr:
        started = time.time_ns()
        process = subprocess.run(command, env=run_env, stdout=stdout, stderr=stderr)
    if process.returncode:
        raise RuntimeError(f'benchmark failed in {dest} with status {process.returncode}')
    result = json.loads((dest / 'result.json').read_text())
    external = json.loads(time_file.read_text())
    record = {
        'version': version, 'ref': ref, 'head': head, 'language': language,
        'workers': workers, 'round': round_name, 'warmup': warmup,
        'metrics': result['metrics'], 'process': external, 'model_sha256': sha(output),
        'effective_cpus': cpus, 'started_unix_ns': started,
        'path': str(dest.relative_to(output_root)),
    }
    (dest / 'summary.json').write_text(json.dumps(record, indent=2))
    return record, output


def compare_model(path, reference):
    with gzip.open(reference, 'rt') as stream:
        expected = json.load(stream)
    return expected == json.loads(Path(path).read_text())


def save_reference(path, reference):
    reference.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('rb') as source, gzip.open(reference, 'wb', compresslevel=1) as target:
        shutil.copyfileobj(source, target)
