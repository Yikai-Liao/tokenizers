"""Extend the exact pinned Chinese source prefix; run only with benchmarks idle."""
from pathlib import Path
import json, sys, subprocess, time
import run_experiment as experiment

BENCH = Path('/root/code/tokenizers-bpe-benchmarks')
sys.path.insert(0, str(BENCH))
from bench.inputs import corpus

OUT = experiment.OUT/'one-gib'

def prepare():
    assert not experiment.busy(), experiment.busy()
    OUT.mkdir(exist_ok=True)
    record = corpus(BENCH/'datasets/wikipedia-zh.json', 1024,
                    OUT/'raw', BENCH/'.bench/shards')
    raw = OUT/'raw/text.txt'
    # Compare the existing 512MiB bytes directly, without replicating paragraphs.
    import hashlib
    prefix = BENCH/'.bench/prezza/zh512/text.txt'
    digest = hashlib.sha256()
    with raw.open('rb') as source:
        remaining = prefix.stat().st_size
        while remaining:
            chunk = source.read(min(remaining, 1 << 20))
            assert chunk
            digest.update(chunk)
            remaining -= len(chunk)
    assert digest.hexdigest() == experiment.sha(prefix)
    prepared = OUT/'words.json'
    job = dict(protocol_version=1, attempt_id='prepare-1g', build_id='main',
        input_id='zh-1024MiB', mode='prepare', input=str(raw), output=str(prepared),
        workers=4, pretokenizer='bytelevel_regex', trainer=dict(vocab_size=50000,
        min_frequency=2, prefix=None, suffix=None, max_token_length=None))
    job_path = OUT/'prepare-job.json'
    job_path.write_text(json.dumps(job, indent=2))
    assert not experiment.busy(), experiment.busy()
    with (OUT/'prepare-result.json').open('w') as stdout, (OUT/'prepare-stderr.log').open('w') as stderr:
        subprocess.run(['taskset','-c','0-3',str(experiment.ROOT/'bin/baseline'),str(job_path)],
                       stdout=stdout, stderr=stderr, check=True)
    result = json.loads((OUT/'prepare-result.json').read_text())
    info = dict(case='zh-1024MiB', requested_mib=1024, path=str(raw),
        bytes=raw.stat().st_size, source_sha256=experiment.sha(raw),
        prepared_path=str(prepared), prepared_sha256=experiment.sha(prepared),
        prepared_bytes=prepared.stat().st_size, unique_words=result['unique_words'],
        prefix_512_sha256=digest.hexdigest(), source_manifest=record)
    (OUT/'input.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)

if __name__ == '__main__':
    prepare()
