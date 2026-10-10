"""Finish the frozen plans serially; no preparation/build overlaps measurements."""
from pathlib import Path
import subprocess, time, json
import run_experiment as experiment

OUT = experiment.OUT
prior_driver_pid = 2480499
while Path(f'/proc/{prior_driver_pid}').exists():
    time.sleep(5)
rows = [json.loads(line) for line in (OUT/'runs.jsonl').read_text().splitlines()]
assert len(rows) == 20 and all(r['valid'] for r in rows), 'diagnostic plan incomplete'
assert not experiment.busy(), experiment.busy()

stages = [
    ('prepare-1g', ['/root/code/tokenizers-bpe-benchmarks/.venv/bin/python',str(OUT/'prepare_1g.py')]),
    ('reference-1g', ['python3',str(OUT/'run_one_gib.py'),'main']),
    ('diagnostic-1g', ['python3',str(OUT/'run_one_gib.py'),'diagnostic']),
    ('plain-1g', ['python3',str(OUT/'run_one_gib.py'),'plain']),
    ('plain', ['python3',str(OUT/'run_experiment.py'),'plain']),
]
for name, command in stages:
    print('Starting '+name, flush=True)
    with (OUT/(name+'.log')).open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('Finished '+name, flush=True)
print('All planned runs complete', flush=True)
