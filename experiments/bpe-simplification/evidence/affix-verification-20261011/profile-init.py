from pathlib import Path
import json
import os
import select
import subprocess
import time

root = Path(__file__).resolve().parent
rows = []
for language, arm, repeat in [('en', 'baseline', 1), ('en', 'flags-only', 1),
                              ('en', 'flags-only', 2), ('en', 'baseline', 2),
                              ('zh', 'baseline', 1), ('zh', 'flags-only', 1)]:
    run = root / 'profile-init' / f'{language}-{arm}-{repeat}'
    run.mkdir(parents=True)
    job = json.loads((root / 'runs' / f'{language}-affix-{arm}' / 'job.json').read_text())
    job['attempt_id'] = run.name
    job['output'] = str(run / 'model.json')
    (run / 'job.json').write_text(json.dumps(job, indent=2) + '\n')
    control_path, ack_path = run / 'control.fifo', run / 'ack.fifo'
    os.mkfifo(control_path)
    os.mkfifo(ack_path)
    control = os.open(control_path, os.O_RDWR)
    ack = os.open(ack_path, os.O_RDWR)
    commands = []
    command = ['perf', 'stat', '--all-user', '-e', 'cycles,instructions,branches,branch-misses',
               '-D', '-1', '--control', f'fifo:{control_path},{ack_path}', '-x', '\t',
               '-o', str(run / 'counters.tsv'), '--', 'taskset', '-c', '0-3',
               str(root / (arm + '-bin')), str(run / 'job.json')]
    (run / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    with (run / 'stdout.log').open('w') as out, (run / 'stderr.log').open('w') as err:
        process = subprocess.Popen(command, stdout=out, stderr=subprocess.PIPE, text=True, bufsize=1)
        for line in process.stderr:
            err.write(line)
            action = None
            if line.startswith('BPE_ALPHABET_SELECTION_SECONDS'):
                action = 'enable'
            elif line.startswith('BPE_INITIALIZATION_SECONDS'):
                action = 'disable'
            if action:
                sent = time.monotonic()
                os.write(control, (action + '\n').encode())
                assert select.select([ack], [], [], 3)[0], 'perf control acknowledgment timed out'
                response = os.read(ack, 4096).decode()
                commands.append(dict(action=action, sent=sent, acknowledged=time.monotonic(), response=response))
        process.wait()
    os.close(control)
    os.close(ack)
    control_path.unlink()
    ack_path.unlink()
    assert process.returncode == 0, (run, process.returncode)
    assert [x['action'] for x in commands] == ['enable', 'disable']
    (run / 'control-times.json').write_text(json.dumps(commands, indent=2) + '\n')
    row = dict(language=language, arm=arm, repeat=repeat, counter_window='selection-log to initialization-log, controlled externally',
               control_times=commands, raw_counters=(run / 'counters.tsv').read_text())
    rows.append(row)
    (root / 'profile-init.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(row), flush=True)
