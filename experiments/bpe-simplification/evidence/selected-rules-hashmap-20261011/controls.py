from pathlib import Path
import hashlib
import json
import re
import subprocess

root = Path(__file__).resolve().parent
rows = []
for case in ['aa', 'reuse']:
    for arm in ['baseline', 'candidate']:
        run = root / 'controls' / (case + '-' + arm)
        run.mkdir(parents=True)
        job = dict(protocol_version=1, attempt_id=run.name, build_id=arm,
                   input_id=case, mode='core', input=str(root / (case + '-control-words.json')),
                   output=str(run / 'model.json'), workers=4, pretokenizer='none',
                   trainer=dict(vocab_size=32, min_frequency=1, prefix=None,
                                suffix='a' if case == 'reuse' else None,
                                max_token_length=None, limit_alphabet=65536))
        (run / 'job.json').write_text(json.dumps(job, indent=2) + '\n')
        with (run / 'stdout.log').open('w') as out, (run / 'stderr.log').open('w') as err:
            result = subprocess.run(['taskset', '-c', '0-3', str(root / (arm + '-bin')),
                                     str(run / 'job.json')], stdout=out, stderr=err)
        assert result.returncode == 0, (case, arm)
        attempts = len(re.findall(r'BPE_INITIALIZATION_SECONDS', (run / 'stderr.log').read_text()))
        assert attempts == (2 if case == 'reuse' else 1), (case, arm, attempts)
        rows.append(dict(case=case, arm=arm, attempts=attempts,
                         model_sha256=hashlib.sha256((run / 'model.json').read_bytes()).hexdigest(),
                         output=json.loads((run / 'stdout.log').read_text())))
    assert rows[-1]['model_sha256'] == rows[-2]['model_sha256'], case
    print(case, 'full model equal; attempts:', rows[-1]['attempts'], flush=True)
(root / 'controls.json').write_text(json.dumps(rows, indent=2) + '\n')
