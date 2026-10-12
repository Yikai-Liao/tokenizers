from pathlib import Path
import subprocess, json, gzip, hashlib

root = Path('/root/code/tokenizers-workspaces/bpe-complete-publisher-results-20261012')
for suite in ['reverse', 'primary-whitespace-repeat', 'controls']:
    with (root/f'{suite}-controller.log').open('w') as log:
        subprocess.run(['python3', str(root/'measure.py'), '--suite', suite], stdout=log, stderr=subprocess.STDOUT, check=True)
    print('completed', suite, flush=True)
# This compression occurs after every normal timing child has exited.
for suite in ['primary', 'reverse', 'primary-whitespace-repeat', 'controls']:
    for row in json.loads((root/f'{suite}.json').read_text()):
        path = Path(row['model_path']); data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == row['model_sha256']
        zipped = path.with_suffix('.json.gz'); zipped.write_bytes(gzip.compress(data, mtime=0))
        assert gzip.decompress(zipped.read_bytes()) == data
        path.unlink()
print('normal models archived losslessly', flush=True)
subprocess.run(['python3', '/tmp/bpe-complete-publisher-diag-build.py'], check=True)
for suite in ['diagnostic', 'diagnostic-controls']:
    with (root/f'{suite}-controller.log').open('w') as log:
        subprocess.run(['python3', str(root/'measure.py'), '--suite', suite], stdout=log, stderr=subprocess.STDOUT, check=True)
    print('completed', suite, flush=True)
