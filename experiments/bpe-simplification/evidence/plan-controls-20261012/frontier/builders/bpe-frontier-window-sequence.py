from pathlib import Path
import subprocess
root=Path('/root/code/tokenizers-workspaces/bpe-frontier-window-results-20261012')
for suite in ['controls','primary','reverse']:
    with (root/f'{suite}-controller.log').open('w') as log:
        subprocess.run(['python3', str(root/'measure.py'), '--suite', suite], stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f'{suite} complete', flush=True)
