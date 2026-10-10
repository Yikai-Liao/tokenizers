from pathlib import Path
import json
from run_full_phases_1g import measure, busy
from run_pretokenizers import prepare, inputs

OUT=Path('/root/code/tokenizers-simplification-results/online-initial')

def run():
    assert not busy(), busy()
    prior=[json.loads(l) for l in (OUT/'runs.jsonl').read_text().splitlines()]
    complete=[r for r in prior if r['case']=='zh-1024MiB' and r['kind']=='phases']
    assert len(complete)==2 and all(r['valid'] for r in complete), '1GiB forward phase pair unfinished'
    prepare()
    for info in inputs():
        for repeat,order in enumerate((('main','baseline','candidate'),),1):
            for arm in order: measure(info,arm,'phases',repeat)
    for info in inputs():
        for arm in ('main','baseline','candidate'): measure(info,arm,'phases-pipeline',1)
    print('Full-phase pretokenizer contrast complete',flush=True)

if __name__=='__main__': run()
