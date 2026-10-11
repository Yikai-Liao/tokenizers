"""Compare the full 2048-ID reuse fixture after fixing only initial heap order.

Builds on a disposable baseline checkout, restoring its index source afterward.
Baseline and candidate still execute their exact commit implementations. No
production source uses this diagnostic hasher. Compare arms within each worker
count; production initial aggregate maps use randomized hash iteration order.
"""
from pathlib import Path
import argparse
import difflib
import hashlib
import json
import os
import shutil
import subprocess

p=argparse.ArgumentParser(description=__doc__)
for name in ['baseline-root','candidate-root','output','target','input']:
    p.add_argument('--'+name,required=True,type=Path)
a=p.parse_args()
a.output.mkdir(exist_ok=True,parents=True)
relative=Path('tokenizers/tk-train/src/trainers/bpe/index.rs')
path=a.baseline_root/relative
original=path.read_text()
needle='let mut states = AHashMap::<Pair, State<Vec<Positions>>>::new();'
replacement='let mut states = AHashMap::<Pair, State<Vec<Positions>>>::with_hasher(ahash::RandomState::with_seeds(11, 13, 17, 19));'
builds={}
try:
    for arm,root in [('baseline',a.baseline_root),('candidate',a.candidate_root)]:
        source=(root/relative).read_text()
        assert source.count(needle)==1
        seeded=source.replace(needle,replacement)
        path.write_text(seeded)
        (a.output/f'{arm}-initial-seed.patch').write_text(''.join(difflib.unified_diff(source.splitlines(True),seeded.splitlines(True),fromfile=str(relative),tofile=str(relative))))
        env=dict(os.environ,CARGO_TARGET_DIR=str(a.target))
        command=['cargo','build','--offline','--release','--locked','--manifest-path',str(a.baseline_root/'experiments/bpe-simplification/runner/Cargo.toml')]
        with (a.output/f'{arm}-build.log').open('w') as log: subprocess.run(command,env=env,stdout=log,stderr=log,check=True)
        binary=a.output/arm
        shutil.copy2(a.target/'release/bpe-bench-runner',binary)
        builds[arm]=dict(command=command,source_sha256=hashlib.sha256(seeded.encode()).hexdigest(),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
finally:
    path.write_text(original)
(a.output/'builds.json').write_text(json.dumps(builds,indent=2))
runs=[]
for workers in [1,4]:
    reference=None
    for block in range(3):
        for arm in ['baseline','candidate']:
            output=a.output/f'{arm}-{workers}-{block}-model.json'
            job=dict(protocol_version=1,attempt_id=output.stem,build_id='seeded-'+arm,input_id='generated-reuse',mode='core',
                input=str(a.input),output=str(output),workers=workers,pretokenizer='none',
                trainer=dict(vocab_size=2048,min_frequency=1,prefix=None,suffix='a',max_token_length=None))
            job_path=a.output/f'{output.stem}-job.json';job_path.write_text(json.dumps(job,indent=2))
            result=subprocess.check_output([str(a.output/arm),str(job_path)],text=True)
            (a.output/f'{output.stem}-stdout.log').write_text(result)
            model=json.loads(output.read_text())
            if reference is None: reference=model
            equal=model==reference
            runs.append(dict(arm=arm,workers=workers,block=block,model_equal=equal,sha256=hashlib.sha256(output.read_bytes()).hexdigest()))
            (a.output/'runs.json').write_text(json.dumps(runs,indent=2))
            assert equal,(arm,workers,block)
            print(arm,workers,block,'complete model agrees',flush=True)
