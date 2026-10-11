"""Rotate three arms in separate processes; exclude each process's first 3 commits."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import statistics
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--probe', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--rounds', type=int, default=5)
p.add_argument('--repetitions', type=int, default=20)
a = p.parse_args()
a.output.mkdir(exist_ok=True, parents=True)
arms = ['baseline','shards_only','candidate']
cases = ['removals','complete','mixed','fresh-small','fresh-spill','fresh-large','reuse-small','reuse-spill','reuse-large']
manifest = dict(binaries={arm:dict(path=str(a.probe / arm), sha256=hashlib.sha256((a.probe / arm).read_bytes()).hexdigest()) for arm in arms},
    rounds=a.rounds, repetitions=a.repetitions, workers=[1,4], cpus='0-3',
    scope='PairIndex::commit only: routing, parallel shard commits and heap publication. Input construction and result decoding excluded.',
    allocator='System wrapper; diagnostic counting disabled for timed runs',
    warmup='Three excluded commits per process',
    cpu=subprocess.check_output(['lscpu'],text=True),rustc=subprocess.check_output(['rustc','-Vv'],text=True))
(a.output / 'commit-manifest.json').write_text(json.dumps(manifest,indent=2))
runs=[]
for workers in [1,4]:
    for case in cases:
        expected=None
        for block in range(a.rounds):
            order = arms[block%3:]+arms[:block%3]
            if block%2: order.reverse()
            for arm in order:
                command=['taskset','-c','0-3',str(a.probe / arm),case,str(workers),str(a.repetitions)]
                env = dict(os.environ)
                env.pop('COMMIT_PROBE_ALLOC',None)
                result=json.loads(subprocess.check_output(command,env=env,text=True))
                if expected is None: expected=result['checksum']
                assert result['checksum']==expected,(case,arm)
                run=dict(arm=arm,block=block,command=command,**result)
                runs.append(run)
                (a.output/'commit-runs.json').write_text(json.dumps(runs,indent=2))
        print(f'{workers} workers {case}: all checksums match',flush=True)
summary=[]
for workers in [1,4]:
    for case in cases:
        selected=[r for r in runs if r['workers']==workers and r['case']==case]
        medians={arm:statistics.median(statistics.median(r['durations_ns']) for r in selected if r['arm']==arm) for arm in arms}
        def delta(before,after):
            return statistics.median((statistics.median(next(r for r in selected if r['arm']==after and r['block']==block)['durations_ns']) /
                                      statistics.median(next(r for r in selected if r['arm']==before and r['block']==block)['durations_ns'])-1)*100 for block in range(a.rounds))
        summary.append(dict(workers=workers,case=case,median_ns=medians,
            paired_delta_percent=dict(shard_ownership=delta('baseline','shards_only'),uniform_smallvec=delta('shards_only','candidate'),total=delta('baseline','candidate'))))
(a.output/'commit-summary.json').write_text(json.dumps(summary,indent=2))
allocations=[]
for case in cases:
    for arm in arms:
        env=dict(os.environ,COMMIT_PROBE_ALLOC='1')
        result=json.loads(subprocess.check_output(['taskset','-c','0',str(a.probe/arm),case,'1','5'],env=env,text=True))
        allocations.append(dict(case=case,arm=arm,layout=result['layout'],
            median_allocations=statistics.median(v[0] for v in result['allocations_and_requested_bytes']),
            median_requested_bytes=statistics.median(v[1] for v in result['allocations_and_requested_bytes'])))
(a.output/'allocations.json').write_text(json.dumps(allocations,indent=2))
