"""Alternating release-runner comparisons, with full parsed-model equality."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

p=argparse.ArgumentParser(description=__doc__)
for name in ['baseline','candidate','output']:
    p.add_argument('--'+name,required=True,type=Path)
for name in ['zh','en','reuse']:
    p.add_argument('--'+name,type=Path)
p.add_argument('--rounds',type=int,default=3)
p.add_argument('--cases',type=Path)
p.add_argument('--workers',type=int,nargs='+',default=[1,4])
p.add_argument('--only',choices=['zh','en','reuse'])
p.add_argument('--zh-label',default='zh-whitespace-16MiB')
p.add_argument('--zh-vocab',type=int,default=32000)
p.add_argument('--en-label',default='en-whitespace-256MiB')
p.add_argument('--en-vocab',type=int,default=50000)
a=p.parse_args()
a.output.mkdir(exist_ok=True,parents=True)
arms={name:str(getattr(a,name).resolve()) for name in ['baseline','candidate']}
cases={
    a.zh_label:dict(path=a.zh,pretokenizer='whitespace',vocab_size=a.zh_vocab,min_frequency=2,suffix=None),
    a.en_label:dict(path=a.en,pretokenizer='whitespace',vocab_size=a.en_vocab,min_frequency=2,suffix=None),
    'generated-reuse':dict(path=a.reuse,pretokenizer='none',vocab_size=256,min_frequency=1,suffix='a'),
}
if a.cases:
    cases=json.loads(a.cases.read_text())
    for case in cases.values(): case['path']=Path(case['path'])
if a.only:
    label={'zh':a.zh_label,'en':a.en_label,'reuse':'generated-reuse'}[a.only]
    cases={label:cases[label]}
missing=[name for name,case in cases.items() if case['path'] is None]
if missing:
    p.error('Input path required for selected case(s): '+', '.join(missing))
def digest(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
manifest=dict(binaries={arm:dict(path=path,sha256=digest(path)) for arm,path in arms.items()},
    cases={name:dict(**{k:v for k,v in case.items() if k!='path'},path=str(case['path']),bytes=case['path'].stat().st_size,sha256=digest(case['path'])) for name,case in cases.items()},
    workers=a.workers,cpus='0-3',rounds=a.rounds,warmup_round=0,
    scope='Public do_train including output construction; input loading and serialization excluded. HWM includes input loading.',
    cpu=subprocess.check_output(['lscpu'],text=True),rustc=subprocess.check_output(['rustc','-Vv'],text=True),
    profile=dict(opt_level=3,lto='fat',codegen_units=1))
(a.output/'training-manifest.json').write_text(json.dumps(manifest,indent=2))
runs=[]
for name,case in cases.items():
    reference=None
    for workers in a.workers:
        for block in range(a.rounds):
            for arm in (['baseline','candidate'] if block%2==0 else ['candidate','baseline']):
                folder=a.output/f'{name}-{workers}-{block}-{arm}'
                folder.mkdir()
                job=dict(protocol_version=1,attempt_id=folder.name,build_id=arm,input_id=name,mode='core',
                    input=str(case['path']),output=str(folder/'model.json'),workers=workers,pretokenizer=case['pretokenizer'],
                    trainer=dict(vocab_size=case['vocab_size'],min_frequency=case['min_frequency'],prefix=None,suffix=case['suffix'],max_token_length=None))
                (folder/'job.json').write_text(json.dumps(job,indent=2))
                command=['taskset','-c','0-3',arms[arm],str(folder/'job.json')]
                env={k:v for k,v in os.environ.items() if not k.startswith(('TK_WORD_COUNTS_CACHE','TK_WRITE_WORD_COUNTS_CACHE','BPE_TRACE_'))}
                env['RUST_BACKTRACE']='full'
                swap=0
                with (folder/'stdout.log').open('w') as out, (folder/'stderr.log').open('w') as err:
                    process=subprocess.Popen(command,env=env,stdout=out,stderr=err)
                    while process.poll() is None:
                        try:
                            status=Path(f'/proc/{process.pid}/status').read_text().splitlines()
                            swap=max(swap,next((int(line.split()[1]) for line in status if line.startswith('VmSwap:')),0))
                        except (FileNotFoundError,ProcessLookupError): pass
                        time.sleep(.025)
                assert process.returncode==0,folder
                output=json.loads((folder/'stdout.log').read_text())
                model=json.loads((folder/'model.json').read_text())
                if reference is None: reference=model
                assert model==reference,(name,arm,workers,block)
                assert swap==0,folder
                run=dict(case=name,workers=workers,arm=arm,block=block,warmup=block==0,command=command,
                    model_equal=True,model_sha256=digest(folder/'model.json'),max_swap_kib=swap,**output)
                runs.append(run)
                (a.output/'training-runs.json').write_text(json.dumps(runs,indent=2))
                print(json.dumps({key:run[key] for key in ['case','workers','block','arm','metrics']}),flush=True)
summary=[]
keys=['train_seconds','train_cpu_seconds','process_hwm_kib_before_validation']
for name in cases:
    for workers in a.workers:
        selected=[r for r in runs if r['case']==name and r['workers']==workers and not r['warmup']]
        row=dict(case=name,workers=workers,formal_pairs=a.rounds-1)
        for arm in arms: row[arm]={key:statistics.median(r['metrics'][key] for r in selected if r['arm']==arm) for key in keys}
        row['paired_median_delta_percent']={key:statistics.median((next(r for r in selected if r['arm']=='candidate' and r['block']==block)['metrics'][key]/next(r for r in selected if r['arm']=='baseline' and r['block']==block)['metrics'][key]-1)*100 for block in range(1,a.rounds)) for key in keys}
        summary.append(row)
(a.output/'training-summary.json').write_text(json.dumps(summary,indent=2))
