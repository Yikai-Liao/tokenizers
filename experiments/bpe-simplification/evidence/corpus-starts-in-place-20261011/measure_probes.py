"""Real CorpusPlan layout checkpoints in separate diagnostic zero-merge jobs."""
import json,os,subprocess,time
from pathlib import Path
here=Path(__file__).resolve().parent
root=Path('/tmp/bpe-corpus-b1-probes-20261011');root.mkdir(exist_ok=True)
rows=[]
for case,config in json.loads((here/'cases.json').read_text()).items():
    reference=None
    for block in range(3):
        for arm in (['baseline','candidate'] if block%2==0 else ['candidate','baseline']):
            folder=root/f'{case}-{block}-{arm}';folder.mkdir(exist_ok=True)
            trainer=dict(vocab_size=0,min_frequency=2,prefix=None,suffix=None,max_token_length=None)
            job=dict(protocol_version=1,attempt_id=folder.name,build_id='probe-'+arm,input_id=case,mode='core',input=config['path'],output=str(folder/'model.json'),workers=4,pretokenizer=config['pretokenizer'],trainer=trainer)
            (folder/'job.json').write_text(json.dumps(job,indent=2)+'\n')
            command=['taskset','-c','0-3',f'/tmp/bpe-corpus-b1-probe-{arm}-bin-20261011',str(folder/'job.json')]
            swap=0
            with (folder/'stdout.log').open('w') as out,(folder/'stderr.log').open('w') as err:
                process=subprocess.Popen(command,stdout=out,stderr=err,env={**os.environ,'RUST_BACKTRACE':'full'})
                while process.poll() is None:
                    try:
                        status=Path(f'/proc/{process.pid}/status').read_text().splitlines()
                        swap=max(swap,next((int(x.split()[1]) for x in status if x.startswith('VmSwap:')),0))
                    except (FileNotFoundError,ProcessLookupError): pass
                    time.sleep(.025)
            assert process.returncode==0,folder
            model=json.loads((folder/'model.json').read_text())
            if reference is None: reference=model
            assert model==reference,folder
            checkpoints=[]
            for line in (folder/'stderr.log').read_text().splitlines():
                if line.startswith('B1_LAYOUT '):
                    _,stage,words,directory_bytes,rss,hwm=line.split()
                    checkpoints.append(dict(stage=stage,words=int(words),directory_bytes=int(directory_bytes),rss_kib=int(rss),hwm_kib=int(hwm)))
            assert [x['stage'] for x in checkpoints]==['measured','prefixed'],folder
            assert swap==0,folder
            row=dict(case=case,arm=arm,block=block,warmup=block==0,command=command,checkpoints=checkpoints,model_equal=True,max_swap_kib=swap,output=json.loads((folder/'stdout.log').read_text()))
            rows.append(row);print(json.dumps(row),flush=True)
            (here/'layout-probe-runs.json').write_text(json.dumps(rows,indent=2)+'\n')
