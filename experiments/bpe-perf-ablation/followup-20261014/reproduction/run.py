import argparse,hashlib,json,os,subprocess,time,gzip
from pathlib import Path
ROOT=Path(__file__).resolve().parent
P=argparse.ArgumentParser();P.add_argument('--arms',nargs='+',default=['main','prepool','current','enterpool']);P.add_argument('--cases',nargs='+',default=['en-bytelevel','zh-bytelevel','zh-whitespace']);P.add_argument('--workers',type=int,nargs='+',default=[1,4]);P.add_argument('--vocabs',type=int,nargs='+',default=[100000]);P.add_argument('--rounds',type=int,default=1);P.add_argument('--modes',nargs='+',default=['pipeline']);P.add_argument('--label',default='screen');a=P.parse_args()
INPUTS=json.loads((ROOT/'inputs.json').read_text());env={k:v for k,v in os.environ.items() if k in ['PATH','HOME','USER','LANG','LC_ALL','TMPDIR']};env['TOKENIZERS_PARALLELISM']='true'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def cpu():return list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]))
for r in range(a.rounds):
 for case in a.cases:
  for w in a.workers:
   for v in a.vocabs:
    for mode in a.modes:
     for arm in (a.arms if r%2==0 else list(reversed(a.arms))):
      d=ROOT/'runs'/a.label/f'{case}-{mode}-w{w}-v{v}-r{r}-{arm}';d.mkdir(parents=True,exist_ok=False);definition=INPUTS[case];job=dict(protocol_version=1,attempt_id=d.name,build_id=arm,input_id=case,mode=mode,input=definition[mode],output=str(d/'model.json'),workers=w,pretokenizer=definition['pretokenizer'],trainer=dict(vocab_size=v,min_frequency=2,prefix=None,suffix=None,max_token_length=None));(d/'job.json').write_text(json.dumps(job,indent=2));cmd=['taskset','-c',f'0-{w-1}',str(ROOT/arm),str(d/'job.json')];before=cpu();t=time.monotonic();swap=0;threads=0;active={};samples=[]
      with (d/'stdout.log').open('w') as out,(d/'stderr.log').open('w') as err:
       proc=subprocess.Popen(cmd,stdout=out,stderr=err,env=dict(env,RAYON_NUM_THREADS=str(w)))
       while proc.poll() is None:
        try:
         status=Path(f'/proc/{proc.pid}/status').read_text();swap=max(swap,next((int(x.split()[1]) for x in status.splitlines() if x.startswith('VmSwap:')),0));threads=max(threads,next(int(x.split()[1]) for x in status.splitlines() if x.startswith('Threads:')))
         times={}
         for p in Path(f'/proc/{proc.pid}/task').glob('*/stat'):
          s=p.read_text().rsplit(')',1)[1].split();times[p.parent.name]=int(s[11])+int(s[12]);active[p.parent.name]=max(active.get(p.parent.name,0),times[p.parent.name])
         if len(samples)<1200:samples.append({'elapsed':time.monotonic()-t,'ticks':times})
        except (FileNotFoundError,ProcessLookupError):pass
        time.sleep(.03)
      after=cpu();record=dict(arm=arm,case=case,mode=mode,workers=w,vocab=v,repeat=r,command=cmd,returncode=proc.returncode,elapsed=time.monotonic()-t,max_swap_kib=swap,max_threads=threads,thread_cpu_ticks=active,cpu_delta=[x-y for x,y in zip(after,before)],binary_sha256=sha(ROOT/arm))
      if proc.returncode: print((d/'stderr.log').read_text());raise RuntimeError(d)
      record.update(json.loads((d/'stdout.log').read_text()));actual=json.loads((d/'model.json').read_text());ref=ROOT/'reference'/f'{case}-{mode}-v{v}.json.gz';ref.parent.mkdir(exist_ok=True)
      if not ref.exists():
       assert arm in ['main','main-phases']
       with gzip.open(ref,'wt') as f:json.dump(actual,f,ensure_ascii=False)
      with gzip.open(ref,'rt') as f:expected=json.load(f)
      record['model_equal']=actual==expected;record['model_sha256']=sha(d/'model.json');(d/'model.json').unlink();(d/'samples.json').write_text(json.dumps(samples));(d/'result.json').write_text(json.dumps(record,indent=2));
      with (ROOT/(a.label+'.jsonl')).open('a') as f:f.write(json.dumps(record)+'\n')
      print(json.dumps({k:record[k] for k in ['arm','case','mode','workers','vocab','repeat','metrics','workers_effective','max_swap_kib','model_equal']}),flush=True);assert record['model_equal'] and swap==0
