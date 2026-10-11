import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent; out=root/os.environ.get('BPE_FEED_PROBE_OUTPUT','probes');out.mkdir(exist_ok=True)
build=json.loads((root/'builds.json').read_text());inputs=json.loads((root/'inputs.json').read_text()); rows=[]
cases=[(inputs[0],6)]+[(next(x for x in inputs if x['case']==f'zh{size}-bytelevel_regex'),1) for size in (128,384,512)]
for item,blocks in cases:
 for block in range(blocks):
  arms=os.environ.get('BPE_FEED_PROBE_ARMS','baseline,unified').split(',');arms=arms if block%2==0 else arms[::-1]
  for arm in arms:
   name=item['case']+'-b'+str(block)+'-'+arm; dest=out/name;dest.mkdir()
   binary=build['builds']['probe-'+arm]['binary'];assert hashlib.sha256(Path(binary).read_bytes()).hexdigest()==build['builds']['probe-'+arm]['binary_sha256']
   job=dict(protocol_version=1,attempt_id=name,build_id='probe-'+arm,input_id=item['case'],mode='pipeline',input=item['source'],output=str(dest/'unused-model.json'),workers=4,pretokenizer='bytelevel_regex',trainer=dict(vocab_size=50000,min_frequency=2,prefix=None,suffix=None,max_token_length=None));(dest/'job.json').write_text(json.dumps(job,indent=2))
   env=dict(os.environ,RAYON_NUM_THREADS='4',LC_ALL='C',BPE_FEED_COST_PROBE='1');env.pop('BPE_FEED_COST_SEED',None)
   command=['taskset','-c','0-3',binary,str(dest/'job.json')];swap=rss=0
   with (dest/'stdout.log').open('w') as stdout,(dest/'stderr.log').open('w') as stderr:
    process=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env)
    while process.poll() is None:
     try:
      fields=dict((s.split(':')[0],s.split()[1]) for s in Path(f'/proc/{process.pid}/status').read_text().splitlines() if s.startswith(('VmSwap:','VmRSS:')));swap=max(swap,int(fields.get('VmSwap',0)));rss=max(rss,int(fields.get('VmRSS',0)))
     except (FileNotFoundError,ProcessLookupError):pass
     time.sleep(.025)
   assert process.returncode==0 and swap==0
   result=json.loads((dest/'stdout.log').read_text());phases=[json.loads(s.removeprefix('BPE_FEED_TABLE ')) for s in (dest/'stderr.log').read_text().splitlines() if s.startswith('BPE_FEED_TABLE ')]
   row=dict(run=name,case=item['case'],block=block,arm=arm,command=command,returncode=process.returncode,max_swap_kib=swap,sampled_max_rss_kib=rss,metrics=result['metrics'],table_phases=phases,diagnostic=result['diagnostic']);rows.append(row);(out/'runs.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
