from pathlib import Path
import os,subprocess,json,time,gzip,hashlib
r=Path(__file__).resolve().parent
inputs=json.loads((r/'inputs.json').read_text())
label='perf-history-buffer';root=r/'runs'/label;root.mkdir(exist_ok=True)
for repeat,arms in enumerate([['main-perf','history-buffer-perf'],['history-buffer-perf','main-perf']]):
 for arm in arms:
  d=root/f'zh-bytelevel-core-w4-v100000-r{repeat}-{arm}';d.mkdir(exist_ok=False)
  job=dict(protocol_version=1,attempt_id=d.name,build_id=arm,input_id='zh-bytelevel',mode='core',input=inputs['zh-bytelevel']['core'],output=str(d/'model.json'),workers=4,pretokenizer='bytelevel_regex',trainer=dict(vocab_size=100000,min_frequency=2,prefix=None,suffix=None,max_token_length=None))
  (d/'job.json').write_text(json.dumps(job,indent=2))
  for name in ['control','ack']:os.mkfifo(d/name)
  fds=[os.open(d/name,os.O_RDWR) for name in ['control','ack']]
  env=dict(os.environ,RAYON_NUM_THREADS='4',TOKENIZERS_PARALLELISM='true',BPE_PERF_CONTROL=str(d/'control'),BPE_PERF_ACK=str(d/'ack'))
  cmd=['perf','record','-F','199','-e','cycles:u','--call-graph','dwarf,16384','--sample-cpu','--timestamp-boundary','-D','-1',f'--control=fifo:{d}/control,{d}/ack','-o',str(d/'perf.data'),'--','taskset','-c','0-3',str(r/arm),str(d/'job.json')]
  (d/'command.json').write_text(json.dumps(cmd))
  with (d/'stdout.log').open('w') as out,(d/'stderr.log').open('w') as err:subprocess.run(cmd,stdout=out,stderr=err,env=env,check=True)
  for fd in fds:os.close(fd)
  with gzip.open(r/'reference/zh-bytelevel-core-v100000.json.gz','rt') as f:expected=json.load(f)
  actual=json.loads((d/'model.json').read_text());assert actual==expected
  result=dict(arm=arm,repeat=repeat,model_equal=True,binary_sha256=hashlib.sha256((r/arm).read_bytes()).hexdigest(),**json.loads((d/'stdout.log').read_text()))
  (d/'result.json').write_text(json.dumps(result,indent=2));(d/'model.json').unlink()
  with (r/(label+'.jsonl')).open('a') as f:f.write(json.dumps(result)+'\n')
  with (d/'report-self.txt').open('w') as out:subprocess.run(['perf','report','--stdio','--call-graph','none','--no-inline','--no-children','--percent-limit','0.3','--sort','symbol','-i',str(d/'perf.data')],stdout=out,check=True)
  with (d/'report-children.txt').open('w') as out:subprocess.run(['perf','report','--stdio','--call-graph','none','--no-inline','--children','--percent-limit','0.5','--sort','symbol','-i',str(d/'perf.data')],stdout=out,check=True)
  print(json.dumps(result),flush=True)
