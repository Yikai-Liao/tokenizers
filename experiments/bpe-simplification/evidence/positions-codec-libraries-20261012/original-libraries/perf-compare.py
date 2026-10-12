from pathlib import Path
import subprocess,json,gzip,os,time,re,hashlib
r=Path(__file__).resolve().parent
assert sum(x['case']=='zh-bytelevel_regex-w1' for x in json.loads((r/'normal.json').read_text()))==8
rows=[]
for repeat,arms in enumerate([['hand','gimli','fmt-slice','fmt-array']]):
 for arm in arms:
  run=r/'perf'/f'zh-bytelevel_regex-w1-r{repeat}-{arm}';run.mkdir(parents=True,exist_ok=False)
  ref=r/'normal'/f'zh-bytelevel_regex-w1-r0-{arm}';job=json.loads((ref/'job.json').read_text())|dict(attempt_id=run.name,output=str(run/'model.json'))
  (run/'job.json').write_text(json.dumps(job,indent=2)+'\n')
  command=['taskset','-c','0-1','perf','stat','-x',';','-o',str(run/'stat.csv'),'-e','instructions:u,branches:u,branch-misses:u,cache-misses:u,task-clock','--','perf','record','--no-buildid-cache','-q','-o',str(run/'perf.data'),'-e','cpu-clock','-F','199','--call-graph','dwarf,4096','--',str(r/(arm+'-bin')),str(run/'job.json')]
  (run/'command.json').write_text(json.dumps(command,indent=2)+'\n');print('start',run.name,flush=True)
  rss=swap=0
  with (run/'stdout.log').open('w') as out,(run/'stderr.log').open('w') as err:
   p=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True)
   while p.poll() is None:
    # Observe all task processes; terminate the process group only on our guard.
    available=int(re.search(r'^MemAvailable:\s+(\d+)',Path('/proc/meminfo').read_text(),re.M)[1])
    if available<2_000_000:
     os.killpg(p.pid,15);p.wait();raise RuntimeError('memory guard interrupted perf')
    time.sleep(.1)
  assert p.returncode==0,(run,p.returncode,(run/'stderr.log').read_text()[-1000:])
  result=json.loads((run/'stdout.log').read_text());model=(run/'model.json').read_bytes();assert model==gzip.decompress((ref/'model.json.gz').read_bytes());(run/'model.json.gz').write_bytes(gzip.compress(model,mtime=0));(run/'model.json').unlink()
  for name,extra in [('self-report.txt',['--no-children','--sort','symbol','--percent-limit','.3']),('callgraph-report.txt',['--children','--sort','symbol','--percent-limit','1'])]:
   with (run/name).open('w') as out:subprocess.run(['perf','report','-i',str(run/'perf.data'),'--stdio','--demangle','--no-show-nr-samples',*extra],stdout=out,stderr=subprocess.STDOUT,check=True)
  row=dict(arm=arm,repeat=repeat,case='zh-bytelevel_regex-w1',diagnostic_only=True,command=command,metrics=result['metrics'],affinity=result['effective_affinity'],model_sha256=hashlib.sha256(model).hexdigest());rows.append(row);(r/'perf-summary.json').write_text(json.dumps(rows,indent=2)+'\n');print('done',run.name,result['metrics'],flush=True)
print('Perf comparisons finished; all models identical',flush=True)
