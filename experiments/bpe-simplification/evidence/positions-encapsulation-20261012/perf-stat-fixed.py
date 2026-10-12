from pathlib import Path
import subprocess,json,gzip,re,time
r=Path(__file__).resolve().parent
assert len(json.loads((r/'overhead.json').read_text()))==38
rows=[]
for case,orders in [('en-whitespace-w6',[['a','b','e','f','h'],['h','f','e','b','a']]),('zh-bytelevel_regex-w1',[['b','h']])]:
 for repeat,order in enumerate(orders):
  for arm in order:
   run=r/'perf-stat'/f'{case}-r{repeat}-{arm}';run.mkdir(parents=True,exist_ok=False);ref=r/'normal'/f'{case}-r0-a';job=json.loads((ref/'job.json').read_text())|dict(attempt_id=run.name,build_id=arm,output=str(run/'model.json'));(run/'job.json').write_text(json.dumps(job,indent=2)+'\n');print('start',run.name,flush=True)
   command=['taskset','-c','0-1','perf','stat','-x',';','-o',str(run/'stat.csv'),'-e','instructions:u,branches:u,branch-misses:u,cache-misses:u,task-clock','--',str(r/(arm+'-bin')),str(run/'job.json')];(run/'command.json').write_text(json.dumps(command,indent=2)+'\n')
   with (run/'stdout.log').open('w') as out,(run/'stderr.log').open('w') as err:p=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True)
   while p.poll() is None:
    available=int(re.search(r'^MemAvailable:\s+(\d+)',Path('/proc/meminfo').read_text(),re.M)[1])
    if available<2_000_000:
     import os,signal
     os.killpg(p.pid,signal.SIGTERM);p.wait();raise RuntimeError('task memory guard interrupted perf stat')
    time.sleep(.1)
   assert p.returncode==0,(run.name,p.returncode);result=json.loads((run/'stdout.log').read_text());model=(run/'model.json').read_bytes();assert model==gzip.decompress((ref/'model.json.gz').read_bytes());(run/'model.json.gz').write_bytes(gzip.compress(model,mtime=0));(run/'model.json').unlink()
   counters={}
   for line in (run/'stat.csv').read_text().splitlines():
    if not line or line.startswith('#'):continue
    parts=line.split(';');counters[parts[2]]=float(parts[0])
   row=dict(case=case,arm=arm,repeat=repeat,diagnostic_only=True,command=command,counters=counters,metrics=result['metrics']);rows.append(row);(r/'perf-stat-summary.json').write_text(json.dumps(rows,indent=2)+'\n');print('done',run.name,counters,flush=True)
print('Perf stat fixed comparisons finished',flush=True)
