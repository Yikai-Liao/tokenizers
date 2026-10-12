from pathlib import Path
import subprocess,json,gzip,re,time,selectors,hashlib
r=Path(__file__).resolve().parent;rows=[]
for repeat,order in enumerate([['b','h','i'],['i','h','b']]):
 for arm in order:
  run=r/'perf-interval'/f'zh-bytelevel_regex-w1-r{repeat}-{arm}';run.mkdir(parents=True,exist_ok=False);ref=r/'normal/zh-bytelevel_regex-w1-r0-a';job=json.loads((ref/'job.json').read_text())|dict(attempt_id=run.name,build_id=arm,output=str(run/'model.json'));(run/'job.json').write_text(json.dumps(job,indent=2)+'\n')
  command=['taskset','-c','0-1','perf','stat','-I','50','-x',';','-o',str(run/'stat.csv'),'-e','instructions:u,branches:u,branch-misses:u,cache-misses:u,task-clock','--',str(r/(arm+'-bin')),str(run/'job.json')]
  (run/'command.json').write_text(json.dumps(command,indent=2)+'\n');print('start',run.name,flush=True);markers=[];start=time.monotonic()
  with (run/'stdout.log').open('w') as out,(run/'stderr.log').open('w') as err:
   p=subprocess.Popen(command,stdout=out,stderr=subprocess.PIPE,text=True,bufsize=1)
   for line in p.stderr:
    elapsed=time.monotonic()-start;err.write(line);err.flush()
    if line.startswith('BPE_'):markers.append(dict(elapsed=elapsed,line=line.strip()))
   p.wait()
  assert p.returncode==0,(run.name,p.returncode);(run/'markers.json').write_text(json.dumps(markers,indent=2)+'\n')
  result=json.loads((run/'stdout.log').read_text());model=(run/'model.json').read_bytes();assert model==gzip.decompress((ref/'model.json.gz').read_bytes());(run/'model.json.gz').write_bytes(gzip.compress(model,mtime=0));(run/'model.json').unlink()
  counters={};events=[]
  for line in (run/'stat.csv').read_text().splitlines():
   if not line or line.startswith('#'):continue
   parts=line.split(';')
   try:elapsed=float(parts[0]);value=float(parts[1]);event=parts[3]
   except (ValueError,IndexError):continue
   events.append((elapsed,event,value));counters[event]=counters.get(event,0)+value
  phases={}
  for marker in markers:
   m=re.match(r'BPE_(\w+)_SECONDS (\S+)',marker['line'])
   if not m:continue
   name,seconds=m[1],float(m[2]);end=marker['elapsed'];begin=end-seconds
   if name not in ['SELECTION','VOCABULARY','PLAN','INITIAL_COUNT','INITIAL_OWNER','INDEX','MATERIALIZE','MERGE','DROP']:continue
   # Exclude the boundary intervals rather than attribute them across phases.
   values={};times=[]
   for elapsed,event,value in events:
    if elapsed-.05>=begin and elapsed<=end:
     values[event]=values.get(event,0)+value;times.append(elapsed)
   phases[name]=dict(wall_seconds=seconds,marker_end=end,covered_seconds=(max(times)-min(times)+.05 if times else 0),interior_interval_counters=values)
  row=dict(case='zh-bytelevel_regex-w1',arm=arm,repeat=repeat,diagnostic_only=True,command=command,counters=counters,phases=phases,metrics=result['metrics'],model_sha256=hashlib.sha256(model).hexdigest(),binary_sha256=hashlib.sha256((r/(arm+'-bin')).read_bytes()).hexdigest());rows.append(row);(r/'perf-interval-summary.json').write_text(json.dumps(rows,indent=2)+'\n');print('done',run.name,counters,phases.get('INITIAL_COUNT'),flush=True)
print('Finished phase instruction measurements',flush=True)
