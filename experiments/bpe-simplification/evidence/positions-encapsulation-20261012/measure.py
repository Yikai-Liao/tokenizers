from pathlib import Path
import subprocess,os,json,time,re,hashlib,resource,gzip,sys
r=Path(__file__).resolve().parent;old=Path('/root/code/tokenizers-workspaces/bpe-positions-simd-results-20261012')
rows=json.loads((r/'normal.json').read_text()) if (r/'normal.json').exists() else [];done={(x['case'],x['arm'],x['repeat']) for x in rows}
cases=[('zh','whitespace'),('zh','bytelevel_regex'),('en','whitespace'),('en','bytelevel_regex')]
# Worker count comparisons share the same two-CPU ceiling. No builds overlap timing.
for workers in [1,6]:
 for number,(lang,kind) in enumerate(cases):
  case=f'{lang}-{kind}-w{workers}';reference=old/'distribution'/f'{case}-r0-distribution';template=json.loads((reference/'job.json').read_text());expected=gzip.decompress((reference/'model.json.gz').read_bytes())
  for repeat in range(2):
   arms=['base','a','b'];shift=number%3;arms=arms[shift:]+arms[:shift]
   if repeat:arms.reverse()
   for arm in arms:
    if (case,arm,repeat) in done:continue
    run=r/'normal'/f'{case}-r{repeat}-{arm}';run.mkdir(parents=True,exist_ok=False)
    job=template|dict(attempt_id=run.name,build_id=arm,output=str(run/'model.json'))
    (run/'job.json').write_text(json.dumps(job,indent=2)+'\n');print('start',run.name,flush=True)
    before=resource.getrusage(resource.RUSAGE_CHILDREN);started=time.monotonic();rss=swap=0
    with (run/'stdout.log').open('w') as out,(run/'stderr.log').open('w') as err:
     proc=subprocess.Popen(['taskset','-c','0-1',str(r/(arm+'-bin')),str(run/'job.json')],stdout=out,stderr=err)
     while proc.poll() is None:
      try:
       status=(Path('/proc')/str(proc.pid)/'status').read_text()
       for key in ['VmRSS','VmSwap']:
        m=re.search(r'^'+key+r':\s+(\d+)',status,re.M)
        if m:
         if key=='VmRSS':rss=max(rss,int(m[1]))
         else:swap=max(swap,int(m[1]))
       mem=re.search(r'^MemAvailable:\s+(\d+)',Path('/proc/meminfo').read_text(),re.M)
       if rss>4_500_000 or int(mem[1])<2_000_000:
        proc.terminate();proc.wait();raise RuntimeError('Stopped at task memory guard; timing rejected')
      except FileNotFoundError:pass
      time.sleep(.05)
    finished=time.monotonic();after=resource.getrusage(resource.RUSAGE_CHILDREN)
    if proc.returncode:raise RuntimeError((run.name,proc.returncode,(run/'stderr.log').read_text()[-1200:]))
    result=json.loads((run/'stdout.log').read_text());model=(run/'model.json').read_bytes();assert model==expected,run.name
    row=dict(case=case,arm=arm,repeat=repeat,**result['metrics'],**result['output'],affinity=result['effective_affinity'],child_sampled_rss_kib=rss,child_max_swap_kib=swap,external_wall_seconds=finished-started,external_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,model_sha256=hashlib.sha256(model).hexdigest())
    (run/'model.json.gz').write_bytes(gzip.compress(model,mtime=0));(run/'model.json').unlink()
    rows.append(row);(r/'normal.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps({k:row[k] for k in ['case','arm','repeat','train_seconds','train_cpu_seconds','child_sampled_rss_kib']}),flush=True)
print('All 48 normal runs completed; full models match baseline',flush=True)
