from pathlib import Path
import subprocess,os,json,time,re,hashlib,resource,gzip
r=Path(__file__).resolve().parent;rows=[]
cases=[(case,['hand','fmt-zero-push','fmt-small']) for case in ['en-whitespace-w6','en-bytelevel_regex-w6','zh-bytelevel_regex-w1']]
for case,arms in cases:
 for repeat in range(2):
  order=arms if repeat==0 else arms[::-1]
  for arm in order:
   run=r/'normal'/f'{case}-r{repeat}-{arm}';run.mkdir(parents=True,exist_ok=False);ref=Path('/tmp/bpe-positions-encapsulation-final-results-20261012')/'normal'/f'{case}-r0-a';job=json.loads((ref/'job.json').read_text())|dict(attempt_id=run.name,build_id=arm,output=str(run/'model.json'));(run/'job.json').write_text(json.dumps(job,indent=2)+'\n');print('start',run.name,flush=True)
   before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic();rss=swap=0
   with (run/'stdout.log').open('w') as out,(run/'stderr.log').open('w') as err:
    p=subprocess.Popen(['taskset','-c','0-1',str(r/(arm+'-bin')),str(run/'job.json')],stdout=out,stderr=err)
    while p.poll() is None:
     try:
      status=(Path('/proc')/str(p.pid)/'status').read_text()
      for key in ['VmRSS','VmSwap']:
       m=re.search(r'^'+key+r':\s+(\d+)',status,re.M)
       if m:
        if key=='VmRSS':rss=max(rss,int(m[1]))
        else:swap=max(swap,int(m[1]))
      available=int(re.search(r'^MemAvailable:\s+(\d+)',Path('/proc/meminfo').read_text(),re.M)[1])
      if rss>4_500_000 or available<2_000_000:p.terminate();p.wait();raise RuntimeError('memory guard stopped task')
     except FileNotFoundError:pass
     time.sleep(.05)
   elapsed=time.monotonic()-start;after=resource.getrusage(resource.RUSAGE_CHILDREN);assert p.returncode==0,(run.name,p.returncode,(run/'stderr.log').read_text()[-1000:]);result=json.loads((run/'stdout.log').read_text());model=(run/'model.json').read_bytes();assert model==gzip.decompress((ref/'model.json.gz').read_bytes());(run/'model.json.gz').write_bytes(gzip.compress(model,mtime=0));(run/'model.json').unlink()
   row=dict(case=case,arm=arm,repeat=repeat,**result['metrics'],affinity=result['effective_affinity'],child_sampled_rss_kib=rss,child_max_swap_kib=swap,external_wall_seconds=elapsed,external_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,model_sha256=hashlib.sha256(model).hexdigest());rows.append(row);(r/'normal.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps({k:row[k] for k in ['case','arm','repeat','train_seconds','train_cpu_seconds','child_sampled_rss_kib']}),flush=True)
print('Library comparisons complete',flush=True)
