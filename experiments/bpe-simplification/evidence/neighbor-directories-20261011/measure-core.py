"""Serial four-arm do_train comparison on fixed real-corpus word maps."""
import hashlib,json,os,statistics,subprocess,time
from pathlib import Path
r=Path(__file__).resolve().parent
out=r/'core';out.mkdir(exist_ok=True)
build=json.loads((r/'builds.json').read_text());inputs=json.loads((r/'inputs.json').read_text())
arms=list(build['builds']);blocks=2
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
for arm in arms:assert digest(build['builds'][arm]['binary'])==build['builds'][arm]['binary_sha256']
for item in inputs:
 assert digest(item['prepared_path'])==item['prepared_sha256']
 assert Path(item['prepared_path']).stat().st_size==item['prepared_bytes']
manifest=dict(builds=build,inputs=inputs,arms=arms,blocks=blocks,workers=[4],
 warmup_blocks=0,cpu=subprocess.check_output(['lscpu'],text=True),rustc=subprocess.check_output(['rustc','-Vv'],text=True),
 timing='public do_train wall and process CPU; map loading and model serialization excluded; HWM remains cumulative including load',
 cache='prepared SHA256 verification reads all input before measurements; ordinary OS cache, no eviction; fresh process each sample',
 order='block 0 baseline/directories/insert/combined; block 1 combined/insert/directories/baseline',
 hashing='fixed [11,13,17,19] caller map seeds; production internal default hashers unchanged',
 correctness='complete parsed vocabulary IDs and ordered merges compared across every arm/block; native tests separately compare per-rule traces at 1/4/8 workers',
 diagnostics='child VmSwap and VmRSS sampled every 25ms; major faults measured across public train; no concurrent builds or benchmarks')
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
records=[];refs={}
for item in inputs:
 for block in range(blocks):
  order=arms if block==0 else arms[::-1]
  for arm in order:
   competitors=[]
   for e in Path('/proc').iterdir():
    if not e.name.isdigit():continue
    try:
     comm=(e/'comm').read_text().strip()
     if comm in ('cargo','rustc') or comm.startswith('bpe-bench'):competitors.append([e.name,comm])
    except (FileNotFoundError,ProcessLookupError):pass
   assert not competitors,competitors
   name=item['case']+'-b'+str(block)+'-'+arm;p=out/name;p.mkdir(exist_ok=False)
   job=dict(protocol_version=1,attempt_id=name,build_id=arm,input_id=item['case'],mode='core',input=item['prepared_path'],output=str(p/'model.json'),workers=4,pretokenizer=item['pretokenizer'],trainer=dict(vocab_size=50000,min_frequency=2,prefix=None,suffix=None,max_token_length=None))
   (p/'job.json').write_text(json.dumps(job,indent=2))
   env={k:v for k,v in os.environ.items() if not k.startswith(('BPE_TRACE_','TK_WORD_COUNTS_CACHE','TK_WRITE_WORD_COUNTS_CACHE','BPE_HEAP_DIAGNOSTIC'))}
   env['RAYON_NUM_THREADS']='4';env['LC_ALL']='C';env['RUST_BACKTRACE']='full'
   command=['taskset','-c','0-3',build['builds'][arm]['binary'],str(p/'job.json')]
   swap=rss=0;begin=time.monotonic()
   with (p/'stdout.log').open('w') as stdout,(p/'stderr.log').open('w') as stderr:
    child=subprocess.Popen(command,env=env,stdout=stdout,stderr=stderr)
    while child.poll() is None:
     try:
      status=Path('/proc/'+str(child.pid)+'/status').read_text().splitlines()
      field=lambda key:next((int(s.split()[1]) for s in status if s.startswith(key)),0)
      swap=max(swap,field('VmSwap:'));rss=max(rss,field('VmRSS:'))
     except (FileNotFoundError,ProcessLookupError):pass
     time.sleep(.025)
   rec=dict(run=name,case=item['case'],block=block,arm=arm,command=command,returncode=child.returncode,max_swap_kib=swap,sampled_max_rss_kib=rss,subprocess_wall=time.monotonic()-begin,concurrent_builds_or_benchmarks=competitors)
   if child.returncode==0:
    rec.update(json.loads((p/'stdout.log').read_text()));model=json.loads((p/'model.json').read_text())
    if item['case'] not in refs:refs[item['case']]=model
    rec['model_sha256']=digest(p/'model.json');rec['model_equal']=model==refs[item['case']]
    rec['valid']=rec['model_equal'] and swap==0 and rec['metrics']['train_major_faults']==0
   else:rec['valid']=False
   records.append(rec);(out/'runs.json').write_text(json.dumps(records,indent=2))
   print(json.dumps({k:rec.get(k) for k in ('run','valid','metrics')}),flush=True)
   if not rec['valid']:raise RuntimeError('failed attempt retained '+name)
keys=['train_seconds','train_cpu_seconds','process_hwm_kib_before_validation']
summary=[]
for item in inputs:
 rows=[row for row in records if row['case']==item['case']]
 row={'case':item['case']}
 for arm in arms:
  row[arm]={k:statistics.median(x['metrics'][k] for x in rows if x['arm']==arm) for k in keys}
 row['paired_delta_percent']={arm:{k:statistics.median((next(x for x in rows if x['arm']==arm and x['block']==b)['metrics'][k]/next(x for x in rows if x['arm']=='baseline' and x['block']==b)['metrics'][k]-1)*100 for b in range(blocks)) for k in keys} for arm in arms[1:]}
 summary.append(row)
(out/'summary.json').write_text(json.dumps(summary,indent=2))
