"""Serial paired public feed + train measurement; canonical full-count and model validation."""
import codecs,hashlib,json,os,statistics,subprocess,time
from pathlib import Path
EVIDENCE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('BPE_FEED_OUTPUT',str(EVIDENCE/'measurements')));ROOT.mkdir(exist_ok=True,parents=True)
ARMS=os.environ.get('BPE_FEED_ARMS','baseline,unified,collected').split(',')
BLOCKS=int(os.environ.get('BPE_FEED_BLOCKS','3'))
BUILD=json.loads((EVIDENCE/'builds.json').read_text())
INPUTS=json.loads((EVIDENCE/'inputs.json').read_text())
CASES=[dict(x,workers=w,scenario=x['case']+'-w'+str(w)) for x in INPUTS for w in [1,4]]
if os.environ.get('BPE_FEED_CASE'): CASES=[c for c in CASES if c['scenario'] in os.environ['BPE_FEED_CASE'].split(',')]
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()
for arm in ARMS: assert digest(BUILD['builds'][arm]['binary'])==BUILD['builds'][arm]['binary_sha256']
for item in INPUTS:
 assert digest(item['source'])==item['source_sha256']
 assert Path(item['source']).stat().st_size==item['source_bytes']
 decoder=codecs.getincrementaldecoder('utf-8')('strict')
 with Path(item['source']).open('rb') as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b''):decoder.decode(chunk)
 decoder.decode(b'',final=True)
manifest=dict(builds=BUILD,inputs=INPUTS,workers=[1,4],arms=ARMS,blocks=BLOCKS,warmup_blocks=0,
 cpu=subprocess.check_output(['lscpu'],text=True),rustc=subprocess.check_output(['rustc','-Vv'],text=True),
 timing='public streaming feed and train_vocab, excludes count export/model serialization; feed_hwm is process high water through feed, pipeline_hwm is process high water before validation',
 cache='source SHA256 and UTF-8 validation read corpora before matrix and warm ordinary OS cache; no eviction; each runner streams input without preload; rotate arms per paired block; no concurrent cargo/rustc/runner',
 correctness='every sample: canonical JSON-key TSV records sorted LC_ALL=C, SHA256 compare across arms/workers; complete parsed vocab/merges compare; exports outside timer',
 limitation='diagnostic only: fixed shared hash seeds per block, identical across arms; same phase trace points; paired wall includes trace overhead; swap polled every 25ms')
(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2))
p=ROOT/'runs.json';records=json.loads(p.read_text()) if p.exists() else []
refs={}
for r in records:
 if r['valid'] and r['case'] not in refs:
  refs[r['case']]={'counts':r['counts_sha256'],'model':json.loads((ROOT/r['run']/'model.json').read_text())}
for item in CASES:
 for block in range(BLOCKS):
  rotate=block%len(ARMS);order=ARMS[rotate:]+ARMS[:rotate]
  for arm in order:
   if any(r['scenario']==item['scenario'] and r['block']==block and r['arm']==arm for r in records):continue
   competitors=[]
   for e in Path('/proc').iterdir():
    if not e.name.isdigit():continue
    try:
     comm=(e/'comm').read_text().strip();exe=str((e/'exe').resolve())
     if comm in ('cargo','rustc') or comm.startswith('bpe-bench') or exe in [BUILD['builds'][a]['binary'] for a in ARMS]:competitors.append([e.name,comm])
    except (FileNotFoundError,ProcessLookupError):pass
   if competitors:raise RuntimeError('competing build/benchmark '+str(competitors))
   workers=item['workers'];name=item['scenario']+'-b'+str(block)+'-'+arm;out=ROOT/name;out.mkdir(exist_ok=False)
   job=dict(protocol_version=1,attempt_id=name,build_id=arm,input_id=item['case'],mode='pipeline',input=item['source'],output=str(out/'model.json'),workers=workers,pretokenizer=item['pretokenizer'],trainer=dict(vocab_size=50000,min_frequency=2,prefix=None,suffix=None,max_token_length=None))
   (out/'job.json').write_text(json.dumps(job,indent=2))
   env={k:v for k,v in os.environ.items() if not k.startswith(('BPE_TRACE_','TK_WORD_COUNTS_CACHE','TK_WRITE_WORD_COUNTS_CACHE','BPE_HEAP_DIAGNOSTIC'))};env['RAYON_NUM_THREADS']=str(workers);env['LC_ALL']='C';env['BPE_FEED_COST_SEED']=str(block)
   command=['taskset','-c','0' if workers==1 else '0-3',BUILD['builds'][arm]['binary'],str(out/'job.json')]
   swap=rss=0;start=time.monotonic()
   with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
    proc=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env)
    while proc.poll() is None:
     try:
      status=Path('/proc/'+str(proc.pid)+'/status').read_text().splitlines()
      field=lambda n:next((int(s.split()[1]) for s in status if s.startswith(n)),0)
      swap=max(swap,field('VmSwap:'));rss=max(rss,field('VmRSS:'))
     except (FileNotFoundError,ProcessLookupError):pass
     time.sleep(.025)
   record=dict(run=name,case=item['case'],scenario=item['scenario'],workers=workers,block=block,arm=arm,command=command,environment={'RAYON_NUM_THREADS':str(workers),'LC_ALL':'C','BPE_FEED_COST_SEED':str(block)},returncode=proc.returncode,max_swap_kib=swap,sampled_max_rss_kib=rss,subprocess_wall=time.monotonic()-start,concurrent_builds_or_benchmarks=competitors)
   if proc.returncode==0:
    record.update(json.loads((out/'stdout.log').read_text()))
    record['table_phases']=[json.loads(line.removeprefix('BPE_FEED_TABLE ')) for line in (out/'stderr.log').read_text().splitlines() if line.startswith('BPE_FEED_TABLE ')]
    assert record['metrics']['feed_workers']==workers
    counts=Path(record['output']['counts_path']);sorted_path=out/'counts.sorted.tsv'
    subprocess.run(['sort','-S','128M','-T',str(out),str(counts),'-o',str(sorted_path)],env=env,check=True)
    record['counts_sha256']=digest(sorted_path);record['count_records']=sum(1 for _ in sorted_path.open('rb'));record['model_sha256']=digest(out/'model.json')
    model=json.loads((out/'model.json').read_text())
    if item['case'] not in refs:refs[item['case']]={'counts':record['counts_sha256'],'model':model}
    ref=refs[item['case']];record['counts_equal']=record['counts_sha256']==ref['counts'];record['model_equal']=model==ref['model']
    record['valid']=record['counts_equal'] and record['model_equal'] and record['count_records']==record['metrics']['feed_unique_words'] and swap==0
    if record['valid']:counts.unlink();sorted_path.unlink()
   else:record['valid']=False
   records.append(record);p.write_text(json.dumps(records,indent=2))
   print(json.dumps({k:record.get(k) for k in ('scenario','block','arm','valid','counts_equal','model_equal','metrics')}),flush=True)
   if not record['valid']:raise RuntimeError('Invalid sample '+str(out))
keys=['feed_seconds','feed_cpu_seconds','feed_hwm_kib','feed_rss_kib','pipeline_seconds','pipeline_cpu_seconds','process_hwm_kib_before_validation','train_seconds','train_cpu_seconds','feed_major_faults','pipeline_major_faults']
summary=[]
for item in CASES:
 rows=[r for r in records if r['scenario']==item['scenario']];row={'scenario':item['scenario'],'case':item['case'],'workers':item['workers']}
 for arm in ARMS:
  ar=[r for r in rows if r['arm']==arm];row[arm]={k:statistics.median(r['metrics'][k] for r in ar) for k in keys}
  row[arm]['feed_mib_per_second']=item['source_bytes']/2**20/row[arm]['feed_seconds'];row[arm]['pipeline_mib_per_second']=item['source_bytes']/2**20/row[arm]['pipeline_seconds']
 row['paired_delta_percent']={arm:{k:statistics.median((next(r for r in rows if r['arm']==arm and r['block']==b)['metrics'][k]/next(r for r in rows if r['arm']==ARMS[0] and r['block']==b)['metrics'][k]-1)*100 for b in sorted(set(r['block'] for r in rows))) for k in keys if 'faults' not in k} for arm in ARMS[1:]}
 summary.append(row)
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
