from pathlib import Path
import subprocess,os,json,time,re,hashlib,resource,argparse,gzip
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--suite',default='pilot');args=parser.parse_args()
rows=[];expected={}
def vm():
 return {k:int(v) for k,v in (line.split() for line in Path('/proc/vmstat').read_text().splitlines()) if k in ('pswpin','pswpout')}
def sample(case,arm,job,repeat=0):
 run=root/args.suite/f'{case}-r{repeat}-{arm}';run.mkdir(parents=True,exist_ok=False)
 job=job|dict(attempt_id=run.name,build_id=arm,output=str(run/'model.json'))
 (run/'job.json').write_text(json.dumps(job,indent=2)+'\n')
 before=vm();usage=resource.getrusage(resource.RUSAGE_CHILDREN);started=time.monotonic();swap=rss=train_rss=load_rss=0
 with (run/'stdout.log').open('w') as out,(run/'stderr.log').open('w') as err:
  env=os.environ | ({'BPE_DIAGNOSTIC_PARALLEL_DROP':'0' if arm.endswith('-serial') else '1'} if arm.startswith('cleanup-') else {})
  proc=subprocess.Popen(['taskset','-c','0-3',str(root/f'{arm}-bin'),str(run/'job.json')],stdout=out,stderr=err,env=env)
  while proc.poll() is None:
   try:
    status=(Path('/proc')/str(proc.pid)/'status').read_text()
    for field in ['VmSwap','VmRSS']:
     found=re.search(r'^'+field+r':\s+(\d+)',status,re.M)
     if found:
      if field=='VmSwap':swap=max(swap,int(found[1]))
      else:
       rss=max(rss,int(found[1]))
       log=(run/'stderr.log').read_text()
       if 'BPE_SELECTION_SECONDS' in log:train_rss=max(train_rss,int(found[1]))
       else:load_rss=max(load_rss,int(found[1]))
   except FileNotFoundError:pass
   time.sleep(.05)
 if proc.returncode:raise RuntimeError((run,proc.returncode,(run/'stderr.log').read_text()))
 result=json.loads((run/'stdout.log').read_text());stderr=(run/'stderr.log').read_text()
 phases={name.lower():sum(float(v) for v in re.findall('BPE_'+name+r'_SECONDS ([\d.]+)',stderr)) for name in ['SELECTION','VOCABULARY','PLAN','INDEX','MATERIALIZE','MERGE','MERGE_RESTART','PREPARE','DROP','APPLY','COMMIT','ROUTE','OWNERS','PUBLISH','PRODUCER','OLD_COMMIT','FINAL_COMMIT','REPLAY']}
 model=(run/'model.json').read_bytes();expected.setdefault(case,model);assert expected[case]==model,(case,arm,'full model mismatch')
 after=vm();final=resource.getrusage(resource.RUSAGE_CHILDREN)
 row=dict(case=case,arm=arm,repeat=repeat,**result['metrics'],**result['output'],phases=phases,token_widths=[int(v) for v in re.findall(r'BPE_TOKEN_WIDTH (\d+)',stderr)],fresh_batches=sum(int(v) for v in re.findall(r'BPE_FRESH_BATCHES (\d+)',stderr)),reuse_batches=sum(int(v) for v in re.findall(r'BPE_REUSE_BATCHES (\d+)',stderr)),external_wall_seconds=time.monotonic()-started,child_max_swap_kib=swap,child_sampled_rss_kib=rss,train_sampled_rss_kib=train_rss,load_sampled_rss_kib=load_rss,window_reports=[json.loads(v) for v in re.findall(r'BPE_WINDOW_DIAGNOSTIC (.+)',stderr)],window_work_reports=[json.loads(v) for v in re.findall(r'BPE_WINDOW_WORK (.+)',stderr)],encoding_reports=[json.loads(v) for v in re.findall(r'BPE_ENCODING_DIAGNOSTIC (.+)',stderr)],publisher_reports=[json.loads(v) for v in re.findall(r'BPE_PUBLISHER_DIAGNOSTIC (.+)',stderr)],producer_reports=[json.loads(v) for v in re.findall(r'BPE_PRODUCER_DIAGNOSTIC (.+)',stderr)],producer_joins=sum(int(v) for v in re.findall(r'BPE_PRODUCER_JOINS (\d+)',stderr)),directory_reports=[json.loads(v) for v in re.findall(r'BPE_DIRECTORIES (.+)',stderr)],allocation_reports=[json.loads(v) for v in re.findall(r'BPE_ALLOCATIONS (.+)',stderr)],child_major_faults=final.ru_majflt-usage.ru_majflt,external_cpu_seconds=(final.ru_utime+final.ru_stime)-(usage.ru_utime+usage.ru_stime),system_swapin_pages=after['pswpin']-before['pswpin'],system_swapout_pages=after['pswpout']-before['pswpout'],model_sha256=hashlib.sha256(model).hexdigest())
 # Archive only after the child exits and all timed metrics have been captured.
 zipped=run/'model.json.gz';zipped.write_bytes(gzip.compress(model,mtime=0));assert gzip.decompress(zipped.read_bytes())==model;(run/'model.json').unlink()
 rows.append(row);(root/f'{args.suite}.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
def job(language,kind,workers):
 path=inputs[(language,kind)]
 return dict(protocol_version=1,input_id=language+'256',mode='core',input=str(path),workers=workers,pretokenizer=kind,trainer=dict(vocab_size=50000,min_frequency=2,prefix=None,suffix=None,max_token_length=None,limit_alphabet=None))
inputs={('zh','whitespace'):Path('/root/code/tokenizers-bpe-benchmarks/.bench/main-yttm-inspired/inputs/zh256-words/words.json'),('en','whitespace'):Path('/root/code/tokenizers-workspaces/bpe-affix-verification-results-20261011/en-prepared-words.json')}
for lang in ['zh','en']:inputs[(lang,'bytelevel_regex')]=Path('/root/code/tokenizers-workspaces/bpe-positions-child-results-20261012/inputs')/f'{lang}-bytelevel.json'

if args.suite in ['primary','reverse','single','single-reverse','diagnostic']:
 workers=1 if args.suite.startswith('single') else 4
 for lang in ['zh','en']:
  for kind in ['whitespace','bytelevel_regex']:
   arms=['eager','single','frontier']
   if (kind=='bytelevel_regex') != ('reverse' in args.suite):arms.reverse()
   if args.suite=='diagnostic':arms=[arm+'-diagnostic' for arm in arms]
   for arm in arms:sample(f'{lang}-{kind}-w{workers}',arm,job(lang,kind,workers),1 if 'reverse' in args.suite else 0)
elif args.suite in ['controls','diagnostic-controls']:
 for case in ['aa','reuse']:
  control=json.loads((Path('/root/code/tokenizers-workspaces/bpe-selected-rules-results-20261011/controls')/f'{case}-baseline/job.json').read_text())
  for arm in (['eager-diagnostic','single-diagnostic','frontier-diagnostic'] if args.suite=='diagnostic-controls' else ['eager','single','frontier']):sample(case,arm,control)
else:raise RuntimeError(args.suite)
