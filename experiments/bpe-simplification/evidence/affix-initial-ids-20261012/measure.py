from pathlib import Path
import subprocess,os,json,time,re,hashlib,resource,argparse
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--suite',default='pilot');args=parser.parse_args()
rows=[];expected={}
def vm():
 return {k:int(v) for k,v in (line.split() for line in Path('/proc/vmstat').read_text().splitlines()) if k in ('pswpin','pswpout')}
def sample(case,arm,job,repeat=0):
 run=root/args.suite/f'{case}-r{repeat}-{arm}';run.mkdir(parents=True,exist_ok=False)
 job=job|dict(attempt_id=run.name,build_id=arm,output=str(run/'model.json'))
 (run/'job.json').write_text(json.dumps(job,indent=2)+'\n')
 before=vm();usage=resource.getrusage(resource.RUSAGE_CHILDREN);started=time.monotonic();swap=rss=0
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
      else:rss=max(rss,int(found[1]))
   except FileNotFoundError:pass
   time.sleep(.05)
 if proc.returncode:raise RuntimeError((run,proc.returncode,(run/'stderr.log').read_text()))
 result=json.loads((run/'stdout.log').read_text());stderr=(run/'stderr.log').read_text()
 phases={name.lower():sum(float(v) for v in re.findall('BPE_'+name+r'_SECONDS ([\d.]+)',stderr)) for name in ['SELECTION','VOCABULARY','VOCABULARY_CPU','PLAN','INDEX','MATERIALIZE','MERGE','MERGE_RESTART','DROP']}
 model=(run/'model.json').read_bytes();expected.setdefault(case,model);assert expected[case]==model,(case,arm,'full model mismatch')
 after=vm();final=resource.getrusage(resource.RUSAGE_CHILDREN)
 row=dict(case=case,arm=arm,repeat=repeat,**result['metrics'],**result['output'],phases=phases,external_wall_seconds=time.monotonic()-started,child_max_swap_kib=swap,child_sampled_rss_kib=rss,child_major_faults=final.ru_majflt-usage.ru_majflt,external_cpu_seconds=(final.ru_utime+final.ru_stime)-(usage.ru_utime+usage.ru_stime),system_swapin_pages=after['pswpin']-before['pswpin'],system_swapout_pages=after['pswpout']-before['pswpout'],model_sha256=hashlib.sha256(model).hexdigest())
 rows.append(row);(root/f'{args.suite}.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
def job(language,kind,workers):
 path=inputs[(language,kind)]
 return dict(protocol_version=1,input_id=language+'256',mode='core',input=str(path),workers=workers,pretokenizer=kind,trainer=dict(vocab_size=50000,min_frequency=2,prefix=None,suffix=None,max_token_length=None,limit_alphabet=None))
inputs={('zh','whitespace'):Path('/root/code/tokenizers-bpe-benchmarks/.bench/main-yttm-inspired/inputs/zh256-words/words.json'),('en','whitespace'):Path('/root/code/tokenizers-workspaces/bpe-affix-verification-results-20261011/en-prepared-words.json')}
for lang in ['zh','en']:inputs[(lang,'bytelevel_regex')]=root/'inputs'/f'{lang}-bytelevel.json'
items=json.loads((root/'inputs.json').read_text())
if args.suite in ['primary','reverse','flags','flags-reverse','diagnostic','cold','cold-reverse','suffix']:
 for decorated in [True,False]:
  if (args.suite.startswith('flags') or args.suite=='suffix') and not decorated:continue
  for item in items:
   lang=item['language'];case=lang+('-affix' if decorated else '-plain')+'-w4'
   arms=['baseline','candidate'] if args.suite in ['primary','reverse','diagnostic','cold','cold-reverse'] else ['candidate','flags']
   if args.suite=='suffix':arms=['baseline','candidate-suffix-first','flags']
   if args.suite in ['cold','cold-reverse']:arms=['baseline','candidate-cold']
   if args.suite=='cold' and decorated:arms.append('flags')
   if args.suite in ['primary','reverse'] and decorated:arms.append('flags')
   if lang=='en':arms.reverse()
   if args.suite in ['reverse','flags-reverse','cold-reverse']:arms.reverse()
   control=dict(protocol_version=1,input_id=lang+'256',mode='core',input=item['prepared_path'],workers=4,pretokenizer='whitespace',trainer=dict(vocab_size=50000,min_frequency=2,prefix='##' if decorated else None,suffix='</w>' if decorated else None,max_token_length=None,limit_alphabet=item['limit_alphabet']))
   for arm in arms:sample(case,arm+('-diagnostic' if args.suite=='diagnostic' else ''),control,1 if args.suite in ['reverse','flags-reverse','cold-reverse'] else 0)
else:raise RuntimeError(args.suite)
