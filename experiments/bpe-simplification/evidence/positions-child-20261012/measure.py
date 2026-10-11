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
 phases={name.lower():sum(float(v) for v in re.findall('BPE_'+name+r'_SECONDS ([\d.]+)',stderr)) for name in ['SELECTION','VOCABULARY','PLAN','INDEX','MATERIALIZE','MERGE','MERGE_RESTART','DROP']}
 model=(run/'model.json').read_bytes();expected.setdefault(case,model);assert expected[case]==model,(case,arm,'full model mismatch')
 after=vm();final=resource.getrusage(resource.RUSAGE_CHILDREN)
 row=dict(case=case,arm=arm,repeat=repeat,**result['metrics'],**result['output'],phases=phases,external_wall_seconds=time.monotonic()-started,child_max_swap_kib=swap,child_sampled_rss_kib=rss,child_major_faults=final.ru_majflt-usage.ru_majflt,external_cpu_seconds=(final.ru_utime+final.ru_stime)-(usage.ru_utime+usage.ru_stime),system_swapin_pages=after['pswpin']-before['pswpin'],system_swapout_pages=after['pswpout']-before['pswpout'],model_sha256=hashlib.sha256(model).hexdigest())
 rows.append(row);(root/f'{args.suite}.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
def job(language,kind,workers):
 path=inputs[(language,kind)]
 return dict(protocol_version=1,input_id=language+'256',mode='core',input=str(path),workers=workers,pretokenizer=kind,trainer=dict(vocab_size=50000,min_frequency=2,prefix=None,suffix=None,max_token_length=None,limit_alphabet=None))
inputs={('zh','whitespace'):Path('/root/code/tokenizers-bpe-benchmarks/.bench/main-yttm-inspired/inputs/zh256-words/words.json'),('en','whitespace'):Path('/root/code/tokenizers-workspaces/bpe-affix-verification-results-20261011/en-prepared-words.json')}
for lang in ['zh','en']:inputs[(lang,'bytelevel_regex')]=root/'inputs'/f'{lang}-bytelevel.json'
if args.suite=='thinvec-cn':
 for kind in ['whitespace','bytelevel_regex']:
  for arm in (['child-inline-fast','child-thinvec'] if kind=='whitespace' else ['child-thinvec','child-inline-fast']):sample(f'zh-{kind}-w4',arm,job('zh',kind,4))
elif args.suite=='single-close':
 for arm in ['child-inline-fast','baseline']:sample('zh-bytelevel_regex-w1',arm,job('zh','bytelevel_regex',1),1)
 for kind in ['whitespace','bytelevel_regex']:
  for arm in (['baseline','child-inline-fast'] if kind=='whitespace' else ['child-inline-fast','baseline']):sample(f'en-{kind}-w1',arm,job('en',kind,1),1)
elif args.suite=='inline24-storage':
 for kind in ['whitespace','bytelevel_regex']:
  for arm in ['child-inline24-final-diagnostic']:sample(f'zh-{kind}-w4',arm,job('zh',kind,4))
elif args.suite=='inline24-cn':
 for kind in ['whitespace','bytelevel_regex']:
  for arm in (['baseline','child-inline-fast','child-inline24'] if kind=='whitespace' else ['child-inline24','child-inline-fast','baseline']):sample(f'zh-{kind}-w4',arm,job('zh',kind,4))
elif args.suite=='inline24-cn-reverse':
 for kind in ['whitespace','bytelevel_regex']:
  for arm in (['child-inline24','child-inline-fast'] if kind=='whitespace' else ['child-inline-fast','child-inline24']):sample(f'zh-{kind}-w4',arm,job('zh',kind,4),1)
elif args.suite=='single-final':
 for lang in ['zh','en']:
  for kind in ['whitespace','bytelevel_regex']:
   for arm in (['child-inline-fast','baseline'] if kind=='whitespace' else ['baseline','child-inline-fast']):sample(f'{lang}-{kind}-w1',arm,job(lang,kind,1))
elif args.suite=='single-cn-reverse':
 for kind in ['whitespace','bytelevel_regex']:
  for arm in (['baseline','child-inline-fast'] if kind=='whitespace' else ['child-inline-fast','baseline']):sample(f'zh-{kind}-w1',arm,job('zh',kind,1),1)
elif args.suite=='final-diagnostic-controls':
 for case in ['aa','reuse']:
  control=json.loads((Path('/root/code/tokenizers-workspaces/bpe-selected-rules-results-20261011/controls')/f'{case}-baseline/job.json').read_text())
  for arm in ['baseline-final-diagnostic','child-inline-fast-final-diagnostic']:sample(case,arm,control)
elif args.suite=='final-controls':
 for case in ['aa','reuse']:
  control=json.loads((Path('/root/code/tokenizers-workspaces/bpe-selected-rules-results-20261011/controls')/f'{case}-baseline/job.json').read_text())
  for arm in ['baseline','child-inline-fast']:sample(case,arm,control)
elif args.suite=='capacity-multi':
 for kind in ['bytelevel_regex','whitespace']:
  for arm in (['child-quarter-growth','child-inline-fast'] if kind=='bytelevel_regex' else ['child-inline-fast','child-quarter-growth']):sample(f'zh-{kind}-w4',arm,job('zh',kind,4))
elif args.suite=='final-storage-multi':
 for kind in ['whitespace','bytelevel_regex']:
  for arm in ['baseline-final-diagnostic','child-inline-fast-final-diagnostic']:sample(f'zh-{kind}-w4',arm,job('zh',kind,4))
elif args.suite=='multi-en':
 for kind in ['whitespace','bytelevel_regex']:
  for arm in (['baseline','child-inline-fast'] if kind=='whitespace' else ['child-inline-fast','baseline']):sample(f'en-{kind}-w4',arm,job('en',kind,4))
elif args.suite=='multi-en-reverse':
 for kind in ['whitespace','bytelevel_regex']:
  for arm in (['child-inline-fast','baseline'] if kind=='whitespace' else ['baseline','child-inline-fast']):sample(f'en-{kind}-w4',arm,job('en',kind,4),1)
elif args.suite=='multi-cn-reverse':
 for kind in ['whitespace','bytelevel_regex']:
  for arm in (['child-inline-fast','baseline'] if kind=='whitespace' else ['baseline','child-inline-fast']):sample(f'zh-{kind}-w4',arm,job('zh',kind,4),1)
elif args.suite=='multi-cn':
 for kind in ['whitespace','bytelevel_regex']:
  for arm in (['baseline','child-inline-direct','child-inline-fast'] if kind=='whitespace' else ['child-inline-fast','child-inline-direct','baseline']):sample(f'zh-{kind}-w4',arm,job('zh',kind,4))
elif args.suite=='owner-inline':
 for kind,workers in [('bytelevel_regex',4)]:
  for arm in (['child-owner-linear','child-inline-fast','child-direct-varint'] if workers==4 else ['child-direct-varint','child-inline-fast','child-owner-linear']):sample(f'zh-{kind}-w{workers}',arm,job('zh',kind,workers))
elif args.suite=='owner-streams':
 for kind,workers in [('bytelevel_regex',4),('bytelevel_regex',1)]:
  for arm in ['child-owner-streams','child-direct-varint']:sample(f'zh-{kind}-w{workers}',arm,job('zh',kind,workers))
elif args.suite=='merge-phase':
 for kind,workers in [('bytelevel_regex',4),('bytelevel_regex',1)]:
  for arm in ['baseline-phase','child-phase']:sample(f'zh-{kind}-w{workers}',arm,job('zh',kind,workers))
elif args.suite=='zh-focus':
 for kind,workers in [('bytelevel_regex',4),('whitespace',1),('bytelevel_regex',1)]:
  for arm in ['child-direct-varint','baseline']:sample(f'zh-{kind}-w{workers}',arm,job('zh',kind,workers))
elif args.suite=='direct-varint':
 for lang,kind,workers in [('en','whitespace',1),('en','bytelevel_regex',1),('zh','whitespace',4)]:
  for arm in (['baseline','child-direct-varint'] if lang=='zh' or kind=='whitespace' else ['child-direct-varint','baseline']):sample(f'{lang}-{kind}-w{workers}',arm,job(lang,kind,workers))
elif args.suite=='cleanup-array':
 for arm in ['cleanup-array-parallel','cleanup-array-serial']:sample('zh-whitespace-w4',arm,job('zh','whitespace',4))
elif args.suite=='cleanup':
 for repeat,order in enumerate([['cleanup-serial','cleanup-parallel'],['cleanup-parallel','cleanup-serial']]):
  for arm in order:sample('zh-whitespace-w4',arm,job('zh','whitespace',4),repeat)
elif args.suite=='owner-array':
 for lang,kind,workers in [('en','whitespace',1),('en','bytelevel_regex',1),('zh','whitespace',4)]:
  for arm in ['child-owner-array','child-fast-push']:sample(f'{lang}-{kind}-w{workers}',arm,job(lang,kind,workers))
elif args.suite=='fast-push':
 for lang,kind,workers in [('en','whitespace',1),('en','bytelevel_regex',1),('zh','whitespace',4)]:
  for arm in ['child-fast-push','child-inline-retire']:sample(f'{lang}-{kind}-w{workers}',arm,job(lang,kind,workers))
elif args.suite=='inline-retire':
 for lang,kind,workers in [('en','whitespace',1),('en','bytelevel_regex',1),('zh','whitespace',4)]:
  for arm in ['child-inline-retire','child-owner']:sample(f'{lang}-{kind}-w{workers}',arm,job(lang,kind,workers))
elif args.suite=='owner':
 for lang,kind,workers in [('en','whitespace',1),('en','bytelevel_regex',1),('zh','whitespace',4)]:
  for arm in ['child-owner','child']:sample(f'{lang}-{kind}-w{workers}',arm,job(lang,kind,workers))
elif args.suite=='pilot':
 for arm in ['baseline','child']:sample('zh-whitespace-w4',arm,job('zh','whitespace',4))
elif args.suite=='v1-control':
 for workers in [1,4]:
  for arm in ['baseline','child-v1']:sample(f'en-whitespace-w{workers}',arm,job('en','whitespace',workers))
elif args.suite=='diagnostic':
 for arm in ['baseline-diagnostic','child-diagnostic']:sample('zh-whitespace-w4',arm,job('zh','whitespace',4))
elif args.suite=='reverse':
 for arm in ['child','baseline']:sample('zh-whitespace-w4',arm,job('zh','whitespace',4),1)
elif args.suite=='controls':
 for case in ['aa','reuse']:
  control=json.loads((Path('/root/code/tokenizers-workspaces/bpe-selected-rules-results-20261011/controls')/f'{case}-baseline/job.json').read_text())
  for arm in ['baseline','child']:sample(case,arm,control)
elif args.suite=='prepare':
 for language in ['zh','en']:
  raw=Path('/root/code/tokenizers-bpe-benchmarks/.bench/owner')/f'{language}256/text.txt'
  output=inputs[(language,'bytelevel_regex')];output.parent.mkdir(exist_ok=True)
  prep=job(language,'bytelevel_regex',4)|dict(mode='prepare',input=str(raw),output=str(output),attempt_id='prepare-'+language,build_id='baseline')
  name=root/'inputs'/f'{language}-prepare-job.json';name.write_text(json.dumps(prep,indent=2)+'\n')
  with name.with_suffix('.log').open('w') as out:
   subprocess.run(['taskset','-c','0-3',str(root/'baseline-bin'),str(name)],stdout=out,stderr=subprocess.STDOUT,check=True)
  print(language,output.stat().st_size,hashlib.sha256(output.read_bytes()).hexdigest(),flush=True)
else:
 for workers in [4,1]:
  for kind in ['whitespace','bytelevel_regex']:
   for language in ['zh','en']:
    for arm in (['baseline','child'] if language=='zh' else ['child','baseline']):
     sample(f'{language}-{kind}-w{workers}',arm,job(language,kind,workers))
