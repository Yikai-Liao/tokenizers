from pathlib import Path
import subprocess,json,os,time,re,hashlib,csv,resource
root=Path(__file__).resolve().parent
inputs=json.loads((root/'inputs.json').read_text())
def vm():
 return {k:int(v) for k,v in (line.split() for line in Path('/proc/vmstat').read_text().splitlines()) if k in ('pswpin','pswpout')}
rows=json.loads((root/'runs.json').read_text())
scenarios = [(i, True) for i in inputs if i['language']=='zh']
# Sequential samples; opposite arm order across languages. Four workers throughout.
for item, decorated in scenarios:
 lang=item['language']
 case='zh-affix-repeat'
 for arm in ['flags-only','candidate','baseline']:
  run=root/'runs'/(case+'-'+arm);run.mkdir(parents=True,exist_ok=False)
  job=dict(protocol_version=1,attempt_id=run.name,build_id=arm,input_id=lang+'256',mode='core',input=item['prepared_path'],output=str(run/'model.json'),workers=4,pretokenizer='whitespace',trainer=dict(vocab_size=50000,min_frequency=2,prefix="##" if decorated else None,suffix="</w>" if decorated else None,max_token_length=None,limit_alphabet=item['limit_alphabet']))
  (run/'job.json').write_text(json.dumps(job,indent=2)+'\n')
  before=vm();usage_before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic();maxswap=0;maxrss=0;majfault=0
  with (run/'stdout.log').open('w') as stdout,(run/'stderr.log').open('w') as stderr:
   p=subprocess.Popen(['taskset','-c','0-3',str(root/(arm+'-bin')),str(run/'job.json')],stdout=stdout,stderr=stderr)
   while p.poll() is None:
    try:
     status=Path('/proc')/str(p.pid)/'status';txt=status.read_text()
     for field in ['VmSwap','VmRSS']:
      m=re.search(r'^'+field+r':\s+(\d+)',txt,re.M)
      if m:
       if field=='VmSwap': maxswap=max(maxswap,int(m[1]))
       else: maxrss=max(maxrss,int(m[1]))
     stat=(Path('/proc')/str(p.pid)/'stat').read_text().rsplit(')',1)[1].split();majfault=max(majfault,int(stat[9]))
    except FileNotFoundError: pass
    time.sleep(.05)
  elapsed=time.monotonic()-start;after=vm()
  if p.returncode: raise RuntimeError((run,p.returncode,(run/'stderr.log').read_text()))
  result=json.loads((run/'stdout.log').read_text());initialization=[float(x) for x in re.findall(r'BPE_INITIALIZATION_SECONDS ([\d.]+)',(run/'stderr.log').read_text())]
  selection=[float(x) for x in re.findall(r'BPE_ALPHABET_SELECTION_SECONDS ([\d.]+)',(run/'stderr.log').read_text())]
  raw_init=[float(x) for x in re.findall(r'BPE_VOCABULARY_INITIALIZATION_RAW_SECONDS ([\d.]+)',(run/'stderr.log').read_text())]
  assert initialization and selection and raw_init
  selection_seconds=sum(selection)
  initialization_seconds=sum(raw_init) - (selection_seconds if arm != 'candidate' else 0.0)
  combined_seconds=sum(initialization)
  assert abs(selection_seconds+initialization_seconds-combined_seconds)<0.01
  final_usage=resource.getrusage(resource.RUSAGE_CHILDREN)
  row=dict(language=lang,case=case,arm=arm,workers=4,affix=decorated,limit_alphabet=item['limit_alphabet'],alphabet_selection_seconds=selection_seconds,vocabulary_initialization_seconds=initialization_seconds,alphabet_plus_initialization_seconds=combined_seconds,attempts=len(raw_init),child_major_faults=final_usage.ru_majflt-usage_before.ru_majflt,**result['metrics'],**result['output'],child_max_swap_kib=maxswap,child_major_faults_sampled=majfault,system_swapin_pages=after['pswpin']-before['pswpin'],system_swapout_pages=after['pswpout']-before['pswpout'],external_wall_seconds=elapsed,model_sha256=hashlib.sha256((run/'model.json').read_bytes()).hexdigest())
  rows.append(row);(root/'runs.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
 for arm in (['baseline','candidate','flags-only'] if decorated else ['baseline','candidate']):
  path=root/'runs'/(case+'-'+arm)/'model.json'
  if arm=='baseline': expected=path.read_bytes()
  else: assert path.read_bytes()==expected,lang+' model differs'
 print(case,'vocabulary and ordered merges equal',flush=True)
with (root/'runs.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
