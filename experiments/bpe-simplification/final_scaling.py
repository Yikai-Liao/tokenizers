"""Same-prefix large Chinese core/phase diagnostics; source and binary identities are recorded.

Requires the immutable runners and prepared 256MiB input described in
manifest-review-final.json and full-review-provenance.json. Writes fresh artifacts
under /root/code/tokenizers-simplification-results/final-scaling. Does not mutate
production source. Refuses concurrent benchmarks/builds and aborts on swapping.
"""
from pathlib import Path
import os,json,hashlib,subprocess,time,argparse,re
ROOT=Path('/root/code/tokenizers-simplification-results');OUT=ROOT/'final-scaling'
SOURCE=Path('/root/code/tokenizers-bpe-benchmarks/.bench/prezza/zh512/text.txt')
CURRENT=Path('/root/code/tokenizers-bpe-benchmarks/.bench/owner/zh256/text.txt')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def job(case,mode,inp,out,arm):
 return dict(protocol_version=1,attempt_id=arm,build_id=arm,input_id=case,mode=mode,input=str(inp),output=str(out),workers=4,pretokenizer='bytelevel_regex',trainer=dict(vocab_size=50000,min_frequency=2,prefix=None,suffix=None,max_token_length=None))
def busy(own=None):
 found=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==own:continue
  try:
   name=(p/'comm').read_text().strip();cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
   if name in ('rustc','cargo') or ('bpe-bench-runner' in cmd and name.startswith('bpe-bench')) or (str(ROOT/'bin')+'/' in cmd and name.startswith('bpe-bench')):found.append(dict(pid=int(p.name),command=cmd))
  except (FileNotFoundError,ProcessLookupError,PermissionError):pass
 return found
def prepare():
 OUT.mkdir(exist_ok=True)
 h=hashlib.sha256()
 with SOURCE.open('rb') as f:
  remain=CURRENT.stat().st_size
  while remain:
   b=f.read(min(remain,1048576));assert b;h.update(b);remain-=len(b)
 assert h.hexdigest()==sha(CURRENT),'Different 256 MiB source prefix; cannot use the existing scale point'
 inputs=[dict(case='zh-256MiB',requested_mib=256,path=str(CURRENT),bytes=CURRENT.stat().st_size,source_sha256=sha(CURRENT),prepared_path='/root/code/tokenizers-perfetto-results/inputs/zh-bytelevel/words.json',prepared_sha256='bb796d0b57b87d965b84de249bc5c8e283b700774638d16a40e4ae5171b13ee0',unique_words=6859987)]
 for size in (384,512):
  d=OUT/'inputs'/f'zh-{size}MiB';d.mkdir(parents=True,exist_ok=False)
  if size==512:inp=SOURCE
  else:
   with SOURCE.open('rb') as f:data=f.read(size*1048576)
   if not data.endswith(b'\n'):data=data.rsplit(b'\n',1)[0]+b'\n'
   data.decode('utf-8');inp=d/'text.txt';inp.write_bytes(data);del data
  j=job(f'zh-{size}MiB','prepare',inp,d/'words.json','prepare');(d/'job.json').write_text(json.dumps(j,indent=2))
  assert not busy(),busy()
  with (d/'stdout.json').open('w') as out,(d/'stderr.log').open('w') as err:subprocess.run(['taskset','-c','0-3',str(ROOT/'bin/baseline'),str(d/'job.json')],stdout=out,stderr=err,check=True)
  result=json.loads((d/'stdout.json').read_text())
  info=dict(case=f'zh-{size}MiB',requested_mib=size,path=str(inp),bytes=inp.stat().st_size,source_sha256=sha(inp),prepared_path=str(d/'words.json'),prepared_sha256=sha(d/'words.json'),prepared_bytes=(d/'words.json').stat().st_size,unique_words=result['unique_words']);inputs.append(info)
  print('prepared '+json.dumps(info),flush=True)
 (OUT/'inputs.json').write_text(json.dumps(inputs,indent=2))
def measure(info,arm,diagnostic):
 assert not busy(),busy()
 case=info['case'];kind='diagnostic' if diagnostic else 'plain';d=OUT/'runs'/case/f'{kind}-{arm}';d.mkdir(parents=True,exist_ok=False)
 name=('full-review-main-geometry' if arm=='main' else 'full-review-final') if diagnostic else ('baseline' if arm=='main' else 'review-final')
 binary=ROOT/'bin'/name;model=d/'model.json';j=job(case,'core',info['prepared_path'],model,arm);(d/'job.json').write_text(json.dumps(j,indent=2))
 env={k:v for k,v in os.environ.items() if not k.startswith(('BPE_TRACE_','BPE_ABLATION_','BPE_SORT_'))};command=['taskset','-c','0-3',str(binary),str(d/'job.json')]
 samples=[];foreign=[];failure=None;beg=time.monotonic();last_check=beg
 with (d/'stdout.json').open('w') as out,(d/'stderr.log').open('w') as err:
  p=subprocess.Popen(command,stdout=out,stderr=err,env=env)
  while p.poll() is None:
   try:
    status=Path(f'/proc/{p.pid}/status').read_text().splitlines();fields={l.split(':')[0]:int(l.split()[1]) for l in status if l.startswith(('VmRSS:','VmHWM:','VmSwap:'))}
    avail=next(int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
    samples.append(dict(seconds=time.monotonic()-beg,available_kib=avail,**fields))
    if fields.get('VmSwap',0)>0 or avail<524288:failure='swap-or-host-memory-guard';p.terminate();break
    if time.monotonic()-last_check>1:
     others=busy(p.pid);last_check=time.monotonic()
     if others:foreign.extend(others);failure='foreign-heavy-process';p.terminate();break
   except (FileNotFoundError,ProcessLookupError):pass
   time.sleep(.02)
  p.wait()
 r=dict(case=case,arm=arm,diagnostic=diagnostic,command=command,binary_sha256=sha(binary),elapsed_seconds=time.monotonic()-beg,returncode=p.returncode,failure=failure,foreign_processes=foreign,max_swap_kib=max([s.get('VmSwap',0) for s in samples]+[0]),minimum_available_kib=min([s['available_kib'] for s in samples]+[99999999]))
 if p.returncode==0:
  r.update(json.loads((d/'stdout.json').read_text()));ref=OUT/'reference'/f'{case}.json';ref.parent.mkdir(exist_ok=True)
  if case=='zh-256MiB' and not ref.exists():ref.write_bytes((ROOT/'reference/zh-core-w4-v50000.json').read_bytes())
  if not ref.exists():assert arm=='main';ref.write_bytes(model.read_bytes())
  r['model_equal']=json.loads(model.read_text())==json.loads(ref.read_text());r['model_sha256']=sha(model)
  if r['model_equal']:model.unlink()
  logs=(d/'stderr.log').read_text().splitlines();r['geometry']=[json.loads(l[13:]) for l in logs if l.startswith('BPE_GEOMETRY ')];r['route']=[l[10:] for l in logs if l.startswith('BPE_ROUTE ')]
  if diagnostic:r['phases']=json.loads(next(l[11:] for l in logs if l.startswith('BPE_PHASES ')))['stages']
 r['valid']=r.get('model_equal',False) and not failure and p.returncode==0 and r['max_swap_kib']==0
 (d/'samples.json').write_text(json.dumps(samples));(d/'result.json').write_text(json.dumps(r,indent=2))
 with (OUT/'runs.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
 print(json.dumps({k:r.get(k) for k in ('case','arm','diagnostic','valid','metrics','geometry','route','phases')}),flush=True);assert r['valid'],d
 return r
def run():
 inputs=json.loads((OUT/'inputs.json').read_text())
 for info in inputs[1:]:
  for arm in ('main','final'):measure(info,arm,False)
 for info in (inputs[0],inputs[-1]):
  for arm in ('main','final'):measure(info,arm,True)
def stats():
 # Compact serde_json writes each word/count entry in this exact shape. Decode
 # each matched JSON string so escapes and Unicode count as real Rust chars.
 pattern=re.compile(r'\["((?:[^"\\]|\\.)*)",([0-9]+)\]')
 infos=json.loads((OUT/'inputs.json').read_text());rows=[]
 for info in infos:
  carry='';words=chars=edges=empty=0
  with Path(info['prepared_path']).open(encoding='utf-8') as f:
   while chunk:=f.read(4*1024*1024):
    carry+=chunk;end=0
    for m in pattern.finditer(carry):
     n=len(json.loads('"'+m[1]+'"'));words+=1;chars+=n;edges+=max(n-1,0);empty+=n==0;end=m.end()
    if end:carry=carry[end:]
  assert words==info['unique_words'],(info['case'],words,info['unique_words'])
  row=dict(case=info['case'],words=words,initial_characters=chars,physical_edges=edges,resident_slots=chars+words+1,empty_words=empty,derived_main_wave_count=(chars+words+((1<<28)-1))//(1<<28))
  rows.append(row);print('geometry '+json.dumps(row),flush=True)
 (OUT/'geometry.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run','stats'));a=p.parse_args();{'prepare':prepare,'run':run,'stats':stats}[a.action]()
