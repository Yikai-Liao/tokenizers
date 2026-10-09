"""Run the existing fixed-word public do_train harness; never time serialization.
Original source, instrumented source with tracing disabled, and two trace levels
share fixed word maps, settings, worker counts and CPU affinity.
"""
import argparse,json,os,signal,subprocess,time,hashlib
from pathlib import Path
ROOT=Path('/root/code/tokenizers-perfetto-results')
SUITE=Path('/root/code/tokenizers-bpe-benchmarks')
SCRIPTS=Path(__file__).resolve().parent
BASE=SUITE/'.bench/builds/badfc052f99e95b871b75ad85a6e24e3067249bde351ab3af7d4bbc9c799b6a8/bpe-bench-runner'
DIAG=ROOT/'runner/target/release/bpe-bench-runner'
CASES={
 'en-whitespace':(SUITE/'.bench/owner/en256-words/words.json','whitespace',268434441),
 'zh-whitespace':(SUITE/'.bench/owner/zh256-words/words.json','whitespace',268435162),
 'en-bytelevel':(ROOT/'inputs/en-bytelevel/words.json','bytelevel_regex',268434441),
 'zh-bytelevel':(ROOT/'inputs/zh-bytelevel/words.json','bytelevel_regex',268435162),
 'code-bytelevel':(ROOT/'inputs/code-bytelevel/words.json','bytelevel_regex',135266254),
}
def job(case,workers,out):
 inp,pre,_=CASES[case]
 return dict(protocol_version=1,attempt_id=out.name,build_id='perfetto',input_id=case,mode='core',input=str(inp),output=str(out/'model.json'),workers=workers,pretokenizer=pre,trainer=dict(vocab_size=50000,min_frequency=2,prefix=None,suffix=None,max_token_length=None))
def fingerprint(path):
 d=json.loads(path.read_text());return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def run(case,workers,arm,rep,system=False):
 out=ROOT/'runs'/f'{case}-w{workers}-{arm}-r{rep}';out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists(): return json.loads((out/'result.json').read_text())
 j=job(case,workers,out);(out/'job.json').write_text(json.dumps(j,indent=2))
 env=os.environ.copy();env['BPE_TRACE_LEVEL']=str({'base':0,'off':0,'coarse':1,'detail':2}[arm])
 if arm in ('coarse','detail'):env['BPE_TRACE_PATH']=str(out/'events.json')
 tracepid=None
 if system:
  with (out/'tracebox.log').open('w') as err:
   x=subprocess.run([str(ROOT/'tools/tracebox'),'--background-wait','--txt','-c',str(SCRIPTS/'system.pbtxt'),'-o',str(out/'system.perfetto-trace')],stdout=subprocess.PIPE,stderr=err,text=True,check=True)
  tracepid=int(x.stdout.strip())
 begin=time.monotonic();p=subprocess.Popen(['taskset','-c',f'0-{workers-1}',str(BASE if arm=='base' else DIAG),str(out/'job.json')],stdout=subprocess.PIPE,stderr=(out/'stderr.log').open('w'),env=env,text=True)
 memory=[]
 while p.poll() is None:
  try:
   vals={}
   for line in Path(f'/proc/{p.pid}/status').read_text().splitlines():
    if line.startswith(('VmRSS:','VmSwap:')):k,v,*_=line.split();vals[k[:-1]]=int(v)
   meminfo=Path('/proc/meminfo').read_text();vals['MemAvailable']=int(next(x.split()[1] for x in meminfo.splitlines() if x.startswith('MemAvailable:')))
   vals['elapsed']=time.monotonic()-begin;memory.append(vals)
  except (FileNotFoundError,ProcessLookupError):pass
  time.sleep(.05)
 stdout=p.stdout.read();(out/'stdout.log').write_text(stdout);(out/'memory.json').write_text(json.dumps(memory))
 if tracepid:
  os.kill(tracepid,signal.SIGTERM)
  # The background recorder flushes and exits on TERM. Poll our log for completion.
  for _ in range(100):
   if not Path(f'/proc/{tracepid}').exists() or 'State:\tZ' in Path(f'/proc/{tracepid}/status').read_text():break
   time.sleep(.1)
 if p.returncode:raise RuntimeError((out,p.returncode,(out/'stderr.log').read_text()))
 result=json.loads(stdout);result.update(case=case,workers=workers,arm=arm,rep=rep,system=system,model_sha256=fingerprint(out/'model.json'),max_swap_kib=max((x.get('VmSwap',0) for x in memory),default=0),min_available_kib=min((x.get('MemAvailable',0) for x in memory),default=0))
 reference=ROOT/'reference'/f'{case}.json';reference.parent.mkdir(exist_ok=True)
 if reference.exists():
  if fingerprint(reference)!=result['model_sha256']:raise RuntimeError('model mismatch '+str(out))
 else:
  if arm!='base':raise RuntimeError('run baseline first')
  reference.write_bytes((out/'model.json').read_bytes())
 # Canonical baseline preserves full vocab IDs and ordered merge sequence.
 (out/'model.json').unlink()
 (out/'result.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(dict(case=case,w=workers,arm=arm,rep=rep,seconds=result['metrics']['train_seconds'],cpu=result['metrics']['train_cpu_seconds'],swap=result['max_swap_kib'])),flush=True)
 if system:
  subprocess.run([str(ROOT/'venv/bin/python'),str(SCRIPTS/'to_native.py'),str(out/'events.json'),str(out/'trace.perfetto-trace'),'--system',str(out/'system.perfetto-trace')],check=True)
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cases',nargs='+',default=['en-whitespace','zh-whitespace']);p.add_argument('--workers',type=int,nargs='+',default=[4]);p.add_argument('--arms',nargs='+',default=['base','off','coarse','detail']);p.add_argument('--reps',type=int,default=3);p.add_argument('--system',action='store_true');a=p.parse_args()
 for rep in range(1,a.reps+1):
  for case in a.cases:
   for w in a.workers:
    # Alternate trace-arm order after each baseline; retain paired replicates.
    arms=a.arms if rep%2 else ([x for x in a.arms if x=='base']+[x for x in reversed(a.arms) if x!='base'])
    for arm in arms:run(case,w,arm,rep,a.system and arm=='detail' and rep==1)
