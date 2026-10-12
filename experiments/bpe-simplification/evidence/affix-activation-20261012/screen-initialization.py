from pathlib import Path
import argparse,subprocess,json,re,hashlib,time
parser=argparse.ArgumentParser();parser.add_argument('--arm',required=True);parser.add_argument('--suite',required=True);parser.add_argument('--reverse',action='store_true');args=parser.parse_args()
r=Path('/root/code/tokenizers-workspaces/bpe-affix-initial-final-results-20261012');items=json.loads((r/'inputs.json').read_text());rows=[]
for item in items:
 lang=item['language'];case=lang+'-affix-w4';arms=['flags',args.arm]
 if lang=='en':arms.reverse()
 if args.reverse:arms.reverse()
 for arm in arms:
  run=r/args.suite/f'{case}-{arm}';run.mkdir(parents=True,exist_ok=False)
  job=dict(protocol_version=1,attempt_id=run.name,build_id=arm,input_id=lang+'256',mode='core',input=item['prepared_path'],output=str(run/'not-produced-model.json'),workers=4,pretokenizer='whitespace',trainer=dict(vocab_size=50000,min_frequency=2,prefix='##',suffix='</w>',max_token_length=None,limit_alphabet=item['limit_alphabet']))
  (run/'job.json').write_text(json.dumps(job,indent=2)+'\n');started=time.monotonic();phases={};status={}
  with (run/'stdout.log').open('w') as out,(run/'stderr.log').open('w') as err:
   proc=subprocess.Popen(['taskset','-c','0-3',str(r/f'{arm}-bin'),str(run/'job.json')],stdout=out,stderr=subprocess.PIPE,text=True,bufsize=1)
   for line in proc.stderr:
    err.write(line)
    match=re.match(r'BPE_(\w+)_SECONDS ([\d.]+)',line)
    if match:phases[match.group(1).lower()]=float(match.group(2))
    if line.startswith('BPE_VOCABULARY_SECONDS'):
     for k,v in re.findall(r'^(VmHWM|VmSwap):\s+(\d+)',Path(f'/proc/{proc.pid}/status').read_text(),re.M):status[k]=int(v)
     proc.terminate();break
   else:raise RuntimeError((run,'process exited before initialization boundary'))
   err.write(proc.stderr.read());proc.wait(timeout=15)
  assert proc.returncode in [-15,-9],(run,proc.returncode)
  row=dict(case=case,arm=arm,phases=phases,status=status,external_wall_before_termination=time.monotonic()-started,returncode=proc.returncode,termination_boundary='BPE_VOCABULARY_SECONDS emitted; no completed training/model validation',binary_sha256=hashlib.sha256((r/f'{arm}-bin').read_bytes()).hexdigest())
  rows.append(row);(r/f'{args.suite}.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
