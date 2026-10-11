from pathlib import Path
import json,subprocess,os,re,hashlib,time
r=Path('/root/code/tokenizers-workspaces/bpe-affix-initial-final-results-20261012');items=json.loads((r/'inputs.json').read_text());rows=[];expected={}
for decorated in [True,False]:
 for item in items:
  lang=item['language'];case=lang+('-affix' if decorated else '-plain')+'-w4'
  arms=['baseline','candidate'] if lang=='zh' else ['candidate','baseline']
  for arm in arms:
   run=r/'allocation-init'/f'{case}-{arm}';run.mkdir(parents=True,exist_ok=False)
   job=dict(attempt_id=run.name,build_id=arm,protocol_version=1,input_id=lang+'256',mode='core',input=item['prepared_path'],workers=4,pretokenizer='whitespace',output=str(run/'unused-model.json'),trainer=dict(vocab_size=50000,min_frequency=2,prefix='##' if decorated else None,suffix='</w>' if decorated else None,max_token_length=None,limit_alphabet=item['limit_alphabet']))
   (run/'job.json').write_text(json.dumps(job,indent=2)+'\n')
   with (run/'snapshot.json').open('w') as out,(run/'stderr.log').open('w') as err:subprocess.run(['taskset','-c','0-3',str(r/f'{arm}-diagnostic-bin'),str(run/'job.json')],stdout=out,stderr=err,check=True)
   snap=(run/'snapshot.json').read_bytes();expected.setdefault(case,snap);assert snap==expected[case],(case,arm,'initializer mapping mismatch')
   log=(run/'stderr.log').read_text();counts=json.loads(re.search(r'BPE_INITIAL_ALLOCATIONS (\[[^\n]+\])',log).group(1));layout=dict((k,int(v)) for k,v in re.findall(r'(rows|row_bytes|ids_bytes|table_bytes)=(\d+)',log))
   row=dict(case=case,arm=arm,allocations=counts[0],reallocations=counts[1],requested_bytes=counts[2],frees=counts[3],layout=layout,snapshot_sha256=hashlib.sha256(snap).hexdigest(),diagnostic_binary_sha256=hashlib.sha256((r/f'{arm}-diagnostic-bin').read_bytes()).hexdigest())
   rows.append(row);(r/'allocation-init.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
