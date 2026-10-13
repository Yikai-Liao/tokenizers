import gzip, hashlib, json, subprocess
from pathlib import Path
root=Path('/root/code/tokenizers-workspaces/bpe-corpus-boundary-20261013/experiments/bpe-simplification/evidence/corpus-boundary-20261013')
definition=json.loads((root/'inputs.json').read_text())['zh-bytelevel']
binaries={'baseline':Path('/tmp/bpe-corpus-binaries-20261013/final-baseline-production'),'candidate':Path('/tmp/bpe-corpus-binaries-20261013/aligned-candidate-production')}
results=[]
for kind, rounds in [('stat',8),('record',2)]:
 for repeat in range(rounds):
  for arm in (['baseline','candidate'] if repeat%2==0 else ['candidate','baseline']):
   directory=root/'zh-aligned-profile'/f'{kind}-r{repeat}-{arm}';directory.mkdir(parents=True,exist_ok=True)
   job=dict(protocol_version=1,attempt_id=directory.name,build_id=arm,input_id='zh-bytelevel',mode='core',workers=1,output=str(directory/'model.json'),**{k:v for k,v in definition.items() if k in ['input','pretokenizer','trainer']})
   (directory/'job.json').write_text(json.dumps(job,indent=2))
   prefix=['perf','stat','-x',';','-o',str(directory/'counters.csv'),'-e','cycles,instructions,branches,branch-misses,cache-references,cache-misses'] if kind=='stat' else ['perf','record','-F','997','-g','--call-graph','dwarf,8192','-o',str(directory/'perf.data')]
   command=prefix+['--','taskset','-c','0',str(binaries[arm]),str(directory/'job.json')]
   with (directory/'stdout.log').open('w') as out,(directory/'stderr.log').open('w') as err:
    subprocess.run(command,stdout=out,stderr=err,check=True)
   model=(directory/'model.json').read_bytes()
   with gzip.open(directory/'model.json.gz','wb') as out:out.write(model)
   (directory/'model.json').unlink()
   result=dict(kind=kind,repeat=repeat,arm=arm,binary_sha256=hashlib.sha256(binaries[arm].read_bytes()).hexdigest(),model_sha256=hashlib.sha256(model).hexdigest(),output=json.loads((directory/'stdout.log').read_text()),command=command,directory=str(directory))
   if kind=='stat':
    counters={}
    for line in (directory/'counters.csv').read_text().splitlines():
     parts=line.split(';')
     if len(parts)>2 and parts[0].strip().isdigit():counters[parts[2]]=int(parts[0])
    result['counters']=counters
   results.append(result);(directory/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
   (root/'zh-aligned-profile-runs.json').write_text(json.dumps(results,indent=2))
assert len({r['model_sha256'] for r in results})==1
