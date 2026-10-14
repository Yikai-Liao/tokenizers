from pathlib import Path
import json,statistics,hashlib
r=Path(__file__).resolve().parent
rows=[json.loads(s) for s in (r/'allocator-mimalloc-zh4.jsonl').read_text().splitlines()]
assert len(rows)==12 and all(x['model_equal'] and x['max_swap_kib']==0 for x in rows)
results=[]
for arm in ['main','history-buffer','current']:
 system=[x for x in rows if x['arm']==arm];mi=[x for x in rows if x['arm']==arm+'-mimalloc']
 assert len(system)==len(mi)==2
 assert {x['binary_sha256'] for x in system+mi}.__len__()==1
 assert all(x['mimalloc_mapped'] for x in mi) and not any(x['mimalloc_mapped'] for x in system)
 def median(rs,k):return statistics.median(x['metrics'][k] for x in rs)
 result=dict(arm=arm,n=2,binary_sha256=system[0]['binary_sha256'],system_wall=median(system,'train_seconds'),mimalloc_wall=median(mi,'train_seconds'),system_cpu=median(system,'train_cpu_seconds'),mimalloc_cpu=median(mi,'train_cpu_seconds'),system_hwm_kib=median(system,'process_hwm_kib_before_validation'),mimalloc_hwm_kib=median(mi,'process_hwm_kib_before_validation'))
 for k,out in [('train_seconds','paired_wall_change_percent'),('train_cpu_seconds','paired_cpu_change_percent')]:
  result[out]=statistics.median((next(x for x in mi if x['repeat']==s['repeat'])['metrics'][k]/s['metrics'][k]-1)*100 for s in system)
 results.append(result)
manifest=json.loads((r/'allocator-mimalloc/manifest.json').read_text())
summary=dict(condition='Chinese ByteLevel 256MiB, vocab 100000, min_frequency 2, workers 4 pinned CPU0-3, two reversed blocks; no observer',allocator=manifest,results=results,all_models_equal=True,max_swap_kib=0)
(r/'allocator-mimalloc-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for x in results:print(json.dumps(x),flush=True)
