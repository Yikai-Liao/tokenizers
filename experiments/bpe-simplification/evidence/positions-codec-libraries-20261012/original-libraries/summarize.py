from pathlib import Path
import json,statistics,re
r=Path(__file__).resolve().parent;rows=json.loads((r/'normal.json').read_text());out={}
for case in sorted({x['case'] for x in rows}):
 group={}
 for arm in ['hand','gimli','fmt-slice','fmt-array']:
  items=[x for x in rows if x['case']==case and x['arm']==arm]
  if not items:continue
  metrics={k:dict(mean=statistics.mean(x[k] for x in items),min=min(x[k] for x in items),max=max(x[k] for x in items),raw=[x[k] for x in items]) for k in ['train_seconds','train_cpu_seconds','process_hwm_kib_before_validation','child_sampled_rss_kib','child_max_swap_kib']}
  phases={}
  for label in ['BPE_INITIAL_COUNT_SECONDS','BPE_PREPARE_SECONDS','BPE_APPLY_SECONDS','BPE_COMMIT_SECONDS','BPE_DROP_SECONDS']:
   vals=[]
   for x in items:
    path=r/'normal'/f"{case}-r{x['repeat']}-{arm}"/'stderr.log';m=re.search(r'^'+label+r' (\S+)',path.read_text(),re.M)
    if m:vals.append(float(m[1]))
   if vals:phases[label]=dict(mean=statistics.mean(vals),raw=vals)
  group[arm]=dict(runs=len(items),metrics=metrics,phases=phases,models_identical_to_reference=True,affinity=items[0]['affinity'])
 if 'hand' in group:
  hand=group['hand']['metrics']['train_seconds']['mean']
  for arm,value in group.items():value['wall_percent_vs_hand']=100*(value['metrics']['train_seconds']['mean']/hand-1)
 out[case]=group
(r/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
for case,group in out.items():print(case,{a:dict(runs=v['runs'],wall=round(v['metrics']['train_seconds']['mean'],3),vs_hand=round(v['wall_percent_vs_hand'],2)) for a,v in group.items()})
