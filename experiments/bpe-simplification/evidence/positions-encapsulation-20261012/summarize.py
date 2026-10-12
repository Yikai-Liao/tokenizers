from pathlib import Path
import json,statistics,re
r=Path(__file__).resolve().parent;rows=json.loads((r/'normal.json').read_text());assert len(rows)==48
summary={'normal_runs':48,'affinity':[0,1],'workers_requested':[1,6],'rounds_per_case_and_arm':2,'all_full_models_equal_baseline':True,'max_sampled_swap_kib':max(x['child_max_swap_kib'] for x in rows),'cases':{},'production_lines':json.loads((r/'production-costs.json').read_text()),'limits':['Six workers share a two-CPU affinity; this is not an unrestricted six-core result.','Two interleaved rounds characterize observed differences, not statistical significance.','Allocation-failure invariants were reviewed in source; no fault injection facility was used.']}
metrics=['train_seconds','train_cpu_seconds','process_hwm_kib_before_validation','child_sampled_rss_kib']
for case in dict.fromkeys(x['case'] for x in rows):
 arms={}
 for arm in ['base','a','b']:
  selected=[x for x in rows if x['case']==case and x['arm']==arm];assert len(selected)==2 and all(x['affinity']==[0,1] for x in selected)
  arms[arm]={metric:dict(mean=statistics.mean(x[metric] for x in selected),min=min(x[metric] for x in selected),max=max(x[metric] for x in selected),samples=[x[metric] for x in selected]) for metric in metrics}
  phases={}
  for x in selected:
   text=(r/'normal'/f"{case}-r{x['repeat']}-{arm}"/'stderr.log').read_text()
   for name,value in re.findall(r'^BPE_(\w+_SECONDS) (\S+)',text,re.M):phases.setdefault(name,[]).append(float(value))
  arms[arm]['phases']={name:dict(mean=statistics.mean(v),samples=v) for name,v in phases.items()}
 for arm in ['a','b']:
  arms[arm]['relative_to_base_percent']={metric:(arms[arm][metric]['mean']/arms['base'][metric]['mean']-1)*100 for metric in metrics}
 arms['b']['relative_to_a_percent']={metric:(arms['b'][metric]['mean']/arms['a'][metric]['mean']-1)*100 for metric in metrics}
 summary['cases'][case]=arms
(r/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for case,arms in summary['cases'].items():
 print(case,'wall',*[f"{a} {arms[a]['train_seconds']['mean']:.3f}s" for a in ['base','a','b']],'A/base',round(arms['a']['relative_to_base_percent']['train_seconds'],2),'B/A',round(arms['b']['relative_to_a_percent']['train_seconds'],2),'RSS MiB',*[f"{a} {arms[a]['process_hwm_kib_before_validation']['mean']/1024:.1f}" for a in ['base','a','b']])
