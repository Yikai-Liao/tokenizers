from pathlib import Path
import json
OUT=Path('/root/code/tokenizers-simplification-results/online-initial')

def summarize():
 rows=[json.loads(l) for l in (OUT/'runs.jsonl').read_text().splitlines()]
 result=[]
 for case in ('zh-512MiB','zh-1024MiB'):
  arms={r['arm']:r for r in rows if r['valid'] and r['kind']=='prepare-ablation' and r['case']==case}
  if not {'control','ordered','lookup','combined'}<=arms.keys():continue
  keymetrics={}
  for arm,r in arms.items():
   stages={p['name']:p for p in r['phases']};detail={p['name']:p for p in r['prepare_detail']['stages']}
   keymetrics[arm]=dict(train_wall=r['metrics']['train_seconds'],train_cpu=r['metrics']['train_cpu_seconds'],hwm_kib=r['metrics']['process_hwm_kib_before_validation'],prepare_wall=stages['prepare']['wall_seconds'],prepare_cpu=stages['prepare']['cpu_seconds'],scan_thread_cpu=detail['scan_collect']['cpu_seconds'],finish_thread_cpu=detail['finish_encode']['cpu_seconds'],setup_thread_cpu=detail['worker_setup']['cpu_seconds'],selected_process_cpu=detail['selected_index']['cpu_seconds'])
  effects={}
  for key in keymetrics['control']:
   c,o,l,b=(keymetrics[a][key] for a in ('control','ordered','lookup','combined'))
   effects[key]=dict(ordered_gain=c-o,lookup_gain=c-l,combined_gain=c-b,ordered_added_to_lookup=l-b,lookup_added_to_ordered=o-b,interaction_gain=o+l-c-b,ordered_gain_percent=100*(c-o)/c,lookup_gain_percent=100*(c-l)/c,combined_gain_percent=100*(c-b)/c)
  result.append(dict(case=case,n_per_arm=1,metrics=keymetrics,effects=effects,scan_counts={a:r['prepare_detail']['counters'] for a,r in arms.items()}))
 (OUT/'small-ablation-summary.json').write_text(json.dumps(result,indent=2))
 return result

if __name__=='__main__':
 for case in summarize():
  print(case['case'])
  for a,m in case['metrics'].items():print(a,m)
  for k,e in case['effects'].items():print(k,e)
