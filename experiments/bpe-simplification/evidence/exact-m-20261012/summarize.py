from pathlib import Path
import argparse,csv,gzip,json,statistics,hashlib
p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args();r=a.evidence
read=lambda path:json.loads(gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes())
normal=[]
for suite in ['controls','primary','reverse']:
 for row in read(r/f'{suite}.json'):normal.append(row|{'suite':suite})
diag=read(r/'diagnostic.json.gz')+read(r/'diagnostic-controls.json.gz')
pairs=[]
for case in sorted({x['case'] for x in normal if x['suite']!='controls'}):
 for suite in ['primary','reverse']:
  rows={x['arm']:x for x in normal if x['suite']==suite and x['case']==case}
  for arm in ['frontier','exact']:
   base=rows['eager'];row=rows[arm]
   pairs.append(dict(case=case,suite=suite,arm=arm,train_seconds=row['train_seconds'],train_cpu_seconds=row['train_cpu_seconds'],wall_delta_pct=(row['train_seconds']/base['train_seconds']-1)*100,cpu_delta_pct=(row['train_cpu_seconds']/base['train_cpu_seconds']-1)*100,hwm_delta_kib=row['process_hwm_kib_before_validation']-base['process_hwm_kib_before_validation'],hwm_budget_kib=max(base['process_hwm_kib_before_validation']*.05,16384),phases=row['phases']))
physical=[]
for row in diag:
 windows=row['window_reports'];work=row['window_work_reports'][0];producer=row['producer_reports'][0]
 reasons={reason:sum(w['stop_reason']==reason for w in windows) for reason in sorted({w['stop_reason'] for w in windows})}
 cohort_counts={str(k):sum(w['cohorts']==k for w in windows) for k in sorted({w['cohorts'] for w in windows})}
 def old_beats_child(w):
  old=w['blocked_priority'];child=w['actual_best_final_child'];return old is not None and child is not None and (old[0]>child[0] or (old[0]==child[0] and old[1]<child[1]))
 physical.append(dict(case=row['case'],arm=row['arm'],work=work,producer=producer,encoding=row['encoding_reports'][0],publisher=row['publisher_reports'][0],directories=row['directory_reports'],allocation=row['allocation_reports'][0],outer_windows=len(windows),inner_cohorts=sum(w['cohorts'] for w in windows),cohort_histogram=cohort_counts,stop_reasons=reasons,guard_stops_with_old_head_beating_final_child=sum(old_beats_child(w) for w in windows if w['stop_reason']=='guard'),retained_checkpoint_bytes_max=max((w['retained_checkpoint_bytes'] for w in windows),default=0),attempt_high_water_bytes_max=max((w['attempt_high_water_bytes'] for w in windows),default=0),pruned_outputs=sum(w['pruned_outputs'] for w in windows),producer_joins=row['producer_joins']))
summary=dict(decision='reject_exact_m',normal_runs=len(normal),diagnostic_runs=len(diag),pairs=pairs,physical=physical)
(r/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
fields=['suite','case','arm','train_seconds','train_cpu_seconds','process_hwm_kib_before_validation','child_max_swap_kib','model_sha256']
with (r/'normal-runs.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(normal)
for case in sorted({x['case'] for x in pairs}):
 m=[x for x in pairs if x['case']==case and x['arm']=='exact'];d={x['arm']:x for x in physical if x['case']==case};e=d['eager-diagnostic'];f=d['frontier-diagnostic'];x=d['exact-diagnostic'];print(case,'cpu',*[round(y['cpu_delta_pct'],2) for y in m],'append',e['work']['positions_append_calls_during_merges'],f['work']['positions_append_calls_during_merges'],x['work']['positions_append_calls_during_merges'],'cohorts',f['inner_cohorts'],x['inner_cohorts'],'windows',f['outer_windows'],x['outer_windows'],'Mcpu',x['producer']['reducer_sum_cpu_ns']/1e9,'MdirGB',next(t['sentinel_logical_bytes'] for t in x['directories'] if t['site']=='coherent')/1e9,'prune',x['producer']['pruned_accepted_starts'],flush=True)
assert len(normal)==30 and len(diag)==18
assert all(x['child_max_swap_kib']==0 for x in normal)
assert all(x['hwm_delta_kib']<=x['hwm_budget_kib'] for x in pairs)
bycase={}
for x in normal+diag:assert bycase.setdefault(x['case'],x['model_sha256'])==x['model_sha256']
