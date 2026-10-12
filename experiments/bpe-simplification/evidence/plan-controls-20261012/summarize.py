from pathlib import Path
import json,statistics,collections,csv
e=Path(__file__).resolve().parent
tables={};all_rows=[];inclusions=[]
for stage,base,arms in [('scratch','baseline',['scratch']),('publisher','baseline',['producer']),('frontier','eager',['single','frontier'])]:
 folder=e/stage;valid=[];normals=[]
 for suite in ['primary','reverse','primary-whitespace-repeat','single','single-cn-reverse','controls']:
  p=folder/f'{suite}.json'
  if not p.exists():continue
  for row in json.loads(p.read_text()):
   record=row|dict(stage=stage,suite=suite)
   included=not(stage=='publisher' and suite=='primary' and row['case']=='zh-whitespace-w4')
   inclusions.append(dict(stage=stage,suite=suite,case=row['case'],arm=row['arm'],included_in_performance=included,reason=None if included else 'Exclude the whole pair: baseline overlapped scratch gzip archiving; replacement pair retained.'))
   normals.append(record);all_rows.append(record)
   if included and (suite in ['primary','reverse','primary-whitespace-repeat']):valid.append(record)
 table=[]
 for case in sorted(set(x['case'] for x in valid)):
  bs=[x for x in valid if x['case']==case and x['arm']==base];assert len(bs)==2,(stage,case,len(bs))
  for arm in arms:
   cs=[x for x in valid if x['case']==case and x['arm']==arm];assert len(cs)==2
   entry=dict(case=case,baseline=base,candidate=arm)
   for field in ['train_seconds','train_cpu_seconds']:
    b=statistics.mean(x[field] for x in bs);c=statistics.mean(x[field] for x in cs)
    entry[field]=dict(baseline=b,candidate=c,delta_pct=100*(c/b-1))
   entry['phases']={}
   for field in bs[0]['phases']:
    b=statistics.mean(x['phases'].get(field,0) for x in bs);c=statistics.mean(x['phases'].get(field,0) for x in cs)
    entry['phases'][field]=dict(baseline=b,candidate=c,delta_pct=100*(c/b-1) if b else None)
   b=max(x['process_hwm_kib_before_validation'] for x in bs);c=max(x['process_hwm_kib_before_validation'] for x in cs)
   entry['peak_hwm_kib']=dict(baseline=b,candidate=c,delta_pct=100*(c/b-1),passes_budget=c-b<=max(.05*b,16*1024))
   entry['pairs']=[]
   for candidate in cs:
    baseline=next(x for x in bs if x['suite']==candidate['suite'])
    entry['pairs'].append(dict(suite=candidate['suite'],cpu_delta_pct=100*(candidate['train_cpu_seconds']/baseline['train_cpu_seconds']-1),wall_delta_pct=100*(candidate['train_seconds']/baseline['train_seconds']-1)))
   table.append(entry)
 tables[stage]=table
 (folder/'all-normal.json').write_text(json.dumps(normals,indent=2)+'\n')

diagnostics={}
for stage in ['scratch','publisher','frontier']:
 p=e/stage/'diagnostic.json';records=[]
 for row in json.loads(p.read_text()):
  record=dict(case=row['case'],arm=row['arm'],directory_reports=row['directory_reports'],allocation_reports=row['allocation_reports'])
  if stage=='publisher':record|=dict(publisher=row['publisher_reports'][-1],producer=row['producer_reports'][-1])
  if stage=='frontier':
   windows=row['window_reports'];work=row['window_work_reports'][-1]
   record|=dict(work=work,encoding=row['encoding_reports'][-1],publisher=row['publisher_reports'][-1],windows=len(windows),cohort_histogram=dict(collections.Counter(w['cohorts'] for w in windows)),stop_reasons=dict(collections.Counter(w['stop_reason'] for w in windows)),guard_false_positives=sum(w['guard_false_positive_at_drain'] for w in windows),coherent_rare_accepted_starts=sum(w['coherent_rare_accepted_starts'] for w in windows),coherent_rare_final_coordinates=sum(w['coherent_rare_final_coordinates'] for w in windows),coherent_rare_final_encoded_bytes=sum(w['coherent_rare_final_encoded_bytes'] for w in windows),max_attempt_high_water_bytes=max((w['attempt_high_water_bytes'] for w in windows),default=0),max_retained_checkpoint_bytes=max((w['retained_checkpoint_bytes'] for w in windows),default=0))
  records.append(record)
 diagnostics[stage]=records
(e/'summary.json').write_text(json.dumps(dict(performance=tables,diagnostics=diagnostics),indent=2)+'\n')
(e/'performance-inclusions.json').write_text(json.dumps(inclusions,indent=2)+'\n')
with (e/'normal-runs.csv').open('w') as f:
 writer=csv.DictWriter(f,lineterminator='\n',fieldnames=['stage','suite','case','arm','repeat','train_seconds','train_cpu_seconds','process_hwm_kib_before_validation','fresh_batches','reuse_batches','child_max_swap_kib','model_sha256']);writer.writeheader()
 for row in all_rows:writer.writerow({k:row[k] for k in writer.fieldnames})
for stage,table in tables.items():
 print(stage)
 for r in table:print(r['case'],r['candidate'],*[round(r[k]['delta_pct'],3) for k in ['train_seconds','train_cpu_seconds']],r['peak_hwm_kib'])
