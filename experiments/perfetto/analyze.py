"""Cumulative capacity accounting over all batches, plus robustness to outliers.
Task spans measure occupied worker wall time. Kernel scheduling separately
measures on-CPU time: do not treat task span time as CPU utilization.
"""
import argparse,collections,csv,json,math,statistics
from pathlib import Path
ROOT=Path('/root/code/tokenizers-perfetto-results')
def ratio(a,b): return a/b if b else 0

def parallel_metrics(phase,tasks,w):
 duration=phase['dur'];busy=sum(t['dur'] for t in tasks)
 if not tasks:return dict(tasks=0,duration_ns=duration,busy_ns=0,capacity_ns=w*duration,boundary_ns=w*duration,insufficient_tasks_ns=0,ready_gap_ns=0,max_task_ns=0,job_lower_bound_ns=0,inner_span_ns=0)
 start=min(t['ts'] for t in tasks);end=max(t['ts']+t['dur'] for t in tasks)
 assert phase['ts']<=start and end<=phase['ts']+duration
 changes=collections.defaultdict(lambda:[0,0])
 for t in tasks:
  changes[t['ts']][0]+=1;changes[t['ts']+t['dur']][0]-=1;changes[t['ts']+t['dur']][1]+=1
 active=0;unfinished=len(tasks);last=start;lack=gap=0
 for ts,(delta,completed) in sorted(changes.items()):
  length=ts-last;capacity=min(w,unfinished)
  lack+=(w-capacity)*length;gap+=(capacity-active)*length
  active+=delta;unfinished-=completed;last=ts
 boundary=w*(duration-(end-start))
 assert abs((w*duration-busy)-(boundary+lack+gap))<5
 return dict(tasks=len(tasks),duration_ns=duration,busy_ns=busy,capacity_ns=w*duration,boundary_ns=boundary,insufficient_tasks_ns=lack,ready_gap_ns=gap,max_task_ns=max(t['dur'] for t in tasks),job_lower_bound_ns=max(max(t['dur'] for t in tasks),busy/w),inner_span_ns=end-start)

def aggregate(rows):
 keys=['duration_ns','busy_ns','capacity_ns','boundary_ns','insufficient_tasks_ns','ready_gap_ns','job_lower_bound_ns','inner_span_ns']
 d={k:sum(r[k] for r in rows) for k in keys}
 d.update(batches=len(rows),task_occupancy=ratio(d['busy_ns'],d['capacity_ns']),lost_ns=d['capacity_ns']-d['busy_ns'],fewer_than_workers_batches=sum(r['tasks']<r['workers'] for r in rows),one_task_batches=sum(r['tasks']==1 for r in rows),existing_granularity_max_saving_ns=d['inner_span_ns']-d['job_lower_bound_ns'],median_tasks=statistics.median(r['tasks'] for r in rows) if rows else 0)
 return d

def analyze_run(run):
 result=json.loads((run/'result.json').read_text());pid,buffers=json.loads((run/'events.json').read_text());events=[]
 for tid,ev in buffers:
  for e in ev:e['tid']=tid
  events.extend(ev)
 byround=collections.defaultdict(lambda:collections.defaultdict(list));byname=collections.defaultdict(list)
 for e in events:byround[e['round']][e['name']].append(e);byname[e['name']].append(e)
 assert len(byname['training'])==1
 training=byname['training'][0]
 workers=result['workers'];rows=[];excluded=collections.Counter()
 for round_id,names in byround.items():
  if not names.get('round'):continue
  r=names['round'][0]
  for phase,jobs in [('prepare','prepare.job'),('commit','commit.owner'),('apply','apply.job')]:
   if len(names[phase])!=1 or (phase=='apply' and not names[jobs]):continue
   if phase=='prepare' and names.get('prepare.aa'):
    excluded['aa_rounds']+=1;excluded['aa_duration_ns']+=names[phase][0]['dur'];continue
   row=dict(round=round_id,phase=phase,workers=workers,vocab_before=r['fields'][0],rules=r['fields'][1],raw_positions=r['fields'][2],**parallel_metrics(names[phase][0],names[jobs],workers))
   rows.append(row)
 summary=dict(case=result['case'],workers=workers,arm=result['arm'],rep=result['rep'],pid=pid,threads=len(buffers),slices=len(events),train_seconds=result['metrics']['train_seconds'],trace_training_seconds=training['dur']/1e9,rounds=len(byname['round']),aa=dict(excluded),stages={n:dict(count=len(v),sum_seconds=sum(x['dur'] for x in v)/1e9) for n,v in byname.items() if n in ['select','prepare','apply','commit','release_candidates','release_events','initial_index','materialize','prepare.aa','vocabulary','corpus_plan','cleanup','output_model']},subphases={n:dict(count=len(v),sum_seconds=sum(x['dur'] for x in v)/1e9) for n,v in byname.items() if n.startswith(('prepare.','commit.','apply.'))})
 # Scope coverage must agree with true timed call; pool worker count must be complete.
 assert abs(summary['train_seconds']-summary['trace_training_seconds'])<.05
 for phase in ['prepare','commit','apply']:
  if phase=='apply' and not byname['apply.job']:continue
  pr=[r for r in rows if r['phase']==phase]
  s=aggregate(pr);ntrim=math.ceil(len(pr)*.01)
  byloss=sorted(pr,key=lambda r:r['capacity_ns']-r['busy_ns'],reverse=True)
  s['remove_top_1pct_loss']=aggregate(byloss[ntrim:]);s['top_1pct_lost_share']=ratio(sum(r['capacity_ns']-r['busy_ns'] for r in byloss[:ntrim]),s['lost_ns'])
  s['by_task_count']={label:aggregate([r for r in pr if test(r)]) for label,test in [('below_workers',lambda r:r['tasks']<workers),('equal_workers',lambda r:r['tasks']==workers),('above_workers',lambda r:r['tasks']>workers)]}
  s['by_size']={label:aggregate([r for r in pr if lo<=r['raw_positions']<hi]) for label,lo,hi in [('<1K',0,1000),('1K-4K',1000,4096),('4K-16K',4096,16384),('16K-64K',16384,65536),('64K+',65536,math.inf)]}
  s['by_progress']={label:aggregate([r for r in pr if lo<=r['vocab_before']<hi]) for label,lo,hi in [('below10K',0,10000),('10K-25K',10000,25000),('25K+',25000,math.inf)]}
  summary[phase]=s
 # Geometry: count direct producers and how cost relates to position counts.
 scans=byname['prepare.scan'];tasks=byname['prepare.task']
 summary['prepare_geometry']=dict(raw=sum(e['fields'][1] for e in scans),matched=sum(e['fields'][2] for e in scans),whole_tasks=sum(e['fields'][2] for e in tasks),direct_tasks=sum(e['fields'][3] for e in tasks),tasks=len(tasks),partial_task_seconds=sum(e['dur'] for e in tasks if not e['fields'][2])/1e9,whole_task_seconds=sum(e['dur'] for e in tasks if e['fields'][2])/1e9)
 if tasks:
  summary['prepare_geometry']['by_range']={label:dict(tasks=len(v),sum_seconds=sum(e['dur'] for e in v)/1e9) for label,test in [('whole',lambda e:e['fields'][2]),('partial',lambda e:not e['fields'][2])] for v in [[e for e in tasks if test(e)]]}
 (run/'analysis.json').write_text(json.dumps(summary,indent=2))
 with (run/'batches.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 return summary

def collect():
 all=[]
 for run in sorted((ROOT/'runs').iterdir()):
  if (run/'events.json').exists() and (run/'result.json').exists(): all.append(analyze_run(run))
 (ROOT/'analysis-all.json').write_text(json.dumps(all,indent=2))
 for s in all:
  p=s['prepare'];c=s['commit'];print(s['case'],s['workers'],s['arm'],s['rep'],'rounds',s['rounds'],'prepare',round(p['task_occupancy'],3),'commit',round(c['task_occupancy'],3),'top1 loss',round(p['top_1pct_lost_share'],3),round(c['top_1pct_lost_share'],3))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run',nargs='?');a=p.parse_args()
 if a.run: print(json.dumps(analyze_run(Path(a.run)),indent=2))
 else:collect()
