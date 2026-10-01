#!/usr/bin/env python3
"""Reproduce saved 43-case fits and logical budget examples; no new training."""
import argparse,pathlib,json,math,gzip,time,statistics,hashlib
BASE=pathlib.Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=pathlib.Path,default=BASE/'posting-threshold-empirical-fits.json')
OUTPUT=parser.parse_args().output
start=time.monotonic(); paths=sorted(p for p in (BASE/'results/posting-distribution').glob('*-r*.summary.json')); rows=[]
idx={'p50':2,'p90':4,'p99':6,'max':7}; models=('constant','linear','power'); predictors=('bytes','physical_edges')
def fit(rs,metric,predictor,model):
 xs=[math.log(r['x'][predictor]) for r in rs]; ys=[math.log(r['y'][metric]) for r in rs]; xm=statistics.mean(xs);ym=statistics.mean(ys)
 gamma={'constant':0.,'linear':1.}.get(model)
 if gamma is None:gamma=sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sum((x-xm)**2 for x in xs)
 return {'x_ref':math.exp(xm),'a':math.exp(ym),'gamma':gamma}
def pred(f,x):return f['a']*(x/f['x_ref'])**f['gamma']
def errors(ps):
 if not ps:return None
 es=[p['prediction']/p['actual'] for p in ps]
 return {'n':len(es),'MAPE':statistics.mean(abs(e-1) for e in es),'geomean_absolute_factor':math.exp(statistics.mean(abs(math.log(e)) for e in es)),'max_absolute_factor':max(max(e,1/e) for e in es),'median_signed_relative_error':statistics.median(e-1 for e in es)}
for p in paths:
 d=json.loads(p.read_text());i=d['input']; t=d['terminal']; caps=d['probe_summary']['capacity_bounds']; agebounds=d['probe_summary']['retired_age_bucket_upper_bounds_rules']; assert d['reached_target'];assert d['invariant_checks']=='PASS'; assert d['probe_summary']['grows']==0
 row={'case':p.name.removesuffix('.summary.json'),'language':i['language'],'split':i['split'],'size_mib':i['size_mib_label'],'rules':d['actual_rules'],'held_out':i['held_out'],'input':i,'x':{'bytes':i['bytes'],'physical_edges':d['physical_edges']},'y':{k:d['live_length_quantiles'][j] for k,j in idx.items()},'heap_p50':d['heap_length_quantiles'][2],'weighted_p50':d['weighted_frequency_quantiles'][2],'live_postings':sum(v for _,v in d['exact_live_length_histogram']),'live_heap_postings':d['inventory']['heap_postings'],'birth_slots':d['probe_summary']['birth_slots'],'physical_rewrites':d['probe_summary']['physical_rewrites'],'upper_birth':d['physical_edges']+2*d['probe_summary']['physical_rewrites'],'allocated_bytes':sum(t['allocated_bytes']),'retired_bytes':sum(t['retired_bytes']),'live_bytes':t['baseline_payload_live'],'baseline_payload_peak':t['baseline_payload_peak'],'policies':[]}
 assert row['birth_slots']<=row['upper_birth']; assert row['allocated_bytes']<=16*d['physical_edges'];assert row['allocated_bytes']-row['retired_bytes']==row['live_bytes']
 for j,c in enumerate(caps):
  ab=sum(t['allocated_bytes'][:j+1]);ac=sum(t['allocated_count'][:j+1]);rb=sum(t['retired_bytes'][:j+1]);rc=sum(t['retired_count'][:j+1]); cc=sum(d['probe_summary']['chosen_bytes'][:j+1]);fc=sum(d['probe_summary']['floor_bytes'][:j+1]);counts=d['probe_summary']['retired_age_counts'];bs=d['probe_summary']['retired_byte_age_sums'];survs=d['survivor_age_counts'];survb=d['survivor_age_bytes']; nb=len(agebounds)
  shortc=sum(counts[k*nb+a] for k in range(j+1) for a,b in enumerate(agebounds) if b is not None and b<=127);shortb=sum(bs[k*nb+a] for k in range(j+1) for a,b in enumerate(agebounds) if b is not None and b<=127)
  longc=sum(counts[k*nb+a] for k in range(j+1) for a,b in enumerate(agebounds) if b is None or (a>0 and agebounds[a-1]>=2047)); longb=sum(bs[k*nb+a] for k in range(j+1) for a,b in enumerate(agebounds) if b is None or (a>0 and agebounds[a-1]>=2047))
  sc=sum(survs[:(j+1)*nb]); sb=sum(survb[:(j+1)*nb]); soldc=sum(survs[k*nb+a] for k in range(j+1) for a,b in enumerate(agebounds) if b is None or (a>0 and agebounds[a-1]>=2047));soldb=sum(survb[k*nb+a] for k in range(j+1) for a,b in enumerate(agebounds) if b is None or (a>0 and agebounds[a-1]>=2047))
  pol={'threshold_bytes':4*c if j<len(caps)-1 else 'all','allocated_count':ac,'allocation_fraction':ac/sum(t['allocated_count']),'allocated_bytes':ab,'allocated_payload_fraction':ab/row['allocated_bytes'],'retired_bytes':rb,'retired_fraction_of_allocated':rb/ab if ab else 0,'retired_count':rc,'chosen_retired_bytes':cc,'floor_retired_bytes':fc,'short_retired_count_le127':shortc,'short_retired_bytes_le127':shortb,'long_retired_count_ge2048':longc,'long_retired_bytes_ge2048':longb,'survivor_count':sc,'survivor_bytes':sb,'survivor_age_ge2048_count':soldc,'survivor_age_ge2048_bytes':soldb,'payload_peak':t['policy_payload_peak'][j],'payload_terminal':t['policy_payload_current'][j],'payload_peak_increase':t['policy_payload_peak'][j]-row['baseline_payload_peak']}
  assert pol['retired_bytes']==cc+fc; assert ac-rc==sc; row['policies'].append(pol)
 # capacity only RSS proxy: each sampled baseline RSS + same-time retained requested bytes;
 # never label this an actual alternative RSS or use it for runtime ranking.
 raw=json.load(gzip.open(p.with_name(p.name.replace('.summary.json','.probe.json.gz'))));tr=raw['trace']; hwm=tr[-1]['memory']['VmHWM']; ini=d['initialize']['memory']['VmHWM']; last=t['memory']['VmRSS']
 row['diagnostic_phase']={'initialize_hwm':ini,'terminal_hwm':hwm,'terminal_rss':last,'trace_points':len(tr),'hwm_was_set_by_initialize':ini==hwm,'sample_max_rss':max(z['memory']['VmRSS'] for z in tr),'initial_route_phase_headroom_proxy':ini-d['initialize']['memory']['VmRSS'],'terminal_headroom_proxy':hwm-last,'policies':[]}
 for j,c in enumerate(caps):
  margin=min(hwm-z['memory']['VmRSS']-sum(z['retired_bytes'][:j+1]) for z in tr); worst=max(tr,key=lambda z:z['memory']['VmRSS']+sum(z['retired_bytes'][:j+1])); row['diagnostic_phase']['policies'].append({'threshold_bytes':4*c if j<len(caps)-1 else 'all','min_requested_retention_margin_proxy':margin,'worst_rules':worst['rules'],'max_rss_plus_requested_retirement_proxy':hwm-margin})
 rows.append(row)
train=[r for r in rows if not r['held_out']];hold=[r for r in rows if r['held_out']];groupkeys=sorted({(r['language'],r['split'],r['rules']) for r in train});local=[]
for lang,split,m in groupkeys:
 rs=[r for r in train if (r['language'],r['split'],r['rules'])==(lang,split,m)]; hs=[r for r in hold if (r['language'],r['split'],r['rules'])==(lang,split,m)]
 for metric in idx:
  for predictor in predictors:
   for model in models:
    f=fit(rs,metric,predictor,model);ps=[{'case':r['case'],'prediction':pred(f,r['x'][predictor]),'actual':r['y'][metric]} for r in hs];local.append({'language':lang,'split':split,'rules':m,'metric':metric,'predictor':predictor,'model':model,'fit':f,'train_errors':errors([{'prediction':pred(f,r['x'][predictor]),'actual':r['y'][metric]} for r in rs]),'holdout_predictions':ps,'holdout_errors':errors(ps)})
cv=[]
for split in ('none','whitespace_split'):
 for m in (4000,16000):
  rs=[r for r in train if r['split']==split and r['rules']==m]; langs=sorted({r['language'] for r in rs})
  for metric in idx:
   for predictor in predictors:
    for model in models:
     ps=[]
     for lang in langs:
      f=fit([r for r in rs if r['language']!=lang],metric,predictor,model)
      ps += [{'case':r['case'],'prediction':pred(f,r['x'][predictor]),'actual':r['y'][metric]} for r in rs if r['language']==lang]
     cv.append({'split':split,'rules':m,'metric':metric,'predictor':predictor,'model':model,'held_languages':len(langs),'errors':errors(ps)})
aggregates=[]
for metric in idx:
 for predictor in predictors:
  for model in models:
   fs=[f for f in local if f['metric']==metric and f['predictor']==predictor and f['model']==model]; ps=[p for f in fs for p in f['holdout_predictions'] if '-32m-' in p['case']]
   aggregates.append({'metric':metric,'predictor':predictor,'model':model,'32m_holdout_errors':errors(ps)})
result={'method':{'fit':'OLS in log space; constant gamma=0; linear through origin gamma=1; power free gamma; normalized x_ref geomean of training x and a geomean training y; each local fit uses three scales, one language, one split, fixed actual rules. All types quantiles include inline.','evaluations':'32MiB holdout; zh512MiB separate extrapolation and source sampling shift; leave-one-language none has four languages, whitespace only two; no runtime rankings.','phase_proxy':'baseline diagnostic RSS + same-time requested retired payload ONLY A CAPACITY PROXY; ignores resident/touched pages, heap free retained pages, arena chunk slack, metadata, altered 24B diagnostic layout and event gaps; not alternative actual RSS.'},'summary_count':len(rows),'train_count':len(train),'heldout_count':len(hold),'elapsed_seconds':time.monotonic()-start,'sources':[{'path':str(p.relative_to(BASE)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],'cases':rows,'local_fits':local,'leave_one_language':cv,'32m_aggregates':aggregates}
print('summaries/train/hold',len(rows),len(train),len(hold),'elapsed',result['elapsed_seconds'])
print('32m holdouts: metric predictor model MAPE meanfactor maxfactor')
for a in aggregates:
 if a['predictor']=='physical_edges':print(a['metric'],a['predictor'],a['model'],{k:round(v,3) if isinstance(v,float) else v for k,v in a['32m_holdout_errors'].items()})
print('512 extrapolation')
for f in local:
 if f['language']=='zh' and f['split']=='none' and f['rules']==16000 and f['predictor']=='physical_edges' and f['model']=='power':print(f['metric'],'gamma',f['fit']['gamma'],f['holdout_predictions'])

# Include the budget examples and compact serialization used by the archived
# empirical-fits JSON. This block was originally a separate inline postprocess.
budgets=[]
for fraction in (.01,.05,.10):
 choices=[]
 for r in result['cases']:
  feasible=[x for x in r['policies'] if x['payload_peak']<=r['baseline_payload_peak']*(1+fraction)]
  selected=max(feasible,key=lambda x:float('inf') if x['threshold_bytes']=='all' else x['threshold_bytes']) if feasible else {'threshold_bytes':0,'allocation_fraction':0}
  choices.append({'case':r['case'],'threshold_bytes':selected['threshold_bytes'],'allocation_fraction':selected['allocation_fraction']})
 budgets.append({'budget_extra_fraction_over_baseline_payload_peak':fraction,'choices':choices,'minimum_allocation_coverage':min(x['allocation_fraction'] for x in choices),'median_allocation_coverage':statistics.median(x['allocation_fraction'] for x in choices),'maximum_allocation_coverage':max(x['allocation_fraction'] for x in choices)})
result['logical_payload_budget_examples']=budgets
OUTPUT.write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n')
