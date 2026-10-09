"""Compare scheduling weights with measured job cost over complete corpora."""
import bisect,collections,json,math
from pathlib import Path
ROOT=Path('/root/code/tokenizers-perfetto-results')
def analyze(run):
 pid,bufs=json.loads((run/'events.json').read_text());jobs=[];byround=collections.defaultdict(list)
 for tid,events in bufs:
  jj=sorted([e for e in events if e['name']=='prepare.job'],key=lambda e:e['ts']);starts=[e['ts'] for e in jj]
  for j in jj:j['matched']=j['scan_ns']=j['finish_ns']=j['groups']=0
  for e in events:
   if e['name'] not in ['prepare.scan','prepare.finish']:continue
   i=bisect.bisect_right(starts,e['ts'])-1
   if i<0:continue
   j=jj[i]
   if e['ts']+e['dur']>j['ts']+j['dur']:continue
   if e['name']=='prepare.scan':j['matched']+=e['fields'][2];j['scan_ns']+=e['dur']
   else:j['groups']+=e['fields'][3];j['finish_ns']+=e['dur']
  jobs.extend(jj)
 for j in jobs:byround[j['round']].append(j)
 def corr(key,weighted):
  xy=xx=yy=0
  for jj in byround.values():
   vals=[key(j) for j in jj];total=sum(vals);time=sum(j['dur'] for j in jj)
   if not total or not time or len(jj)<2:continue
   w=time if weighted else 1
   for j,v in zip(jj,vals):
    x=v/total-1/len(jj);y=j['dur']/time-1/len(jj);xy+=w*x*y;xx+=w*x*x;yy+=w*y*y
  return xy/math.sqrt(xx*yy) if xx*yy else 0
 out=dict(jobs=len(jobs),raw=sum(j['fields'][2] for j in jobs),matched=sum(j['matched'] for j in jobs),scan_ns=sum(j['scan_ns'] for j in jobs),finish_ns=sum(j['finish_ns'] for j in jobs),within_batch_time_weighted_correlation={name:corr(key,True) for name,key in [('raw_positions',lambda j:j['fields'][2]),('matched_positions',lambda j:j['matched']),('groups',lambda j:j['groups'])]})
 (run/'job-costs.json').write_text(json.dumps(out,indent=2));return out
if __name__=='__main__':
 for run in sorted((ROOT/'runs').glob('*w4-detail-r[123]')):
  d=analyze(run);print(run.name,round(d['matched']/d['raw'],4),d['within_batch_time_weighted_correlation'])
