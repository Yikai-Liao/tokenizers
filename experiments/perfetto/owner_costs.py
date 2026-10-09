"""Explain owner cost over every batch, rather than selected slow examples."""
import bisect,collections,json,math,statistics
from pathlib import Path
ROOT=Path('/root/code/tokenizers-perfetto-results')
SUB=['commit.group','commit.counts','commit.completed','commit.aggregate','commit.encode_publish','commit.prefix']
def analyze(run):
 pid,bufs=json.loads((run/'events.json').read_text());owners=[];byround=collections.defaultdict(list)
 for tid,events in bufs:
  oo=sorted([e for e in events if e['name']=='commit.owner'],key=lambda e:e['ts']);starts=[e['ts'] for e in oo]
  for o in oo:o['sub']=collections.Counter();o['positions']=o['groups']=0;o['tid']=tid
  for e in events:
   if e['name'] not in SUB:continue
   i=bisect.bisect_right(starts,e['ts'])-1
   if i<0:continue
   o=oo[i]
   if e['ts']+e['dur']>o['ts']+o['dur']:continue
   o['sub'][e['name']]+=e['dur']
   if e['name']=='commit.encode_publish':o['positions']+=e['fields'][2];o['groups']+=e['fields'][3]
  owners.extend(oo)
 for o in owners:byround[o['round']].append(o)
 critical=[max(oo,key=lambda o:o['dur']) for oo in byround.values()]
 costs={scope:{name:sum(o['sub'][name] for o in oo) for name in SUB} for scope,oo in [('all_owners',owners),('critical_owner_each_batch',critical)]}
 for scope,oo in [('all_owners',owners),('critical_owner_each_batch',critical)]:costs[scope]['total_ns']=sum(o['dur'] for o in oo)
 counts=sum(o['fields'][1] for o in owners);completed=sum(o['fields'][3] for o in owners);positions=sum(o['positions'] for o in owners)
 out=dict(owners=len(owners),batches=len(byround),changes=counts,completed_births=completed,routed_positions=positions,routed_groups=sum(o['groups'] for o in owners),costs=costs,counts_ns_per_ref=costs['all_owners']['commit.counts']/counts,completed_ns_per_birth=costs['all_owners']['commit.completed']/completed)
 # Fractions normalized within each batch separate owner imbalance from batch size.
 def correlation(key):
  xx=[];yy=[]
  for oo in byround.values():
   values=[key(o) for o in oo];total=sum(values);duration=sum(o['dur'] for o in oo)
   if total and duration and len(oo)>1:
    xx.extend(v/total-1/len(oo) for v in values);yy.extend(o['dur']/duration-1/len(oo) for o in oo)
  sx=sum(x*x for x in xx);sy=sum(y*y for y in yy)
  return sum(x*y for x,y in zip(xx,yy))/math.sqrt(sx*sy) if sx*sy else 0
 out['within_batch_cost_correlation']={name:correlation(key) for name,key in [('changes',lambda o:o['fields'][1]),('completed_births',lambda o:o['fields'][3]),('routed_positions',lambda o:o['positions'])]}
 (run/'owner-costs.json').write_text(json.dumps(out,indent=2));return out
if __name__=='__main__':
 for run in sorted((ROOT/'runs').glob('*w4-detail-r[123]')):
  o=analyze(run);print(run.name,'ref',o['changes'],'completed',o['completed_births'],'routed',o['routed_positions'],'ns/ref',round(o['counts_ns_per_ref'],2),'ns/completed',round(o['completed_ns_per_birth'],2),'critical%',{k:round(v/o['costs']['critical_owner_each_batch']['total_ns']*100,1) for k,v in o['costs']['critical_owner_each_batch'].items() if k!='total_ns'},o['within_batch_cost_correlation'])
