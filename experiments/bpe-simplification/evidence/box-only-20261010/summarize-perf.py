"""Summarize whole-process perf counters, separately from training-only metrics."""
import csv,json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1])
records=json.loads((root/'core-runs.json').read_text())
rows=[]
for r in records:
    path=root/f"core-{r['case']}-b{r['block']}-{r['arm']}"/'perf.csv'
    values={}
    for fields in csv.reader(path.read_text().splitlines(),delimiter=';'):
        if not fields or fields[0].startswith('#'):continue
        assert len(fields)>=5,fields
        assert fields[0] not in ('<not counted>','<not supported>'),fields
        values[fields[2]]=float(fields[0])
        assert float(fields[4])>=99,fields
    values['instructions_per_cycle']=values['instructions']/values['cycles']
    rows.append(dict(case=r['case'],arm=r['arm'],block=r['block'],warmup=r['warmup'],counters=values))
summary=[]
for case in sorted(set(r['case'] for r in rows)):
    formal=[r for r in rows if r['case']==case and not r['warmup']]
    result=dict(case=case,scope='whole process including loading, serialization and all threads; hardware counters are supplied by the VPS virtual PMU')
    for arm in ['enum','box']:
        samples=[r for r in formal if r['arm']==arm]
        result[arm]={k:statistics.median(r['counters'][k] for r in samples) for k in samples[0]['counters']}
    result['paired_median_delta_percent']={k:statistics.median((next(r for r in formal if r['arm']=='box' and r['block']==b)['counters'][k]/next(r for r in formal if r['arm']=='enum' and r['block']==b)['counters'][k]-1)*100 for b in sorted(set(r['block'] for r in formal))) for k in formal[0]['counters'] if all(r['counters'][k]>0 for r in formal)}
    summary.append(result)
(root/'perf-runs.json').write_text(json.dumps(rows,indent=2)+'\n')
(root/'perf-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for result in summary:print(result['case'],result['paired_median_delta_percent'])
