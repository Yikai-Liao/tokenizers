from pathlib import Path
import json, statistics
OUT=Path('/root/code/tokenizers-simplification-results/online-initial')

def stats(values):
    return dict(n=len(values),median=statistics.median(values),minimum=min(values),maximum=max(values),raw=values)

def summarize():
    rows=[json.loads(l) for l in (OUT/'runs.jsonl').read_text().splitlines()]
    groups=[]
    for case,kind,arm in sorted({(r['case'],r['kind'],r['arm']) for r in rows if r['valid']}):
        rr=[r for r in rows if r['valid'] and (r['case'],r['kind'],r['arm'])==(case,kind,arm)]
        group=dict(case=case,kind=kind,arm=arm,n=len(rr),metrics={})
        for key in rr[0]['metrics']:
            group['metrics'][key]=stats([r['metrics'][key] for r in rr])
        if rr[0].get('initial_phase'):
            group['initial_phase']={k:stats([r['initial_phase'][0][k] for r in rr]) for k in rr[0]['initial_phase'][0]}
        if rr[0].get('phases'):
            group['phases']=[dict(name=p['name'],**{k:stats([next(q[k] for q in r['phases'] if q['name']==p['name']) for r in rr]) for k in p if k!='name'}) for p in rr[0]['phases']]
            group['unassigned_wall_seconds']=stats([r['metrics']['train_seconds']-sum(p['wall_seconds'] for p in r['phases']) for r in rr])
            group['unassigned_cpu_seconds']=stats([r['metrics']['train_cpu_seconds']-sum(p['cpu_seconds'] for p in r['phases']) for r in rr])
        groups.append(group)
    result=dict(attempts=len(rows),valid=sum(r['valid'] for r in rows),excluded=[r for r in rows if not r['valid']],groups=groups)
    (OUT/'summary-full.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    result=summarize()
    for case,kind in sorted({(g['case'],g['kind']) for g in result['groups'] if g['kind'].startswith('phases')}):
        groups={g['arm']:g for g in result['groups'] if (g['case'],g['kind'])==(case,kind)}
        if not {'main','candidate'}<=groups.keys():continue
        a,b=groups['main'],groups['candidate']
        m=lambda g,k:g['metrics'][k]['median']
        print(case,kind,'main/candidate wall',m(a,'train_seconds'),m(b,'train_seconds'),'CPU',m(a,'train_cpu_seconds'),m(b,'train_cpu_seconds'),'HWM',m(a,'process_hwm_kib_before_validation'),m(b,'process_hwm_kib_before_validation'))
        for x,y in zip(a['phases'],b['phases']):
            print(x['name'],round(x['wall_seconds']['median'],3),round(y['wall_seconds']['median'],3),round(y['wall_seconds']['median']-x['wall_seconds']['median'],3),round(y['cpu_seconds']['median']-x['cpu_seconds']['median'],3))
