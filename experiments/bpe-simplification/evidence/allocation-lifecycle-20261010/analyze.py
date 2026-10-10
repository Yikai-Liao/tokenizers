from pathlib import Path
import json, statistics
root=Path('/tmp/bpe-allocation-confirmation')
records=json.loads((root/'core-runs.json').read_text())
summary=[]
for case in sorted(set(r['case'] for r in records)):
    rows=[r for r in records if r['case']==case]
    arms={}
    for arm in ['arena','arena-off','box-mutex','box-tls']:
        samples=[r for r in rows if r['arm']==arm]
        if not samples: continue
        phases={name:{metric:statistics.median(r['diagnostic']['times'][i][j] for r in samples) for j,metric in enumerate(['wall_seconds','cpu_seconds'])} for i,name in enumerate(samples[0]['diagnostic']['phases'])}
        fields=list(samples[0]['diagnostic']['counts'][0])
        totals={field:statistics.median(sum(c[field] for c in r['diagnostic']['counts']) for r in samples) for field in fields}
        ending_free=[sum(c['heap_frees'] for c in r['diagnostic']['counts'][8:11]) for r in samples]
        arms[arm]=dict(samples=len(samples),metrics={k:statistics.median(r['metrics'][k] for r in samples) for k in ['train_seconds','train_cpu_seconds','process_hwm_kib_before_validation']},phases=phases,totals=totals,ending_heap_frees=statistics.median(ending_free),ending_free_percent=statistics.median(v/sum(c['heap_frees'] for c in r['diagnostic']['counts'])*100 for v,r in zip(ending_free,samples)),retained_bump_bytes=statistics.median(r['diagnostic']['retained_bump_bytes'] for r in samples),retained_scratch_capacity_bytes=statistics.median(r['diagnostic']['retained_scratch_capacity_bytes'] for r in samples),all_heap_allocations_released=all(sum(c['heap_allocations'] for c in r['diagnostic']['counts'])==sum(c['heap_frees'] for c in r['diagnostic']['counts']) for r in samples),all_arena_handles_retired=all(sum(c['arena_allocations'] for c in r['diagnostic']['counts'])==sum(c['arena_retires'] for c in r['diagnostic']['counts']) for r in samples))
    comparisons={}
    for a,b in [('arena','arena-off'),('box-mutex','box-tls'),('arena-off','box-mutex'),('arena','box-tls')]:
        blocks=sorted(set(r['block'] for r in rows if r['arm']==a)&set(r['block'] for r in rows if r['arm']==b))
        if not blocks:continue
        comparisons[f'{a}_to_{b}']={key:statistics.median((next(r for r in rows if r['arm']==b and r['block']==block)['metrics'][key]/next(r for r in rows if r['arm']==a and r['block']==block)['metrics'][key]-1)*100 for block in blocks) for key in ['train_seconds','train_cpu_seconds','process_hwm_kib_before_validation']}
    summary.append(dict(case=case,arms=arms,paired_median_delta_percent=comparisons))
(root/'phase-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for item in summary:
    print(item['case'])
    for arm,v in item['arms'].items():
        cleanup=sum(v['phases'][p]['wall_seconds'] for p in ['drop_index','drop_corpus','drop_codec'])
        print(arm,'wall/CPU',round(v['metrics']['train_seconds'],3),round(v['metrics']['train_cpu_seconds'],3),'cleanup',round(cleanup,5),'end/total frees',v['ending_heap_frees'],v['totals']['heap_frees'],'bump MiB',round(v['retained_bump_bytes']/1024/1024,1),'scratch KiB',round(v['retained_scratch_capacity_bytes']/1024,1),'all released',v['all_heap_allocations_released'])
    print('comparisons',item['paired_median_delta_percent'])
