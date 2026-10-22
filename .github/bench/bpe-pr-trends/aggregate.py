"""Combine version artifacts; summarize three-run medians and exact-model checks."""
from pathlib import Path
import argparse,collections,csv,json,statistics,os

HERE=Path(__file__).resolve().parent


def load(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=3);args=p.parse_args()
    plan=load(HERE/'plan.json');expected={0,-1,*range(1,len(plan['points'])+1)}
    selected={}
    for path in args.artifacts.glob('bpe-curve-*/completion.json'):
        completion=load(path);n=completion['number'];attempt=int(completion.get('run_attempt') or 1)
        if n not in selected or attempt>selected[n][0]:selected[n]=(attempt,path.parent)
    errors=[];records=[];builds={};points={};environments={}
    for n in sorted(expected):
        if n not in selected:errors.append(f'Missing completed version {n}');continue
        _,folder=selected[n];completion=load(folder/'completion.json')
        if completion['rounds']!=args.rounds:errors.append(f'Round count mismatch at {n}')
        rows=[json.loads(x) for x in (folder/'results.jsonl').read_text().splitlines()]
        build=load(folder/'builds'/f'{n:02d}'/'manifest.json');builds[n]=build
        environments[str(n)]=load(folder/'environment.json')
        if len(rows)!=2*args.rounds:errors.append(f'Incomplete samples at {n}')
        source=plan['base'] if n==0 else plan['original'] if n==-1 else plan['points'][n-1]
        if completion['head']!=source['head'] or build['head']!=source['head']:
            errors.append(f'Source head mismatch at {n}')
        if n!=-1 and build['source_tree']!=source['tree']:
            errors.append(f'Source tree mismatch at {n}')
        attempts=['main','repeat']+[f'round-{i}' for i in range(3,args.rounds+1)]
        keys=collections.Counter((r['language'],r['attempt']) for r in rows)
        if keys!=collections.Counter((l,a) for l in ('en','zh') for a in attempts[:args.rounds]):
            errors.append(f'Duplicate or missing language/round at {n}')
        for row in rows:
            row={**row,'artifact_name':folder.name,'artifact_local_path':str(folder),
                 'artifact_relative_run_path':row['artifact_relative_run_path']}
            if not row['model_equal_baseline'] or row['process']['exit_code']!=0:errors.append(f'Quality failure at {n}')
            if len(row['effective_affinity'])!=4:errors.append(f'Affinity mismatch at {n}')
            if any(row[k]!=build['head'] for k in ('head','job_head','measured_head','binary_source_head')):
                errors.append(f'Measurement head mismatch at {n}')
            if row['binary_sha256']!=build['binary_sha256'] or row['source_tree']!=build['source_tree']:
                errors.append(f'Measurement binary/source mismatch at {n}')
            records.append(row)
    for field in ('rustc','profile','runner_sources','observer_sha256','canonical_dependency_lock_sha256'):
        values=[b[field] for b in builds.values()]
        if values and any(v!=values[0] for v in values):errors.append(f'Build environment mismatch: {field}')
    references={language:{r['model_sha256'] for r in records if r['number']==0 and r['language']==language}
                for language in ('en','zh')}
    for language,hashes in references.items():
        if len(hashes)!=1:errors.append(f'Non-deterministic or absent upstream reference for {language}')
    for row in records:
        if row['model_sha256'] not in references[row['language']]:errors.append(f'Model differs from upstream: {row["number"]}/{row["language"]}')
    for n in sorted(expected):
        for language in ('en','zh'):
            rows=[r for r in records if r['number']==n and r['language']==language]
            if len(rows)!=args.rounds:continue
            rows.sort(key=lambda r:r['process']['maxrss_kib']);representative=rows[len(rows)//2]
            metrics={key:statistics.median(r['metrics'][key] for r in rows)
                     for key in ('pipeline_seconds','pipeline_cpu_seconds','feed_seconds','train_seconds','train_cpu_seconds')}
            points.setdefault(str(n),{})[language]={**metrics,'rss_mib':statistics.median(r['process']['maxrss_kib'] for r in rows)/1024,
                'rss_min_mib':min(r['process']['maxrss_kib'] for r in rows)/1024,
                'rss_max_mib':max(r['process']['maxrss_kib'] for r in rows)/1024,
                'max_swap_mib':max(r['max_process_swap_kib'] for r in rows)/1024,
                'peak_phase':representative['phase']['global_peak_phase'],
                'representative_artifact':representative['artifact_name'],
                'representative_run':representative['artifact_relative_run_path'],
                'time_samples':[r['metrics']['pipeline_seconds'] for r in rows]}
    summary={'complete':not errors,'errors':errors,'rounds':args.rounds,'size_mib':128,'workers':4,
        'versions':len(selected),'samples':len(records),'expected_samples':len(expected)*args.rounds*2,
        'statistic':'median of per-process feed+train wall/CPU and median of per-process full maxRSS',
        'precision':'Shared GitHub-hosted runners; three rounds per version on one VM. Compare broad trends; machines can differ between versions.',
        'plan':plan,'points':points,'environments':environments,
        'selected_artifacts':{str(n):{'attempt':a,'name':f.name} for n,(a,f) in selected.items()}}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    (args.out/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    table=['# BPE PR trends:128MiB,4 workers',f'{args.rounds} rounds per language; medians. Exact model checks: {"passed" if not errors else "incomplete/failed"}.',
        '', '| Version | Language | feed+train s | train s | CPU s | full maxRSS median MiB | RSS range MiB | swap max MiB | peak phase |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    csvrows=[]
    for n in [*range(len(plan['points'])+1),-1]:
        for language in ('en','zh'):
            r=points.get(str(n),{}).get(language)
            if not r:continue
            label='frozen-original' if n==-1 else str(n)
            table.append(f'| {label} | {language} | {r["pipeline_seconds"]:.3f} | {r["train_seconds"]:.3f} | {r["pipeline_cpu_seconds"]:.3f} | {r["rss_mib"]:.1f} | {r["rss_min_mib"]:.1f}–{r["rss_max_mib"]:.1f} | {r["max_swap_mib"]:.1f} | {r["peak_phase"]} |')
            csvrows.append({'number':n,'language':language,**{k:v for k,v in r.items() if not isinstance(v,list)}})
    if errors:table+=['','## Incomplete checks',*['- '+s for s in errors]]
    text='\n'.join(table)+'\n';(args.out/'summary.md').write_text(text)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'],'a') as stream:stream.write(text)
    if csvrows:
        with (args.out/'summary.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=csvrows[0].keys());writer.writeheader();writer.writerows(csvrows)
    print(json.dumps({'complete':summary['complete'],'versions':len(selected),'samples':len(records),'errors':errors}))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
