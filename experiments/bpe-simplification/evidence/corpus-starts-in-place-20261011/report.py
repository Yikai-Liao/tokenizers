"""Archive raw benchmark records and summarize measured layout checkpoints."""
import hashlib,json,shutil,statistics
from pathlib import Path
here=Path(__file__).resolve().parent
training=Path('/tmp/bpe-corpus-b1-training-results-20261011')
archive=here/'training';archive.mkdir(exist_ok=True)
for name in ['training-manifest.json','training-runs.json','training-summary.json']:
    shutil.copy(training/name,archive/name)
shutil.copy('/tmp/bpe-corpus-b1-training-20261011.log',archive/'stdout.log')
runs=json.loads((archive/'training-runs.json').read_text())
for run in runs:
    folder=Path(run['command'][-1]).parent
    dest=archive/'samples'/folder.name;dest.mkdir(parents=True,exist_ok=True)
    for name in ['job.json','stdout.log','stderr.log']:
        shutil.copy(folder/name,dest/name)
probes=json.loads((here/'layout-probe-runs.json').read_text())
for run in probes:
    folder=Path(run['command'][-1]).parent
    with (folder/'model.json').open('rb') as f:run['model_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
    dest=here/'layout-samples'/folder.name;dest.mkdir(parents=True,exist_ok=True)
    for name in ['job.json','stdout.log','stderr.log']:
        shutil.copy(folder/name,dest/name)
(here/'layout-probe-runs.json').write_text(json.dumps(probes,indent=2)+'\n')
summary=[]
for case in dict.fromkeys(r['case'] for r in probes):
    selected=[r for r in probes if r['case']==case and not r['warmup']]
    before_after=lambda r:[r['checkpoints'][i] for i in [0,1]]
    row=dict(case=case,formal_pairs=2,words=selected[0]['checkpoints'][0]['words'])
    for arm in ['baseline','candidate']:
        arm_rows=[r for r in selected if r['arm']==arm]
        row[arm]={key:statistics.median(r['checkpoints'][1][key] for r in arm_rows) for key in ['directory_bytes','rss_kib','hwm_kib']}
        row[arm]['rss_growth_during_prefix_kib']=statistics.median(r['checkpoints'][1]['rss_kib']-r['checkpoints'][0]['rss_kib'] for r in arm_rows)
    row['directory_bytes_saved']=row['baseline']['directory_bytes']-row['candidate']['directory_bytes']
    row['paired_median_prefixed_rss_delta_kib']=statistics.median(next(r for r in selected if r['block']==block and r['arm']=='candidate')['checkpoints'][1]['rss_kib']-next(r for r in selected if r['block']==block and r['arm']=='baseline')['checkpoints'][1]['rss_kib'] for block in [1,2])
    assert row['directory_bytes_saved']==row['words']*8
    summary.append(row)
(here/'layout-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
assert len(runs)==24 and len(probes)==24
assert all(r['model_equal'] and r['max_swap_kib']==0 for r in runs+probes)

# Archive the final integrated comparison separately from the isolated baseline.
final_root=Path('/tmp/bpe-corpus-b1-integrated-results-20261011')
final_archive=here/'integrated/training';final_archive.mkdir(parents=True,exist_ok=True)
for name in ['training-manifest.json','training-runs.json','training-summary.json']:
    shutil.copy(final_root/name,final_archive/name)
shutil.copy('/tmp/bpe-corpus-b1-integrated-training-20261011.log',final_archive/'stdout.log')
final_runs=json.loads((final_archive/'training-runs.json').read_text())
for run in final_runs:
    folder=Path(run['command'][-1]).parent
    dest=final_archive/'samples'/folder.name;dest.mkdir(parents=True,exist_ok=True)
    for name in ['job.json','stdout.log','stderr.log']:
        shutil.copy(folder/name,dest/name)
assert len(final_runs)==24
assert all(r['model_equal'] and r['max_swap_kib']==0 for r in final_runs)
cross=[]
for case in dict.fromkeys(r['case'] for r in final_runs):
    old=next(r for r in runs if r['case']==case and r['block']==0 and r['arm']=='baseline')
    new=next(r for r in final_runs if r['case']==case and r['block']==0 and r['arm']=='baseline')
    old_model=Path(old['command'][-1]).parent/'model.json'
    new_model=Path(new['command'][-1]).parent/'model.json'
    equal=json.loads(old_model.read_text())==json.loads(new_model.read_text())
    cross.append(dict(case=case,initial_baseline_model_sha256=old['model_sha256'],integrated_baseline_model_sha256=new['model_sha256'],full_model_equal=equal))
(here/'integrated/upstream-model-comparison.json').write_text(json.dumps(cross,indent=2)+'\n')
assert all(r['full_model_equal'] for r in cross),cross
print('24 integrated models match; four upstream comparisons match')
