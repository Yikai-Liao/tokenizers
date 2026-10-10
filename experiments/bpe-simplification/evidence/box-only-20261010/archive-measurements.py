"""Archive metrics, jobs and one compressed reference model per input case."""
import gzip,hashlib,json,shutil,sys
from pathlib import Path
root=Path(__file__).resolve().parent
label=sys.argv[1]
source=Path(sys.argv[2])
dest=root/label
dest.mkdir(exist_ok=True)
for name in ['core-manifest.json','core-runs.json','core-summary.json']:
    shutil.copy2(source/name,dest/name)
records=json.loads((source/'core-runs.json').read_text())
assert all(r['valid'] and r['model_equal'] and r['max_swap_kib']==0 and not r['concurrent_builds_or_benchmarks'] for r in records)
jobs={}
reference_dir=dest/'reference-models'
reference_dir.mkdir(exist_ok=True)
seen=set()
for r in records:
    run=source/f"core-{r['case']}-b{r['block']}-{r['arm']}"
    jobs[run.name]=dict(job=json.loads((run/'job.json').read_text()),stdout=json.loads((run/'stdout.log').read_text()),stderr=(run/'stderr.log').read_text())
    if (run/'perf.csv').exists(): jobs[run.name]['perf_csv'] = (run/'perf.csv').read_text()
    if r['case'] not in seen:
        model=(run/'model.json').read_bytes()
        assert hashlib.sha256(model).hexdigest()==r['model_sha256']
        packed=gzip.compress(model,mtime=0)
        (reference_dir/f"{r['case']}.json.gz").write_bytes(packed)
        assert hashlib.sha256(gzip.decompress(packed)).hexdigest()==r['model_sha256']
        seen.add(r['case'])
(dest/'jobs-and-output.json').write_text(json.dumps(jobs,indent=2)+'\n')
print(label,len(records),'validated runs',len(seen),'reference models')
