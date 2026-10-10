from pathlib import Path
import gzip,hashlib,json,runpy,shutil,statistics,subprocess
work=Path('/root/code/tokenizers-workspaces/bpe-heap-experiment')
n=runpy.run_path('/tmp/bpe-cleanup-archive.py');archive=n['archive'];copy=n['copy']
n['measurements']('/tmp/bpe-cleanup-measurements','final-training')
for src,name in [('bpe-cleanup-build.py','build.py'),('bpe-cleanup-builds.json','builds.json'),('bpe-cleanup-measure.py','measure.py'),('bpe-cleanup-candidate-build.log','candidate-build.log'),('bpe-cleanup-controller.log','controller.log')]:copy('/tmp/'+src,'final-training/'+name)
copy('/tmp/bpe-cleanup-archive.py','archive.py');copy('/tmp/bpe-cleanup-write-report.py','write-report.py')
summary=json.loads((archive/'final-training/core-summary.json').read_text());runs=json.loads((archive/'final-training/core-runs.json').read_text())
assert len(runs)==12 and sum(not r['warmup'] for r in runs)==8
assert all(r['model_equal'] and r['valid'] and r['returncode']==0 and r['max_swap_kib']==0 and not r['concurrent_builds_or_benchmarks'] for r in runs)
builds=json.loads((archive/'final-training/builds.json').read_text())
for path,expected in builds['builds']['candidate']['source_hashes'].items():assert hashlib.sha256((work/path).read_bytes()).hexdigest()==expected,path
for case in [s['case'] for s in summary]:
 a=json.loads(gzip.open(archive/'initial-heap'/f'{case}-reference-model.json.gz').read());b=json.loads(gzip.open(archive/'final-training'/f'{case}-reference-model.json.gz').read());assert a==b,case
validation=dict(final_source_commit=builds['builds']['candidate']['commit'],full_models_equal_across_initial_and_final_comparisons=True,initial_complete_runs=12,final_complete_runs=12,final_formal_runs=8,final_excluded_warmups=4,all_final_runs_zero_swap=True,no_concurrent_builds_or_task_benchmarks=True,source_hashes_match_final_commit=True,default_library_tests=20,no_default_library_tests=20,all_target_clippy_warnings_denied=True,changed_files_rustfmt=True,diff_whitespace_check=True)
(archive/'validation/final-validation.json').write_text(json.dumps(validation,indent=2))
text='''The final comparison uses the same pre-change `d4e3fa45` baseline binary and
final `a31898fa` binary. Thus both arms use pop/push heap certification; remaining
training-source differences are the five simplifications. Serialization is
outside this training timer. As in the initial comparison, each input has an
excluded AB warmup, followed by formal BA and AB blocks (12 runs total, 8 formal).
All 12 full models agree, including with the initial comparison's reference
models. No measured child swap or concurrent task builds/benchmarks were observed.

| Input | Baseline / final wall (s) | Baseline / final CPU (s) | Paired wall change | Paired CPU change | Paired HWM change |
| --- | ---: | ---: | ---: | ---: | ---: |
'''
for s in summary:
 a,b=s['baseline'],s['candidate'];d=s['paired_median_delta_percent']['candidate'];label='ByteLevel' if s['case']=='zh-256MiB' else 'Whitespace'
 text+=f"| {label} | {a['train_seconds']:.3f} / {b['train_seconds']:.3f} | {a['train_cpu_seconds']:.3f} / {b['train_cpu_seconds']:.3f} | {d['train_seconds']:+.2f}% | {d['train_cpu_seconds']:+.2f}% | {d['process_hwm_kib_before_validation']:+.2f}% |\n"
text+='''
Absolute values are medians of two formal samples per arm; deltas are medians
of within-block percentage changes. A ratio of the displayed absolute medians
can differ from the paired delta. Individual CPU changes reverse direction:
'''
for s in summary:
 f=[r for r in runs if r['case']==s['case'] and not r['warmup']];ds=[]
 for block in [1,2]:
  p={r['arm']:r for r in f if r['block']==block};ds.append((p['candidate']['metrics']['train_cpu_seconds']/p['baseline']['metrics']['train_cpu_seconds']-1)*100)
 text+=f"{s['case']}: {ds[0]:+.2f}% and {ds[1]:+.2f}%; "
text+='''these observations do not establish a stable speedup or a consistent
training slowdown. Source/build metadata, raw runs and model validation are in
[final-training](final-training).
'''
p=archive/'README.md';s=p.read_text().replace('FINAL_RESULTS_PENDING',text);p.write_text(s)
p=work/'experiments/bpe-simplification/STATUS.md';s=p.read_text();a=s.index('## Serde delegation');b=s.index('The fixed-size storage audit found',a)
intro='''## Serde delegation and state simplification, 2026-10-10

The final implementation retains WordCounts Map.serialize and Entries.collect_map
with the same flat-map format and no additional allocation or dependency.
A standalone preallocated-Vec serialization comparison observes Entries CPU
+4.65% and Map +0.98%; JSON bytes match and both implementations allocate zero
inside this measurement. The user explicitly chose to retain collect_map and
accept its local serialization cost.

PairIndex::best returns to one pop/push certification implementation. Initial
PeekMut timing is inconsistent across the two inputs. An injected pair with two
historical cohorts (snapshot counts 10/5, current count 5) shows that equal-priority
cohort selection can change the complete merge trace. This is a directed queue
state, not proof of a naturally failing corpus. The retained regression checks
all expected rules and the entire vocabulary/merge model. A fresh/reuse branch
was rejected in favor of the unified previous protocol.

The five state/interface simplifications remove infallible Result propagation,
return symbol-scan break reasons directly, store an unfed Trainer in Builder,
collect reuse word IDs through adjacent deduplication, and read Rule's pair from
its owning Candidate instead of a duplicate field. Sorting and overflow checks,
UTF-8 affix boundaries, public builder methods and serialized fields are retained.
Initial dense-count staging and frozen-position storage are unchanged.

Collection-only measurements with actual Positions iteration show a tradeoff:
with 64 occurrences per word, Vec capacity falls from 2 MiB to 32 KiB and CPU
falls about 6.6%; with no/low duplication, capacity is unchanged and CPU rises
about 7–9%. Both low-duplicate implementations grow through 16 reallocations.
This is not a whole-reuse training or process-RSS measurement.

The final d4e3fa45 / a31898fa training comparison uses two Chinese 256 MiB inputs,
four workers on CPUs 0–3, identical locked release builds, excluded warmup, then
BA and AB formal blocks. Public training timing excludes loading, serialization
and validation; these no-affix cases use fresh training. All 12 full models match
and measured swap is zero. Median paired changes are:

| Input | Wall | CPU | HWM |
| --- | ---: | ---: | ---: |
'''
for ss in summary:
 d=ss['paired_median_delta_percent']['candidate'];label='ByteLevel' if ss['case']=='zh-256MiB' else 'Whitespace';intro+=f"| {label} | {d['train_seconds']:+.2f}% | {d['train_cpu_seconds']:+.2f}% | {d['process_hwm_kib_before_validation']:+.2f}% |\n"
intro+='''
Individual formal changes reverse direction within each input; these small-sample
shared-VM observations do not establish a stable performance gain. Default and
no-default-feature library suites each pass all 20 tests, including generated
per-rule/full-model oracle comparisons at 1, 4 and 8 workers. All-target Clippy
passes with warnings denied, changed files pass rustfmt and whitespace checks.
Whole-crate rustfmt still has pre-existing import-order differences in five
untouched files. Complete protocols, sources, raw samples, model hashes and
validation are in [the comparison report](evidence/serde-heap-simplification-20261010/README.md).

'''
s=s[:a]+intro+s[b:];p.write_text(s)
copy('/tmp/bpe-cleanup-finalize.py','finalize.py')
manifest={str(p.relative_to(archive)):hashlib.sha256(p.read_bytes()).hexdigest() for p in archive.rglob('*') if p.is_file() and p.name!='archive-sha256.json'}
(archive/'archive-sha256.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(validation,indent=2))
