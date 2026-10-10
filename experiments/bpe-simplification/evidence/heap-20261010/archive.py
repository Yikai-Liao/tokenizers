"""Freeze compact reproducible evidence after both benchmark groups finish."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

work = Path('/root/code/tokenizers-workspaces/bpe-heap-experiment')
dest = work / 'experiments/bpe-simplification/evidence/heap-20261010'
root = Path('/tmp/bpe-heap-measurements')
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for mode in ['clean','diagnostic']:
    for name in ['manifest','runs','summary']:
        shutil.copy2(root/f'{mode}-{name}.json', dest/f'{mode}-{name}.json')
    # Repeated traces live in runs.json; preserve raw stdout/stderr and jobs separately.
    for case in root.glob(f'{mode}-*-b*-*'):
        if not case.is_dir(): continue
        out=dest/'raw'/case.name
        out.mkdir(parents=True,exist_ok=True)
        for name in ['job.json','stdout.log','stderr.log']:
            shutil.copy2(case/name,out/name)
scale=Path('/tmp/bpe-heap-scale6-measurements')
assert len(json.loads((scale/'clean-runs.json').read_text()))==16
assert all(r['valid'] and r['model_equal'] and r['max_swap_kib']==0 and not r['concurrent_builds_or_benchmarks'] for r in json.loads((scale/'clean-runs.json').read_text()))
for name in ['manifest','runs','summary']:
    shutil.copy2(scale/f'clean-{name}.json',dest/f'scale6-{name}.json')
for case in scale.glob('clean-*-b*-*'):
    out=dest/'raw'/f'scale6-{case.name}'
    out.mkdir(parents=True,exist_ok=True)
    for name in ['job.json','stdout.log','stderr.log']:
        shutil.copy2(case/name,out/name)
timing=Path('/tmp/bpe-heap-timing-measurements')
assert len(json.loads((timing/'runs.json').read_text()))==4
assert all(r['valid'] and r['model_equal'] and r['max_swap_kib']==0 and not r['concurrent_builds_or_benchmarks'] for r in json.loads((timing/'runs.json').read_text()))
for name in ['manifest','runs','summary']:
    shutil.copy2(timing/f'{name}.json',dest/f'heap-timing-{name}.json')
for case in timing.glob('*-w*'):
    out=dest/'raw'/f'timing-{case.name}'
    out.mkdir(parents=True,exist_ok=True)
    for name in ['job.json','stdout.log','stderr.log']:
        shutil.copy2(case/name,out/name)
for src,name in [('/tmp/bpe-heap-timing-instrument.py','timing-instrument.py'),('/tmp/bpe-heap-timing-measure.py','timing-measure.py'),('/tmp/bpe-heap-timing-build.log','build-timing.log'),('/tmp/bpe-heap-lines.json','lines.json')]:
    shutil.copy2(src,dest/name)
patch=subprocess.check_output(['git','diff','--','tokenizers/tk-train/src/trainers/bpe/engine/index.rs','tokenizers/tk-train/src/trainers/bpe/engine/mod.rs'],cwd='/root/code/tokenizers-workspaces/bpe-heap-timing')
(dest/'timing-candidate.patch').write_bytes(patch)
for src,name in [('/tmp/bpe-heap-instrument.py','instrument.py'),('/tmp/bpe-heap-measure.py','measure.py'),('/tmp/bpe-heap-diag-baseline-sampled-build.log','build-diagnostic-baseline-sampled.log'),('/tmp/bpe-heap-diag-candidate-sampled-build.log','build-diagnostic-candidate-sampled.log')]:
    shutil.copy2(src,dest/name)
for arm in ['baseline','candidate']:
    diag=Path(f'/root/code/tokenizers-workspaces/bpe-heap-diagnostic-{arm}')
    patch=subprocess.check_output(['git','diff','--','tokenizers/tk-train/src/trainers/bpe/engine/index.rs','tokenizers/tk-train/src/trainers/bpe/engine/mod.rs','tokenizers/tk-train/src/trainers/bpe/engine/positions.rs'],cwd=diag)
    (dest/f'diagnostic-{arm}.patch').write_bytes(patch)
for old,name in [('/tmp/bpe-heap-fullscan-measurements','earlier-fullscan'),('/tmp/bpe-heap-sampled-priority-attempt','interrupted-before-clean-priority')]:
    old=Path(old)
    out=dest/name
    out.mkdir(exist_ok=True)
    for file in ['diagnostic-manifest.json','diagnostic-runs.json']:
        if (old/file).exists(): shutil.copy2(old/file,out/file)
    for case in old.glob('diagnostic-*-b*-*'):
        if not case.is_dir(): continue
        case_out=out/case.name
        case_out.mkdir(exist_ok=True)
        for file in ['job.json','stdout.log','stderr.log']:
            shutil.copy2(case/file,case_out/file)

clean = json.loads((root/'clean-runs.json').read_text())
diagnostic = json.loads((root/'diagnostic-runs.json').read_text())
assert len(clean)==34 and len(diagnostic)==10
assert all(r['valid'] and r['model_equal'] and r['max_swap_kib']==0 and not r['concurrent_builds_or_benchmarks'] for r in clean+diagnostic)
engine=Path('tokenizers/tk-train/src/trainers/bpe/engine')
global_manifest=json.loads((work/'experiments/bpe-simplification/evidence/global-20261010/performance-manifest.json').read_text())
manifest=dict(
    baseline_commit='177302e2873649deb8c11d8be371f2710e44ed8f',
    candidate_patch_sha256=digest(dest/'candidate.patch'),
    clean_sources_sha256={str(p.relative_to(work)):digest(p) for p in sorted((work/engine).rglob('*.rs'))},
    diagnostic_sources_sha256={arm:{str(p.relative_to(Path(f'/root/code/tokenizers-workspaces/bpe-heap-diagnostic-{arm}'))):digest(p) for p in sorted((Path(f'/root/code/tokenizers-workspaces/bpe-heap-diagnostic-{arm}')/engine).glob('*.rs'))} for arm in ['baseline','candidate']},
    runner_lock_sha256=digest(work/'experiments/bpe-simplification/runner/Cargo.lock'),
    crate_lock_sha256=digest(work/'tokenizers/tk-train/Cargo.lock'),
    compiler=global_manifest['rustc'],
    profile='release opt-level3 fat LTO codegen-units1; runner tk-train dependency no-default-features',
    complete_model_equality=dict(clean4_runs=34,clean6_runs=16,diagnostic_runs=10,timing_runs=4,all_equal=True,all_child_swap_zero=True),
    checks=[dict(command='/root/.cargo/bin/cargo test --locked --manifest-path tokenizers/tk-train/Cargo.toml --target-dir /tmp/bpe-global-tests',log='native-default.log',result='17 native + 1 doctest passed'),dict(command='/root/.cargo/bin/cargo test --locked --no-default-features --manifest-path tokenizers/tk-train/Cargo.toml --target-dir /tmp/bpe-global-tests',log='native-nodefault.log',result='17 native + 1 doctest passed'),dict(command='/root/.cargo/bin/cargo clippy --locked --all-targets --manifest-path tokenizers/tk-train/Cargo.toml --target-dir /tmp/bpe-global-tests -- -D warnings',log='clippy.log',result='passed'),dict(command='/root/.cargo/bin/cargo fmt --manifest-path tokenizers/tk-train/Cargo.toml --check',result='passed'),dict(command='python3 experiments/bpe-simplification/count_lines.py',log='lines.json',result='2120 production / 799 tests'),dict(command='git diff --check',result='passed')],
    limitations=['Original suite passed; expanded reference differences and unresolved equal-priority reuse cohort ordering mean experiment only.','Diagnostic Chinese first commit then every32 commits plus finish; synthetic every commit. Stale maxima are checkpoint observations.','Earlier fullscan diagnostic candidate and first sampled baseline interrupted by author to reduce diagnostic traversal overhead and prioritize clean Chinese HWM/performance. These incomplete records excluded from final summaries.','Default benchmark jobs have no affixes; model equality on these inputs does not prove alias/reuse equivalence.','Shared VM, three measured pairs; descriptive changes only.','Only CPU0-5 available; four and six worker tests do not establish scaling on 16/32 CPU hosts.','Phase profiles have one sample per case/worker. Queue-path elapsed includes count-map lookups/removal/free/index cleanup; it is an upper-bound attribution, not pure heap CPU. Parallel owner Candidate creation/counting excluded.'],
    unrun='Miri not rerun: production positions.rs/unsafe unchanged; native codec checks retained.')
manifest['artifacts_sha256']={str(p.relative_to(dest)):digest(p) for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='validation.json'}
(dest/'validation.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
