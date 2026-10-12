from pathlib import Path
import json, statistics, re, hashlib, datetime

r = Path(__file__).resolve().parent
opt = Path('/tmp/bpe-positions-codec-optimized-results-20261012')
lib = Path('/tmp/bpe-positions-codec-libraries-results-20261012')
old = Path('/root/code/tokenizers-workspaces/bpe-positions-simd-results-20261012')
selected = json.loads((r / 'decision.json').read_text())['selected_arm']
normal = json.loads((r / 'normal.json').read_text())
validation = json.loads((r / 'validation.json').read_text())
assert len(normal) == 18 and len(validation) == 20
assert 'Remaining case validation complete' in (r / 'validation-controller.log').read_text()
rows = [x for x in normal + validation if x['arm'] in ['hand', selected]]
groups = {}
for case in sorted({x['case'] for x in rows}):
    groups[case] = {}
    for arm in ['hand', selected]:
        items = [x for x in rows if x['case'] == case and x['arm'] == arm]
        assert len(items) == 2
        assert all(x['affinity'] == [0, 1] and x['child_max_swap_kib'] == 0 for x in items)
        assert len({x['model_sha256'] for x in items}) == 1
        metrics = {}
        for key in ['train_seconds', 'train_cpu_seconds', 'process_hwm_kib_before_validation', 'child_sampled_rss_kib']:
            values = [x[key] for x in items]
            metrics[key] = dict(mean=statistics.mean(values), raw=values, min=min(values), max=max(values))
        phases = {}
        for label in ['BPE_INITIAL_COUNT_SECONDS', 'BPE_PREPARE_SECONDS', 'BPE_APPLY_SECONDS', 'BPE_COMMIT_SECONDS', 'BPE_DROP_SECONDS']:
            values = []
            for x in items:
                text = (r / 'normal' / f"{case}-r{x['repeat']}-{arm}" / 'stderr.log').read_text()
                values.append(float(re.search(r'^' + label + r' (\S+)', text, re.M)[1]))
            phases[label] = dict(mean=statistics.mean(values), raw=values)
        groups[case][arm] = dict(metrics=metrics, phases=phases, models_identical=True, model_sha256=items[0]['model_sha256'])
    assert groups[case]['hand']['model_sha256'] == groups[case][selected]['model_sha256']
    baseline = groups[case]['hand']['metrics']['train_seconds']['mean']
    current = groups[case][selected]['metrics']['train_seconds']['mean']
    groups[case]['delta_seconds'] = current - baseline
    groups[case]['delta_percent'] = 100 * (current / baseline - 1)

perf = {}
for case in ['en-whitespace-w6', 'en-bytelevel_regex-w6', 'zh-bytelevel_regex-w1']:
    perf[case] = {}
    for arm in ['hand', selected]:
        values = {}
        for stat in sorted((opt / 'perf-final').glob(f'{case}-*-{arm}/stat.csv')):
            for line in stat.read_text().splitlines():
                fields = line.split(';')
                if len(fields) >= 5 and fields[2] in ['instructions:u', 'branches:u', 'branch-misses:u', 'cache-misses:u', 'task-clock']:
                    assert float(fields[4]) == 100
                    values.setdefault(fields[2], []).append(float(fields[0]))
        perf[case][arm] = {k: dict(mean=statistics.mean(v), raw=v) for k, v in values.items()}
    perf[case]['instruction_delta_percent'] = 100 * (perf[case][selected]['instructions:u']['mean'] / perf[case]['hand']['instructions:u']['mean'] - 1)

costs = json.loads((r / 'production-costs.json').read_text())
decision = json.loads((r / 'decision.json').read_text())
decision['pending'] = None
decision['validation_complete'] = True
decision['chinese_cases_within_five_percent'] = all(v['delta_percent'] <= 5 for k, v in groups.items() if k.startswith('zh'))
(r / 'decision.json').write_text(json.dumps(decision, indent=2) + '\n')
findings = dict(
    selected=decision,
    production_integration=json.loads((r / 'production-integration.json').read_text()),
    final_eight_case_comparison=groups,
    independent_earlier_four_arm_comparison=json.loads((opt / 'summary.json').read_text()),
    single_byte_comparison=json.loads((r / 'summary.json').read_text()),
    original_library_comparison=json.loads((lib / 'summary.json').read_text()),
    perf_final=perf,
    workload_counts_from_separate_diagnostic=json.loads((old / 'distribution-summary.json').read_text()),
    production_costs=costs,
    native_tests='32 BPE tests per default/no-default configuration; clippy all-targets; known baseline fixture skipped',
    proofs='32772 append-state and 1008 chunk comparisons per candidate; 52 allocation/reallocation/free cases',
    normal_runs=dict(original_libraries=len(json.loads((lib / 'normal.json').read_text())), optimized=len(json.loads((opt / 'normal.json').read_text())), single_byte=len(normal), remaining_validation=len(validation)),
    limits=['All measurements limited to CPU 0 and 1, including requested 6 workers.', 'Perf counts include input loading, training, output validation/serialization, and recorder overhead; they are not pure training counts.', 'Normal performance uses frozen SmallVec 1.16.2; native checks use 1.16.3. Both upstream packages are unmodified.', 'Two rounds are sufficient to reveal large regressions, not to establish a sub-percent gain on this VPS.', 'The original library comparison was stopped after 25 completed runs; its unfinished Whitespace comparison is not a completed result.'],
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
)
(r / 'final-findings.json').write_text(json.dumps(findings, indent=2, ensure_ascii=False) + '\n')
for case, value in groups.items():
    print(case, round(value['hand']['metrics']['train_seconds']['mean'], 4), round(value[selected]['metrics']['train_seconds']['mean'], 4), round(value['delta_percent'], 2))
print('Chinese <= 5%:', decision['chinese_cases_within_five_percent'])
