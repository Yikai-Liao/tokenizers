from pathlib import Path
import json, statistics, re, shutil, hashlib, datetime

r = Path(__file__).resolve().parent
w = Path('/root/code/tokenizers-workspaces/bpe-positions-encapsulation-20261012')
report = w / 'experiments/bpe-simplification/evidence/positions-codec-libraries-20261012'
rows = json.loads((r / 'normal.json').read_text())
assert len(rows) == 16
assert 'Unrestricted six-worker comparisons complete' in (r / 'normal-controller.log').read_text()
assert all(x['affinity'] == list(range(6)) and x['child_max_swap_kib'] == 0 for x in rows)
groups = {}
for case in sorted({x['case'] for x in rows}):
    group = {}
    for arm in ['hand', 'fmt-zero-push']:
        items = [x for x in rows if x['case'] == case and x['arm'] == arm]
        assert len(items) == 2 and len({x['model_sha256'] for x in items}) == 1
        metrics = {}
        for label in ['train_seconds', 'train_cpu_seconds', 'process_hwm_kib_before_validation', 'child_sampled_rss_kib']:
            values = [x[label] for x in items]
            metrics[label] = dict(mean=statistics.mean(values), raw=values, min=min(values), max=max(values))
        phases = {}
        for label in ['BPE_INITIAL_COUNT_SECONDS', 'BPE_PREPARE_SECONDS', 'BPE_APPLY_SECONDS', 'BPE_COMMIT_SECONDS', 'BPE_DROP_SECONDS']:
            values = [float(re.search(r'^' + label + r' (\S+)', (r / 'normal' / f"{case}-r{x['repeat']}-{arm}" / 'stderr.log').read_text(), re.M)[1]) for x in items]
            phases[label] = dict(mean=statistics.mean(values), raw=values)
        group[arm] = dict(metrics=metrics, phases=phases, models_identical=True, model_sha256=items[0]['model_sha256'])
    assert group['hand']['model_sha256'] == group['fmt-zero-push']['model_sha256']
    hand = group['hand']['metrics']['train_seconds']['mean']
    library = group['fmt-zero-push']['metrics']['train_seconds']['mean']
    group.update(delta_seconds=library - hand, delta_percent=100 * (library / hand - 1), measurement_cpu_affinity=list(range(6)))
    groups[case] = group

(r / 'summary.json').write_text(json.dumps(groups, indent=2) + '\n')
findings_path = report / 'single-byte-and-final/final-findings.json'
findings = json.loads(findings_path.read_text())
findings['superseded_two_cpu_six_worker_comparison'] = {k: v for k, v in findings['final_eight_case_comparison'].items() if k.endswith('-w6')}
findings['six_worker_correction'] = dict(reason='Six-worker measurements were inappropriately restricted to two CPUs. Withdraw their use as normal six-worker performance.', replacement='six-cpu-correction/summary.json', source_binaries_unchanged=True, all_six_cpus_allowed=True, completed_normal_runs=16, utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
findings['final_eight_case_comparison'].update(groups)
for case, value in findings['final_eight_case_comparison'].items():
    if case.endswith('-w1'):
        value['measurement_cpu_affinity'] = [0, 1]
findings['normal_runs']['six_cpu_correction'] = 16
findings['perf_final_scope'] = 'Two-CPU diagnostic only; no inference about unrestricted six-worker wall time.'
findings['limits'][0] = 'Single-worker results retain the original CPU 0,1 affinity. Corrected six-worker results allow all CPUs 0-5. Earlier six-worker results restricted to two CPUs are superseded and do not represent normal six-worker performance.'
findings['selected']['chinese_cases_within_five_percent'] = all(v['delta_percent'] <= 5 for k, v in findings['final_eight_case_comparison'].items() if k.startswith('zh'))
findings_path.write_text(json.dumps(findings, indent=2, ensure_ascii=False) + '\n')

text_path = report / 'FINDINGS.txt'
lines = text_path.read_text().splitlines()
lines[0] += ' — 六 CPU 更正'
lines.insert(2, '更正：此前擅自把 6 worker 限制在两个 CPU 上，不能据此评价正常六核性能。所有旧六 worker 的两 CPU 耗时结论撤回；原始数据保留作受限配置记录。下表的四项六 worker 结果已用相同 ELF、允许全部 CPU 0–5 的两轮对照替换，完整证据见 six-cpu-correction。')
for i, line in enumerate(lines):
    if line.startswith('最终完整训练对照：'):
        lines[i] = '更正后的完整训练对照：同一冻结语料和基准模型，每组两轮，顺序交错并反转，使用系统分配器。单 worker 沿用 CPU 0、1 的允许范围；6 worker 使用全部 CPU 0–5，未设置 taskset 限制。手写版与库版均复用原冻结二进制，不重编译。以下是同组均值。'
    for case, value in groups.items():
        if line.startswith(case + ' '):
            hand = value['hand']['metrics']['train_seconds']['mean']
            library = value['fmt-zero-push']['metrics']['train_seconds']['mean']
            lines[i] = f"{case:28s} {hand:12.4f} {library:11.4f} {value['delta_seconds']:9.4f} {value['delta_percent']:9.2f}"
    if line.startswith('最初的 5% 目标'):
        white = groups['zh-whitespace-w6']['delta_percent']
        byte = groups['zh-bytelevel_regex-w6']['delta_percent']
        lines[i] = f'最初的 5% 目标没有在所有配置上达到。中文 Whitespace 单 worker 约 +5.62%，选择保留简单库调用并接受这项代价。纠正资源配置后，真实六 CPU / 六 worker 的中文 Whitespace 为 {white:+.2f}%，ByteLevel 为 {byte:+.2f}%。此前约 +8.5% 只属于六 worker 被限制在两 CPU 的配置，已撤回其作为正常六 worker 性能结论的使用，不用于正式判断。各轮原始耗时与有效 CPU 亲和范围见 JSON 和 job/结果记录。'
    if line.startswith('perf 使用正式计时'):
        lines[i] = '以下 perf 数据属于旧的两 CPU 诊断配置，仍能检查调用与汇编差异，不能用来解释真实六 CPU 的六 worker 耗时。' + line
    if line.startswith('验证：'):
        lines[i] = line.replace('87 次正常训练', '103 次正常训练（其中新增 16 次正确六 CPU 对照）')
    if line.startswith('文件：'):
        lines[i] += ' six-cpu-correction 保存此次正确六 CPU 的完整对照，并替代旧六 worker 性能结论。'
text_path.write_text('\n'.join(lines) + '\n')

destination = report / 'six-cpu-correction'
assert not destination.exists()
shutil.copytree(r, destination)
manifest = []
for p in sorted(report.rglob('*')):
    if p.is_file() and p != report / 'MANIFEST.json':
        manifest.append(dict(path=str(p.relative_to(report)), size=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(report / 'MANIFEST.json').write_text(json.dumps(dict(files=manifest, excludes=['probe-target', 'target', '__pycache__', 'hold-validation']), indent=2) + '\n')
for case, group in groups.items():
    print(case, group['hand']['metrics']['train_seconds']['raw'], group['fmt-zero-push']['metrics']['train_seconds']['raw'], round(group['delta_percent'], 2))
