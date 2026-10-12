from pathlib import Path
import json, shutil, hashlib, os

r = Path(__file__).resolve().parent
w = Path('/root/code/tokenizers-workspaces/bpe-positions-encapsulation-20261012')
report = w / 'experiments/bpe-simplification/evidence/positions-codec-libraries-20261012'
assert not report.exists()
findings = json.loads((r / 'final-findings.json').read_text())
sources = {
    'original-libraries': Path('/tmp/bpe-positions-codec-libraries-results-20261012'),
    'optimized': Path('/tmp/bpe-positions-codec-optimized-results-20261012'),
    'single-byte-and-final': r,
}
skip = {'probe-target', 'target', '__pycache__', 'hold-validation'}
total = sum(p.stat().st_size for source in sources.values() for p in source.rglob('*') if p.is_file() and not any(part in skip for part in p.relative_to(source).parts))
free = os.statvfs(w).f_bfree * os.statvfs(w).f_frsize
assert free > total * 2 + 1_000_000_000, (free, total)
for name, source in sources.items():
    destination = report / name
    for p in source.rglob('*'):
        relative = p.relative_to(source)
        if any(part in skip for part in relative.parts):
            continue
        if p.is_file():
            assert p.stat().st_size < 100 * 1024 * 1024, str(p)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)

groups = findings['final_eight_case_comparison']
lines = [
    'Positions 的 LEB128 库替换与性能取舍（2026-10-12）',
    '',
    '正式采用 leb128fmt 0.1.0 的切片编解码 API。编码写入 Leaf 已预留的 SmallVec 缓冲，解码同样调用库；绝对 u64、每 128 个位置的重启点和目录仍是本模块的内部格式。保留原有单元素内联表示。没有新增 unsafe，没有修改上游库，没有采用双元素内联、TLS 草稿缓冲或单字节分支。',
    '',
    '生产代码：Positions + codec + merge 相对 3436bba7 净减少 20 行非空、非注释代码，其中此前封装减少 17 行，本次库替换再减少 3 行。统计使用 Rust AST 排除 cfg(test)，未靠压缩格式减少行数。源码、统计脚本和各候选提交可在 single-byte-and-final/production-source 与 production-costs.json 核查。',
    '',
    '最终完整训练对照：同一冻结语料和基准模型，每组两轮，顺序交错并反转，使用系统分配器。全部进程限制 CPU 0、1，6 worker 配置不是无限制六核测试。以下是最后一组对照的均值，百分比相对同组手写版本。',
    '',
    'case                         handwritten(s)  library(s)  delta(s)  delta(%)',
]
for case, value in groups.items():
    hand = value['hand']['metrics']['train_seconds']['mean']
    current = value['fmt-zero-push']['metrics']['train_seconds']['mean']
    lines.append(f"{case:28s} {hand:12.4f} {current:11.4f} {value['delta_seconds']:9.4f} {value['delta_percent']:9.2f}")
lines += [
    '',
    '最初的 5% 目标没有在所有配置上达到。中文 Whitespace 单 worker 约 +5.62%，选择保留简单库调用并接受这项代价。6 worker 限制在两 CPU 时，中文 Whitespace 两轮分别约 +13.1% 和 +3.7%，平均约 +8.5%；该结果必须保留，不能解释为真实六核性能。英文差异按绝对秒数评估：任务较短，少量差异和调度波动会形成较大的百分比。原始两轮耗时、CPU 时间、进程峰值 RSS 与阶段时间见 final-findings.json。',
    '',
    '独立的前一组四候选测试中，中文 ByteLevel 单 worker：手写 65.5335 秒，直接库调用 66.0938 秒（+0.86%），TLS 69.1579 秒（+5.53%），gimli 内联候选 82.5850 秒（+26.02%）。后一组同一二进制的直接库调用为 +3.32%，说明该 VPS 上连较长任务也存在环境波动；两个测量组分别保留，不挑选最好的一组作为统一结论。',
    '',
    '差异定位：原始 gimli-rs leb128 经 SmallVec 自带 Write 逐字节写入，实际汇编每个字节进入切片插入路径并调用 memmove；原始 fmt 切片方案每个 gap 调用未内联的 resize 并补零，数组方案调用复制路径。直接按宽度 push(0)、随后调用库编码，并让调用者内联，去掉了原始切片方案的 resize 调用。这些差异来自调用方式与编译结果，52 组实际系统分配/重分配/释放对照均相同，不是新增堆分配造成的。',
    '',
    'perf 使用正式计时的同一 ELF，没有为了采样重编译。最终直接库调用的指令数：英文 Whitespace +2.83%，英文 ByteLevel +3.15%，中文 ByteLevel +5.09%。计数覆盖加载、训练、验证/序列化和 recorder，不能当作纯训练阶段指令数，也不能把它与耗时百分比直接相减来计算 VPS 影响。英文两轮计时会反转，而指令数很稳定；并行调度、指令类型与内存行为也可能影响耗时。未使用虚拟 PMU 的 cycles/IPC 作因果解释。',
    '',
    '语料差异的诊断证据：英文 Whitespace 累计约 1,013 万个位置、4.49 万次合并；中文 ByteLevel 累计约 4.69 亿个位置、4.97 万次合并。合并次数接近但每次处理的位置规模相差很大。统计覆盖完整叶片生命周期，包括丢弃碎片，来自独立诊断构建，不是唯一语料 token 数。语言、预分词方法与 worker 数同时变化，不能把这些结果归结成语言本身的固有开销。',
    '',
    '单字节分支实验：只增加 4 行，gap < 128 时调用库 encode_u64 后取第一个字节。LLVM 已消除库调用和临时十字节数组，但保留了独立 SmallVec::push 调用。512 位置的编码加释放微基准约快 11%，完整中文 ByteLevel 两轮均值却为 66.6960 秒，直接库调用为 66.5911 秒，仅约 +0.16%；英文完整训练也未显示一致改善。保留 a81b2416 候选和原始证据，未增加正式代码的分支。perf-small.py 仅准备，性能接受后未执行额外采样。',
    '',
    '验证：最终候选 default 与 no-default BPE lib 测试各 32 个通过，clippy all-targets 通过；跳过已知基线 JSON fixture，允许基线已有 redundant_field_names lint。32772 个逐次 append 状态的编码字节、容量与 spill 状态一致；1008 组分片边界与内容一致；52 组系统分配/重分配/释放一致。87 次正常训练以及已完成的诊断训练均逐字节核对完整模型；正常训练 Swap 为 0。原始库比较中另有一个中断的 Whitespace 运行，明确标记，没有计作完成结果。',
    '',
    '正常比较冻结 SmallVec 1.16.2，并统一启用 write 与两个库依赖以保持比较条件一致；正式 manifest 仅增加 leb128fmt，保留 SmallVec union。native 检查使用 SmallVec 1.16.3。SmallVec 两版本、leb128、leb128fmt 的全部 registry 文件均与 Cargo lock 校验过的原始 crate 包相同；证明见 upstream-integrity.json。',
    '',
    '文件：original-libraries 保存原始三个库接法与 perf；optimized 保存六种调用优化与 TLS；single-byte-and-final 保存单字节候选、最后八场景验证、正式源代码与总结果。MANIFEST.json 列出所有归档文件的尺寸和 SHA-256。构建 target 和临时等待标志不归档。输入来自此前冻结的预处理语料，绝对路径见 job.json；这里保存输入引用、源码、脚本、ELF 和模型结果，未重复上传整份原始语料。',
    '',
    '接口来源：https://docs.rs/leb128fmt/0.1.0/leb128fmt/ ，https://docs.rs/leb128/0.2.7/leb128/ ，https://docs.rs/smallvec/1.16.2/smallvec/ 。',
]
(report / 'FINDINGS.txt').write_text('\n'.join(lines) + '\n')
manifest = []
for p in sorted(report.rglob('*')):
    if p.is_file():
        manifest.append(dict(path=str(p.relative_to(report)), size=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(report / 'MANIFEST.json').write_text(json.dumps(dict(files=manifest, excludes=sorted(skip)), indent=2) + '\n')
print('Archived', len(manifest), 'files, bytes', sum(x['size'] for x in manifest))
print(report)
