#!/usr/bin/env python3
"""Persist configurations, algorithms, observations and limits in Markdown."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('parallel',type=Path)
    parser.add_argument('pr',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    rows = [json.loads(s) for s in args.parallel.read_text().splitlines()]
    pr = json.loads(args.pr.read_text())
    if any('failure' in r for r in [*rows,pr]):
        raise SystemExit('cannot summarize failed observations as completed comparisons')
    reference = rows[0]
    signature = lambda r: (r['model_sha256'],r['actual_vocab'],r['actual_merges'],r['unique_words'])
    assert all(signature(r)==signature(reference) for r in [*rows,pr])
    env = json.loads(args.parallel.with_suffix('.environment.json').read_text())
    prenv = json.loads(args.pr.with_suffix('.environment.json').read_text())
    serial,parallel,atomic = rows
    s = reference['indexed_stats']
    names = ['非原子 1 worker','非原子 4 worker','原子 4 worker','PR 4 worker']
    lines = ['# HF 并行关键对照与 PR #2348', '',
             '## 测量范围', '',
             f'本轮只完成一个真实中文案例、四次独立进程训练，每种配置一次。日期为 {env["date_utc"].split("T")[0]}。全部算法确定后再做完整 benchmark；本表用于确认大工作集并行效果、检查原子访问及定位主要阶段，不是多次重复的稳定性结论。', '',
             f'- 当前基线提交：`{env["base_commit"]}`，另加工作区中的实验实现。',
             f'- 未合并 PR #2348 的固定版本：`{prenv["pr_head"]}`；使用该提交的干净 checkout，未把后续改动混入。',
             f'- CPU：{env["cpu_model"]}，操作系统可见 {env["visible_cpus"]} 个 CPU；`{env["rustc"]}`，release 构建，无并行构建/测试/下载干扰。',
             '- feed 固定串行。当前训练使用每训练建立一次的专用 Rayon pool；所有 merge 复用它。PR 的 runner 在 feed 后启用条件并行，global Rayon 为 4 worker，库的线程控制也显式设为 4。', '',
             '## 输入与共同参数', '',
             f'输入为固定 revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` 的 Wikimedia Wikipedia 中文文章段落。按 shard、文章及段落顺序读取，将连续空白归一为一个空格，保留 32–8192 字节的段落，再取完整行前缀。没有重复拼接或人工扩充。原始 1 GiB 文件来自前两份 shard；关键输入约 512 MiB。原始 shard URL、SHA256、选择规则在 `.build/gb-corpus/manifest.json`，其内容也已嵌入 environment JSON。', '',
             '| 参数 | 值 |', '|---|---|',
             f'| 实际输入字节 | {reference["input_bytes"]:,} |',
             f'| 输入 SHA256 | `{env["corpora"]["zh-512m"]["sha256"]}` |',
             '| 预处理 | `none`：保留整行及换行，feed 聚合相同片段的词频 |',
             '| 目标词表 / 最小频率 | 50,000 / 2 |',
             '| prefix / suffix / 特殊 token / alphabet limit / max length | 未设置 |',
             f'| 初始 alphabet | {env["preflight"]["alphabet"]:,} |',
             f'| feed 后不同片段 | {reference["unique_words"]:,} |',
             f'| 实际词表 / 合并规则 | {reference["actual_vocab"]:,} / {reference["actual_merges"]:,} |',
             f'| 当前 corpus 槽数 N / 初始未加权边数 E | {s["initial_slots"]:,} / {s["initial_edges"]:,} |',
             f'| 初始过滤后不同 pair / 物理地址块 | {s["initial_pairs"]:,} / {s["initial_blocks"]} |',
             f'| 完整词表与有序 merges 的共同 SHA256 | `{reference["model_sha256"]}` |', '',
             '四次完整模型摘要、词表大小、规则数和不同片段数均一致。初始 alphabet 小于目标，测量实际包含 29,243 次合并。', '',
             '## 使用的算法与配置', '',
             '| 配置 | Corpus / posting | 候选与更新算法 | 线程与批次 |', '|---|---|---|---|',
             '| 当前非原子 1 worker | u16 ID，u32 全局 posting；本输入装入一个 2³² 地址块 | 唯一 pair owner；16B SmallPosting；8B 出生链；单调低频剪枝；OctonaryHeap lazy refresh；HF canonical ID | workers=1，batch cap=256 |',
             '| 当前非原子 4 worker | 与上项同宽度 | 同一算法；只读规划、按空间排序的 16B Plan、split_at_mut 独占写入、并行 owner 提交 | workers=4，batch cap=256 |',
             '| 当前原子 4 worker | 同宽 AtomicU16 Relaxed load/store；posting 相同 | 与非原子完全相同的批次、规划、排序、路由、切片写入和提交 | workers=4，batch cap=256 |',
             '| PR 4 worker | WordArena：每字符 Symbol{u32 ID,u32 length}，8B；candidate 携带历史 word cohort Vec<u32> | pair 计数串行；每轮一条规则；扫描候选词的 live symbol run，原地压缩；scratch/delta buffer 复用；OctonaryHeap；保持 HF 历史 cohort 账本 | Rayon=4；候选词数≥1000 才并行；默认阈值显式设为1000 |', '',
             '当前 AA 规则独占批次，用每 4096 个位置的摘要传播跨区间 parity；尾 run 先近邻顺查 8 项，再二分。reserved canonical 输出 ID 也采用单规则批次，保持同频次序。当前空 affix 的精确批次和低频剪枝证明见 [REVIEW.md](REVIEW.md)、[PAIR_MONOTONICITY.md](PAIR_MONOTONICITY.md)。非空 affix 仍走串行 cohort 路径，本案例没有测量其性能。', '',
             '## 总体结果', '',
             'Train 从 feed 完成后计到训练 API 返回，包含初始化、merge、输出及内部存储释放；Total 另外包含读取、feed 和词频汇总。输出 JSON 与模型摘要计算不计入。', '',
             '| 配置 | Feed s | Train s | Total s | 初始阶段 s | Merge s | 峰值 RSS GiB |', '|---|---:|---:|---:|---:|---:|---:|']
    for name,r in zip(names,[*rows,pr]):
        if r['indexed_stats']:
            init=r['indexed_stats']['initialize_ms']/1000
            merge=r['indexed_stats']['merge_ms']/1000
        else:
            init=sum(r['stage_ms'][k] for k in ['special_tokens','alphabet','tokenize_words','count_pairs'])/1000
            merge=r['stage_ms']['merges']/1000
        lines.append(f'| {name} | {r["feed_ms"]/1000:.3f} | {r["train_ms"]/1000:.3f} | {r["elapsed_ms"]/1000:.3f} | {init:.3f} | {merge:.3f} | {r["maxrss_kib"]/(1<<20):.3f} |')
    lines += ['',
              f'- 同算法非原子 4/1 worker：Train 加速 **{serial["train_ms"]/parallel["train_ms"]:.2f}×**，Merge 加速 **{serial["indexed_stats"]["merge_ms"]/parallel["indexed_stats"]["merge_ms"]:.2f}×**。',
              f'- PR 4 worker / 当前非原子 4 worker：Train **{pr["train_ms"]/parallel["train_ms"]:.2f}×**，Total **{pr["elapsed_ms"]/parallel["elapsed_ms"]:.2f}×**。',
              f'- 原子相对非原子 4 worker：Train 高 **{100*(atomic["train_ms"]/parallel["train_ms"]-1):.2f}%**，Merge 高 **{100*(atomic["indexed_stats"]["merge_ms"]/parallel["indexed_stats"]["merge_ms"]-1):.2f}%**。这个量级小于本轮单次、共享主机观测能可靠区分的范围，不能判断原子访问更快或更慢。', '',
              'PR 的初始阶段包括构造 symbol arena、计数和候选建堆；当前还包括专用 pool 建立与空间统计。阶段并非完全相同的源码边界，因此跨实现主要比较完整 Train。此前 4 MiB 结果采用不同输入、目标词表和旧源码，不能据两个规模的差值计算此次修复收益。', '',
              '## 阶段成本', '',
              '| 当前阶段 | 非原子1 s | 非原子4 s | 原子4 s |', '|---|---:|---:|---:|']
    for label,k in [('HF初始化/语料构造','tokenize_ms'),('初始位置路由','initial_route_ms'),('初始owner计数','initial_count_ms'),('初始过滤/建堆','initial_heap_ms'),('规则选择','select_ms'),('posting过滤/Plan拼接/排序/AA','plan_ms'),('delta与出生路由','delta_ms'),('语料写入','rewrite_ms'),('owner频率聚合/出生posting填充','commit_ms'),('提交后字典路由','route_ms')]:
        lines.append('| '+label+' | '+' | '.join(f'{r["indexed_stats"][k]/1000:.3f}' for r in rows)+' |')
    lines += ['', '| PR 阶段 | 秒 |', '|---|---:|']
    for k,v in pr['stage_ms'].items():
        lines.append(f'| {k} | {v/1000:.3f} |')
    p=parallel['indexed_stats']
    lines += ['',
              f'当前 4 worker 的 delta 与 commit 共 {(p["delta_ms"]+p["commit_ms"])/1000:.2f}s，占 merge 的 {100*(p["delta_ms"]+p["commit_ms"])/p["merge_ms"]:.1f}%；planning 共 {p["plan_ms"]/1000:.2f}s，占 {100*p["plan_ms"]/p["merge_ms"]:.1f}%；实际语料写入仅 {p["rewrite_ms"]/1000:.2f}s。HF 初始化及语料构造 {p["tokenize_ms"]/1000:.2f}s 仍为串行，并占完整 Train 的 {100*p["tokenize_ms"]/parallel["train_ms"]:.1f}%。', '',
              '源码确定的额外工作是全局 16B Plan 的串行拼接、混合规则空间排序，以及有效 posting 过滤后第二次走访生成 delta。原型普通批次在过滤时直接路由，使用4B starts，并没有这个全局排序。当前 plan_ms 包含这些工作及必要过滤/AA；它只是整个 planning 阶段耗时，不能把该值全部归给排序。delta/commit 也包含原型原有的必要更新，不能把它们全部称为额外成本。', '',
              '已修复的迁移偏差：每小块重复创建 worker 路由缓冲、Unicode 查询与大 pair 表更新交织、初始化建堆串行、分块 owner 重复扫所有计数、coordinator 串行安装分块出生 posting。当前采用每 worker 连续区间、保序 posting 和 owner/block 并行；与原型动态任务游标、BinaryHeap 仍有差异。两轮独立审查列出完整边界与剩余差异。本轮没有直接跑原始 Prepared API 原型，不能凭这张表断言其与迁移版本的完整速度差距。', '',
              f'三种当前配置均为 {p["batch_rounds"]:,} 个批次、最大 {p["max_batch_rules"]} 条规则、{p["posting_visits"]:,} 次 posting 访问。1/4 worker 同时改变 owner 分区和各表的工作集，因此初始计数的观测加速包含布局影响，不是仅减少同一表的执行时间。', '',
              '## 内存与 swap 观测', '',
              f'运行前上界估计：当前峰值 {env["estimated_peak_bytes"]/(1<<30):.2f} GiB，PR RSS 峰值 {prenv["estimated_rss_peak_bytes"]/(1<<30):.2f} GiB。按用户要求留出 1 GiB；每 0.5 秒监控可用内存和进程 VmSwap，达到下限或观察到进程 swap 即停止。', '',
              '| 配置 | 最低可用 GiB | 采样进程 VmSwap MiB | 全局换入页 | 全局换出页 |', '|---|---:|---:|---:|---:|']
    for name,r in zip(names,[*rows,pr]):
        m=r['memory']
        lines.append(f'| {name} | {m["minimum_available_bytes"]/(1<<30):.3f} | {m["sampled_peak_process_swap_bytes"]/(1<<20):.3f} | {m["pswpin_delta"]} | {m["pswpout_delta"]} |')
    core=sum(p[k] for k in ['initial_corpus_bytes','initial_posting_bytes','initial_pair_table_bytes','initial_block_table_bytes','initial_directory_bytes','initial_heap_bytes'])
    lines += ['',
              '训练进程采样没有观察到 swap，所有运行的可用内存均高于 1 GiB。主机在开始前已有其它进程占用 swap，全局仍有后台换页；全局计数不能归属给某个训练进程。这是共享主机上的关键单次观测，原子访问的微小差异尤其不能作稳定结论。两次更严格的早期全局换出监控中止记录保留在 parallel-key-512.jsonl 与 parallel-key-512-retry.jsonl，未计入结果表。', '',
              f'当前三个配置在初始化计数、过滤、建堆完成时的核心源码布局估算均约 {core/(1<<30):.3f} GiB；它不包含输入字符串/词频 map、临时初始化路由、分配器、线程栈或训练中 Plan。各组件真实容量字段在 JSON 中。PR 未加入相同初始化容量计数，不作相同精度的布局比较。', '',
              'PR 的 symbol Vec 按 UTF-8 字节数预留，每容量槽8B，本输入的虚拟 capacity 上界约4 GiB；只有实际写入的 Unicode 字符页成为 RSS。容量公式和物理常驻内存是两个口径，不能用预留尾部推断它已经用了 swap。完整初始化布局定义见 [MEMORY_LAYOUT.md](MEMORY_LAYOUT.md)。', '',
              '当前并行公开字段 stale_posting_visits、peak_birth_bytes 尚未收集，零值不代表零成本；corpus_bytes/posting_bytes 为初始化赋值，不代表最终或峰值。plan_ms 与 commit_ms 仍需细分，保留为下次有针对性的优化观测项。', '',
              '## 正确性与独立审查', '',
              '26 项索引测试通过，包含1,500组逐轮HF差分、跨块AA、四种存储布局的原子/非原子、完整ID域回退及跨worker全局出生阈值；之后单独补充并通过大于u32地址的4096计划、四线程递归切片写入测试。两个独立审查轮次及该新增测试复核见 [REVIEW.md](REVIEW.md)。测量后只增加这一测试，没有修改生产算法。', '',
              '## 数据与复现', '',
              f'- [当前三项逐次数据](results/{args.parallel.name}) 与 [environment](results/{args.parallel.with_suffix(".environment.json").name})：完整输入、二进制、源码与依赖锁摘要。',
              f'- [PR逐次数据](results/{args.pr.name}) 与 [environment](results/{args.pr.with_suffix(".environment.json").name})：固定PR提交、临时探针源码、runner线程开关、二进制与锁摘要。',
              '- PR只在临时副本加入六个Instant阶段探针；原checkout保持干净。build_pr_compare.py 固定提交并使feed串行、merge启用4worker。',
              '- run_parallel_key.py / run_pr_key.py 是关键测试脚本；run.py 的完整矩阵本轮没有运行。', '',
              '```bash',
              'cargo build --manifest-path benchmarks/hf-bpe/Cargo.toml --release --locked',
              'python3 benchmarks/hf-bpe/run_parallel_key.py benchmarks/hf-bpe/.build/gb-corpus --output benchmarks/hf-bpe/results/new-key.jsonl',
              'python3 benchmarks/hf-bpe/build_pr_compare.py benchmarks/hf-bpe/.build/pr-head',
              'python3 benchmarks/hf-bpe/run_pr_key.py benchmarks/hf-bpe/results/new-key.jsonl --output benchmarks/hf-bpe/results/new-pr-key.jsonl',
              'python3 benchmarks/hf-bpe/summarize_parallel_key.py benchmarks/hf-bpe/results/new-key.jsonl benchmarks/hf-bpe/results/new-pr-key.jsonl --output benchmarks/hf-bpe/PARALLEL_REPORT.md',
              '```', '',
              '语料准备由 prepare_gb_corpus.py、corpus_preflight.rs 完成，原始文本及构建目录不提交。首次准备需要pyarrow；来源和固定摘要随已保存environment保留。输出文件已存在时脚本会拒绝覆盖，应使用新文件名。']
    args.output.write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
