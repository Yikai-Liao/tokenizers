# 分片堆、精确 leader 与并行校验窗口

## 结果

保留统一实现：所有线程数均使用一层 leader 堆与每分片 4 项校验窗口，没有单线程特化。候选分支为 `bpe/validation-window`，最终实现提交 `bbf25ac4`，基点是当前出生提交优化版本 `37f7599e`。

协调线程不再为每个候选扫描全部 N 个分片。每轮在 O(N) 时间构建 leader 堆；每次消费或降低一个 leader 只维护该分片，全局维护为 O(log N)。局部堆操作、stale 查询与动态批次规则仍保留原来的精确语义。

当前单/四线程的完整 Trainer 对照没有发现稳定的整体回退。中文四线程两对方向混合，视为持平；当前单线程与英文四线程的改善不能全部归因于队列，未改阶段和随机 owner 哈希布局也有波动。49 次模拟支持移除未来几十分片下的逐候选线性扫描，模拟由 4 个真实 worker 执行，不代表测过 64 核机器。

## 实现与正确性

原实现每次调用所有 owner 的 `peek_current`，查询各自频率账本、校正 stale heap top，再比较全局最大。新实现保留这些局部堆和账本，增加两个状态：

- 局部窗口：owner 的既有提交任务结束时，校验并弹出最多 4 个局部精确候选。初次初始化也复用已有 owner 工作。没有新增每次 pop 的并行调度或等待 barrier。
- 全局 leader：保存每个分片当前 top 的上界。验证全局最高 leader；下降后更新它并重新比较，直到该 leader 等于局部精确 top。消费后只更新这个 owner 的上界。

频率在 batch 选择期间冻结，旧 pair 的真实频率不超过其 heap 上界。局部校验得到一个精确有序前缀。若全局最高上界与相应局部精确值相等，它也不低于其它全部分片上界，故是精确全局最大值。频率与 pair ID 的同频顺序都参与比较。

协调线程沿用原来的冲突、AA、预留 ID、词表容量和长度规则。窗口耗尽时按需刷新获胜分片。提前停止后把所有未使用窗口候选归还，在下一次频率更新前清除本轮认证。worker 对各自 owner 独占写入，选择与更新之间沿用已有阶段边界；没有新增共享原子协议或 unsafe。

`leader` 控制只懒校验全局最高上界；`cached` 控制缓存局部精确 top 但仍扫描 N 个分片；`serial` 控制保留原逐分片校验。最终默认 `bulk4` 在所有线程数使用相同流程。中间为单 worker 选择 `serial` 的 `06d1be3f` 已按用户要求撤销，相关记录保留为未采用方案。

## 真实语料：单、四线程采用条件

原 Trainer API、none、min frequency 2；中文 512 MiB 词表 50,000，英文 16 MiB 词表 18,124。feed 串行并固定 word map seed `(11,13,17,19)`；owner 内部的默认 AHash seed 没有固定。比较完整训练函数，包括初始化、合并、最终清理。数据、编译、库测试和模拟计时均错开。

原最佳版本使用已冻结的 `native-dense-birth-commit-seeded`；最终候选使用 `native-validation-window-v3`。不同 binary 的来源、依赖、源码及实际输入摘要均在 environment 文件中。公开模型 schema 未改。

| 语料 / worker | 原最佳 train | 统一方案 train | 变化 |
|---|---:|---:|---:|
| 中文 512 MiB / 1 | 52.531 s | 50.754 s | −3.38% |
| 中文 512 MiB / 4，第 1 对 | 16.523 s | 16.631 s | +0.66% |
| 中文 512 MiB / 4，反转顺序第 2 对 | 16.646 s | 16.483 s | −0.98% |
| 英文 16 MiB / 1 | 4.228 s | 4.102 s | −2.99% |
| 英文 16 MiB / 4 | 1.830 s | 1.756 s | −4.05% |

中文四线程两对平均 16.585 → 16.557 s，约 −0.17%，按基本持平处理。两对 merge 时间反而略增约 0.3%，不宣称已有中文四线程端到端提速。该有限对照没有证明所有输入严格零退化。

中文首对的选择阶段：单线程 332 → 294 ms、四线程 218 → 179 ms；嵌在提交阶段的 worker 校验时间之和分别为 16 ms、47 ms。它不能直接作为墙钟加到多个并行阶段，也不能只报告 `select_ms` 而遗漏搬到 commit 的工作。

最初同 binary 的四线程控制中，原选择路径、leader、bulk4 的完整训练为 19.419 / 18.099 / 17.774 s；未改初始化也从 5.424 降到 4.615 s，commit 有明显波动，因此没有将这组 8.5% 差额写成队列收益。

所有中文模型为 `d50fb836…`，英文为 `854e76ea…`，完整有序 merges 与 vocab 摘要一致。中文各模式同样处理 203,230,114 初始边、125,409,599 posting visits、1,550 batch。最终单线程峰值约 3.83 GiB，四线程约 3.33–3.38 GiB，与各自原版本接近；所有 native 调用的 posting 生命周期计数相等、进程 VmSwap 为零。

## 几十分片模拟

模拟直接编译实际 `CandidateHeap`、`Owner::peek_current` 与 `validation_window` 实现。频率表使用相同 16 B 空 posting 载荷，保留 32 B 的 key/value 记录布局；不建立 corpus 和 rewrite。固定 owner map seed，使用实际 hash 分片函数。每次选择 16,000 项，执行相同的动态冲突停止、旧频率下降/删除与新身份出生。

40 次固定工作模拟：1,048,576 初始候选，4/16/32/64 分片，0%/75% 初始 stale，serial/cached/leader/bulk4/bulk16。另有 9 次增长负载：每分片 262,144 候选，16/32/64 分片，75% stale，serial/leader/bulk4。所有模式的完整有序 `(pair, frequency)` SHA256、选择数量和 batch 数一致。

下表是每种负载的 64 分片结果；pipeline 包含选择、状态更新与预校验，不只统计 coordinator。

| 负载 | 原方案 pipeline | leader | bulk4 | 原方案探查次数 | bulk4 探查次数 |
|---|---:|---:|---:|---:|---:|
| 固定 1,048,576 项，fresh | 103.9 ms | 55.1 ms | 62.8 ms | 1,062,528 | 72,218 |
| 固定 1,048,576 项，75% stale | 114.2 ms | 98.4 ms | 72.6 ms | 1,062,976 | 72,563 |
| 增长至 16,777,216 项，75% stale | 114.0 ms | 79.4 ms | 70.7 ms | 1,030,976 | 41,568 |

fixed fresh 的 truth 查询从 1,062,528 降到 bulk4 的 158,784；stale 从 1,115,159 降到 212,888；增长负载从 1,083,059 降到 92,068。O(N) 重复查询与比较被移除，局部 stale 校正的一部分由现有 worker 执行。

较大的窗口不合算：fresh 64 分片的 bulk16 做了 634,880 次 truth 查询，pipeline 129.1 ms，慢于原方案。bulk4 为 62.8 ms。即使 bulk4 也会校验未消费候选，pure leader 在 fresh 场景更省工作；这些结果支持保留小窗口，不证明 4 对所有工作负载最优。

49 次模拟均无进程换页，最大峰值约 1.31 GiB，最低 MemAvailable 约 6.28 GiB。该模拟改变的是分片数及工作集规模，实际 worker 仍是 4；它不能预测真正 64 核的带宽、调度和 NUMA 行为。

## 验证与复现

初始统一方案完整 66 项库测试通过。撤销临时单线程分派后重新核验两个核心差分：真实 BPE 在 1/4/32 worker、16/32 offset、宽权重/零权重、预留 ID、长度限制下的完整轨迹；局部队列在 1/4/32/64 分片、packed/wide、缺失/下降/出生/提前停止下的精确序列。最终 native 完整模型再次核验。

共完成 49 次模拟、19 次 native 调用，其中含 5 次烟雾检查与 1 次未采用的临时单线程分派测量。结果目录含全部实际记录，未把中间方案写成最终收益。

- [开发计划](VALIDATION_WINDOW_PLAN.md)
- [完整分析](results/validation-window/analysis.json)
- [固定总量模拟](results/validation-window/fixed-simulation-summary.json)
- [增长负载模拟](results/validation-window/weak-simulation-summary.json)
- [最终轨迹检查](results/validation-window/unified-trace-tests.log)
- [最终队列协议检查](results/validation-window/unified-protocol-tests.log)

构建使用 `build_validation_window.py WORKTREE --label validation-window-v3`；真实语料使用 `run_validation_window.py --mode auto --workers 1|4 ...`。模拟使用 `build_validation_sim.py WORKTREE` 和 `run_validation_sim.py`，增长负载增加 `--weak`。结果文件名拒绝覆盖，重跑须更换 case 名。首轮 native 与模拟来自 `10e2a2e2`，最终统一实现在 `bbf25ac4`；原始 provenance 保留各自准确版本。

队列探索到此停止。后续权重排序与等权重区间另起分支，从保留的统一版本开展完整训练对照，并将排序成本计入收益。
