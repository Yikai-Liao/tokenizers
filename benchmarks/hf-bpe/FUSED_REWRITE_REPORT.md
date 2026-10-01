# BPE 融合 rewrite 与出生 commit 对照

日期：2026-10-01。开发计划见 [FUSED_REWRITE_DEVELOPMENT_PLAN.md](FUSED_REWRITE_DEVELOPMENT_PLAN.md)，完整调用清单、阶段计数与摘要见 [analysis.json](results/fused-rewrite/analysis.json)。

## 结果与选型

普通规则与 AA 共用的融合遍历已完成。它直接遍历原 posting，按地址及 token 长度划分独占写区，立即校验、计算 delta 并 rewrite；支持跨 block、完整 64 位基址及普通/atomic 槽位。全部 68 个库测试通过。

这轮最有价值的改动是出生 commit：按规则和方向保存出生分组，用邻居 ID 目录汇总所有任务片段，再一次生成最终 posting。它也适用于原 prepare/rewrite 阶段。512 MiB 中文中，“旧遍历 + 新 commit”在两次与融合普通 u32 的对照中都更快，因此推荐独立分支 `bpe/dense-birth-commit`；该分支全部 64 个库测试通过。融合实现保存在 `bpe/fused-aa-rewrite`。

独立 release 构建通过原 Trainer API 的最后一对验证如下。每个单元格为实际完成的一次调用，包含 train 返回前的 posting 与 arena 释放。

| 输入 | 实现 | 完整 train | initialize | merge | commit | 峰值 RSS |
|---|---|---:|---:|---:|---:|---:|
| 中文 512 MiB，50k 词表 | 原基线 | 21.029 s | 5.332 s | 15.352 s | 5.754 s | 3.383 GiB |
| 同上 | 出生 commit 优化 | **19.307 s** | 5.382 s | 13.623 s | **4.099 s** | 3.361 GiB |
| 英文 16 MiB，18,124 词表 | 原基线 | 2.285 s | 0.306 s | 1.945 s | 0.479 s | 0.269 GiB |
| 同上 | 出生 commit 优化 | **1.953 s** | 0.282 s | 1.644 s | **0.347 s** | 0.257 GiB |

中文完整训练减少 **8.2%**，commit 减少 **28.8%**；英文完整训练减少 **14.5%**，commit 减少 **27.5%**。这些是本机这次独立构建对照的结果。独立对照每个输入一对，前面的同二进制实验另列；没有把不同构建的结果混为重复样本。

推荐代码版本为 `a0f5801809d0b799ad43f969901fe876081955b4`，父版本为 `adb219cda4600e52ea2809d2537ac551c168dd27`。公开训练参数和模型 schema 保持原格式。旧任务切分、有效位置表与 rewrite 屏障继续提供并发语义。原冻结分支保留供对照。

## 正确性：旧状态也能计算 CD/AB

本批选定规则时已分配 replacement ID：`(C,D)→CD`、`(A,B)→AB`。处理 CD 的右边界时，读取旧 `A,B` 并查询本批规则，即可删除旧边 `(D,A)`、生成最终边 `(CD,AB)`。处理 AB 时识别左侧 CD 已被本批选定，跳过左边界。delta 使用旧语料与规则映射即可投影最终邻接。

原实现先完成全部旧状态读取，再统一 rewrite。直接在旧规则 posting 数切分上融合写入，会出现跨线程中间状态：CD 先写 `C→CD`、尚未改写 D，AB 可能看到 `CD,D`，错误地生成 `(D,AB)`。仅使用 AtomicU32 无法修正这个协议。

本次融合实现把相交或相邻的潜在合并跨度放进同一个任务。CDAB 的两个区间因此不会跨任务；任务内按规则顺序处理，识别本批新 token，规范化旧邻居与最终邻居。两种顺序都只生成一次 `(CD,AB)`。任务间的未改写间隙以只读切片借用，独占写区通过安全的 `split_at_mut` 划分；不保存语料旧值副本。

切分只读取原 posting 地址、数量、规则长度及少量样本。每规则的分段累计下标支持跨 block 定位；邻接组件调整覆盖 stale posting 的潜在跨度。单规则连续段使用附近检查与二分跳转。普通长 ABAB 与 AA 连续链可留在一个任务；20,000 个跨 block posting 的测试验证了该行为。模拟基址超过 2^32 的测试覆盖地址组合；未实际训练 100 GiB 输入。

## 出生 commit 优化

出生 key 包含本批新身份，该身份确定规则。左出生的邻居未参与本批合并；两段都被选中时，最终边由左段的右侧产生。因此最终出生 key 可唯一归入 `(规则, 左/右方向)` 桶。

线程局部 Scratch 将出生组直接追加到对应 owner 的桶，省去出生 flush 的哈希入表。owner commit 用一个词表大小的邻居目录跨任务累加权重、数量和片段链，省去临时出生 key 哈希表。频率门槛在全部片段汇总后检查；接受的 key 一次分配最终 posting，逆向填充原 Node 链，最后插入 ledger 并进入 heap。

目录在桶间复用，只重置触及项。每 owner 的目录为 `4 × 身份数` 字节；现有聚合路径的身份上限为 65,536，四个 owner 的目录合计至多 1 MiB。桶向量、合计组及片段链还有随实际分组数增长的存储，完整 RSS 已计入。

推荐独立分支用于原 shared-corpus、非 AA、flat、身份数不超过该上限的 prepare 路径。AA、generic 多 block、高身份数及原非 shared-corpus 路径使用原 commit；身份别名配置沿用串行兼容入口。旧边删除仍会查询 ledger，Node 链仍须读取并写入最终 posting。

## 融合候选的实际结果

### 最初两版

v1 删除了独立有效位置表与 rewrite 遍历，但中文 merge 未改善，commit 变慢。v2 将出生片段先汇总再一次插入 ledger，减少逐片段全局目录查询。v2 同二进制中文对照为：

| 实现 | train | merge | commit |
|---|---:|---:|---:|
| 原基线 | 20.689 s | 15.224 s | 5.794 s |
| 融合 AtomicU32 | 20.900 s | 15.746 s | 6.316 s |
| 融合普通 u32 | 21.598 s | 16.695 s | 6.832 s |

空间切分把同一个 pair 分散到更多任务输出中。实际 delta 分组片段由 **57,835,113** 增至 **91,036,137**，增加 **57.4%**；原 posting 访问数都为 **125,409,599**。这是新增汇总工作的直接计数。

英文 flat v2 完整训练为 1.945→2.047 s，未改善；强制 block bits=16 的 generic 同输入为 6.882→5.883 s，减少 14.5%。generic 结果仅完成一对，对应实验分支；推荐独立分支的 generic 路径仍沿用基线。

### 同时优化 commit 后

v3 在同一二进制中比较旧遍历/旧 commit、旧遍历/新 commit、融合/新 commit。源码为 `fa6934882237f423e226f5d8e013062d20599d1c`。

| 中文组合 | train | merge | commit |
|---|---:|---:|---:|
| 旧遍历、旧 commit | 23.850 s | 17.621 s | 7.632 s |
| 旧遍历、新 commit，首次 | 19.384 s | 14.050 s | 4.692 s |
| 融合 AtomicU32、新 commit | 19.984 s | 14.388 s | 5.073 s |
| 融合普通 u32、新 commit，首次 | 19.476 s | 14.176 s | 4.792 s |
| 融合普通 u32，反向对照 | 20.178 s | 14.974 s | 4.792 s |
| 旧遍历、新 commit，反向对照 | 18.420 s | 12.987 s | 3.977 s |

英文同二进制旧遍历/旧 commit 为 1.899、1.899 s，新 commit 为 2.029、1.899 s，方向混合；融合普通 u32 为 2.033 s。独立构建验证见首表。初始化和代码生成/表布局的变化没有归入 commit 的局部收益，不把新 commit 的收益记作融合 rewrite 的收益。

## L1 data-load 为什么增加约 49%

此前的“读取次数”指 perf 的整段 merge **L1 data-load 事件数**，包含语料、posting、聚合组、哈希表、描述链等访问。它不是原语料读取次数。

v2 整段 merge 的诊断中，L1 data-load 由 173.190 亿增至 257.233 亿，增加 48.5%；instructions 增加 37.4%。随后用独立阶段开关重测原路径与 v2 的哈希出生融合路径，获得：

| 计数范围 | 原路径 L1 data-load | 融合哈希路径 | 增幅 |
|---|---:|---:|---:|
| 旧 prepare + plan + rewrite / 融合遍历 | 140.229 亿 | 184.304 亿 | +31.4% |
| commit | 30.324 亿 | 68.403 亿 | +125.6% |

在这组阶段诊断中，额外事件约 46.4% 来自 commit，约 53.6% 来自遍历。commit 有更多分组片段，以及新增描述链和多层索引；融合遍历还需要解读已改写邻居、访问 stream 元数据并 flush 更多任务聚合组。这些是代码中增加的工作。阶段计数定位了增量范围，尚未逐指令证明每一种操作贡献多少。

阶段开关只在对应窗口启停事件，使用 perf FIFO 确认；每批有启停开销，诊断计时不参与排名。阶段诊断关闭任务日志；早期整段 merge 诊断另有任务日志。两组计数来自独立调用，不要求阶段和精确复原早期整段值。

### 缓存命中与并行空泡

v2 整段诊断的 L1 miss 比例从 11.68% 降至 8.77%，但 miss 总数从 20.223 亿增至 22.548 亿，增加 11.5%。比例下降同时发生在更多 load 上，不能据此报告减少了缓存未命中。通用 cache-misses/cache-references 为 66.69%→63.27%，该事件不代表所有缓存层。

各可用事件的 running 比例为 100%；保留原 CSV。该 KVM 主机的 cycles 事件在 CPU 工作探针中为 0，判为不可用，未计算 IPC 或基于 cycles 的收益。报告仅使用实际事件计数，未声称某条语料 load 产生了精确数量的 miss。

v2 merge worker CPU 时间比例为 73.3%→78.6%，同时 worker 总 CPU 时间由约 49.9 s 增至 55.3 s。更高比例伴随额外工作，不能单独证明空泡减少。任务轨迹中旧 prepare 的执行跨度覆盖约 78.2%，旧 rewrite 约 64.8%，融合约 85.1%；旧轨迹覆盖非 AA flat 批次的 98.55% 原 posting，融合轨迹覆盖全部批次，且日志有开销。

原 posting 数量的均衡效率：旧 prepare 约 99.998%，融合约 96.326%。融合消除了一个单独 rewrite 阶段，但这轮没有得到足以将端到端收益归因于“更高缓存命中”或“更少并行空泡”的证据。

## 核验与实际调用数

实际完成 **44 次训练调用**：25 次正式对照、11 次非诊断模型检查、8 次诊断。阶段诊断的小语料检查计入诊断。未将构建、计划调用、单元测试或取消的调用计入训练样本。

所有保存调用通过完整模型签名、posting 分配/退休闭合及进程无 swap 检查。正式训练与编译、库测试及其它本任务 CPU 重工作分开；诊断不参与正式排名。MemAvailable 低于 1 GiB 停止；峰值取内核 HWM 与采样 RSS 最大值。

中文模型 SHA-256：`d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd`。英文 16 MiB：`854e76eae8d379f8f2cb26f22363c55c56b6429c595649b31f05c29a55ed7d9e`。

同一输入、词表、min_frequency=2、none 前端、4 个初始化/merge worker、动态 arena 策略及 release 参数。feed 串行且哈希种子为 `(11,13,17,19)`，固定语料物理顺序；没有固定所有内部临时哈希表的种子。分析脚本检查配对输入、模型、初始边数、posting 访问数和批次数一致。

## 复现与保留内容

- [build_fused_rewrite.py](build_fused_rewrite.py)：同二进制恢复基线及候选；`--phase-controls` 支持分阶段事件与原哈希出生候选。
- [run_fused_rewrite.py](run_fused_rewrite.py)：模型/生命周期/内存门控、perf 窗口；`--mode baseline-dense` 单独测旧遍历的新 commit。
- [fused_rewrite_diagnostics.py](fused_rewrite_diagnostics.py)：benchmark overlay 的线程 CPU 时间、可选任务轨迹和事件启停。
- [build_standalone_bpe.py](build_standalone_bpe.py)、[run_standalone_bpe.py](run_standalone_bpe.py)：独立 source 的原 Trainer API 对照。
- [analyze_fused_rewrite.py](analyze_fused_rewrite.py)：只读取实际完成的调用，生成清单与配对比较。

每个 case 保留 control、environment 或诊断 provenance、stdout/stderr、summary 与 JSONL；perf case 另有事件 CSV。构建目录保存完整源码 overlay 与 `instrumentation.patch`，environment 保存实际源码、二进制及输入摘要。不同 case 不覆盖已有结果。

示例（使用本机 checkout 与已保存输入；更换 case 名再执行）：

```bash
python3 benchmarks/hf-bpe/build_standalone_bpe.py \
  /root/code/tokenizers-worktrees/dense-birth-commit --label dense-birth-commit-seeded
python3 benchmarks/hf-bpe/run_standalone_bpe.py \
  --variant dense --worktree /root/code/tokenizers-worktrees/dense-birth-commit \
  --label dense-birth-commit-seeded --case zh512m-native-dense-repro \
  --corpus benchmarks/hf-bpe/.build/gb-corpus/zh-512m.txt --vocab 50000 \
  --expected-model d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd
python3 benchmarks/hf-bpe/analyze_fused_rewrite.py
```
