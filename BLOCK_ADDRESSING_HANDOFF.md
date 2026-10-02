# BPE 局部地址与统一 posting 设计交接

更新日期：2026-10-02。工作目录：`/root/code/tokenizers`。

## 1. 用户真正要求解决的问题

**重新设计 block 存储及执行流程，使大语料仍能使用局部 u32 地址，并尽可能保持现有 flat 的速度。单 block 应极其接近 flat；未来应能使用统一实现，不再维护一套专门的 flat 算法。**

用户明确指出，BPE 具有局部性，并不需要让每条位置记录都承担强全局寻址。扩展到大语料时，应把区域身份放在较粗粒度的描述里，而不是扩大每条位置记录或复制整套 pair 索引。

当前 block 设计已经出现两个实质问题：

- 单 block 字典路线比正常 flat 训练慢 **3.78 倍**，不能解释成少量块管理开销。
- 完整中文 512 MiB 的 u16 多 block 测试未完成，目标进程 RSS 达到 **6.58 GiB** 后触发可用内存门槛。

用户要求重新分析算法设计，并认为需要大幅调整。**本次交接时尚未实施新的统一存储设计。** 用户最后要求先写本文件，换模型继续分析；当前代理停止这项实现工作。

语言差距分析是另一个独立任务，现已由其所有者完整交付。用户随后授权 review/提交；当前代理仅轻量核验并归档交付文件，没有重跑采样、编译或 native。结论与入口见第 11 节。

## 2. 当前源码及工作区

| 项目 | 当前状态 |
|---|---|
| ROOT 分支 | `bpe/experiments` |
| ROOT 源码交付提交 | `0e899a416110a8f0533d7eaee806c58ee2e9c562`；后续语言分析与交接文档提交未改生产源码 |
| HEAD 内容 | 已交付的 affix 快速引擎迁移、第一轮 block 融合 prepare、历史报告及结果 |
| 性能测试用干净源码 | `/root/code/tokenizers-worktrees/block-fused-prepare` |
| 干净源码提交 | `1959202f30673fc21e68ab2868ba40213062c409` |
| 源码关系 | ROOT 的整个 `tokenizers` 子树与上述提交一致；不是把少量补丁套在旧归档代码上 |
| 第一轮 block 前基线 | `fd300ca81d1531fdaec622d7947d5072731431e4` |
| 第一轮片段实现 | `d4723fc2be33306889d24b57d9c6346c1c2a652b` |
| 后续小调整 | `1959202f` 将每个切片的有效偏移 Vec 合并为每个 job 的连续 Vec |

`1959202f` 的正常 flat 路径是当前可复用的快基线。**后续候选必须与最快的有效 flat 组合比较，不能只与已经明显较慢的 block 路线比较。**

历史 `7c37e202` 是旧归档源码，不应成为本次优化基点。之前失败的六项新想法也不是当前最佳基线。

### 尚未提交的相关工作

- [run_affix_analysis.py](benchmarks/hf-bpe/run_affix_analysis.py)：修正测试脚本保留完整语料缓冲区的问题，见第 7 节。
- [build_posting_width_comparison.py](benchmarks/hf-bpe/build_posting_width_comparison.py)：隔离构建中强制 `flat = false`，用于观察同一 block 算法在 u32 单 block 下的成本；不修改生产源码。
- [run_posting_width_comparison.py](benchmarks/hf-bpe/run_posting_width_comparison.py)：完整中文 512 MiB 三个布局顺序运行，失败也保留。
- [posting-width-zh512 结果](benchmarks/hf-bpe/results/posting-width-zh512/)：三次实际调用，两个成功、一个资源停止。

语言分析的 Markdown、四个脚本及 `results/language-training-gap/` 已获所有者授权，由本轮交付提交归档；本交接文件也随同提交。`pelican-bicycle.html` 是用户的无关文件；生成的 `results/affix-analysis/input/*.txt` 也仍未提交，均须保留。不要使用无差别 `git add .`。

## 3. 完整中文 512 MiB 对照：已有强证据

### 输入与条件

- 输入：`benchmarks/hf-bpe/.build/gb-corpus/zh-512m.txt`。
- 实际字节数：`536870289`。
- SHA256：`a0d40d4102cba1e933f25e5ccd17552d2eaebaec4ef770bc39d8e46e740b4d4f`。
- 不设置 prefix/suffix；整行 feed，保留换行，没有额外预分词。
- vocab `50000`，min_frequency `2`，merge workers `4`，初始化线程池配置 `4`。
- 语料槽为 `AtomicU32`、每槽 4 字节。这里改变的是 posting offset 类型，不是 token ID 槽宽。
- 两个成功案例均有 `206089945` 个初始槽位、`204660029` 个初始符号、`203230114` 条初始边、`1429915` 条唯一整行。
- 两个成功案例均有 `2697517` 个初始不同 pair、`29243` 次 merge、`1550` 个批次、`1435` 个融合非 AA 批次、`125409599` 次 posting visits。

### 三个实际案例

| 案例 | 布局及算法 | train s | initialize s | merge s | RSS GiB | VmSwap |
|---|---|---:|---:|---:|---:|---:|
| `zh512m-u32-flat` | 正常 u32 单 block、原生产 flat 路线 | 16.288 | 4.355 | 11.590 | 3.297 | 0 |
| `zh512m-u32-block` | u32 单 block、隔离强制 block 字典路线 | 61.546 | 31.220 | 29.623 | 4.175 | 0 |
| `zh512m-u16-block` | u16 局部 offset、多 block、新融合 block prepare | 未完成 | 无最终统计 | 无最终统计 | 6.576，停止前观测 | 0 |

u16 在 **15.396 秒**触发 `MemAvailable <= 1 GiB`，最低观测可用内存为 `1045401600` 字节。目标 RSS 为 `7060516864` 字节。它没有产出完整模型或最终阶段统计，不能写成完整训练耗时，也不能声称已完成 u16/u32 的耗时对照。

此前在 commentary 中将 u16 停止归为初始化阶段。**现有日志没有阶段标记，这个精确阶段尚未独立证实。** 源码和已完成 256 MiB 结果支持初始块字典内存是重点调查方向；若需要确定失败瞬间的阶段，应加有限阶段标记，而不是把推断当成测量。

两个成功案例的完整模型 SHA 均为：

```text
d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd
```

它们匹配已有正确结果，主/推测 posting 分配 session 的请求与释放检查通过。实际 OS 线程峰值均为 5。

### 阶段差距指向哪里

| 指标，ms | 正常 flat32 | 强制单 block 字典32 |
|---|---:|---:|
| initial route | 901.867 | 29150.407 |
| initial count | 2318.113 | 461.681 |
| initialize 合计 | 4354.928 | 31219.716 |
| delta，包含融合 prepare | 6845.346 | 9074.277 |
| commit | 3688.305 | 4639.174 |
| route | 17.748 | 14855.493 |
| rewrite | 635.344 | 685.224 |
| select | 182.931 | 230.226 |

注意：两种 initial route/count 字段的工作划分不同。字典的 `initial_route_ms` 包含物理块内构建和分组，不是单纯复制或分发位置；不能按字段名直接比较某个微操作。融合 prepare 已包含在 delta 和 merge 内。

merge 总差额是 `18032.725 ms`，route 字段差额为 `14837.744 ms`，算术上约占 **82.3%**。这将调查范围明确指向出生安装及目录更新；它还不是对某个哈希函数、分配器或锁的 CPU 归因。

### 分配与存储差异

| 指标 | flat32 | 强制单 block 字典32 |
|---|---:|---:|
| 初始 posting 分配容量，bytes | 802967868 | 1147872496 |
| 初始 owner pair 表估计，bytes | 138412096 | 276824128 |
| 初始块内 pair 表估计，bytes | 136 | 209715352 |
| arena buffers，累计 | 9150979 | 13349666 |
| posting grows，累计 | 0 | 6528097 |
| heap buffers，累计 | 172733 | 416324 |
| heap requested，累计 bytes | 876966116 | 2022603928 |

初始字典路径逐项 push 后保留扩容空余，owner 表和 block 表都保存 pair 相关状态。flat 则稳定分组后按最终数量分配 posting。累计请求量、分配容量、表容量估计和 RSS 是不同指标，不能互换。

原始结果、环境和构建 provenance 均在 [结果目录](benchmarks/hf-bpe/results/posting-width-zh512/)。`plan.json` 记录参数、三个命令和成功/失败事实。每个案例有 `.jsonl`、`.environment.json`、stdout/stderr；`provenance/` 保存三个构建 manifest 和完整隔离 patch。

## 4. 当前设计为何没有自然退化为 flat

### 状态归属存在两套含义

主文件：[parallel.rs](tokenizers/tk-train/src/trainers/bpe/indexed/parallel.rs)。

- `Entry` 约第 232 行：`frequency` 和 `blocks: SmallPosting`。
- `Owner` 约第 237 行：全局 pair 账本、候选堆和选择窗口。
- `Block` 约第 268 行：`base`、权重 metadata，以及自己的 `postings: AHashMap<pair, PackedPosting>`。
- `flat` 判定约第 872 行。

`Entry.blocks` 在 flat 路径里直接保存位置，在字典路径里保存 block 目录。同一字段承担两种含义，初始化、选择、prepare、提交和退休都据此分岔。**因此当前 single block 字典并不是 flat 只加了一个 base：它仍完整经过另一套索引和安装流程。**

### 初始化没有统一

flat 在初始 ID 域允许时使用：

1. [owner_route.rs](tokenizers/tk-train/src/trainers/bpe/indexed/parallel/owner_route.rs) 直接填入最终 owner 的有序记录流。
2. [radix_count.rs](tokenizers/tk-train/src/trainers/bpe/indexed/parallel/radix_count.rs) 稳定排序、分组、统计频率、按数量安装 posting。
3. posting 和频率直接由 owner 持有。

字典路径约在 `parallel.rs` 第 972 行后：

1. 按物理 block 分 wave，每块调用 [bounded_initial.rs](tokenizers/tk-train/src/trainers/bpe/indexed/parallel/bounded_initial.rs)。
2. 块内逐项建 posting 哈希表；表大后按 tile 排序，再逐项 push 到 posting。
3. 将每块 pair 频率汇总到 owner；owner 还保存 block 目录。
4. 所有块完成后再应用全局门槛，并在块表中过滤。

**强制单 block 时只有一个块初始化任务。** `initialization_workers=4` 说明线程池配置，不说明这段块字典构建实际有四路并行。该限制与逐项建表/扩容一起构成需要取消的设计成本，不应包装成已经充分并行。

### 出生结果仍被二次组织

第一轮新实现：[fused_batch/block.rs](tokenizers/tk-train/src/trainers/bpe/indexed/parallel/fused_batch/block.rs)。

已经解决：

- 非 AA 批次不再建立逐位置 `Plan { position, rank }` 或排序全部位置。
- `Selected` 读取旧语料恢复相邻选中规则；支持跨块邻居和跨多个 block 的长 token。
- 每个 job 使用局部 u32 有效偏移、8 字节出生链节点；片段保存目标 block。
- prepare join 后才并行改写；选中源 posting 在最后一个切片完成后释放。

但 `Prepared::install` 约第 596 行仍做：

1. 全部片段按目标 block 路由。
2. 每个 block 重新建 pair→count 哈希表。
3. 在块内 posting 哈希表插入最终 posting。
4. 再遍历片段并再次查块内 pair，填入链节点。
5. 产出目录，再查 owner 的 pair 表并追加 block ID。

该安装主要按 block 并行，single block 下只有一个安装任务。它与 flat 的 [flat_commit.rs](tokenizers/tk-train/src/trainers/bpe/indexed/parallel/flat_commit.rs) 差异很大：后者按 owner 并行，利用“规则、方向、邻居”的唯一生产者关系做 dense 聚合，直接分配和填入最终 owner posting。

所以前一轮只取消 Plan，没有收回块字典、目录和出生安装的重复工作。小型强制多块测量的 modest 收益不能证明这个设计满足大语料目标。

## 5. 多 block 内存证据及范围

完整 512 MiB 的 u16 案例没有最终计数。已有完整中文 256 MiB u16 字典结果可帮助理解存储成本：

- 1555 个真实物理 block，每块 65536 个槽位。
- 初始全局不同 pair：1901618。
- 初始 `(block, pair)` 条目：51055597。
- 块内表估计容量：2545967408 字节，约 2.37 GiB。
- 峰值 RSS 约 4.90 GiB。

同一个 pair 在许多块出现时，当前结构在每个块复制一个带哈希桶和 posting 对象的条目，又在全局 owner 保存目录。offset 从 4 字节缩为 2 字节，没有同时消除这些元数据。

上述 256 MiB 条目数是实测；**不要直接将它乘二写成完整 512 MiB 的已测条目数。** 内存门槛不是语料改变或候选偷换：512 MiB 输入确实完整传入运行器，目标因资源检查停止。测试没有终止其他用户服务，也没有降低门槛。

## 6. 待新模型评估的方向

下面是候选假设，尚未实现或测量，不能直接当作正确、快速的最终设计。

### 方向：owner 持有统一的分段 posting

每个 pair 只有一个全局 owner 及一份频率账本。posting 按局部地址区域分段：

```text
pair entry
  frequency
  posting
    segment descriptor: region/base + position range/count
    positions: local u32 offsets
```

base/region 保存一次，热循环遍历局部 offset；全局地址只在语料访问或跨段边界需要时还原。单 block 的 segment 表示应尽量保持当前 SmallPosting 的布局与访问成本。允许单段内联表示，不应为它复制一整套选择、初始化和提交算法。

需要同时设计的事项：

1. **初始化**：推广当前 owner 直接路由与稳定分组，最终仍归 owner；不要先重建一套块内 pair 表再转回。局部频率必须跨所有区域汇总后才能应用门槛。
2. **选中 posting 调度**：按规则及有序 posting segment/slice 拆任务；并行度由工作量决定，single block 的大 posting 仍可拆给多个 worker。
3. **融合 prepare**：保留旧语料与 Selected 的判断；有效位置只保存局部地址，region 身份按任务保存。
4. **出生提交**：推广现有 flat dense commit 的规则/方向/邻居聚合，直接生成最终分段 posting。目标区域是出生边起点所属区域；避免“按 block 路由→建表→安装→再建 owner 目录”。
5. **旧 pair 退休**：由 owner 一次销毁其全部 posting segment，避免再通过目录逐块查同一个 key。
6. **元数据成本**：segment 描述必须按真实段数计量。即便去掉哈希表，若为每个单次出现分配一个 Vec 或大描述，也可能继续失败。尤其 u16 很多小块的情形必须估算和测量。
7. **规模边界**：每段局部 offset/count 的上限、跨段总位置数、链节点索引、零权重段、全局 base 的类型需要明确。不能因地址改为局部 u32，就把跨段总长度或权重也误缩为 u32。

应评估是否还能减少全局查询、分配和数据搬运。用户的目标是局部整数地址加低额外成本；不要只把字段改名、加入新的适配层或再造一个单 block 特化分支。

## 7. 测试脚本的已修正内存干扰

旧 `run_affix_analysis.py` 在 `--require-absent-affixes` 下无条件执行 `raw = corpus.read_bytes()`，即使 prefix/suffix 均为 None，也将整个输入缓冲保留到 native 训练结束。512 MiB 对照因此会额外消耗父进程约 512 MiB；它不在目标进程 RSS 中，但会影响主机 MemAvailable。

当前工作区已修正：NONE 跳过 marker 读取；实际非空 marker 使用 1 MiB 分块扫描并保留跨块 overlap。空 marker 的包含语义也保留。跨读取边界的 marker fixture 已核验。

第 3 节三次新 512 MiB 运行均使用这个修正。更早 [block-fused-prepare](benchmarks/hf-bpe/results/block-fused-prepare/) 测量仍是旧脚本版本，记录原样保留；不能将那些失败资源数字与新测量混为一份结果。

## 8. 保留的语义与已有验证

重构应保留这些约束：

- 与 HF 的完整 merge trace、词表及 merges 一致，包含 tie ordering。
- 非 AA 批次的区间不重叠证书、首次激活 replacement、Selected 的唯一出生生产者关系。
- 读完全部旧语料后才改写，改写 join 后才提交及开始下一批。
- AA 的奇偶/连续段传播必须跨地址区域正确，不能简单独立处理每块。
- 左出生所属区域按 `p - 左 token 真实跨度` 计算，不能仅用前一个 block。
- 同一 pair 的位置按实际顺序拼接；临时逆序链填回后必须有序且不重复。
- 全局精确权重先汇总，再用频率门槛；局部零权重但物理存在的位置须保留。
- 有限长度、special/reserved ID、u16/u32 token 槽位、权重超 u32、普通/原子槽位等既有边界。
- 所有非空 affix 的首次激活检查和实际 ID 复用后的完整 HF cohort 重建；不能退回只支持少数 affix 字面值。
- 主/推测 posting allocation session 的释放完整性。

既有第一轮 block 源码 `d4723fc2` 完整 tk-train **86/86** 测试通过；`1959202f` 的偏移 Vec 合并另通过 full-trace 矩阵与直接跨块 fixture。日志见 [block 结果](benchmarks/hf-bpe/results/block-fused-prepare/)。这些只能证明那一轮语义，不能提前证明新设计。

affix 前置工作及证明见 [AFFIX_FAST_PATH_MIGRATIONS.md](benchmarks/hf-bpe/AFFIX_FAST_PATH_MIGRATIONS.md)、[PAIR_MONOTONICITY.md](benchmarks/hf-bpe/PAIR_MONOTONICITY.md)。旧 [BLOCK_FUSED_PREPARE_REPORT.md](benchmarks/hf-bpe/BLOCK_FUSED_PREPARE_REPORT.md) 记录第一轮取消 Plan 的结果；它早于本次强制 u32 single-block 与完整 512 MiB 对照，不是新目标的验收报告。

## 9. 下一轮应如何判断设计成立

1. 先列出每种数据结构按全局 pair 数、真实 segment 数、位置数增长的成本，并指出每个阶段消除了什么重复工作。
2. 审查初始化、选择、prepare、提交、退休的完整数据流，证明 single block 能自然退化到接近当前 flat 的工作量；没有这一步，不应先宣称已达到目标。
3. 若现有阶段计时不足以定位具体操作，有限采样当前 `u32-block` 的 route 与 initializer；验证真正的优化 ELF、DWARF、IP、调用栈后归因。语言分析的 profile 是公共 flat32，不能用它直接归因字典 route。
4. 实现一个可完整训练的统一候选，复用已有差分；补充实际改变的所有权、跨段顺序和地址上限检查。
5. 先以完整中文 512 MiB 的当前正常 flat32 **16.288 秒**为有效基准，测 unified 单段以及真实多段。必须固定输入、词表、权重、语料槽宽和线程数。
6. 主目标是大语料局部 **u32**；不要让 u16 小块实验替代它。可在相同完整输入上缩小真实地址区域形成多个 u32 段，同时明确这是地址区域尺寸控制，而非真的分配了超过 2^32 个槽位。
7. 单段应接近 flat 的完整时间、内存和并行工作量；多段不应重新出现按块复制 pair 哈希表、按块串行安装或不必要的全局排序。
8. 报告成功、失败、取消及资源门槛，不将 partial run 当成完成。保持与编译和其他 CPU 重任务错开。所有 metadata 容量不能直接相加声称同时峰值。

真正超过 2^32 槽位的完整性能范围尚未覆盖。当前机器的其他进程与内存压力也必须保留；不能为了制造成功结果终止用户服务。资源不足时，区分协议/地址语义验证与实际全规模性能验证。

### 常用执行入口

`tk-train` 被上层 workspace exclude；使用它自己的 manifest，避免 `cargo -p tk-train` 找不到包：

```bash
cd /root/code/tokenizers
/root/.cargo/bin/cargo test --manifest-path tokenizers/tk-train/Cargo.toml --lib
```

隔离构建和单个 native 运行分别使用 [build_affix_analysis.py](benchmarks/hf-bpe/build_affix_analysis.py) 与 [run_affix_analysis.py](benchmarks/hf-bpe/run_affix_analysis.py)。`--posting-block-bits 16` 当前会选择 u16 offset；`32` 的正常小语料会进入 flat。强制字典的 wrapper 仅是失败路径诊断工具，未来统一候选的正式测量应记录其真实表示及路径。

已有 `run_posting_width_comparison.py` 拒绝覆盖本次结果。复测必须使用新 case/result 目录和不可变 build label，并记录实际二进制 SHA。`.build/` 的 binary/source 存在于本机但被 git ignore；提交中的 source freeze 和结果中的 manifest/patch 是其可恢复来源。

## 10. 本次交接时的停止位置

- 源码尚未进行新的统一 posting 重构。
- 三次完整 512 MiB 布局测试已结束，当前代理没有运行中的编译或 native benchmark。
- posting 宽度对照的新脚本/结果及父 runner 缓冲区修正仍未提交；语言分析与本交接文档已单独归档，未推送。ROOT 生产源码仍是 `0e899a41` 对应的 `1959202f` 子树。
- 本文件用于让下一模型继续设计审查和实现；不能把候选的 owner 分段方案视为已经证明有效。

## 11. 独立语言差距分析：已交付并复核

完整报告：[LANGUAGE_TRAINING_GAP_ANALYSIS.md](benchmarks/hf-bpe/LANGUAGE_TRAINING_GAP_ANALYSIS.md)。最终数据：[训练摘要](benchmarks/hf-bpe/results/language-training-gap/final-none32.json)、[perf 分析](benchmarks/hf-bpe/results/language-training-gap/final-perf-analysis.json)、[原始验证记录](benchmarks/hf-bpe/results/language-training-gap/final-validation.json)、[完成状态](benchmarks/hf-bpe/results/language-training-gap/status.json)。

正式公共 release label 为 `language-gap-formal32`，binary SHA256 为 `3499c17aabf1fa09b2df5ceb0de019f0b15436116249e9461ddf286d18e80214`。源码 freeze 为 `1959202f`，与第 3 节正常 flat32 对照相同；这是公共 bits32/flat32，16 MiB 两案例未执行多 block fusion。构建证据另归档于 [正式 manifest](benchmarks/hf-bpe/results/language-training-gap/provenance/language-gap-formal32/build_manifest.json) 和 [诊断 manifest](benchmarks/hf-bpe/results/language-training-gap/provenance/language-gap-dwarf32/build_manifest.json)，同目录保存 instrumentation patch 与编译日志。

NONE、EN16/ZH16、merge/init 4/4、V30k/min2 条件下：

- 正式 EN train `2118.011 ms`，ZH `950.019 ms`，比值 **2.22944**。
- merge 墙钟差额 `1080.5232 ms`，占全训练差额 **92.5112%**。
- EN posting visits 为 ZH 的 **6.651 倍**，有效 rewrites 为 **4.896 倍**；AA 改写量很小。
- 真正观测到的热点包括端点校验、ledger 邻居聚合、birth 链与 posting 组装。全命令 instructions/cycles 及 cache MPKI 仅作相关证据，不把周期当墙钟或将 cache miss 比例当因果拆分。

总计 8 次 native 调用均完成模型、线程、资源和分配 balance 检查；最初 EN formal 与诊断编译重叠，已排除并用独立 EN 复测替换。为守预算省略 EN 未采样诊断 control，英文采样开销未独立隔离。每语料仅一次 census，stat/record 各一次，原始 perf 记录无 reported loss；未知 leaf period 为 EN `9.61%`、ZH `13.16%`。限制已写入报告。

本轮父复核仅检查小文件哈希、Python 语法、本地链接、既有 gate 及构建证据对应关系；未重新读取语料、哈希 binary、解析 perf 或运行 native。语言分析的采样是 flat32，**不能直接用来解释第 3 节单 block 字典路线的 route 成本**。
