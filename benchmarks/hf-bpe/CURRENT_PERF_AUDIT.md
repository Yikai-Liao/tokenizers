# 当前候选版本的 PERF 审计

日期：2026-10-01。本报告由 GPT-6 Luna 独立审计候选分支 `bpe/initial-owner-waves`，源码固定为 `029ab45bd0b0f446035f71acbd247c804bea60b8`，采样前工作树干净。采样与摘要保存在 [PERF 归档目录](results/current-perf-audit)。原始 `perf.data` 为 296,943,378 字节，约 283 MiB，保留在 Git 忽略的 `.build` 目录；SHA-256 为 `1b8ec30aa074de3b238929477faedffaec7c679ed2ab0b7a7c936441857beaa3`。本次审计没有修改候选或生产源码。

## 1. 当前成绩与结论

**已经实现约 19 秒的训练，后续候选还把峰值内存从约 4.43 GiB 降到了约 3.33–3.46 GiB。** 前次摘要引用的 25.577 s / 29.838 s 是较早的 DE→H 历史成绩，不是当前候选回退后的成绩。为避免混淆，下表把版本、分配策略和计时范围一起列出。

以下均为同一份 512 MiB 中文语料、`none`、目标词表 50,000、最低频率 2、初始化与合并各四个 worker；完整模型与工作量核验通过。训练 API 耗时包含初始化、合并及收尾；端到端另包含输入处理。

| 版本与实测配置 | 训练 API 耗时 | 输入处理 + 训练 | 峰值内存 | 样本口径 |
|---|---:|---:|---:|---|
| 历史 H，标准分配器 | 25.577 s | 29.838 s | 约 4.43 GiB | 两对 DE→H 的中位数 |
| J，标准分配器 | 24.351 s | 28.957 s | 4.434 GiB | 与下一行同场对照，各一次 |
| J，全量 posting arena | **19.126 s** | **23.105 s** | 4.435 GiB | 已实现并实测的 arena 候选 |
| 后续低峰值候选，直接路由、普通 heap、T256 | 17.893 s | 21.889 s | 3.377 GiB | 一次完整调用 |
| 后续低峰值候选，直接路由、压缩 heap、T256 | 18.105 / 20.954 s | 22.003 / 25.067 s | 3.359 / 3.333 GiB | 两次调用，按相同顺序列出 |
| 后续低峰值候选，直接路由、压缩 heap、全量 arena | 16.470 / 20.067 s | 20.306 / 24.409 s | 3.439 / 3.458 GiB | 两次调用，按相同顺序列出 |

T256 是固定阈值：容量不超过 256 字节的 posting 载荷使用 arena，更大的载荷使用普通分配器；256 的单位是字节，不是 token 数。全量 arena 则将全部堆分配 posting 载荷放入 arena。压缩 heap 是候选队列的条件表示，与已经撤回的 8B 初始化记录方案不同。表中配置保留在候选实现和实验归档中，生产 J 尚未迁入这些后续改动。**Arena 策略尚未完成最终选型，也没有实现按语料动态计算阈值。** 本次 PERF 的当前候选使用默认系统分配器，未启用 Arena。

19.126 s 是训练耗时，对应端到端 23.105 s；25.577 s 和 29.838 s 则是历史 H 的两种计时范围。因此不存在用训练从 19 秒回退到 25 秒来换取内存的结论。低峰值候选已经测到接近或快于此前 19 秒的训练，同时峰值内存降低约 22–25%。两次复测有明显主机波动，具体成绩按表中实测值保留，不把 16.470 s 宣布为稳定耗时。

早期 DE→H 两对对照的训练中位数为 32.397→25.577 s，减少 21.1%；端到端为 36.766→29.838 s，减少 18.8%。随后 H→J 的筛选对照、J 的 arena 对照，以及后续初始化优化继续推进。历史与 arena 证据见 [优化报告](OPTIMIZATION_REPORT.md) 和 [全量 arena 对照](results/j-bump-retain.summary.md)，低峰值候选见 [初始化峰值报告](INITIALIZATION_MEMORY_REPORT.md)。

本次 PERF 使用当前候选的标准分配器诊断版，用于检查剩余热点；它没有启用此前取得 19 秒成绩的 arena 策略。其带采样耗时不参与上述成绩排名。当前主要成本是 posting 校验、邻边频率变化的构造，以及 owner 更新；尚未找到占比高、可以明确删除的重复工作，**因此停止这一轮优化**。

## 2. 工作负载与测量方法

通过原 Trainer API 完成一次完整训练，使用固定的真实 512 MiB 中文 Wikipedia 输入：536,870,289 字节，SHA-256 为 `a0d40d4102cba1e933f25e5ccd17552d2eaebaec4ef770bc39d8e46e740b4d4f`。配置为 `none` 切分、reference 前端、目标词表 50,000、最低频率 2、原子 u32 语料、初始化与合并各四个 worker。完整模型摘要与标准结果一致：`d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd`；最终词表 50,000，合并规则 29,243 条，去重片段 1,429,915 个。

候选源码从干净工作树复制到独立诊断构建目录，Cargo target 也使用独立目录。优化后的 release binary 保留 DWARF level 2 和符号，ELF 含 `.debug_info`、`.debug_line`、`.debug_abbrev` 和 `.debug_str`。Binary SHA-256 为 `45d0147acc733cf9d66e593916495ff164aec115f5de68bdbcf9d35107059b52`，Build ID 为 `7faa1d5a8f37cd203ba61fb94200d5ba80ebfa42`。运行程序调用原 Trainer API，使用 Rust 默认系统分配器，仅增加已有内部统计的输出和 worker 设置。构建细节及源码 hash 见 [build.json](results/current-perf-audit/build.json)，运行参数与资源监控见 [current.provenance.json](results/current-perf-audit/current.provenance.json)。

与历史 H 运行的工作量一致性检查通过：去重片段 `1,429,915`、初始符号 `204,660,029`、语料槽位 `206,089,945`、物理边 `203,230,114`、pair key `2,697,517`、posting 访问 `125,409,599`、剪枝 pair `21,518,771`，以及完整模型 SHA-256 均相同。另一个库存统计中的 `stored_positions=207,224,101` 采用不同口径，不参与这里的工作量对照。

本次以 99 Hz 同时采集 `cycles:u` 和通用 `cache-misses:u`，调用栈配置为 `--call-graph dwarf,16384`。周期样本 8,719 个、缓存未命中样本 8,447 个，共 17,166 个，丢样为零。按样本 period 加权后，事件量分别为 221,481,049,753 个周期和 1,376,613,550 次缓存未命中。未解析的采样指令地址分别占加权事件量的 9.66% 和 12.96%。以下百分比均为采样指令地址对应的事件份额，不是墙钟耗时占比。通用缓存事件无法区分缓存层级，采样地址也可能偏离实际触发事件的指令，因此不能据此认定某条 load 导致了这些未命中。

资源始终高于停止门槛：最低 `MemAvailable` 为 4.28 GiB；目标进程采样 `VmRSS` 峰值为 3.24 GiB，`VmHWM` 为 3.38 GiB；采样 `VmSwap` 为 0；系统 `pswpin`、`pswpout` 增量均为零。诊断运行的训练耗时为 31.294 s，从输入处理开始到训练返回为 35.923 s。这些时间包含诊断构建与采样开销，不纳入正式速度排名。

## 3. 实际执行分支与阶段成本

本次选择 `parallel_u32_flat32`，初始化只有一个 block，计数后端为 `stable_radix16`。完整 key 的通用自适应分支没有执行：`initial_bounded_tiles=0`、`initial_bounded_groups=0`。因此，这份 PERF 覆盖当前 512 MiB 的 flat 主路径，不用于判断超过 `2^32` 位置或通用分块回退路径的性能。

| 测量范围 | 耗时 | 口径 |
|---|---:|---|
| 输入处理（feed） | 4.630 s | 包括读取输入和片段频率准备 |
| 初始化 | 7.028 s | `initial_count_ms` 为 4.271 s；radix 排序 1.437 s、posting 安装 2.048 s、路由 1.357 s。这些统计有嵌套或重叠，不能直接相加 |
| 合并 | 19.442 s | 融合 prepare/delta 为 8.945 s，owner 提交 8.879 s，语料改写 0.867 s，路由 0.021 s，候选选择 0.269 s。`delta_ms` 已包含融合 prepare 区间，不重复累加 |
| 完整训练 | 31.294 s | 初始化与合并没有覆盖全部 Trainer 准备和收尾；剩余 4.824 s 无法由这些阶段统计单独归因 |

本次带采样的训练耗时为 31.294 s，历史 H 诊断为 28.569 s，观察差额为 2.725 s，约 9.5%。这是两次诊断运行的记录；采样开销、代码版本与统计探针均有影响，不能把差额直接归因于候选算法，也不改变前述正式对照已经证明的收益。

当前主要热点如下。这里的直接采样占比（self）只计采样指令本身落在该函数内的事件，不含被调用函数。

| 热点 | 周期事件占比 | 缓存未命中事件占比 |
|---|---:|---:|
| `train_in_pool` 的 owner 提交闭包 | 23.35% | 19.20% |
| `fused_batch::prepare_with_mode` 闭包 | 22.13% | 17.20% |
| 稳定 radix 排序 | 5.24% | 9.85% |
| `HashMap<u64, Entry>::insert` | 4.75% | 2.77% |
| `cfree` | 3.37% | 1.59% |
| `Prepared::apply` | 2.65% | 7.03% |

完整热点摘要见 [event-summary.json](results/current-perf-audit/event-summary.json)。

使用本次实际 binary 的 DWARF，对少量高频采样地址做了源码核验。Owner 提交闭包的热点分别落在输出 Group 遍历（`parallel.rs:1215`）、`ledger.entries.get_mut` 与 hashbrown 的 `find_inner` 探测（`parallel.rs:1223`），以及频率下限检查（`parallel.rs:1228`）。Prepare 闭包遍历候选 posting、校验相邻 token、读取权重并生成精确的 remove/birth 频率变化（`fused_batch.rs:178–258`）。语料写入由另一个 `Prepared::apply` 路径执行，包括 `fused_batch.rs:300` 的替换写入。这样可以区分查询、频率变化构造和语料改写，但不能把整个闭包的采样量都归给其中某一步。地址与源码核验保存在 [top-ip-addresses.txt](results/current-perf-audit/top-ip-addresses.txt)。

完整合并过程访问了 125,409,599 个 posting，本次过期 posting 访问数为零。Prepare 和 owner 提交为这些位置执行边界校验、邻边频率统计及 owner 账本更新。目前没有找到能在保持 BPE 结果的前提下删除的大规模重复扫描或冗余更新。

## 4. 与上次完整 PERF 的比较

上次保存的完整 512 MiB DWARF PERF 来自 H，源码为 `00216d914186e42458d45e72276b13c700749c6a`，不是 J 或当前候选。它使用相同输入与 Trainer 参数、优化后的 release 构建、DWARF level 2、相同的 `cycles:u` 和 `cache-misses:u`、99 Hz 采样及 DWARF 调用栈。其 binary SHA-256 为 `b712704589be5228a7ba1295d39f989993ac1d987b8f96197f9504c30b701d34`，Build ID 为 `e80ac816f0da8a7b8019205f97b665ba08d13045`。详见 [H 的历史采样报告](results/optimization-h-debug-profile.md) 和原始运行元数据。

下表斜杠两侧依次为周期、缓存未命中事件占比。包含被调用函数的采样占比标为 inclusive，其口径与直接采样 self 不同。

| PERF 采样类别 | 历史 H | 当前候选 |
|---|---|---|
| 周期 / 缓存未命中样本数 | 8,084 / 7,813 | 8,719 / 8,447 |
| 未解析指令地址的事件份额 | 9.55% / 9.63% | 9.66% / 12.96% |
| `Output::birth` 直接采样 | 5.35% / 1.69% | 已不在主要热点中；当前任务内聚合 birth 为 2.47% / 2.12% |
| `Output::remove` 直接采样 | 5.46% / 3.25% | 1.53% / 0.80% |
| Owner 提交闭包 | 历史闭包的直接周期采样合计 18.94% | 23.35% / 19.20% |
| Prepare 闭包 | 包含被调用函数的占比为 34.08% / 24.30% | 直接采样为 22.13% / 17.20%；包含被调用函数的周期占比为 32.36% |
| Hash 表扩容 | `RawTable` 直接采样合计 3.40% / 7.41%，主要是 `reserve_rehash` | 全部已解析 `reserve_rehash` 指令地址合计 2.55% / 7.49% |

这些行对应的源码范围并不完全相同。H 的 `Output::birth/remove` 按 occurrence 处理；当前源码先在任务局部 `Scratch` 中聚合邻边变化，再按 owner 提交路由表。H 的 inclusive prepare 也不能直接与当前 self prepare 相减。采样确认了旧 birth/remove 热点已经被结构性改写，成本分布转移到 prepare、聚合与 owner 提交；具体收益仍需正式阶段和端到端计时衡量。

当前 `reserve_rehash` 汇总覆盖完整样本流中的全部已解析符号，不只取前 25 个热点。主要项为：`BirthGroup` 1.54% / 4.62%，`(u64, u32)` 0.70% / 2.33%，`Entry` 0.24% / 0.45%，`CompactString` 0.07% / 0.09%。这些事件份额既不是扩容次数，也不能独立估算预留容量能节省多少时间。完整符号汇总保存在 [event-summary.json](results/current-perf-audit/event-summary.json)。

J 的分组改动已有独立的正式端到端证据：原 API 的 H→J 512 MiB 筛选中，prepare 减少 8.1%，完整训练减少 11.4%，端到端减少 9.6%。同一次对照中，未修改的初始化也快了 2.011 s，因此完整收益不能全部归因于分组改动。这些数据支持保留 J。本次 PERF 用于识别剩余热点，不额外计算一个新的加速比；两次采样的加权事件量也不能当作绝对工作量对照。

## 5. 剩余机会与停止决定

当前最大的成本是精确 prepare 循环，以及真实 posting 对应的 owner 聚合和账本更新。这些路径已经实施了任务内邻边变化聚合、减少 owner 路由查询、直接填充最终初始化流，以及低 scratch 稳定排序等结构性优化。

剩余热点中，尚未找到占比高、可以用简单改动明确删除的冗余操作。扩容采样分散在不同表中，`Entry` 表扩容只占约 0.24% 的周期事件；`BirthGroup` 的缓存事件占比不能直接解释为可回收的缓存延迟。增加预留容量会增加内存，目前缺乏能带来明显完整调用收益的证据。单独的 `cfree` 热点也无法说明哪些分配可以省掉，或能节省多少墙钟时间。

**对已经测量的 512 MiB flat 工作负载，停止继续寻找新的内存内算法。** 已有端到端收益明确，当前证据没有支持新的高收益算法候选。Arena 策略仍是尚未完成的选型事项：中文 512 MiB 的低峰值候选中，全量 Arena 两次比 T256 快，峰值略高；其他语料的速度与峰值表现不同，现有实验没有选出适用于所有语料的默认阈值。完整 key 的通用 `>2^32` 自适应路径不在本次采样范围内，其性能仍需按单独的代表性工作负载评估。

## 6. 归档与复现

- [构建脚本](results/current-perf-audit/build_current_perf.py)：在独立构建与 target 目录生成优化后的 DWARF 诊断 binary，记录修改后的源码和运行程序输入 hash，保存 [诊断补丁](results/current-perf-audit/instrumentation.patch)。
- [运行脚本](results/current-perf-audit/run_profiled_call.py)：只执行一次完整诊断调用，保存运行来源和结果；`MemAvailable <= 1 GiB` 时停止。
- [汇总脚本](results/current-perf-audit/summarize_perf.py)：从原始 PERF 文件重建按 period 加权的指令地址摘要，无需展开每条完整调用栈。
- [原始 PERF 头信息](results/current-perf-audit/perf-header.txt)、[采样输出](results/current-perf-audit/perf-record.stderr)、[训练结果](results/current-perf-audit/current.jsonl) 和 [DWARF 地址映射](results/current-perf-audit/top-ip-addresses.txt)：保留诊断细节。
- 原始 PERF 数据：`.build/native-j-current-perf.perf`，SHA-256 记在 `current.provenance.json`；大型二进制文件未提交到 Git。

在新的审计 checkout 中复现时，将同一 hash 的语料放到记录路径，依次运行 `build_current_perf.py`、`run_profiled_call.py` 和 `summarize_perf.py`。脚本会拒绝覆盖已有输出。
