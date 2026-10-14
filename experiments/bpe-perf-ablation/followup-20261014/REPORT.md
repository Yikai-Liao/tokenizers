# 历史复测与训练区间 perf：2026-10-14

当前版相对 fork main 的退化已复现。保留 buffer 和 Complete/Partial 快路径的历史版也存在显著差距，因此追加压缩和后来取消 Complete 发布不能解释全部退化。当前版的准备与提交合计解释约三分之二的训练墙钟增量；旧 buffer 版的差距更分散。

当前提交阶段已确认一个具体机制：忽略 whole-producer 完整性，令大量低频 birth 流和完整结果重复经过 fragment 路由、分组与归并。两项独立控制分别减少约 1.67 s 和 0.63 s 提交 wall。准备、初始化及其余阶段仍有未消除差距。

这是诊断报告。表中每个系列使用自己的 main 配对；阶段计时、分配计数和 perf 使用隔离源码，不能把它们混成一组正式性能样本。后续生产修订必须同时满足清晰的接口/结构体设计和实际性能收益。

## 条件与校验

- 本机六核 VPS，四 worker 固定 CPU 0–3；既有 English 单 worker 对照固定 CPU 0。
- 原中文文本 268435162 bytes，ByteLevel/Whitespace prepared words、100K vocab、min frequency 2、无 affix。输入哈希见 `input-manifest.json`。
- 训练区间仅覆盖 public `do_train`；输入加载、模型序列化和完整模型校验在区间外。历史提交整棵源码用 `git archive` 导出，不改生产 Rust 文件。
- 同一编译器、release opt-level 3、fat LTO、一个 codegen unit；共同依赖版本一致。正式历史复测不带 observer，每轮反转 arm 顺序。
- runner 比较完整 vocabulary IDs、special tokens 和有序 merges；所列历史/细分运行全部模型一致，普通计时 runner 检查 swap 为 0。
- main 是 `e4f787dc189d9be7192107490d652096cde7480e`，current 是 `36c819c5`；其余完整提交 SHA 在 source manifests。

## 不改源码的历史提交复测

中文 ByteLevel，四 worker，两个相反顺序的配对块。百分比先逐块相对该块 main 计算，再取中位数。

| 历史节点 | 训练 wall 中位数/s | CPU 中位数/s | 相对 main 的配对 wall 变化 |
|---|---:|---:|---:|
| fork main | 15.989 | 52.432 | +0.0% |
| 177302e2：Arena、窄坐标 builder、完整 producer | 22.499 | 70.715 | +40.8% |
| 1396e736：owned、TLS codec scratch | 21.366 | 66.906 | +33.7% |
| 02620e0a：u64 buffer、Complete/Partial | 23.040 | 70.463 | +44.2% |
| 36c819c5：当前版 | 24.577 | 76.036 | +53.7% |

从 Arena 版到 owned 版并未新增主要退化，本机这一系列反而变快。主要差距在较早简化实现里已经存在。另一组包含 `7495cb8b`（保留 selected-rule 快查表）、buffer、appendable-child 和 current 的复测也支持这一结论，见 `historical-summary.json`。早先 buffer/child 实验的近似持平，是在已经比 main 慢的局部基线上比较，不能推导它们与 main 持平。

## 增量分布

各列在自己的 main 配对系列中计算，单位是训练墙钟秒。准备和提交需合并比较：两套实现将位置编码安排在不同边界。

| 区段 | 旧 buffer 版增量 | 当前版增量 |
|---|---:|---:|
| 总训练 | +7.166 | +9.143 |
| 准备 | +2.297 | +2.706 |
| 提交 | +1.369 | +3.356 |

旧版约 7.17 s 的总增量中，准备约 2.30 s、提交约 1.37 s、初始化约 1.42 s、materialize 约 0.49 s、回收约 1.11 s、选择约 0.44 s；较快的 vocabulary/plan 抵消部分增量。约 1.35 s 的路由差距只有总增量的约五分之一，不能作为主要原因。

当前版提交再分为 root 路由、owner joined work 和 root 发布。wall 时间可依次相加；owner 内 CPU 是所有执行线程的累计，不能与 wall 相加。

| 当前版提交细分 | main wall/s | current wall/s | 增量/s |
|---|---:|---:|---:|
| 事件路由 | 0.248 | 1.517 | +1.268 |
| owner joined work | 1.659 | 3.339 | +1.681 |
| root 发布 | 0.004 | 0.404 | +0.400 |

其中 owning BirthFragment 的 comparison sort 约 2.11 s 是所有 owner 的累计 CPU，不是额外 2.11 s wall。只换分桶的现有原型未恢复提交差距；不能把这一符号的样本或 CPU 直接当成全部提交退化。

准备任务的累计 CPU：main 的 scan 约 19.21 s、finish 约 5.19 s；current 的 scan 约 32.34 s、setup 约 1.52 s、finish 约 1.05 s。编码边界变化使 scan/finish 分别不可直接对齐；合计仍多约 10.5 s CPU。旧 buffer 的 setup+scan+finish 约 34.9 s CPU，说明这项准备成本早已存在。

## perf 和工作量证据

`perf record -F 199 -e cycles:u --call-graph dwarf,16384` 通过 control/ack FIFO 在输入加载后启用、训练完成后关闭。分别采样 main/current 和 main/旧 buffer，各两组反向顺序运行，全部完整模型一致。perf 运行时间与细分计时分别保存，没有用单次样本百分比相减推导因果比例。

`perf-source-summary.json` 和 `perf-history-source-summary.json` 使用每个 sampled instruction 的 LLVM inline source stack，按最近的项目源码行独占归属。完整原始 DWARF perf.data、LLVM annotate 和 source-attribution 在本机独立诊断目录，Git 中保留紧凑 summaries 和符号报告。

较旧 buffer 的热点覆盖 endpoint loads、initial pair 的 SmallVec push、neighbor query/record、position encode、count ledger 和回收。current 另有 streaming Positions push、owning birth fragment 排序和路由。符号不一一对应，因此 source/sample 占比支持定位操作，不能单独证明这些操作各自贡献多少 wall。

分配计数在独立运行记录，包含 realloc 请求的新容量；requested bytes 是累计容量请求，不是 RSS 或实际写入字节。

| prepare 分配工作 | main | 旧 buffer | current |
|---|---:|---:|---:|
| alloc calls | 1472213 | 9920642 | 8197801 |
| realloc calls | 4046469 | 18796112 | 11257232 |
| 累计 requested bytes | 12539877870 | 46131610944 | 43188729427 |

这证明更多分配/增长请求已经发生，但尚不能把次数折算成耗时。短流分布、被 floor 丢弃的 births/positions 和任务细分计数保存在 `fine-results-raw.json`。

## 单变量反证与正向证据

- 纯 NeighborScratch 生命周期消融，在当前 pool 行为不变时，English 四 worker 准备约 0.952→0.617 s；Chinese 一对约 11.498→10.491 s。反复目录初始化是已证实的额外工作，但不是整个 Chinese 差距。详情在上一级 README 与 local-diagnostics.json。
- 仅替换 fresh birth grouping 的原型未恢复提交退化，且 Chinese 变慢。
- 外部八 worker 的 dense selected-rule 与 U24 slot 消融分别只有约 +1.3% 提速和约 1% 变慢，作为外部提示记录；不把这些数字作为本机已验证结果。
- 本机拆掉 main 的小规则任务合并，保留编码、producer 完整性及发布机制，准备 wall 约 8.065→8.674 s、训练约 16.045→16.805 s；它有贡献，但无法解释大头。

完整 producer 临时存储控制，仍保留 main 的位置编码、floor 和发布：

| 控制 | 训练 wall/s | 准备 wall/s | 准备任务 CPU/s |
|---|---:|---:|---:|
| main 默认形状反馈 | 15.637 | 7.755 | 24.886 |
| 完整 producer 始终独立连续 Vec | 15.739 | 7.974 | 25.554 |
| 完整 producer 始终共享链式存储 | 18.530 | 10.673 | 34.870 |

## 分配器计时与编码控制

另用 1/128 分配请求采样，在 native System alloc/realloc/dealloc 前后读取线程 CPU 时钟。每个样本紧接着测一对空时钟调用，逐线程/阶段/操作扣除实时基线。固定启动校准的第一版偏差明显，未用于结论；第二版原始记录见 `allocation-cpu-v2-results.json`。

第二版 prepare 分配器内部 CPU 估计：main 两轮约 7.87/7.87 s，旧 buffer 约 15.03/18.13 s。该估计仍受采样长尾、时钟与调用包装开销影响，不能当作已经通过修改收回的生产 CPU。独立 perf 的全训练 libc 用户态 cycles 占比约 main 7.96%/8.77%、旧版 16.12%/16.55%；包含分配器和内存复制，且大部分 native 调用栈未恢复，不能据此划分 BPE 阶段或换算 wall。

对旧版单独给分配点加标签，测到邻居原始位置缓冲约 459 万 alloc、643 万 realloc；冻结位置编码约 456 万 alloc、1044 万 realloc。标签 observer 的结果只用于定位，运行时间不与普通 runner 混合。

再做不带 allocator observer 的编码控制：

| 独立系列 | 原版 wall/s | 控制 wall/s | 原版 CPU/s | 控制 CPU/s |
|---|---:|---:|---:|---:|
| 保留 Vec，不转 Box | 22.334 | 22.189 | 70.026 | 69.047 |
| 先测精确字节数，再一次预留 | 22.564 | 21.796 | 70.097 | 67.959 |

Vec→Box 在 capacity 大于 len 时先收缩，可能请求 realloc；这项机制不能推出当前 workload 中收益很大。取消转换也会增加对象和驻留容量，因此该控制包含其实际布局/内存代价。精确预留控制增加一次测宽扫描。上述两项全模型均一致，但都未收回 main 的主要差距。

另外只给旧版 Neighbors::group/record 与 FreshSnapshot::record_neighbors 加 inline(always)，训练中位数约 22.93→22.19 s；不是大头，CPU 的两轮改善也不一致。完整 producer 强制 Vec 近似持平、强制链式变慢，不能据此把“恢复链或池”当成关键修复。

## whole-producer 过滤与提交消融

当前版在 owner 中丢弃约 1339 万条未达 floor 的 birth 流，全部只有一个位置。main 的 complete producer 能在编码/路由前根据完整质量过滤。单纯改变分组算法仍处理这些流，不能消除这部分工作。

独立 current 控制保留现有位置表示、线程池和 scratch：真实 chunk iterator 只有一块时，将其标记为 WholeRule；AA、reuse、分片及可能由两方向合并的 (id,id) 留在原路径。Index 在知道完整频次后提前过滤，移除动作仍保留。类型携带实际 producer coverage，没有新增全局有效位或镜像索引。

| current 配对控制 | 训练 wall/s | 准备 wall/s | 提交 wall/s | route wall/s | owner join wall/s |
|---|---:|---:|---:|---:|---:|
| current | 25.230 | 10.712 | 5.047 | 1.497 | 3.215 |
| owner 的相同取模改为 power-of-two mask | 23.972 | 10.153 | 4.386 | 1.143 | 2.945 |
| whole producer 提前过滤 | 23.015 | 10.172 | 3.373 | 0.909 | 2.154 |

提前过滤将进入 BirthFragment 的流从 2117 万条减至 789 万条，owner sort 累计 CPU 从约 2.1 s 降至约 0.75 s；仍由 owner 丢弃的低频流约 10.8 万条。两轮完整模型一致。准备阶段尚未改变，也没有由此消除全部退化。

相同取模控制不改变任何 owner 映射：2 的幂用 hash & (n-1)，其他 worker 数仍用原取模。route 在该系列由约 1.50 s 降至约 1.14 s；总训练差额还混有其他阶段波动，不全部归因于这行算术。

再在提前过滤控制上，对已达 floor 的 whole-producer 直接发布 count 和冻结 positions；部分流继续原分组路径。root owning queue 仍保持原实现。该系列单独配对：

| 对照 | 训练 wall/s | 提交 wall/s | route wall/s | owner join wall/s | owner sort CPU/s |
|---|---:|---:|---:|---:|---:|
| 提前过滤 | 22.389 | 3.333 | 0.913 | 2.122 | 0.739 |
| 提前过滤并直接发布完整 producer | 21.735 | 2.705 | 0.928 | 1.473 | 0.028 |

不带 phase、fine 或 allocator observer 的独立复测：

| 实现 | 训练 wall/s | 训练 CPU/s |
|---|---:|---:|
| fork main | 15.607 | 51.584 |
| current | 23.986 | 74.662 |
| 仅保留 whole-producer 并提前过滤/直接发布 | 22.101 | 70.405 |

该原型的现有 32 项 BPE 单元/串行 oracle 测试全部通过，覆盖随机 min frequency、重启、overflow、AA、reuse、affix 和 1/4/8 worker 语义对照。裸 runner 依赖仍用冻结 lock；测试图添加 dev-dependencies，保留所有 runner 原有包/版本。该复测仍仅是中文 ByteLevel、100K、四 worker、本机两个配对块。

## 整次训练只入池一次

在不带 observer 的完整 producer 候选上，入口将整次训练（含 Fresh/Reuse 重试）放到既有 Rayon pool 的 worker 内。已在 pool 中的调用和串行设置直接执行；不创建专用线程池，也不修改每轮算法。该系列固定二进制并反转执行顺序，共两个配对块。

| 案例 / worker | main wall/s | current wall/s | producer + 一次入池 wall/s |
|---|---:|---:|---:|
| Chinese ByteLevel / 4 | 16.220 | 24.293 | 21.647 |
| English ByteLevel / 1 | 1.937 | 3.231 | 2.677 |
| English ByteLevel / 4 | 1.289 | 2.429 | 2.199 |

同系列 Chinese 四 worker 中，完整 producer 单独为 22.141 s，叠加一次入池为 21.647 s。原 current 为 24.293 s、main 为 16.220 s。所有完整模型一致、swap 为 0；该候选仍未恢复 main 的性能。


上述源码 patch 是隔离的因果实验，尚不是生产修订。全模型对照与适用范围之外的语义测试须一并看；阶段计时 observer 也不能替代正式无 observer 复测。后续不应继续堆积独立 fast-path 状态，而应让 producer 完整性、最终表示和 index 的完整提交操作各自有明确归属。


## mimalloc 直接对照

同一二进制、仅替换 native allocator 的 12 次训练结果见 [mimalloc 对照](mimalloc/README.md)。旧 buffer 版配对 wall 下降约 12.8%，current 约 5.1%，main 约 5.9%；current 与 main 的绝对差距基本未缩小。

按用户要求另做了 main Arena 消融：只移除 Arena 后端，保留 system/mimalloc 两种 allocator 对照。system allocator 下训练中位数增加 2.41 s，mimalloc 下无可测差异；退役阶段从 0.07 s 增至 1.19 s。完整方法、补丁和阶段数据见 [main Arena 消融](main-arena/README.md)。

## 后续修订的设计条件

目前对 current 的建议是进一步减少已量出的实际工作，不能将诊断控制、TLS 原型或旧 main 的机制直接叠进生产实现。

每个修订须明确临时与冻结表示的边界、状态权威、移交和最后使用时间；接口负责完整准备/提交操作，调用者不用维护额外有效位和并行镜像。需要复用的资源优先由一次训练的明确生命周期管理。替换实现时，同轮清理失效路径和状态，并同时验证模型语义、性能和结构质量。以上是用户要求的验收条件。

## 数据与复现

原始历史 records、细分 observer、分配记录、perf summaries 和输入/源码 manifests 在本目录。每个 source patch 都标明 control→variant；先导出指定历史提交，再在隔离树应用 observer，不能在分支 tip 混用 main/current patches。

- main `e4f787dc` → `patches/main-to-main-detail.patch`。
- current `36c819c5` → `patches/current-to-current-detail.patch`。
- buffer `02620e0a` → `patches/history-buffer-to-history-buffer-detail.patch`。
- main-detail 之后单独应用 single-job、vectors 或 chains patch，各自是独立控制，不累积。

训练 runner 和 control/ack gate 见 `reproduction/`；执行前按目标隔离树调整 Cargo.toml 的依赖路径，保持 profile 与 lock 的共同依赖版本，显式 CPU affinity 和实际 worker。普通计时与 perf gate 运行分别构建；perf gate 需要父进程先创建并以 O_RDWR 打开 control/ack FIFO，然后启用 perf --delay=-1 --control=fifo:...。完整模型校验必须留在训练区间外。
