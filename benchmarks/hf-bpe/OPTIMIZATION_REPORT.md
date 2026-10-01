# BPE 热点优化与完整组合选型

## 当前选择：J规则邻居聚合

当前选择 **J `376363d2`**（H+Task内左右neighbor目录聚合），worktree `/root/code/tokenizers-worktrees/prepare-rule-aggregate`，branch `bpe/prepare-rule-aggregate`。H带debug采样确认route桶探测，独立计数得到birth/remove各173.16M次；J把同Task、同侧、同neighbor的weight更新聚合后flush，保留原owner路由与Node链。

原48tests及[独立静态审查](RULE_AGGREGATE_REVIEW.md)通过；一次H→J原API512MiB关键对照的直接prepare与完整操作同向改善，按预算不追加多组重复。

| 单次同期对照 | H | J |
|---|---:|---:|
| prepare（含scratch初始化、group与flush） | 8.400 s | 7.723 s |
| 完整初始化 | 7.467 s | 5.456 s |
| merge | 15.726 s | 15.044 s |
| 完整train | 27.790 s | 24.613 s |
| feed+train elapsed | 31.762 s | 28.701 s |

prepare少8.1%，train少11.4%，elapsed少9.6%。**未改初始化也快2.011秒，不能把全训3.177秒差额都归因于J。** 这是n=1筛选结果，未证明稳定复现；H的既有两对基线保留。新增scratch capacity汇总峰2,106,304B（约2.01MiB），不是实测同时驻留峰值。RSS均约4.43GiB，VmSwap0；完整模型与N/E/pairs及corpus/posting容量一致，详见[J筛选](results/optimization-rule-aggregate.summary.md)。

K `288ad858`仅并行释放owner，47tests通过；两对清理区间平均少32.1%，但完整train/elapsed未建立收益，停止，不加入J。诊断定位到owner销毁占post-merge缺口99.48%，说明大项存在，也说明单项优化仍须兑现完整结果，见[成本定位](results/optimization-h-cost-probe.md)与[K两对结果](results/optimization-owner-parallel-drop.aggregate-n2.md)。

### 用户追加测量

- [J单线程/四线程](results/rule-aggregate-scaling.summary.md)：512MiB同binary各一次，初始化3.43×、merge3.45×、train2.99×、feed+train2.70×；模型/工作量一致。feed由benchmark显式关闭并行，生产实现已有并行路径。
- [J posting分配统计](results/j-posting-inventory.summary.md)：terminal独立堆posting7,274,631个，61.13%容量≤8个u32，inline3,297,497个。诊断扫描191ms，不参与排名；此前H精确计时将大额收尾归于owner drop，词表/merges构造约21ms。
- [原efficient_bpe Rust粗略对照](results/efficient-rough.summary.md)：同约16MiB文本前缀、逐行权重、四线程、40,759规则，J train1.478s，原Rust ebpe call2.155s，约1.46×。各一次，无模型一致性检查；输入API与release profile差异明示，不能外推512MiB。

### 停止依据与剩余成本

最后复用已有完整H采样，按真实owner commit符号的IP核对源码和反汇编，未新增训练。主要热点是旧pair ledger probe和frequency载荷更新、出生ledger probe、posting长度读取与Node链逆序填充。采样属于H，J改变prepare后模块份额可能变化；具体地址的cycles/cache份额不等于精确load或整数减法成本，详见[owner具体操作](results/optimization-h-debug-owner-commit.md)。

I减少出生ledger查询的改法回退，K并行释放未兑现完整收益，J减少route重复查询后尚未找到代价清楚、能大幅减少剩余工作的局部方案。本轮停在J；大幅替换语料布局、allocator或owner结构仍是未验证研究方向。本次停止不表示全局最优。

## 此前H完整组合选型：权重快路径

H **`00216d91`**（DE+权重1空间桶认证）是J的已验证基点。每256位置只需一个bit，整桶权重为1时省掉pivot搜索；混合桶保留原精确搜索。在512MiB中文none/50k/min2、u32、初始化/merge各4线程的原Trainer接口下，两个交错pair的直接模块及完整训练都改善。用户要求收益明显时停止重复测试，因此完成n=2后取消第三对。

| 两对中位 | DE | H |
|---|---:|---:|
| 初始分组统计 | 3.268 s | 0.327 s |
| prepare（已含在delta） | 12.075 s | 8.170 s |
| 完整初始化 | 8.585 s | 5.580 s |
| owner commit | 5.986 s | 5.883 s |
| 完整train | 32.397 s | 25.577 s |
| feed+train elapsed | 36.766 s | 29.838 s |

H/DE配对train比0.821/0.757，elapsed比0.834/0.789；两个直接模块和完整训练均同方向改善。完整模型签名、N/E/pairs、语料和posting容量一致；所有进程swap采样0，RSS约4.43GiB。统计为n=2描述值，范围与sampleCV保留在 [H对照](results/optimization-weight-stability.summary.md)，初次screen单列。

**内存增加100,632B，约98KiB**，并非降低内存。该工作集约99.1%的空间桶全部权重1；这是空间覆盖，未测实际查询命中率。WeightLookup总容量由3,220,160增到3,320,792B，初始化/merge复用同一分配，bytes不能重复相加。全47tests与 [独立审查](WEIGHT_ONE_BUCKET_REVIEW.md) 通过；无显式SIMD。

I `e3a1954c` 全48tests与审查通过，但一次H→I的commit5.971→6.876秒、train26.187→28.538秒，停止推进，见 [I筛选](results/optimization-commit-screen.summary.md)。此后按用户要求补齐H实际binary debug信息、源码采样与动态计数后，再验证J，详见[debug采样](results/optimization-h-debug-profile.md)。

## 此前DE完整组合选型

此前推荐 **DE `c8702374`**：基础endpoint/owner/posting/batch/pool，C直接语料构造，B2融合与查询目录，D初始radix，以及E批量posting安装/commit。512MiB中文none/50k/min2、u32、4线程原接口。B2/D/DE三个循环块复测，DE每块均胜B2；train中位31.332秒、elapsed35.493秒，配对中位比例0.8715/0.8840，sample CV6.91%/6.10%。详细运行见 [三组合报告](results/optimization-radix-combination-stability.summary.md)。

B2和D的train中位35.952/35.442秒；D完整初始化11.058→9.147秒有明确改善，端到端逐组0.8501/1.0065/1.0002并未稳定胜B2。DE初始posting安装中位0.624秒（D0.931）；commit跨组会换号，不能把整个train差值都归因bulk。归因的限制不妨碍根据完整数据推进DE。

随后G2的自动向量化尝试：与DE三对的prepare比中位1.0195、elapsed比中位1.0018，未有稳定收益，停止推进。F单例出生的两个直接模块也在screen回退，停止推进；未把未改init变快算作F/G的收益。随后由DE热点采样定位权重查询并实施H，自动SIMD只作为可能手段，不按是否出现向量指令选方案。见 [自动向量化检查](AUTO_VECTORIZATION.md)。

### 相比建立worktree制度之前

之前同512MiB/4线程的u16 corpus、u32 posting train56.430秒，当前DE三次中位31.332秒：历史观测少44.5%、快1.80倍；旧RSS3.005GiB、现在约4.43GiB。输入与模型签名一致，但ID宽度和测量时间改变，因此不是严格同期隔离加速比。统一u32后最早count4 train54.760秒，相比DE观测少42.8%、快1.75倍。

### DE历史热点（上述三组合样本中位）

| DE阶段 | 秒 |
|---|---:|
| prepare（包含在delta中） | 11.880 |
| owner commit | 5.955 |
| 初始分组/权重频率 | 2.907 |
| 初始radix排序 | 1.902 |
| 初始route（含compact） | 1.478 |
| corpus构造 | 1.110 |
| 初始posting安装 | 0.624 |

prepare与commit仍占merge大头。DE/G2同期的DE prepare中位11.330秒，绝对值有波动；优先优化每有效posting上的查询/索引管理。全部source与组合依赖在 [OPTIMIZATION_CATALOG.md](OPTIMIZATION_CATALOG.md)。

## 历史C/B2选型

C/B2三对train中位42.887→34.942秒、elapsed47.701→38.965秒，B2每对胜出，成为后续D/E的parent。共同pre-write定义为C plan+delta、B2 plan+delta+目录构建，中位18.928→12.124秒，fused_prepare不再与delta相加。见 [C/B2组合稳定性](results/optimization-combination-stability.summary.md)。失败B与B2比较只作诊断。

## 首轮单次记录

固定512MiB中文 Wikipedia、none预处理、目标50,000/min2、串行feed、4线程初始化与merge，u32 corpus/posting；全部通过原Trainer接口。模型完整摘要、50,000词表、29,243规则和1,429,915个唯一片段一致。每候选只做关键计时；A的追加同期诊断用于解决实际merge回退，未展开矩阵。

| 版本 | 初始化 s | Merge s | Train s | RSS GiB | 最低可用 GiB | commit |
|---|---:|---:|---:|---:|---:|---|
| [原并行版](results/native-fair-512.count4.jsonl) | 22.975 | 27.457 | 54.760 | 3.390 | 4.424 | `b6a28768` |
| [同期诊断原版](results/optimization-a-diagnostic.baseline.jsonl) | 23.760 | 29.810 | 57.982 | 3.395 | 4.390 | `b6a28768` |
| [A并行构造](results/optimization-a-diagnostic.candidate.jsonl) | 15.421 | 29.756 | 49.737 | 3.417 | 4.337 | `98ca7fc1` |
| [C直接构造](results/optimization-c-512.jsonl) | 9.767 | 26.092 | 39.805 | 3.425 | 4.383 | `fbdc0b2b` |
| [B融合首版](results/optimization-b-512.jsonl) | 13.251 | 31.887 | 49.697 | 3.418 | 4.398 | `c1ee2019` |
| [B2查询优化](results/optimization-b2-512.jsonl) | 14.399 | 21.386 | 40.265 | 3.386 | 4.415 | `a0832c48` |

首轮单次训练最小值为 **C：39.805秒**，相比初始并行版54.760秒快1.38倍；初始化从22.975降到9.767秒，快2.35倍。此单次值不再用于选型；当前推荐由上述同期配对决定。

**B2是有效的merge热点候选**：相对C merge从26.092降到21.386秒，耗时减少18.04%；相对B首版31.887秒减少32.93%。B2整次训练40.265秒，较C多1.16%，尚未证明全训收益，当时未按局部结果取代C；随后同期配对确认整个B2组合更快，已选择B2。

所有采样进程VmSwap峰值为0；停止条件仅MemAvailable≤1GiB。系统换页属于主机范围，完整记录保留在jsonl。计时与静态正确性复核并行；本轮B2计时时CPU测试已经结束，没有并发构建/测试/下载。按用户最新要求，后续构建完成后性能与正确性检查同步推进，失败的正确性结果会使该候选性能记录作废。

## 热点变化

| 阶段 s | A同期诊断 | C | B首版 | B2 |
|---|---:|---:|---:|---:|
| alphabet | 2.563 | 0.347 | 0.413 | 0.345 |
| corpus measure | 0.199 | 0.187 | 0.184 | 0.189 |
| corpus allocate | 0.265864 | 0.000046 | 0.000025 | 0.000035 |
| corpus fill | 2.407 | 0.555 | 0.512 | 0.681 |
| tokenize合计 | 5.436 | 1.090 | 1.110 | 1.216 |
| 初始route | 0.460 | 0.573 | 0.664 | 0.867 |
| 初始count | 9.445 | 8.015 | 11.370 | 12.162 |
| Plan/AA fallback | 4.261 | 3.877 | 0.040 | 0.039 |
| delta（含fused prepare） | 15.979 | 13.888 | 23.142 | 12.196 |
| fused prepare（不可再与delta相加） | — | — | 22.858 | 11.889 |
| rewrite | 0.727 | 0.621 | 0.787 | 0.765 |
| owner commit | 8.431 | 7.368 | 7.481 | 7.952 |

A先把词划成连续独占区域并行写最终数组；C进一步移除逐字符UTF-8字符串hash查询和最终数组串行清零。None alphabet改为每worker存在位图，Some沿用原HF频次裁剪。C tokenize1.090秒，热点转移到初始pair计数和merge索引管理。

B首版虽然省掉大部分Plan排序，却把prepare推到22.858秒。按规则访问丢失了全局空间有序访问的局部性，并新增selected邻居查询；32MiB perf诊断中prepare worker约占30% CPU samples，但内联使单个查询成本未能独立量化。没有把全部回退归给某一种操作。

B2仅替换上述查询：每256位置一个pivot下界目录，桶内精确搜索；selected head/tail用ID直接表，重复head/tail通过小hash表回退。prepare降到11.889秒；这两个改动一起测量，不单独量化各自贡献。owner commit仍7.952秒，是剩余merge热点。

## 改动范围与依赖

| 改动 | 直接影响 | 本轮保留的算法 |
|---|---|---|
| A/C构造 | 字母集合收集、字符ID查询、最终数组初始化、词权重元数据 | merge过滤/排序、delta、write、owner commit |
| B融合 | flat非AA的过滤、取消全局Plan排序、邻边delta、写入调度、出生访问顺序 | 初始化构造/计数算法、选择证书、ID分配、AA/多块/非空affix路径 |
| B2查询 | 融合prepare的权重查询与selected邻居查询，merge起点一次目录构建 | B的出生协议、apply屏障、owner提交算法及全部初始化模块 |

B/B2入口实例化AtomicU32槽位；历史对照已验证相同槽位大小及较小的实测时间差异，用户明确取消本轮重复Atomic计时。初始化算法源码沿用C，实际slot实例化与C不同；B2相对B没有再次改变slot类型。B2新目录是在initialization返回后才建立，初始计数不消费它。C/B/B2初始count测到8.015/11.370/12.162秒，数据结构规模与输入一致；这段差异的原因仍未确定，不用merge查询改动解释它。

A最初同样有一次merge变慢，而原版/A同期诊断merge29.810/29.756秒，没有复现6.5秒回退。源码未改不能代替性能证据；记录全部测量及CPU/fault/context-switch/主机负载，不指定未经证实的NUMA、allocator、编译器或后台负载原因。

## 内存与正确性

历史A/C/B/B2的初始化核心容量相同：slots824,359,780字节、length166,056字节、词权重25,165,824字节、posting1,147,872,496字节，另有owner哈希表/heap。它不是进程RSS，空间口径沿用 [MEMORY_LAYOUT.md](MEMORY_LAYOUT.md)。

C的alphabet临时数组11.57MiB、字符直接表4.25MiB，构造返回后释放；无第二份完整corpus。B2在merge常驻权重目录3,220,160字节（3.07MiB），建立11.784ms；每批selected表峰值800,128字节（0.76MiB），有效起点缓冲峰值17,301,504字节（16.50MiB）。后两项按分配capacity计，不包含allocator开销。RSS实测仍约3.39GiB。

C完整42项库测试通过；B把既有1500-case逐轮HF差分扩到新flat Atomic路径并增加相邻批次/共享头尾/长度/跨worker排序案例，完整43项通过；B2增加目录与原全量pivot查询的逐位置等价测试，完整44项通过。模型签名gate包含10项运行：`d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd`。

独立静态审查分别在 [C审查](CORPUS_DIRECT_REVIEW.md)、[融合及查询审查](FUSED_DIRECT_REVIEW.md)。unsafe数组转换、安全屏障、出生key唯一规则生产者、连续任务顺序、重复head/tail与有限长度协议都有对应推导，审查与计时并行。

## 源码与复现

各候选是独立干净worktree与本地commit，中央分支只保存实验账本。路径与branch在 [WORKTREES.md](WORKTREES.md)，过程及用户要求在 [OPTIMIZATION_PLAN.md](OPTIMIZATION_PLAN.md)、[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)。

在 `benchmarks/hf-bpe`：

```bash
python3 build_native_fair.py /root/code/tokenizers-worktrees/prepare-rule-aggregate --label prepare-rule-aggregate
python3 run_native_fair.py --case rule-aggregate-reproduction \
  --worktree /root/code/tokenizers-worktrees/prepare-rule-aggregate \
  --build-root .build/native-prepare-rule-aggregate --binary target/release/hf-bpe-native-prepare-rule-aggregate \
  --corpus .build/gb-corpus/zh-512m.txt --output results/rule-aggregate-reproduction.jsonl \
  --initialization-workers 4 --merge-workers 4 --atomic-corpus \
  --require-stats fused_prepare_ms --require-stats weight_lookup_bytes \
  --require-stats peak_prepare_aggregate_bytes
```

每次environment JSON锁定实际worktree源码、临时探针副本、runner/Cargo.lock、二进制、输入和脚本hash；probe只在原do_train返回前输出私有统计。实际调用原train_vocab，未增加公有Trainer选算法API。计时原始数据与阶段解释见 [C结果](results/optimization-c-512.summary.md)、[B结果](results/optimization-b-512.summary.md)、[B2结果](results/optimization-b2-512.summary.md)。旧PR/native公平四项见 [PARALLEL_REPORT.md](PARALLEL_REPORT.md)。

当前选择J；H两对完整组合证据作为保留基线。失败B/F/G/I/K均保存来源与结果，不纳入推荐。没有追加Atomic对照或宽度/线程矩阵。各新增候选由实际采样、阶段边界与动态计数支持，完整收益与归因边界分开记录。
