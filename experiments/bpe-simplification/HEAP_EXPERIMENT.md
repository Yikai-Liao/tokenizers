# 普通队列与 reuse cohort 统一堆实验：2026-10-10

本实验按用户要求实现并测量延迟释放成本。对照是 `177302e2` 的 BPE 引擎，
分支为 `simplify/bpe-heap-20261010`。它是实验实现，尚未作为可替换主线的
完整语义等价结论。

## 实现与回收

原版在各 owner 内持有 `Pair -> State<Positions>` 和轻量 `Priority` 堆；reuse
另有拥有历史 occurrence 列表的 cohort 堆。实验版将各 owner 简化为
`Pair -> count`，所有 Positions 都由一个全局 `Candidate` 堆拥有。选择时统一
修正过时 count；fresh key 消失时弹出 Candidate 才回收列表。reuse 仍保留
signed ledger、历史 cohorts 及逐 action 的 remove-before-birth 次序。

fresh count 降到 floor 以下后，count map 可以立即删除，heap 中的 Positions
可能继续存在。Arena 和动态阈值 `max(256, sqrt(items / 256))` 保留；inline、
Arena／owned 分配机制也保持原样。这里的 heap 是候选优先队列，变更的是
列表的拥有者和回收时机。partial 与 complete birth 汇总后串行 push。没有另加 handle、
清扫或撤销目录。格式化后的生产实现 **2176 → 2120** 行，测试仍为 **799**
行；减少 56 行来自同一套选择、take 和 birth 发布流程。

## 测量边界

真实输入为既有冻结的中文 256 MiB 源语料，分别由 ByteLevel 和
Whitespace 准备词频。两边加载同一份词频，4 workers，CPU affinity 0–3，
vocab 50k，min frequency 2，无 affix。无插桩版本每个输入执行一组预热、
三组成对测量，顺序交替 AB／BA；预热不进入汇总。

训练 wall 和 CPU 只覆盖 public `do_train`；HWM 在完整模型序列化前读取，
包含输入加载。完整 vocab／merge model 在计时结束后比较；子进程 swap 为
0 才接受样本。每次启动检查 cargo、rustc 和实际 runner executable 的并发。
共享 VM 上三组样本只用于描述观察，不声称统计提速或等价。

诊断版单独执行一组成对运行。Positions 按已经验证的实际 Layout 统计 owned
allocation 字节，包括 header、restart directory 和压缩 stream，排除 allocator
metadata、inline 和 Arena。全局原子 live／peak 还包含编码中的临时片段；
index-owned 仅统计 checkpoint 上归索引拥有的列表。首次 commit、之后每
32 轮 commit 及完成时采样；压力输入每轮采样。所有任务已 join 后，再统计
heap 中 fresh key 已消失的 stale-owned 列表和进程 RSS。
Arena 已分配块、worker scratch capacity、heap descriptor capacity、map 元素
载荷 capacity 分别报告；map 估算不包含 bucket/control metadata。

诊断每次采样遍历整个索引并查 count map，尤其中文候选达数百万，时间开销显著。
诊断耗时不能用于比较实现速度。这里的 stale 峰值指 joined checkpoint 中的
最大观测值；全局 owned peak 由分配／释放原子计数记录。

首次诊断逐轮遍历，完成两组英文和中文 ByteLevel 对照后，统一堆中文诊断的
采样开销显著。作者停止该未完成样本，保留原始记录，随后改为上述固定采样
并重新测量中文；未完成样本不纳入最终汇总。按用户后续要求聚焦中文，没有
继续执行英文无插桩矩阵。

## 压力输入

零权重长词 `ab` 重复 N 次保存大量 AB 坐标；`abc:1` 使 AB 初始 count 为 1，
`bc:2` 使 BC count 为 3。另有 2048 个端点互不共享的双字符词，权重均为 2。
BC merge 消耗 AB 唯一正权重 occurrence 后，旧版立即删除并释放 AB 列表；
统一堆把旧 AB 列表留在优先级 1 的候选里，等待更高优先级候选处理完。
零权重坐标是公共词频输入契约的一部分，不能按其权重忽略。

N 分别为 2²⁰、2²³、2²⁵。vocab 10k，min frequency 1，其余设置相同；
prepared words 按字典序保存，caller AHashMap seeds 为 `[11,13,17,19]`。
这是隔离回收机制的压力输入，不能代表真实语料的典型分布。初始化编码时的
临时 Builder／片段峰值可能高于回收后的差异，须同时看 commit 后的字节与
RSS，不能把 HWM 变化小解释为延迟释放没有代价。

## 正确性与复核限制

原 default/no-default 各 **17 native tests＋1 doctest**、Clippy all-targets
`-D warnings` 已通过。扩展探针仅放在隔离副本，最终生产源码和原测试已恢复：

- 简单同 pair／同 count alias fixture 重复 32 次匹配参考的完整 model／trace。
- 生成对照从 64 扩到 1024 时在 case 84 停止。对照 `177302e2` 与统一堆的
  engine trace 逐字一致，二者均不等于 reference；这是对照已存在的差异。
- alias fixture 加入 64 个独立高优先 pair 后，两边都出现 reference 差异，
  两次调用方 map 的 ID 顺序不同，不能视为跨实现精确等价证明。

初始 priority 唯一不足以保证后续等优先历史 cohort 的弹出次序。此问题未关闭，
因此即使本次资源测量的模型相等，也只证明这些输入上的一致性。独立只读
复核没有发现新增的所有权、数值更新或错误清理缺陷，完整记录见
[independent-review.md](evidence/heap-20261010/independent-review.md)。

## 实测结果

4／6 workers 的无插桩结果已经完整，以下为三组成对变化的中位数；绝对值分别是
各实现三次测量的中位数，百分比按每对计算后取中位数，二者不一定能直接互除。

| 中文输入 | HWM GiB：对照 → 统一堆 | HWM 成对变化 | wall 秒：对照 → 统一堆 | wall 成对变化 | CPU 成对变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ByteLevel，4 workers | 2.440 → 2.373 | -3.11% | 19.754 → 19.312 | -2.70% | -2.18% |
| Whitespace，4 workers | 1.819 → 1.885 | +3.63% | 16.774 → 17.660 | +4.61% | +2.95% |
| ByteLevel，6 workers | 2.401 → 2.399 | -0.30% | 14.977 → 15.065 | +0.13% | -0.56% |
| Whitespace，6 workers | 1.835 → 1.789 | -2.40% | 13.629 → 13.657 | +0.20% | -0.56% |

诊断中的失效 owned 列表与总体 HWM 不是同一指标：中文 ByteLevel 的最大
采样值是 **230.05 MiB**，完成时仍持有 **203.68 MiB**；Whitespace 的最大
采样值为 **2.44 MiB**。两边也有不同的 queue descriptor 和 map capacity
分布，allocator／Arena／scratch 驻留共同决定进程峰值。因此额外 held bytes
不应直接加到对照进程 HWM 上。

| 压力长词中的 AB 数量 | 延迟持有的精确 owned MiB | 首次 commit 后 RSS 差 MiB | 无插桩 HWM 成对变化 |
| --- | ---: | ---: | ---: |
| 2²⁰ | 1.117 | +0.211 | -1.06% |
| 2²³ | 8.938 | -0.375 | +0.12% |
| 2²⁵ | 35.750 | +35.656 | +0.009% |

三种压力输入中，失效 AB 列表均额外保留 **8 轮 commit**，直到 merge 2048
后的更高优先级候选处理完。旧版 index-owned 在第一轮即回收；统一堆直到它
到达堆顶才释放。释放 owned layout 字节不保证对应同量 RSS 下降，较小两组
RSS差异也包含 allocator 与其他驻留变化。最大压力组确实多驻留约 35.7 MiB，
但整体 HWM 被早期约 400 MiB 的初始化峰值覆盖，最终几乎不变。

## 处理队列占训练时间的比例

独立轻量计时二进制只在队列阶段边界计时，不遍历索引、不对 occurrence 计时。
每个输入分别跑 4／6 workers 一轮，完整模型与无插桩对照一致。分母是同一次
profile 的 public `do_train` wall，区间互不重叠。

| 中文输入／workers | 初始建堆 | best＋take | 串行 birth 入队 | 最终 index 清理 | 队列相关合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ByteLevel／4 | 0.003% | 2.42% | 0.329 s／1.73% | 0.46% | **4.60%** |
| Whitespace／4 | 0.45% | 1.32% | 0.201 s／1.20% | 0.55% | **3.52%** |
| ByteLevel／6 | 0.003% | 3.02% | 0.312 s／2.07% | 0.57% | **5.66%** |
| Whitespace／6 | 0.56% | 1.57% | 0.216 s／1.49% | 0.55% | **4.17%** |

best／take 包含 count-map 校验、过时修正、列表释放；index 清理也包括 count
maps 与 routes。这是队列相关路径的 elapsed 上界归因，不能称纯 heap CPU。
并行 owner 生成 Candidate、更新计数和编码不在其中。计时器与原子记录有
少量开销，单轮 profile 不是用于判断实现提速的统计样本。

普通模式新增串行的是 joined birth 发布，以及全局初始建堆；候选选择和最终
清理在对照中也由 coordinator 串行执行。不能把队列总占比全算成新增串行
工作。实测串行 birth push 占 **1.20%～2.07%**，绝对时间 **0.201～0.329 s**，
在当前中文输入与 4／6 CPU 上占比较小。4→6 workers 时绝对时间基本不变，
占比随总训练缩短略升。机器仅允许 CPU 0–5，没有执行 16 workers 的超订阅
测量，也不能据此外推 16／32 CPU 的扩展性。

64 次无插桩、驻留诊断及轻量阶段计时的完整模型全部一致，child swap 均为
0；每次启动未观察到并发编译或其他被测 runner。4 workers 的差异约为
几个百分点，6 workers 的训练 wall 变化约 +0.1%／+0.2%。当前环境没有观测
到明显串行退化，资源代价不构成直接排除这个简化的证据。

源码、二进制、锁文件与原始日志哈希见
[validation.json](evidence/heap-20261010/validation.json)。无插桩逐次结果为
[clean-runs.json](evidence/heap-20261010/clean-runs.json) 和
[scale6-runs.json](evidence/heap-20261010/scale6-runs.json)，阶段计时为
[heap-timing-runs.json](evidence/heap-20261010/heap-timing-runs.json)，驻留轨迹为
[diagnostic-runs.json](evidence/heap-20261010/diagnostic-runs.json)。早期被作者
中止的诊断样本单独保留，不进入最终汇总。

## 后续结构候选

按用户要求另一个 subagent 只读审计整个六模块流程，只找到一项值得实验的
候选：准备阶段统一交付压缩 `Positions`。完整 producer 保留提前 floor
剪枝；partial／AA／reuse 交付 owned 压缩片段；owner 用已有 Fragments
归并、单片段移交，reuse 仍解码排序。

它可能一起退出 Birth 的 Partial／Complete 双表示、独立发布分支和
Builder::append，让提交不再解释 producer 完整性。Partial 多一次 codec，
complete 多一次分组，须实测，不能提前声称净收益。详细源码证据、语义边界
和验证入口见 [next-simplification-review.md](evidence/heap-20261010/next-simplification-review.md)。
