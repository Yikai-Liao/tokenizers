# BPE 全流程简化迭代：2026-10-10

按用户要求，连续执行可回退的完整切片、必要测试、冻结中文成本测量和新上下文
审查。搜索与采用分开：有待验证成本的思路先实现，实测后决定；一次审查只读
整个引擎，不按文件把全流程拆散。每轮同时寻找后续可退出的表示、状态协调和
平行流程，直到新审查没有值得启动的完整切片。

起点为 `simplify/bpe-heap-20261010` 的 `3a298346`。该提交统一普通与 reuse
候选优先队列，生产实现为 2120 行；Arena 及阈值保留。此前资源测量见
[HEAP_EXPERIMENT.md](HEAP_EXPERIMENT.md)。后续分支为
`simplify/bpe-iterations-20261010`。

## 统一 birth 的交付表示

隔离 prototype 让准备阶段统一交付压缩 Positions。完整普通 producer 仍在准备时按 floor
剪枝，并按现有 Arena 阈值编码；partial、AA 和 reuse 使用 owned 片段，
在 owner 汇总 count 后决定保留。Fresh 单片段移交，多片段使用既有 Fragments
编码；reuse 解码、排序并发布历史 cohort。Owner 不再解释 Partial／Complete，
Birth 双表示、Complete 独立发布和 Builder::append 一起退出。

首版生产实现 2120→2065，测试仍为 799。default／no-default 各 17 native
tests＋1 doctest、all-target Clippy、fmt 和行数预算通过。直接导入实际 positions
模块的 Miri 两项测试也通过。新上下文通读六模块和完整边界，没有找到新增的
已证明缺陷，记录见
[independent-review-v1.md](evidence/birth-20261010/independent-review-v1.md)。

首版统一分组为每个单片段 birth 新增 Vec allocation。真实中文的三组
交替成对测量显示如下变化：

| workers | 输入 | wall 成对变化 | CPU 成对变化 | HWM 成对变化 |
| ---: | --- | ---: | ---: | ---: |
| 4 | ByteLevel | +8.80% | +9.21% | +0.10% |
| 4 | Whitespace | +9.20% | +10.22% | -0.86% |
| 6 | ByteLevel | +13.13% | +12.28% | -0.25% |
| 6 | Whitespace | +5.28% | +5.43% | -0.15% |

**按用户决定拒绝并回退此切片。** 第二版尝试已有 SmallVec 内联首片段，
通过 native tests、Clippy 和构建，但没有完成成本测量。它也已回退；测试通过
不能被解释为修复性能。两版源码和日志保留在 evidence，当前引擎保留原有
Partial Builder／Complete Positions 及完整 producer 的提前剪枝和直接发布。

拒绝后，另一独立上下文重新审查原接口：Birth 两 variant 是必要的内部能力
契约。Partial 表示局部结果，owner 必须先归并总频率；Complete 表示 fresh
ordinary producer 已覆盖完整列表、按 floor 剪枝并冻结，可以直接发布。
完整性推导留在 merge，owner 只消费结论，coordinator 和公共调用者不接触
Birth 分派。因此原接口仍符合深模块原则。把分派藏进 wrapper、把 Complete
包成 Candidate 或移动三阶段调用，没有消除这些职责，也没有强结构收益。
本轮仅就近补充两种 Birth 的契约注释。

## 统一写计划的实验

上一轮新审查提出让 fresh 与 reuse 都保存完整 Match。Fresh 已取得完整 Match，
旧版丢弃 right／after，只存 start 与规则几何，apply 时再重构；reuse 直接存
Match。隔离实现退出 Writes 两 variant、几何镜像、坐标往返和两处分派，
record 也无需返回永远成功的 Result。准备及应用的 joined 次序和不同 matcher
保留。纯 Writes 切片减少 42 行生产实现，测试没有删除。

本次成本比较仅用 `3a298346` 加纯 Writes 补丁，SHA256
`fa86dced33ea3220a0346491e3114de7bf3e64bf03a56d4fa5e37dd26a4db956`。
生产实现 2120→2078，测试仍为 799。default／no-default 各 17 native tests＋
1 doctest、all-target Clippy 和 fmt 已通过，positions unsafe 没有变化。
更早同时含 birth prototype 的试跑归入 excluded-mixed-prototype，不作为
独立 Writes 的采用依据。

**按用户决定拒绝并回退 Writes。** 当前引擎与隔离试验工作树都恢复原有
Compact／Occurrences 写缓存。用户判断少 42 行的收益不值得本次真实语料
观察到的代价；剩余测量因此停止。有效结果如下，变化均按正式配对计算：

| workers | 输入 | 已完成正式配对 | wall 变化 | CPU 变化 | HWM 变化 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4 | ByteLevel | 3／3 | +3.94% | +1.12% | +5.30% |
| 4 | Whitespace | 1／3 | +2.35% | +1.14% | +0.02% |
| 6 | 两种输入 | 未启动 | — | — | — |

ByteLevel 的三组原始训练时间如下。汇总是三组变化的中位数，因此不等于
分别取两组时间中位数再求比值。

| 配对 | baseline wall（秒） | candidate wall（秒） | wall 变化 | CPU 变化 | HWM 变化 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 20.4404 | 20.3255 | -0.56% | -1.06% | +5.17% |
| 2 | 20.2060 | 21.1806 | +4.82% | +1.12% | +5.79% |
| 3 | 19.8869 | 20.6695 | +3.94% | +4.02% | +5.30% |

共有 12 个完整进程，包含预热；完整模型均一致、child swap 均为 0，未观察到
并发编译或其他被测 runner。Whitespace 第二组的 baseline 被主动终止，
不计入结果，也不补成完整矩阵。详情保存在
[real-workers4-runs.json](evidence/writes-20261010/real-workers4-runs.json)、
[real-workers4-summary.json](evidence/writes-20261010/real-workers4-summary.json)
和 [measurement-status.json](evidence/writes-20261010/measurement-status.json)，
包括完整 pair 明细、binary／source／input 哈希与中断状态。

独立新上下文固定在纯 Writes 补丁，通读六模块和公共边界，新增具体缺陷 0、
值得下一轮真实实验的强候选 0。早先混合 prototype 的静态结论单独保存，
没有冒充纯 Writes 的固定快照审查。静态正确性没有被当作成本接受依据。

长 AB／AA 单词压力实验仅用于说明临时写缓冲的机制。64 位平台 fresh 写
记录由常见 4 字节 start 增至 24 字节 Match；right／after 原可由固定 rule
geometry 重建。32M 次 AB 的单词压力中，进程 HWM 从 631.6 MiB 增至
1290.4 MiB；对应约 640 MiB 的额外写记录内容。这个构造不代表真实中文，
不能凭它推断正常训练的增长，更不能单凭它否决候选。它的独立 36 个样本
全部模型一致、swap 为 0；真实数据和压力数据分开报告。

## 共同验证边界

真实测量沿用固定 `wikimedia/wikipedia`、configuration `20231101.zh`、revision
`b04c8d1ceb2f5cd4588862100d08de323dccfbaa` 的中文文本完整行前缀，
268,435,162 字节、666,896 行。来源、选择方法和哈希见
[real-corpus-manifest.json](evidence/writes-20261010/real-corpus-manifest.json)。
沿用该 256 MiB 源语料的 ByteLevel／Whitespace prepared words，
birth 使用 4／6 workers，Writes 原计划同样比较，实际仅运行上述 4-worker
明细。对应 CPU affinity 0–3／0–5，vocab 50k、min frequency 2、
无 affix。一组预热不计入汇总，三组 AB／BA 成对测量比较完整模型，检查 child
swap 和并发编译／被测进程。训练 wall／CPU 仅覆盖 public do_train；进程 HWM
包含加载。三组共享 VM 样本描述本次观察，不声称统计提速或完整语义等价。

所有语义检查保留 alias、AA、reserved 身份、重启、零权重、strict length、
signed action 次序、historical cohort、完整 u64 和 scoped owned／Arena 回收。
前轮同优先级 reuse cohort 次序尚未完全证明，既有扩展 reference fixture 的
差异也仍存在。这轮不把普通中文模型一致解释为关闭这些历史限制。

## 最终恢复与停止条件

本轮两项候选均已回退。相对起点 `3a298346`，最终生产源码只有四行 Birth
契约注释，逐字去掉这四行后与原文件一致；其他 engine 生产源码未变化。
生产实现仍为 2120 行，测试仍为 799 行，fmt、diff 检查和行数预算通过。
最终是注释改动，因此继承起点 native 验证，没有重新执行训练测试。

最新独立 reviewer 在完整六模块及公共边界审查后，复核这个最终恢复对象：
新增具体缺陷 0、值得下一轮实验的强全局简化候选 0；原有深模块边界成立。
报告见 [final-independent-review.md](evidence/iterations-20261010/final-independent-review.md)，
恢复校验和命令输出见 [validation.json](evidence/iterations-20261010/validation.json)。
用户要求的“测试、成本测量、fresh review，直到没有值得试验的新思路”迭代
据此结束，没有为维持迭代而再引入已拒绝方案。
