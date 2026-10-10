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

## flat 分支的状态表达与互斥存储整理

本轮基线为 `simplify/bpe-flat-20261010` 的 `499fd7ab`，按软件设计哲学与
clarity 的代码表达规则重新审查，范围排除 parity。保留五项整理：attempt
显式返回 Complete／RestartForReuse；统一初始化与扫描的装饰位计算；空
Builder 接管后直接返回；Group 的 ordered／unordered 缓冲改为互斥表示；
FreshSnapshot 的普通查表与 AA starts 改为互斥表示。共享 neighbor 算法、
Birth／Writes 双表示和既有存储边界继续保留。

生产实现增加 57 行，收益是减少隐含有效缓冲规则和状态转译。当前目标平台
Group 布局由 64 降至 40 字节；AA 不再构造普通 selected／heads／tails。
各项依次构建和验证，再比较 baseline、局部整理、Group、完整版本四个累积
版本。冻结中文 ByteLevel／Whitespace 各排除一组预热，保留两组正式轮换
测量，四 workers 固定在 CPU 0–3；24 次完整模型均一致，child swap 均为 0。

| 完整版本相对本轮基线 | wall 成对变化 | CPU 成对变化 | HWM 成对变化 |
| --- | ---: | ---: | ---: |
| 中文 ByteLevel 256 MiB | −1.70% | −0.13% | −0.14% |
| 中文 Whitespace 256 MiB | −1.42% | −0.64% | −0.10% |
| 长 AA 定向负载 | +1.17% | +0.53% | +2.04% |
| affix／reuse 定向负载 | −2.56% | −3.27% | −0.62% |

定向负载各排除 AB 预热，再比较正式 BA／AB；12 次完整模型均一致，无 swap。
AA 的正式 HWM 中位数由 489,182 增至 499,162 KiB，约多 9.7 MiB；该差异已在
推送前向用户报告。用户接受约 1%–2% 量级差异，决定保留，并授权直接推送。
本轮不声称普遍提速或所有负载的峰值内存不回退，布局缩小也不等于 RSS 必降。

default／no-default 各 20 项现有 library tests、all-target Clippy（warnings
denied）、改动文件 rustfmt 与 diff 检查通过。完整源码、逐次模型验证、布局
probe、input／source／binary 哈希、测量命令与日志见
[本轮记录](evidence/readability-20261010/README.md)。

## flat 分支的内部命名校准

在 `04553f26` 上完成独立 clarity／软件设计审查，并以 Hugging Face 官方
main 的 `fb49a292` 核对命名惯例。保留结构边界，修正以下内部名称：
`Corpus::has_been_activated` 明确激活历史在最后一次出现消失后仍保留；
`CodecScratch` 明确位置压缩的复用 byte／offset 缓冲，相关字段和局部量用
scratch；`Positions::iter_from` 明确返回迭代器，调用处用 index 区分列表
下标与语料坐标；`SelectedRules::replacement` 明确查表返回替换 token ID。
上游的 EncodeScratch 服务于 tokenizer encode，职责不同，因此不直接沿用
该名称。真正的解码 Cursor、SelectedRules、FreshMatches 等名称保留。
Codec 构造参数 num_threads 表示预期执行线程数，仅用于预留 scratch 容量；
index 的路由参数及局部量改为 shard_count，明确 count owner 分片与执行线程
没有绑定关系。任务本地目录与 codec scratch lease 的旧 worker 描述同步修正。

同步修正 README／DESIGN／COVERAGE 的历史激活和缓冲说明，以及上一轮记录
误写的 arena thresholds。证据文档的源码哈希比对明确指向当时的保留提交，
历史源码快照和原始测量数据保持原样。
改名后由独立 sub-agent 复核完整普通 BPE 源码、测试、调用点和当前文档。
发现的角色描述、测试身份复用说明、restart 时序和 lease 并行范围歧义均已
修正；最终 diff 未发现剩余命名遗漏或源码与文档矛盾。真实执行线程的
workers／parallel_workers 与解码 Cursor 保留。
本轮只涉及命名、注释和格式，运算、分支顺序、分配与数据布局均未改动；
未重新执行性能基准。default／no-default 各 20 项现有 library tests 和
all-target Clippy（warnings denied）通过，改动文件 rustfmt 与 diff 检查通过。

## 所有冻结 positions 改用 Box 字节流

按用户要求移除 Empty／One／Two 内联特化，所有非空列表统一使用原有
restart／delta 编码，空列表使用空 Box。positions.rs 的生产代码净减少
48 行（334 → 286，排除测试、注释和空行）；x86-64 的 Positions 从
24 → 16 bytes，Candidate 从 40 → 32 bytes。保留 full-u64 坐标、检查算术、
独立所有权和并发只读迭代。代码已在 `9f5b553d` 推送。

四种真实 256 MiB 输入分别做一个排除的热身和四组正式交替配对。
英文 ByteLevel 的 CPU／wall 配对中位差为 −2.99%／−1.98%，峰值 +0.14%；
英文 Whitespace 为 +1.15%／+4.57%，峰值 +10.41%，绝对增加约 13.7 MiB，
用户接受该英文低基数增长。中文 ByteLevel 为 −3.48%／−3.67%，峰值 −1.65%；
中文 Whitespace 为 +5.26%／+6.23%，峰值 −2.16%。所有 40 次完整模型一致，
无 swap、无并发构建或基准。

针对中文 Whitespace 的 VPS 波动疑问，再分别以 min_frequency 2／3 做
各四组正式配对。阈值 2 的 CPU 中位差为 +4.94%，合并两轮八组为 +5.26%；
此前约 9.8% 只是单次配对，不能作为整体结果。阈值 3 的差距降至 +2.64%，
尚不能认定相同。whole-process perf 指令／周期中位差分别为
+1.27%／+4.13% 和 +0.34%／+1.67%；该边界含加载和序列化，且使用 VPS
虚拟 PMU，因此只能辅助说明额外工作，不能精确归因训练 CPU。

独立诊断统计的是累计冻结调用，不是唯一或同时驻留列表：阈值 2 时
非空列表中的 52.77% 长度为 1／2。初始化单位置列表的局部权重有 96.35%
为 1，符合用户判断，但初始化先冻结再汇总全局频率。提高阈值到 3 后，
complete-birth 的短列表从 2,047,383 → 155,353，初始化的 4,518,165 个
短列表不变。用户随后要求优先尝试全局过滤后再编码，再评估内联容量；
该后续实验的成本、语料规模和分布另行记录，不混入本轮保留结果。

普通 BPE 范围的 12 项测试通过（按用户要求排除既有 tokenizer encoding
roundtrip 问题）；all-target no-default Clippy（warnings denied）、改动文件
rustfmt 与 diff 检查通过。测量排除加载和模型序列化，峰值则含 caller map。
完整源码、布局、逐次数据、模型哈希、诊断 patch 与计数口径见
[Box-only 记录](evidence/box-only-20261010/README.md)。

## positions 独立构造与统一 u64 临时存储

按用户要求恢复冻结列表的 Empty／One／Two／Compressed Enum，保留短列表
内联，移除 Codec、CodecScratch、Lease、Input、临时编码 scratch 和 tk-train
的 thread_local 依赖。Positions 接收有序 slice，或消费有序 fragments；单个
非空 fragment 直接返回。较长列表在最终 Vec 中一遍写入 restart／delta，
填充前缀目录再转 Box；Vec 增长及 boxing 仍可能重新分配。

随后移除自定义 Builder／Buffer 与 u32→u64 promotion，生产端直接使用
`SmallVec<[u64; 2]>`。冻结 Positions 为 24 bytes，Candidate 为 40 bytes；
临时 SmallVec 为 24 bytes。u64/4 控制为 40 bytes，未显示稳定收益。
用户优先考虑大规模、低重复中文输入，并接受数个百分点的 CPU 波动，
选择统一 u64/2；提出的 u32/4 后续比较在构建、测量前取消。

已完成的四种 256 MiB 输入，相对原 Enum＋编码 scratch 的正式配对中位差
（CPU／wall／峰值 RSS）：英文 ByteLevel −3.12%／−6.49%／+8.30%；
英文 Whitespace +6.09%／+5.10%／+17.84%；中文 ByteLevel
+1.01%／+2.83%／+5.28%；中文 Whitespace +2.79%／+1.49%／+0.37%。
每个输入仅两组正式配对，属于 VPS 上的描述性观察；不声称普遍提速或
内存改善。u32 阈值是预处理、去重后原始符号及分隔符的坐标，不能从原文
GB 数直接推断 promotion；本轮没有 GB 规模测量。

四种表示、八个 16／256 MiB 输入的 96 次 screen，加 AA／reuse 的四次
定向完整模型比较，共 100 次模型一致且无 swap。首版直接编码的 14 项
普通 BPE 检查通过，已有 tokenizer encoding 断言在未改 baseline 同样失败，
按用户指示排除。direct/narrow 与 u64/4 的 all-target Clippy（warnings
denied）通过，最终 u64/2 完成 release 构建与逐模型比较；保留源码哈希
与测量快照完全一致。用户要求停止追加测量并直接整理推送。
完整源码、已有逐次数据、协议、模型与哈希见
[positions 构造记录](evidence/positions-value-20261011/README.md)。

## positions 读取接口与临时格式规则

Positions 用借用 Chunk 封装压缩块范围，片段仅提供 len／iter；三个 block
方法全部收为私有。merge 仅保存片段，按实际片段长度与候选列表长度判断
完整 producer，保留目标向上对齐后仍可能只有一个完整任务的行为。AA
过滤后的 slice 始终走部分 producer。按语料坐标定位并迭代合并为
iter_from_value，列表下标转换留在 Positions 内部。独立 prefix helper
内联到字节流访问中，函数间补齐空行。

新增覆盖 chunks 全量拼接、跨边界重复坐标、空／短列表、非 restart 倍数
目标、完整／部分 publication 分类，以及 127／129／257 ABC 重复次数
的完整模型与逐规则 oracle 比较。no-default 普通 BPE 17 项检查及
all-target Clippy（warnings denied）通过；沿用此前排除的既有 tokenizer
encoding 断言。本轮没有追加性能测量。

临时 [format-changed.py](format-changed.py) 与 BPE 目录 AGENTS.md 纳入 Git。
脚本只处理相对指定 base 有变更的 BPE Rust 文件，并关闭子模块递归；
rustfmt 后用语法树补函数间空行，保留文档与属性相邻，跳过 raw string
中的伪函数。独立临时 Git fixture 验证只读检查、重复执行稳定以及未改文件
字节完全不变。提交上游 PR 前可以一并删除脚本与临时目录规则。

## positions 关键注释、测试整理与中文 256 MiB 复测

补齐压缩布局与目录偏移基准、lower_bound 退一块的理由、内联数组的零填充、
跨 restart 的非递减校验，以及 Cursor 依赖内部编码格式和已知元素数的前提。
bytes 改为 compressed_bytes，要求 Compressed 变体，内联误用走 unreachable。
PairIndex::best 就地说明等 pair／priority 仍可能对应不同位置 cohort，保留
pop→修正→push，避免 peek_mut 改变同优先级批次顺序。

五个存储测试目的保留；删除 SmallVec 自身操作与重复空片段准备，用小表
保留分片边界案例，降序 restart 检查移入非法输入测试。测试由 219 → 187
格式化行，Positions 全文件由 507 → 487 行；精确字节、单片段原指针复用、
跨块重复值、u64 边界、并发读和 Miri 采样均保留。17 项 no-default 普通
BPE 检查、all-target Clippy（warnings denied）与变更文件格式检查通过，
沿用排除的既有 tokenizer encoding 断言。

按用户要求，以接口调整前的 26aa5926 为基线，对当前借用 Chunk／坐标
迭代实现重新测量中文 256 MiB 的 ByteLevel 和 Whitespace。各做一个排除
的预热与四组交替正式配对：CPU／wall／峰值 RSS 中位差分别为
+0.81%／+0.87%／−0.43%，以及 −0.15%／+0.89%／−0.56%。20 次
完整模型一致，无 swap、无并行构建或测试。ByteLevel 前两组 CPU 曾为
+2.93%，完整四组降至 +0.81%，不能由早期样本归因于 Chunk。
用户决定零点几的差异不继续优化；尚未构建的单字节解码试改已撤回，
保留源码与本轮已测快照哈希一致。
协议、源码、构建与二进制哈希、逐次数据和参考模型见
[中文读取接口复测](evidence/positions-read-20261011/README.md)。


## Merge 完整批次与扫描边界

以 d1ede6dd（98524a4 之后的注释与测试整理）为基线，先单独提交完整批次
入口，再整理内部准备流程。Batch::commit 消费选中的规则，依次汇合并行
准备、汇合并行端点写入、调用 PairIndex::commit 并行更新各 owner。
删除 Prepared 类型及外层 prepare→apply→index.commit 协议；训练循环
只负责选择、记录规则、执行批次和更新进度。准备失败不会写入端点；
提交仍可能在写入或部分计数更新之后失败，错误需废弃整个训练尝试。

Fresh 的左右事件放进 record_neighbors，共同说明相邻选中匹配之间的
边界由左侧匹配负责，右侧跳过重复登记，即使两个匹配属于不同任务。
任务只携带 rank 与借用位置片段，由 snapshot 按 rank 取得 Rule，删除
rank 与 Rule 引用必须相符的双重身份。CohortPreparation::prepare 收回
构造和执行顺序，任务入口负责初始化、逐词调用及收尾；prepare_word
保持完整顺序扫描。floor 只用于收尾，移出扫描上下文。Job、两种写入
几何、Partial／Complete、Directories／touched 和 FreshSnapshot 保留。

两个阶段各通过 18 项 no-default BPE 检查，完整模型及逐条规则与独立
顺序实现在 1／4／8 workers 下比较；沿用排除的既有 tokenizer encoding
断言。新增准备失败回归：一个独立任务可正常准备，另一个任务在邻接
计数累加时溢出，三种线程数下所有端点保持原值。原 AA 测例只有 4097
个符号，覆盖 restart 块但未跨准备任务；扩到 8195 个符号产生 4097 个
非重叠起点，确保一个线程下也跨 4096 起点的任务边界。affix/reuse、
严格长度门限、零权重、完整／部分发布和等优先级历史 cohort 顺序均回归。
最终 all-target Clippy（warnings denied）与变更文件格式检查通过。
本轮未测性能，不据结构调整声称吞吐或内存收益。
