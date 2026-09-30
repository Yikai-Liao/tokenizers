# BPE 实验记录

## 当前比较规则

- 使用固定来源的真实 Wikipedia 段落，同一输入、预处理、词表与最小频率；核对完整词表及有序 merges 的摘要。
- 新的公平比较统一 u32 token ID 与 u32 posting。PR 的每 Symbol 长度字段属于其算法布局，另列；不拿窄 ID 的内存优势解释算法收益。
- 双方先用 1 线程初始化计数、4 线程 merge；我方 4 线程初始化另列为优化项。
- 只做关键单次测试，完整矩阵等算法确定后再运行。
- 仅 MemAvailable ≤ 1 GiB 时停止；进程与系统 swap 如实记录，少量 swap 可以接受。构建、测试、下载与计时错开。
- 按实现创建独立 worktree、branch、commit，直接使用原 BpeTrainer::train_vocab/train 接口。

## 2026-09-30：旧 u16 并行访问对照

固定 Wikipedia revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`，先准备约 1 GiB 独立原文，再取完整行前缀 `536870289` 字节。输入 SHA256 为 `a0d40d4102cba1e933f25e5ccd17552d2eaebaec4ef770bc39d8e46e740b4d4f`。目标词表 50,000，min_frequency=2，none 预处理，无 affix/special/长度限制。

| 旧版本配置 | Train s | Merge s | RSS GiB | 最低可用 GiB |
|---|---:|---:|---:|---:|
| u16 corpus、u32 posting、非原子1 worker | 176.063 | 83.933 | 3.062 | 3.046 |
| 同布局、非原子4 worker | 56.430 | 26.627 | 3.005 | 3.046 |
| 同布局、Relaxed 原子4 worker | 56.545 | 26.656 | 3.005 | 3.093 |

三个模型摘要均为 `d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd`；alphabet 20,757，实际29,243条规则，1,550批，最大75规则/批。数据在 [parallel-key-512-final.jsonl](results/parallel-key-512-final.jsonl)，来源、源码、依赖锁及二进制摘要在相邻 environment JSON。

收获：同版本4/1 worker训练3.12×、merge3.15×；原子与非原子4 worker差异仅约0.2%，单次共享主机观测无法区分。4 worker的delta/commit约21.2秒、planning约4.35秒、实际写入约0.65秒。串行HF初始化/语料构造约13.48秒仍需优化。源码确定的新增工作是全局16B Plan拼接、混合规则排序、过滤后第二次delta走访；现有planning计时不能把全部成本归给排序。

这些是历史u16访问实验，不与新的u32公平内存对照混用。

## 2026-09-30：未合并 PR 固定版本

PR #2348 head `6ac0de5359d9e0e1ed0608422575a360ef91b908`。WordArena每Symbol为u32 ID+u32 length，共8B；串行pair计数，一轮一条规则，历史word cohort索引，候选词数≥1000才启用4 worker扫描；复用scratch/delta缓冲。feed串行，训练Rayon4；原checkout干净，仅临时副本插入六个Instant阶段探针。

第一次PR512测试因我错误地把少量进程swap设成停止条件而提前结束；该失败记录保留，不作为训练耗时结论。初始化RSS估算4.57GiB也没有覆盖历史cohort、候选、出生记录的训练增长，后续估算必须加训练期峰值。

按用户纠正后的>1GiB余量规则，GPT-6 Luna完成同一512MiB PR4测试：Train约308.111秒，峰值RSS约6.57GiB，最低可用约1.52GiB，完整模型摘要与旧三项一致。最终数值、各阶段与采样记录在 [pr-fair-512.jsonl](results/pr-fair-512.jsonl) 及 environment JSON。此处不将PR与u16版本的内存差直接解释为算法收益；新的公平控制会使用u32。

## 2026-09-30：原接口 worktree 与统一 u32 比较完成

七种实现分别建本地branch/worktree：未改HF reference、固定PR、串行endpoint、fused，以及u32串行初始化并行merge、u32并行初始化、u32原子访问。五个新实现的 `do_train/train_vocab/Trainer::train` 直接选固定内部引擎；公开字段、builder与serde schema保持原样，实验配置与结果类型改为模块私有。中央根目录 `bpe/experiments` 继续保存历史接口以复现旧记录，开发基点 `bpe/migration-base` 固定为 `8c968e10`。源码索引见 [WORKTREES.md](WORKTREES.md)。

增加 `initialization_workers: Some(1)` 控制项，owner划分和merge线程数仍为4；初始化专池与merge池每次训练各建一次并复用，初始化完成屏障之后才merge。delta权重查询改为有序Plan游标：同词复用、近邻最多8项、远距离二分剩余pivot；等价性经测试与独立审查，本次没有单独隔离其速度收益。

GPT-6 Luna按count1→count4→atomic串行完成以下关键计时，父agent只编辑文本、读取源码；先前PR已完成，无重复计时。相同536,870,289字节真实语料，u32 ID/u32 posting，feed1/merge4，none、vocab50,000、min2；三个native batch cap256，无affix/special/max-length。

| 版本 | 测量commit | 初始化s | Merge s | Train s | RSS GiB | 最低可用GiB | 进程swap采样MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| 固定PR串行计数 | `6ac0de53` | 110.979 | 188.533 | 308.111 | 6.569 | 1.521 | 0 |
| u32初始化1/merge4 | `63e384b8` | 61.722 | 30.859 | 96.906 | 3.626 | 4.281 | 0 |
| u32初始化4/merge4 | `c07a6e39` | 22.975 | 27.457 | 54.760 | 3.390 | 4.424 | 0 |
| AtomicU32初始化4/merge4 | `f2c5415f` | 22.365 | 26.342 | 52.240 | 3.391 | 4.453 | 0 |

PR初始化是special/alphabet/tokenize/count四阶段和，阶段边界与native不同。四项完整(model SHA256、vocab数、merges数、unique_words)签名全部通过：digest `d50fb836…62cd`，50,000词、29,243规则、1,429,915片段。native都是1,550批、最大75规则/批、125,409,599 posting访问。统计与实际编译源码、runner、依赖锁、输入、二进制摘要固定在中央提交 `1396274f`；原始数据及签名见 [native-fair-512.summary.md](results/native-fair-512.summary.md)。

### 本轮收获

1. 在统一u32宽度下，串行初始化控制相对PR训练快3.18倍、RSS少44.8%；并行初始化版快5.63倍、RSS少48.4%。这是该语料的一次进程测量，不能推成所有预处理/affix配置的收益。
2. 我方单次初始化并发1→4使训练快1.77倍；merge两者都是4线程。计数子阶段超线性差异包含缓存与工作调度状态，不称为纯4核加速。
3. 原子版这次训练少4.60%、merge少4.06%，RSS相近；旧u16两项只有约0.2%差异。本轮没有重复样本，无法证明原子或非原子普遍更快。实际写入约0.68秒，管理delta/posting的成本更大。
4. 初始化结构容量约2.158GiB，实测峰值3.39–3.63GiB；两者口径不同。PR的8字节Symbol与我方4字节slot只解释约0.76GiB，剩余约3.18GiB总RSS差距还没有逐结构峰值归因。系统swap存量与页计数单独记录，不能把后台换页归给训练；各次VmSwap采样均为0。
5. 原型以外的16字节全局Plan、混合规则拼接/排序和第二次delta走访仍存在。当前delta+commit约22.36秒；Plan子计时包含其它工作，不能全部归给排序。Halfword/H2.5/H3仍未迁入这些版本。
6. 独立审查复核了新游标、init专池屏障、原tuple API/serde和同算法u32/AtomicU32，见 [WORKTREE_REVIEW.md](WORKTREE_REVIEW.md)。审查发现leaf benchmark残留 `indexed` feature引用私有API；随后删除这项feature及条件分支，追加只涉及runner/docs的提交，引擎源码和测量结果保持对应。中央runner也明确将merge-workers限定为4；原测量脚本可在 `1396274f` 取出。

完整RSS、系统换页、阶段耗时、初始化结构容量与来源见 [PARALLEL_REPORT.md](PARALLEL_REPORT.md)。本次按用户要求只做关键单次比较，串行endpoint/fused/HF reference没有新增512MiB矩阵。

## 2026-09-30：初始化收益、热点与选型结论

本节把测量后的计算、源码解释与下一步建议一起固定。时间来自上述三个native结果和固定PR结果；加速比统一为“基准耗时÷被比较版本耗时”。下面是单次结果，不含重复采样置信区间。

### 初始化并行值得保留

控制项与并行项都是4个owner、4线程merge，仅初始化执行并发从1改为4。

| 阶段 | 初始化1线程 s | 初始化4线程 s | 加速比 |
|---|---:|---:|---:|
| 位置路由 | 2.517 | 0.515 | 4.88× |
| pair计数与过滤 | 45.225 | 9.296 | 4.87× |
| 整个初始化 | 61.722 | 22.975 | 2.69× |
| 整次训练 | 96.906 | 54.760 | 1.77× |

结论：保留初始化并行。计数子阶段的超线性结果包含缓存、调度与运行状态；alphabet/语料构造仍串行约13秒，因此整个初始化加速低于计数阶段。两者merge并发相同，merge时间差异不算初始化并行的直接收益。

### 相对PR的加速比

| 我方配置 | 训练加速比 | merge加速比 | RSS GiB | 相对PR的RSS降幅 |
|---|---:|---:|---:|---:|
| u32初始化1/merge4 | 3.18× | 6.11× | 3.626 | 44.8% |
| u32初始化4/merge4 | 5.63× | 6.87× | 3.390 | 48.4% |
| AtomicU32初始化4/merge4 | 5.90× | 7.16× | 3.391 | 48.4% |

分母基准是PR训练308.111秒、merge188.533秒、RSS6.569GiB。原子版本次略快的原因尚未隔离，不能据此认定稳定收益。

### 热点与优化优先级

以u32初始化4/merge4为排序依据：

| 路径 | 时间 s | 当前判断 |
|---|---:|---|
| 邻边delta、出生记录与路由 | 14.668 | 最大merge阶段；需要细分权重查询、哈希及出生链成本 |
| alphabet与UTF-8语料构造 | 13.068 | 单个coordinator执行，整次训练的重要串行部分 |
| 初始化计数与过滤 | 9.296 | 并行已获明显收益，仍有大量哈希与posting工作 |
| owner频率与posting提交 | 7.696 | 与delta合计约22.36秒，应优先处理 |
| Plan生成、筛选、拼接与排序 | 4.057 | 原型之外的中间存储与多次走访；尚未单独测排序 |
| 实际语料写入 | 0.681 | 当前不是主要耗时 |

优化主线按以下顺序研究；这些是建议，尚未实现或测得收益：

1. 融合有效位置检查与delta生成，减少过滤后再次走访及中间Plan。
2. 减少出生记录的哈希查询、分配与posting安装工作。先细分14.668秒delta及7.696秒commit，避免把权重查询当作全部成本。
3. 并行构造语料，保留canonical初始化与ID分配语义。
4. 探索移除混合规则全局排序，同时重设计相邻selected occurrence识别、权重查询和独占写区组织。排序目前服务多个正确性与存储约束，不能直接删除。

目前已有阶段计时足够确定优化区域，尚无指令级profile或各子函数成本，不能精确断言哪条哈希/读写指令最贵。

### Atomic的内存与默认方案

在当前机器上用Rust直接检查：`size_of::<u32>()=4`、`align_of::<u32>()=4`，`size_of::<AtomicU32>()=4`、`align_of::<AtomicU32>()=4`。现有atomic实现仅把槽位load/store换成Relaxed访问，没有每元素锁或附属数组。两版初始槽数组均为824,359,780字节；峰值RSS分别3.390与3.391GiB。因此当前AtomicU32方案不增加槽数组容量，RSS也基本相同。

当前选型建议：已验证的分区写算法默认使用普通u32，保留Atomic对照分支。依据是独占切片与阶段屏障已满足并发安全，无需额外原子同步；本次数据没有证明非原子更快。这个建议不等于已经合并或删除原子版本。

如果下一版改成原型式共享语料融合遍历，优先评估AtomicU32与Relaxed访问。此时必须重新证明邻边读取、最终出生位置和阶段顺序；Atomic保证单次访问的原子性，不能单独保证整批BPE语义。该方案尚未实现，不能沿用当前分区写算法的测量加速比。

### 后续实验记录要求

每次关键实验完成后，先归档原始结果、配置/提交/输入/二进制来源和签名核对，再同步计算阶段与整体加速比、RSS及余量、系统换页、热点排序、已确认结论、选型建议和未验证方案。进展与最终汇报应从这份记录取数，避免等到追问才补分析。

### 语料构造为何仍串行，以及可行的并行方式

源码核对：`parallel::train` 先串行执行 `compute_alphabet`，之后 `train_in_pool` 又串行遍历字符串计算capacity，再串行解码字符、查只读ID表并push到corpus，同时更新lengths、块边界、pivots/weights和总计数。现有池中执行这一循环不会自动把它并行化。13.068秒的tokenize计时还包含alphabet、池建立和容量计算，不能全归给数组写入。

这是尚未迁移的实现步骤，普通字符路径没有要求串行构造的算法依赖：alphabet完成后，每个保留字符的canonical ID已经确定，填充阶段只查询ids，不创建新ID。独立词的最终数组区间可以分开写，不需要Atomic。

可行方案是先按现有词遍历顺序建立连续词块，在各块并行计算保留字符数；对块长度做前缀和后，一次分配最终corpus，用独占切片并行解码/填充各块，按块顺序合并词边界与权重。lengths的初始活跃标记和计数采用局部结果归并，避免多个worker写同一个ID项。应直接写最终布局，避免每worker构造完整临时ID数组再拼接。

alphabet频率统计也可按worker局部表归并，再保留原字符排序及canonical ID分配步骤。有限limit_alphabet的同频裁剪继承HF哈希遍历细节，改变归并顺序可能改变边界保留字符，必须保持现有行为或对此配置保留既有初始化。非空affix走generic串行路径，其装饰token可能在遍历中新增ID，不把它直接纳入上述并行填充方案。

这项方案尚未实现或测量。优先拆分alphabet、capacity与填充计时；优化后核对实际N/E、权重、vocab与完整merges签名，重新记录初始化时间和RSS。
