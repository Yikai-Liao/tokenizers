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

## 2026-09-30：候选A并行构造已实现并测量

计划先提交 `832bf4c9`，候选worktree `/root/code/tokenizers-worktrees/corpus-parallel`、branch `bpe/corpus-parallel`，源码提交 `98ca7fc1`，parent `b6a28768`。新的私有corpus模块冻结原词遍历顺序，按连续词块计长、前缀和分割最终数组、独占并行填充；以局部bitset归并初始ID活跃标记，按词序重建跨块权重。alphabet与canonical分配保持原逻辑，affix走generic原路径。

40项既有库测试和1项新增构造边界测试通过；独立只读审查无阻塞，详见 [CORPUS_PARALLEL_REVIEW.md](CORPUS_PARALLEL_REVIEW.md)。构造依然先串行default初始化最终数组，fill子计时含边界与长度元数据归并；新增临时元数据按词数及regions×ID位图增长，没有额外4N语料副本。

GPT-6 Luna完成一项512MiB关键计时，四个新增计时字段强校验通过。alpha2.322秒、区域测量0.228秒、最终数组分配0.876秒、并行填充及归并2.371秒；tokenize合计5.799秒，初始化16.090秒、merge33.964秒、train55.207秒、RSS3.397GiB、最低MemAvailable4.257GiB、进程VmSwap采样0，系统pswpin147页/pswpout10,164页。core槽/posting容量及四项模型签名均与基线一致，完整来源在 [optimization-a-512.summary.md](results/optimization-a-512.summary.md)。

结论：相对旧非原子4线程版本，tokenize约2.25×，整个初始化约1.43×；但merge从27.457升至33.964秒，整次训练略慢0.82%。merge算法源码未变，差异原因尚未隔离，不能只报局部加速便宣称整次获益。A暂保留候选，下一步从它派生匹配Atomic控制与融合候选，测量实际组合效果。

## 2026-09-30：同期诊断没有复现A的merge回退

用户指出只改初始化却出现merge多6.5秒，要求查清。源码核对显示A与基线从merge主循环开始逐字相同，Trainer路由、compact/AA/posting相同；输入、实际线程、布局、N/E/pairs、批次数、posting访问数与pruned数一致。随后先提交诊断计划/脚本 `94ed8412`、Linux CPU计数修正 `48dcb25d`，GPT-6 Luna连续跑旧版与A各一次。

| 同期配置 | Init s | Merge s | Train s | RSS GiB | 最低可用GiB |
|---|---:|---:|---:|---:|---:|
| 旧非原子初始化4/merge4 | 23.760 | 29.810 | 57.982 | 3.395 | 4.390 |
| 候选A | 15.421 | 29.756 | 49.737 | 3.417 | 4.337 |

此次初始化1.54×、整次训练1.17×；merge只差0.054秒。plan4.275/4.261、delta15.856/15.979、rewrite0.732/0.727、commit8.554/8.431秒，逐阶段均相近。先前6.5秒差异没有复现，具体原因仍未知；缺少先前运行的CPU/缺页记录，不能断言是负载、NUMA、allocator或compiler造成。

新增进程user/sys CPU分别149.061/6.792秒与149.369/5.218秒；minor faults543,902/458,511、major faults0/1；voluntary context switches35,212/35,515、involuntary38,378/37,572。主机busy约48.2%/54.1%，load变化与steal原始ticks单独保存，这些背景值不证明成因。两次进程VmSwap采样0、完整模型签名一致。完整数据见 [optimization-a-diagnostic.summary.md](results/optimization-a-diagnostic.summary.md)。

结论：有同期证据支持A降低初始化及总训练时间；没有证据支持“初始化并行导致merge算法退化”。初次A和此次诊断都保留，避免用后来较好的结果覆盖先前观测。以后对局部改动的整体收益与回退，采用同期基线并保存进程资源数据。

## 正在实现：候选C继续降低初始化串行与哈希成本

用户要求提高A的构造加速比。拆分显示初次A的alphabet2.322秒与最终数组默认初始化0.876秒仍串行；后者在同期诊断降到0.266秒，说明单次墙钟值本身也有变动。

新worktree `/root/code/tokenizers-worktrees/corpus-direct`，branch `bpe/corpus-direct` 从A `98ca7fc1` 分叉。无limit_alphabet时，字符频次不参与筛选，只需出现集合；候选用每worker Unicode位图归并，再保持字符排序/特殊ID分配。启用limit继续原HF频率与同频边界逻辑。只读字符ID查询表替代逐位置UTF-8字符串哈希。最终数组MaybeUninit由独占区域完整写入，完成join与精确覆盖检查后才转换为Vec；避免串行预先写零和第二份语料数组。该局部unsafe转换需要专门边界测试与独立审查。

候选C尚未测量；测试、源码commit、审查与最终计时将补在此处。B融合方案与同构造Atomic控制仍保留，先完成C再选择联合候选的parent。

## 2026-09-30 初始化候选 C

源码 `bpe/corpus-direct/fbdc0b2b` 从A `98ca7fc1` 派生：None alphabet按worker字符存在位图统计，Some仍调用原频率裁剪；每字符只读Unicode→canonical ID直接表；MaybeUninit最终数组由独占worker直接写入，join和完整覆盖断言后原allocation转换，无第二份完整语料。原公开接口与merge源码不变。完整42项测试与独立静态审查通过，证明和计时按用户要求并行。

固定512MiB一次：init9.767、merge26.092、train39.805秒，tokenize1.090秒（alphabet0.347/measure0.187/allocate0.000046/fill0.555），初始route0.573/count8.015秒。RSS3.43GiB、minimum available4.38GiB、VmSwap0，模型完整签名一致。scratch11.57MiB，直接表4.25MiB，返回构造后释放。新增成本很小；当前初始化热点已转为pair计数。与历史结果相比route也发生明显变化，但未隔离first-touch/位置布局/运行环境，不给出单因果归因。原始信息与CPU/fault/host counters见 [C报告](results/optimization-c-512.summary.md)，安全条件见 [C复核](CORPUS_DIRECT_REVIEW.md)。

C选作B起点；从C分叉Atomic控制27dc7fd0保留旧Plan算法，融合候选另分支，分别做一次关键比较。A上的Atomic起点继续保留但未测，不混用其结果。

## 2026-09-30 merge 融合首版 B：回退

`fused-direct-atomic/c1ee2019` 从C Atomic入口27dc7fd0派生，flat非AA一次遍历过滤/delta，连续规则posting任务维持每个出生key有序，join后共享原子端点写入。初始化代码未改，AA/多块/generic不改。43项测试通过，新增1500-case Atomic32逐轮差分及相邻批次/长度/跨worker出生顺序案例。按用户要求取消重复Atomic对照计时。

512MiB唯一测量：prepare22.858、plan0.040、delta23.142、commit7.481、merge31.887、init13.251、train49.697秒；RSS3.42GiB、最低可用4.40GiB、VmSwap0，完整签名一致。Plan排序成本省下但prepare成本抵消收益，C仍最快。相同初始化源码的count变11.370秒，不能声称融合影响了初始化算法。原始信息见 [B报告](results/optimization-b-512.summary.md)。当前不把regression归因Atomic，继续聚焦按规则遍历改变的权重查询局部性与新增selected边界查询。

## 2026-09-30 完整组合稳定性与优化目录

用户纠正：B是失败方案，以B比较不能选最快组合。已完成B/B2三组只留作查询修复诊断（prepare median22.301/11.704秒，CV5.62/3.91%），不作为最终竞争基线。随后C/B2三组交错实际比较端到端，B2每组train/elapsed/merge都更快；train median42.887/34.942秒、CV4.46/.85%，elapsed median47.701/38.965秒、CV4.21/.47%。配对中位比例train.815、elapsed.817、merge.699；等价prewrite median18.928/12.124秒（B2目录建立含在内，fusedprepare不另加）。签名全一致，VmSwap0。现在选B2作为当前完整组合，取代按历史单点选择C。见 [组合稳定性](results/optimization-combination-stability.summary.md)。

用户明确最终目标为发现多个可组合优化并选稳定最快端到端。已提交 [完整清单](OPTIMIZATION_CATALOG.md)，标注正交/依赖/互斥、未融合项、配置条件、失败项和下一队列。测量期间同步编写初始计数D：稳定radix分组替代逐边hash、低频先裁剪、精确预留posting和owner表，公开ID/位置仍u32；从B2分叉d15c18cc。新增route/sort成本、约3.03GiB record+scratch临时量全部计入，性能待测；不能只比较省掉的hash工作。后续具体候选是批量posting填充，省每node push检查和再次reverse，可能同时作用于D安装与owner commit；先保持热点集中。

## 2026-10-01 初始计数D筛选与正交posting组合

D `d15c18cc` 完整45项测试与独立审查通过，固定512MiB screening：train33.265/init9.617/merge19.267秒，route2.563、sort2.045、group2.683、posting install1.037秒，initial count包括后3项及释放共5.764秒。稳定B2 train34.942秒作为当前已证快组合；D单次差距不充分确证更快，继续组合后同期验证。内存暂态成本兑现：RSS4.43GiB、最低可用3.39GiB、VmSwap0；最终posting从1147.872MB到802.968MB、owner估计从276.824MB到138.412MB，corpus不变。重复initial/merge weight_lookup_bytes是同一分配，不能相加。见 [D报告](results/optimization-d-512.summary.md) 和 [初始化审查](INITIAL_RADIX_REVIEW.md)。

同步实现E `35eaf03c`：只在flat commit已预留posting的suffix倒序直接填，len全部初始化后一次发布；由SmallPosting私有模块封装容量/内联/heap/panic所有权。D与E可组合；DE `c8702374` 使用同方法从初始record尾部填sorted posting，再用于commit。明确改动模块为install与commit，而原子、alphabet/构造、radix频率、prepare/rewrite不再改。候选分别有worktree/commit，性能和正确性复核并行。

## 2026-10-01：DE完整组合胜出，停止F/G小收益方向

B2/D/DE九次交错比较与完整模型签名gate完成：train中位35.952/35.442/31.332秒，elapsed40.151/39.740/35.493秒。DE/B2配对train中位比0.8715、三对均快；D/B2中位比1.0002，D独立收益不稳定。DE与D排名在第一对互换，install中位0.931→0.624秒、commit7.023→5.955秒，但差值不足解释整个训练差距。当前推荐实际DE组合c8702374。见 [九次比较](results/optimization-radix-combination-stability.summary.md)。

F 4a2f148a从D派生，46tests与审查通过。D/F一次筛选的prepare11.472→12.062、commit6.362→6.543秒均回退；完整train33.125→32.468秒的改善伴随未改init9.479→7.907秒，因此没有直接模块支持。停止F，DE+F 7504cbfd仅保存源码，未构建/计时。

G1 cbb935b2用join阶段屏障取得安全只读corpus切片；G2 2f238505按128项缓冲端点，纯比较循环由Rust/LLVM自动生成SSE2。全47/48tests与独立审查通过。DE/G2三对prepare比0.9783/1.0195/1.0409，elapsed比0.9572/1.0018/1.0528，无稳定收益；停止G方向。用户要求聚焦大块热点，取消仅隔离小比较kernel的后续microbenchmark，没有执行。详见 [向量化证据与取舍](AUTO_VECTORIZATION.md)。

## 2026-10-01：从实测热点筛选权重快路径H

DE仅一次32MiB perf：1776 cycles:u samples、lost0，WeightLookup占top-IP样本8.16%，prepare worker13.51%、owner commit15.65%；9.57%符号未解析另列。该诊断挑出同时影响初始group与prepare的权重查询，不能将采样占比当作可实现的墙钟收益。见 [热点来源](results/optimization-de-hot-profile.md)。

H worktree weight-one-buckets，branch bpe/weight-one-buckets，00216d91从DE派生。每256槽一个认证bit，任何非1权重区间触及的桶清bit；命中直接返回1，混合桶沿用原精确pivot搜索。初始化已建lookup复用于merge，bitmap额外约100KiB；权重边界静态，不随token端点重写改变。新增测试逐位置对比旧全量搜索，涵盖重复pivot、空区间、256及64桶边界、尾桶、零和u64::MAX权重；完整47tests通过。审查及DE→H原接口512MiB筛选进行中，尚未计入推荐组合。


H独立审查通过，DE→H screen完整模型gate通过：group2.640→0.328、prepare11.320→8.548、train32.860→26.538秒，RSS约4.43GiB均无进程swap；bitmap100,632B增加约98KiB，空间认证桶比例99.104%，语料/posting载荷相同。两个直接模块与全训支持继续三个交错pair；screen与复测分开报告，尚未替换DE推荐。见 [H screen](results/optimization-weight-screen.summary.md) 与 [H审查](WEIGHT_ONE_BUCKET_REVIEW.md)。


## 2026-10-01：H选作新基点；推进commit候选I

用户要求明显改善时停止多组复测、继续优化。已取消第三对，当时p2.de在运行，完成后保留两个完整pair，n=2。H/DE配对group比0.120/0.085、prepare0.687/0.666、train0.821/0.757、elapsed0.834/0.789，完整模型及实际N/E/pairs/语料posting容量均匹配。H train中位25.577、elapsed29.838秒；额外bitmap100,632B、RSS约4.43GiB。选择H源码00216d91作为后续parent，不把小kernel SIMD作为目标。

I源码e3a1954c，worktree commit-direct-assemble，branch bpe/commit-direct-assemble。flat commit聚合来源Group的同时以16B描述项prepend来源链，accepted key直接一次reserve+反向bulk填最终posting，再一次插入ledger/heap；省原来的第二遍逐key ledger查询/逐group bulk调用。全来源floor、owner路由与producer顺序不改，非flat旧路径。新增peak_commit_descriptor_bytes记录所有owner的临时Vec capacity，独立审查与48tests通过，完成H/I各一次关键对照。

I直接commit5.971→6.876秒、train26.187→28.538秒，完整模型gate通过，descriptor容量峰2,097,152B。直接模块与整体均回退，停止I，保留H。J源码376363d2保存每rule/task左右neighbor目录聚合的原型；尚未通过测试和性能筛选。用户要求先详查H成本，因此暂停J推进，不列为已验证候选。


用户进一步要求先形成临时skill，再自主迭代至无明显收益。已在标准skills目录创建performance-optimization-draft，主指引和H/I/G2/DE案例引用，格式校验通过；轻量行为审查后的测量预算要求已纳入。随后用户质疑perf binary debug，readelf确认实际H release没有DWARF line/info。暂停J并补独立release/debug2诊断构建，正式H不覆盖；已有函数/偏移采样保留原证据范围。源码级分析必须核验新binary的地址/行映射后重新采样。

## 2026-10-01：H源码采样与生命周期诊断

同H源码的独立opt3/debug2 binary经readelf和addr2line核验后，仅一次512MiB cycles/cache双事件采样。8,084/7,813样本，lost0；事件分别按period加权，top-IP未解析份额9.55%/9.63%。birth/remove的self周期合计10.81%，其中实际inline hot IP落在route entry的hashbrown桶探测，但函数也包括group和Node操作，不将全部称为hash。RawTable self另为3.40%/7.41%，约96–97%来自reserve_rehash，不能称为纯probe。训练worker与prepare的inclusive份额不相加。详见[debug采样](results/optimization-h-debug-profile.md)。

一次独立H cost/count probe保持原算法，额外group扫描开销653ms单列，诊断不参与排名。计时定位train-init-merge缺口4.780秒，显式post-merge4.777秒，其中owner销毁4.752秒（99.48%），结果字符串转换21.3ms。终点仍有10.57M entry/14.23M heap item。实际birth/remove各173.16M，局部delta group57.84M只是按output去重下限，不等于J实际flush数。模型签名和N/E/pairs/载荷一致，RSS4.43GiB、min available3.33GiB、VmSwap0。见[成本诊断](results/optimization-h-cost-probe.md)。

该证据支持先做K：H派生owner-parallel-drop/288ad858，最后一次commit及所有读者join后，输出持有独立字符串，既有pool并行消费drop各owner。新增一次调度，无unsafe、无额外索引或公开Stats字段。正常释放全部对象，allocator竞争仍须测量。正确性检查和release构建进行中；随后仅H→K一次关键对照，J继续暂存。

K47tests通过后首对清理3.835→2.727秒（-28.9%），但未改merge多1.793秒、train+3.1%。为解决具体方向冲突，仅追加一对反向K→H；清理4.136→2.683秒（-35.1%），train-0.8%但elapsed+0.5%。n2平均清理少1.280秒/32.1%，完整train+1.2%、elapsed+1.1%，未兑现完整收益。停止K，保存两对及[汇总](results/optimization-owner-parallel-drop.aggregate-n2.md)，不扩大allocator/线程矩阵。

J376363d2从H继续，未叠加K。已有inline route probe与173.16M次birth/remove计数支持重复工作方向；output group57.84M仍仅为理论下限。48tests和[独立静态审查](RULE_AGGREGATE_REVIEW.md)通过，native release完成；只做一次H→J512MiB筛选，scratch初始化/LocalGroup/flush全部纳入prepare及完整操作。

筛选完成：[J结果](results/optimization-rule-aggregate.summary.md) prepare8.400→7.723秒（-8.1%），train27.790→24.613秒（-11.4%），elapsed31.762→28.701秒。scratch capacity汇总峰2,106,304B，RSS约4.43GiB、VmSwap0，模型及N/E/pairs/载荷完全一致。未改init少2.011秒，不能把全训3.177秒都归因J；n=1只作为当前筛选，未保证稳定复现。当前选择J，不叠加K，不追加多组。

最后仅离线复用原完整H事件，对实际owner commit符号的15个热点IP批量addr2line；root另以30地址核对并用有界objdump确认频率load、control-byte probe、posting长度与Node-next读取/逆序循环。closure direct self18.94%cycles/15.76%cache，但临时born聚合未单独定量，通用cache/skid与CPU→wall限制保留，见[具体操作](results/optimization-h-debug-owner-commit.md)。原callchain export与全15,897条IP记录逐项event/period一致，inclusive份额只在此完整覆盖依据下保留。

本轮停止：剩余具体大项涉及账本访问/更新、posting链读取和正常释放；I/K简单方向未兑现收益，J之后未找到代价明确的下一项大幅局部改善。保存全部分支、原始结果、正确性证据与复现入口。没有宣称全局最优，也没有把尚未实现的布局或allocator改造计为收益。

## 2026-10-01：J扩展、posting对象统计与原Rust粗略对照

用户追加三个报告，当前J源码保持376363d2。仅benchmark runner支持实际1/4 workers，默认4，保留旧binary与源码副本。同J512MiB单线程→四线程各一次：init19.027→5.547s，merge46.630→13.528s，train70.539→23.626s，feed+train74.307→27.541s。模型、工作量和resource gate通过；[完整阶段与限制](results/rule-aggregate-scaling.summary.md)。生产feed已有并行路径，本benchmark显式关闭feed并行以沿用历史口径。

独立对象诊断一次：10,572,128 terminal entries里7,274,631个posting堆分配、3,297,497个inline；堆分配61.13%容量≤8个u32。新增191ms扫描不参与排名，模型gate通过。此前H post-merge99.48%在owner drop；本次计数支持小对象释放机制，尚未单独计时map扫描和allocator。[分配统计与补丁](results/j-posting-inventory.summary.md)。

原efficient_bpe当前选中Rust ebpe 8eb3cc6c，same16MiB prefix逐行权重，4线程/max40,759rules/min2，各一次：J train1.477984s，Rust call2.155127s，约1.46×。按用户要求不查模型一致，N/E/rule数/posting visits对齐；J字符前端/输出构造与Rust Prepared输入、LTO等profile差异明示。[粗略对照](results/efficient-rough.summary.md)。原仓库未编辑。

## 2026-10-01：全量arena实际内存代价

用户要求把退休posting也保留到底实际测峰值。独立J副本添加纯Rust bumpalo3.20.3、每专用Rayon worker TLS arena，所有指针存活至train_typed返回，owner/任务结束后统一broadcast释放，原算法不变。16MiB smoke gate通过，随后512MiB同期标准J→Bump各一次：HWM只多936KiB（4.4337→4.4346GiB），train24.351→19.126秒。累计payload1.307GiB/live.754GiB，退休buffer保留567.155MiB，arena chunks实际1.860GiB/51块，growth0，最终release65.341ms。模型、操作量及资源gate通过。诊断统计开销单列，不将n1当稳定排名或推广其它库配置；[完整报告](results/j-bump-retain.summary.md)。

用户继而要求独立subagent推导posting/阈值随Wikipedia规模、merge数量、语言、去重权重、物理数据量和片段长度变化的关系；已明确授权独立分析。并强调只需回收影响全程峰值的部分，初始化临时工作集释放后可允许更大arena保留。当前全量方案已经满足资源门槛，不预设必须有规模增长超参；继续数学推导、跨语言固定规则诊断与阈值敏感性分析。
