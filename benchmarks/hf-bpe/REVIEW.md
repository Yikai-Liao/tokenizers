# 并行迁移独立审查

## 审查范围与结论

本报告由独立审查 agent 完成。只读对照当前 `indexed/parallel.rs`、`aa_parity.rs`、`small_posting.rs`、`indexed.rs`，原型 `efficient_bpe/rust/src/parallel/`，以及本仓库 HF reference 的 `bpe/mod.rs`、`word.rs`。审查期间没有构建、执行测试或运行 benchmark；下面的正确性结论来自源码推导，性能项只描述确定发生的额外工作，不断言其耗时占比。

在普通字符初始化、空 prefix/suffix、非负权重且计数没有溢出的适用范围内，没有发现阻塞正确性的可达反例。原子与非原子版本目前共用规则选择、计划、路由、提交和安全写入算法，可以比较访问方式本身的耗时。当前实现仍与原型有若干明确差异，尤其是全局计划复制、混合规则排序及计划与路由分离；因此这种原子对照不能单独解释迁移版本与原型的速度差距。

### 固定的源码版本

以下行号对应这些 SHA256。生产算法沿用第二轮审查版本；其后仅扩展 `exclusive_writes_use_full_global_addresses` 测试，`parallel.rs` 的文件 hash 从 `e6cd2d2d3f1d1db40b908d42252c5176a70527abe9f177e09a46d182fb342b29` 变为下表值。该测试扩展已单独只读复核。源码变化后应按报告末尾的范围复核。

| 文件 | SHA256 |
|---|---|
| 当前 `indexed/parallel.rs` | `5d2223e96222a72ac344a216d3ebaa6e2fd8d8733a9a0155e81878d9c8a597ad` |
| 当前 `indexed/aa_parity.rs` | `cf5f6307513f6fe1f4aac905bc42b503099bd76d5c741c6da4e2c905504d68a3` |
| 当前 `indexed/small_posting.rs` | `a5d8e66ca0465006c54fc5af197f2f9f2cf7c72c8decef48a1511d474b9ffb7d` |
| 当前 `indexed.rs` | `ace05f9720d1a120866d4586895266c76f24e0b6e6580580e8fabdaaa70f4da3` |
| 当前 `bpe/mod.rs` | `bbce327e126ae424c5491ae86aa48c838e2ea60092eda637935f319185ecc295` |
| 当前 `bpe/word.rs` | `b81cd9c9819c7be9a76b6556fd58f0c1d04fb00992998e224f360aa7706553bd` |
| 原型 `parallel/mod.rs` | `cddf31972e31b72fe8e4e1c9f7cd6c8e3cef4c9cae3339eac75656780d7a9623` |
| 原型 `parallel/aa_parity.rs` | `f8eaaab11cae2550bebc583181a6bad1b7ca53c3a3d943761a9430ce039e8f99` |
| 原型 `parallel/small_posting.rs` | `5707dc55c2ed8e45681adf4b5f06b7a34d1364333a5128cc9a695dfc090b9ce5` |

## 一、阻塞正确性

当前适用范围内未发现阻塞错误。下列两个范围约束必须与结论一起保留：

- 非空 affix 仍由 `indexed.rs:571` 附近路由到串行 HF cohort 引擎；这部分不属于已证明的普通并行路径。
- HF reference 的账本仍使用 i32（`bpe/mod.rs:440`、453、647）。当前并行路径支持更宽的计数，不能声称与 reference 整数溢出后的行为一致。正常逐轮差分须在 reference 计数范围内进行。

## 二、仍存在的性能和迁移差异

### P1：普通 AB 批次新增全局计划复制和混合规则排序

当前 [parallel.rs:788](/root/code/tokenizers/tokenizers/tk-train/src/trainers/bpe/indexed/parallel.rs:788) 先生成 `Vec<Vec<Plan>>`，852–856 再由 coordinator 将每个 chunk 复制进完整 `Vec<Plan>`；混合规则在 859–860 进行整批空间排序。64 位下 `Plan` 为两个 usize，共 16 字节。拼接时，目标完整数组与仍未消费的 chunk 数组同时存在，额外承担一轮串行内存复制和临时内存。

原型 `parallel/mod.rs:629`–685 的普通 AB 路径在检查有效出现后，直接生成 worker 的路由和 4 字节有效 starts；691–729 按任务应用。普通批次没有全局空间排序。当前排序使 `split_at_mut` 的空间所有权和按索引判断相邻计划更容易证明，但它是移除语料原子访问时引入的算法成本。

原子版本也保留这一算法，因此原子/非原子对照只隔离访问方式，无法估算“迁移原型时新增的排序与复制”成本。应把 `plan_ms` 拆出过滤、拼接、普通规则排序、AA 摘要/选择后，才能定位差距。是否改用空间分区、归并有序 posting 或原型任务结构，应依据这些计时决定。

### P1：有效位置检查和 delta 路由分成两次走访

当前 788–850 过滤 posting，909–946 再走访所有选中计划，重新读取邻居和权重并生成 delta/birth。原型普通路径在 `prepare_batch` 内一次完成检查与路由。

这增加一次计划数组读写和阶段边界。当前结构同时实现了统一的相邻计划处理及独占语料写入，不能仅删掉第二次走访而保留现有依赖；应在确认该阶段占比后再改变计划表示。

### P2：所有新 posting 都为保持顺序付出反链和 reverse 成本

当前 flat 出生链仍为 8 字节节点，频率先聚合、posting 按计数一次预分配，与原型对应。区别在 [parallel.rs:1016](/root/code/tokenizers/tokenizers/tk-train/src/trainers/bpe/indexed/parallel.rs:1016)：每个出生链填入 posting 后再 reverse，并按空间输出顺序追加。原型不维护全局 posting 顺序，AA 被选中时另行排序。

当前选择把工作从 AA 轮提前到所有出生 posting，换来单规则过滤和 AA 免排序。是否划算取决于 AA 轮比例、posting 长度与缓存行为，尚未量化。

### P2：静态空间任务与原型动态任务分配不同

当前 flat 初始化 476–496 为至多 workers 个连续空间范围各建一份路由；delta 阶段 908–914 也按空间等计划数切成至多 workers 份。原型初始化和路由使用 worker 私有 buffer 加动态 task cursor。

当前已经消除了第一轮版本“每 65536 个语料位置都创建 workers 个路由 Vec”的分配问题，并保持初始 posting 有序及线性权重游标。仍有负载均衡方式差异：每份计划数量接近，但邻居数量、哈希表增长和 key 分布可能不同。这个差异需要测量；不能据源码断言静态或动态分配必定更快。

### P2：字典路径仍有每块每轮的遍历、容器和复制

新版 [parallel.rs:1083](/root/code/tokenizers/tokenizers/tk-train/src/trainers/bpe/indexed/parallel.rs:1083) 已把出生 posting 的安装交给各字典并行完成，按计数一次预分配；1124–1135 由各 owner 并行追加目录，消除了原来的 coordinator 全量复制和串行安装。

仍会逐轮遍历所有 blocks，为每块创建 workers 个目录 Vec，查找每个输出在该块的出生数据，再复制到最终 local posting。字典的输出 buffer 本身仍用嵌套 block→pair→posting，而非 flat 路径的 8 字节出生链。这是需要处理超出 u32 全局地址范围时新增的实现，不能视为已经完整照搬原型；dict16 块数很多时尤其值得独立计时。默认 flat32 路径不承担这些字典成本。

### P2：部分公开统计字段尚未兑现

并行路径当前没有更新 `stale_posting_visits` 或 `peak_birth_bytes`。`corpus_bytes`、`posting_bytes` 只在 687–688 赋为初始化值，没有计算末尾占用或训练峰值。原型有 stale visits、实际合并数、出生节点与 route 容量、heap 刷新等观测。

现有 `plan_ms` 将过滤、全局复制、排序、AA 摘要和选择合为一项；`commit_ms` 将频率 reduce 和出生链填充合为一项。总阶段时间可用，但字段为零不能说明没有 stale 或出生内存，也不能据此定位峰值开销。建议完善需要用于本次归因的字段，或在 benchmark 输出中明确这些字段的当前语义。

### 其它配置和测量边界差异

| 项目 | 当前版本与原型的关系 |
|---|---|
| pair owner 的混合函数 | 相同混合；power-of-two 使用掩码，与模运算分配等价 |
| AHash | 默认对应；原型另有 Std hash 配置，当前未暴露 |
| heap policy | 当前对应原型 lazy policy；原型 eager 配置未暴露 |
| heap 类型 | 当前 OctonaryHeap，原型 BinaryHeap；属于实现变化，效果未测 |
| token length 表 | 当前 usize，原型 u32；支持更大物理跨度，但元素宽度增加 |
|任务粒度 | 原型 chunk_size 可配，当前阶段分别固定为 4096 或 workers 空间分区 |
| 初始化范围 | 当前包括 alphabet、字符串查询、语料构造和 pool 建立；原型接受 Prepared，相关时长须分开比较 |

## 三、已证明成立的机制

### 精确规则批次、频率和 tie

候选排序按频率降序、pair ID 字典序升序，与 reference 和原型对应。规则冲突条件与原型相同：不允许一个选中规则的 tail 等于另一规则的 head；共享 head 或共享 tail 的不同 pair 不共享实际 token，因而无需额外排除。AA 必须独占批次。

普通 BPE 的 replacement 身份首次激活。每个出生 pair 有一个旧邻边作为 witness：`(L,Z)` 对应旧 `(L,A)`，`(Z,R)` 对应旧 `(B,R)`；相邻两个选中规则产生的 `(Zi,Zj)` 对应旧 `(Bi,Aj)`。出生次数不超过 witness 的旧加权频率，而 witness 与生成它的规则存在 cross-conflict。同频时，含 fresh numeric ID 的出生 pair 字典序排在 witness 后。

所以，若某个 birth 会排在后续旧候选前，它的 witness 一定更早到达堆顶，并使当前批次停止。其频率达到候选 floor 时，witness 也达到 floor；birth 通过有限长度门控时，跨度更短的 witness 也不会被门控排除。这个 witness 条件才支持一次认证整个无冲突批次。仅有“birth 频率不超过生成规则频率”不足以证明批次正确，因为生成规则可能高于后续旧候选。

HF 允许预留低 ID 的 canonical 字符串。当前在非空批次遇到 reserved replacement 时结束批次，并在首次激活 reserved ID 后独占提交（约 726–779）。因此，低 ID 的出生 pair 能在下一轮重新参与 tie，而不会被跳过。普通字符串身份的首次激活证明是这部分以及 flat commit 区分 old removal/birth 的前提；详见本目录 `PAIR_MONOTONICITY.md`。

### 有限长度与相邻合并

当前 926、939 使用严格 `< max_token_length`，与 `Word::merge` 128–143 对新邻边的条件一致；初始化字符 pair 全登记，已选规则本身不因长度逐位置跳过。普通身份唯一决定正文跨度，所以同一个出生 pair 的全部出现接受相同门控结果。

相邻选中计划在 922–934 只扣一次原来的中间边，并直接生成最终 replacement/replacement 边。它与顺序执行时先产生中间 birth、随后删除该 birth 的净效果相同；最终长度门控也相同。

### AA 跨块和混合边界搜索

初始 flat 路由按连续空间块的顺序 collect。每轮 flat 出生链局部 reverse 后按输出空间顺序追加。dict 的计划从 indexed blocks iterator 按块号 collect，块内多规则显式排序。因此当前所有单规则有效 starts 保持空间顺序。

有效 AA starts 的 gap 至少为 token length。于是“从 index i 到末尾全在同一 run”可以由总跨度是否等于边数乘 length 判定，而且谓词单调。`aa_parity.rs:42` 的实现先向左检查附近最多 8 步，再二分定位长尾 run，符合缓存邻近访问与远距离搜索混合的要求。

摘要间的 incoming parity 仅串行处理 chunk 数量；空摘要保留已有 run 状态，跨字典块的连续链不重启。实际 greedy 选择由 workers 处理。没有完整串行 AA posting 扫描。

### 非原子独占写入与阶段屏障

[write_plans:294](/root/code/tokenizers/tokenizers/tk-train/src/trainers/bpe/indexed/parallel.rs:294) 的写入位置仅为 `pos`、`pos + left_len`、必要时的 `after - 1`，全部在 `[pos, after)`。

计划按位置递增，规则冲突检查和 AA greedy 选择保证 `previous.after <= next.pos`。每次切分点取右半首个计划的 `pos`，所以左分支所有写入严格小于切分点，右分支所有写入大于或等于切分点。`split_at_mut` 提供互不重叠的可变切片，`rayon::join` 完成后才进入后续读阶段。

语料写入没有 unsafe 引用别名。AtomicU16/U32 的 `token(&self)` 使用 Relaxed load，`set(&mut self)` 使用 Relaxed store；同样在这些互斥切片内执行。读 snapshot、写入、提交之间都有完成屏障，不要求更强 atomic ordering。

### 地址范围与 u16 token ID

- u16 slot 的 65535 留作 separator。选择条件 `strings.len().max(vocab_size) <= 65535` 包含 special token 和强制 alphabet；此后新增 ID 仅到目标词表大小，预留 ID 仅首次激活。最大活 ID 为 65534。基础 token 数超出范围时选择 u32，包括 AtomicU32。
- flat 路径仅在容量上界不超过 `2^32` 时使用；有效位置转 u32 不截断。
- dict 路径保留 usize 全局地址，使用 `position >> bits` 和块内 offset。u32 offset 没有保留 sentinel，允许 `u32::MAX`；块目录数量另有上限检查。计划及独占写入的地址仍为 usize。
- 字典只是拆分 posting 的存储。全局 `initial_pairs` 统计 owner keys；`initial_block_pairs` 统计 block/pair 条目。后者可因同一 pair 分布在多块而增大，不意味着复制原始语料或增加真实 pair 出现次数。

### SmallPosting 所有权

堆 allocation 只有一个 PackedPosting owner；共享切片仅访问已初始化 Copy 值，可变操作要求 `&mut self`。保存 Vec 的实际 capacity；重分配前 `mem::take`，然后由唯一重建的 Vec 管理旧指针。检查失败或展开不会导致双重释放。u32 两个 inline 值、u16 四个 inline 值都保持当前使用实例的 16 字节结构布局。

### 持久线程池

`train_typed:377` 在单次训练开始建立一次私有池。所有 merge round 复用该池；不是每轮重建线程。线程池建立属于当前初始化计时范围，原型单独记录 pool 时间。

## 四、测试覆盖和最小补充

新版已加入：

- 既有跨 AA 地址块、非均匀权重、有限长度用例的 atomic/non-atomic 和 slot 宽度组合。
- `birth_floor_is_applied_after_all_worker_counts:1258`：约 20000 个计划分为四个空间输出，每份约 5000，局部低于 15000 floor，而全局出生计数高于 floor；该用例覆盖 flat/dict、两种 slot 宽度及 atomic 开关。按源码，确实覆盖了旧两个位置用例触及不到的跨输出聚合条件。
- `exclusive_writes_use_full_global_addresses:1329` 已扩展为额外生成 4096 个有序、相邻的双字符 plans，使用大于 `u32::MAX` 的 base 和四线程私有池。4096 plans 先切为两个 2048，再切为四个 1024，确实进入两层 `split_at_mut` 递归。普通 u16 与 AtomicU16 都执行相同写入，并逐槽对照独立计算的 7/8 交替预期值，检查末尾 separator 保留。**高全局地址下递归写入分支未覆盖项已关闭。**本审查只验证了测试逻辑，未重新执行测试或 benchmark。

仍建议补一个地址范围测试：

**不分配 GB 语料的 u32 字典地址单测。** 为 dict 输出直接生成 `position = 2^32 + offset` 及边界 offset，检查块号、局部 offset、目录与权重；现有长词训练只跨 dict16 的块，未跨 dict32 的真实全局边界。

随机逐轮 HF 差分目前主要通过 `indexed.rs` 的 dict16 配置；flat32 有定向用例，缺少同规模随机配置覆盖。可将现有一部分普通随机 case 再走 flat32，用于验证其出生链反转与 posting 顺序；无需重复全部 affix fallback case。

## 五、修改后必须复核的部分

| 改动范围 | 必须复核的依赖 |
|---|---|
| 初始化路由、worker 动态 cursor | posting 是否仍全局有序；线性权重游标是否仍成立；单规则与 AA 能否继续免排序 |
| born 输出布局、链追加或 reverse | 每个 pair 的位置严格递增；跨输出和跨字典位置顺序；计数与最终 posting 长度一致 |
| 普通 batch 取消全局排序或更换相邻判断 | 中间边只扣一次；replacement/replacement 出生；有限门控；写入区间不重叠 |
| reserved ID 批次条件 | fresh-ID 的 tie 证明能否继续使用；低 ID 出生是否重新参与下一轮选择 |
| Slot 或 atomic 访问方式 | ID 与 separator 编解码；访问类型与所有权；读写阶段屏障；对照是否仍为相同算法 |
| 全局/局部地址表示 | 大于 `2^32` 的地址、最大 local offset、目录索引和跨块左邻 birth |

本轮已解决的首轮发现包括：flat 初始化过多路由 buffer、初始 owner prune/heap 串行、nonflat owner 重复扫描无关 key、字典出生 coordinator 串行复制与安装，以及 heap 计时混入后续空间统计。剩余差异已列在第二节，尚不能宣称迁移与原型全部对齐。
