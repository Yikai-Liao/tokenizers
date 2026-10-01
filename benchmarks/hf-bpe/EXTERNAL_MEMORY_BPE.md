# 小 RAM 上的精确贪心 BPE：外存架构与研究边界

独立调研日期：2026-10-01。本文从加权动态图收缩、外存优先队列、缓冲字典、倒排索引和压缩域计算重新审视问题。它补充 [SCALE_UP_ANALYSIS.md](SCALE_UP_ANALYSIS.md) 和 [ALGORITHM_FRONTIER_MAP.md](ALGORITHM_FRONTIER_MAP.md)，没有修改 Trainer。参考工作树在本次源码读取时为 `add44e14d793c31a7595bb40f8321d0ec038a6aa`；容量情景仍取已归档的 `56dd3227` 512 MiB JSONL，而不是假定当前源码已完成大规模训练。

**可实现的近期路线是：外部精确去重与排序初始化，磁盘上的有序 posting，保留现有 4N endpoint corpus 表示并增加受预算控制的页缓存，最后按 K/H 的真实容量决定是否把 ledger/queue 也移到磁盘。** 只有 bounded 初始化仍不足以在小 RAM 上训练数十 GiB。完整逐规则扫描提供简单的精确兜底，但其磁盘流量随规则数增长；RLZ/grammar 上直接计算则是更远的、依赖重复性的研究路线。

## 1. 应抽象成什么问题

预分词后有加权路径集合 `(s_i,w_i)`。路径顶点是 token span，边是相邻 token 的 canonical pair。每轮必须选择加权边质量最大的 pair，按 `(left_id,right_id)` 升序打破 tie，再在每条路径内从左到右收缩所有不重叠出现位置。收缩删除旧邻边、产生新邻边，并改变下一轮的全局统计。

因此任务是 **动态加权邻接关系上的批量收缩，加上严格全局最大值选择**。它同时含两种索引：按 pair 查位置，按位置查当前邻居。磁盘排序解决前者的构建，页缓存解决后者的访问；单独替换 hash map 或 heap 没有覆盖整个任务。

沿用现有符号：`R` 原始字节，`U` 唯一片段数，`N_s` 含 separator 的 slot 数，`E_u` 初始物理边数，`E_0` 初始加权边质量，`K_t` 活跃 pair 数，`Q_t` pair×地址块目录项数，`H_t` queue 记录数，`P_t` 保存的历史位置数。另设 `r` 已执行规则数，`m` 实际物理收缩总数，`Z` 生成的 ledger 更新记录数，`J` 所有候选验证所访问的 posting 数。`r`、`m`、`E_0` 三者不能互换。普通无非空 affix 路径中，`m≤E_u`，总物理 birth≤`2m`，故保存全部历史 posting 的逻辑上界是 `P≤3E_u`；这是表示上界，不含目录、扩容、临时文件或 allocator。

原始输入可以顺序读取，下一条规则却依赖已完整读取并更新的全局状态。一次不可重放的 raw stream 通常不能在末尾到来前决定第一条精确规则：两个拥有同一前缀的输入，后缀可以让不同 pair 成为唯一最大值。可重放文件、保存后的 corpus 或足够的充分统计是必要的持久状态。

## 2. 精确契约决定哪些研究能转移

**最终标准是与原始 HF 实现在同一输入和配置下的训练结果对齐，包括 vocab、merge 次序和 model signature。** AA 专门化可以自由改变内部表示、遍历或调度，只要输出一致；无需满足文献中所有 Re-Pair 变体。下表用于识别影响 HF 输出的条件，不额外引入新的训练目标。

| 契约 | 外存实现必须满足 |
|---|---|
| weighted dedup | 完全相同的预分词片段才合并权重；hash 碰撞必须核对原文。重复文本只增加权重，不增加物理位置。 |
| global frequency/floor | 所有 shard 的计数先相加，再做 floor。局部低频与局部 top-k 均不能提前永久删除。零权重位置仍参与物理训练状态。 |
| full priority | 使用 u64 frequency 和完整 u32/u32 canonical key；磁盘 comparator 与内存 comparator 一致。 |
| AA | `AAAAA` 对 `AA` 的候选计频是 4，而规则执行合并 2 次；排序/页边界不得重置左到右的 overlap 状态。 |
| identities | special/reserved ID、首次激活、token 字符长度、请求 vocab 与 ID 上界分别处理；“每轮新分配数字 ID”不是现有证书的全部前提。 |
| visibility | 规则 t 的最终 births、deaths、corpus 状态在规则 t+1 的精确选择中可见；落盘时机可以延迟，逻辑可见性不能延迟。 |
| affix alias | 非空 affix 可使旧 pair 增频，当前一般 cohort/serial fallback 必须保留。若它的容量也超界，需单独的外存一般 ledger；不能静默应用普通单调算法。 |

现有普通路径的旧 pair 在首次完整 birth 后只会减频，低于 floor 后可永久退役。证明与反例见 [PAIR_MONOTONICITY.md](PAIR_MONOTONICITY.md)、`indexed.rs` 的 `suffix_alias_really_increases_an_old_pair`。这是可复用的项目证书；外部文献的“exact RePair”不自动等价于当前 Trainer。

## 3. 直接相关的 Re-Pair 研究：可行性与真正的限制

### 3.1 精确外存 Re-Pair 已有理论结果

Köppl 等的 *Re-Pair in Small Space*（Algorithms 2021）§6 明确研究外存构造。其 Theorem 4 给出

```text
min(4 Sort(n), (r·n/M) Scan(n) + Sort(n) + O(n log_(M/B) 2))
    + r Scan(n)
```

的 I/O 上界，`r` 是论文的 turns。heap 方案将全局统计放在外存，仍逐规则扫描文本完成替换。这给出结构可行性及扫描成本，未消除逐轮文本访问。论文 §1.3 对 AA 使用不重叠计频 `floor(n/2)`，tie 任意；当前 HF 使用相邻计频 `n−1` 与 canonical tie。因此不能直接引用这个 theorem 为当前加权 Trainer 的复杂度证明。扫描/排序框架可适配，计数、weight 宽度和更新次数的摊还证明需重做。[作者 PDF，§1.3 与 §6](https://koeppl.github.io/bin/paper/algorithms21repair.pdf)

### 3.2 少内存仍可能保留 O(N) 物理文本

Bille、Gørtz、Prezza（DCC 2017）提出低工作空间 Re-Pair 算法，在可重写的 n-word 文本之外再使用约 `(1+ε)n+√n` words，或以更多计算换约 `n+√n` words。它适合减少 RAM 常数，输入与工作数组仍随 n 线性增长；不能证明任意 100 GiB 输入可在固定 8 GiB RAM 中完成。其 high/low frequency 分阶段思想值得参考，但 weighted frequency 很大时，不能用 frequency 下界推导相同的物理 occurrence 下界。[Space-Efficient Re-Pair Compression](https://arxiv.org/abs/1611.01479)

### 3.3 直接在压缩表示上训练是另一条路线

Sakai 等（DCC 2019）把任意表示 T 的 grammar 重构为 RePair(T)，避免完整展开。其空间依赖压缩 grammar 大小 n、输出变量数 m 和 `log N`，给出 `O(min(N,nm log N))` 空间及相应时间界。这说明精确贪心并不必然要求显式保存每个位置，但该 bound 在不够重复的语料上仍可退回 O(N)，也需要重做 HF 的 ties、重叠计频、pretoken separator 与权重语义。[RePair in Compressed Space and Time](https://arxiv.org/abs/1811.01472)

2025 年 RLZ-RePair 预印本进一步在 reference+phrase 表示上做规则替换，目标是保留 exact RePair grammar。作者自己说明：实现依赖 heap 的 tie 策略与 unordered occurrence 遍历，可能产生不同 grammar。可转移的机制是“相同内部串共享一次物理更新，边界贡献显式记账”，而不是直接采用其模型输出。当前完整片段去重已经是最简单的共享；RLZ 可共享不同片段的内部重复，收益应由 parse/reference 大小与 boundary 更新量证明。该工作尚不能当作 HF 精确 Trainer 的现成实现。[Efficient Grammar Compression via RLZ-based RePair，2025 preprint](https://pmc.ncbi.nlm.nih.gov/articles/PMC12330530/)

BigRePair/Re²Pair 先做 prefix-free parsing，再分别对 dictionary/parse 构造 grammar；改变局部构造顺序。它们展示了高度重复集合的容量实践，但没有提供当前 canonical 全局贪心序列的等价保证。可用来构造前一段研究路线的压缩输入，不能直接取代训练规则。[Re²Pair 的算法描述](https://pmc.ncbi.nlm.nih.gov/articles/PMC11275962/)

## 4. 全局 max：外存队列可以精确，但有代价

| 一手研究/实现 | 给出的机制与边界 | 在 BPE 中的转移 |
|---|---|---|
| [Arge buffer tree，1996](https://www.brics.dk/DS/96/3/BRICS-DS-96-3.pdf) | 树节点缓存批量更新，将大量小操作变成块传输。 | 可缓存 signed count delta 和 posting 元数据；立即查询必须看到路径上的缓冲消息。每轮 flush 整棵树会失去收益。 |
| [DecreaseKeys are Expensive，STOC 2017](https://arxiv.org/abs/1611.00911) | external PQ 支持 decrease-key 有不同于只 Insert/ExtractMin 的下界。 | 不能把任意动态优先级更新当作免费排序。此下界针对其操作模型，不是 BPE 的端到端下界。 |
| [Jiang–Larsen，2018](https://arxiv.org/abs/1806.07598) | 带 decrease-key 的期望摊还 I/O 改进。 | max 可反向 comparator；HF 旧 frequency 下降对应 min-priority 上升，并非原生 decrease-key 的直接同向操作。任意更新须核对 API 或用 delete+insert。 |
| [Iacono–Jacob–Tsakalidis，ESA 2019](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ESA.2019.60) | 带可选参数的更新、提取和空间 tradeoff。 | 更适合一般动态图、alias 路径的长期研究；tradeoff 必须同时计提取成本，不能只摘最便宜的 update bound。 |
| [Brodal 等，ESA 2025](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ESA.2025.5) | Insert 摊还 `O(1/B)` I/O，DeleteMin 摊还 `O((1/B) log_(M/B)(N/B))`，只需 `M≥2B`。 | 普通路径的 lazy upper-bound 候选可以避免原地 decrease-key，更接近其接口；真正 ledger 验证与 corpus I/O 仍另算。 |
| [Lazy B-Trees，MFCS 2025 作者页及 erratum](https://www2.wild-inter.net/publications/rysgaard-wild-2025) | 查询驱动的局部排序/分区，连接 dictionary 与 PQ；作者纠正了 pointer maintenance 的漏算。 | 不采用 conference 版宣称的有利 decrease-key bound。作者说明维护全体 element pointer 时，直接实现的 delete-min 可升到 `O(log N)` I/O；只作为需重新核对的研究候选。 |
| [STXXL priority_queue，固定 1.4.1 文档](https://stxxl.org/tags/1.4.1/classstxxl_1_1priority__queue.html) | 提供磁盘 priority queue 与 push/top/pop、配置和内存核算接口。 | 可作为行为/布局原型参照；C++ 依赖、比较器 sentinel、缓冲预算及 Rust 边界仍有集成成本，本文未引入它。 |

### 普通路径可继续使用保守上界

对每个已发布 pair 保存候选 `upper(q)`，满足 `upper(q)≥true_count(q)`。旧 pair 只有减频，可以延迟修改 queue。出生必须等全部 shard 汇总，按完整 final frequency 入队。取 queue 最大项后，从 ledger 读取包括所有待处理 delta 的真实 count：若已退役则丢弃，若下降则以新值重插，只有该项在完整 priority 上仍胜出时才执行。所有候选保留 canonical key，因此相等 frequency 时也能比较正确。

这个证明允许物理 compaction 落后，不允许新 births 暂藏于未参加选择的 memtable。重复 queue records 若采用版本号，generation/tombstone 也必须持久化并计入 record width。普通无再激活证书下可以维持每 pair 一个候选，pending decreases 不必逐条产生新候选；alias 路径需要在每次可能增频时发布上升的候选，或采用真正精确的动态 queue。

`Insert` 和 `ExtractMax` 的 I/O 优良，不代表每一次随机 ledger Get 或每个 pair 的 occurrence 验证也享有同一 bound。把精确全局选择做成每个 shard 的局部 top-k，还需证明未报告项不能赢；通常要保留精确上界并进行追加查询，单纯固定 k 没有这个证书。

## 5. 缓冲字典与倒排索引：应用机制，不继承错误语义

### 5.1 Signed delta 适合 buffered upsert

普通收缩改变常数条邻边，可发出 `(pair,epoch,signed_delta)`，按 pair 分组相加。在允许范围内 signed sum 具有可合并性，但中间算术要 checked；超过原有 i64 mass 不能通过改用磁盘而隐式放宽。读取 pair 必须包含当前 epoch 之前的全部消息。

Bε-tree 的 upsert 在查询时收集路径上的消息，可避免每次计数更新先随机读取。2022 年研究给出具有 worst-case 更新成本的外存字典变体：对固定 ε，更新 `O(B^{ε−1}log_B N)`，点查 `O(log_B N)`；这是字典界，不是 max-query 界。count→priority 的二级索引若需要先读旧 count，可能重新引入昂贵点查，ordinary lazy queue 正好避开这一需求。[External-memory dictionaries with worst-case update cost](https://arxiv.org/abs/2211.06044)、[Bε-tree upsert 的原始实践介绍](https://www.usenix.org/system/files/login/issues/login_oct15_issue.pdf)

RocksDB Merge Operator 可以把增量与 base value 在读取或 compaction 时合并，适合 exact counter delta。对符号宽度和错误输入必须定义自己的 checked merge，不能复制示例的无检查 u64 wraparound。把一个热 pair 的全部 positions 放在一个不断 concat 的 value 中会在读取时形成巨型对象；应拆成 `(pair,segment)` keys 和可流式消费的 payload。[RocksDB Merge Operator](https://github.com/facebook/rocksdb/wiki/Merge-Operator)

### 5.2 Posting payload 与字典分离

WiscKey 的核心机制是只在 LSM 中移动小 key/指针，大 value 在独立 append log 中保存。这里最适合放在 log 中的是有序 posting extent；LSM 维护 pair 的 frequency、extent 指针和 epoch。value GC 需知道退休 extent 与所有 reader 的生命周期；峰值还包括新旧 extent 同时存在。论文的 KV benchmark 不能转为 BPE 收益。[WiscKey，FAST 2016](https://www.usenix.org/system/files/conference/fast16/fast16-papers-lu.pdf)

Lucene 的 immutable segments、term dictionary→posting 指针、segment-local IDs 与 global base，和现有 u32-local posting/64-bit base 结构很接近。借用其磁盘表示，可以把 positions 存为按 pair、address-block、offset 排序的 extent，并用 sparse block index/skip 信息跳过不需要的范围。Lucene 的删除是 document 语义，BPE 的边失效与出生是动态 adjacency；不能只替换 docID 就假定其统计保持正确。[Lucene segment 说明](https://lucene.apache.org/core/9_12_0/core/org/apache/lucene/index/package-summary.html)、[Lucene912 posting 格式](https://lucene.apache.org/core/9_12_2/core/org/apache/lucene/codecs/lucene912/Lucene912PostingsFormat.html)

对 ordinary pair，初始/首次出生后不会有未来新增 occurrence，因此其 posting extent 可一次建立、随后以 corpus 验证和最终退休处理失效。一个 birth epoch 内的跨 shard 分段应在 barrier 后整体发布。这比为每个失效位置立即产生 delete tombstone 更适合现有算法；若要清理冷 extent，则显式建 validity bitmap 或重写 extent 并核算额外 I/O。

### 5.3 总 RAM 预算必须覆盖隐藏缓存

RocksDB 的 RAM 包括 block cache、memtables、index/filter blocks、iterator pinning；OS page cache 还可能保存另一份数据。只设置 block cache=2 GiB 不能证明 RSS≤2 GiB。各列族共享预算、限制 iterator fan-in、计入 compaction/decompression scratch，必要时显式管理 I/O/cache。[RocksDB memory accounting](https://github.com/facebook/rocksdb/wiki/Memory-usage-in-RocksDB)

## 6. 图系统的可转移部分

| 一手系统 | 借用的机制 | 当前任务的限制 |
|---|---|---|
| [GraphChi，OSDI 2012](https://www.usenix.org/conference/osdi12/technical-sessions/presentation/kyrola) | 图分 shard，parallel sliding windows 把随机边访问变成较少的顺序区间。 | BPE 路径天然有空间顺序，可做 address-shard 读写；GraphChi 的异步迭代不等价于 BPE 规则的精确全局排序。 |
| [X-Stream，SOSP 2013](https://www.cl.cam.ac.uk/~ey204/teaching/ACS/R244_2021_2022/papers/roy_SOSP_2013.pdf) | edge-centric scatter/gather，按 streaming partition 做顺序读取、更新日志。 | 适合稠密早期规则或重算基线；每条稀疏后期规则都流过全图会产生 `r·Scan(N)`。不能引用其 graph benchmark 预测 Trainer。 |
| [GridGraph，ATC 2015](https://www.usenix.org/conference/atc15/technical-session/presentation/zhu) | 两级分块与 active-block selective scheduling，跳过没有活跃更新的块。 | posting 已经指出选中 pair 的活跃地址块。可据密度选择点页、区间或全 shard；需要跨块 AA 与邻居状态一致。 |
| [GraphMP，2018](https://arxiv.org/abs/1810.04334) | semi-external 模式与 selective scheduling/cache。 | “顶点状态全驻 RAM”在本任务意味着 N slots 仍驻 RAM；这里只能参考缓存机制，不能直接宣称解决 4N 超 RAM。 |

这些系统主要改善访问组织，不解除每条 BPE 规则之后的全局依赖。worker 可以并行处理同一条规则在不同路径/shard 的出现位置；不同规则只有已有的安全 batch 证书允许共同处理。预取多个可能候选的页可以保留，预先执行未证明独立的规则会改变训练结果。

## 7. 建议的三阶段架构

### A. 外部预处理与初始化

1. 完全复用既有 normalization、pretokenization、alphabet filtering 与 newline 边界。预分词器若会输出无界长片段，外存接口应支持连续 chunk 和 carry，不能用不够大的字符串缓冲来改变边界。
2. 在磁盘进行 exact dedup：按完整片段键排序并相加权重，或 hash partition 后核对完整字节。保存稳定 word/span ID 和权重；处理 alphabet/special tokens 的确定性顺序。片段跨 run 不等于新的片段。
3. 生成 4-byte endpoint slots 文件和 u64 global base+u32 local offset 的地址目录。保留现有 token length 表，因此不需要每个 token 两条 64-bit linked-list pointer。separator、pivots/previous_weight 和跨块边沿用现有语义。
4. 用预算内 run 生成和有限 fan-in 归并构建 `(pair,global-position)` 序列。混合权重若附带显式 u64 weight，full record 是 24 B；16 B record 必须能从已驻留/可顺序连接的 word metadata 恢复 weight。初始 ID 满足条件时的 8 B record 是另一个局部证书。
5. 对同 pair 完成全局频率汇总后才做 floor，然后流式写 retained posting extents 和 `(pair,block)` 目录。一个热 pair 的完整 group 可以大于 RAM；group 需流式消费，不先收成 Vec。若 floor 决策前 positions 不能丢弃，可暂存该 group 或第二遍读取排序文件。所有 runs、Q directories 和 producer summaries 均不能一次 collect 到 RAM。

STXXL 的 run creation/streamed merge 可作为外部 sorter 参考；DuckDB 2025 sort 使用更新后的 spillable page layout，说明成熟实践已将可换出表示与排序 kernel 分开。这里可直接比较 `(pair,position)`，无需依赖 sorter 的 stability。对 raw string dedup 而言 record 长度可变，不能直接套 pair 的 fixed-width 体积。[STXXL sort 文档](https://stxxl.org/tags/1.4.1/design_algo_sort.html)、[DuckDB 2025 sorting redesign](https://duckdb.org/2025/09/24/sorting-again)

### B. Semi-external 训练：先让 N/P 离开 RAM

pair ledger 与 candidate queue 在 RAM，corpus 与 posting payload 在磁盘。每轮取精确最大 pair，按 global position 增序流式读 posting；按地址范围预取所需 corpus 页，验证当前 pair/长度并执行左到右收缩。在页缓存中保留修改后状态，产生 checked local delta/birth runs，全局聚合后再发布下轮。

endpoint 表示每个实际收缩只访问常数个 slot，但它们可能位于多个页：左端、右端、前一个 token 的尾/起点、后一个 token 起点、新 span 的尾。按完整 token length 查页，不能假设只需相邻物理页。长 token、跨 shard AA 与 separator 需要显式 carry 或协调。对密集位置，读整个 address shard 往往更便宜；对稀疏位置，读取有限页/区间。页缓存可淘汰 dirty 页，但每次读取必须得到当前 epoch 的逻辑数据。

此阶段仅在 K/H/必要目录能驻 RAM 时成立。现有 physical block 大小 2^16/2^32 slots 是地址编码选择，不适合直接当 4 KiB I/O 页或缓存 shard；新增这些层应保持地址身份，避免把 I/O 重分块误当 posting ID 重编号。

### C. Fully external ledger/queue：针对 K/H 超预算

frequency/目录移到 buffered dictionary，posting extents 单独保存，queue 用真正外存 PQ 或有精确上界的多层候选结构。每条规则的 delta/birth 可分块写入，候选验证查询必须读到该轮的所有未 compaction 消息。**逻辑 barrier 不要求每轮全局 fsync/compact**；同一过程可在 cache/memtable 中观察完整更新。若要求恢复训练，另以 epoch manifest/WAL 管理 corpus、ledger、births、rule log 的一致性，恢复时只能公布完整提交的 epoch。

一般 affix fallback 暂留现有实现，并给出清楚的容量 gate；若用户必须让这类输入也完全外存，应另实现一般 cohort 状态的外存版本。普通 compact 的 `P≤3E`、永久 floor 剪枝和 lazy upper bound 都不自动覆盖它。

## 8. 参数化 RAM、磁盘与 I/O 模型

### 8.1 RAM

```text
M_total ≥ M_vocab + M_ledger_resident + M_queue_resident
        + M_corpus_cache + M_posting_buffers + M_sort_buffers
        + M_directory_cache + M_delta_buffers + M_decode/compaction
        + M_threads + M_OS_headroom

semi-external logical payload: M_ledger ≈ a·K, M_queue ≈ h·H
fully-external:               resident ledger/queue caches bounded by budget
```

`a=32 B`、`h=16 B` 只是选定 dense representation 的例子；当前 hash bucket rounding/control 与 queue capacity 需另加。若 `H=K`，10M/100M/430.85M 的逻辑 48K payload 分别为 0.447/4.470/19.261 GiB；不能把 K 从 512 MiB retained count 线性外推超过 `V_0²`，也不能用 K 当 H。

一个**预算配置示例**是总 16 GiB：8 GiB corpus cache、2 GiB sort/validation、2 GiB ledger+directory cache、1 GiB queue、1 GiB delta/decode/worker、2 GiB OS/headroom。各阶段可复用预算。它是外存实现需要遵守的 allocation budget，不是当前 Trainer 的 RSS 预测，也不保证此 cache 足以获得好吞吐。保留原 `M_feed` 的 word-string map 会破坏 fixed-RAM 目标，必须实现前述外部 dedup 和生命周期释放。

### 8.2 磁盘容量

设 `θ` 为初始保留 posting 比例，`p` 平均每条 position 的压缩后字节，`d` 每目录项字节，`a_d/h_d` 为磁盘 ledger/PQ record 字节，`S=sE_u` 是完整排序 payload。

```text
D_live ≈ 4N_s + M_wordmeta_disk + p·P + d·Q + a_d·K + h_d·H
       + D_uncompacted_deltas + D_WAL + D_vocab

D_init_peak ≈ D_raw + D_dedup_staging + 4N_s + M_wordmeta_disk
            + 2S + p·θE_u + d·Q_0 + a_d·K_0 + h_d·H_0 + overhead
```

`2S` 是旧 runs 与 merge 输出同时存在的保守配置；可边消费边回收降低峰值，但需要具体 sorter 证明。压缩 `p<4` 的收益由 gap 分布决定；一个热 pair 的差分位置可能很小，稀疏 pair 的 header 反而更大。历史 `P≤3E_u` 可因及时退休显著减少，也不能直接当实测。descriptor/LSM/extent GC 会增加新旧副本。

### 8.3 I/O

以固定宽 record 为单位，`Scan(n)=Θ(n/B)`，`Sort(n)=O((n/B)·max(1,log_(M/B)(n/B)))`。I/O 页字节 b 与 record width s 的关系是 `B=floor(b/s)`。RAM run 容量与 page cache、PQ 缓冲不能同时重复使用同一份 M。

初始化包含 raw/dedup 的排序/扫描、corpus 写入、`Sort(E_u)` 和 posting 输出。若有 L 轮完整外部归并，sorting 流量约 `2(L+1)S`：生成 runs 一读一写，随后每轮一读一写。跨多个设备的并行 I/O 不等于免费流量。[Vitter 外存算法原始综述](https://users.cs.duke.edu/~reif/courses/alglectures/vitter.papers/Vit.IO_survey.pdf)

训练可分解为：

```text
I/O_train = I/O_posting(J,p,b)
          + Σ_t (corpus_page_misses_t + dirty_page_writes_t)
          + I/O_delta_dictionary(Z)
          + I/O_queue(I inserts, X extracts/revalidations)
          + I/O_extent/directory_compaction
```

如果完全冷缓存、每条稀疏收缩或 posting 验证都碰独立页，corpus I/O 可以接近 `O(J+m)` 页，而不是 `O((J+m)/B)` 页；它会把几字节更新放大到几 KiB。若同一轮热点覆盖 h_t 个页，且缓存/调度只载入一次并最终写一次，则请求量接近 `2b·h_t`。应该实测每轮 unique touched pages、重载次数、dirty eviction 与 gap locality，再选择预取页/区间/shard。仅按 `m` 乘 slot bytes 会低估磁盘成本。

简单 streaming 重算每轮执行 `Sort(current_edges)`，并读/写 compact token stream，约 `Σ_t Sort(E_t)+Σ_t Scan(N_t)`；成熟的增量 scan-based 方案可减少统计重算，但仍有逐轮 corpus 访问。用 buffered delta 逐轮外排序时，也要计 `Σ_t Sort(Z_t)` 与每轮小 run 的固定开销，不能未经证明改写成单次 `Sort(ΣZ_t)`。

条件时间下界为 `max(D_read/β_read,D_write/β_write,Q_random/IOPS,C_cpu/ρ_cpu)`，设备共享读写带宽时应合并约束。顺序速度、随机 IOPS、访问依赖和 queue-depth 均应从目标设备实测，不由论文 benchmark 或本项目 RAM 计时推算。

## 9. 32/100 GiB 情景：先检查磁盘，再谈吞吐

由 [容量脚本](results/external-memory-bpe/capacity_model.py) 和 [JSON](results/external-memory-bpe/capacity_model.json) 重现。假设和已有 512 MiB 中文 none 输入具有相同 `N_s/R`、`E_u/R`、`U/R`，u32 corpus，初始 positions 全保留，mixed-weight word metadata 逻辑 12U B。不假定 K/H/Q 随 raw 线性增长；这是条件算术，未经大样本实测。

| raw GiB | N_s 十亿 | E_u 十亿 | 4N corpus GiB | 4E 初始 posting GiB | 12U metadata GiB | corpus+posting+metadata GiB |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 13.190 | 13.007 | 49.136 | 48.454 | 1.023 | 98.612 |
| 100 | 41.218 | 40.646 | 153.549 | 151.418 | 3.196 | 308.164 |

| raw GiB | 16E sort payload GiB | 24E sort payload GiB | 保守 init 峰值，16 B records GiB | 保守 init 峰值，24 B records GiB | 3E 历史 posting，4 B/位置 GiB |
|---:|---:|---:|---:|---:|---:|
| 32 | 193.816 | 290.723 | 518.244 | 712.059 | 145.362 |
| 100 | 605.674 | 908.510 | 1619.511 | 2225.185 | 454.255 |

init 峰值列包含 raw、corpus、metadata、初始全部 posting 与 2 份完整 sort payload；**另需增加** dedup staging、K/H/Q、WAL、headers/allocator。16 B record 假设 weight 已可恢复；24 B 是显式 `(u64 pair,u64 position,u64 weight)`。顺序 count-first/two-pass 和更窄初始 ID records 可以降低 staging，不能把该表当最小容量。重复率更高的输入可大幅降低 N/E/U；不同语言、预分词和 alphabet 也会改变密度。

把每条规则都扫固定 4N endpoint 文件作为量级警示：30,000 规则仅 corpus 读取就约 1,439.5 TiB / 4,498.5 TiB；若每轮一读一写则约 2,879.0 / 8,997.0 TiB，尚不含计数排序。compact stream 会随收缩变小，实际应使用 `Σ4N_t`，这个固定长度情景没有预测最终运行时间。它解释了为什么后期稀疏规则必须通过 postings/selective scheduling 避免全扫。

本机该文件系统调研时仅约 58 GiB 空闲；这个条件下即使 32 GiB 情景的 corpus 能勉强容纳，corpus+posting 和排序 staging 也不能同时容纳。没有尝试大规模文件分配。

## 10. 预研交付与未来验证顺序

按最新任务范围，外层方案与外存路线仅做预研。交付包括此文、一手来源清单与可复算容量模型；本文没有实现外存 Trainer、paged endpoint、disk ledger/PQ 或恢复协议，也没有完成它们的 HF 差分和大规模性能验证。以下是将来决定实现时的顺序，当前不据此启动实现。

下一步顺序由当前缺口决定：

1. **先测容量与访问轨迹。** 在现有完整调用内记录 N/E/U/K/Q/H、posting 历史/寿命/gap、每轮所触页和 cache reuse distance、raw dedup 峰值。成本代理必须直接来自 Trainer trace。
2. **落地 bounded 初始化。** full-key、u64 count、u32-local+u64-base、globally applied floor、按物理位置恢复顺序；这是 RAM 内路径和外存路径共同需要的基础。
3. **实现外部 word store+disk postings+endpoint 页访问。** 先用 RAM ledger/PQ 的 semi-external 模式隔离语义风险。用很小的缓存强制 eviction，与现有 Trainer 的整个 model signature 差分，覆盖跨页/跨块 AA、reserved activation、混合 weights/zero、length boundaries；一般 affix 继续 fallback。
4. **K/H 超预算才引入外存 ledger/PQ。** buffered delta、lazy ordinary candidates、出生发布 barrier、cold pair 验证与受预算 compaction 都要进入完整调用检查。
5. **并行研究 compressed-domain HF BPE。** 以真实 corpus 的 RLZ/grammar 大小和边界数量为 gate，先实现 HF adjacent-AA+canonical tie 的小模型等价证明。不能用 lossy model signature 或近似 greedy 换取容量。

来源与本地证据的可审计清单见 [provenance.json](results/external-memory-bpe/provenance.json)。本文的架构、估算与排序建议是独立推导；没有把理论 bound 或论文吞吐转成数十 GiB 的性能承诺。
