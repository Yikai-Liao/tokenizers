# 初始 pair 索引：算法问题、替代路线与独立原型

日期：2026-10-01。生产基点为 J `376363d2`。独立研究阶段只改原型；主实验已完成整合与完整 Trainer 对照，见 [INITIALIZATION_MEMORY_REPORT.md](INITIALIZATION_MEMORY_REPORT.md)。

## 当前最值得整合的结果

**稳定的块复用 radix 已整合进内存内候选 Trainer。** 找到了作者公开的 Radsort 参考代码，做了 C 原型和纯 Rust 移植。它保持 `(pair key, physical position)` 的 8 字节记录，使用已消费输入块保存输出，把第二份全量排序数组替换为约 2 MiB 加块目录。真实中文 1、4、16 MiB 的加权频率和每个位置均与独立 oracle 相同，位置严格递增。

单线程中文 16 MiB 初始化索引原型中：普通稳定 radix 为 320.6 ms，作者 C Radsort 为 202.4 ms，Rust 移植为 196.1 ms。对应原型进程 HWM 为 157.7、151.3、151.9 MiB。**这些是索引原型的结果，尚不构成完整 Trainer 的收益。** 独立原型没有当前 J 的 owner 路由、每个 SmallPosting 的分配及后续 merge。

两遍哈希和三遍 bitmap/rank 直接填充也通过相同 oracle。它们进一步降低原型峰值，但 16 MiB 下耗时约为 radix 的 3.4 倍，暂不优先整合。

## 1. 把初始化写成一个稀疏关系求逆问题

设平坦语料为 `x[0..S)`，分隔符为 NONE。有效物理边集合为：

```text
P = { p | p + 1 < S，x[p] != NONE，x[p + 1] != NONE }
g(p) = (x[p], x[p + 1])
w(p) = 该位置所在去重片段的权重
E = |P|，K = 不同 g(p) 的数量，A = 初始 alphabet 大小
```

每个 key 的输出包含三个彼此不同的量：

```text
physical_count[k] = #{ p in P | g(p) = k }
weighted_frequency[k] = sum{ w(p) | p in P，g(p) = k }
positions[k] = 按 p 递增排列的全部物理位置
```

频率低于 floor 的 key 不进入最终索引。高权重的单个物理位置可能通过 floor；不能把 frequency 当成 posting 长度。位置次序决定 AA 重叠匹配的从左到右边界，不能用不稳定排序直接代替。

这相当于把“位置 → key”的稀疏二元关系转置成“key → 有序位置”。排序是其中一种构造方法。SciPy 的 `csr_tocsc` 是现成的同类实践：计数、前缀和、按输入行顺序散射，输出行号自然有序。它的源码给出线性复杂度；映射到 BPE 后，主要新增问题是 key 空间稀疏、权重独立，以及并行写入顺序。[SciPy CSR 转 CSC 源码](https://github.com/scipy/scipy/blob/main/scipy/sparse/sparsetools/csr.h)

### 本工作负载的规模

512 MiB 中文的已记录规模：去重字符数 204,660,029，片段数 1,429,915，有效物理边 E = 203,230,114，alphabet A = 20,757，floor 后保留 key 数 2,697,517。这里 floor 前 K 尚未从当前统计中取得，不能用保留数替代全部计数表的容量。

| 表示或工作 | 容量或工作量 |
|---|---:|
| 当前 `(u32 key, u32 position)` 全量记录 | 8E = 约 1.514 GiB |
| 当前稳定 LSD 的全量 scratch | 又一个 8E |
| 全部物理位置按 `u32` 输出的上界 | 4E = 约 775.3 MiB |
| 用紧凑 alphabet 编号的整个 pair universe | A² = 430,853,049 个槽 |
| universe membership bitmap | A² / 8 = 约 51.4 MiB |
| 每个 64-bit bitmap word 的 `u32` rank 目录 | A² / 16 = 约 25.7 MiB |

`4E` 是当前未压缩输出表示的上界，不是所有算法的信息论下界：单元素/双元素 posting 可以内联，低频 key 可以过滤，单调位置还可以压缩。更根本地，原始 corpus 已经编码了 `g(p)`，无需另一份索引也能扫描得到位置，代价是查询时间。

若从 corpus 独立保存全部 key 标签，给定各组物理长度时，标签排列的信息量约为 `log₂(E! / ∏ n_k!)`，渐近接近 `E × H(key)`。这说明 key 的分布和输出访问约束比原始文件字节数更适合用于判断空间路线。

### 时间成本应分开算

- 顺序扫描：读 corpus，以及顺序推进片段权重游标。
- 随机查询：每条边访问哈希桶、rank 目录或最终 posting cursor。
- 数据重排：把 8E 字节的记录读写多轮，可能触发 cache/TLB/写分配成本。
- 并行通信：worker 路由、跨 worker 的计数归并、posting 填充前缀和。
- 输出：只写 floor 后保留的物理位置；尽量避免与临时记录同时长时间存活。

两遍算法的扫描次数较少，仍可能被每条边的随机访问拖慢。数据库 join 的研究也发现分区、payload 大小及实际查询组合会改变微基准结论，完整工作负载才能决定选择。[SIGMOD 2021：To Partition, or Not to Partition](https://15721.courses.cs.cmu.edu/spring2023/papers/11-hashjoins/bandle-sigmod21.pdf)

## 2. 六条路线及其适用条件

| 路线 | 主空间 | 主要时间代价 | 初步选择 |
|---|---|---|---|
| 稳定块复用 radix | 8E 记录 + 小 scratch + 输出重叠 | 4 个数字轮次、块目录、finalize | **已有原型，优先整合** |
| 两遍稀疏哈希，直接填 posting | O(K) 计数表 + 输出 | 每条边两次随机 key 查询 | 已测，较慢，内存优先时备选 |
| 三遍 membership bitmap + rank | 3A²/16 字节目录 + O(K) + 输出 | membership、计数、填充各扫一次；依赖随机读 | 已测，当前不优先 |
| 热字符 pair 密集计数，冷边 radix | O(H²) + 8E_cold + 冷排序 scratch + 输出 | 判断热/冷；热边两遍填充 | 有潜力，需测热×热覆盖率 |
| 只排序 `u32 position`，key 回读 corpus | 4E 记录 + 4E scratch + 输出 | 排序后对 corpus 的间接访问 | 空间中间档，访存风险较大 |
| 单调 posting 压缩或延迟物化 | 取决于编码/候选位置索引 | merge 解码、过滤、更新或额外扫描 | 更大改动，后续边界研究 |

### 2.1 稳定块复用 radix：保留分组算法，消除大 scratch

Radsort 是 Clausecker 和 Schintke 于 2026 年 7 月发布的预印本。其稳定 LSD 算法复用已消费的输入块，用逻辑块置换记录输出顺序。理论额外空间为 `O(b + E/b)`，选择 `b = Θ(√E)` 得到 `O(√E)`。作者实验版本固定 b = 512：8 字节元素的 scratch 为 2 MiB，块目录为每块约 9 字节。论文报告大输入下优于其传统 LSD 对照，具体机器和负载不等同本项目。[Radsort 论文](https://arxiv.org/html/2607.05302v1)

对本项目的推导：四个 key 字节轮次不变，稳定性满足 posting 次序；position 和 canonical key 不需改类型。4 个 owner 的理论排序 scratch 从约 1.514 GiB 降到约 8 MiB 加目录。固定 block 版本仍有 `O(E/512)` 目录，不能把实际实现写成严格 `O(√E)`。

整次初始化的收益受“记录与最终 posting 同时存在”限制。排序时减少了 scratch 后，峰值可能迁移到 install；owner 分波、尽早释放已安装 owner 的记录可以继续降低重叠。主线程的 owner 分波路线可与它组合。

作者实现的最后一步将逻辑块搬回连续数组。可以在下一阶段直接按 `perm` 和有效 block 长度消费分组结果，省去 finalize 的搬运。这是论文 §4.1 的接口用法。当前 Rust 原型先保留了连续 finalize，便于直接替换原 `sort(records)`；逻辑块消费尚未实现。

传统原地 MSD，例如 PARADIS，也能减小 scratch，但其原地置换会改变相同 key 的次序。若随后另按 position 排序，额外代价可能抵消收益。PARADIS 论文重点是原地并行置换和负载均衡，本项目对位置次序的要求需要另行处理。[PARADIS，PVLDB 2015](https://www.vldb.org/pvldb/vol8/p1518-cho.pdf)

### 2.2 两遍稀疏计数和精确散射：把临时空间从 E 转向 K

第一遍用哈希表得到每个 key 的物理 count 和 weighted frequency，过滤后精确分配。第二遍沿位置顺序查询 cursor 并填充，不生成全量 `(key, position)` 记录。

这是标准 group-by/count 的直接应用。DuckDB 的公开实践使用线程局部计数表，再按分区合并；其说明强调表容量由组数决定，而不是输入行数。[DuckDB 并行 grouped aggregation](https://duckdb.org/2022/03/07/aggregate-hashtable)

并行方案需要保持顺序，可以用每个语料 chunk 对每个 key 的 count 前缀和分配互不重叠的输出区间，或采用有界块路由、owner 顺序消费。前者最坏有 W×K 元数据，后者引入 barrier 和路由。原型只验证串行下的随机查询成本和语义，不宣称并行版本已经实现。

中文 16 MiB 原型明显慢于 radix。输入从 E = 637 万边扩大到 2.03 亿边后，哈希表及输出超出 cache 会改变成本；当前证据足以把它放到内存优先备选，不支持宣称小数据的比值能线性外推。

### 2.3 Membership bitmap + rank：用稀疏整数集合代替哈希

初始字符 ID 可映射到紧凑 `[0,A)`，key 变为 `a×A+b`，保持 pair 身份双射。第一遍只建立 A²-bit membership。每个 word 记录先前 set bit 数，key 的 dense index 为：

```text
rank(key) = prefix[key / 64]
          + popcount(bitmap[key / 64] & ((1 << (key % 64)) - 1))
```

第二遍更新 dense count/frequency，第三遍直接写最终 posting。时间是三遍 E，加上 `O(A²/64)` 的目录构建；辅助空间约 `3A²/16 + O(K)` 字节。rank 公式及此 BPE 实现是本研究推导，不归属于引用论文。

这个方法对 A 很敏感，须先压紧初始 ID：按完整 16-bit pair universe 建 bitmap 会需要 512 MiB，rank 又需 256 MiB。A = 20,757 时二者合计约 77.1 MiB，才有吸引力。编码后的 key 仅用于初始化内部，正式 key 仍恢复 canonical `(a,b)`。

已验证的 rank 原型比 radix 慢约三倍。目录和计数的依赖读、全 K 写入以及第三次扫描，是它新增的成本；目前没有 perf 证据将具体比例归因于其中某项。

### 2.4 热×热密集计数，冷边排序：同时削减记录和排序工作

选择 H 个常见初始字符，将二者都来自该集合的 pair 放入 `H×H` 小密集阵列，精确统计 frequency 和 physical count。其余边进入 radix。最终热 posting 沿 corpus 第二遍填充，冷 posting 由排序结果安装。热/冷分类是空间选择，所有 key 的统计都保持精确。

若热边占 q，cold 边数为 `(1-q)E`，临时 radix 记录和 scratch 都按 cold 边缩小。主要同时存活空间可按各阶段估算：

```text
sort:    16E_cold + 12H² + 4E_hot
install:  8E_cold + 12H² + 4E_retained + key metadata
```

安排热 posting 的分配时机后，sort 阶段也可以不保留 `4E_hot`。这些是容量模型，尚未测得 q 和最终 RSS。先测 H = 256/512/1024 的**物理边覆盖率**，再决定是否做原型；weighted 覆盖率不足以估算记录空间。512 个热字符的 count/frequency 阵列约 3 MiB，每 worker 一份时需乘 worker 数。

Zippy 的 top-k aggregation 工作把热候选和冷分区分开处理，用 cache 内结构减少数据搬运，这支持探索热冷分路的机制。但它允许利用 top-k 上界剪掉非候选组；BPE 后续还可能选择冷 key，不能直接采用这种剪枝。这里仅借鉴热冷物理路径，保留全部 exact 统计。[Zippy，PVLDB 2023/2024](https://www.vldb.org/pvldb/vol17/p644-siddiqui.pdf)

### 2.5 只保留位置：用 corpus 重新计算 key

将记录改为 `u32 p`，排序时通过 `corpus[p], corpus[p+1]` 取 key。稳定位置数组和 scratch 各减半，所有实际类型和模型输出仍不变。缺点是前几个数字轮次后，p 的读取次序已经打散，读取 corpus 可能形成不连续访问。预计是明确的容量收益和不明确的速度代价，尚未尝试。

另一种做法是先顺序按 left token 聚类位置，获得密集的 A 行，再在每行统计 right token；它相当于稀疏矩阵按行处理。峰值 scratch 可以按最大的 left-token 行限制，但热点行长度及 workload 倾斜必须测量。

### 2.6 改变最终 posting 表示或延迟物化

位置本来严格递增，Elias–Fano 每组理论约 `n_k×(log₂(S/n_k)+2)` bit，外加访问目录，可减小最终位置表示。Vigna 的 Quasi-Succinct Indices 对单调倒排表做了工程验证；这些检索结果不等同于 BPE 的频繁删除和新增 posting。[Quasi-Succinct Indices，WSDM 2013](https://vigna.di.unimi.it/ftp/papers/QuasiSuccinctIndices.pdf)

本项目的可行变体是初始 posting 不可变压缩，merge 时过滤失效位置；需要追加新位置的 pair 转为当前 SmallPosting。要测冷 posting 的实际寿命、解码扫描量和重建成本，不能只根据理论压缩比判断。

更激进的变体是仅统计所有初始 key 的频率，选中 pair 后再枚举位置。若每次扫描整个 corpus，成本变成 merge 数乘 N，明显不适合。若用单字符位置表，在选中初始 `(a,b)` 时枚举较少一侧的字符位置并验证相邻 token，可把扫描量降到 `min(n_a,n_b)`。它能消除初始 pair 的排序，但最终单字符位置表仍接近 4N，重复候选检查还可能拖慢 merge。wavelet 序列索引也能支持这类倒排访问，已有研究讨论其 inverted-list 表示；当前没有对应 BPE 原型。[Wavelet trees 与 inverted lists](https://arxiv.org/abs/1011.4532)

## 3. 独立原型的共同约束

源码在 [results/initialization-algorithms/prototype](results/initialization-algorithms/prototype)。输入使用现有真实 zh Wikipedia 1、4、16 MiB 文件；逐行 `read_line`，保留 LF 和 CR，按整段字符串去重并累加权重。字符 ID 按 Unicode scalar 排序，片段按首次出现顺序铺平，NONE 严格分隔片段。

这个顺序与 J 的 AHashMap 遍历顺序不同。原型中的全部候选和 oracle 使用同一顺序，因此可逐位置差分；正式整合仍需在同一 Trainer feed 上检查模型签名。

四个算法都输出同一种连续 posting 表、每个 key 的 count 和 exact weighted frequency，floor = 2。原型没有把计数频率当物理长度，也没有把低频 key 的位置混入输出。checksum 覆盖排序后的 key、频率、物理长度和所有位置；oracle 对每个 key 和每个位置分别比较，checksum 只是产物摘要。

### 正确性验证

- 真实三份语料的四个原型与 Rust 移植均通过完整 oracle，相同输入 checksum 一致。
- 小语料覆盖空输入、单字符、空片段、AA 连续重叠、长于 256/512 的片段、LF、CR、零权重、混合权重，以及 floor = 1/2/9/100000。
- Rust 移植另覆盖 14 种长度 × 5 种 key 分布，包括 block 边界、全同 key、交替 key、随机 key、`0`、`u32::MAX` 和最高位。
- frequency 加法使用 checked arithmetic；真实小语料均未溢出。当前独立原型不覆盖完整 Trainer 的协议和后续 merge。

## 4. 实测结果与计时边界

单线程，无 target-cpu=native；Rust release `opt-level=3`、`codegen-units=1`，ahash 0.8.12，种子固定 `[1,2,3,4]`；作者 C 版本 `cc -O3 -DNDEBUG`。CPU 为虚拟机暴露的 Intel Xeon Gold 6140。没有与主线程 512 MiB 正式计时重叠；原型轮次可能与开发编译等小任务重叠，因此速度仅用于候选筛选。

下表使用修正前处理后的那一轮；每配置一次。Rust 移植随后另做了一次。

| 语料 | 算法 | 索引墙钟 ms | 进程 HWM MiB | 索引分配容量模型 MiB |
|---|---|---:|---:|---:|
| zh 1 MiB | radix | 25.3 | 13.17 | 6.69 |
| | C Radsort | 41.1 | 12.50 | 6.69 |
| | hash 两遍 | 148.6 | 11.77 | 4.98 |
| | bitmap/rank 三遍 | 108.0 | 14.82 | 8.95 |
| | Rust Radsort | 14.2 | 12.53 | 6.69 |
| zh 4 MiB | radix | 102.8 | 41.39 | 24.59 |
| | C Radsort | 65.4 | 39.84 | 22.65 |
| | hash 两遍 | 341.1 | 35.21 | 17.73 |
| | bitmap/rank 三遍 | 221.0 | 41.41 | 20.98 |
| | Rust Radsort | 56.9 | 39.82 | 22.65 |
| zh 16 MiB | radix | 320.6 | 157.73 | 97.32 |
| | C Radsort | 202.4 | 151.33 | 90.00 |
| | hash 两遍 | 1116.6 | 125.66 | 52.76 |
| | bitmap/rank 三遍 | 1083.6 | 121.68 | 58.30 |
| | Rust Radsort | 196.1 | 151.89 | 90.00 |

HWM 在 oracle 和 checksum 计算**之前**读取，包含 corpus 前处理及 allocator 保留页。源码记录 before-index RSS/HWM；例如 16 MiB 的各算法 before-index RSS 均约 60.4–60.6 MiB。容量模型只包含索引临时 Vec/计数表与输出，没有 corpus、feed、allocator metadata；hash 表容量是按 hashbrown load-factor 估算，不能当成实测 RSS。

`index_ms` 从记录生成/第一遍计数开始，到最终连续 index 填好为止；不含前处理、checksum 和 oracle。每个 `.time` 文件的 user/system CPU 是**整个进程**，包含前处理和 oracle；不能拿它归因到初始化 kernel。第一轮原型结果另保留在 `preprocessing-v1/`，其中前处理曾先收集全部字符再 sort/dedup；之后改为固定 Unicode membership 表，消除了这一不必要的大临时 Vec。第一轮未混入上表。

1 MiB 的 C Radsort 第一轮为 18.0 ms、第二轮为 41.1 ms，hash 也有明显波动。保留两个事实；没有将它们拼成统计中位数或把小差异写成确定胜负。4、16 MiB 下 Radsort 的方向一致，足以支持进入完整 Trainer 验证。

## 5. 原型来源与复现

- [manifest.json](results/initialization-algorithms/manifest.json)：四算法第二轮的源码、binary、输入 SHA-256、命令及输出。
- [rust-port-manifest.json](results/initialization-algorithms/rust-port-manifest.json)：Rust 移植 binary/source/input SHA-256 和逐配置结果。
- [原型 main.rs](results/initialization-algorithms/prototype/src/main.rs)：同一语料前处理、四个算法及 oracle。
- [Rust Radsort](results/initialization-algorithms/prototype/src/radsort_u64.rs)：按 high32 key 稳定排序，包含连续 finalize 和边界差分。
- 作者仓库：[clausecker/radsort](https://github.com/clausecker/radsort)，精确版本 `f69e816c3cd79d312cd67aea5b9cf1c338c1b371`。
- 参考文件 `radixsort_permuted.c`、`radixsort.h` 及 [BSD-2-Clause COPYING](results/initialization-algorithms/prototype/vendor/COPYING) 原样保留。Rust 移植来自该参考实现；集成时需要保留许可证和来源。

```sh
cd benchmarks/hf-bpe/results/initialization-algorithms/prototype
/root/.cargo/bin/cargo test --release --offline
/root/.cargo/bin/cargo build --release --offline
target/release/initialization-algorithm-prototype selftest -
cd ..
python3 run.py
python3 run_rust_port.py
```

## 6. 如何决定速度与内存的甜点

先把当前原型 Radsort 整合到 J 的 owner 路由和 SmallPosting 上，保持已验证的 merge 逻辑，测整次训练的模型签名、初始化时间、完整训练时间、初始化 HWM、merge HWM 和全程 HWM。再与 owner 并发宽度、posting Arena 阈值组合。

选择指标是完整训练在内存预算内的最快配置：

```text
Peak_total = max(Peak_initialization, Peak_merge, Peak_other_phases)
选择完整训练时间最短，且 Peak_total <= 预算的配置
```

当 merge 已经决定全程峰值时，再降低 init 峰值仍能减小阶段 RSS，但不会继续降低全程 HWM。不能把较高初始化峰值造成的“峰值余量”当成 Arena 保留空间的预算依据。

当前有界研究支持的顺序是：**Radsort → 与 owner 分波组合 → 按实测整次峰值重选 Arena 阈值**。两遍哈希/rank 保留为内存优先候选；热冷密集分路是进一步改变记录量的下一项，先取得物理覆盖率再决定是否实现。更换最终 posting 表示和延迟物化会改变 merge 的访问边界，放在较大结构演化阶段。
