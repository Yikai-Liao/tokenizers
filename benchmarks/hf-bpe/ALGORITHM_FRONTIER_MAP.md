# 精确 BPE 的算法地图与可接入的前沿机会

调研日期：2026-10-01。本文覆盖整个优化过程；初始化的详细替代原型继续见 [INITIALIZATION_ALGORITHM_RESEARCH.md](INITIALIZATION_ALGORITHM_RESEARCH.md)，本文不重跑该研究。源码观察基点为 J `376363d2` 及 `/root/code/tokenizers-worktrees/initial-owner-waves` 的 staged Radsort 实现。原 `efficient_bpe` 仅作只读算法来源核对。

当前已收获按 chunk 前缀直接填最终 owner 路由数组，以及有范围证书的 8 字节候选记录。最新方向是**数十 GiB 及以上规模的通用算法优先**：先处理超过 `2^32` corpus slots 时的 block-dictionary 初始化、完整 key/频率宽度与历史 posting 生命周期，再考虑小范围特化。单独的 [SCALE_UP_ANALYSIS.md](SCALE_UP_ANALYSIS.md) 给出源码分支、条件容量表和 bounded 初始化提案；本机没有运行数十 GiB，不能把当前 flat 分支收益推给超界 fallback。

两项局部替换已集成，源码 tip 为 `8bd048f6`（六次正式计时 binary 对应 `56dd3227`）；主任务报告 52 项 Trainer tests、2 项针对性测试、4 个小模型 smoke case，以及新增非空 affix fallback test 通过。六次正式调用均通过 D50。candidate 压缩须为普通、无非空 affix，且 `V_bound≤65,536`、`E_0≤u32::MAX`；wide fallback保持完整 u64 frequency/pair。arena all两组较快，但repeat超过历史B2峰值预算；阈值选择按最新要求留到算法完成。
最新论文确实提供了可用原语：Radsort 已进入主任务并扩展到full-key有界批次；SimdQuickHeap 适合后续比较，TPHT 和 Zombie Hashing 提供字典替换路线；因现有Rust库与PR接受成本降优先级。但当前负载并不支持立即把整个 owner 表或候选队列换掉。以下按实际成本和接口条件说明原因。

## 1. 统一问题、语义与证据口径

输入是去重片段 `(s_i, w_i)`，`w_i` 是非负 u64 权重。预分词、保留换行、alphabet filtering、special token 和 canonical ID 分配均属于语义。令过滤后的片段字符长度为 `L_i`：

```text
N_u = Σ L_i                         去重后的字符槽数
E_u = Σ max(L_i − 1, 0)             物理初始边数
E_0 = Σ w_i · max(L_i − 1, 0)       加权初始边质量
V_0 = special + alphabet 完成后的唯一身份数
V_bound = max(V_0, requested_vocab_size)
```

平坦语料含分隔槽，故其分配容量并非 `N_u`。在本文普通 compact 路径中，对尚未选择的 pair `q`，`frequency(q)` 是物理边权重之和，`posting_len(q)` 是物理位置数；重复片段不复制 posting。`E_0`、`E_u`、原文件字节数不可互换。

全局 greedy 次序为频率降序，再按 canonical `(left_id, right_id)` 升序。规则执行在每个片段内从左到右；`AA` 的重叠必须选择交替起点。普通无 affix 路径只合并、不拆分；身份首次激活、旧 pair 不增频、低频永久退役的条件证明见 [PAIR_MONOTONICITY.md](PAIR_MONOTONICITY.md)。非空 affix、任意 dropout 或改变 pretoken 边界的算法不能直接采用这些证书。

安全批次还要求：保留 greedy 的逐条排名与 ID 分配；规则写入跨度不重叠；新边采用该批次最终邻居；所有 worker 出生贡献聚合后才能做 floor 过滤；每个出生 key 的 position 严格递增且唯一。Relaxed priority queue、近似 heavy hitter、采样训练、改变 tokenization objective 均不满足本任务契约。

### 已测成本决定排序

J 的同 binary 1/1 与 4/4 线程对照每配置一次，见 [rule-aggregate-scaling.summary.md](results/rule-aggregate-scaling.summary.md)。四线程：train 23.626 s，init 5.547 s，merge 13.528 s；prepare 6.769 s，commit 5.449 s，rewrite 0.622 s，select **0.282 s**。route 1.644 s 内含 compact 0.756 s；sort 1.826 s。阶段父子项不能相加，CPU 采样份额也不是墙钟份额。

这意味着只改善 select 的理想完整训练节省上界约 1.2%，即便把它完全消除。因此候选压缩的主要理由是内存；队列微基准的 2× 不能直接转成 Trainer 2×。prepare、commit 和大数组搬运仍有更大的时间空间。

[终点 inventory](results/j-posting-inventory.summary.md)测得 10,572,128 个 owner entry、7,274,631 个独立 posting backing buffer、14,225,344 个候选记录。posting 请求容量 771.62 MiB，候选 backing capacity 329.29 MiB，owner table 估算容量 528.00 MiB。候选只在四个连续数组中；posting 才是数百万次独立小分配。61.13% 的 posting buffer 容量不超过 8 个 u32，却只占 posting 请求字节的 10.75%。这些容量统计不等于 RSS。

主任务最新 staged 同 binary 四次调用结果：classic sort2/install2/all：init 7.820 s、train 21.127 s、HWM 约 3.68 GiB；block sort4/install2/T256：init 5.507 s、train 20.003 s、HWM 约 3.38 GiB；block sort4/install2/all：train 20.021 s、HWM 约 3.55 GiB；block/T0：train 23.065 s、HWM 约 3.38 GiB。这是主任务回传的 **n=1 筛选**，正式来源由其结果文件维护。它支持先从 T256 组合继续，不能宣称精确最优阈值或稳定排序。

早先 all arena 不增 HWM，是因为经典初始化已建立 4.43 GiB 的峰值；Radsort 降低该峰值后，合并期 live+retired arena 的峰值重新成为约束。见 [POSTING_ARENA_THRESHOLD_REPORT.md](POSTING_ARENA_THRESHOLD_REPORT.md)。

### 新集成的完整对照与阈值反序 repeat

主任务在同一最终诊断 binary、中文512MiB、none、50k/min2、init4/merge4、block sort4/install2上完成下列四次调用，全部通过完整模型D50签名。单项收益保持相邻control：route legacy→direct固定wide/T256；heap wide→auto固定direct/T256；arena T256→all固定direct/auto。

| 配置 | init s | full train s | route s（含compact） | HWM GiB |
|---|---:|---:|---:|---:|
| [legacy route / wide / T256](results/frontier/zh512m-legacy-wide-t256.summary.json) | 5.740548 | 19.730086 | 1.747130（compact 0.883248） | 3.368232728 |
| [direct route / wide / T256](results/frontier/zh512m-direct-wide-t256.summary.json) | 4.356681 | 17.893389 | 0.724478（compact 0） | 3.376522064 |
| [direct route / auto packed / T256](results/frontier/zh512m-direct-auto-t256.summary.json) | 4.327508 | 18.105217 | — | 3.359176636 |
| [direct route / auto packed / all](results/frontier/zh512m-direct-auto-tall.summary.json) | 3.922882 | 16.469754 | — | 3.439247131 |

直接route的完整训练本次少9.31%，route整体少58.53%；init少24.11%。全训变化不能全部归因route，因为未改merge仍会波动；route直接阶段及机械删除的compact搬运共同支持保留。auto packed的本次train稍慢（约1.18%），**没有独立速度改善证据**；其heap backing capacity实测从329.2867432 MiB降至164.6433716 MiB，candidate数同为14,225,344，表示收益精确为一半。本次E_0=203,974,788、V_bound=50,000，满足普通compact路径的数值gate。

all 相对 auto/T256 首轮 train 少 9.03%，HWM 增加约 82.0 MiB，接近历史 B2 约 3.442 GiB 的预算边界。随后同 binary 完成 all→T256 反序 repeat，仍全部通过 D50：

| 反序 repeat 配置 | init s | full train s | HWM GiB |
|---|---:|---:|---:|
| [direct / auto packed / all](results/frontier/zh512m-direct-auto-tall-repeat.summary.json) | 4.891775 | 20.067069 | 3.458663940 |
| [direct / auto packed / T256](results/frontier/zh512m-direct-auto-t256-repeat.summary.json) | 4.974043 | 20.953533 | 3.332626343 |

all 在 repeat 对内 train 少 4.23%，HWM 增加 129.0625 MiB，**峰值已经超过约 3.442 GiB 的历史预算**。T256 的两次峰值均低于该预算。按最新要求，阈值选择留到算法完成；当前 T256 仅作固定对照，all 不能被宣称满足同一峰值预算。完整训练跨组波动明显，每阈值仅两次，不能据此宣称精确最优或稳定百分比。不要把更早 staged 组合的 20.003 s 当这六次新 binary 的对照。


## 2. 全过程的高层算法地图

表中的现状 ID 对应 [OPTIMIZATION_CATALOG.md](OPTIMIZATION_CATALOG.md)。算法复杂度只描述工作量；随机访问、分配、字节搬运及并行屏障决定本机常数。

| 阶段与现状 | 高层数学/算法问题 | 当前已采用的原语及关键约束 | 前沿或实践判定 |
|---|---|---|---|
| feed：片段去重加权 | 字符串 exact group-by、非负计数 reduction | CompactString + AHash；必须保留 PT 的真实片段与换行 | 并行 hash aggregation 是成熟实践；benchmark 的 feed 明确串行，不能把开启 feed 并行当内核新收益 |
| I3 字母表 | Unicode universe 的集合并，或频率排序裁剪 | unlimited 用 worker bitmap OR；limited 保持 HF selector 的 tie 行为 | 位图 OR 已比逐字符 hash 更贴合有界宇宙；SIMD 只可能替换内核，裁剪语义不能变 |
| I4 字符→ID | 静态有界整数字典查询 | 0x110000 项直接表，约 4.25 MiB，canonical ID 固定 | 已采用直接寻址；MPHF/learned index 对这份小密集 universe 没有明确优势 |
| I2/I5 corpus 构造 | 分段长度 scan + 稳定 scatter + 独占 first-touch | 连续片段区间、checked 长度、最终 MaybeUninit 区域一次写入 | 标准 prefix/disjoint scatter；SIMD UTF 解码可讨论，但过滤与 ID 映射仍需逐字符工作 |
| S1 endpoint 表示 | 动态路径图的边收缩；稳定物理地址 | 起点/终点保存 token ID，跨度表找邻居；不搬移整片段 | 已省 per-position prev/next/word_id；Halfword 是替代表示，需新并发/跨度证书 |
| 初始 owner 路由 | 稳定 partition，映射 `position→owner(pair)` | chunk 内递增位置，chunk 间按物理顺序拼接 | 直接 count-prefix-scatter 可以省 compact；详见 §4 |
| I6 pair 初始化 | 稀疏关系转置：key→有序 occurrence；exact weighted group-by | 8 字节 record，稳定四轮 LSD，floor 后才装字典 | Radsort已接入；full-key adaptive batches已做完整Trainer对照，中文初始化约3.6–5%；hash2/rank3原型慢；finalize重写机会小 |
| S4 owner 字典 | 可变 exact map，唯一写所有权 | AHashMap<u64, Entry>，Entry 为 frequency + posting | 当前已有 SwissTable/SIMD 探测；TPHT 值宽度不匹配，Zombie 需要完整替换 |
| S2 候选堆 | 动态 exact argmax，旧上界 lazy repair | 8 叉堆；频率减小只在 top 被观察时 pop/reinsert | 8 字节条件表示是局部替换；SimdQuickHeap 是后续候选；radix heap 无现成 owner 局部单调证书 |
| 跨 owner 选择 | 少量已校正局部最大值的 tournament | coordinator 比较所有 owner heads，仍使用 canonical tie | P=4 时扫描很小；大 P 可缓存头或 tournament，但需新 P 的实际成本证据 |
| S5 退役 | 单调状态上的永久资格剪枝 | 初生聚合后低于 floor 丢弃；旧 pair 仅减频 | 属于已有证明支持的精确剪枝；top-k/近似 sketch 不可替代 |
| P2 批次 | 带 greedy 次序约束的独立边集合/可交换重写 | AA、tail/head 冲突与 reserved-ID barrier | 并行图 MIS/relaxed order 不保证 greedy trace；保留当前具体证书 |
| P4 AA | 排序整数集合的连续 run 检测 + parity scan | chunk 摘要、尾 run 近邻检查/二分、incoming parity | 已把 coordinator 工作从 occurrence 降到 chunk；一般 parallel scan 可替换摘要传播，当前占比很小 |
| M1/M3/M8 权重 | 静态 predecessor query + certified uniform interval | 稀疏 pivots、256 槽目录、weight=1 位图、近邻 cursor | learned predecessor/EF 需额外访问和构建；大部分桶已可跳过查询，先看剩余 mixed-bucket 成本 |
| M2 posting 验证 | 对历史位置表做 snapshot filter，再生成局部图增量 | 检查 endpoint，读旧 corpus，join 后才写 | SIMD blocking G2 已测未收益；压缩历史 posting 是表示替换，不能直接保留 slice API |
| M4 同批最终邻居 | 小规则集合上的 exact lookup | head/tail 直接目录；多重规则小 hash 回退 | 当前有界目录匹配≤65,536 ID；无需另建一般动态图邻接表 |
| M10 规则邻居聚合 | 每 Task×侧×neighbor 的局部 group-by | touched-ID 目录累计，按原链次序 flush | 已减少逐事件 hash；进一步压 group/cached route 要依据当前 J 的实际 group 数 |
| P3 delta 路由 | 稀疏多生产者→唯一消费者 reduction | 每 job×owner route；8 字节出生链节点 | 先聚合 count/frequency 后 exact 预留；unordered completion 必须恢复生产序号 |
| M5 owner commit | exact segmented assembly + 隐式 CSR 构造 | 全 worker count 汇总，预留最终容量，逆序链直接写 suffix | 已避免扩容与多余 reverse；直接组装 I 反而变慢，不重复采用 |
| S3 posting | 大量不可扩容短向量的混合表示与生命周期管理 | 16 字节 SmallPosting 内联 2 个 u32，长向量 exact capacity | arena T256 已测；可回收 size-class pool 可省 retired retained，但属于中等风险后续 |
| P1/P5 调度/销毁 | fork-join DAG 调度与批量内存回收 | 持久 Rayon 池；coordinator 在池内；arena 整块回收 | 已避免反复建池；parallel drop K 未兑现全训收益；每轮更细任务不能破坏 posting 次序 |
| 输出构造 | merge DAG 的字符串物化与模型序列化 | 保留原 vocab/merge 输出格式 | H 独立诊断 string 构造约 21 ms，owner drop 约 4.75 s；输出字串不是当前主要热点 |

限定长度与 affix 的可扩展证书分别见 [LENGTH_PRUNING.md](LENGTH_PRUNING.md)、[AFFIX_PRUNING.md](AFFIX_PRUNING.md)。它们是特定配置的算法选择；当前 none/unlimited 大样本不会因实现这些证书变快。

## 3. 最小局部收获：8 字节候选的完整证书

这是本项目推导的表示优化，不归属于某篇新论文。它保留当前 8 叉堆和 owner ledger，仅在充分条件成立时改变 backing element。

### 3.1 精确 eligibility

```text
supports_compact(trainer)  // prefix/suffix 均为 None 或空字符串
AND max(strings_after_special_and_alphabet.len(), trainer.vocab_size) <= 65_536
AND checked_weighted_initial_edge_mass <= u32::MAX
```

**本证书只适用于普通、无非空 affix 的 compact 路径。** 该条件由 `do_train_indexed_parallel` 的 `supports_compact(trainer)` 入口和 packed eligibility 本身共同保证；非空 prefix/suffix 回退一般串行 ledger，不能复用本证书。显式重复 guard 防止以后入口复用时遗漏。`corpus::build` 的 `Region.weighted_edges:i64` 已逐片段 checked multiplication/addition，并 checked 汇总；把汇总作为 `Prepared.weighted_edges:u64` 返回即可，**无需新增 corpus 扫描**。`L_i` 必须用其已有 retained 字符数。保留原 i64 溢出错误；超过 u32 上界时选择 wide heap，不截断、不饱和，也不新增拒绝输入。

65,536 是身份**数量**上界，最大合法 ID 是 65,535。它不同于 u16 corpus 的 65,535 数量上界，后者需给 separator 留出 65,535 码值。候选 pair 没有 u16 separator；corpus 仍可为 u32/AtomicU32。forced alphabet 和 special token 已包含在 `strings.len()`；零权重片段仍参与 alphabet，但贡献零 edge mass。

### 3.2 频率上界证明

以下证明假定 `supports_compact(trainer)` 成立，prefix/suffix 均为 None 或空字符串；一般 affix ledger 的别名与历史 cohort 不在证明范围。每个当前 token 是初始字符序列的连续区间。合并只删除 token 区间间的边界，因此每个片段当前边数 `e_i(t) ≤ max(L_i−1,0)`。权重非负：

```text
Σ_i w_i e_i(t) ≤ E_0
0 ≤ frequency_t(q) ≤ Σ_i w_i e_i(t) ≤ E_0
```

在上述普通 compact 路径中，初始候选、新生候选和 lazy heap 中的旧 frequency 都是某个合法 snapshot 的 subset 边计数，所以均≤E_0。AA 的频率可以包含相互重叠的两个相邻边；它仍是物理边的子集，不能误用“实际选中 merge 数”作上界。partial birth 聚合都是非负贡献，最终精确总和已被同一 bound 覆盖。有限 length gate 只限制登记哪些新边，不增加物理边。

reserved ID 的首次激活不扩大身份数。追加新字符串只发生在 `ids.len()<vocab_size` 的循环中，每个唯一新身份同时追加 strings 与 ids；已有 special 字串复用不追加。故最终 `strings.len() ≤ V_bound`。该证明依赖当前 ids/strings 唯一身份一致性，不能推广到存在字符串别名的任意新训练器。

### 3.3 顺序嵌入与 fallback

```text
code = (left_id << 16) | right_id
packed = (frequency << 32) | (!code as u32)
```

以 u64 max heap 比较 packed，正好是 frequency 降序、code 升序；code 的 lexicographic pair 次序与原 `(left<<32)|right` 相同。decode 后恢复原 u64 canonical key，owner 分片函数继续吃原 key。**不能直接把原 u64 pair key cast 为 u32**，那会丢失 left ID。

低风险接口为 `enum CandidateHeap { Packed(OctonaryHeap<u64>), Wide(OctonaryHeap<Candidate>) }`，公开到本模块的方法继续返回 Copy Candidate。peek/pop/push 都可局部转换，无需引用稳定性或 PeekMut。初始化先选 variant 再构建，避免先收集 wide 再转 packed 的临时峰值。heap bytes 按实际 capacity×8/16 统计，不能继续固定×16。

本轮实际记录 heap backing **329.29→164.64 MiB**，容量精确减半；allocator/RSS 不按此数字线性下降。频率还在 ledger 中保持 u64，posting/corpus/key API 都不改变。

[独立 arithmetic oracle](results/algorithm-frontier-map/packed_candidate_oracle.py)验证 10,064 个 record 的 roundtrip 与完整排序，含 0/u32::MAX 与 0/65,535 边界，并检查 6 个 eligibility 边界。结果见 [JSON](results/algorithm-frontier-map/packed_candidate_oracle.json)。它只验证顺序数学，不代替 Trainer 差分、overflow 输入、reserved-ID、AA、floor 和模型签名测试。主任务已实现Packed/Wide wrapper、直接narrow collect、意外range escape时的wide promotion，并报告52项Trainer tests通过；主任务另报告high-weight和vocab=70,000的fallback针对性测试通过，采用selected J的wide indexed serial oracle：legacy HF将weight cast为i32，不适合作为超u32频率的oracle。首轮同binary计时已回传（见§1）；packed没有独立速度收益，backing容量实测减半。

## 4. 第二项局部收获：省 owner route compact

当前 radix route 已做两次 corpus 扫描：第一次为每 chunk×owner 精确计数，第二次写对应 chunk-local Vec，随后逐 owner append 到最终 Vec。直接 scatter 保留这两次扫描，只在其间多一个小 prefix 阶段。

设按物理位置有序的 chunk 为 `c=0..C`，owner 为 `o=0..P`，第一次扫描得到 `h[c,o]`。在各 owner 的最终缓冲区内：

```text
offset[c,o] = Σ_{j<c} h[j,o]
owner_len[o] = Σ_c h[c,o]
worker(c) writes owner[o][offset[c,o] .. offset[c,o]+h[c,o]]
```

每个 worker 顺序扫描自己的位置，向所持 owner slice 推进 cursor。不同 `(c,o)` 的区间不相交，全部区间精确覆盖最终数组；chunk 顺序对应物理位置顺序，所以每个 owner 的所有输入位置仍递增。稳定 radix 后同 key 的位置仍递增，AA 与初始 posting 安装的证明不变。

实施可在每个 owner Vec 的 spare/MaybeUninit 区域按 counts 切出互斥 slices，再将这些 slice 句柄转置成每 chunk 一个 job；slice 本身没有复制 payload。join 完成并验证各 cursor 后再发布 Vec 长度。无需全局原子 cursor；原子 fetch_add 只能保证区间唯一，不能保证同 key 的物理顺序。失配/错误时不可发布未初始化记录。

此项省去 compact 的一次 8E_u 读取和一次 8E_u 写入：本负载约 **3.03 GiB 请求流量**，并消除 old chunk buffers 与正在形成的最终 owner buffers 重叠。它不是精确 DRAM 流量预测：cache、write allocation 和 copy 实现仍影响机器流量。元数据 O(CP)，当前 4×4 很小。source 的 owner hash 与两遍扫描次数都不变；开销是 prefix、slice 分配句柄、屏障和 second-pass 多流写入。

低风险主要指语义边界局部且有直接等价证明；MaybeUninit/切片覆盖仍需内存安全审查。首先比较 route（含 compact）与完整 init/train/HWM，不能只宣布删除一个计时字段就等于省了该时间。主任务随后已实现owner_route.rs并报告52项库测试通过。独立只读source审查没有发现阻塞问题：safe split_at_mut区间精确分割，indexed zip恢复chunk对应关系，emit计数assert和join发生在MaybeUninit→u64转移之前，panic不会发布未初始化u64。现有weighted retained-posting oracle不直接观察floor丢弃的stream；更强的物理oracle是directstream与独立sequential ownerpartition逐record比较。首轮同binary对照route及完整train同向改善（见§1）；每配置n=1。

## 5. 最新原始论文与实现的接入筛选

以下来源均在调研日核对原论文或作者实现。发表时间、作者 benchmark 和本项目结果分开；论文加速数不作为本项目预期。

| 来源与日期 | 可提取的原语 | 与当前实现的实际兼容条件 | 判定 |
|---|---|---|---|
| [Radsort，2026-07](https://arxiv.org/html/2607.05302v1)；[作者 C](https://github.com/clausecker/radsort) | stable LSD 消费输入块后复用；逻辑 block permutation；小 scratch | 8B record、四个 key byte、稳定位置顺序完全吻合；当前纯 Rust 已用原内核 | **已收获**。后续可直接消费逻辑块，跳过 finalize；parallel 协议见 §6 |
| [SimdQuickHeap，2026-04](https://arxiv.org/html/2604.25681v1)；[作者 Rust](https://github.com/RagnarGrootKoerkamp/quickheap) | 相邻 pivots + bucket partition；SIMD classify/partition；push/pop 无 decrease-key handle | library 只支持 32/64bit scalar；packed u64 可反转成 min-order；需 AVX2/AVX512 和正确 fallback | **暂后排**。原论文已比较 8叉堆，不只是二叉堆；当前 select 较小；多 bucket slack 的内存待测 |
| [Tiny Pointer Hash Tables，PVLDB 2026](https://arxiv.org/abs/2607.28892)；[standalone C](https://github.com/Xilinion/TPHT) | byte-sized pointers、quotienting；Chained 省空间，Flattened 追求单 cache miss | standalone value 是 1–8B；当前 Entry value 为24B，BirthGroup亦超8B。需 map→handle + Entry arena，新增间接访问与生命周期 | **有价值但不是直接替换**。可先用实际 owner 操作 trace 验证；不把作者64key/64value负载推广到32B entry |
| [Zombie Hashing，SIGMOD 2025](https://users.cs.utah.edu/~jeffp/papers/zombieht.pdf)；[作者实现](https://github.com/saltsystemslab/ZombieHT) | 小窗口 tombstone redistribution，保持高 load 的 churn 吞吐 | owner dictionary 有 birth/remove，但当前 SwissTable并非同一种 linear probing布局；要移植 map与value布局 | **中长期候选**。论文95% fullness收益需抵扣访问/迁移成本；当前profile主要证据是growth及route probes，未隔离严重churn退化 |
| [FractalSortCPU，2026-05](https://arxiv.org/abs/2605.10390)；[论文所列 artifact](https://github.com/mikdanana/fractalsort_cpu/)（本次未能取得） | 稀疏树 histogram、counter width tapering | 输出 sorted keys 的 histogram expansion 不能自动保留不同 position payload；32bit稀疏pair+stable8B records还缺相应接口证书 | **不接入当前排序**。headline带宽结果不能代替稳定key/payload oracle |
| [External-Memory Priority Queues with Optimal Insertions，ESA 2025](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ESA.2025.5) | buffered layers，摊还 block transfer | 理论 external-memory模型；当前实测无swap，缺对应可直接调用的小型Rust内核 | **无立即收获**。只有队列远超cache且选择成为大项时再研究 |
| [Faster Superword Tokenization，2026-04](https://arxiv.org/html/2604.05192v1) | 对候选超词按频率聚合，避免保留完整documents | 本任务普通BPE已按pretokens去重；Boundless/SuperBPE跨边界改变目标 | **已采用相同基础思想**；可迁移的weighted dedup收益已存在，不改变PT去获取论文headline加速 |
| [Stream VByte 原论文](https://arxiv.org/abs/1709.08990)；[维护实现](https://github.com/fast-pack/streamvbyte)；[2026 forward-index研究](https://arxiv.org/abs/2602.05445) | control/data分流的SIMD整数解码，delta-compressed顺序列表 | posting sorted且birth后不可追加的生命周期有利；但现有as_slice、AA随机检查、job切分需block目录或解码scratch | **中期表示候选**。先选长冷posting、保留inline；小列表头部和频繁解码可能抵消收益 |
| [simdutf官方实现](https://github.com/simdutf/simdutf) | vector UTF8→UTF32、ASCII fast path，CPU dispatch | Rust str已有效；仍需canonicalID查表与过滤，直接UTF32数组可能增加临时4N_u | **暂后排**。可bounded tile解码再映射，但当前fill .482s，新增依赖/拷贝须有更强热点证据 |
| [SwissTable原设计](https://abseil.io/about/design/swisstables)；[hashbrown](https://github.com/rust-lang/hashbrown) | control byte fingerprint + SIMD probe | 已用于当前AHashMap；改变hashing不等于换掉scalar链式旧表 | **继续沿用成熟实践**。H profile有SSE2 match_tag实际执行证据 |

### 代码许可、固定来源与实现状态

| 实现 | 核对版本 | 许可/接入注意 |
|---|---|---|
| clausecker/radsort | `f69e816c3cd79d312cd67aea5b9cf1c338c1b371` | BSD-2-Clause；主任务保留作者notice，移植审查必须使用Rust实际pointer provenance |
| klauspost/radsort | `0341b6e9caa4ac560058144dcca8c6cd6a281026` | LICENSE正文为BSD两条款，GitHub API标记NOASSERTION；Go移植保留Clausecker/Post notice；仅作parallel协议参考 |
| RagnarGrootKoerkamp/quickheap | `fdc1de6a6899c4a89aa494f5daa855a42a4fca74`，2026-09-30 | Cargo.toml声明MIT；GitHub license 字段为空，正式vendoring须核对完整license文件/发行包notice；edition2024，依赖ensure_simd/wide等 |
| Xilinion/TPHT | `d021bf85cfba8241f0e01804785e52c8067dcaaf` | BSD-2-Clause；embedded XXH3同许可，需同时保留THIRD-PARTY-NOTICES；standalone自称tentative release |
| saltsystemslab/ZombieHT | `5064a381fd3b7a540594fadd6f7595fdcc748a6c` | BSD-3-Clause；artifact不同变体/子模块要分别检查，不把论文CC-BY当代码许可 |
| mpdn/radix-heap | `8d3ccc17789ca0717b2520bbb19486c2c96f0d04` | MIT；强制monotone insertion契约，见下一节排除原因 |
| orxfun/orx-priority-queue | `1caf7b8c5771ef526aefdebe3f97d81e1028c0c5` | Apache-2.0；论文8叉对照用此库，当前Trainer实际为dary_heap0.3.6，不能把两者当同一实现 |
| simdutf | `cf8715fad4d55c87aad3006a9a82531f740605b8` | 官方提供Apache-2.0或MIT选择；benchmark competition子目录不当通用可用源码 |
| StreamVByte / Rayon / hashbrown / bumpalo | 本文链接其官方当前源；未vendor | StreamVByte官方为Apache-2.0；相关版本须在实际选型时固定。Rayon、hashbrown和bumpalo均可选MIT或Apache-2.0；实际vendoring继续保留相关notice；不在此报告引入依赖 |

### 容易误接的队列与压缩方案

普通无非空 affix compact 路径的旧 pair frequency 单调下降不等于每个 owner 队列满足 radix-heap 的“新插入≤上次取出”条件。别的 owner 选中的规则可向该 owner 产生高于其上次局部取出 frequency 的 pair；full priority还包含canonical tie，reserved ID首次激活会影响tie。没有本地monotone证书时，[radix-heap契约](https://github.com/mpdn/radix-heap)禁止直接替换。全局统一queue需改变owner选择结构，另有证书和成本。

候选堆当前不会为每次减频都追加一份记录：peek只在top旧值不等于ledger时pop+push修正。终点heap比entry多约365万条，主要可由retired key尚未pop解释；不能将它称为无限live duplicate膨胀。按live entry重建堆可清理陈旧key，但需O(K)哈希表扫描及heapify，shrink还可能产生短暂双buffer。它可以列为内存应急工具，当前不是已有时间收益。

posting的immutable birth cohort很适合长表压缩：出生后只过滤失效位置，旧表不需要随旧pair减频重建。可先让长初始posting采用block delta+StreamVByte，保留小表/新生表。但历史位置跨度、低频长尾、AA随机读、解码到u32scratch峰值都要计入；Elias–Fano的理论bit公式不能直接作为RSS节省。此前u16分块字典也会增加重复pair目录，不能只把payload宽度减半。

## 6. 多核并行：现状、可证明扩展与优先级

owner是由canonical pair key的混合hash指定的数据分片，其唯一ledger/posting/heap写所有权贯穿训练。当前`owner_count=config.workers`，merge workers通常为4；`initialization_workers`可独立创建池。owner不会固定绑定某个物理核或Rayon线程。Radsort算法自身不要求owner，当前多份owner stream提供天然独占并行：**sort4已经并发**，install2是限制posting与record同时存活的内存策略。

用户最新方向是已有owner并行时不急于再并行化内核，因此本节保留可行路径和触发条件。

### 6.1 单个大stream的parallel Radsort

论文§4.3给出每一digit round按逻辑block分chunk、每chunk独立head-start/scratch/bucket状态，join后按digit和原chunk次序拼合block目录的协议。顺序拼合保留稳定性；fixup顺序执行，finalize可保持串行。额外scratch随worker数增加，固定block版目录仍O(E/b)。[Go实现parallel.go](https://github.com/klauspost/radsort/blob/0341b6e9caa4ac560058144dcca8c6cd6a281026/parallel.go)实现了该结构，源码正文许可已核对；其128K阈值和最多8worker是作者机器参数，不移植成项目默认值。

Rust迁移最重要的依赖是：本轮每个worker持有互斥physical blocks，其输出只能复用自己已消费的输入块；fixup只能在join之后读全部bucket末端；下一digit必须等fixup结束；最后不能对有覆盖依赖的finalize循环直接par_iter。当前raw-pointer模块的独占证明只覆盖单thread一份state，给每条record循环加Rayon会破坏它。

当owner数量小于可用cores、或一个heavy-owner决定sort尾部时才考虑内核并行。已经四owner占四cores时，嵌套4×4不会创造16cores；新的小scratch、调度和带宽竞争可能使它更慢。

### 6.2 P>W的虚拟分片

让P个owner独立于W个worker，可给Rayon更多commit/sort任务；hash只保证key层面的统计分散，不保证按occurrence字节数或出生量平衡。**单个热pair始终属于一个owner，增加P不能拆它**。初始化不必永久增owner：可用暂态key分桶排序，再将完整pair组路由回现owner，但会增加分区与安装的接口工作。

更大P的成本包括更多map/heap头、round到全ownerhead检查、每job的P份route、count-prefix元数据和容量取整。若prepare保持W个job，route数量O(WP)；若job也增到J，变O(JP)。更多owner的hash容量可能因独立power-of-two取整增加，不能假定sharding只有收益。

最小触发证据为各owner的E_o、retainedK_o、sort/group/install elapsed、每batchcommit耗时和birth bytes；计算`max(E_o)/(E/P)`并看真实阶段尾部。工作下界`max(total_work/W, max_shard_work)`只有测到shard项明显占优时才支持P>W。当前J init/merge约3.4×的4线程加速不证明无偏斜，也不足以要求重组owner。

### 6.3 更细prepare job与stage overlap

当前prepare按posting访问数分约W个连续job；无效历史位置与有效位置的birth工作成本不同，所以相同visit count不等于相同CPU成本。可以有J>W个**有序连续**job并用indexed collect恢复job序号；不能按completion顺序提交posting。由序号保持同key producer的position次序，AA则保留chunk摘要与incoming parity屏障。代价是每job的ID目录、routes、validbuffer与聚合scratch增加；仅在job尾部空转证据下尝试。

不同owner的sort完成后，可在内存许可下启动其group/install；它们只有读稳定corpus的公共依赖。但至少需要限制在途posting容量的budget，并保证先释放已安装record后再接下一任务。当前sort4/install2阶段式结构简单且已测；pipeline可能抢同一memory bandwidth，不能仅用“少一个barrier”推断收益。

全局stable Radsort可把owner分区从排序算法中移走，但单线程全局sort会失去当前并发；parallel全局sort后还要把完整key组装入多个owner。排序结果按key有序不等于owner连续。若再生成owner数组，可能恢复被省掉的一次搬运。因此较小的直接route scatter优先于全局sort结构重写。

## 7. 面向大规模的后续顺序与最小门槛

已集成项目的现有收益保留；以下顺序反映最新的规模目标，候选范围证书与性能证据分开。当前 stable radix/direct route 只在 `bits=32 && corpus.len()≤2^32 && V_0≤65,536` 分支执行。`requested_vocab>65,536` 本身不关闭初始 radix，但 forced alphabet把 `V_0` 推高会关闭。source依据和容量公式见 [SCALE_UP_ANALYSIS.md](SCALE_UP_ANALYSIS.md)。

| 顺位 | 候选 | 时间/内存机制 | 风险与最低门槛 | 状态 |
|---:|---|---|---|---|
| 1 | block fallback 的uniform/sparse delta计数 | `posting.len·w`，或`posting.len+Σ(w−1)`；只对非单位权重hash，保持16B posting value | 0/非unit权重、i64 checked加法、全局floor、跨block；最坏mixed权重新增每key lookup | 已实现16次proxy对照；临时map容量缩小，速度混合；详见初始化报告 |
| 2 | bounded block/batch 初始化 | records由batch预算决定；两遍全局count与保留posting assembly | local/global地址、fullu64key、单表u32length、globalfloor与Q目录；最终O(N+K+Q+P)不消失 | 提案；大样本无法实测，先强制多块oracle及容量诊断 |
| 3 | direct scatter/low-scratch radix的完整宽度和block扩展 | 保留稳定count-prefix写finalslice；删除compact临时搬运 | 当前flat已实现；widecode需8digit与宽record，不能直接去掉guard | flat n=1 route少58.53%、train少9.31%；超界branch尚未提供此收益 |
| 4 | 可回收小posting pool/slab | 退休slot按归属重用，减少累计arena backing | class rounding/free-list/owner回收与生命周期；先量live/retired容量 | 提案；阈值选型待算法完成 |
| 5 | 长冷posting压缩 | delta codec减payload与访存 | as_slice、AA随机索引、block目录、解码scratch；真实gap/寿命样本 | 未实现；比低范围heap更面向大规模 |
| 6 | block summary分wave消费 | 已消费频率vectors立即释放；从全体Q降到Q_wave临时量 | owner独占汇总、递增block directory、稳定producer顺序 | 已实现57tests通过；13block同binary摘要48→16MiB，进程峰值479.94→431.57MiB；初始化+12%、全训+0.55%，各n=1 |
| 7 | route map容量复用/reserve | 减少BirthGroup rehash/分配 | retainedcapacity抬RSS、job几何变化；先观测当前J增长 | H有growth证据，J尚缺当前证据 |
| 8 | owner dictionary TPHT/Zombie | key/metadata压缩与高load更新 | TPHTvalue宽度不符、Rust依赖/PR成本；adapter准备未测Trainer | 研究候选，降优先级 |
| 9 | 额外kernel并行/P>W/pipeline | 降重owner/job尾部 | hotpair不因P增大而拆分；更多scratch/routes与带宽竞争 | 用户已降优先级；仅偏斜/idlecore数据支持时做 |
| 10 | 逻辑block消费者省finalize | 跳过排序后block恢复到连续数组 | 跨blockgroup、两遍访问与iterator；独立成本证据 | 最新fixedT256 probe四owner compact CPU总492.5ms、max134.76ms；train21.280s，理想wall机会约0.63%，降后排 |
| 条件优化，11 | packed8B candidate +wide fallback | 候选backing同capacity减半 | ordinaryaffix +32bitmass+16bitID证书；超界用wide | 已集成、backing实测减半，独立train略慢；为范围内存收益 |
| 条件优化，12 | packedu64上的SimdQuickHeap | SIMD分区与bucket局部性 | ISA/依赖/notice/slack；当前select小且packed条件受限 | 未实现；queue成本真正变大后再比真实trace |

[sort cost probe](results/frontier-sort-cost/zh512m-sort-cost-t256.summary.json) 的CPU总和不可当墙钟节省；上述0.63%只按最大owner并发criticalpath给理想机会，不宣称大样本收益。
可回收pool应按allocator归属而非“当前执行线程”找到free-list。posting在prepare、coordinator和ownercommit之间移动，Rayon不固定线程；直接把释放推入当前TLS可能归还到错误pool。安全的方案可让owner显式回收其slot，selectedposting在join后送回对应owner，或者slot带pool标识并批量路由回收。任意路径的pointer都不能交给std Vec::from_raw_parts释放arena内存。pool空间为各class在各owner上的最大同时活跃量乘class大小，加目录与chunkslack；这只比累计分配模型更贴近live，并不保证小于现有精确Vec空间。

未采用项也属于调研结果：F singleton birth延迟物化、G2自动SIMDblock验证、I直接ownerposting组装、Kparallel drop均有本项目失败证据；它们没有因为换成新论文术语而重新成为优先候选。下一次测量从当前最快有效组合派生，一项局部假设决定一项保留/丢弃，不把与历史慢版本的比较当当前收益。

本报告没有修改Trainer、efficient_bpe或其它agent文件，没有运行构建与大样本计时。新增文件包括此报告、[规模分析](SCALE_UP_ANALYSIS.md)、[候选JSON](results/algorithm-frontier-map/ranked_candidates.json)、[固定实现来源](results/algorithm-frontier-map/primary_implementation_provenance.json)和bounded编码oracle；提出的空间数字是表示推导，正式性能由完整调用决定。

## 本轮从调研到实测的补充（2026-10-01）

full-key有界排序已扩展稳定Radsort到u128记录，不再要求初始alphabet16位。始终排序在英文回退25.2%；8B临时记录仍回退30.3%且中文整训未获益，用户要求撤回。当前029ab45b按实际block字典大小选择：小字典hash扫描，达到65,536项后后续tile稳定分组；中文初始化3.58%/4.09%，13block四块并行4.95%，英文不分配排序缓冲。新增16次完整Trainer调用、最终60lib tests与完整模型/工作gate通过。与flat已有的8B heap条件分支是不同改动。见 [初始化报告5.2](INITIALIZATION_MEMORY_REPORT.md)。

早期DE→H的train中位32.397→25.577s、feed+train中位36.766→29.838s；随后J全量arena已测train19.126s、feed+train23.105s。后续低峰值候选的arena配置已测train16.470–20.954s、峰值约3.33–3.46GiB；历史25.577s不代表当前成绩。GPT-6 Luna随后完成当前512MiB flat主路径的完整PERF独立审计。上次完整DWARF PERF来自H00216d91，两版本分别核对各自binary与源码。当前大项是posting校验、邻边统计和owner提交；未发现高占比且明确可删除的重复工作，按用户要求停止本轮优化。采样未进入generic adaptive分支，不据此判断数十GiB。详见 [当前PERF审计](CURRENT_PERF_AUDIT.md)。
