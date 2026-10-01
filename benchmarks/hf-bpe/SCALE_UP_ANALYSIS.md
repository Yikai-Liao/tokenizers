# 精确 BPE 扩到数十 GiB：地址边界、容量与优先算法

分析日期：2026-10-01。源码基点：`/root/code/tokenizers-worktrees/initial-owner-waves` 的 `8bd048f6`；已完成的六次正式计时 binary 对应 `56dd3227`。本机没有运行 10–128 GiB 训练；下文的容量表是明确条件下的算术情景，不是速度或 RSS 预测。全文算法地图见 [ALGORITHM_FRONTIER_MAP.md](ALGORITHM_FRONTIER_MAP.md)。

用户已将目标定为数十 GiB 及更大规模。**优先扩展有界初始化、完整 ID/频率宽度和生命周期管理；8 字节候选是条件优化，排在这些通用工作之后。** arena 阈值暂保留为对照配置，待算法完成再做选择。

## 1. 原文件大小不能代替算法规模

定义下列量，避免把 raw GiB 直接乘训练时间：

| 符号 | 含义 | 当前对应 |
|---|---|---|
| `R` | 预分词前原文件字节 | benchmark `input_bytes` |
| `U` | 去重后片段数 | `unique_words` |
| `N_u=ΣL_i` | 过滤后唯一片段的字符槽数 | `initial_symbols` |
| `N_s=1+Σ(L_i+1)` | 实际平坦 corpus 槽数，含每片段 separator | `initial_slots` |
| `E_u=Σmax(L_i−1,0)` | 初始物理边数 | `initial_edges` |
| `E_0=Σw_i·max(L_i−1,0)` | 初始加权边质量 | checked `weighted_edges` |
| `K_0` / `K_t` | 初始 / 时刻 t 的全局保留 pair 数 | owner ledger entries |
| `Q_0=Σ_b K_{0,b}` | 初始 `(address block,pair)` 的不同组合数 | block dictionaries 的 entry 总数 |
| `H_t` | 候选 heap 记录及实际 capacity | 包含尚未 pop 的退休 key，不能等同 `K_t` |
| `P_t` | 当前保存的历史物理 posting 位置数 | 可含已失效位置；不能等同当前 token 边数 |
| `G_t` | 截至 t 累计产生的物理 birth 位置数 | 与权重相乘前的真实位置 |
| `A_t` | arena 到 t 的累计申请量及 backing | 包含退休但未回收的 payload 和 chunk slack |

重复已有片段只增加 `w_i`，可能只增加 `E_0`，保持 `N_s/E_u/U/K_0` 不变。加入新文本才会增长物理存储；中文、ASCII、ByteLevel、whitespace 和 none 的 `N_s/R`、去重率及 pair 组合分布各异。预分词边界与 newline 必须保持原语义。

固定身份数量给出有限宇宙：`K_0≤min(E_u,V_0²)`，`K_t≤V_t²`；分块表可多次存同一 key，`K_0≤Q_0≤E_u`，更细分块通常增大 `Q_0`。它们不是固定“每 raw GiB 字节数”。同理，固定 50k vocabulary 下把终点 owner 数按 raw GiB 无限线性延长，会超过 `50,000²`，算术情景本身就不再成立。

普通无非空 affix 路径中，每次实际物理 merge 删除一个边界，并最多生成两个新邻边。令实际 merge 次数为 `m_t`，则 `m_t≤E_u`、`G_t≤2m_t`。若历史出生位置只登记一次，历史位置数有结构上界 `P_t≤E_u+G_t≤3E_u`。这不是分配或 RSS 上界：Vec capacity、临时生产者副本、map 桶、退休 arena 与 allocator slack 需另计。一般 affix alias/cohort ledger 不复用此证明。

## 2. 当前实现的三个独立分支边界

以下条件来自 [parallel.rs](/root/code/tokenizers-worktrees/initial-owner-waves/tokenizers/tk-train/src/trainers/bpe/indexed/parallel.rs:577) 和 [corpus.rs](/root/code/tokenizers-worktrees/initial-owner-waves/tokenizers/tk-train/src/trainers/bpe/indexed/parallel/corpus.rs:118)。

| 条件 | 当前作用 | 超界行为与限制 |
|---|---|---|
| 普通无非空 affix | parallel compact 算法证书 | 非空 prefix/suffix 回退一般串行 ledger；`8bd048f6` 入口及 packed eligibility 都显式检查；affix fallback test 已通过 |
| `bits=32 && N_s≤2^32` | `flat`：posting 直接存 u32 global position | 更大 corpus 使用 `usize` global base + block-local offset + owner 的 u32 block directory；不是地址截断 |
| `flat && V_0≤65,536` | `radix_eligible` | 当前 Radsort/direct route **只在此分支执行**；`N_s>2^32` 转 physical-block hash，不能引用当前 radix 收益 |
| `flat && V_0>65,536` | 完整 u32/u32 pair key 的 flat hash 初始化 | 先 route `Vec<u32>` 位置，再由 owner 读取 corpus 建表；未采用 direct final record scatter |
| `max(V_0,requested_vocab)≤65,536 && E_0≤u32::MAX`，另需普通 affix 条件 | 8 字节 candidate heap | 不成立时用 16 字节 Candidate；ledger frequency 始终 u64 |
| `narrow_corpus && max(V_0,requested_vocab)≤65,535` | u16 corpus slot | 更大 vocab/forced alphabet 使用 u32 slot。65,535 给 separator 留码值，与 pair 压缩的 65,536 数量上界不同 |
| `i64` checked 每片段权重、`w_i·edges` 与全局和 | 当前 compact count 可接受范围 | 超界返回错误；u64 ledger 不代表输入已经支持超过 i64 的加权总量 |
| `ceil(N_s/2^bits)≤u32::MAX` | block directory 可表达范围 | 更大 block 数明确报错；64 位 usize 也仍受实际 allocation/内存限制 |
| 单个 PackedPosting 的 len/capacity 为 u32 | local positions 和 owner block directory | 单表不能装 `2^32` 条；严格小于 `2^32` 的 scan batch 并不能自动保证跨 batch 汇总后的单表满足该条件 |

`V_0` 是 special token 与 alphabet 完成后的完整身份数，包括 forced alphabet。只有 requested vocabulary 大于 65,536、而 `V_0` 仍较小时，**初始 radix 仍可用**，candidate heap/corpus 则可以是 wide。相反，forced alphabet 把 `V_0` 推高，即便 requested vocabulary 小，也必须走完整 key 的初始化。

地址模型已经跨 block：`Plan.position`、corpus 长度和 block base 是 usize；word starts 用 u32 local pivot，跨块词由 `previous_weight` 延续权重；AA parity 跨块继续传播。当前只有 16/32-bit block 配置，不能把任意 batch 长度直接当成已有 address block。middle block 的单一热 pair 也可能占满 `2^32` 个位置，仍需处理单表长度界。

## 3. 超过 flat 界后的实际工作与存储

### 3.1 当前 physical-block hash 初始化

[源代码](/root/code/tokenizers-worktrees/initial-owner-waves/tokenizers/tk-train/src/trainers/bpe/indexed/parallel.rs:676) 按 address block 并行扫描 corpus。每个活跃 block 同时建 `counts: HashMap<pair,u64>` 和 `postings: HashMap<pair,PackedPosting<offset>>`，逐位置 hash lookup、累计权重和 growable posting push。再输出 block×owner 的 `(key,frequency)` vectors，全部 collect 后各 owner 汇总 frequency，并登记 block ID。**全局 floor 在所有 block 的频率相加后才应用**，然后 block maps 删除不保留的 key。

现有代码正确保留了跨 block 累计后才能做 floor 的条件。局部 `frequency<floor` 不能提前删除：同一 pair 在多个 block 各出现一次，floor=2 时仍应保留；零权重片段和混合权重同样需 exact 汇总。

代价包括：逐位置两份 key table 访问、posting 多次扩容、全体 `Q_0` 的 summary vectors、重复的 block-key 目录、global owner map；floor 前已经为所有物理 pair 分配 posting。Rayon 同时执行的 block 数限制活跃 counts maps，但完成 block 的 postings 和 summary vectors会继续存活。因此当前并非 O(一个 batch) 的总初始化临时量，也没有全局 `8E_u` radix record 数组。

临时 `routes`/`frequencies` 在其初始化分支结束时已经释放，再构建候选 heap；这是已有实践。Radsort 路径也在 owner install wave 完成后释放该 owner 的 records。继续优化应寻找剩余重叠，不能再次把这些已有释放算成新增收益。

### 3.2 参数化内存式

`T_s(c)` 表示容量为 c、每 bucket payload s 字节的当前 hash table 估算：`next_power_of_two(ceil(8c/7))·(s+1)+16`。它计 control bytes 和桶取整，但没有 allocator 或 RSS 证明。当前 owner 的 `(key,Entry)` payload 是 32B，block `(key,PackedPosting)` 为 24B，counts `(key,u64)` 为 16B。

```text
flat live ≈ M_feed + c·N_s + M_wordmeta + M_vocab
          + Σ_owner T_32(owner_capacity)
          + offset_width·posting_heap_capacity
          + candidate_width·heap_capacity

block live ≈ flat live（把 flat posting 换成各 block 的 local posting）
           + Σ_block T_24(block_capacity)
           + 4·owner_directory_heap_capacity + O(number_of_blocks)

block init extra includes:
    Σ_active_block T_16(counts_capacity)
    + 16·summary_vector_capacity + growth overlap

radix init extra includes:
    8·remaining_record_capacity + radix_scratch + group_capacity·sizeof(Group)
```

`c=2/4`，posting offset width 为 2/4，candidate width 为 8/16。内联短 posting 已在 Entry/PackedPosting 的 bucket payload 内，不再作为 heap bytes 相加。mixed weights 需 pivot/weight 两列（逻辑每 word start 12B，实际有 capacity），uniform weights 则跳过它们。语料 build 暂存的 word references、region starts/active bitmaps也有峰值；feed 读取是流式，去重字符串表却在训练期间仍保留，`M_feed` 不能删去。

arena 中可用 live payload 与累计 backing 必须分开。ALL arena 退休后不复用，容量随累计申请和几何 chunk rounding 增长；repeat 的 requested 约 1.308 GiB、backing 约 2.158 GiB，不能把 requested 当峰值。可回收 pool 的目标是按 size class 的**同时活跃最大量**复用，而不是再次优化一个固定阈值；其 class rounding、owner 归属和跨 worker 回收成本仍需记入。

## 4. 10/32/64/128 GiB 的条件容量算术

参考行：[zh512m direct/auto/T256 JSONL](results/frontier/zh512m-direct-auto-t256.jsonl)。`R=536,870,289 B`，`U=1,429,915`，`N_u=204,660,029`，`N_s=206,089,945`，`E_u=203,230,114`，`E_0=203,974,788`，`V_0=20,757`，保留 `K_0=2,697,517`。**表中假定新输入保持相同的唯一片段字符/字节密度、separator 比例和权重分布；没有假定 K 或 heap 随 raw 线性增长。** 换预分词、语言或重复率必须重新填参数。

| raw GiB | `N_s` 十亿槽 | `E_u` 十亿位置 | `E_0` 十亿权重 | current 32-bit address blocks | u32 corpus GiB | `4E_u` 逻辑位置 GiB | 假想全量 `8E_u` records GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 4.122 | 4.065 | 4.080 | 1 | 15.355 | 15.142 | 30.284 |
| 32 | 13.190 | 13.007 | 13.054 | 4 | 49.136 | 48.454 | 96.908 |
| 64 | 26.380 | 26.013 | 26.109 | 7 | 98.271 | 96.908 | 193.816 |
| 128 | 52.759 | 52.027 | 52.218 | 13 | 196.543 | 193.816 | 387.631 |

`4E_u` 是把所有初始物理位置按 u32 展开的逻辑体积；不是已分配 posting backing，也没有计 floor、inline、扩容、历史 births。最后一列描述“继续用全量 records”将遇到的额外体积；**当前 >2^32 slots 会回退 hash，不会实际分配这份全局 record 数组。** fixed `V_0` 的 `K_0≤V_0²=430,853,049`，因此不能按 512MiB 的 retained K 任意延长到 128GiB；block 的 `Q_0` 又不受同一单表界限制。

在这个情景中 flat-slot 界约为 raw **10.420 GiB**，packed `E_0` 界约 **10.528 GiB**。10GiB 情景仍能满足数值范围，但要容纳约15.36GiB corpus和30.28GiB records，以及去重输入、owner、posting等状态；“packed 能开”不等于机器能训练。32GiB及以上当前已失去初始 Radsort 路径。高度重复的 128GiB 输入也可能一直处于 flat 物理范围，只增长权重而令 packed 关闭。

更大规模需先测/预计算 `N_s,E_u,E_0,U,V_0,K_0/Q_0`；只用 `R` 不能通过容量 gate。上述表没有外推全训秒数，也没有把小样本 n=1/2 的百分比代入大样本。

## 5. 通用的下一步：让额外 records 由预算决定

### 5.1 先做小范围、可审查的 fallback 改善

当前 block fallback 给同一 key 维护 counts 与 posting 两张表。主任务提出更窄的替换：**posting.len 已提供物理数量，仅对非单位权重累计 signed delta**，无需永久加宽 block posting value。

```text
uniform Some(w): local_frequency(q) = posting(q).len · w
otherwise:      delta(q) = Σ_valid_edges_of_q (w−1)
                local_frequency(q) = posting(q).len + delta(q)
```

uniform 的 w 可以是0/1/2或其他非负值，用 checked u64 multiplication；该分支完全不建 counts/exception map。nonuniform 只在 `w!=1` 的有效边更新 `HashMap<pair,i64>`，然后遍历所有 block postings，用 `delta.get(key).unwrap_or(0)` 恢复 frequency。每个有效物理位置仍加入 posting；零权重贡献−1，不能跳过其位置。key、local offset、扫描/目录顺序、完整全局 reduction 和 floor 位置都保持。

local frequency为0也仍须输出block/key summary：同一key可能在别的block获得正frequency并通过global floor；跳过零local summary会漏掉该block的物理positions。missing delta代表零修正，不代表没有该key。正负修正抵消为0的exception entry可保留，避免重复移除/重建；容量统计仍应包括它。

独立算术审查：既有 build 先检查每个 weight 可转 i64，且非负全局 `E_0≤i64::MAX`。每 block 至多 `2^32` 个位置，所以任意前缀的负delta不小于−`2^32`；正delta不超过这些边的Σw，也不超过 `E_0`。因此 signed delta可表达；仍用checked_add，`len+delta`经checked i64相加与非负u64转换。最终结果精确等于Σw，不需要新的 u32 frequency 证书。单表 u32 len/capacity 界保持原样。

mixed 权重下，收益取决于**非单位权重有效边比例**和它们涉及的不同key数，不是word数。最坏情况全部边非单位、权重又不uniform：exception map的key数可等于旧counts map，逐边两次hash仍在，而且恢复frequency增加每个block-key一次lookup及branch；不能保证所有输入加速。unit-heavy 情景减少计数hash与exception容量。现有 weight-one bucket比例仅表明256槽区域的证书比例，不等于有效边比例。应在已有扫描中采样/统计 `unit_valid_edges / E_u`、`K_exception/Q_0`、exception capacity和完整调用，避免再扫描corpus。

这项适用于dict16/dict32、usize base、完整u64 pair key和既有i64 mass范围；它不扩展 flat radix地址，也不改变一般affix fallback。主任务采用same-binary legacy/sparse对照，用较小输入强制physical blocks验证分块地址模型；已实现并完成56项lib测试、16次same-binary proxy调用；固定词序对照确认临时map容量缩小，但初始化/全训速度混合。详见 [初始化报告第5节](INITIALIZATION_MEMORY_REPORT.md#5-通用-block-路径的实际收获与限制)。

另一个已实现的局部收获是分 wave 产生 block summaries并立即并入唯一 owner ledger，释放已消费 summary Vec，而非 collect 全部 `Q_0` 再 reduce。数值加法可分 wave；每个 owner directory 必须仍按递增 block ID append。它把 summary 临时量从全体 `Q_0` 降为 wave 内 `Q_wave`，owner/block 最终状态仍是 O(K+Q+P)。稀疏计数已实现并完成proxy测量，尚未证明通用加速或进程峰值收益；summary分wave已继续实现并测量：13block摘要48→16MiB、进程峰值约少48MiB；初始化+12%、全训+0.55%，n=1。

先增加 branch、`Q_0`、最大 block pair/position count、counts table bytes、summary capacity 和每 wave 峰值诊断，用现有 16-bit block 测试路径验证跨块语义。模拟地址分块可以验证算法；它不能证明 >2^32 真正分配成功或数十 GiB 的吞吐。

### 5.2 两遍 bounded 初始化，推广稳定 sorting/direct scatter

为每个 scan batch 选 **严格少于 `2^32` 的局部位置范围**，额外 record 预算记为 M，可有 W 个在途 batch。record 保存 local offset，batch descriptor 保存 usize base；full pair key 用 u64，只有初始 ID 条件成立时才用 u32 pair code。scan batch 可小于 address block，但不直接改变现有 block 身份。

1. **第一遍只做 exact 加权频率。** 每 batch 产生有序 local records，以 stable radix 或完整 key sorter 分组，向唯一 owner 汇总 `(canonical key, weighted count)`。各 batch 汇总完成后才判断全局 floor。records 消费后释放，不能 collect 全部 runs 到 RAM。若 hash 部分更合适，也可替换 local grouping，保持同一边界。
2. **第二遍生成保留的 positions。** 读取同一 unchanged corpus，batch 内分组后只为全局 retained keys 产生 positions。按 `(address block,key)` 汇总精确长度，然后直接填最终 local posting；沿 batch 物理序号恢复位置次序，不按 worker completion 顺序拼接。已有 count-prefix scatter 可复用为 final-slice writer。
3. **发布 owner block directories 和 heap。** 每 key 全局 frequency 等于各 producer 的 checked sum；block directory 递增且唯一；每 block posting 的 offset 递增且唯一；候选排序仍按 u64 frequency 降序、完整 pair key 升序。heap 选择在普通/u32 mass证书成立时压缩，否则 wide。

若第二遍在一个 address block 内分多个 batch，不能简单追加并宣称“exact capacity”：需要该 block-key 的总 count 或分段 posting 接口。前者可在第二遍 count-prefix 后再 fill，增加局部扫描；后者改变 `as_slice`/AA/prepare API。若引入较小 address block size，则需扩展目前只允许16/32bit的配置，并核对 block directory/Q 增量。**先复用现有地址块做局部 prototype，再选择接口；这些是当前待解的实现工作，不是假定已经提供的能力。**

期望额外 records 为 `O(W·M)`，不是 `O(E_u)`；sort/group scratch 还受 batch 内 key 数约束。全局 frequency、block-key 计数和最终 postings仍占 `O(K+Q+P)`，若为了记精确 block counts 收集全部 `Q_0`，只能说 records 有界，不能说全部 extra memory 为常数。原始 corpus/word table依旧在RAM，**这不是完整 out-of-core Trainer**。

两遍增加扫描及重新分组，用确定的带宽/CPU工作换较小峰值。逐 owner 重扫全部 corpus会变成 `O(P·N_s)`，不应把它当免费实现。高初始 vocab 完整 u64 key需要8个byte digits，宽record可能由8B升到16B；排序内核、scratch和group索引也须适配，不能仅删除当前 eligibility guard。

### 5.3 若必须 spill：先限定初始化范围

经典 external sorting 以 RAM预算生成有序 runs，再用有限 buffers 做多路 merge；STXXL 的 sorter公开区分 run creation与streamed output。它是可核对的实践来源，**没有建议在项目中直接引入C++依赖**。[STXXL sorter 文档](https://stxxl.org/tags/1.4.0/classstxxl_1_1sorter.html)

为每条记录比较 `(canonical pair key, global position)` 可显式保证 position 次序，即便外部 sorter 本身不 stable；STXXL `sort` 文档也明确不保证稳定。run merge 可按 key 汇总频率并构建 posting，但 floor 决策之前需保存/二次读取相应 positions。[STXXL sort 设计](https://stxxl.org/tags/1.4.1/design_algo_sort.html)

外存的顺序 scan 成本随 `n/B`，标准 sort 的 I/O 量为 `O((n/B)·log_(M/B)(n/B))`；这里 n 是 records、M 是RAM records、B是I/O块records。增加 spill 会增加SSD读写和临时磁盘容量，不能用内存radix的固定pass速度替代外存时间。merge 阶段仍随机访问全 corpus，完整 out-of-core merge另需页/块驻留和posting访问设计。[Vitter 原始综述](https://users.cs.duke.edu/~reif/courses/alglectures/vitter.papers/Vit.IO_survey.pdf)

## 6. 只给工作量与带宽条件，不报虚构速度

对某阶段，设实际 DRAM请求字节为 D、有效带宽为 `β_mem`，必须完成的算术/哈希工作量为 C、有效处理率为 `ρ_cpu`；条件下界为 `max(D/β_mem,C/ρ_cpu)`，串行依赖、随机latency、页缺失和屏障会继续加时。工作量可以按 `N_s/E_u/Q/P/G` 计，不能把某次 train 秒数乘 `R/R_ref`。

例：旧 owner compact 删除的读+写请求约 `16E_u`。在上述 32GiB 情景，该体积约193.816GiB；若另行实测有效拷贝带宽为βGiB/s，**只可写“此体积的带宽下界为193.816/β秒”**。它不是当前fallback已有可删时间，也不是directroute的大样本预期speedup。当前 block radix额外目录约每512records9B，仍随records线性增长，固定scratch不代表总sortingmetadata为O(1)。[Radsort 原论文与接口](https://arxiv.org/html/2607.05302v1)

merge 随历史 posting visits、有效 merges/births和实际规则数变化；vocab固定使规则上界有限，但 posting规模及分布仍改变每条规则工作。多core收益受热pair和owner尾部、内存带宽限制，当前四owner并发不应直接外推到更多core。

## 7. 按大规模目标重排

| 优先级 | 工作 | 可交付的最小 gate |
|---:|---|---|
| 1 | 先测 fallback 的uniform/sparse delta计数；再做bounded waves、稳定 block/batch 初始化、完整 u64 key/local offset路径 | unit有效边比例与exception容量；global floor、跨block AA、0/非unit权重、forcedalphabet oracle；记录W·M、Q、RSS并测完整调用 |
| 2 | 复用已采用的direct scatter / low-scratch radix，释放consumer已消费临时存储 | 各宽度/地址分支单独报告；full key排序及单表u32长度界；不把flat收益写成dict32实测 |
| 3 | 生命周期回收：可回收小posting pool、长冷posting压缩 | live/retired bytes和class/gap/寿命样本；owner回收归属、解码scratch、AA访问证书；阈值待算法完成再定 |
| 4 | 局部summary副本减少，按实际大stream成本再考虑逻辑block消费者 | 当前固定T256 probe的四owner finalize CPU总492.5ms、最大134.76ms；若并发且全训21.280s，理想墙钟机会仅约0.63%，故当前降优先级；不能把CPU总和当墙钟节省 |
| 5 | 真实owner trace后再考虑dictionary或queue替换；偏斜后再加并行 | 当前宽entry、ISA、实际select成本；真实hotkey/owner工作量，不依据论文headline |
| 条件优化，后排 | 8字节候选、u16 corpus、低vocab目录等范围特化 | 每个range单独证书和wide fallback；对大规模权重/ID超界不承担通用收益 |

已完成两项局部收益保留在报告中；重新排序改变的是后续投入。T256 两次峰值较低、ALL 两次较快但一次超过历史B2预算，属于本机已测事实；用户最新要求将阈值选择留到算法完成，本报告不将T256宣称为全规模最优配置。
