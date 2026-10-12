# BPE 训练优化实施与实验方案

2026 年 10 月 12 日 · 面向精确模型一致性与可维护性的下一轮实施

## 结论与实施顺序

最值得做的不是再换一层 Positions 表示，而是分别验证三种实际工作削减：

1. **把邻居目录的生命周期延长到一次训练 attempt**，消除反复初始化越来越大的 ID 目录。这是最小、最独立的控制。
2. **在一个 producer 的所有任务完成后发布完整 birth key**，把 birth 的 owner 路由从碎片数 H 降到存活完整键数 K_live，并移除该接缝的比较排序。先在现有单 cohort 上验证。
3. **只消费窗口入口旧 ID 的 joined 窗口**，中途只维护旧键损失，最后创建一次 newborn Positions。这才是删除中间 birth 历史的算法变化。

窗口先试较小的 FRONTIER：第一遍只产生 accepted starts 和 old losses，不计算 newborn family 摘要。只有真实数据表明需要更长窗口或整 producer 稀有性剪枝时，再以 exact M 替换它作实验。**不预设同时维护两套长期运行路径，也不先造新 Engine、公开 Builder/Chunk 或异步调度器。**

中文 Whitespace 当前主要耗时在初始化索引，另开 count-first 与 materialize 融合的独立实验有充分理由；它不能记作窗口或 commit 的收益。动态频率下界和异步选择性就绪都有可用数学结论，但应排在上述可独立归因的实验之后。

本方案依据只读源码审查、作者已发布的原始实验数据，以及独立的加权列表、边界与事件数学模型。**这些提案尚未由本次研究原生实现、编译或 benchmark；所有速度、分配和 RSS 改善均待验证。** 证明、反例与证据范围另见《BPE 原理证明与证据附录》。

## 一 当前基线与真正要优化的阶段

以 [890ea1f58e65f8d8503f7cd3fc97ad1651eaf284](https://github.com/Yikai-Liao/tokenizers/commit/890ea1f58e65f8d8503f7cd3fc97ad1651eaf284) 为公开代码和证据锚点。02:32 UTC 的树核对确认 corpus、index、merge、mod、positions 五个核心文件与本轮 a2b20f32 审查基线相同；最新提交记录的是右 token prefetch 的拒绝实验，并未采纳该候选改动。

作者四 worker、256 MiB 输入的当前基线均值如下。它们是作者测量，不是本次重跑。

| 工作负载 | train wall | prepare wall | initial index wall | prepare / train |
|---|---:|---:|---:|---:|
| 中文 Whitespace | 20.742 s | 2.790 s | 10.925 s | 13.45% |
| 中文 ByteLevel | 22.841 s | 9.219 s | 3.147 s | 40.36% |

[原始运行数据](https://github.com/Yikai-Liao/tokenizers/blob/890ea1f58e65f8d8503f7cd3fc97ad1651eaf284/experiments/bpe-simplification/evidence/prefetch-right-20261012/all-normal.json)中的 `index` 是初始索引构建；`prepare` 是一次 `batch.prepare` 的 joined wall，已包含在 merge loop 中。**merge 减 prepare 不是 commit 时间**，还包括选择、写入、记录和循环开销；prepare 也不全是可以删掉的 birth 工作。

当前 fresh `Positions::concat` 已经按顺序转移压缩叶子与 payload，**不解码、重编码坐标**。新方案可省的是本不该创建的叶子、碎片处理与后续失效工作，不能再宣称省掉 concat 的坐标重编码。

作者拒绝右 token prefetch 的决定有数据支持：四 worker 英文 Whitespace 两种运行顺序都回退；中文 ByteLevel 的 prepare 收益很小。该结论只针对那一个 hint 改动，不能推出端点访存不重要，也不构成新方案的性能证据。

## 二 术语与全程不变量

- producer：一个被接受的合并规则及其唯一输出 ID；它可以有很多并行 position tasks。
- cohort：现有选择器接受的一段兼容优先级前缀；自合并 AA 保持单独 cohort。
- window：若干仍然逐个 join 的 cohort，共用固定旧输入 ID 边界，延迟发布 newborns。
- floor F：有效最低频率，至少为 1；优先级是精确加权 count 降序，再按 pair ID 字典序升序。
- birth：合并后新出现的相邻 pair；old loss 是被破坏的旧 pair 的精确加权损失。

实施时不可为了窗口放松下面的现有约束：

1. 每次迭代仍选兼容 cohort，仍从当前精确堆头开始，不跳过冲突头去找“更好并行”的规则。
2. 保留 pair 内并行，即使本批规则数 B 小于 worker 数 P；一个巨大 producer 仍可占满任务队列。
3. 保留 batch 全局任务目标和原有分片规则；在大小上限未触发的普通情形，每 producer 至多约 P 片，AA 至多约 4P。不要每个 producer 重新按 N_r/P 切成 P 片。
4. count 始终来自现有标量计数和 checked 增减；**任何优先级或下界都不得靠扫描 Positions 重算频率**。坐标数量也不是加权频率。
5. 零权重出现仍参与物理改写；完整存活 key 的全部坐标必须保留，包括贡献为零的碎片。
6. birth 长度资格沿用源码的 span 和严格小于上限语义；初始 pair 的既有长度例外必须保留。
7. 实际停止条件是 `vocabulary.len()` 达到 target，不是规则数、reservation 数或假定的 256 字母表。
8. checked u64 溢出、codec/坐标错误和所有已启动任务的错误仍传播；成功输出一致不等于资源失败顺序完全一致。

## 三 独立控制一 复用 attempt 内的邻居目录

### 要删除的工作

现有准备使用 `map_init(Directories::default, ...)`，每个实际进入 prepare 的新状态把两个 `Vec<u32>` 填成 `u32::MAX`，长度为当前 ID domain V。一次状态首用约填 8V 字节；map_init 是 Rayon job/folder 局部状态，并非保证每 worker 恰好一个。

即使每批只有一个状态，完成 K 个新增 ID、每批至多 C=256 条规则，也存在跨批累计 `Ω(K²/C + A₀K/C)` 的逻辑元素初始化。A₀ 包括已有但尚未激活的预留 ID。此项独立于边界事件数量，不应被“每次合并只改常数条边”的摊还账本遗漏。

### 最小接缝

在现有 `train_attempt` 的 materialize 之后、merge loop 之前创建 attempt-local scratch owner，例如 `ThreadLocal<RefCell<Directories>>`；只向现有私有调用链传共享引用。没有新增 Engine 或 Corpus 字段的必要。

每个顺序 `prepare_job` 借用当前线程目录，返回前释放。后续顺序 reducer 若使用同一目录，也必须先完成自己的 inner join，再借用，且在再次调用并行任务前释放。**不可持有 RefCell 借用去等待嵌套 Rayon 工作。** thread_local 如被选用，需要直接依赖声明；不能假设间接依赖已经可导入。

现有目录会验证缓存序号是否仍指向本次 groups 的同一邻居，因此无需每 job 清空或新增 epoch。ID domain 单调增长时只初始化新增尾部。令 H_slot 为整个 attempt 实际参与的 scratch slots，总逻辑填充至多 `8 × H_slot × V_final` 字节；`H_slot≤P` 依赖固定线程池和非重入约束。

### 代价与验收

- V=50,000 时每参与线程约 400,000 字节初始化元素，尚未计算 capacity；16 个 slot 约 6.4 MB。
- 缓冲跨 commit 和空闲阶段驻留；TLS 查找、RefCell 检查、扩容复制、capacity slack 和最终释放均需计费。
- 在 attempt 结束及原输入 reuse restart 时释放；同一 ID domain 的一次 live reindex 可以继续使用。
- 先统计实际初始化状态数、填充长度、线程数和扩容次数。O(NV) 是 split-local 状态的粗最坏上界，不是当前负载已发生 N 倍浪费的证明。
- 可用 `with_min_len` 作更小的调度控制，但它只减少批内状态数量，不能消除跨批反复填充，也可能削弱负载均衡。
- 接受条件是独立对照证实总 wall/CPU 改善且长期 scratch 内存可接受；逻辑字节数下降不能代替计时。

## 四 独立控制二 先完成 producer 再发布 birth

### 单 cohort 就能验证

保持现有 prepare、apply、任务布局及 Positions。apply 的私有结果保留 replacement ID；按现有任务顺序聚合相邻 producer 的任务结果，然后并行处理不同 producer。

每 producer 用左右两个 validated direct-ID directory，以“方向加最终邻居 ID”定位稀疏组。按原始任务和空间顺序移动非空叶子，checked 累加完整 count；完整 key 达到 F 才调用一次现有 concat。old losses 仍按原语义送到 owner。producer 不是固定 worker 绑定；owner 仍负责旧计数分片的独占更新。完整 birth 归约可以从 owner 剥离，本实验并未证明或要求彻底删除 owner。

新增一个狭窄的 fresh complete-Candidate 插入接缝：owner 直接插入完整 count 并转移 Positions，所有 owner 完成后再发布 heap。**完整 Candidate 可能已有 Root，不能改装成 Change 再送进只接收 Leaf 的 concat。**

### 真正省下什么

设 H 为非空 birth 碎片，K_live 为过 floor 的完整 birth keys：birth 的 owner hashes/routes 从 H 减到 K_live，fresh owner 的对应比较排序消失。grouping、checked 求和、floor 判断与 concat 被移到 producer，并没有消失；坐标编码量与最终 payload 也没有因此下降。

收益主要来自**同一完整 key 分散到多个 jobs 的碎片**。H≈K_live 时几乎没有路由压缩，只是归约位置改变和排序方式改变。不要靠加密分片制造一个看似有利的 H/K 比。

若 L 个原 Change 有 old loss，E 个 Change 的 loss/birth 恰好同 shard，原 pending actions 为 `L+H−E`，分开后为 `L+K_live`。所以总 action 数减少量是 `H−E−K_live`，甚至可以为负；`H−K_live` 只适用于 birth 路由。窗口最终 birth-only drain 没有这一同帧 coalescing 项，之前的 losses 仍须独立计费。

### 新尾部和 staging 风险

每 producer 的归组至少有 O(H_r)，混合 Change 扫描可能是 O(G_r) 的顺序尾部。一个巨大 producer 可能失去原先按 owner 并行排序的分散性；减少总工作不保证缩短临界路径。

先试一遍处理、singleton-inline 的叶子暂存。每 key 一个普通 Vec 会在单碎片键很多时新增大量分配。只有测到 staging 分配确实重要，才考虑两遍 flat 方案：它省 per-key staging 分配，但增加第二次目录访问、prefix/range 处理、H 个初始化槽和搬运，且仍保留 concat 的最终 child storage。

保留当前 batch 全局任务粒度、原始 key 内叶子顺序、零权重叶子和 checked count。不要增加公开 Builder/Chunk、通用 scatter 框架或新 Positions 枚举。独立 publisher 先与旧 publisher 比较，胜出后才允许与窗口组合。

## 五 核心算法实验 延后创建中间 birth

### 窗口规则与最小状态

窗口入口固定旧 ID 边界 B；所有窗口输入 ID 小于 B，输出必须是唯一新追加 ID，且窗口中不得消费这些输出。局部状态只需 B、按 cohort 排列的输出描述、accepted-start/job 缓冲、一个汇总 guard 和 retained-capacity 计数。

**256 是整个 window 累计接受的规则上限，不能每 cohort 重新获得 256。** 下一 cohort 只能使用窗口剩余规则容量，同时受实际 target 余量限制；若容量截断 compatible prefix，就在这次 cohort 闭合后结束整个 window，不得再把同一 plateau 切成下一批。

采用**不查询原始 witness 集合 K0**的最小版本。旧键只会减少，不会重新出生，因此可保留现有 selected-key 删除、checked old loss、低于 floor 删除和 absent-key lazy heap repair。无需 witness snapshot、tombstones、retirement log 或额外清理阶段；这删掉的是提案中的新机械结构，并非宣称删掉现有源类型。

不做 witness 查询时，最终重放无需只为它保留 original ports。输出 ID/cohort 与正常 span 描述仍需活到所有最终读者结束。省去 witness 过滤可能多处理一些最后会低于 floor 的 family，但 exact M 的 at-floor guard 与整 producer 拒绝结果不受影响。

### 每次迭代仍然同步

1. repair/peek 精确 old head，检查先前 cohort 的汇总 guard、兼容性、window 剩余累计规则容量和实际目标边界。
2. 在上述检查通过后才 resolve identity；只有确认为真正唯一追加输出，才在当前窗口 take candidate 并 activate。reserved/alias 进入下述 drain/restart 分支，不能无条件激活；guard 失败不能留下 vocabulary/alias 副作用。
3. 完成现有兼容 cohort 的 preparation，join 全部读者；随后 apply 并 join；再提交 old losses 并 join。
4. 只在这三个闭合点之后考虑下一 cohort，保留每个 producer 的 pair 内并行。
5. guard 失败、旧堆为空、规则/内存边界或 identity 边界均触发正常 drain；不得把“旧堆空”当作没有隐藏 births。
6. drain 从最终邻居产生一次 birth，完整归约并发布后，恢复普通精确选择。

guard 失败只拒绝尚未接受的堆头，已接受工作不回滚、不重新 prepare。其直接检查很小，但新内核已经承担的 replay、family reduction 和驻留开销不能被说成“只有一次比较”。

inactive reserved ID 不满足追加 ID 的 tie 证明：先 drain，再按现有 singleton 路径重新验证处理。已激活或本 attempt 先前已产生的输出 alias 保持**从原输入开始的 reuse restart**，并丢弃本次窗口与 scratch 状态。不能提前处理一个尚未通过 hidden-priority guard 的 alias。

### 边界归属与最终发射

旧输入窗口内，一条原始 exterior cut 的两个端点至多各变化一次。跨 cohort 时较早 producer 拥有它；同 cohort 时物理左侧 producer 拥有它。

对输出 X 的最终邻居 Y：右侧 Y 若也是窗口输出，只有 `cohort(X)≤cohort(Y)` 才发射；左侧则只有 `cohort(X)<cohort(Y)` 才发射。XX 只由物理左 X 的右端发射，不能两端重复计数。

例：AB→X 先于 CD→Y，词 CDAB 最终成为 YX；拥有 YX 的是右侧较早的 X。不能用“小 ID 优先”或“最终总选左 producer”代替时间归属。

最清楚的正例是 n 个不同、单位权重的 ABBCD_i，A<B<C<D_i，F=2，target 留有空间。AB 与 BC 初始 count 都为 n；先 AB→X，再 BC→Y。正确 tie guard 允许 BC 胜过隐藏 XB。eager 创建 n 个 XB 再创建 n 个 XY；窗口只创建最终 n 个 XY。

这删除的是同一 cut 的**两版变一版**，不是任意长 birth 链。每 cut 最多两版，且同 cohort 的两端改写早已被当前源码合并，不能再计一次收益。加入整 producer 稀有性拒绝后可以完全不发射其最终版本，那是另一来源。

### FRONTIER 作为较小的首试内核

普通非 AA cohort：在 selected keys 已取走、但本 cohort 写入和 old losses 尚未发生时，捕获精确剩余 old head 的 `(u, pair.head)`。每个可选择 newborn 都有当时的未选旧边界 witness，因此该快照就是隐藏 birth 的优先级上界。

后续 old candidate `(n,(a,b))` 通过单个摘要的条件是 `n>u`，或 `n=u 且 a≤head`；多个 prior cohort 取最高 count，并在并列时取最小 head。AA 用其 selected raw count 与输入 head 作保守 fallback，不能沿用普通 empty-frontier 结论。

第一遍只保留 accepted starts 和唯一归属的 old losses。右侧只需原 next ID，不再预测 selected-right replacement，也不作 newborn 长度判断或 family 计数；左侧同 cohort 前驱抑制和更早输出的归属抑制仍必须保留。最终长度检查留到 drain。

若普通 cohort 的 pre-write frontier 为空，可在 apply/readers 闭合后跳过它的最终重放；这证明没有可选择 births，不是不存在物理低频边界。AA 不享有这个 shortcut。

FRONTIER 的保留 guard 把后续 admission 限在同一个 count/head plateau；按现有 compatible-prefix、AA 规则和上述累计窗口 cap，且容量截断即结束窗口时，**一个窗口至多四个 cohort**。它无第一遍 birth-summary 开销，但可能过早 drain，也无法识别一般的 M<F 分散稀有 family。

### exact M 作为更强而更贵的替代

第一遍计数每个 producer 的完整 coherent families：固定输出、方向、coherent 邻居；只计唯一归该 producer 所有、且 coherent 端点 span 满足当前 birth 严格长度门槛的 families。length-ineligible family 不计入本节定义的 exact M；初始 key 的长度例外仅留在旧 frontier。跨其所有 jobs checked 相加，得到 `M=max_f Σ_j mass(j,f)`，并只在全局最大 family 中求最小 tie key。左右 direct directories 足够，不需要排序、hash family map 或持久 eligibility 表。

摘要可压成 `(M, ell)`：ell 是全局最大 family 中最小的旧左邻居 ID，若不存在取 B。old `(n,(a,b))` 通过条件为 `n>M`，或 `n=M 且 a≤ell`。family 后来可分裂成多个最终 key，却不能合并成另一个 family 的共同后代；因此上界有效，但不必等于最终最大 child。

M<F 时，这个 producer 的全部最终 children 都不可选。完成必要读者、apply 和 old losses 后，可释放 starts，跳过整次最终重放；保留输出描述供其他 producer 判断邻居归属。此收益即使只有一个 cohort 也成立。

一个稀疏正例：A=0、B=1、D=2、c_i=3+i，500 行 ABc_i:1 与 500 行 DBc_i:1，再加 c₀A:2，F=2，无长度限制，target≥507。AB500/DB500 形成兼容 cohort，Xc_i/Yc_i 各仅为 1，故两 producer 的 M 都为 1，可省 1,000 个 birth 坐标 append 及其碎片处理；普通 pre-write frontier 仍有 count=2 的旧 witness，FRONTIER 不能同样整批提前拒绝。这不依赖高重复权重，也不是省掉 1,000 次 heap 分配或 heap push：这些完整 births 本来就会在 floor 处被拒绝。

相反，一个 producer 有 n 个 mass=1 family 和一个 mass=F family，M 仍为 F。最小方案不能跳过它的 n 个低频 family，哪怕“稀有 family 比例”接近 100%。必须测整 producer 被拒绝的 starts/bytes 份额。

exact reducer 每 producer 是 O(H_r) 顺序扫描，另有 O(K_r) 稀疏标量空间和目录访问；随后存活 producer 仍要最终计数/发射。不要把原 map_init 的目录误写成 reducer 自动可复用的分配。

**源码陷阱：**现有 `Neighbors::finish` 只保留 removed_weight 非零或 Positions 非空的记录。禁用 push 后必须显式保留正 born_weight，否则会丢掉 positive birth-only family。ABCD:1、AB:3、CD:2 的兼容 AB4/CD3 cohort 会产生分离的 XY1 birth-only 记录，正好暴露此错误。

## 六 删除工作与新增契约的账本

| 变化 | 实际可删工作 | 仍在或新增的工作 |
|---|---|---|
| attempt scratch | 重复全域 sentinel 初始化和相应分配 | 每 job 查询、增长复制、长期容量、最终 drop |
| producer 完整发布 | birth fragment owner 排序与 H→K_live 路由 | 归组/求和迁移、叶子 staging、producer 顺序尾部和 join |
| FRONTIER 窗口 | 中间 birth 构造、计数、叶子、路由及实际后续失效处理 | starts 重放、最终一次发射、旧损失与各 cohort 三个 join |
| exact M 窗口 | 上述中间版本及整拒绝 producer 的最终发射 | coherent-family 完整归约、存活 starts 重放、顺序 reducer 尾部 |
| 无 witness 过滤 | 新增 K0/tombstone/log/清理需求 | 部分低频无效 family 可能继续处理到最终 floor |

Batch、Corpus、PairIndex、匹配器、ordered ReadTasks 和统一 Positions 仍存在；reuse 的有序 signed 更新路径也仍存在。绕过 fresh birth 字段不等于相关 Change/Event 类型已从整个系统删除。

设 N₁ 为一次变化的最终 owned cuts，N₂ 为在不同 eager cohorts 变化两次的 cuts，N=N₁+N₂。无其他过滤时，eager 有 N+N₂ 次 birth 计数/编码，FRONTIER 最终有 N 次。exact M 另有 N 次 family tally，若整拒绝覆盖 N_R 个最终事件，则最终只计数/编码 N−N_R 次。

因此 exact M 的编码削减为 N₂+N_R，而额外标量事件净数为 N−N_R−N₂；FRONTIER 没有那一遍 family tally。实际 length/floor 筛选、碎片布局、stale decode 和 allocator 会改变成本，需按真实事件计数，不用统一“少 30%”假设。

窗口新增五项公共契约：旧输入/唯一输出、跨 cohort 归属、starts/描述生命周期、完整 drain、资源触发 drain；另加一种 guard 契约。exact M 用更强的 family 完整性与 tie 证明替换 FRONTIER，并增加整 producer 提前释放条件。目录与 publisher 也各有独立生命周期/排序契约，不能因组合在同一 patch 就视为免费。

## 七 内存是独立否决条件

保留 raw starts 会把源实现“apply 后释放”的缓冲延长到最终发射。以 ABBCD_i 正例 n=500、P=4 为例，额外一个 cohort 的 starts 已用空间为 4,000 字节；省掉的 500 个短 gap XB 坐标四片合计仅 560 个编码已用字节，尚未计两边容量与描述。工作减少并不推出峰值内存下降。

禁止整窗口保留完整 Batch 或 selected 历史 Positions。当前 FreshSnapshot 借用 Batch，历史候选可在本 cohort prepare 返回后释放；想提前按 producer 释放需单独改所有权，不能假定已经实现。

同样，`clear`、`retain` 或 `mem::take` 内层 payload 不会释放外层 Change Vec 的 backing capacity。完成 old losses 后消费并释放不再需要的 buffers，只保留必要 starts/job 记录；capacity 留着就继续计费。

第一版可用无固定参数的相对 drain：H 为本 attempt 已完成 cohort 中最大的 starts 与 job-container 实际容量费用；继续之前保留 R≤H，下个 cohort 闭合后更新 H，若 R>H 就 drain。checkpoint 上旧 R 加新 cohort 费用至多 2H，另加描述空间。

**2H 只是 joined checkpoint 的持有容量界，不是总 HWM 或 RSS 界。** 单个大 cohort 可大幅抬高 H；AA canonical vector、历史候选、scalar groups、directory、任务收集/扩容以及最终 payload 重叠都另计。256 是规则上限，不是字节上限；H 不应每窗口重置后再宣称可支持长窗口。

若用户资源目标要求硬上限，需要在消费候选和激活 ID 之前，依据候选坐标上界及真实容器增长规则估算新增容量；仅用 `8×Positions.len()` 不够。超大 singleton 不能靠反复 drain 缩小，必须明确允许一次超限、报资源错误或另做更深的分块算法。

若 raw starts 实测不可接受，再独立试统一的现有 Positions 存 starts；它会增加一次 accepted-start 编码、apply 解码和最终再解码，并需要 iterator apply 接缝。不要先引入 raw/compressed 双模式、专用新 codec 或回放全部历史 candidates。

## 八 动态频率下界 单独验证和推进

“最终第几个 merge 的频率至少多少”必须有剩余全过程的存活证书；target 本身或当前第 K 大 count 都不够。竞争 pair 会互相破坏，当前高频集合可能很快消失。

只在**完整、fully drained fresh frontier**上建立证书。令 T=target、V=实际 vocabulary.len()、R=从未激活的现有预留 ID 数；仅在 T>V 仍需训练时，剩余规则安全 horizon 为 `H=(T−V)+R=T−已激活ID数`。T≤V 时不作无符号 T−V，也没有未来末条 merge 需要认证，但仍保留初始计数校验。使用 activation history，不能用当前有正权重支持的 ID 数。

例如预留 AB，输入 AB:5、CD:1，T=6，初始 V=5：AB 激活预留 ID 不增加 V，随后 CD1 才填满 target。只按剩余一个 slot 得出 floor=5 会错误删掉最后规则。

证书可以来自阈值图的未触及 witness、定向角色 hit budget 或 frozen-key excess-mass depletion。数学细节见附录。简单匹配证书通常需要约 `2H−1` 个互不占用端点的 witness，初始 256 个 active symbols 不能据此许诺 50,000 词表 horizon。

未 drain 时不能仅扫描“可见旧键”立证书。定向反例的正确选择频率为 `[12,12,1]`，遗漏 hidden birth 后错误认证 floor 2 会只剩 `[12,12]` 并 underfill。精确可见 old counts 不等于完整 frontier。

有效 floor L 允许淘汰完整 count<L 的候选索引与 payload，也可在完整 birth count 或完整 family 上界<L 时省其创建；等于 L 必须保留。它不能删除其他改写仍需的 corpus 端点，不能忽略存活 key 的零权重坐标，也不能按任务局部小计作拒绝。

当前 heap candidate 拥有 Positions，单删 count-map key 只使候选失效，旧 payload 通常要到 `best` 弹出或既有 retire 接缝才释放。若要求立即回收，需显式队列筛除/重建或可复用的回收接缝，其遍历、heapify 和 drop 成本另测；提高 floor 本身不能当作即时 RSS 下降。

同一 target、同一 fresh attempt 内 floor 只保留或提高。后来较弱的证书不使此前证明失效，也不需要降 floor。若 horizon/语义被改变，降低标量阈值不能恢复已丢信息；须完整重建或从原输入按安全 floor restart。进入 reuse 时撤销证书及其有损索引。

只在 commit 才提高 floor，通常已经晚于 prepare 的 Positions::push，省不到这些 append。要提前省写，必须把完整上界接在 append 之前；不增加一遍 Positions recount。证书扫描和 wide arithmetic 自身也要计费，证明算术溢出应保守放弃证书而非误认证。

## 九 中文 Whitespace 的初始化专项

当前 initial index 占中文 Whitespace train 约 52.67%，值得与 merge 优化并行开发、串行测量。最明确的候选是：第一次只做 exact pair count，完整聚合决定 admission；第二次在现有 materialize 遍历中写 token，同时只给已准入 pair 构建 Positions。

这仍是基线已有的**两次完整 symbol 遍历**，不是另外加第三次。代价是新的 phase boundary、count 和 positions 分开路由，以及 materialize 中的 admission/local lookup。可选简单 global-admission lookup，或先把 range count directory 消费转换成 admitted Positions directory；不要同时造两套生产接口。

拒绝 key 的压缩 append、叶子分配/drop 可以真正删除。所有 key 都被准入时则新增 lookup 和路由却几乎不省 payload；必须有 all-admitted 负控。

峰值也可能回退：resident token plane 提前与 admitted fragments、暂存目录及 concat 共存。基线原本把 index 临时数据结束后才 materialize；不能只比较被删掉的 rejected bytes。

保留 exact `ids.symbols`、range 顺序、affix/初始 ID 行为、零权重坐标和 checked 初始总数。target 已满足的 zero-merge 调用仍需要既有计数校验。materialize 加入 fallible append 后需返回 Result，不能泄露半成品 corpus/index；reuse 沿用原路径。

## 十 异步就绪 只作为后续研究

没有“所有 producer 永远都必须等”的一般定理。对未选 old q=(u,v)，可能改变它的较早 producer r=(a,b) 满足 `u=b 或 v=a`；等这些相关 loss 真正闭合即可让 q 的数值精确，其他 old counts 可暂为乐观上界。共享头或共享尾本身不等于冲突。

另有实质性的渐进 birth guard：选择时固定 raw count f，每个已完成 job 计算 accepted contraction mass m_j 和完整本地 child 最大值 c_j，保持 `U=f−Σ(m_j−c_j)`。以一次非负 credit `m_j−c_j` 连贯发布，不能先减 m 再加 c 暴露假低值。它不额外扫描 Positions，但增加 accepted-mass 求和与完整 local-max 摘要。

一个符合源码四 worker 分片的独立模型正例中，初始 AB3000、冲突 BA2000、独立 EF1600。AB 的 1,500 个坐标分为 384/384/384/348；前三 job 各有 m=768、c=256，完成后 U=1464，BA 的剩余 upper count 为 464，EF1600 已能在最后一个 job 未完成时数值获准。它越过的是现有选择器真正会停下的 BA 冲突，不是本来就能同批的两个兼容 pair。

这给出可重叠的正面机会，但不是吞吐提升。EF 的 loss 必须确已不受未闭合 producer 影响；还需真实物理读写 P/W readiness、所有 future tasks 预登记、metadata 可见性、消息/ACK 生命周期及错误收口。prepare 完成、write 完成、loss delivery 完成是不同事件；一项 credit 或暂时零任务数不能代替最终闭合。

因此当前结论是“存在足够的选择性依赖条件”，不是已证明一个原生异步 scheduler 正确或更快。第一版 joined 不引入这一组状态、原子计数、唤醒和共享 map 生命周期契约。

word-clipped 读区间可减少部分物理假冲突，但 fresh 基线 materialize 后主动释放 word starts。保留它需每词一个 usize 的 O(words) 常驻空间和每非空 job 两次 O(log words) 查找；不是免费利用现成目录。对真实历史 same-pair posting 的 seed/tail bounds，write clipping 通常不再缩小范围。

离线最终 ByteLevel 模型可分为中文 228、英文 231 个 old-input-only 窗口，却仍包含 1,585、1,338 个兼容 inner cohorts。它没有当时的 count、未被选择的阻塞头或物理坐标，不能当作 FRONTIER 可达到的窗口数，更不能推断这些 inner joins 全被删除。

## 十一 实施与实验安排

### 阶段 A 建立可归因基线

固定提交、编译参数、输入 hash、worker/CPU 绑定、target、floor、affix、长度策略和终端行为。先加有明确开关的计数/阶段诊断；计时版与重诊断版分离，并记录其开销；原生测试与 benchmark 应在获得实施授权后执行。

分别做 scratch-only 与 publisher-only 候选；都保持现有 eager emitter。独立有效才组合。多个实现分支可用不同 worktree 并行开发，但 benchmark 及竞争性编译串行，避免相互抢 CPU、带宽或内存。

### 阶段 B 窗口的三臂对照

- E：现有 eager birth kernel。
- S：候选 deferred kernel，但强制每个现有兼容 cohort 后 drain。
- W：同一个 deferred kernel，按已证明 guard 允许多个 cohort。

三臂必须使用同一 publisher、同一 scratch 生命周期、相同资格/终端政策，尽量保持任务规则。S−E 暴露 staging/replay 基础代价，W−S 识别跨 cohort 延迟的增益；不能把新 publisher 或 terminal elision 混进 W 后全部记为 window 收益。

先跑 FRONTIER。只有 guard 保守停止频繁且最终 drain 的真实 child priority 显示明显机会，或全 producer 稀有性有显著物理工作份额，再试 exact M。若 exact M 见效，可加一次“同 M、关闭整 producer early-prune”的诊断，分离更强 admission 与稀有性剪枝。诊断不等于要保留长期模式矩阵。

### 阶段 C 精确性与对抗形状

用简单独立 eager 加权 token-list oracle 比较完整有序 merges、每步 pair/count、ID 顺序和最终 corpus；再比较原生完整 vocab/merges 模型。至少包括：

- ABBCD_i 正例、没有跨 cohort 改写的零收益例、单巨大 producer、同 key 多碎片和 H≈K 单碎片例。
- AA 长 run、raw count 与非重叠 replacements 不等、跨 task ownership、XX 单端发射、反向 earlier-owner YX。
- 零权重长词和存活 key 的零权重碎片；局部低频但全局达 floor；positive birth-only `finish` 陷阱。
- 同频 lexicographic ties、initial 长度例外、严格 birth cap、target 已满足、目标不足和隐藏 births 下的 empty old heap。
- inactive reserved singleton、已激活/计划输出 alias、affix、原输入 reuse restart、多次 train/reuse 调用。
- checked u64 局部/全局溢出、u32 ID/ordinal 边界、narrow/wide 坐标转换、无效 codec/位置和 started-work 错误传播。
- 内存 drain、超大后续零权重 cohort、stale-heavy candidates、empty/tiny jobs、早期大 H 后大量小 cohort。

测试取短而清楚的名字，复用现有 helper 与参数化覆盖；保持既定测试预算不超过 1,500 行。不要通过删除现有正确性覆盖来凑预算。独立数学模型不替代 Rust 借用、并发、allocator 和真实 corpus 测试。

### 阶段 D 测量范围与重复策略

先以四 worker 做针对性成对比较：中文 ByteLevel 看 prepare/window，中文 Whitespace 看 initial index，并检查英文两种预分词负控。使用既有 256 MiB 中英文 ByteLevel/Whitespace 输入及固定 prepared-input hash。

预算允许时做两组成对、第二组反向顺序；只有接近噪声、两次方向相反或阶段归因不清时再加重复。没有“每项必须五次”的机械要求。确认有意义的候选后，才扩到 1、4、16 workers；16-worker 数字需注明实际硬件/配额，不能与四核亲和运行无条件混比。

终端窗口单独标记。暂不把 terminal discard、动态 floor、初始 count-first、compressed starts 或 async 同时放进全排列矩阵；一次回答一个可删除工作的假设。

## 十二 必须收集的指标与停止条件

### 操作和时间

- selected 历史坐标解码 D、accepted starts A、stale 比例；邻居读取、selected-right 预测和 weight 查找次数。
- 实际 Positions append、已用编码 bytes、allocated capacity、非空 fragments、完整拒绝/准入 keys、heap pushes。
- publisher 的 H/K_live、loss L、coalescing E、owner hashes/actions；exact reducer 的 H_r/K_r、总 CPU 和最大 producer 尾部。
- 每窗口 cohort/rule 数与 stop reason；跨 cohort 两版 cut 数、整拒绝 producer 的 A/bytes、guard 与最终真实 child priority 的差距。
- directory 实际 state/thread 数、sentinel 填充、增长/复制与长期 capacity；不能复用含义不同的 Positions directory 计数冒充它。
- 分开计 prepare decode/match/loss、family reducer、apply、old-loss route/owner、final emit、birth route/group/concat/heap；记录真实 join 数。
- train wall、process CPU、关键 phase wall、最大 reducer/drain tail、分配/重分配次数、drop 时间。只移到别的 phase 的工作不算删除。

### 同时存活内存

对 starts、job vectors、selected candidates、Change capacity、AA scratch、groups、directories、staging、最终 leaves/roots 做时间对齐的容量账本。另报进程 RSS/HWM；加载输入留下的 HWM 与 allocator 缓存都不能自动归因给训练峰值。capacity、已用 bytes 和实际 RSS 分开列。

### 接受与回退

接受某接缝需同时满足：完整模型/必要错误语义一致；counter 显示预期工作确实消失；目标工作负载端到端收益可重复并超出噪声；零收益形状与单大 producer 无不可接受回退；峰值内存符合实施前明确的预算。

若只看到 outer windows 变少、prepare 降低而 final drain 增长，或组合收益其实来自 scratch/publisher，则不能采纳该窗口解释。若 FRONTIER 至多四 cohort 的机会不足以付费，就删掉它；不靠逐步叠异步机制来挽救一个未通过控制的内核。

运行时 guard/cap/memory stop 只做正常 drain。参考路径切换限于 attempt 入口或完整 drained frontier，不能在 hidden births 存在时直接退回 eager。错误仍走现有 attempt 丢弃与 reuse restart；不增加自适应策略引擎或未证明的局部回滚。

最终保留最少的胜出接缝和运行路径。若三项都未过门槛，结论应是保留当前实现，并留下准确的操作计数与被否证原因，而不是提交“结构更先进”的框架。

## 主要源码与公开证据

- [选择、准备与 apply 的现有边界](https://github.com/Yikai-Liao/tokenizers/blob/890ea1f58e65f8d8503f7cd3fc97ad1651eaf284/tokenizers/tk-train/src/trainers/bpe/merge.rs#L16-L123)
- [Change、direct directory 与 finish 保留条件](https://github.com/Yikai-Liao/tokenizers/blob/890ea1f58e65f8d8503f7cd3fc97ad1651eaf284/tokenizers/tk-train/src/trainers/bpe/merge.rs#L147-L260)
- [任务粒度及左右邻居处理](https://github.com/Yikai-Liao/tokenizers/blob/890ea1f58e65f8d8503f7cd3fc97ad1651eaf284/tokenizers/tk-train/src/trainers/bpe/merge.rs#L360-L482)
- [owner 路由、coalescing 与完整键归约](https://github.com/Yikai-Liao/tokenizers/blob/890ea1f58e65f8d8503f7cd3fc97ad1651eaf284/tokenizers/tk-train/src/trainers/bpe/index.rs#L194-L430)
- [只接受 Leaf 且不重编码的 concat](https://github.com/Yikai-Liao/tokenizers/blob/890ea1f58e65f8d8503f7cd3fc97ad1651eaf284/tokenizers/tk-train/src/trainers/bpe/positions.rs#L46-L85)
- [attempt 生命周期与 reuse restart](https://github.com/Yikai-Liao/tokenizers/blob/890ea1f58e65f8d8503f7cd3fc97ad1651eaf284/tokenizers/tk-train/src/trainers/bpe/mod.rs#L335-L417)
- [作者右 token prefetch 实验说明及逐次数据入口](https://github.com/Yikai-Liao/tokenizers/blob/890ea1f58e65f8d8503f7cd3fc97ad1651eaf284/experiments/bpe-simplification/evidence/prefetch-right-20261012/README.md)

证据使用原则：源码给出可修改的真实接缝；数学证明给出有条件的安全边界；独立模型检验提供反例搜索和机制交叉核对；只有授权后的原生实现与受控实验，才能决定这些接缝是否值得留下。
