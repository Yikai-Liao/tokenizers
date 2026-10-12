# BPE 整夜研究：完整阶段日志

更新时间：2026-10-12 11:00（Asia/Shanghai，UTC+08:00）。
验收时间：2026-10-12 11:00（Asia/Shanghai）。
当前已核验分支 head：`890ea1f58e65f8d8503f7cd3fc97ad1651eaf284`（最近一次持续核验：10:59:35（工具完成时间约至秒））；早期结论保留当时的固定提交来源。
研究要求：持续进行实质探索至验收时间；不能提前停止，也不能提前定稿后等待定点发送。本文按实际活动逐阶段整理；后续更正保留在原阶段之后。

本日志记录实际完成或正在进行的活动、证据、结果及尚未解决的问题。所有时间均为北京时间；精确到分钟的时间段表示目前可核实的阶段边界，不表示已经取得逐秒运行记录。没有记录的时段不补造活动。模型验证、源码审阅、理论推导和实际性能测量分别标明。

## 阅读说明

- 主时间轴为北京时间（Asia/Shanghai，UTC+8）；每个编号阶段另列已有起止时间的 UTC 对照。UTC 10 月 11 日与北京时间 10 月 12 日是同一晚。未记录的秒数、进程起止和中间活动均不补造。
- “通过”只表示该段指定的证据范围成立。独立数学模型是新写的标准库检查，没有运行、导入或修改被审仓库；源码推导也不等于 Rust 并发、接口或性能验证。作者发布的测试与 benchmark 单独标明。各模型的配置、轨迹或状态可能重叠，不能将各条计数相加为独立语料规模。
- 本文保留已否定的方向、适用条件、开发断言修正、证据误读更正及后续取代关系。后来的正确结论不会倒写成最初就已知道。
- 研究文件名仅用作模型辨认；长校验值压为指纹。可公开核查的源码及论文保留固定链接。阶段编号按记录归档顺序排列，重叠研究请以实际起止为准。早期“进行中”是当时状态；后续验证可能另列阶段，不表示该方向一直运行至验收。

## 研究目标与固定约束

- 优先消除 prepare 和 commit 的实际工作，再考虑并行、流水线及调度重排。
- 保持精确 weighted counts、merge 顺序、rank、输出 ID、词表、tie-break、floor、长度限制、reuse、AA 和停止行为。
- 不扫描 Positions 重新获得计数；不采用首次重数后转 eager 的变体。
- child Positions 使用统一的快速 append/concat 路径；不因 producer 数量增加专门通道。
- 理论结果不冒充实现结果；数学模型 PASS 不冒充 Rust 并发安全证明或 benchmark 加速。

## 已核实的阶段记录

### 01｜源码基线核验

- UTC 对照：开始：10-11 17:52:48；结束：10-11 17:52:48。

- 开始：2026-10-12 01:52:48。
- 结束：2026-10-12 01:52:48（核验事件的记录时间）。
- 工作：通过 GitHub API 核对 `Yikai-Liao/tokenizers` 的 `simplify/bpe-flat-20261010` 分支。
- 结果：成功核验分支指向 `02620e0ad37d8b165b5923902a05184e483cbd88`。
- 证据：[固定提交](https://github.com/Yikai-Liao/tokenizers/tree/02620e0ad37d8b165b5923902a05184e483cbd88)。重点审阅入口为 [prepare 与边界记录](https://github.com/Yikai-Liao/tokenizers/blob/02620e0ad37d8b165b5923902a05184e483cbd88/tokenizers/tk-train/src/trainers/bpe/merge.rs#L229-L517)、[选择与计数](https://github.com/Yikai-Liao/tokenizers/blob/02620e0ad37d8b165b5923902a05184e483cbd88/tokenizers/tk-train/src/trainers/bpe/index.rs#L147-L235)、[routing 与 commit](https://github.com/Yikai-Liao/tokenizers/blob/02620e0ad37d8b165b5923902a05184e483cbd88/tokenizers/tk-train/src/trainers/bpe/index.rs#L354-L446)。
- 限制：远端提交不覆盖 child Positions 的未提交改动；实施前仍需核对实际工作区。

### 02｜阶段性研究稿交付

- UTC 对照：结束：10-11 17:57:42。

- 开始：未单独记录。
- 结束：2026-10-12 01:57:42。
- 工作：交付阶段性 Markdown 研究稿。
- 结果：已交付。后续 02:25:58 的 v1 汇总稿包含更完整的 B/R、阶段重排与验证结果。

### 03｜B/R 统计、阶段重排、SPA 与有界模型汇总

- UTC 对照：开始：10-11 18:14；结束：10-11 18:25；10-11 18:25:58。

- 开始：2026-10-12 02:14。
- 结束：2026-10-12 02:25；v1 研究报告于 02:25:58 交付。
- 工作：整理加权路径收缩模型、边界关系差分、admission-sensitive B/R 分组、组末 loss 投影、先 apply 后读取最终边界的阶段重排，以及 fresh producer-run SPA 的适用边界。
- 结果：形成《BPE 精确计数的工作消除与并行研究建议》。优先候选为阶段重排、B/R 与组末 loss 合并、fresh producer-run SPA；跨规则滚动发布继续作为后续研究。
- 核实结论：
  - B/R 分组中，birth count 为 B，旧边界 removal mass 为 B+R。长度 admission 与 frequency floor 必须分开；floor 只能在完整归约后应用。
  - 固定 fresh batch 的最终边界可通过当前批次 output→inputs 恢复旧 witness；两端 fresh 的边界须唯一归属。
  - producer 输入流可依赖的单调量是规则 rank，不能误用 bucket 或 pair 单调性。
  - reuse 的历史 cohort 顺序仍须保留，不能将 fresh 的唯一 producer 推导直接扩展到 reuse。
- 明确未完成：实际布局成本、内部错误契约、Rust aliasing、真实并发 interleaving、atomics/join、分配与峰值内存、仓库测试及性能 benchmark。
- 验证证据：见下列两个独立模型的已保存结果；本阶段没有修改仓库代码，也没有运行仓库测试或 benchmark。

#### 03-A｜fresh batch 阶段重排与 B/R 模型

- 单次运行起止时间：原结果仅记录耗时，未记录墙钟起止；完成结果已在 02:25:58 交付前纳入报告。
- 运行耗时：36.874 秒。
- 结果：PASS，unexpected failures 为 0。
- 范围：160,451 个完整位置路径案例；151,675 个唯一候选记录顺序；1,966,062 个 weighted/cap counter cases；1,427,448 个外边界事件实例；1,600 个双 word 语料组合；10,984 次 fresh 历史转换；94,928 次归档候选校验。
- 观察：812 个 mixed B/R group 实例；16,206 个案例严格减少 group key 集合。
- 负面对照：五项有意错误覆盖旧 witness 恢复错误、共享切口双计、真实 ID 0 与 sentinel 混淆、未清除被消费 token start，以及重复 live record 导致即时 apply 与 validate-all-before-write 的 multiplicity 不一致。
- 适用限制：等价结果限于唯一 fresh starts；正权重枚举；固定 token span；当前批次 fresh 输出身份可准确识别。独立顺序 duplicate 拒绝 fixture 不构成生产等价证据。
- 可复验材料：`verify_fused_boundaries.py`、`results.json`、阶段研究材料。
- 源文件 SHA-256：`9e8bbe3b4769…`。

#### 03-B｜两遍原地 loss 投影模型

- 单次运行起止时间：原结果仅记录耗时，未记录墙钟起止；完成结果已在 02:25:58 交付前纳入报告。
- 运行耗时：4.676 秒。
- 结果：PASS，unexpected failures 为 0。
- 范围：423,864 个有效状态案例；1,808,320 个 group 实例；154,560 次 self-target、526,944 次 earlier-target、465,216 次 future-target 更新；116,256 次目录项校验后回退。
- 定向验证：四项额外有效 fixture，包括零权重但 Positions 非空的片段。该片段与正权重片段合并达到 floor 后，三个位置全部保留。
- 负面对照：present-key 目录被破坏、只验索引范围而不验 old witness、过早重置 future-target。
- 适用限制：当前存在的 old-key group 必须具有正确目录映射；全部初始化先完成再投影；B 与 Positions 不变。结果仅验证可选原地代数布局，不验证默认 Loss 输出数组的 Rust 实现，也不消除跨 producer、side、chunk 归约。
- 可复验材料：`verify_loss_projection.py`、`projection_results.json`、阶段研究材料。
- 源文件 SHA-256：`e5317ff9e7a1…`。

### 04｜确认整夜研究范围

- UTC 对照：开始：10-11 18:29:25；结束：10-11 18:29:51。

- 开始：2026-10-12 02:29:25。
- 结束：2026-10-12 02:29:51。
- 工作：确定继续研究 prepare/commit 工作消除、精确计数与 weighted IDs/ties、不重数 Positions、统一快速 child Positions 的方向。
- 结果：研究继续至 11:00 验收；要求保留各阶段的时间、活动、成败和验证证据。

## 02:30 起各研究方向的初始记录

以下保留各方向首次记录时的结论；后续完成、修正和验证在对应时间段及编号阶段中续记。

### 05｜端点分解与旧 loss 的 canonical 归约

- UTC 对照：开始：10-11 18:30；最新记录：10-11 18:32。

- 开始：2026-10-12 02:30。
- 结束：进行中。
- 最新记录：2026-10-12 02:32。
- 已做工作：推导受影响 old pair 的 tail-row 优先、否则 head-column 的唯一归属及按 key 汇总方法。
- 当前结论：归属分解可避免重复归属。若原有 M 条 loss 记录归约为 D 个不同 key，owner hash/probe 次数有机会从 M 降至 D。
- 已排除的过度结论：算术操作并未因此自动减少。M−D 次求和加 D 次减法仍为 M 次；radix/tile 分组、搬运、缓冲和热点倾斜必须计入成本。
- 成败状态：代数归属已推导；实现收益与端到端速度尚未验证。
- 新发现的反例：初始 count=10、floor=5、顺序 losses=[7,7] 时，旧行为在第一次减少后退休该项，随后忽略第二条；直接把 loss 聚合为 14 会发生下溢。该例属于无效或故障注入状态，说明不能未经错误契约审查就把任意顺序更新替换为简单总和。

### 06｜精确自适应 greedy 与 scalar debt

- UTC 对照：开始：10-11 18:30；结束：10-11 18:38。

- 开始：2026-10-12 02:30。
- 结束：本轮于 02:38 完成，最终结果见阶段 12；后续紧性研究见阶段 15。
- 最新记录：2026-10-12 02:32。
- 已做工作：探索延后 scalar 更新时如何维持候选选择、tie-break 与权重语义。
- 当前结论：token mass 可用于候选上界的方向值得继续验证；上界不能冒充精确计数。
- 成败状态：探索中；尚无完整认证条件或可替代现有计数维护的结论。

### 07｜稀疏更新与 immutable fresh birth rows

- UTC 对照：开始：10-11 18:30:34；结束：10-11 18:39:51；10-11 18:43:49。

- 开始：2026-10-12 02:30:34。
- 结束：理论报告于 02:39:51 完成；物理布局补充于 02:43:49 完成，见阶段 16—17。
- 最新记录：2026-10-12 02:32。
- 已做工作：研究 fresh birth row 单次创建及只读布局；分析半满阈值 compaction 的摊还成本。
- 当前结论：fresh row 一次创建的结构性质有初步依据；半满 compaction 具有可继续建立摊还界的方向。
- 风险：loss 的 canonical anchor 与 birth row 布局未必一致，重排代价可能抵消 batched search 的收益。
- 成败状态：结构推导进行中；布局、内存及性能未验证。

### 08｜空间边界 provenance 与 zipper

- UTC 对照：开始：10-11 18:30；结束：10-11 18:34。

- 开始：2026-10-12 02:30。
- 结束：2026-10-12 02:34（本轮分析报告完成）。
- 最新记录：2026-10-12 02:34。
- 已做工作：研究按空间边界 provenance 重建相关邻接并减少重复定位。
- 当前阻碍：当前设想的 spatial zipper 依赖 word directory，而该目录在现有流程中已退休。恢复或延长目录生命周期的成本尚未闭合。
- 本轮结论：spatial zipper 有机会省去 accepted-start 存储，但需要支付 k-way merge、边界 halos 及所需目录的成本。
- 成败状态：本轮分析完成；未得出无条件收益，不能把该方向记录为已取得性能加速。

### 09｜补充动态图库文献检索

- UTC 对照：开始：10-11 18:32；最新记录：10-11 18:32。

- 开始：2026-10-12 02:32。
- 结束：进行中。
- 最新记录：2026-10-12 02:32。
- 已做工作：检索到 graph batch/dynamic 相关条目 `1903.08794`、`1908.01956`，以及 GraphTango，DOI `10.1007/s10766-024-00768-x`。
- 状态：目前仅完成发现，尚未阅读核验；未将这些文献的结论用于 BPE 正确性或性能论证。

## 02:33—02:34 研究进展

### 10｜全局 token mass 上界与永久失活候选

- UTC 对照：开始：10-11 18:33；最新记录：10-11 18:34。

- 开始：2026-10-12 02:33。
- 结束：进行中。
- 最新记录：2026-10-12 02:34。
- 新假设：维护 token 的全局加权出现质量 M；pair frequency 的上界为两端质量的较小值。对 selected non-AA 规则，尝试用选中频率更新 M，目标更新量为 O(batch)。
- 候选用途：若在 fresh 阶段能证明某 token 的全局质量已经低于 floor 且不再增加，相关 pair 可以永久失去入选资格，可能省去相应 loss 统计与 birth Positions；不通过扫描 Positions 重数。
- 待核验：全局质量与已维护 pair count 的语义是否一致、u128 或其他防溢出表示、初始质量统计成本、reuse 边界、零权重位置，以及永久失活的完整证明。
- 成败状态：新假设；尚未完成验证，不作为可以直接实施的结论。

### 06 补充｜pair provenance forest 与发布认证

- 记录时间：2026-10-12 02:33—02:34。
- 新假设：fresh pair key 的唯一出生 epoch 与唯一 old witness，可能构成适合推迟 scalar debt 的 provenance forest。
- 潜在收益与代价：lazy exact scalar debt 或许能省去永远不会再查询的更新，但 inverse index、保留的记录存根和积累债务的空间成本尚未闭合。
- 新假设：对尚未完成 commit 的变化维护上界，可能在不读取全局一致快照时认证候选选择。
- 待核验：physical reader stability、完整 tie/lexicographic 上界、遗漏 birth 的风险，以及认证失败时的进展条件。
- 成败状态：仅提出候选结构与验证问题，未建立完整正确性或收益结论。

### 07 补充｜immutable rows 与 owner 布局

- 记录时间：2026-10-12 02:33—02:34。
- 新假设：对 fresh immutable birth rows 使用按 row 排序的 batch 更新，并交付互不重叠的 mutable slices，可能避免部分 hash-owner 更新。
- 未决问题：row 方向与 canonical loss anchor 不匹配；排序、搬运和 compaction 可能抵消收益。此方向仍与保留现有 owner 的主线分开评估。
- 成败状态：结构探索，尚未验证。

## 02:35—02:36 研究进展

### 10 补充｜免初始化的保守 upper mass

- 记录时间：2026-10-12 02:35—02:36。
- 改进候选：对训练开始时已存在的 token，使用不会减少的无穷上界；对 fresh output，令其上界 U 等于 selected frequency f。这样尝试避免额外扫描语料初始化全局 token mass。
- non-AA 更新候选：两个不同旧输入的 U 若有限，各减少 f；无穷上界保持无穷。
- AA 更新候选：设实际 non-overlap replacement mass 为 m，使用 `2m ≥ f` 的关系，旧输入 U 只减少一次 f，输出上界取 f；不重数 Positions。该保守性及全部边界条件仍待验证。
- 源码约束：有效 floor 为 `min_frequency.max(1)`，不能按用户原始 min frequency 直接处理零值情形。
- 粗粒度过滤候选：在 producer 层预先计算 side mask。若相关旧输入上界低于 floor，尝试整侧省略 loss，而非每个 occurrence 再加载四个端点上界。
- 成败状态：候选推导正在复核；还不能宣称保持全部错误行为、所有计数语义或实现加速。

### 11｜按激活顺序封存 birth rows

- UTC 对照：开始：10-11 18:35；最新记录：10-11 18:36。

- 开始：2026-10-12 02:35。
- 结束：进行中。
- 最新记录：2026-10-12 02:36。
- 新候选：post-apply 后，两端都是新输出的共享边界由激活 rank 较新的端点归属，以支持 sealed birth rows。
- 潜在目标：减少或删除部分 hash shards、owner routing 和 BTree 维护。
- 待核验：归属唯一性、Positions 顺序、reuse 与历史 cohort 的边界、热点倾斜、compaction 成本，以及这种布局与当前分区要求是否兼容。
- 成败状态：更大范围的架构候选；尚无实现、仓库测试或性能证据。

### 07 补充｜退休 token 的稀疏清扫势能

- 记录时间：2026-10-12 02:35—02:36。
- 新候选：令 R 为退休 token 关联度的累计量，D 为实际 dead keys，提出 `R/2 ≤ D ≤ R`。若活跃存储量为 E，在 `R ≥ E/2` 时清扫，则至少有 E/4 个 dead keys，可据此尝试建立按 erased key 计的常数摊还成本。
- 待核验：R、D、E 的精确定义、重复退休/删除的计数、压缩后重置规则与适用图结构。
- 成本限制：维护 degree 与退休状态需要额外更新，不能把该费用省略。
- 成败状态：摊还证明待复核；未进行仓库执行或性能测试。

## 02:37—02:38 研究进展

### 12｜精确 greedy 的 commit 消除分析完成

- UTC 对照：开始：10-11 18:30；结束：10-11 18:38。

- 开始：2026-10-12 02:30（对应阶段 06）。
- 结束：2026-10-12 02:38。
- 工作：完成 fresh birth-provenance forest、lazy scalar debt、inverse-index/range-sum 限制、独立 loss 向量及 token-bound 剪枝的理论分析。
- 已形成结论：fresh provenance 的直接子节点初始质量可表达旧 pair 的 loss；不能改用子节点当前 count，也不能把所有后代质量重复扣除。lazy debt 可省掉从未被再次查询的旧 pair 更新，但索引、路由、标量存根及内存回收仍有成本。
- 排除结果：eager child-sum 只是把旧 count 更新换成另一种表示；range sums 或 Euler tours 不能免费构造 old-witness 的逆索引。
- 有限信息论结论：构造 weighted words 可让一个 fresh merge 产生任意独立 D 维 loss 向量；仅靠 token marginals 一般不能恢复所有精确 pair counts。该结论不是所有计算模型下的普适下界。
- 错误契约：跳过未查询 debt 可能漏掉旧路径会报告的错误；将全部 loss 求和也可能产生旧路径退休后会忽略的错误。未解决前不视为 drop-in 方案。
- 来源：固定提交源码；[Factorised Representations of Query Results](https://www.cs.ox.ac.uk/dan.olteanu/papers/oz-icdt12.pdf)、[LaB-PQ 所在原始论文](https://arxiv.org/html/2105.06145v2)、[Kinetic Data Structures](https://graphics.stanford.edu/~comba/papers/socg.pdf)、[Space-Efficient Re-Pair Compression](https://www2.compute.dtu.dk/~phbi/files/publications/2017serpcC.pdf)。文献只支持相应表示或数据结构背景，BPE 推导是本次独立分析。
- 成败状态：本轮理论报告完成；推荐优先继续验证 token-mass 剪枝，通用 lazy scalar forest 暂不列为首选实现。

### 13｜局部发布与 guarded certification 分析完成

- UTC 对照：结束：10-11 18:37；10-11 18:38。

- 开始：未单独记录。
- 结束：2026-10-12 02:37—02:38。
- 工作：分析 incomplete commit 的候选覆盖、publication guards、短 admission 校验、共享边界组件，以及局部 reader/apply phases。
- 当前结论：全局一致快照是精确选择的充分条件；若所有未完成变化都有覆盖其完整候选优先级的保守 guard，则理论上可在不等待所有 owner 完成时认证更高优先级候选。
- 必需条件：候选的 authoritative count 校验与 admission 必须原子化或版本重验。只检查 guard 而复用更早读到的 count 不安全。
- 已找到反例：两个 fragments 各为 60，完整 birth 为 120；对 fragment 只取 max=60 不能覆盖完整候选。guard 不能只覆盖 birth，还须覆盖半完成的旧 count 及未齐全 Positions。
- 仍未解决：物理 Corpus reader stability。原子 slots 不会自动把多个端点写变成原子路径重写。whole-word stripes 提供保守路线，但引入 false dependencies、目录/元数据成本及大 word 瓶颈。
- 来源：[Greedy Sequential MIS and Matching](https://jshun.csail.mit.edu/mis.pdf)、[Calvin](https://www.cs.umd.edu/~abadi/papers/calvin-sigmod12.pdf)、[Aria](https://www.vldb.org/pvldb/vol13/p2047-lu.pdf)、[Naiad](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/11/naiad_sosp2013.pdf)。未将随机优先级深度界或数据库协议当成 BPE 性能证明。
- 成败状态：协议层分析完成；没有完整实现、内存序审计或性能验证，不能宣称彻底消除 barrier。

### 14｜token-mass certificate 的独立全流程模型

- UTC 对照：研究开始：10-11 18:38:08；最终执行开始：10-11 18:42:27.780993。

- 研究开始：2026-10-12 02:38:08。
- 复核经过：初轮结果摘要后，修正最大值 fixture 并重新执行；以下只报告最终保存结果，不混用旧摘要统计。
- 最终执行开始：2026-10-12 02:42:27.780993。
- 最终执行结束：2026-10-12 02:42:30.212338；执行耗时 2.432 秒。
- 报告完成与结果交付：2026-10-12 02:44:03 前。
- 工作：使用 Python 标准库建立完整 serial greedy fresh-BPE reference，与维护增量 index 且实际省略 globally certified losses/birth Positions 的算法比较。参考 oracle 的直接路径重数仅用于独立验证，不是生产优化方案。
- 结果：PASS；完整 merge history、selected frequencies、output IDs 和最终 weighted-corpus token paths 一致，没有 unexpected failures。
- 范围：75,417 个配置（72,900 穷举、2,500 固定种子随机、16 个定向配置，以及一个负面对照所配的正确 reference 比较）；35,239 个非空 history；16,366 个 multi-epoch history；最长 23 条规则。
- 检查量：61,267 条 selected rules，其中 23,566 条 AA；241,929 次 token-bound 检查；29,325 次 retirement-safety 检查；17,636 个 lower-numeric fresh-output 非空 history；5,701 次零权重 word 内的实际 selected occurrence。
- 实际省略：13,284 个 old loss 事件、43 个 newborn position records。它们是模型中的逻辑事件数，不是 CPU 操作、分配减少量或速度测量。
- 四项负面对照：数值饱和后继续减法造成下界；AA 错减 2f；按局部 fragment 应用 floor；用更新后的 token 上界取消已经 reserved 的规则。
- 正确性范围：sticky infinity；有效 floor 为 max(minimum,1)；fresh 输出从未激活且不复用；固定 token span；全部被替换 occurrence 都计入 selected raw frequency；所有 authoritative candidate 查询遵守退休判定。
- 仍未验证：生产 compatible-batch selector、并行发布/共享输入更新、reuse 转换、Rust 整数与错误时序、惰性回收和峰值内存。
- 可复验材料：`verify_token_mass_certificate.py`、`token_mass_results.json`、阶段研究材料；源 SHA-256 为 `6bee6d7bf2d5…`。
- 边界：未导入、运行或修改仓库实现；上述耗时不是 trainer benchmark。

### 15｜token-bound 紧性与更强低成本证书

- UTC 对照：开始：10-11 18:38；结束：10-11 18:53。

- 开始：2026-10-12 02:38。
- 结束：2026-10-12 02:53。
- 工作：研究只观察 old pair bound 和两个 token marginals 时，`min(old_pair_bound, U_left, U_right)` 是否已达到可保证的最紧上界；尝试构造 observable marginals 相同、pair joint counts 不同的真实语料。
- 最终理论报告已完成。不同端点的 min-bound 紧性由可实现 weighted 2×2 family 与 inclusion–exclusion 推导支持；AA 的整数例外为 `max(M[a]−1,0)`。更强证书需要额外上下文或剔除已证无关的上下文。
- 实验优先顺序调整为：当前统一 child-Positions 基线上的 membership gating；token envelopes 其次；保留历史 scalar provenance 的重型方案后置。该排序依据结构与成本分析，不是性能测量排名。

## 02:39—02:44 研究进展

### 16｜稀疏更新与 sealed rows 理论报告完成

- UTC 对照：开始：10-11 18:30:34；结束：10-11 18:39:51。

- 开始：2026-10-12 02:30:34。
- 结束：2026-10-12 02:39:51。
- 工作：审阅固定提交的 index 实现，分析 collect-reduce、endpoint anchoring、sealed birth rows、floor 退休、65-bit 分组 key、垃圾回收势能及 stale Positions 生命周期。
- 结论：消除重复更新与把字典更新并行化是两件事。对 M 条本地已归约 loss、D 个 key，进一步 coalesce 可省 M−D 次 authoritative 查找/写；frozen atomics 本身不省这些查找。
- 新来源：[SPAA 2023 stable semisort / collect-reduce](https://arxiv.org/pdf/2304.10078)。heavy-key reduction、稳定分发与缓存行为更贴近当前问题，但仍须支付分区、scratch 和搬运成本。
- exactness 结果：若保留原有每 key 的更新顺序，可用有界 prefix-sum 结构找到第一次跨 floor 的位置，区分正常退休与 underflow；这是正确性候选，不能宣称删除全部算术或保留跨 owner 的错误先后。
- 内存限制：count-key 存储的上界不能自动约束 heap 中 stale Positions 的总存活量。
- 成败状态：本轮理论报告完成；没有编译实现、仓库执行或 benchmark。

### 17｜sealed rows 物理布局补充分析完成

- UTC 对照：开始：10-11 18:40:35；结束：10-11 18:43:49。

- 开始：2026-10-12 02:40:35。
- 结束：2026-10-12 02:43:49。
- 工作：比较每行一个 Vec 的 AoS 布局与单一 flat Vec；分析 sparse split_at_mut/Rayon 行更新、非空行列表、串行原地与并行非原地 compaction，以及按实际 capacity 计峰值。
- 结果：行描述符宽度、Vec 扩容或整理、compaction 复制、临时新旧存储同时存在，都可能抵消移除 hash-owner 层的收益。不能把删除 owner routing 记成免费收益。
- 避免的新工作：用非空行列表避免每次扫完整 V；只对实际触及的互斥 slices 更新，避免隐藏的全域遍历。
- 成败状态：物理布局方案及代价已整理；没有安全 Rust 实现、编译结果或性能数据。

### 18｜历史证据审计完成

- UTC 对照：开始：10-11 18:39:48；结束：10-11 18:42:31。

- 开始：2026-10-12 02:39:48。
- 结束：2026-10-12 02:42:31。
- 工作：检查固定提交中 evidence 报告及其 provenance、manifest、summary；区分历史诊断、合成 microbench 和当前基线所需证据。
- 结果：已检查的材料未建立当前 child-Positions 基线下的 M/D、per-owner 工作倾斜、token 退休覆盖率、sealed-row degree/utilization，或 guarded pipeline 的合法 overlap 与尾部等待。
- 关键判断：文件名含“current”不代表当前源码；旧版阶段耗时和 Positions 分布不能代替本次候选的性能证据。局部短 fragment 也不代表全局 pair 低于 floor。
- 来源示例：[index-shards 历史结果与构建来源](https://github.com/Yikai-Liao/tokenizers/blob/02620e0ad37d8b165b5923902a05184e483cbd88/experiments/bpe-simplification/evidence/index-shards-20261011/README.md)、[阶段 provenance](https://github.com/Yikai-Liao/tokenizers/blob/02620e0ad37d8b165b5923902a05184e483cbd88/experiments/bpe-simplification/evidence/phase-provenance.json)、[Position probe](https://github.com/Yikai-Liao/tokenizers/blob/02620e0ad37d8b165b5923902a05184e483cbd88/experiments/bpe-simplification/evidence/box-only-20261010/position-probe.json)。
- 成败状态：审计完成并确认当前证据缺口；未复现历史 benchmark，也未运行新诊断。

### 19｜sealed rows 独立反例审查

- UTC 对照：开始：10-11 18:44:24；结束：10-11 18:57:18。

- 开始：2026-10-12 02:44:24。
- 结束：2026-10-12 02:57:18。
- 工作范围：独立检查 sealed birth-token row 的证明与反例，重点为 phase、内存及 ownership 的失败情形。
- 02:47:34 源码审阅子阶段完成：核对 row-design、固定提交的 merge.rs/index.rs/Cargo.toml。fresh AA 使用 nonoverlap starts 执行，final pair counts 仍含重叠邻接；基线共享切口由 physical-left 归属；fresh heap comparator 仅依据 count/pair。
- Rayon 1.10.0 官方实现观察：精确长度的 Vec::par_extend 可以并行写入新增 tail；filter/flat_map 会走 list-of-Vec 加串行 append fallback；collect_into_vec 会截断旧内容。布局方案不能把这些 API 当成相同的增量并行追加。
- 最终结论：fresh no-rebirth 定理通过 AA 与低数值 reserved ID 的反例审查；共有 cut 的 loss 与 birth ownership 必须一致移动或明确拆开，不能只改 birth 而遗漏/双计旧 loss。
- 新增成本：固定提交没有 activation-order 表，只有已激活标志，故 semantic rank 是额外 O(V) 元数据。
- 反例：每 pair count 合法不保证整行总和适合 u64；巨大稀疏行的单点 lookup 有 logarithmic work；紧 memory cap 下反复为单个 birth 打包 K 个存量 cell，可产生 Θ(K²) movement。
- 简化：没有额外 token-mass 懒退休的普通 rows，可用 scalar tombstone count 触发清扫，避免 degree array。当前 reuse 路径重启整个 attempt，因此不需要 live-state row→signed-ledger 转换。
- Rayon 来源：[1.10.0 Vec extension](https://github.com/rayon-rs/rayon/blob/v1.10.0/src/iter/extend.rs#L530-L556)、[collect/append](https://github.com/rayon-rs/rayon/blob/v1.10.0/src/iter/collect/mod.rs#L8-L37)。manifest 声明 rayon=1.10，未审计实际 lockfile 解析版本。
- 成败状态：反例审查及独立模型完成，布局仍没有运行时加速证据。
- 边界：只读理论检查，不运行或编辑仓库代码。

## 02:45 研究进展

### 15 补充｜marginal 上界的可实现反例与 AA 例外

- 记录时间：2026-10-12 02:45。
- 新构造：weighted 2×2 context family 的完整 old-pair histogram、全部 token masses、selected frequencies 及 post-merge raw directional masses 相同，但 UV residual 可在 0 至 4 之间变化。
- 当前结论：对不同端点，仅凭这些 marginals 一般不能将 min-bound 收紧为精确 joint count；已构造可实现语料支持该限制。
- AA 特例：有限、正的整数质量下，同 token 邻接具有 run 边界损失，可得到 `f(aa) ≤ M[a]−1`；零质量等边界仍须明确处理，不能照搬不同端点的紧性结论。
- 继续研究：从完整的、达到 floor 的 birth groups 维护 active outgoing/incoming candidate-mass envelopes，避免增加逐 occurrence 端点统计；另研究 frozen old-parent membership 的剪枝证书，暂限定 non-AA compatible batch。
- 成败状态：反例与特殊界已推导；后续更强证书与完整报告仍在进行。

### 20｜frozen old-witness membership 独立模型

- UTC 对照：研究开始：10-11 18:45:15；最终执行开始：10-11 18:50:58.504379。

- 研究开始：2026-10-12 02:45:15。
- 最终执行开始：2026-10-12 02:50:58.504379。
- 最终执行结束：2026-10-12 02:51:00.623467；耗时 2.119 秒。
- 报告完成：2026-10-12 02:52:18 前；结果于 02:52:28 报告。
- 工作：独立验证 non-AA compatible fresh batch 的 frozen old-witness active-membership 证书，覆盖共享切口、全局 witness 唯一性、零权重、较低数值 fresh ID、多 word 与长度 gate。
- 结论：每个 final child key 全局具有唯一 old witness；在完整、冻结的 active index 中，非 selected 的 old witness 缺席可证其 below-floor 或永久长度不合法，从而省略其全部 loss/birth。membership snapshot 必须是在 selected keys take 后、当前批次 commit 前的完整状态。
- 结果：PASS，64,129 个完整 batch；81,390 次全局 child-witness 检查；83,052 个外切口，其中 3,108 个 shared new/new cuts；最终 active frequencies 与完整 Positions 精确一致。
- 实际省略：6,404 个 loss 事件与对应的 6,404 个 birth candidates，其中 4,700 个本来通过长度 gate 的潜在 birth Positions。另保留了 748 个必要的零权重 birth Positions。
- 负面对照：selected AA、incomplete frontier、shared-output alias 三项均展示错误变体失败。特别是 selected-key 缺席不能在 AA 下直接等同于低频。
- 限制：这是 whole-batch 状态变换模型，不验证生产 multi-batch selector、并发查询、Rust 安全、overflow/故障时序或 reuse。数学上允许较低数值 fresh ID，不授权取消实际 selector 的 reserved-ID 隔离规则。
- 成本：仍须识别 witness 并查询稳定 active membership；hash/cache/synchronization 的新增费用需要和省下的 grouping、累加、Positions、routing 比较，没有性能结论。
- 可复验材料：`verify_frozen_witness_certificate.py`、`frozen_witness_results.json`、阶段研究材料；源 SHA-256 为 `05cd838c8173…`。
- 边界：先前模型及结果保持不变；没有运行或修改仓库实现。

### 21｜fresh ID 顺序源码审计

- UTC 对照：开始：10-11 18:45:00；结束：10-11 18:46:43。

- 开始：2026-10-12 02:45:00。
- 结束：2026-10-12 02:46:43。
- 工作：核验真实实现如何区分 reserved ID、fresh 输出及 reuse。
- 初步源码结果：`merge.rs` 74—90 将所有 reserved output 隔离为 singleton，包括 inactive low ID。非空 batch 在 take 前停止；如果 reserved 是首项，则 take 后立即封 batch。已经 active 的 reserved ID 会重启为 reuse，cap 为 1。
- 影响：多规则 fresh batch 的 ID 排序前提已有源码依据；滚动调度仍须保留 reserved drain/barrier，不能把 singleton 的处理与普通 batch 混用。
- 来源：[固定提交的 batch admission](https://github.com/Yikai-Liao/tokenizers/blob/02620e0ad37d8b165b5923902a05184e483cbd88/tokenizers/tk-train/src/trainers/bpe/merge.rs#L74-L90)。
- 手工反例：special token `pq=0`，weighted words 为 `pqz:100, pq:100, ab:100`。PQ200 激活 ID0 后生成 `(0,z)100`，在 equal-frequency tie 中排在 AB100 之前，而旧 witness QZ100 排在 AB 之后。删除 reserved isolation 会允许错误的第二条规则。该 fixture 是手工推导，未执行。
- 结论：当前源码 gate 正确隔离了此情形；没有发现当前 batching bug。滚动设计只设 batch-size=1 仍不够，必须完成相关 publication 或提供更强的 tie-aware guard。
- 成败状态：源码审计完成；未运行仓库测试。

### 22｜更强的阈值剪枝与增量维护文献研究

- UTC 对照：开始：10-11 18:46；结束：10-11 18:53。

- 开始：2026-10-12 02:46。
- 结束：2026-10-12 02:53。
- 工作：阅读前述 fresh-BPE 计数和退休报告，并开始检索 anti-monotone boundary-witness pruning、增量维护/provenance、graph contraction 的原始来源。
- 目标：寻找能够省略更多 prepare/commit 工作且边界明确的证书。
- 完成结果：整理八项原始框架/论文来源，形成 weighted unique-witness 推导、early membership filter、AA/post-take 例外、snapshot false-negative 风险、两种证书互不包含的 fixtures，以及 capped-loss / partition-budget 推论。
- 代表来源：[Apriori](https://rsrikant.com/papers/vldb94.pdf)、[Provenance Semirings](https://www.cs.ucdavis.edu/~green/papers/pods07.pdf)、[DBToaster](https://dbtoaster.github.io/papers/pvldb2012-dbtoaster.pdf)、[Parallel Tree Contraction](https://www.cs.cmu.edu/~glmiller/Publications/Papers/MR1-89.pdf)、[Distributed Thresholded Counts](https://doi.org/10.1145/1142473.1142507)。
- 成败状态：报告及来源核验完成；没有把近似监控、数据库或 tree-contraction 的 benchmark/调度保证移植成 BPE 结论。
- 边界：不重新扫描 Positions 求计数，不运行或修改仓库实现。

## 02:49 研究进展

### 22 补充｜hereditary witness pruning 与吸收式退休

- 记录时间：2026-10-12 02:49。
- 来源核验：Apriori（1994）、provenance semirings（2007）、DBToaster（2012）、Miller–Reif tree contraction，以及 thresholded-count monitoring（2006）的原始资料已查阅。
- 当前结论：在唯一 old-witness 的 support injection 前提下，可研究直接查询已有 frozen active membership 的 complete-key 剪枝；不必额外维护 provenance tree 或 negative-border 状态。
- 修正：不能声称 witness membership 剪枝全面优于 post-batch token-mass retirement。已构造两个方向的例子，说明两种证书适用范围互不包含。
- 次级候选：capped loss `min(c−F+1, total_loss)` 可表达保留 exact above-floor count 与 absorbing retirement；其潜在收益只是在退休阈值后省去部分 loss 累加，不能因此省掉 birth 或全部 routing。
- 成败状态：理论与来源已推进；完整报告及实现成本审查仍在进行，尚无生产性能结果。

## 02:50 研究进展

### 19-A｜sealed-row cut ownership 独立有限模型

- 所属研究开始：2026-10-12 02:44:24。
- 模型执行开始：未单独记录墙钟起点。
- 模型执行完成：2026-10-12 02:50:07 前；保存结果记录耗时 0.95 秒。
- 工作：检查 fresh old-key count 单调性、按 activation rank 的最终切口唯一归属及每种方向的 position mapping 顺序。
- 结果：10,400 次 sequential fresh transitions；109,152 个 disjoint-cut owner cases；5,796 个 self boundaries；3,554 次输出 ID 数值小于两个选中输入的转换。上述检查及明确的 floor-fragment/zero-weight 示例均通过。
- 可复验材料：`check_model.py` 与 `results.json`；源 SHA-256 为 `288659a54105…`。
- 限制：独立数学模型，不是仓库执行、完整 trainer parity 或并发/布局实现证明。整体反例审查最终于 02:57:18 完成，见阶段 19。

### 23｜physical-cut 永久剪枝后续研究

- UTC 对照：开始：10-11 18:49:57；10-11 18:50；结束：10-11 18:55。

- 开始：2026-10-12 02:49:57—02:50。
- 结束：2026-10-12 02:55。
- 工作：研究具有真实 below-floor lineage 的 physical cut，在 fresh AA 情形下是否仍可永久剪枝，以及是否能使用惰性不可逆 bit marker 而不扫描退休 Positions。
- 初步反例：selected-key 被移出 active index 不是死亡证明。`AAAA`、floor=1 时，仍存活的旧 AA 外切口可以生成 eligible XX。
- lazy bitmap 候选：初始为 unknown；首次 frozen witness 确认低于 floor 后，才标记不可再激活。目标是避免同一低频物理边界在后续收缩中重复查询和分组。
- 待计费用与前提：cut 地址必须稳定；额外 N bits、atomic 竞争、reuse 清空都要计算；如果已有每组缓存足够，逐 cut 标记可能反而增加工作。
- 完成结论：真正的 below-floor physical-cut lineage 在 fresh AA 下仍可永久剪枝，并可形成不可逆 segmentation；各段仍共享全局 counts 与 greedy 选择，不能各自独立训练后拼接。
- 两种不安全捷径：selected AA key 被移除，以及 post-cohort old witness death，都不是 transition 前低频证明；后者由 ABC→XC 反例说明。
- 量化成本：200 million slots 的单 bit 表需 25 MB（约 23.84 MiB），另有逐边界 loads 与首次 atomic mark；不改善现有总边界触及 O(N) 的渐近界。
- 成败状态：本轮理论报告完成；安全范围与成本已明确，没有建立比每组缓存更优的实际收益。
- 边界：只读理论，不执行仓库代码或新增退休 Positions 扫描。

## 02:53 研究进展

### 15 补充｜membership gating 的查询时刻与成本边界

- 记录时间：2026-10-12 02:53。
- 必须分开的三个判定：parent 在 selection 前是否 active；old pair 在 batch 后是否 ineligible；final child 在 batch 后是否 ineligible。第一个可省 loss 和 birth，后两者分别只决定 loss 或 birth。
- 反例：ABC:10、floor=2，执行 AB→X 后 B 的 mass 为 0，旧 BC 已无资格，但 XC:10 必须出生。不能用 post-batch parent death 代替 pre-selection membership；冻结了 map 也不能让 helper 混入新更新的退休 flags。
- AA 扩展候选：以 selection 前的逻辑 membership 为语义 oracle；当前 after-take map 缺席时，对 singleton AA 的 selected key 作明确 override。non-AA 不需要该 override。该统一路径仍需对应独立验证。
- 明确成本账：inactive parent 的查询大体从 commit 前移到 prepare；active parent 多一次 prepare read 且通常仍需 commit mutable lookup。first-touch negative cache 需要有效槽位/标记，不能假定目录工作全部消失。
- 零权重特例：基线可能本来就不路由 aggregate loss=0，因此新增 first-touch lookup 不一定只是移动旧查询。
- 组合限制：B/R 把 old/final groups 合并后，membership 的 old-group cache 可能消失；不能同时无条件计入 B/R 省 old-group lookup 和 membership 复用该 group 的收益。应先单独比较变体。
- 阶段限制：prepare 新增 index membership 读取，要求该 view 保持冻结；不能同时假定 owner commit 可无条件修改同一 view。

### 24｜active candidate row/column envelopes

- UTC 对照：开始：10-11 18:53；结束：10-11 18:53。

- 开始：2026-10-12 02:53 前；准确启动秒数未记录。
- 结束：本轮理论推导于 02:53 完成，独立验证尚未报告。
- 候选：在稳定 frontier，用全部 eligible above-floor pairs 的 outgoing/incoming counts 之和给单个候选上界。fresh 唯一 witness 保证相关 active row/column mass 不会因邻居变新而增加。
- 成本来源：从完整 birth groups 收集新输出的 S_out/S_in，不新增逐 occurrence 的 word-end 统计；对 new/new birth 分别贡献一次 outgoing 和 incoming。
- 额外费用：每个 retained complete birth 最多两个 scalar contributions、按输出 ID 归约及 O(batch) updates；两个 u64 数组为 16V bytes。与总 token bound 同时保留为 24V bytes，并新增更新。
- 限制：完整 birth 归约后才可取得 S，不能宣称它省掉生成同一 S 所需的 gather。distinct-input 更新不能直接套在 AA 上；保守 AA 扩展可保留输入侧 envelope 不变。
- 成败状态：语义推导完成；成本、实际覆盖率和独立模型仍待验证，尚无性能推荐。

### 25｜AA-aware preselection witness oracle 扩展验证

- UTC 对照：研究开始：10-11 18:52:37；最终执行开始：10-11 18:55:09.960816。

- 研究开始：2026-10-12 02:52:37。
- 最终执行开始：2026-10-12 02:55:09.960816。
- 最终执行结束：2026-10-12 02:55:10.330801；耗时 0.370 秒。
- 报告完成：2026-10-12 02:56:49；结果于 02:56:58 报告。
- 工作：独立检查 singleton AA 在 take 后恢复 selected-key membership 的语义 oracle；增加 post-cohort token-mass guard 错误隐藏原本 active witness 的组合负面对照。
- 结论：fresh singleton AA 的全局唯一 witness 与 support injection 仍成立。正确 oracle 为 raw frozen after-take membership，或 witness 等于本次 selected AA key；后者仅恢复语义资格，不重建已被 take 的旧 count entry。
- 结果：PASS；45,432 个有界配置中 7,278 个具有可选择 AA 并执行完整 batch，其余 38,154 个未执行 AA batch。13,924 次 global witness checks；15,900 个外切口；320 个必要的零权重 Positions 被保留。
- 对照：blind after-take oracle 在 2,430 个案例中与 reference 不一致；AA-aware oracle 在所有有效 batch 中一致。ABC×10 的 post-token-flag 组合负例也已验证。
- 可复验材料：`verify_aa_witness_supplement.py`、`aa_witness_results.json`、阶段研究材料；源 SHA-256 为 `f790921a24b5…`。
- 依赖：仅复用独立 frozen-witness 数学模型的 helpers，依赖 SHA-256 仍为 `05cd838c8173…`。
- 限制：这是对 earlier raw after-take 限制的明确修正，不是否定其反例。未验证 Rust、并发、allocation、故障时序、reuse 或性能；先前 certificate 模型与结果未修改。

### 26｜parent-budget publication certificate

- UTC 对照：开始：10-11 18:54:13；结束：10-11 18:59:58。

- 开始：2026-10-12 02:54:13。
- 结束：2026-10-12 02:59:58。
- 工作：研究以旧 parent 的质量预算约束尚未完整发布的 child candidates 的证书。
- 本轮结论：对固定 fresh-compatible target，令 C 为旧 witness 精确 pre-batch count、L 为已唯一归属的消失切口质量、b_c 为各 child 已归约质量，则 `U=C−L+max(b_c)` 是整个 family 的单调上界。
- 示例：C=100、L=60，两个 child 各积累 30，得到 U=70；若全部 60 集中到同一 child，则 U 仍为 100。进度本身不保证上界下降。
- 关键修正：U 约束最终逻辑 target，不自动约束尚未扣 loss 的物理 count 或 stale heap。优先保留旧 guard，直至相应 parent loss 和 child owner updates 可见并得到一致 acknowledgment，才发布更紧 guard。
- 工作代价：可复用完整 child totals 做 O(1) max/tie 更新，但按 parent 归属、路由、完整 fragment closure、零权重尾片段及物理 reader stability 仍须处理。仅证明 rejected mass 的 `C−J` 是较便宜的候选 fallback。
- 成败状态：理论报告完成，包含 AA/selected-parent、overflow、重复切口/fragment 反例与费用清单；没有完成 pipeline 或速度结论。
- 边界：不执行仓库实现或修改代码。

### 27｜first-touch witness gating 最小接口草案

- UTC 对照：开始：10-11 18:56；结束：10-11 18:59。

- 开始：2026-10-12 02:56。
- 结束：2026-10-12 02:59。
- 工作：对固定提交的 Neighbors 设计只读接口草案，假设当前统一 child Positions 路径；prepare 期间 PairIndex 不变，每个 old group 复用判定，born-only groups 避免无用 membership read，AA 采用 allow-on-miss。
- 约束：不新建 engine、trait 或锁层。
- 草案结果：prepare-only `Option<&PairIndex>` 加 local `Option<bool>` cache；born-only groups 不查询；raw frozen membership 加 selected-AA allow-on-miss；交付 owned Jobs 后结束 shared borrow，再开始 commit。
- 规模估计：约新增 35—70 行 production code，仅为未生成 patch 的估算，不是已实现行数。padding、Sync 和 API 适配须实际核验；不顺带加入 B/R 或 stable handles。
- 成败状态：接口草案作为阶段材料完成；未编辑、编译或执行仓库实现。

## 02:58 研究进展

### 28｜实验决策计划汇总

- UTC 对照：开始：10-11 18:54:41；结束：10-11 18:58:10。

- 开始：2026-10-12 02:54:41。
- 结束：2026-10-12 02:58:10。
- 工作：综合 token-bound 限制、threshold-pruning、历史证据审计与 sealed-row 分析，形成分阶段、彼此隔离的实验计划。
- 产出内容：membership、token envelope、scalar reducer、row-layout 四类独立变体；最小 join-aggregated counters；break-even 成本账；capacity/allocation/final-free 费用；P 大于 batch pair 数时的 work/span 情形；exactness 与错误路径 fixtures。
- 测量建议：初始采用两组配对、交错的 full-run 重复，仅在结果含糊时增加第三组。此项是计划，尚未执行。
- 修正同步：稀疏更新报告已根据独立源码审阅，修正 activation-rank 属于新增元数据及 reuse 采用整次 attempt restart 的描述。
- 成败状态：决策计划完成；没有新增 instrumentation、仓库执行或当前基线 benchmark。历史结果仍只用于设计实验，未作为当前加速证据。

### 29｜context-sensitive admission

- UTC 对照：开始：10-11 18:58:46；首份阶段材料写成：10-11 19:09:35；10-11 19:11:00。

- 开始：2026-10-12 02:58:46。
- 首份阶段材料写成：2026-10-12 03:09:35；修订至 03:11:00，独立审阅仍在进行。
- 工作：基于局部发布、token-bound 限制与 parent-budget 研究，检查实际 occurrence overlap/dependency 信息能否安全放宽 `H ∩ T = ∅`。
- 约束：不通过 Positions 重数，不增加与整个 Corpus 同规模的辅助元数据；只读固定提交和原始文献。
- 已完成条件推导：actual-matching 的 exact priority prefix；raw-presence OR 与 potential-producer closure；virtual post-A neighbor IDs 及左坐标 `p−combined_span`；nondependent cohort 中每个 surviving cut 至多两次 transition。
- 排除：scalar/pruned count=0 不等于没有 conflict；不能跳过真正 maximum 选较低独立候选；late addition 不能免费改写旧 births；atomics 不允许任意 live mixed Corpus 读取。
- 03:10 新阻碍：late index.take 的物理 count deletion 会与 frozen membership-map readers 冲突，需要区分 logical reservation 与 physical row deletion。
- 状态：首份阶段报告及条件证明完成，等待独立 source/frontier 审阅；没有仓库执行或 benchmark。

### 30｜研究持续时间与验收范围加强

- UTC 对照：用户明确要求：10-11 18:58:56；确认：10-11 18:59:06。

- 用户明确要求：2026-10-12 02:58:56。
- 确认：2026-10-12 02:59:06。
- 范围更新：禁止提前停止探索，或提前写好最终报告等待定点发送；须充分利用至北京时间 11:00 的研究时间，持续实质探索。
- 对既有产出的影响：所有已完成报告、模型与实验计划均是可继续修正的阶段材料，不表示最终方案已锁定。
- 状态：要求已确认并纳入后续工作；11:00 前的完成记录只表示某个明确子问题或本轮检查结束，不代表整夜任务结束。

## 03:00 起的继续研究

### 31｜压缩表示中的 contraction 工作

- UTC 对照：开始：10-11 19:00；本轮阶段材料完成：10-11 19:09:46。

- 开始：2026-10-12 03:00。
- 本轮阶段材料完成：2026-10-12 03:09:46；相关压缩图研究继续。
- 工作：结合已完成的 greedy/证据分析，深入阅读 [RePair in Compressed Space and Time](https://arxiv.org/abs/1811.01472) 第 2—5 节，并继续研究 context-summary closure 与 dynamic index。
- 新核验：论文 AA 优先级按最大 nonoverlapping occurrences 计，与本项目 raw adjacency count 不同；改进界仍包含各轮 boundary-summary 重建的 `Σ_h n_h` 工作；论文实际实现的是较慢的 full-grammar-scanning 版本。
- 来源状态：这是对已引用论文的进一步核验，不记作新论文发现。
- 本轮产出：整理 compressed RePair、静态/动态 q-gram indexes、BPE automata、Re²Pair 语义，以及 weighted-DAG 与 AA-run 成本分析；给出统计不足反例和 compatible non-AA batch 的 four-window 公式。
- 成败状态：首份阶段材料完成；上述结果限制可迁移范围，不构成压缩表示已经适合本项目的结论。
- 边界：未执行或修改仓库代码。

### 32｜parent-budget guard 独立协议模型

- UTC 对照：研究开始：10-11 19:01:00；最终执行开始：10-11 19:08:46.027855。

- 研究开始：2026-10-12 03:01:00。
- 最终执行开始：2026-10-12 03:08:46.027855。
- 最终执行结束：2026-10-12 03:08:51.017278；耗时 4.990 秒。
- 报告完成：2026-10-12 03:10:16 前；结果于 03:10:22 报告。
- 工作：独立枚举 parent-budget guard 协议状态，验证 pooled-child 上界、coherent guard 降低、具体 candidate admission、acknowledgments 与 Positions closure；检查 race/order 负面对照。
- 结果：PASS；27,790 个 global state instances、826,932 次具体 admission 检查，311,113 次被允许的选择均等于 fixed final target 的精确 winner；24,520 次 monotone guard publications、57,436 次 partial-ack 保留 guard 检查、72,320 次零权重 payload 未齐全检查。
- 反例：五项必要负面对照，另加 unsafe intermediate arithmetic。包括 stale heap priority、低数值 reserved ID 的错误 tie witness；`C+M` 可能溢出，而在预算前提下 `(C−L)+M` 安全。
- 开发修正：初轮 harness 误要求“仍有任何 family 未关闭时，被选中的只能是外部 independent candidate”。实际已关闭 family 的候选也能合法胜出；精确 target-winner 比较当时已通过，修正的是过强断言。开发失败 trace 已单独保留，未发现 protocol-envelope 反例。
- 可复验材料：`verify_parent_budget_protocol.py`、`parent_budget_results.json`、阶段研究材料；源 SHA-256 为 `58eba18a15d9…`。
- 限制：模型假设固定 target、support 唯一归属、exact-once summaries、合法全局 rejection 和 coherent admission；不验证真实 owner acknowledgments、Rust memory ordering、新增 generations、deadlock/liveness、bounded memory 或性能。
- 状态：本轮独立协议验证完成，仍为中间研究材料；未执行仓库代码。

### 33｜sealed count rows 与短期 physical slots 组合

- UTC 对照：开始：10-11 19:02:11；本轮阶段报告完成：10-11 19:14:12。

- 开始：2026-10-12 03:02:11。
- 本轮阶段报告完成：2026-10-12 03:14:12。
- 工作：研究第一次 local witness touch 时解析短期 physical count slot 的设计，与 sealed rows 组合检查 exactness/lifecycle、scalar 操作节省、稀疏互斥写及 atomic 替代方案。
- 条件结论：layout 冻结时，first-touch 保存的 slot 可省第二次 commit lookup；但 G 次 local first-touch 查询可能多于 coalesce 后的 D 次查询，不能只计删除的一端。
- 最小形态：复用 group creation 与 `Option<slot>`，AA 处理留在 prepare 范围；需要 batch/attempt stamp，只有 compaction epoch 不够。
- 原子替代：合法输入流下保留 raw residual 至 join 是公平对照；立即清零再 fetch_sub 不安全。boxed OnceLock rows 可提供 scoped safe reclamation，但 current-count membership 与 unknown-row losses 仍需额外语义依赖。
- 成败状态：中间设计与有限模型完成；没有实际 layout measurement、Rust 编译或速度结论。
- 边界：只读研究，不执行或修改仓库代码。

### 34｜two-frontier 限制式流水线

- UTC 对照：开始：10-11 19:04:13；结束：10-11 19:06:30。

- 开始：2026-10-12 03:04:13。
- 结束：2026-10-12 03:06:30。
- 工作：推导 two-frontier 的受限执行时序，并结合源码核验需要满足的安全条件。
- 条件性结果：A1 apply 后 Corpus 稳定时，A2 prepare 可与 A1 gather/commit 重叠；A2 apply 必须等 A1 readers，A2 count commit 必须等 A1 finalization。
- 新义务：candidate birth family 已 closure 或由 guard 排除；部分发布期间禁用不完整 index 的 membership pruning；同步 count/heap；A2 span 元数据 staging 不得使 A1 readers 失效；双缓冲按 bytes 限额；晚发错误丢弃 attempt。
- 成败状态：受限时序分析完成，属于条件成立的协议候选；未验证真实并发实现或速度收益。
- 边界：数学与源码分析，不执行仓库代码。

## 03:06 研究进展

### 29 补充｜实际 occurrence 相容性与 virtual prepare

- 02:59—03:02：读取固定提交的 merge.rs、corpus.rs、index.rs、positions.rs，确认 H/T selection stop、共享切口唯一归属、零权重处理、Relaxed endpoint atomics，以及当前 joined prepare/apply/commit。
- 03:02—03:05：推导 watched-next-candidate 的 presence OR；closure 只覆盖语法上可能 overlap 的 active rules。找到零权重/剪枝相关反例，以及 late-admission 的 `ABBC→XB→XY` 依赖。
- 03:05—03:06：推导在未改写 Corpus 上，用有界映射执行 virtual post-A prepare 的候选，目标是不增加 Positions 遍历。output-span 生命周期、per-owner generation 顺序、中间 payload 和错误顺序仍需明确。
- 当前理论结果：实际 occurrence matching 可作为已有 newborn-witness priority 推导的更细相容条件；安全选取该信息与实际执行协议仍在完善。
- 文献背景：commutativity/gatekeeping、Galois cautious operators、deterministic reservations 与 exact greedy MIS 的原始框架；未据此宣称 BPE 获得对应调度界。
- 成败状态：上述子阶段完成，整体 context-sensitive admission 报告仍在进行；没有仓库执行或 benchmark。

### 31 补充｜有限 context summary 的闭包限制

- 子阶段开始：2026-10-12 03:02。
- 子阶段结束：2026-10-12 03:06:55；整体研究仍继续。
- 反例：语料 `{abab, cbad}` 与 `{abad, cbab}` 有相同 pair/trigram 统计，但 AB→X 只在前者生成 XX，BA loss 也不同。pair 与 trigram 统计不能单独决定该更新。
- 新推导：固定 compatible non-AA batch 的新 pair counts 可由旧 four-token windows 决定；但固定 q-gram profile 在连续收缩下不闭合，AA 还涉及任意长度 run 的 parity。
- 文献核验：dynamic TST-index 的 substring edit 仍须更新 O(k+q) transformed symbols 及 q-gram trie paths，廉价查询不等于删除 contraction 维护工作。
- 修正：BPE DFA 原始构造给出 `Q₀+m·dloc(A₀,1)` 界，不能笼统声称必有指数状态膨胀；论文也未证明 weighted training 的适配。
- Re²Pair 核验：dictionary/parse 独立 RePair 与 phrase completion 产生有效 grammar，不等于复现单一全局 greedy merge 顺序。
- 成败状态：已形成反例和明确的文献适用限制；整体压缩表示研究仍继续。

### 35｜统一 candidate/family frontier

- UTC 对照：开始：10-11 19:07；本轮阶段报告完成：10-11 19:13。

- 开始：2026-10-12 03:07。
- 本轮阶段报告完成：2026-10-12 03:13。
- 工作：结合 parent-budget guards 与 deferred-forest 限制，研究复用现有 optimistic heap entries 的统一 frontier；检查 exact best-first、candidate masking/readiness、family closure、pending membership 与多代嵌套。
- 核心判据：统一 heap 是否真正减少工作和状态，还是仅改变表示。
- 03:10 来源进展：Hjaltason–Samet（TODS 1999）提供统一 nodes/leaves priority queue 的直接先例，并明确比较 one-queue 与 two-queue 及 coverage invariant；已据此推导 BPE 的覆盖条件。
- 新边界：可按 `u∈T 或 v∈H` 惰性识别 affected family。只延后 queue entries 可保留普通 exact scalar publication；若把不完整 scalar births 延至后续 merges，则还需 pending eligibility 和 signed birth/loss debt，重新引入 deferred-forest 的费用。
- 条件结论：已有 optimistic old-pair heap entry 可覆盖 unresolved birth family；一套 frontier 在 coverage、priority upper bound、leaf readiness 和原子 coverage transfer 条件下成立。static old snapshot 可避免第二个 guard heap 及 budget bookkeeping。
- 限制：不能保持普通 best() 完全不变；family 不能按 parent residual 修低或退休，仍需 tag/registry/generation。收益可能只是省第二队列，不能自动删除 child counts、fragment storage 或 closure。
- 来源：[Hjaltason–Samet，Distance Browsing](https://infolab.usc.edu/csci599/Fall2007/papers/a-2.pdf)、[Lawler k-best](https://pubsonline.informs.org/doi/10.1287/mnsc.18.7.401)、[Eppstein](https://www.ics.uci.edu/~eppstein/pubs/Epp-TR-94-26.pdf)、[Fagin–Lotem–Naor](https://www.wisdom.weizmann.ac.il/~naor/PAPERS/middle_agg.pdf)。
- 成败状态：中间报告完成，整理十项反例；未建立生产实现、bounded-memory 或速度结果。

## 03:10 研究进展

### 33-A｜ephemeral slot 独立有限模型

- 所属研究开始：2026-10-12 03:02:11。
- 模型执行起点：未单独记录墙钟时间。
- 完成：2026-10-12 03:09:53 前；保存结果记录耗时 0.889 秒。
- 结果：22,353 个有效 frozen batches 通过，其中 20,196 个 singleton、2,157 个 compatible multi-rule；98,474 次 first-touch queries；18,158 个 rejected cut events。
- 语义覆盖：14,052 个 selected-AA allow-without-loss 切口事件；23,796 个必要零权重 Positions 保留；409,010 个有效 atomic updates 的串行线性化顺序检查。
- 三项预期反例：zero-clamp 加 fetch_sub 的 wrap；dynamic atomic membership 丢失必需 fragment；compaction 后仍在范围内但指向错误对象的 stale slot。
- 可复验材料：`check_model.py` 与 `results.json`；源 SHA-256 为 `6a468767507c…`。
- 限制：独立数学模型；串行线性化枚举不是 Rust atomics/hardware concurrency 执行，也不是 trainer parity。

### 36｜CM-Graph 压缩图成本核验

- UTC 对照：开始：10-11 19:10；本轮核验完成：10-11 19:19:48。

- 开始：2026-10-12 03:10。
- 本轮核验完成：2026-10-12 03:19:48。
- 工作：继续阅读 CM-Graph（HPEC 2024），检查是否仍有需要计算的遍历与归约成本。
- 截至 03:11 的来源限制：两个 host 的全文直接读取失败；官方 search 索引只提供 Algorithm 2 的原始 CSR traversal 加 hash reduction 片段。没有把索引片段记录为全文阅读。
- 03:19:48 补充：从原始 PDF 的官方 indexed passages 恢复 Algorithms 1—4 及分析；直接全文 PDF 访问仍失败，公开仓库仅见 LICENSE/README，未当成可检查的实现。
- 反例：反复转移原始 adjacency 可产生 Θ(N log N) 工作，而相同路径只需 N−1 次 contraction。压缩邻接表示不会自动删除维护工作。
- 成败状态：本轮来源和成本审阅完成，证据仍限于 indexed primary passages；没有测量或实现对照。

### 37｜跨领域原始文献再核验

- UTC 对照：开始：10-11 19:09；结束：10-11 19:11。

- 开始：2026-10-12 03:09。
- 结束：2026-10-12 03:11（本轮检索/阅读）。
- [Amorphous Data-parallelism](https://www.cs.utexas.edu/ftp/techreports/tr09-05.pdf)：阅读第 2—3 节和 4.2 节，确认 ordered dynamic tasks 即使邻域不相交也可能有生成任务的优先级依赖，用作 context-sensitive admission 的对照。
- [The Role of Semirings in IVM，2026-06-05 v1](https://arxiv.org/html/2606.07795v1)：阅读摘要、引言、定理 1—2 和 delta rules；insert-only 等前提限制迁移，不能直接得到 BPE 复杂度结论。
- [CM-Graph，HPEC 2024](https://ieee-hpec.org/wp-content/uploads/2024/09/111.pdf)：两处全文直读未成功；官方索引可见算法片段仍包含 CSR 遍历与 hash 归约。记录为部分来源证据，未宣称完整读过该文。
- 成败状态：前两项完成本轮来源核验；CM-Graph 全文读取受阻。没有把其他领域算法的速度或调度界直接移植给 BPE。

### 38｜将 parent-budget 协议连接到可实现 BPE histories

- UTC 对照：研究开始：10-11 19:12:12；执行开始：10-11 19:17:55.269210。

- 研究开始：2026-10-12 03:12:12。
- 执行开始：2026-10-12 03:17:55.269210。
- 执行结束：2026-10-12 03:17:55.832117；耗时 0.563 秒。
- 报告完成：2026-10-12 03:20:35.022147。
- 工作：从实际 weighted BPE corpus histories 产生 parent-budget families，模拟 chunk publication，并比较完整串行 reference。
- 结果：PASS；8,638 个完整 histories、12,165 条 selected rules 的 `(pair, frequency, output ID)` trace 与最终 token paths 一致；118 次 early reservations 全部履行，其中 66 次仍有 physical jobs pending，71 次由 guard tightening 放行。后两项是可重叠的不同分类，不能互换或相加。
- 可实现正例：202 个 physical occurrence jobs 中先完成 2 个，将 BC family 的 C100 分为 residual40、两个 child 各 30，guard 降至 70，可预留 EF80；仍有 200 jobs 待完成。调度这两个有用 jobs 的发现成本没有被证明为免费。
- 集中反例：同样 loss 集中到单个 child 时 guard 仍为 100，正确阻止 EF80。提前 reservation 不等于提前执行，模型等待当前 cohort 全部 Corpus/Positions 发布后才执行预留规则。
- 范围：每个 frontier 最多预留下一 cohort 的一条规则；candidate 读取是 canonical count，未实现 production lazy heap。包括 3,467 个 AA singleton fallback cohorts、97 个必要零权重 birth positions、2,899 个 lower-ID 非空 histories。
- 可复验材料：`bpe_reachable_verify.py`、`bpe_reachable_results.json`、阶段研究材料；源 SHA-256 为 `89370b127996…`。
- 状态：可实现 history 验证作为中间结果完成；保留此前模型哈希，不运行仓库实现，不声称 Rust 并发或性能通过。

### 29 补充｜virtual prepare 的代价与 membership epoch

- 开始：2026-10-12 03:11。
- 本轮修订完成：2026-10-12 03:12:31。
- 新限制：virtual q 的 prepare 如果长于 A 剩余尾部，会延长 stable-read fence 并推迟 A commit；提前启动不保证更快。
- 新反例：virtual key XB 在旧 frontier F 中本来不存在，不能直接用 frozen-F membership 判死；应映射回 original-F witness BB，或使用已认证的 logical post-A membership。
- 顺序结果：每 key 的左位置坐标都减去固定 endpoint span，顺序保持，因此 virtual prepare 本身不需要新增 Positions scan/sort。
- 来源补充：Pingali 等原始报告 Figure 6 说明新生成 ordered work 的依赖，进一步确认 BPE 仍需要独立的 witness-priority 条件。
- 状态：阶段材料继续修订，没有实现或速度结论。

## 03:15 研究进展

### 35 补充｜static family 的进展限制与 stale upper snapshot

- 记录时间：2026-10-12 03:15。
- 修正：若原 cohort 已被同一个最高优先级 conflict parent 阻塞，static C 只简化 coverage 表示，不能放行较低候选；要取得新的提前进展仍需更紧上界或 family closure。
- 新推导：若 heap snapshot `C_hat ≥ C`，则 `U_hat=C_hat−L+max(b_c)` 仍安全，携带固定 slack `C_hat−C`。正确性不必强制额外捕获 exact C，但较松上界会削弱提前放行能力。
- 成败状态：中间报告已修订；没有把表示简化等同于调度提速。

### 39｜新增定期源码检查与更新基线要求

- UTC 对照：用户指令：10-11 19:15:09；确认：10-11 19:15:13.767。

- 用户指令：2026-10-12 03:15:09。
- 确认：2026-10-12 03:15:13.767。
- 范围更新：研究期间定期检查新 commits，按最新代码及实验数据继续分析。
- 用户报告：Positions 优化已推送，回归结果可接受。此处记录为用户提供的状态；最新 SHA、实际改动和实验证据等待源码核验。
- 对当前研究的影响：原先以 `02620e0…` 为依据的结论仍保留其版本来源；03:18:21 后的源码基线更新为 `ea1d4d56…`，具体接口和性能记账须重新核对，不能再把已推送 child-Positions 描述成未提交改动。
- 状态：新增要求已确认，开始最新推送审计；持续研究和 11:00 验收要求不变。

### 40｜最新 Positions 推送审计

- UTC 对照：开始：10-11 19:15:24；完成：10-11 19:21:22。

- 开始：2026-10-12 03:15:24。
- 完成：2026-10-12 03:21:22。
- 工作：核对最新分支与 `02620e0…` 的差异，读取 Positions 优化及其新实验数据。
- 03:15:36 首轮读取：GitHub compare 与 commits list 均返回分支仍为 `02620e0ad37d8b165b5923902a05184e483cbd88`，common base 未变。继续核对 direct refs 及其他 branches；这次读数不能证明用户没有推送。
- 03:18:21 新推送已可见：[ea1d4d56e30f8b7e5a1d91a405959a5054c8e30d](https://github.com/Yikai-Liao/tokenizers/commit/ea1d4d56e30f8b7e5a1d91a405959a5054c8e30d)，比原 `02620e0…` 正好前进一个 commit，merge base 相同且无分叉。
- 改动范围：positions/index/merge/mod 四个代码文件；该次 diff 没有新增 benchmark-report 文件。
- 新源码确认：非递归 fixed roots 与 appendable leaves；opaque ReadTask；Complete/Partial 协议删除；floor 统一在 owner 归约后处理；owner stable grouped sort 保留同 key fragment 顺序；reuse 用 kmerge；成功结束时 joined parallel retirement。
- 具体计数：四个变更文件的非测试、非空白/注释 production LOC 共 1491→1551，Positions+106、merge−54、index/mod 各+4；两个版本四文件均未发现 unsafe。
- [exact-head CI](https://github.com/Yikai-Liao/tokenizers/actions/runs/38167402222) 于 03:19:41 只见 Secret Leaks success，combined statuses 为空，不能当作 BPE tests 或 benchmark PASS。checked-in tests 已阅读但未执行。
- 成败状态：固定 SHA 审计完成，03:15—03:17 旧读数已被替代。没有新的 benchmark artifacts，用户报告的可接受回归仍未独立复现。

补充检查记录：2026-10-12 03:11:12.771，PR2501 状态检查确认未变化；没有新增结果通知。此项不作为 Positions 新推送的验证证据。

### 41｜ephemeral slots 的接口简化

- UTC 对照：开始：10-11 19:15:44；本轮补充报告完成：10-11 19:24:30。

- 开始：2026-10-12 03:15:44。
- 本轮补充报告完成：2026-10-12 03:24:30。
- 工作：检查 private consuming joined-round ownership 能否删除 runtime batch stamp；研究是否可直接在现有 sharded hash tables 中安全复用 round 内 offsets，而不先引入 sealed rows。
- 版本限制：等待最新 Positions 基线审计，再确认具体字段与接口；不将旧版结构假设当成新代码事实。
- 最小变体：保留 action 和 Positions 顺序，循环中只逻辑退休；保存物理 cell 至循环结束，仅把 retired offsets 降序排列后 swap-remove。
- 接口结论：private single-index、joined、consume-once flow 可免 runtime stamp；未声称编译器单独保证 semantic index identity。IndexMap 公共 API 可提供 per-shard offset 对照，不需 sealed rows 或 application unsafe；更换字典本身仍有成本。
- 成败状态：有界补充分析完成，已按 ea1d4d56 基线核对；布局数字均为估算，没有 Rust measurement 或 trainer benchmark。

### 42｜图收缩维护框架的补充文献核验

- UTC 对照：开始：10-11 19:15:42；结束：10-11 19:16:23。

- 开始：2026-10-12 03:15:42。
- 结束：2026-10-12 03:16:23（本轮阅读）。
- [SPoCH，ICS 2025 / arXiv v3](https://arxiv.org/html/2412.18008v3)：阅读 4.1—4.2 节和 evaluation 限制，提取同类 phase-concurrent updates、延后 CSR rebuild 及 buffered additions 仍须可见的维护思路。
- 迁移限制：CH 的 min 聚合是幂等的，BPE additive counts 必须 exactly-once；放宽 contraction 顺序会改变 hierarchy，不能直接用于 exact greedy BPE。LazyCSR 仍有 count 可见性、buffer memory 和延后 full scan 的成本。
- [Batching BPE Tokenization Merges](https://arxiv.org/html/2408.04653v1)：重新打开已有参考文献，尚无新增结果。
- 成败状态：维护框架来源核验完成；没有引用其速度结果作为 BPE 加速证据。

## 03:18 研究进展

### 43｜coarse child-bucket family guards

- UTC 对照：开始：10-11 19:16；本轮推导完成：10-11 19:18:35。

- 开始：2026-10-12 03:16。
- 本轮推导完成：2026-10-12 03:18:35。
- 工作：推导固定 coarse child buckets 的 `U=C−L+max(bucket)`、k=2 形式及 single watched-q certificate，以减少精确逐 child budget 维护。
- 来源：[Count-Min 原始论文](https://www.cs.ox.ac.uk/people/graham.cormode/pubs/papers/cm-full.pdf) 第 3—4.1 节于 03:17—03:18 阅读；只借用非负汇总的安全上界思路，不引入近似 winner 或文献性能承诺。
- 进展：已有独立推导确认及可实现 strict-win 示例；分开的数学模型验证仍在进行。
- 成败状态：中间理论材料完成，尚无生产数据上的 bucket 分布、费用或加速结论。

基线沟通记录：2026-10-12 03:18:40.047，已向用户确认新提交可见；源码审计继续，未声称回归已被独立复现。

## 03:20 研究进展

### 44｜weighted corpus DAG 的局部更新恒等式

- UTC 对照：开始：10-11 19:10；本轮推导完成：10-11 19:19:48。

- 开始：2026-10-12 03:10（与 CM-Graph 后续核验同一轮）。
- 本轮推导完成：2026-10-12 03:19:48。
- 工作：在 unit-edge、weighted-final、acyclic corpus automaton 上，推导一次 fresh non-AA rewrite 的局部 α/β mass 恒等式，以及 cab/abc/abab 的 count delta。
- 独立检查：03:14 的 inline Python 检查通过 8,303 个 synthetic corpora；工具 wall time 为 0.826453929 秒，03:14:21 时钟提醒是完成上界，未单独保存精确运行起止，未补造时间。
- 有限范围：27 个长度 3 的三元字符串的全部 1/2/3 元子集，共 3,303；另用 seed=137 生成 5,000 个有限语料，word 长度 1—8、抽词 1—20 次、权重 0—3。只检查一次 non-AA 规则的α/β、trigram/abab delta 与直接替换 oracle，不含μ独立断言、多轮、AA 或复杂度势能。
- 归档：原 inline 脚本于约 03:24 事后原样保存为`dag_non_aa_identity_check.py`，记录为`dag_non_aa_identity_check.first-run.txt`；没有把归档当成新运行。归档源 SHA-256 为`d144df33d95b…`，并非原执行同时采集的哈希。
- 未闭合部分：sparse incoming-label μ update 已有分析推导；copying-potential 的摊还界仍未证明，需独立审查。
- 成败状态：建设性局部公式与阶段材料完成，架构总工作/空间优势未建立；没有仓库程序、测试或 benchmark 执行。

### 45｜固定 child-key bucket 的独立模型补充

- UTC 对照：研究开始：10-11 19:20:40；执行开始：10-11 19:22:15.608986。

- 研究开始：2026-10-12 03:20:40。
- 执行开始：2026-10-12 03:22:15.608986。
- 执行结束：2026-10-12 03:22:19.780245；耗时 4.171 秒。
- 报告完成：2026-10-12 03:23:39.221201。
- 工作：验证固定 child-key buckets 的上界、coherent 发布及完整 greedy history；另以真实固定数组模拟 incremental publication。
- 结果：PASS；51,822 个 history/partition 配置（8,637 个语料配置×6 种分区）保持完整 trace/paths；523 次预留全部履行，349 次仍有 physical jobs pending，241 次由 tightening 放行，后两类可重叠。
- 增量检查：80 个 directed reachable batches 的全部 job arrival order/partition 组合共 5,568 次；包含 rejected births、零权重 payload 和 u64 边界。
- 关键正反例：同一 child 全部映射到固定 bucket，碰撞只使 guard 变松；按 job/thread/chunk 分桶会把同一 XC80 拆成 40+40，得到错误 U40 并放行 EF60，负面对照已捕获。
- tie 限制：bucket total 不提供精确 child tie witness；模型使用所有可能 old/child keys 的保守最小 key，不声称凭最大 bucket 即可获得紧 tie 界。
- 可复验材料：`bpe_reachable_bucket_verify.py`、`bpe_reachable_bucket_results.json`、阶段研究材料；源 SHA-256 为`394d3ca0feea…`。
- 独立数学依赖：`bpe_reachable_verify.py`哈希仍为`89370b127996…`。
- 状态：中间模型补充完成，旧模型未改；没有仓库执行、生产性能或 Rust memory-model 结论。

### 29 / 43 补充｜实际 overlap 与 coarse guard 的分析例

- 03:13—03:17：推导更强的一步 actual-overlap certificate，要求 residual `R=C−L`、`R≥L` 及对其他 old priority 上界的支配；virtual prepare 只局部过滤已失效 starts，不重数 counts。
- 共享切口例：`ABCD:10, AB:110, CD:100, BC:80, EF:70` 的序列为 AB120、CD110、BC80、EF70、XY10；BC loss10 只记一次。等值例中 BC10 后会先出现 XY10，再轮到 EF9，不能盲目继续旧 prefix。
- 03:17—03:20:40：独立核对固定 k buckets 的 `max_child ≤ max_bucket ≤ L` 与 coherent anytime guard；可实现 AU110/BU105/UV100 例可收紧为 UV40、XV30、YV30。
- 费用及边界：other-old 的保守 C2、potential q-loss producer closure、metadata/readiness、strict tie 都须保留。full scalar closure 后可另用 `max(R,max_bucket)`，不能提前套用。
- 状态：分析与阶段文档修订完成；这些例子是手工推导，没有被描述成穷举测试或 benchmark。

### 46｜新 Positions 基线的独立可维护性复核

- UTC 对照：开始：10-11 19:19:27；结束：10-11 19:20:23。

- 开始：2026-10-12 03:19:27。
- 结束：2026-10-12 03:20:23。
- 工作：读取 `ea1d4d56…` 的 Positions production 实现、index commit/retire/merge_positions，以及 merge/positions 的前后 diff。
- 确认：跨模块 complete 协议被删除；内部增加 custom leaf growth、directory move 与 root task 处理。维护工作从外部阶段协议转到 Positions 内部，不能仅以删除旧分支或代码行数判断总体变简单。
- 代码量：审计报告的 production LOC 为 1491→1551；这是源码规模统计，不是运行成本或可维护性评分。
- 用户沟通：03:20:39.975 已给出可维护性分析；没有声称新 benchmark 已独立复现。
- 同轮理论修订：coarse guard 的 full scalar closure 后界 `max(R,max_bucket)` 于 03:20:23 补入阶段材料。
- 状态：本轮独立源码复核完成；最新基线审计与后续定期分支检查继续。

### 41-A｜joined-round slot 简化的数学检查

- 所属研究开始：2026-10-12 03:15:44。
- 检查完成：2026-10-12 03:21:28；独立运行起点和耗时未记录。
- 接口修正：private joined consume-once round 可以由所有权保证 slots 不跨轮存活，无须 runtime epoch/stamp；只有允许跨阶段持有或防止伪造接口误用时才需要相应动态校验。此处收窄了早先“必须 stamp”的表述。
- 源码/API 对照：IndexMap 2.14 提供 get_full 与 indexed mutation，可作为每 shard 的独立 slot-fusion 对照；现有 AHashMap 没有相同的安全 ordinal API，不能直接假定已有此能力。
- 结果：32,767 个 descending-removal subsets、364,250 个 owner-action sequences 通过；其中 201,131 个 underflow 序列的报错步骤与参考一致，不是意外失败。
- 负例：ascending removal 会使后续 offsets 错位；重排 mixed birth actions 可破坏 Positions 顺序。
- 可复验材料：`check_model.py` 与 `results.json`；源 SHA-256 为 `b093d4479af8…`。
- 限制：只检验 Python dictionaries/lists 模型，没有 Rust/trainer 执行；补充设计文档仍在进行。

### 47｜确定性 weighted Misra–Gries family summary

- UTC 对照：开始：10-11 19:22；本轮理论补充完成：10-11 19:29。

- 开始：2026-10-12 03:22。
- 本轮理论补充完成：2026-10-12 03:29。
- 工作：审查 fixed-partition guards，并研究用 1—4 个 key slots 的确定性 weighted Misra–Gries summary 上界化 max-child；检查 anytime 单调更新、task 合并、对抗碰撞及与固定 buckets 的费用对比。
- 版本背景：已读取新 `ea1d4d56…` 基线审计。
- 理论结果：one-key weighted cancellation 的 `E=balance+D`，以及 k-slot 的 `E=max(residual)+D`，可作为确定性 family maximum 上界；coherent `U=C−L+E` 单调不增。
- 合并规则：replay 与 bulk-prune 两种 task merge 已推导；bulk-prune 使用质量不等式，不能误套单流的等式。
- 来源与限制：MG、mergeable summaries、Count-Min、SpaceSaving 原始来源已核验；材料包含碰撞/歧义、不互相支配的 fixtures、closure/tie/zero/u64 条件。独立模型仍在运行，未建立生产速度结论。

### 48｜weighted Misra–Gries / Boyer–Moore 独立模型

- UTC 对照：研究开始：10-11 19:24:00；开发轮：10-11 19:26:58.509914；10-11 19:27:07.951085。

- 研究开始：2026-10-12 03:24:00。
- 开发轮：03:26:58.509914—03:27:07.951085，通过后增加 bulk merge 后直接更新和 arrival-order precision fixtures。
- 最终执行：2026-10-12 03:27:40.819051—03:27:49.349198；耗时 8.530 秒。
- 报告完成：2026-10-12 03:29:43.081047。
- 工作：验证 coherent publication、可实现 histories、u64 边界及 unsafe counters，核对 exact invariant 和 mergeability。
- 结果：PASS；29,524 个 bounded stream/k cases、69,096 个完整 history/configuration 比较；828 次预留全部履行，504 次仍有 physical jobs pending。bulk/replay merge、zero/rejected payloads 和负面对照均按规格检查，未出现有效案例失败。
- 重要限制：merge/arrival 顺序可改变 summary state 与界的精度，不声称字节级结合性或同样 tightness。MG 与固定 bucket 互不全面优于对方；不同 tie policy 的预留次数不能当成纯 summary 质量或速度对比。
- 可复验材料：`bpe_misra_gries_verify.py`、`bpe_misra_gries_results.json`、阶段研究材料；源 SHA-256 为`3043feae59bc…`。
- 数学依赖哈希仍为`89370b127996…`。
- 状态：中间模型完成；没有仓库执行、内存序或性能结论，不重复计入基线已实现的 Complete/Partial 收益。

### 49｜进一步收敛 Positions 接口与读取路径

- UTC 对照：用户提问：10-11 19:21:53；10-11 19:21:57.807；本轮只读分析结果：10-11 19:23:51.123。

- 用户提问：2026-10-12 03:21:53；确认：03:21:57.807。
- 本轮只读分析结果：2026-10-12 03:23:51.123 已回复用户；准确审阅起点未单独记录。
- 具体候选一：当前 lower_bound 已解码 prefix，随后 cursor(start) 又重复解码；返回 restart index 后复用 iterator.skip_while，可删除重复 prefix decode 和 cursor 的任意 skip 路径。
- 具体候选二：leaf 的有效 bytes 恰好到末尾且没有 padding，Cursor.end 可能冗余。
- 必须保留：small-leaf coalescing，避免把一个简单化接口变成大量 Job。
- 状态：两项明确的收敛机会已给出；独立模型待完成，没有改动仓库，也没有新增性能承诺。整夜研究继续。

### 41 补充｜避免新增 fresh/reuse 存储分支

- 源码复核时间：2026-10-12 03:25:20。
- 发现：当前 Shard.counts 不通过迭代来发布 candidates；initial candidates 来自单独 states 归约，后续 candidates 来自 sorted fragments。
- 推论：IndexMap 对照可保持 fresh/reuse 共用同一 storage type，reuse 继续 key-based signed actions，不必引入新的存储 enum。
- 状态：源码层论证已补入材料；没有 parity run，不能据此直接认定替换字典无行为变化。

### 50｜count-slot 方案的 work/span 与真实偏斜

- UTC 对照：开始：10-11 19:25:50；本轮完成：10-11 19:36:14。

- 开始：2026-10-12 03:25:50。
- 本轮完成：2026-10-12 03:36:14；期间布局问题的处理已单列。
- 工作：构造可实现 fixed-owner key skew，分析 scalar-slot parallel reduction、真实 sort/scratch 成本、eager local 查询与 late global-key 查询，以及 fresh-valid 与 reuse/错误语义的不同要求。
- 基线：`ea1d4d56…` 的统一 child Positions。
- 源码界：N 为 ordinary selected candidate coordinates（含 stale），B 为 ordinary rules，P 为 workers，T 为 prepare tasks；当前 cap 前 T≤P+B，cap 后 T≤ceil(N/2^26)+B，单个 old key 至多 2T 条 local records。新 child concat 后，高 frequency 或多 coordinates 不自动等于大量 owner scalar 工作。
- 构造结果：给出可实现 one-key skew 及对抗性 many-key owner collisions；后者只证明存在，没有搜索实际 ahash collisions。
- 算法结果：已排序 scalar slots 上，parallel segmented scan 加 run-tail disjoint count-slice writes 可达到理想 post-sort O(M/P+log M)；packing、排序、birth 及 publication 仍另计。Rayon 已知 unstable partition 为串行，未宣称整个 sort 具有同一理想 span。
- 独立模型（03:33:54 完成）：512 个 associativity triples、27,063 个 scalar streams、216,504 个 disjoint count partitions、256 个 compatible weighted cohorts、12,416 个 valid floor cases 通过。无效 loss 序列[6,8]、count10、floor5 仍区分旧退休行为与 net underflow。
- 源文件：`check_model.py`，SHA-256 为`de824ce1f6aa…`；结果为`results.json`。模型精确起止/耗时未单独记录。
- 成败状态：有界成本与模型材料完成；没有仓库执行、Rayon 计时或当前真实语料偏斜数据。

### 51｜weighted DAG 独立反例与摊还审查

- UTC 对照：开始：10-11 19:25；10-11 19:26；本轮独立审查完成：10-11 19:29。

- 开始：2026-10-12 03:25—03:26。
- 本轮独立审查完成：2026-10-12 03:29。
- 已核对前提：single-root、unit-edge、exact deterministic DAG；零 final weight 用显式 acceptance flag 区分；先从旧图构造 shortcuts，再 cloning，并重定向全部 a edges。
- 新边界：多 initial states 的扩展需要 START arrival class，才能审查 potential 与复用的安全性。
- 条件证明：single-root、trimmed、unit-edge、fresh non-AA 模型中，以 structural suffix counts σ计费，shortcut 使势能增加不超过 3C；共享 arrival-class 拆分支付 row copying，singleton row 复用和 dead-copy omission 保持 language。抽象累计 copied augmented row sizes≤4N。
- 验证一：20,000 个随机 DAG 中 15,369 个 nonempty trimmed 案例通过 language/α/β/μ与势能检查；seed=1102026，2—9states，labels={a,b,c}，final weights0—3。
- 验证二：5,000 个随机 DAG 中 3,955 个 nonempty 案例执行 10,540 次 non-AA 重写，singleton reuse/dead-copy 省略、精确 language 和逐步/累计势能界均通过；seed=112026。模型精确运行起止未单独记录，完成于本轮审查内。
- 源文件：`independent_weighted_dag_review.py`，SHA-256 为`b14691daadf1…`；`independent_weighted_dag_reuse_check.py`，SHA-256 为`a7c8d66d0cc5…`。
- 限制：4N 是 unit-cost 图 transition-copy 预算，不是 O(N) trainer 定理；具体 pair index、AA、数值/故障时序、Positions 整合和峰值内存仍未验证。多 initial 和 weighted-edge 扩展不能直接套用。
- 状态：条件势能证明和有界检查完成；没有仓库执行或生产推荐。

## 03:29 研究进展

### 52｜weighted DAG 多轮 α/β/μ 与 AA 检查

- UTC 对照：研究阶段：10-11 19:25；10-11 19:28；执行开始：10-11 19:28:22.249765。

- 研究阶段：2026-10-12 03:25—03:28。
- 执行开始：2026-10-12 03:28:22.249765。
- 执行结束：2026-10-12 03:28:39.291329；耗时 17.041416 秒。
- 工作：核对多轮图重写的 α/β/μ、精确带权接受语言（含零权重）和 pair delta，与直接字符串替换 oracle 对照。
- 结果：8,303 个语料、42,200 次重写全部通过，其中 AA20,175 次、non-AA22,025 次。
- 有限范围：3,303 个三元长度 3 words 的 1/2/3 词子集只跑 AA 单步；另以 seed=67319 生成 5,000 个语料，word 长度 0—12、抽词 1—24 次、权重 0—3，每个至多 8 轮当前存在的 pair。不是完整 greedy 选择顺序或生产 trainer 验证。
- 实现范围：没有 trim/reuse/minimize，也未验证 copying-potential 摊还界。
- 可复验材料：`dag_local_mass_multirank_check.py` 与 `dag_local_mass_multirank_check.first-run.txt`；运行时与当前源 SHA-256 均为`7f3a9dc8ca88…`。
- 补充材料：为独立审查形成最小 weighted-DAG rewrite 规格；单 root、unit-edge、deterministic、显式 zero-weight acceptance 等前提继续保留。
- 状态：局部多轮数学检查完成，研究继续；没有仓库执行或速度 benchmark。

## 03:32 研究进展

### 53｜One/Two/Three 与 32-byte Positions 布局问题

- UTC 对照：用户提问：10-11 19:28:53；10-11 19:28:58；官方布局/NonZeroU64 文档核验：10-11 19:29:35；10-11 19:29:48；10-11 19:30:35。

- 用户提问：2026-10-12 03:28:53；确认：03:28:58。
- 官方布局/NonZeroU64 文档核验：03:29:35；当前源码与独立布局审阅：03:29:48—03:30:35。
- 第一个结论：直接给现有 32-byte Leaf/Root 外层 enum 增加 One/Two/Three，预计会增至 40 bytes。32-byte Leaf 的单个 NonZero 无效值已用于 Root；Root 自身 24 bytes 不表示额外 variant 免费。此处为布局推断，repr(Rust)不保证该结果。
- 环境事实：检查 command -v rustc 及标准 binary 位置未找到编译器，因此没有报告实测 size_of 或编译验证。03:30:17.871 向用户说明了这一限制。
- 第二轮分析：03:30:57—03:31:34。若将 compressed Leaf 长度移入 byte header，使 Compressed(SmallVec) payload 为 24 bytes，再用 Leaf enum 承载 Empty/One/Two/Three，外层 Leaf-or-Root 可能复用未用 discriminants，预计仍可 32 bytes。
- 代价：每个 compressed leaf 增加 8-byte header 及 capacity/header-load 影响；2/3 项可能免分配；从 Three promotion 最多重编码旧三值及新值。具体 SmallVec features 与 toolchain layout 仍须实测。
- 成败状态：区别了“原样加 arms 预计 40B”和“改表示后可能 32B”，没有把两者混为一谈；新表示仅是未编译候选，未采纳或修改仓库。
- 工作衔接：本轮布局问题处理后继续 count-slot 的 work/span 分析。

### 54｜exact count 闭合前的 child identity 证书

- UTC 对照：阶段材料完成：10-11 19:31；10-11 19:33。

- 本轮准确开始时间：未单独记录。
- 阶段材料完成：2026-10-12 03:31；03:33 报告并复核相关模型。
- 候选条件：严格 `b > C−J−b`，并支配所有 outside envelopes，可在 exact count 尚未闭合前确定下一 child 身份。
- 限制：identity 证书不等于 count、Positions 或 Corpus 可执行性已完成；仍需 payload/read fences 及最终 exact traces，不能把 lower bound 当作最终频率。
- 新基线源码：production MergeRecord 仅保存 pair，count 属于 cfg(test)；但当前 Candidate API 仍要求 exact priority 和 owning Positions，因此不能把此证书视为现有接口可直接接受的输入。
- 成败状态：只读理论与接口边界材料完成；未实现或测试新的 early-execution 协议。

03:33 文献/模型交叉复核：MG 理论补充已将独立有限模型状态更新为完成，引用阶段 48 的 29,524 stream/k cases 及 69,096 history/configuration 检查；没有重复计算为新一轮执行。

### 55｜dynamic terminal-frequency 下界

- UTC 对照：开始：10-11 19:34:35；本轮理论材料完成：10-11 19:40。

- 开始：2026-10-12 03:34:35。
- 本轮理论材料完成：2026-10-12 03:40。
- 工作：基于 ea1d4d56 审计，研究剩余训练 horizon 内的 survival/depletion certificate，以及可用于剪枝的终止频率下界。
- 本轮产出：disjoint-token matching/vertex-cover、directed threshold-graph top-r hit budgets、frozen-key excess-mass depletion（统一与 distribution-aware）、weighted dual packing、整个 threshold 集合 compatible 等条件证书。
- 语义边界：H 为未填 vocab slots 加 inactive reserved IDs；需处理 underfill/没有 final merge、AA raw counts、max-length、fresh/reuse restart、严格<L 退休，以及 prepare 阶段须具备 complete-birth bound。
- 有界 smoke checks：两份脚本各枚举相同 2,044 个长度 2—10 的 binary words，H=1—7、L=1 至初始最大 raw frequency；在完整未剪枝 fresh reference 轨迹上验证 depletion，以及 depletion+directed top-r hit 条件。均 exit_code=0，不是 pruned storage lifecycle 测试。
- 原运行工具 wall time 分别 0.058502972 秒、0.28198188 秒；未捕获精确起止。结果分别早于 03:36:37 和 03:38:41 提醒，但提醒不是精确执行结束时间。
- 归档：`check_floor.py` SHA-256 为`0514a6742a05…`；`check_floor_graph.py`为`b55a8fb1b6f6…`；README 保留原 stdout 和复现命令，没有为补日志重跑。
- 完整 pruned lifecycle 独立验证另见阶段 58。
- 文献限制：[Theoretical Analysis of BPE](https://arxiv.org/abs/2411.08671)、[A Formal Perspective on BPE](https://aclanthology.org/2023.findings-acl.38/)提供总压缩 utility/近似分析，不直接给本任务的 last-frequency 下界。
- 状态：中间条件推导完成；没有普适安全 floor 或性能结论，无仓库执行或编辑。

## 03:36 研究进展

### 56｜用户新增目标：按最终词表预算动态提高安全剪枝界

- UTC 对照：首次要求：10-11 19:33:53；10-11 19:34:00；补充要求：10-11 19:34:03；10-11 19:34:09。

- 首次要求：2026-10-12 03:33:53，希望由最终 vocab size 推导安全的最终频率下界，以提前过滤/释放；03:34:00 确认。
- 补充要求：03:34:03，询问动态更新；03:34:09 确认。
- 工作：03:34 起分别进行构造性推导与词表预算反例审查，核验 horizon、fresh/reuse、ID 及停止语义。
- 初步充分条件：fresh-only 且存在 count≥L 的 pairwise vertex-disjoint keys 集合 S 时，`|S|≥2K−1`可能保证随后 K 条规则频率≥L；推广方向是 high-count 图的 vertex-cover 下界大于`2(K−1)`。
- 适用性：该证书可能只在训练后段、较短剩余 horizon 才有用；最终 vocab slots 与实际规则次数的对应仍须明确。
- 排除简单猜测：`aababa`与`abaaba`都有 AA1、AB2、BA2，但下一阶段最高频率分别为 1 和 2，当前第 k 大 pair 频率不能直接充当最终下界。
- 成败状态：新目标与动态要求已纳入研究，正例和反例是初步理论材料；尚未提交最终安全/收益结论。

### 57｜weighted DAG 的 AA 独立扩展审查

- UTC 对照：开始：10-11 19:32；本轮完成：10-11 19:36。

- 开始：2026-10-12 03:32。
- 本轮完成：2026-10-12 03:36。
- 工作：对 H-only odd-prefix parity recurrence 使用独立全图 parity oracle，检查 AA 语言、zero/nonzero finals、α/β/μ、context deltas 与 local cleanup。
- 条件势能结果：raw AA 邻接 R 满足 C≤R≤2C（C 为实际 nonoverlap contractions）；沿用 shortcut/copy 计费，可得 mixed AA/non-AA 的抽象 copied-row 预算 D_total≤7N。不是 trainer 运行时间界。
- 反例与修正：`aaaa`证明 non-AA 的“第二 incoming label 表示仍有 live old prefix”不适用于 AA；给出基于精确α/O 和 structural acceptance 的 local cleanup predicates，并与 full trim oracle 比较。
- 随机验证：15,000 个 2—10state DAG 中 12,787 个 nonempty 案例、10,996 次 AA 步骤通过；seed=11102026，final weights0—3。额外 192 个长度 1—96 的 long-run cases、912 次 AA 步骤均通过。
- 可复验材料：`independent_weighted_dag_aa_check.py`与同名前缀`.result.txt`；源 SHA-256 为`73561d9401a9…`。各模型精确运行起止未单独记录。
- 限制：single-root、trimmed、acyclic、deterministic、unit-edge/final-weight 前提不变；具体 pair index、ordered failures、memory/Positions 整合仍开放，没有仓库执行或生产建议。

03:36:14 布局材料补充：Positions inline/header 设计说明已保存，加入 Vec 压缩 header 替代方案；全部 size 与 allocation 取舍仍未编译或测量。

### 58｜动态终止频率下界的独立 BPE 模型

- UTC 对照：开始：10-11 19:37:50；最终执行：10-11 19:44:37.043926；10-11 19:44:38.166388；10-11 19:46:10.174414；10-11 19:46:17。

- 开始：2026-10-12 03:37:50。
- 最终执行：2026-10-12 03:44:37.043926—03:44:38.166388（约 1.122 秒）；报告完成 03:46:10.174414，03:46:17 修正文档空格。
- 工作：对完整未剪枝 greedy 参考与动态 floor 剪枝模型进行比较；实际模拟永久 index 退休及低频 lineage 事件省略。比较全部 `(pair, raw weighted frequency, output ID)` 规则历史和最终词路径，保留等于 floor 的键及零权重 Positions。
- 证书：endpoint-disjoint matching、定向 threshold graph touch 预算、uniform/distribution-aware excess-mass 消耗预算，以及整个 threshold 集的兼容性证书。行动上界 H=未填槽数 K+未激活 reserved ID 数量 Z；使用历史已证明下界的最大值，只退休严格低于下界的键。
- 成功：16,988 次完整剪枝/参考比较通过，包含 24,828 条选择规则、7,540 条 AA 规则、6,853 次零词表增长的 reserved 动作；4,679 次 floor 提高中 2,054 次发生在后期。退休 5,915 个当前键，省略 5,834 个低频 lineage 事件，保留 21,072 个等于 floor 的键和 308 个零权重 birth 位置。
- 专项：304 个整个 threshold 兼容集的对抗 horizon 配置全部通过；784 个选中动作均达到下界。覆盖共享 head/tail、集外低频 AA、new/new 边界、长度限制、较小数值的 fresh ID 及 reserved 动作。
- 反例：K-only 行动预算、reuse alias、仅兼容子集、等号错误退休、同 histogram 推断终止频率等五类控制按预期失败。9,964 个目标 underfill 配置提醒：不一定存在“填满词表的最终合并频率”。
- 证据：阶段研究材料、`verify_dynamic_floor.py`、`dynamic_floor_results.json`、`dynamic_floor_run.log`。源 SHA-256：`3daa586d5864…`；参考依赖 SHA-256：`89370b127996…`。
- 边界：这是有限标准库数学模型。直接重数仅用于参考 oracle；没有提出生产 Positions 重数。没有执行仓库代码、验证 Rust 并发/完整错误顺序，或测量证书维护成本与速度。03:42 时的 16,683 通过摘要由本次 16,988 最终结果取代。

### 59｜weighted DAG 的共享条件与数值边界

- UTC 对照：开始：10-11 19:30；本轮材料更新：10-11 19:38。

- 开始：2026-10-12 03:30。
- 本轮材料更新：2026-10-12 03:38。
- 工作：构造没有整词重复却可共享/不可共享的两类语料，区分≤4N/≤7N 防 copy 膨胀界与真正共享收益；推导有限显式语料中α/σ≤U 的见证界及β溢出反例。
- 新来源：[Daciuk 等，Computational Linguistics 2000](https://aclanthology.org/J00-1002.pdf)，核读右语言等价、sorted construction 及成本；当前历史日志不足以估计 weighted-DAG 共享程度。
- 03:35—03:37 源码检查：只读 ea1d4d56 的 vocabulary.rs/corpus.rs；无限 alphabet 只记录 presence，有限 alphabet 使用普通 usize 加法，不能据此保证 token mass 适合 u64。初始 pair counts 在 floor 前做 checked accumulation，为只保存初始≥2 边可达状态β提供新的候选界。
- 待审查：该更细β界仍需独立证明，不作为已完成的机器数值安全结论。
- 修正：旧 AA 例子混用了物理与加权 contraction；`aaa:2`只有一次物理 contraction，weighted replacement mass 为 2。物理 N 的摊还预算不以 word weight 放大。
- 成败状态：阶段材料已更新；没有执行仓库、读取新语料或进行 benchmark。

## 03:39 研究进展

### 60｜词表预算下界的独立反例与 fresh-lineage 审查

- UTC 对照：开始：10-11 19:34；本轮报告完成：10-11 19:39。

- 开始：2026-10-12 03:34。
- 本轮报告完成：2026-10-12 03:39。
- 结论：当前 histogram 与剩余预算不能唯一确定最终频率，当前第 K 大 count 不安全；但“histogram 永远不能给非平凡下界”的 blanket 结论同样错误，2K−1 个 eligible high-count disjoint keys 已给充分证书。
- 紧性构造：strict max-length 配置下，相同 histogram 的 rank-K 频率可为 1 与任意 N；matching cardinality 仅 2K−2 不足。独立 toy 检查 K=2—9、N=1000，通用任意 N 为数学构造结论。
- fresh lineage：每个未来 pair 有唯一 snapshot boundary 祖先，count 不超过该祖先，包含 AA 和 inactive reserved ID；可支持严格低于有效 terminal bound 的永久剪枝。等于界的 ties 必须保留，active identity reuse 不在此证明内。
- 预算修正：剩余 vocab 增长 K 不总等于规则次数；若 Z 界定可一次性激活的 inactive reserved identities，且不发生已 active identity reuse，则 K+Z 可作保守 action 预算。动态 max-past horizon 证书的适用条件已整理。
- 模型材料：`vocab_budget_counterexample_model.py`，当前 SHA-256 为`e4f458ee1ab4…`。已读取源码与报告；完整调用记录/运行起止待补，未把默认__main__搜索当成全部 gadget 验证入口。
- 状态：有界反例及条件证明作为阶段材料完成；不执行仓库，不把 toy 结果当成生产实现通过。

### 61｜dynamic floor 的 cover/packing 推广

- UTC 对照：推导时间：10-11 19:35；10-11 19:38；10-11 19:38:40。

- 推导时间：2026-10-12 03:35—03:38；来源检索：03:38:40。
- 工作：将 threshold pair graph 的 vertex-cover 下界推广为 oriented action touch-set 的 set-cover dual；任意可行整数 packing 满足总权重大于`(R−1)×每 action 容量`时可形成候选证书，正确性不要求求出 LP 最优解。
- 来源状态：03:38:40 仅发现 snippet；03:43:10 已实际阅读[Trevisan 讲义](https://www.cs.cmu.edu/afs/cs/academic/class/15750-s16/Handouts/Trevisan.pdf) §8.2，印刷页 62—63/PDF 页 67—68。元素非负 charges、每 set 总 charge≤cost 所给的 cover lower bound 支持 packing 解释，并非 BPE 定理。
- 来源校验：一处“at most”文字疑似方向笔误，与前文 lower-bound 说明不一致；本次论证使用公式及 weak duality，不依赖该句。
- 另一候选：单 word 长度 m 的 ceil(log2 m) survival 界，只能在剩余相关 edges 始终 eligible 等前提下使用；高 word frequency 情形的适用性有限，未声称中文语料收益。
- 成败状态：推广方向已推导并继续审查；没有 benchmark 或普适可用性结论。

## 03:42 研究进展

### 62｜定期源码检查：cursor 简化与新实验证据

- UTC 对照：检查开始：10-11 19:42:41；本轮检查完成：10-11 19:43:45。

- 检查开始：2026-10-12 03:42:41。
- 本轮检查完成：2026-10-12 03:43:45。
- 新 head：[a2b20f329d1b4a0d39b5721f4a7a678a80d0d9b2](https://github.com/Yikai-Liao/tokenizers/commit/a2b20f329d1b4a0d39b5721f4a7a678a80d0d9b2)，标题为“Record Positions cursor simplification checks”，parent 为 c1b6557a。
- ancestry：head 比 ea1d4d56 前进 3 个 commits，无分叉；d9170827 为证据、c1b6557a 为 cursor 简化、a2b20f32 为证据。
- 源码：Positions 先对齐 restart，再单次 cursor.skip_while；Cursor.end 已删除。index/merge/mod 的 blob 与 ea1d4d56 相同。阶段 49 的这两项候选已实现，后续不能重复计为待做收益。
- 新作者验证：每配置 29 项 targeted tests，明确排除已知 JSON-roundtrip 失败；5 个 pairs.json 哈希匹配。这是作者报告，不是本研究重新执行，也不是全套测试无例外通过。
- Cursor 作者测量：four-worker wall 变化为−4.31%至+4.51%，没有一致加速。
- Root/leaf 作者测量：English 单 worker ByteLevel 回归+8.92%至+11.62%，Whitespace+14.17%至+16.44%；Chinese retained memory 上升。用户已表示可接受该优化的回归，不能把结果简写成所有 case 都≤5%。
- 来源：[cursor 证据](https://github.com/Yikai-Liao/tokenizers/blob/a2b20f329d1b4a0d39b5721f4a7a678a80d0d9b2/experiments/bpe-simplification/evidence/positions-read-20261012/README.md)、[root/leaf 证据](https://github.com/Yikai-Liao/tokenizers/blob/a2b20f329d1b4a0d39b5721f4a7a678a80d0d9b2/experiments/bpe-simplification/evidence/positions-child-20261012/README.md)。
- 成败状态：最新代码与作者实验资料已纳入研究，当前基线为 a2b20f32；没有编译或复跑作者测量。

阶段 58 补充：截至 03:42，独立动态 floor lifecycle 模型报告初步 16,683 个 passing cases，正在加入 whole-threshold compatibility 补充。该数字是进行中的统计，未当成最终保存结果或完成声明。

### 63｜动态 floor 的实用增量证书

- UTC 对照：开始：10-11 19:43；10-11 19:43:21；结束（进行中；所列为目标时间）：10-11 19:46；10-11 19:47；10-11 19:49；10-11 22:58。

- 开始：2026-10-12 03:43（阶段材料分钟精度；此前 03:43:21 已确认启动）。
- 结束：初份设计 03:46 完成；03:47—03:49 补充 physical-edge potential。06:58 回查保存材料，关闭初始“进行中”状态。
- 工作：将 terminal-frequency 充分条件转成常数状态 N、S、M 摘要，比较少量 tail pivots 与精确 count histogram 的更新/查询成本。使用较早 ea1 审计副本，不冒充当时已读完整 a2 源码。
- 结论：常数状态 depletion 证书便宜但可能弱；histogram 需 logarithmic 更新和额外状态。floor 提高不自动释放 heap 中埋藏 Positions，queue filtering/重建和 count-table retirement 仍是独立扫描/生命周期成本。post-commit 丢弃已完成 birth 不等于省去 prepare append。
- 报告内支持检查：约 03:44 的 count-multiset 枚举 57,906 cases 核对 distribution-aware predicate 单调性与 streaming budget；仅报告约 1.22 秒 tool wall time，原精确起止和该次源哈希未在本次回查中取得，不能称为完整 BPE lifecycle 验证。
- 证据：阶段研究材料当前 SHA-256 `1d0f15df1cc6…`（含后续补充）。无仓库执行或测得收益；更后期证书/heap 改进分阶段续记。

## 03:45 研究进展

### 64｜不同的未完成词序列提供 horizon 下界

- UTC 对照：初始推导：10-11 19:44；10-11 19:45；10-11 19:51。

- 初始推导：2026-10-12 03:44—03:45；完整推导与源码核查 03:45—03:51。
- 结果：H 个不同、尚未完成的 snapshot token 序列，各自 weight≥L 且剩余边始终 eligible，在 fresh substitution 的可逆展开下保持不同；一次规则至多完成一种序列，因此保证接下来 H 次 greedy 频率均≥L。较小数值的首次激活 reserved ID 不破坏此结论。
- 更强容量证书：若存在不变 token decoder，N 种不同合格 word string 与 B 个已证明不可能表示这些 word 的现有 ID 满足 N≥Vtarget−B，则填满目标前始终有高频词未完成。这可以绕过 accepted-rule 预算；复用保持 decoder 时频率证书仍可能成立，但不能据此永久删除 reuse 中当前低频 pair。
- 动态更新：最坏情况每次同时减 Q 和 H 不会增强证书；需要在已有 prepare 边界访问中识别实际整词完成，或更新保守容量信息。只需候选统计的推导不等于已经证明其实现免费。
- 前提与反例：不能直接把原始文字唯一性当成过滤后 token 序列唯一性；alphabet 过滤和 affix 可能 alias。全词保留符号 span 严格小于 max length 可保证持续 eligible；仅当前边 eligible 不足。等于 floor 的 pair 保留，birth 需完整组上界，零权重词不能支持正下界。
- 源码依据：在`a2b20f32`只读核查 CorpusPlan、vocabulary 及 WordCounts；CorpusPlan 已有降序 word weight 排序与 retained-symbol 测量，可在现有 pass 提取初始 rank-floor，无需另排序或 Positions 重数。初始 count 和 append 先于全局 floor 聚合，因此没有宣称省去初始 Positions 追加。
- 证据：阶段研究材料；固定提交的 `tokenizers/tk-train/src/trainers/bpe/{corpus,vocabulary,word_counts}.rs`。
- 状态：理论与源码适配条件已写成阶段材料，独立完整生命周期模型进行中；没有仓库执行、真实覆盖率或性能结果。Chinese 大量 weight=1 可能限制有效 L。

### 65｜lazy birth Positions materialization

- UTC 对照：开始（截至记录，非精确起点）：10-11 19:45；结束（进行中；所列为目标时间）：10-11 19:47；10-11 22:58。

- 开始：截至 2026-10-12 03:45 已启动；精确起点未保存。
- 结束：2026-10-12 03:47 阶段报告完成；06:58 回查保存材料，关闭初始“进行中”状态。
- 工作：研究 adaptive postings materialization 能否省去未被使用的 birth 压缩流，并分析 sibling 重扫、raw 父 posting 不是 accepted starts、cold sibling 保留整族等反例。
- 结论：无限共享父列表过滤会出现Θ(n×siblings)准备工作。可证明的 compressed partition 与 one-shot family promotion 仍需新 accepted-anchor 编码、共享所有权/分区、临时双份流和延迟峰值；现有 eager newborn appends 已是每 accepted occurrence 至多两条。没有证明统一 fresh/reuse 的 lazy cohort 语义。
- 决定：不把该路线列为无条件 memory-tight/fast 优化；one-shot promotion 仅保留为有真实未使用 family 证据后可比较的研究控制。没有 Positions 重数计数方案，也没有把现有 child 无重编码 concat 再计作收益。
- 证据：阶段研究材料 SHA-256 `3043259c47d3…`。只读较早 ea1 实现副本及有明确版本限制的 corpus 来源，无仓库运行、实现或 benchmark。后续 FFBPE hot-window 审查另见阶段 111。

### 66｜marked-state β 数值界的支持性检查

- UTC 对照：本方向间隔：10-11 19:41:30；10-11 19:44:33；执行开始：10-11 19:45:19.605295。

- 本方向间隔：2026-10-12 03:41:30—03:44:33 未新增实质研究结果，等待独立审查；该间隔未算作新验证，其他研究方向继续。
- 执行开始：2026-10-12 03:45:19.605295。
- 执行结束：2026-10-12 03:45:24.163238；耗时 4.557812 秒。
- 有限结果：20,000 个 random draws 中 5,286 个满足 initial pair≤15 的语料，执行 21,468 个 AA/non-AA 步骤；所有 marked β≤15 且每个被使用端点均有 mark。unmarked β>15 共观察 32,425 次，这不是不同状态总数。
- 证据用途：支持“仅必要 marked 状态可能享有更细β界”的方向；toy 上限 15 不等于已证明 u64 安全，也不是独立审查完成。
- 可复验材料：`dag_used_beta_bound_check.py`与同名前缀`.first-run.txt`；运行时及当前源 SHA-256 为`59c1cb0d104d…`。
- 状态：本轮压缩表示材料整理完成，但 marked-state β明确保持未决；整体整夜研究继续，没有仓库执行或性能结论。

### 67｜完整词见证下界的独立模型

- UTC 对照：开始：10-11 19:46:22；最终执行：10-11 19:51:09.662521；10-11 19:51:10.645056；10-11 19:52:32.951902；10-11 19:52:49。

- 开始：2026-10-12 03:46:22。
- 最终执行：03:51:09.662521—03:51:10.645056（0.983 秒）；报告完成 03:52:32.951902，03:52:49 空格整理。
- 工作：验证局部 H 个未完成 token 序列证书与全局 N≥Vtarget 的固定 decoder 容量证书，比较完整 fresh greedy 规则/频率/输出 ID 历史、最终路径、每步保留 count 与 Positions；模拟永久退休和 lineage 事件省略。
- 成功：15,802 次完整剪枝/参考配置全部通过，覆盖 21,072 条规则、2,512 次 floor 提高（432 次后期）、1,404 个退休键、3,564 次省略 lineage 事件、8,199 次恰好等于 floor 的选择、6,895 次 reserved 零增长动作，保留 560 个零权重 birth 位置。每步“至多完成一种 snapshot 序列”与 decoder 不变性检查通过。
- 范围：12,150 次小规模枚举、3,600 次 seeded 随机配置及 52 次定向完整历史配置；这些包含不同证书模式，不是 15,802 个独立语料。全局正证书在小模型中较少出现。
- 反例：原始词经过过滤 alias、同一 rendering 的不同 tokenization 错误合并权重、仅当前边 eligible 的长度捷径均被否定。共享 BC 的高频词可在一次收缩后全部失去 eligible 边，下次频率降至 1。
- reuse 专项：一次保持 string decoder 的实际复用与三次完整 index 重建支持全局频率证书；没有跨 reuse 保留 fresh 永久剪枝。N≥Vtarget−B 的可排除 ID 细化尚未实现于该模型。
- 开发修正：一个 fixture 原期望 floor0，实际低权重 filler 合法证明 floor1；修正断言并保存原 failure JSON/log。未发现有效历史对比失败。
- 证据：阶段研究材料、`verify_word_witness_floor.py`、`word_witness_floor_results.json`、`word_witness_floor_run.log`。源 SHA-256：`12b01014b3b6…`；依赖 SHA-256：`3daa586d5864…`，旧模型未改动。
- 边界：标准库数学验证；全词去重、维护和统计成本未测，production decoder 与 span 需按源码限定。没有仓库执行、并发或完整错误路径证明、性能结论。

### 68｜按本地组归约的 atomic old-loss 对照

- UTC 对照：开始：10-11 19:49:42；10-11 19:58:01。

- 开始：2026-10-12 03:49:42；阶段结束 03:58:01。
- 结果：固定 count-map 结构、所有 frozen-witness prepare 已 join 后，每个本地组一次 fetch_sub 或 checked CAS 可实现有效 fresh loss。旧的 count-map shard 可只作为寻址分区，不必独占该 shard 的所有标量更新；old-only 记录需在并行 job 内丢弃，才真正省去对应串行 scatter。
- reuse：同一 AtomicU64 存储可在独占阶段通过 get_mut 运行原有顺序 signed 算术；不需要新增存储枚举，但仍需保留 fresh/reuse 调度差别。
- 关键限制：below-floor 首次 crossing 唯一，可只记录 R 个退休键；物理删除仍需路由、capacity 扫描或保留历史垃圾，最坏 R=D=M，瓶颈回归。count 删除并不立即释放 heap 中的 Positions。birth 分组、插入、Positions 及 publication 仍存在。
- 错误边界：raw fetch_sub 遇非法输入可能先 wrap 再发现，需要 join 后整 attempt 丢弃；checked CAS 可避免先写入，但重试/争用有成本。clamp-to-zero 再 fetch_sub 不安全，任意 atomic 顺序也不能保持 reuse 原有失败步骤。不能把“immutable map 结构”当成冻结的标量语义视图。
- 独立模型：03:54:45 完成 3,261 个 lookup/RMW/remove 配置、75,273 个 interleaving 状态、5,629 个终态；确认唯一 floor-crossing 与借用的退休节点安全条件。包含 count10、floor5、loss[4,7]与[7,4]错误顺序差异，以及 reuse signed overflow 反例。执行起点和耗时未单独记录。
- 证据：阶段研究材料、`check_model.py`与`results.json`；源 SHA-256：`21c9fa16b282…`。
- 文献对照：SCC/Papaya 仅作为未 benchmark 的清理对照，需计 bucket locking 与 guard reclamation；2025 GROUP BY 的 delete-free ticketing 前提不同。
- 边界：数学线性化模型和只读设计审查；不是 Rust/并发容器运行、内存模型证明、布局测量或硬件 benchmark。当前 index/merge 主体与 ea1d4d56 一致，未重复计算已有 child-Positions 收益。

### 69｜PR 2501 状态复查

- UTC 对照：检查时间：10-11 19:50:41。

- 检查时间：2026-10-12 03:50:41。
- 结果：仍为 open、未合并；本次检查未发现新事件（返回事件数组为空，timeline 数量 8）。无需重复发送无变化通知。

### 70｜静态词权重与“无完成动作”动态 rank 模型

- UTC 对照：开始：10-11 19:53:42；最终执行：10-11 19:57:30.634304；10-11 19:57:31.318173；10-11 19:59:06.924584。

- 开始：2026-10-12 03:53:42。
- 最终执行：03:57:30.634304—03:57:31.318173（0.684 秒）；报告完成 03:59:06.924584。
- 方案：固定初始 top min(H0,Q0)合格 word weights；rank=H0−已成功提交且没有完成任何 word 的规则数。每个规则跨 physical occurrences 做 completion OR，包含未选为见证的词和零权重词；无需按 word 身份删除权重。它等价于每遇完成规则就保守删去当前最大 weight。
- 成功：9,922 条完整剪枝/参考历史全部一致；14,592 次存活见证 quantile 审计通过。1,858 次 floor 提高中 752 次为后期提高，其中 585 次从初始 Q0<H0 后期变得可用；退休 759 个键、省略 2,066 个 lineage 事件，保留 369 个零权重 birth 位置及 6,258 次等于 floor 的选择。
- 批次验证：14,102 次 batch counter 检查，包括 426 个兼容多规则 batch；同一批中必须分别对每条 selected rule 做 OR，然后汇总 false 数量。AB10/CD5/EF1 反例显示，整个 batch 只做一个 OR 会把正确 floor1 误抬成 5。
- 更强直方图反例：在严格 span<3 下，ABCD:N 加 singleton B:N/C:N/UV:1，与 AB:N/BC:N/CD:N/UV:1，有相同 pair histogram、character histogram、word-weight multiset、加权和非加权符号总数及词数；BC 优先后，第二频率分别 1 与 N（模型 N=1000）。全词长度资格不同，不能用这些廉价统计直接推断终止频率。
- 开发修正：ABCD 并不会在所设两条规则后完成，定向 fixture 改成 ABC；原 failure 诊断保留。有效生命周期比较未发现反例。
- 证据：阶段研究材料、`verify_initial_word_summary.py`、`initial_word_summary_results.json`与`initial_word_summary_run.log`。源 SHA-256：`c4abce77f957…`；依赖 SHA-256：`3daa586d5864…`。
- 源码与成本边界：已有 prepare 可读 separator 与 weight；rank 只在成功闭合 commit 后更新。无需运行期词身份 map 或可变 histogram 的代数证明已成立，但初始化 distinctness、geometry 及实际实现成本仍需证明/测量。没有仓库执行、性能或内存模型结论。
- 文献核查：subword-nmt 原始实现 threshold 保留并恢复 big_stats，不能作为永久低频退休的正确性依据。

### 71｜动态下界阶段反馈与 rewrite 依赖前沿

- UTC 对照：阶段反馈时间：10-11 19:55:31.288209；第一轮研究：10-11 19:55；10-11 20:02；10-11 20:08；10-11 20:11。

- 阶段反馈时间：2026-10-12 03:55:31.288209。已说明 word/vocabulary 条件可提供动态 floor，大量 weight=1 词可能使证书很弱；继续通用 count 界及 prepare/commit 工作消除，未锁定最终方案。
- 第一轮研究：03:55—04:02；第二轮细化 04:08—04:11。
- 结果：string contractions 可交换，不代表独立 prepare 的 neighbor deltas 可直接相加；相邻 disjoint contractions 有状态相关 cross term，包含 scalar 与坐标。优先级、occurrence 支持、corpus 读写、count/payload 闭合及错误 publication 需分别证明。
- 候选：有限个完整词 region、局部 logical-prefix watermarks 和全局已认证 rule tickets，可允许某 region 先 prepare child，不必等待其他 region 完成 parent。初步限制为一个 open parent generation 加一个 identity-certified child；不是当前 Candidate/take 的直接替换。
- 成本边界：region 数量限制同时推进的 phase，不禁止 region 内原有 pair 并行；控制状态 K/R 不界定 payload 字节，必须另算 sparse descriptor/leaf/journal 与实际 byte 预算。单个巨大 region 可使严格 byte 上限阻塞进度。
- 闭合细化：分别记录 dispatch delivered、corpus executed、effects delivered 三种 watermark；seal 表示“今后不会再有消息”，不是当前 queue 空。一个已冻结兼容 batch 可把 closure 记账做到 O(K+实际 tasks)，独立 rolling tickets 仍可能 O(KR)；不要求 K×R 矩阵，但不能声称发现依赖免费。
- producer 界：fresh 出生 key 可由正确的完整 producer-rule 闭合界定 payload；同批 new/new 共享 cut 的 canonical 物理左 producer 拥有 birth。唯一 rule 仍可有任意多 jobs；payload 闭合不等于邻域读稳定。旧 pair loss 可能来自多个 producer。
- 有限检查：报告 04:01:54 执行独立脚本`rewrite-frontier-model.py`，2,880 个三 token alphabet、词长 0–6、非 AA 且支持 disjoint 的 rule/word 配置验证 contraction 交换及坐标 cross term。SHA-256：`1c476142f5d5…`；未单独归档精确进程起止/stdout。
- 证据：阶段研究材料，只读 ea1d4d56 相关源码与明确标识的旧 corpus 几何副本。没有仓库执行、regional 并发协议实现、rank admission/overflow 或 Rust 安全证明。

### 72｜重词的 substring diversity 下界

- UTC 对照：推导时间：10-11 19:58；10-11 19:59。

- 推导时间：2026-10-12 03:58—03:59。
- 结果：在当前 snapshot token alphabet 中，合格高权重词内 D_l 种不同长度 l 子串，要求至少 ceil(D_l/(l−1))条 fresh binary 规则才能全部完成；单条新规则至多增加 l−1 种跨边界子串。
- 前提与限制：如果按旧 base alphabet 展开，需扣除已有 token 覆盖的子串；不能直接与 word-count 下界相加，也不能把 D2 等同任意高 aggregate pair 数量。
- 证据：阶段研究材料 §13，引用 grammar、string attractor 及 substring complexity 主来源。此处为理论扩展；没有新增 substring index、语料扫描、实现或性能验证。

### 73｜weighted DAG marked-state 数值域独立审查

- UTC 对照：审查开始：10-11 19:52:35；10-11 19:55:44；成功：10-11 19:45。

- 审查开始：2026-10-12 03:52:35；结束 03:55:44。
- 成功：在 exact-language、fresh rewrite、初始全部 pair checked 计数成功等前提下，用初始 state origin 与 token expansion 不变量证明所需 marked-state β≤u64 上界；包含 AA、共享 prefix、重新汇合、内部接受节点、零权重和暂时 unreachable copies。03:45 支持性模型的“待独立证明”状态由此得到条件性解决。
- 限制：mark 按“存在长度≥2 的初始 prefix”定义，不能用最短深度；generic αβ node mass 仍可能溢出，任意压缩 DAG 的α也不受该β结论限制。显式有限输入的α/μ/parity 需要另行 size-bound 证明。
- 数值反例：ababab 权重(2^64−1)/3 的所有初始 pair 均合法，L+R 却会先溢出；应按 L+(R−z)等非负顺序求真实 loss。不能先分别扣 L/R 再补 z，也不能把合法 u64 loss 强塞为 negative i64。
- 初始短词 alias 可使 final-weight 汇总超过 u64 且无 pair 可检查；忽略这些词的β不等于全语言可用 u64。仍需保留原始输入检查和全部初始 pair 检查，不得用标记β替代。
- 独立算术执行：03:53:51.611993—03:53:51.612032。`weighted_dag_numeric_counterexamples.py`及`.result.txt`；SHA-256：`b5c46c75082c…`。只有 L+R 构造被脚本直接断言，其他构造为报告中的数学证明和打印见证。
- 证据：阶段研究材料。部分本地 corpus 副本与固定提交 blob 不同，报告明确标出；关键 checked-before-floor 顺序由固定 index 副本核对。没有重新执行早期 DAG 模型、仓库或 benchmark；未证明生产 first-error 等价。

### 74｜prepare 作为稀疏 batched join 的条件性方案

- UTC 对照：研究区间：10-11 19:49；10-11 19:54。

- 研究区间：2026-10-12 03:49—03:54。
- 方案：每条 selected pair 的 cursor 在正常扫描时筛出有效 matches；按位置合并 accepted cursors，并缓存其他 stream 最小值以连续处理赢家 run。tile 内部从相邻 accepted matches 推断共享 cut，仅 tile 两端查询 selected-pair map；保留 deferred apply starts 及 child Positions。
- 条件收益：原来约 2K 次 selected-neighbor probes 可降至约 2J 个 tile 端点 probes；在 O(M)解码/验证之外，merge 控制 O(K+R log(B+1))，R 是 accepted 同 stream runs。没有省去必要的加权累加、birth 写入或最终 pair-major 归组。
- 新成本：composite-key 分组、更多 fragments、seek、restart planner 及额外比较。D 个 restart descriptor 的串行 planner O(D log(B+1))；很多小 leaf 时 D 不等于 M/128。需要与 pair 内部 task 切分对照，不能仅因 P>B 就判断赢。
- 有限模型：报告 1,310,700 个边界 case 通过，覆盖四 token 长度≤7 词、ordinary 兼容规则子集、tile 边界、不同 span 及 stale 候选；另 20,000 个随机 cached-competitor merges 和 100,000 个 restart-seed rank 误差检查通过。重复坐标只验证 merge 排序，不证明重复 BPE 应用安全。原始执行时间与 stdout 未保存于现有归档，无法核验精确运行时刻；未为补日志重跑。
- 证据：阶段研究材料；`check.py` SHA-256 `fb660ad61f7e…`；`merge_check.py` SHA-256 `168a5570c90e…`。
- 范围：ea1d4d56 本地源码只读审查；后续 a2 只改 cursor 的结论来自另一次审计。AA/reserved/reuse 留在原有特殊路径。没有仓库运行、硬件性能或错误路径验证。

### 75｜祖先质量预算下界与独立模型

- UTC 对照：理论探索开始（截至记录，非精确起点）：10-11 19:59；10-11 20:01:59；最终执行：10-11 20:05:44.412532；10-11 20:05:46.621853；10-11 20:07:59.593324。

- 理论探索开始：截至 2026-10-12 03:59 已开始，精确时间待补。独立模型开始 04:01:59。
- 最终执行：04:05:44.412532—04:05:46.621853（2.209 秒）；报告完成 04:07:59.593324。
- 不变量：fresh 收缩只删除原始物理 cut，每个当前 pair 具有唯一 snapshot family。j 步实际删除质量≤top-j family 质量；未被选择的 family 保留其全部 chosen 质量，其 descendant key 数量≤(1+x_a)(1+y_b)。这里约束物理删除，不能与旧键 label 变化的 loss 混为一谈。
- 已检验三种证书：剩余总质量与 key 容量比较、threshold/role cover、整数 packing。H 次 horizon 用 r=H−1，L=1 单独处理；整数 split cost 不用浮点 sqrt，宽整数聚合可能超过 u64。
- 成功：2,058 条物理边界 trace、8,597 个 state、9,426 个成立证书的精确 horizon 检查通过；另 8,800 条完整剪枝/参考历史一致，含 11,892 条规则、885 次后期 floor 提高、864 个退休键、1,079 个省略 lineage 事件、244 个保留零权重 birth 位置。
- 重要限制：当前 eligible 不等于未来持续 eligible。ABC100/BC1/UV1、严格 span<3 反例使三种未经资格筛选的公式均误判 L2,H2，真实下一频率 1。模型关闭这种使用，或仅纳入全词 span<cap 的 qualified matrix；该矩阵的生产获取成本尚未建立。
- 其他负例：AA raw frequency 不等于实际删除质量（AAAA:10 首先 raw30 而只删除 20）；同一 family 可被多步选择；错误用 r=H−2 会产生假证书。三类负控制均检出，无意外正例失败。
- 与冻结 3M 比较：在非平凡 ungated 配置中，三种 ancestry 界均找到 3M 未证实的情形，也均存在 3M 可证而自己不能证实的情形；两类证书互补，不能声称普遍支配。
- 证据：阶段研究材料、`verify_ancestry_floor.py`、`ancestry_floor_results.json`与`ancestry_floor_run.log`；源 SHA-256：`31d1228d4252…`，依赖 SHA-256：`3daa586d5864…`。
- 边界：独立 interval/cut 模型与有限完整历史测试；没有仓库执行、额外 Positions 生产重数、实现成本或性能结论。

### 76｜分支定期复查

- UTC 对照：检查时间：10-11 19:59:58。

- 检查时间：2026-10-12 03:59:58。
- 结果：`simplify/bpe-flat-20261010`保持`a2b20f329d1b4a0d39b5721f4a7a678a80d0d9b2`，没有新提交。
- 来源：GitHub branch API 的本次只读结果。

### 77｜当前性能证据身份校正与 heap 清理条件

- UTC 对照：校正时间：10-11 20:01。

- 校正时间：2026-10-12 04:01。
- 重要发现：Chinese `merge-current-perf` 的 job.json 与 flat.txt.gz 标识实际为较早`child-direct-varint`实验臂，即使 README 使用 final 措辞，也不能把 prepare32.82%、owner10.92%、append9.28%、write7.82%当作当前最终代码 profile。
- 可确认对照：English `merge-final32-en-perf`标识为`child-inline-fast`、单 worker；报告 prepare20.39%、owner14.32%、heap max_sibling8.48%、commit sort4.85%、Cursor4.22%、hash insert3.51%、push_restart1.73%、write1.28%。这些为作者 profile 数据，尚无 child-read 最终版本的细粒度 profile。
- 状态：更完整证据审计材料待交付；不从过时的 Chinese 百分比推断当前瓶颈。
- 新审查：fresh heap 候选唯一条件下，以 D=queue 长度−active count 记录数计 dead 候选；dead≥一半时重建是否能按退休 candidate 摊销为线性正在审查。计数单位是 candidate，不是 Positions 字节；尚未实现。

### 78｜frozen-witness gate 的最小源码接口审查

- UTC 对照：开始：10-11 20:03:48；10-11 20:09:50。

- 开始：2026-10-12 04:03:48；阶段结束 04:09:50。
- 工作：在最新 merge/index 及 provenance 源码中核验逐 group 缓存与 finish-group gate 的取舍、旧 witness 精确重建、接口改动与布局/工作估计。
- 源码结果：born-only Change.removed 当前是 synthetic pair，不能直接查 membership。把原 old_neighbor 传入 group 构造，可在现有零 loss 字段保存真实 witness，实现 finish gate 而无需新增每组存储。early gate 只需 bool 与现有 created 信号，因为 fresh born-only group 不会变成 old-role。
- 证据：阶段研究材料，以 ea1d4d56 的相关未变源码为依据。Change72/local wrapper80 的布局估计、borrow 接口、错误行为保持及性能 break-even 均未编译/测量；没有仓库执行或修改。
- 独立模型执行：04:07:23.459081—04:07:23.947648。5,452 个有效 batch、13,133 个 boundary 事件通过，含 2,131 个 AA 配置、4,250 个较小数值 fresh output 配置及 200 个保留零质量 fragment 检查。early bool cache 与 finish-group 真实 witness gate 均匹配完整 count/Positions。
- 反例：直接对当前 synthetic removed Pair 做 finish membership，在 406 个配置中错误；必须恢复真实旧 witness。
- 证据：`check_model.py`及`results.json`，SHA-256：`e6b28d519b09…`。整数/list 模型，不是 Rust 布局、分配或性能验证。

### 79｜历史高 floor 同模型证据与安全试探候选

- UTC 对照：证据发现（截至记录，非精确起点）：10-11 20:04:24；10-11 20:05；10-11 20:12；10-11 20:12:13。

- 证据发现：2026-10-12 04:04:24；试探 floor 推导于 04:05 开始，截至 04:12 已有阶段报告。独立更深模型审查 04:12:13 开始。
- 历史校正：`core-runs.json`中 Chinese Whitespace min_frequency2/3 的 20 个报告运行均 actual_merges32,241、actual_vocab50,000，模型 SHA 为`6ba955cc1c9c…`，与最新 child-read 模型一致。此前“min3 会改变此模型”的假设不成立。没有 min4 测量。
- 含义：支持所报告配置的终止频率≥3；不是任意语料的事前证书，也不证明 3 为最佳 threshold。
- 理论结果：固定更高 floor 的 fresh attempt 始终选取基础 floor 参考的精确 prefix；若达到原目标，其整个模型相同。underfill 或 active reuse 时必须丢弃该 attempt，从原 WordCounts 以基础 floor 回退；不能直接降 floor 继续已剪枝的状态。最多一次试探，回退不再试探。
- 早停候选：按唯一 ancestor family 质量计算未来可选择高频规则的上界；AA 每次实际删除至少 ceil(L/2)，非 AA 为所选频率。若剩余 family 容量不足剩余词表槽，可提前放弃试探；尚需完整模式与成本核验。
- 错误/成本边界：保留初始 checked counts 和输入限制；跳过 Positions 分配可能改变资源失败行为，target-fill 证书不能检测从未执行的 baseline 失败。试探错误回退不能自动修复这一差异。progress 输出也需明确 attempt 语义。失败代价 T_trial+T_base，不宣称≤2 倍 wall；drop 后 RSS 未必立即下降。
- 作者支持模型：10,000 个 seed382921 随机 weighted corpus，四符号、词长 2–12、权重 0–7、trial floor2–12，检查 family 同源、AA-aware 规则容量、higher-floor 精确 prefix 与 success/fallback 结果均通过。`check_speculative_floor.py` SHA-256 `78f587b9766c…`，结果`check_speculative_floor_results.json`。没有保存精确执行起止，不能据文件时间补造。
- 证据：阶段研究材料。支持模型不含 batch/reserved/reuse/affix/length/error/实际 prepare 剪枝；独立审查正在扩展。没有仓库实现或新 benchmark。

### 80｜PR 2501 再次复查

- UTC 对照：检查时间：10-11 20:12:18.043。

- 检查时间：2026-10-12 04:12:18.043。
- 结果：open、未合并，三个 assignment/request 相关数组为空；完整 timeline 仍 8 个事件，没有新 assignment/request/removal。未重复发送无变化通知。

### 81｜边界守恒代数与 global birth-to-loss projection

- UTC 对照：开始：10-11 20:12:57；10-11 20:25:17。

- 开始：2026-10-12 04:12:57；阶段结束 04:25:17。
- 结果：精确 old-key loss 为 L(p)=R(p)+Σ[witness(q)=p]B(q)。local B/R 记录省去 admitted boundary 的一次重复加法，某些情形省目录访问；finish 可恢复现有 Change 形状。reuse 在有效输入上恢复相同 group、坐标和 signed action 顺序。
- 新细化：利用已有 global birth-key aggregation 后的完整 B(q)再投影 old loss，R 照常路由。必须在 child floor 丢弃前投影，包括最终低频 child。它增加第二个 count-update/join，不能在中间 frontier 选择候选或做 witness gate，也不能直接统一 reuse 调度。
- 工作对照：需要比较 R 记录数+完整 child 数与原 old-group 数。高 fragmentation fixture 为 64→2，低 fragmentation fan-out16 却 1→16；已全局 coalesce 到 D 个 old key 的替代基线需与 D 比较。新增 scalar buffers、owner skew、第二 barrier 及 candidate 保留时间均计成本。
- 初轮模型：04:20:47.347819—04:20:49.008532，5,448batch/13,181events；当时脚本 SHA `cde6ce51ba87…`。
- fixture 修正与最终执行：源码 task 按 128-position restart 切大 leaf，原抽象(ABC)^128/chunk4 不对应此约束；改成可达(ABC)^4096/chunk128。04:24:25.076586—04:24:26.615411 最终通过 5,448 个 batch、21,117 个 boundary 事件；64→2 结论保留。346,201 个 ordered reuse 序列、26 个 overflow partition、1,643 个 successful selection-range 检查保持通过。
- 源码界：当前 selector 保证成功 fresh batch 的 output 是 selection-order 连续 range（reserved 为独立 singleton），可暂时解码 witness，无需 persistent inverse map；不是“所有 fresh ID 数值都更大”的假命题。
- 错误边界：B+R 需在 apply 前 checked 验证，和原每次 record 加法相比可能改变 malformed 首个错误顺序；有效 record 等价不自动证明完整故障语义等价。
- 04:57 维护性补充：让 Change.removed_weight 在 build 时表示 rejected mass、finish 后表示 total removal 虽数学可行，却增加阶段依赖含义，不符合清晰简化的直接目标。采用命名明确的 private accumulator/conversion 或稳定 B/R 字段会增加接口/验证成本；需与省加法/查表一起衡量，尚无实现决定。
- 证据：阶段研究材料、`check_model.py`及`results.json`；最终 SHA-256 `d47ce2f126cb…`取代初轮当前文件哈希。只读核查固定 corpus/vocabulary/merge/index；没有仓库执行、Rust 布局或性能测量。

### 82｜阈值工作集、压缩与图收缩主来源扩展

- UTC 对照：时间：10-11 20:13；10-11 20:15。

- 时间：2026-10-12 04:13—04:15。
- 工作：进一步对照 subword-nmt learn_bpe.py 的动态工作集 pruning 与保留 big_stats 机制；检索 2017 Space-Efficient Re-Pair 的 high/low 阶段、ICS2025 并行 Contraction Hierarchies 及 SEA2026 parallel BlockTree。
- 结论边界：工作集缓存恢复与 destructive floor/restart 语义不同；图/语法算法只作为结构与成本类比线索，未转用其 benchmark 为 BPE 收益。
- 来源：[Space-Efficient Re-Pair](https://www2.compute.dtu.dk/~phbi/files/publications/2017serpcC.pdf)、[parallel CH](https://doi.org/10.1145/3721145.3725744)。BlockTree 具体适用结果仍待核读。

### 83｜initial count-first admission 与已有 materialize 融合

- UTC 对照：开始（截至记录，非精确起点）：10-11 20:14；10-11 20:18:46。

- 开始：2026-10-12 04:14；截至 04:18:46 已完成阶段材料，精确结束秒未记录。
- 结果：先用 count-only 初始 pass 算全部精确 count，再把合格 Positions emission 融合进已有 materialize pass，可避免被拒绝 pair 的初始 compressed append，仍为现有两次完整 symbol traversal，不增加第三次。
- 两种接口：丢掉本地 count 目录后，在 materialize 做 global admission+local position 查找；或保留 range 目录到 admission，按 local distinct key 转换为 admitted Positions 目录后 drop scalar scratch，每条 edge 只查该本地目录。
- 新成本：每条 edge 增加 admission/目录查找，O(local distinct)转换与检查，scalar 和 Positions 两轮 fragment routing；resident slots 与尚未归并的 admitted fragments 共存，峰值内存未必更低。
- dense 限制：token_count≤256 时，global admission bitmap≤8KiB，count-only 平面≤512KiB/active range；Positions 平面、seen-key 与 scratch 另计，不能把完整结构称为 512KiB。
- 正确性条件：初始全部 checked count 先于 floor，目标已满/零合并也保留输入检查；使用原 ids.symbols 及 range 顺序，保留合格 key 的零权重位置，冻结 global membership，reuse 不能直接套 fresh 筛除。materialize 由 infallible 变成含 Positions 写入的 fallible 路径需重新处理错误语义。
- 证据：阶段研究材料，只读 a2 CorpusPlan/vocabulary 与未变的 ea1 index。CorpusPlan 已有 word 排序，symbol_count complete-alphabet 捷径不是免费 pair-ID 扫描，alphabet 频率也不是可直接复用的 endpoint 质量证书。
- 状态：具体工作与成本推导，无仓库实现、验证或 benchmark。全局 birth projection 用于 loss reduction 的另一路推导仍在验证。

### 84｜试探 floor 的成本与共用实现审查

- UTC 对照：时间（截至记录，非精确起点）：10-11 20:12；10-11 20:16；10-11 20:18:46。

- 时间：2026-10-12 04:12—04:16（近似阶段区间），截至 04:18:46 材料完成。
- 证据复核：独立读取 a2 固定提交的历史 runs/manifest、当前 read pairs 与 child environment。20 个 min2/min3 运行来自同一 250,116,097-byte prepared input，SHA `ed057a8fdb27…`；相同目标、无 affix/cap、4workers、affinity0–3，均得到相同模型。
- 历史统计：排除 warmup 后，enum min2/3 训练 median22.0151/21.3360 秒，box23.6086/22.2769 秒，描述性差−3.08%/−5.64%。它们不是当前代码试探或 gate 对照，不能推断当前 speedup；同一确定性语料的重复成功不能作为部署成功概率样本。
- 当前作者 phase 记录：Chinese Whitespace train21.7140 秒，index11.5807、merge6.4553、drop2.0206；Chinese ByteLevel train23.7437、merge16.1832。Whitespace 的 index 重，失败试探再建 index 可能吃掉 merge 收益；这些为现有记录，不是新实验。
- 成本决定：应在同一实际 suite 比较成功节省总量与失败 attempt/teardown/fallback 开销，含失败样本。时间预算只在 joined checkpoint 检查，不能保证跨不可中断 index、batch、drop 的硬截止。
- 新源码事实：CorpusPlan 已用 retained symbol 数算 u128 weighted edge mass 后丢弃；暴露该 scalar 可在 index 前用 floor(E/ceil(L/2))<剩余槽数判定试探不可能填满。它包含 low/ineligible 边，界很弱；判失败仍需回退执行基础输入检查。
- 实现方向：proof-based 与 speculative floor 共用一个 numeric-floor kernel，外层只区分需 target-fill 的 attempt 与基础合法 underfill；不必在每 occurrence 判断证明来源。失败 trial 在最终 model/string publication 前 drop；不把任意 invariant 错误静默吞成正常重试。
- 证据：阶段研究材料。资源失败/日志观察语义仍需明确；没有新实现、benchmark 或通用 L=3 默认建议。

### 85｜分支复查与读取路径恢复

- UTC 对照：检查时间：10-11 20:18:46。

- 检查时间：2026-10-12 04:18:46。
- 结果：GitHub compare 确认 branch 与`a2b20f329d1b4a0d39b5721f4a7a678a80d0d9b2`相同，ahead0/behind0。
- 读取情况：此前 generic fetch 的 API branch 路径返回 400，web API 读取也未成功；支持的 compare 读取随后成功，没有持续访问阻碍。未发送无变化通知。

### 86｜试探 floor 与完整回退的独立挑战模型

- UTC 对照：开始：10-11 20:12:13；最终执行：10-11 20:18:22.843975；10-11 20:18:23.432043；10-11 20:20:08.378625。

- 开始：2026-10-12 04:12:13。
- 最终执行：04:18:22.843975—04:18:23.432043（0.588 秒）；报告完成 04:20:08.378625。
- 模型：独立 attempt/restart 状态机、真实保留 index 与 lineage 省略、reserved/reused identities、单调 floor、strict length、zero weights、ties 及错误控制；fallback 完全从原始输入重启，保留用户原 min_frequency 配置。
- 成功：13,669 个 wrapper/reference 配置完整结果一致。6,012 次接受 trial 中 1,659 次实际执行规则，4,353 次原目标已满足；后者是平凡 target 检查，不能计为剪枝收益。7,657 次完整回退；所有 4,209 次 early-abort 都由另一次不早停 trial 确认确实无法填满。
- 容量检查：1,670 条 family trace、2,457 条规则、168 个 mid-run snapshot 验证每 family 未来动作≤1+floor((c−L)/ceil(L/2))（c<L 为 0），总上界不足剩余槽即不能填满。AA 实际删除单独统计；这是动作上界，单调 length admission 丢失只会使它更保守，不需要下界证书的 persistent eligibility 前提。
- 8 类负控制：包括错误 drop task-local fragment、drop 整个低权重 word、跨 active reuse 继续 fresh pruning，均可“填满目标但模型错误”；target-fill 不能代替这些前提。跳过 Positions 失败的模拟也验证数学模型相同不等于避免的 fallible storage 行为相同。
- 其他统计：trial 退休 16,415 个 key，省略 1,916 个 lineage 事件，保留 120 个零权重 birth 位置；3 次 reuse abort、2 次 trial error 有完整回退。无意外有效配置失败。
- 证据：阶段研究材料、`verify_speculative_floor.py`、`speculative_floor_results.json`、`speculative_floor_run.log`；SHA-256 `c6181db730a7…`；依赖 SHA-256 `89370b127996…`。
- 边界：独立 Python 数学模型；recount/no-abort 重跑只作 oracle，不是生产方案。没有仓库执行、Rust 错误/并发或性能结论。

### 87｜barrier fragment 与 grammar floor 独立验证

- UTC 对照：开始：10-11 20:22:08；最终执行：10-11 20:25:56.138216；10-11 20:25:58.362892；10-11 20:28:10.304901。

- 开始：2026-10-12 04:22:08。
- 最终执行：04:25:56.138216—04:25:58.362892（2.225 秒）；报告完成 04:28:10.304901。
- 成功：13,937 个 snapshot（426 个 mid-run）及 5,403 条完整剪枝/参考历史通过。4,081 条非空历史共 7,816 条规则；1,439 次 floor 提高含 216 次后期提高，386 次仅靠 grammar 界而 Q 不足；省略 2,787 个 lineage 事件，退休 1,802 个 key，保留 307 个零权重 birth 位置。
- 不变量审计：59,686 次 protected-cut、37,620 次 class synchronization、32,008 次完成 grammar 容量检查。覆盖 5,182 条高频 AA 规则、1,526 次 reserved first-activation 动作，以及 2,054 条 source-style 初始 admission 例外历史。
- 负控制：任意重叠 substring 冒充不重叠 fragment、raw 过滤 alias、affix 不同 ID 却合并 weight、仅当前 eligible、active reuse、错误 Q_short、partial count 假 barrier 等均检出。k4 构造确实强于 k2；没有意外有效配置反例。
- 语义细分：原始 pair 先计数、不受后续 strict newborn gate 的例外与 uniform strict 模型分开；不能把初始例外 pair 错误视为低频 missing barrier。合格 fragment 仍保守要求完整 span<cap。
- 证据：阶段研究材料、`verify_barrier_fragments.py`、`barrier_fragment_results.json`及`barrier_fragment_run.log`；SHA-256 `69d6b3696cd2…`，依赖 SHA-256 `89370b127996…`。
- 边界：独立有限数学验证，未执行仓库、Rust 并发、分配/错误兼容性或性能测量。动态 exact/hashed D2 模型 04:26:00 另开，尚在进行，保留本次文件与哈希。

### 88｜低频边界保护的重复 fragment 证书与 grammar 细化

- UTC 对照：阶段整理：10-11 20:22；10-11 20:24；04:24—04:26 细化：10-11 20:24；10-11 20:26。

- 阶段整理：2026-10-12 04:22—04:24；更早推导精确起点未记录。
- 条件证明：用完整 snapshot pair count<L 的 cut 与词端切出不重叠 maximal fragments；≥L 的 fresh 规则不能跨这些 cut。精确相同 token 序列的 fragment 累加 weight 后，可像 weighted word 同步收缩，包括 AA；Q 种合格、持续 eligible fragment 保证 Q 条高频规则。
- grammar 细化：合格 fragment 内 D_k 种不同 k-gram 与长度 2..k−1 的 target class 数 Q_short(k)，给出 fresh production 下界 Q_short(k)+ceil(D_k/(k−1))，与 Q 取 max。只计真实 target 短 class，不能把所有短 substring 当成额外必需 root。
- unit-weight 示例：三个唯一上下文包围的 abac，L3 时 whole-word 证书为 0，fragment Q1，D2=3 可证明 3 条高频规则。19-symbol 二进制 core 有 D4=16，k4 界 6 强于 D2 界 4 和 depth 界 5。都是构造示例，无 Chinese 真实分布结论。
- 作者模型：12,000 个 seed12387231 配置通过；5,336 个非零证书、3,336 个 grammar 强于 Q。`check_barrier_fragment_floor.py` SHA-256 `d6ad72337d18…`，结果同名`_results.json`。
- 独立支持模型：5,516 个 case、9,500 个正 grammar 界、3,260 个高频 AA 步骤无违反；`check.py` SHA-256 `970109a50850…`及`results.json`。两份模型未归档精确起止；没有补造时间或重跑。richer lifecycle 模型另在进行。
- census 成本：完整 pair counts 之后才能找 barrier；可融合既有 materialize，但增加每 edge 查表、精确 fragment 比较/hash 与 dictionary，且到达太晚，不能反向省掉该轮此前初始 append。plain/完整 alphabet/无 affix 才可借用原 UTF-8 slice 作为精确键；一般过滤/alias 情形需精确 token 序列。
- 04:24—04:26 细化：动态 unseen D2 义务得到独立理论支持。可用 hashed occupied bitset 替代 exact bigram set；每条 selected rule 都清其 snapshot join 对应 bit，碰撞只削弱证书，开始跟踪后绝不重新插入。fragment weight 资格仍必须精确，不能用哈希桶错误合并不同 fragment。
- 新假设：有界 exact-key census cache，eviction 只导致漏计，能否控制 census 内存仍在审查。
- 来源：阶段研究材料；[LATIN2020 substring complexity](https://users.dcc.uchile.cl/~gnavarro/ps/latin20.pdf)提供δ与 grammar 背景，不是上述 BPE threshold 定理来源。没有生产实现、额外语料读取或性能结果。

### 89｜已有数据对新证书与 gate 的规模约束

- UTC 对照：审计补充时间：10-11 20:27；10-11 20:28。

- 审计补充时间：2026-10-12 04:27—04:28。
- 历史 min2→3 记录中，complete birth 非空调用数减少 40.61%，相关 Positions 数量减少 9.43%；这是历史 threshold 变化，不能称为当前 frozen-witness gate 命中率或其速度证据。
- 初始 D2 上限：当前 Chinese ByteLevel 已接纳初始 pair 只有 15,049，因此即使全是合格 fragment bigram，也不足证明约 50k 剩余 growth 的初始 horizon。后期 activation 或 k≥3 为另一路可能性，不能混入这个初始界。
- Chinese Whitespace 已接纳初始 pair 为 1,721,017、剩余 growth32,241，数量上有空间，但未测 fragment 资格或真实证书大小。
- 动态 bitset 证明细化：在严格有效 lifecycle 下，首次选到<L 的规则前，所有 witness fragments 已完成，bits 已清空，故可能无需每次显式 selected-count≥L 分支。active reuse 停止、开始后不 reinsertion 等条件仍不可取消；独立模型继续核验。

### 90｜完整 birth projection 与 atomic count-map 组合审查

- UTC 对照：开始：10-11 20:32:23；10-11 20:38:10。

- 开始：2026-10-12 04:32:23；阶段结束 04:38:10。
- 结果：complete B(q)归约后可直接 atomic 扣旧 witness，fresh joined batch 中省单独 scalar-loss 路由/更新阶段；却未必省总 phase。共享 count-map 读取未 join 前不能结构插入/删除，结构 publication 仍需另一个并行 join，或改为串行 map 工作。
- 对照：本地 group atomics 可在 apply 内完成 loss，随后复用原 mutable birth-owner phase；global projection 只在完整 child 归并减少的 atomic 数量大于失去 old-parent 共享时有机会赢，并承担更晚闭合。不是“无 owner commit”。
- 模型执行：04:35:49.735488—04:35:49.842243。15,410 个有效 parent、61,958 个任意 R/B 串行化、27,826 个 R-before-B 串行化、17,504 个正但低于 floor 的 child 贡献均保持正确 residual 和唯一 floor crossing。
- 证据：阶段研究材料、`check_model.py`与`results.json`；SHA-256 `2a706150a639…`。
- 边界：只读固定源码与 Rust/Rayon 主 API；整数 linearization 验证不等于硬件 memory model、Rust 执行、布局或速度测试，reuse 仍保留顺序路径。

### 91｜动态 exact/hashed D2 义务完整生命周期验证

- UTC 对照：开始：10-11 20:26:00；10-11 20:32:12.829452；10-11 20:32:14.733768；10-11 20:33:50.286371。

- 开始：2026-10-12 04:26:00；最终执行 04:32:12.829452—04:32:14.733768（1.904 秒）；报告完成 04:33:50.286371。
- 成功：16,152 个完整剪枝/参考 setting 一致（12,531 个非空、29,273 条规则，含 6,822 条 AA）；2,311 次证书激活，92 次由初始不足后期激活。13,119 个 hash setting 中 1,801 次激活，含 3,027 个 constant-hash 反向强度控制、2,172 个 mid-run setting、4,989 个初始 admission 例外。
- 不变量：36,342 次检查 live bits≤尚未引入的真实 snapshot bigram 义务；143 次不相关碰撞清 bit 均只削弱证书。7,037 次 bounded exact fragment cache 构造检查，证明遗漏/eviction 只漏计的限定用法。
- 关键条件：所有 partial bitset 构造、OR 与 clear 使用同一 mapping/seed；每条 committed fresh rule 都清其(last(left),first(right)) snapshot join，包括未命中见证 fragment 的动作；开始 tracking 后绝不 reinsertion；mid-run 原有 ID 全部作为该 snapshot 原子重新初始化端点。
- 负例：错 seed/clear、reinsertion、错误 endpoint、漏清 batch 内规则、用 hash 碰撞合并 fragment 质量、忽略 reserved 预算和 reuse 均被检出。一个 late-activation fixture 期望修正，原 failure 记录保留。
- 证据：阶段研究材料、`verify_dynamic_bigram_bits.py`、`dynamic_bigram_bits_results.json`及 run log；SHA-256 `115e405685e7…`，依赖 SHA-256 `69d6b3696cd2…`。
- 边界：逐 committed-rule 数学模型，不验证 atomic OR 或 Rust 并发。bitset 内存之外仍需 snapshot first/last metadata 与 exact census；生成 setting 不是生产收益率，未 benchmark。

### 92｜hashed 义务强度的合成实验及解释修正

- UTC 对照：材料时间：10-11 20:31；10-11 20:33。

- 材料时间：2026-10-12 04:31；04:33 发现并校正摘要歧义。
- 结果：任意 hash 的证书安全性是确定性的；独立均匀 hash 只用来研究强度，非正确概率。constant hash 可能完全失效。推导 occupancy 期望/方差并做 20 个 case×400 次及 8 个动态 case×200 次的合成模型，不含实际 fragment census 或训练时间。
- 校正：先前“初始 100%激活、500 次 clear 后 15%”比较固定 H 的两个独立静态 case，不能解读为合法轨迹中证书消失。真实每条规则 H 减 1、live bits 至多减 1，所以已成立证书不会失效，可在激活后释放 tracker。
- 强度限制：无 witness slack 时碰撞可使 O(H) bitset 很难激活；较大空间可减延迟但无普遍保证。摘要与模型 metadata 已明确静态/动态区分。
- 证据：阶段研究材料、`model.py`、`results.json`；当前源 SHA-256 `5dfa0a45d7ed…`。精确执行起止未归档；没有重新运行或生产分布/速度结论。

### 93｜fresh underfill 后从 live corpus 重建 index 的恢复候选

- UTC 对照：理论修正：10-11 20:33；10-11 20:33:58；独立模型执行：10-11 20:38:53.943232；10-11 20:38:54.183293；10-11 20:39:52.294194。

- 理论修正：2026-10-12 04:33；独立模型开始 04:33:58。
- 结论：精确 prefix 可保留当前 corpus/vocabulary/merges，在基础 floor 完整重建缺失 count 与 Positions 后继续。原 WordCounts 全重启是现有架构最简单路线，不是数学必需；直接降 floor 使用不完整 index 仍错误。
- 新条件：只在 fully committed fresh underfill 恢复。live token 遍历须按 span 跳洞、不双计 endpoint，保留固定词 separator、weight_region 与零权重 occurrence；O(P)粗 whole-word checkpoint 可用于并行划分，仍需独立核验。
- admission 细节：初始 span1 token pair 的计数例外与后续 strict newborn gate 必须区分，不能统一全放行或全套 cap。未来 active reuse 仍走原输入 reuse 重启，live fresh index 不能补造 reuse 历史 cohort。
- 成本：先 drop 旧 index，再重建避免双 payload，但临时计数与 resident corpus 并存，峰值可比原初始化更高。新扫描/builder 及复杂度可能吃掉避免 prefix replay 的收益，尚无实现与测量。
- 独立模型执行：04:38:53.943232—04:38:54.183293（0.240 秒）；报告完成 04:39:52.294194。1,309 个完整参考配置一致；886 次 live rebuild，424 次在非空 prefix 后，重建后执行 1,016 条 suffix 规则。423 次无需重建 target-fill 中 185 次初始已满、238 次非空成功。
- 物理扫描验证：3,544 个 whole-word partition 扫描与独立 token-path oracle 一致，重建保留 248 个零权重位置。scanner 仅用 slot plane、span、压缩 weight regions 及 coarse checkpoints，不使用旧 Positions 或全 word-start 目录。span 跳跃避开 stale interior 与内部 tombstones。
- admission 证明：初始 span 均 1 且 fresh 输出 span≥2 时，当前 edge 合法 iff 两端 span1 或 combined span<cap。strict-only、admit-all、任意 stale-slot 起点、原始 composite-span 输入都有反例；不能把这一结论扩展到任意初始 geometry。
- 证据：阶段研究材料恢复 addendum；阶段研究材料、`verify_live_reindex.py`与 results/run log，SHA-256 `b9de7872c688…`，依赖 SHA-256 `89370b127996…`。
- 范围（当时记录）：只验证 fully committed fresh underfill，原材料曾将 reuse/partial errors 合写为重启；08:12 已明确更正（阶段 165）：只有 explicit active reuse 原输入重启，真实 partial errors 丢弃并传播。没有仓库执行、Rust 安全、并行或资源错误/性能结论。

### 94｜分支稳定与 alphabet 统计可复用性核查

- UTC 对照：分支检查：10-11 20:41:35。

- 分支检查：2026-10-12 04:41:35，GitHub compare 确认 head 仍为`a2b20f329d1b4a0d39b5721f4a7a678a80d0d9b2`，内容相同。
- 源码核查：select_alphabet 第 335–367 行显示默认 unlimited 模式只算 presence，不算 character frequencies；token-mass 初始化不能免费复用不存在的 alphabet 频率。
- 状态：没有新提交、无需重复通知；live-reindex 及 atomic/global-projection 均已作为阶段材料归档，仍保留各自实现/性能缺口。

### 95｜birth-fragment 有序分组与 giant-key concat 审查

- UTC 对照：开始：10-11 20:42:12；10-11 20:53:09；模型执行：10-11 20:50:23.717815；10-11 20:50:26.919053。

- 开始：2026-10-12 04:42:12；阶段结束 04:53:09。
- 源码结果：每个 fresh birth key 只有一个 canonical emitting selected rule，包括 new/new cut；可在临时 producer region 分组，无需 persistent sealed rows。成功 fresh 语义只要求同 key 稳定空间次序，跨 key publication 可用 stable semisort；reuse 仍需原 ordered path。
- 反驳免费改进：task-local changes 为 first-touch/left-before-right 顺序，并非 key-sorted runs；merge tree 不能免去分组。当前 concat 仅接受 leaf 且 one-fragment 不分配 child array，不宜加入 recursive concat 或每 tiny group 一个 Vec。
- 更紧 task 界：ordinary cap 前每 fresh birth key fragment≤P，AA≤4P；old-key 记录≤2T，ordinary T≤P+B−1。高 weighted frequency 不等于 giant descriptor group；需要 many-key owner 分布、allocation、serial routing 与 wall imbalance 证据。
- 模型执行：04:50:23.717815—04:50:26.919053。8,590 个 corpus/rule case、34,360 个 task partition、70,100 个 boundary 及 927,720 个 weight/length/floor publication 配置通过；另各 10,000 个 leaf task/batch bounds 检查，unstable 同 key 顺序反例检出。
- 证据：阶段研究材料 SHA-256 `4fc1d3d91b68…`；`check_model.py`及`results.json`，模型 SHA-256 `da97a6526017…`。
- 成本边界：random-hash load bound、descriptor memory 估计及算法比较均未测 Rust 布局/硬件；现有 stdstable sort 已适应重复 key。没有仓库运行或 benchmark。
- 04:54:32 标准容器补充完成：transient per-producer map+SmallVec<[Positions;1]>可 expected-linear 稳定分组且 singleton 无 staging heap allocation，但 multi-fragment 可能有双 descriptor 数组，producer 归组仍串行Θ(F_r)。没有建立测量收益或更简单统一 fresh/reuse 架构。
- 补充后报告 SHA-256 `033de42a66e7…`取代前述报告哈希；模型未改动。

### 96｜延迟 scalar loss 的条件性与负面结果

- UTC 对照：本轮探索：10-11 20:42；10-11 20:45；10-11 20:44。

- 本轮探索：2026-10-12 04:42—04:45；阶段材料标记 04:44。
- 结论：旧精确 count 可作为 upper bound 并延后 reconcile，但这是已有 lazy heap 修复的延伸，本身不省工作。绝对 threshold 下任意多 one-unit loss 可分别改变 eligibility/tie，不能套 relative heavy-hitter 的小状态保证。
- 路由限制：不按 parent 索引而保留 producer rows，会使精确 candidate 查列需要 inverse projection。自适应先 scan、累计代价达 M 后 transpose，抽象 expected-hash 模型可给 O(M+Q)及 2M+Q 界，避免 quadratic；最坏并不优于 eager O(M)归约。只有 debt 在 query 或内存压力前永久无关才真省工作。
- 支持模型：10,414 个 scan/promotion 调度通过界检查。sparse-star 示例 512 条非零 loss 对应 262,144 次无索引 row probes。`model.py` SHA-256 `1498c2d2f03f…`，结果`results.json`；精确运行时间未归档。
- 证据：阶段研究材料。抽象操作计数，无语料模拟、hash/cache/allocator 计时或仓库运行；本轮未得到无条件更优 prepare/commit 方法。

### 97｜fragment 指纹、多阈值 census 与 prepare 重复工作新审查

- UTC 对照：活动区间：10-11 20:42；10-11 20:45。

- 活动区间：2026-10-12 04:42—04:45。
- 正在推导：compositional whole-fragment root fingerprints、更强 k-gram capacity deficit 及共用 multi-L census；独立核验 resident-slot key 与 affix 身份。
- prepare 源码候选：AA snapshot 已经验证 starts，后续重复 AA validation 与两次 binary search 可能替换成 trusted snapshot range；生产 Source::Slice 只由 snapshot 产生，任意 malformed slice 来自 private fixture。需要把 fixture 放在真实 snapshot validation 边界验证，而非跳过应有检测。
- birth 审查初步：现有 std stable sort 已有 duplicate 适应性；giant group 的 descriptor 数受 task 数限制，不等于 position frequency。完整报告与模型待完成。
- 状态：未实现的源码与理论发现，不是最终建议或 native speedup。

### 98｜阶段证据分层与后续验证优先级

- UTC 对照：更新时间：10-11 20:47。

- 更新时间：2026-10-12 04:47。
- 当前优先候选：完整 frozen-parent gate 是最直接、可能大幅省去真实 prepare 工作的方向，但仍没有实际 gate-hit trace。历史短 fragment/min3 统计仅是动机，不能当成 gate 速度。
- 研究分层：root-group/grammar 证书提供更强正确性工具，尚未转成 native 收益；AA 重复 validation/search 的源码清理应独立列为较小范围的精确工作删除，先收紧 private source invariant，而非埋入新调度架构。
- 状态：继续研究和核验，不锁定最终方案，不提前定稿等待。

### 99｜实用 fragment census 的驻留 slot 视图审查

- UTC 对照：实际开始：10-11 20:42:41；最终模型执行：10-11 20:50:05.988350；10-11 20:50:06.131092；10-11 20:51:54.153952。

- 实际开始：2026-10-12 04:42:41。
- 最终模型执行：04:50:05.988350—04:50:06.131092（0.143 秒）；报告完成 04:51:54.153952。
- 结果：可消费 materialize 已有 resolver 输出，以完成的 resident fragment(start,length,fingerprint)作暂时 key，实际 slot IDs 比较完整相等，免去额外 resolver pass 与 token 序列拷贝。适用于过滤、affix 与 decorated-ID alias，但不是普通借用 u32 slice，必须使用真实 slot accessor。
- 工作/内存成本：每 edge 完整 count 查表、per-token fingerprint、bounded cache probes、精确相等 slot 读取、qualified gram 读取及 reduce/OR 仍存在。P 个 cache 输出同时保留时额外状态 O(PB+Pm)，不是全程 O(B+m)。key 必须在所有 census 读取完成后、merge 写入前销毁。
- 安全构造：whole-word ranges 不切碎 fragment；cache 只用精确相等累加，counter cap 于 L；eviction/readmission 从 0 只漏计。重复资格事件可重复 OR gram bits，不可直接重复加 Q。各 partial 只消费一次，同 hash mapping 合并。
- 验证：3,045 个 setting 通过，含 2,661 个 affixed、1,819 个 constant cache hash；832 个非空证书、2,213 个空证书。记录 342 次 cache eviction、1,388 次重复 qualification 及 3,662 次 disjoint cache reduction，无有效配置反例。
- 负控制：重复/重叠 partial、class hash 碰撞当相等、truncated fragment、错误 raw slice 身份、重复资格 Q 计数、OR seed 不一致等均否定。模型不代表实际 Rust resolver 执行。
- 证据：阶段研究材料、source manifest、`verify_practical_census.py`及 results/run log；SHA-256 `cd4e849a4194…`。
- 源码：a2 固定提交 materialize/resolver/slot accessor 只读核查；slot accessor 另行取得，未把局部 corpus 摘录当完整文件。无仓库执行、Rust 借用/atomic publication、分配错误或速度测量。

### 100｜whole-root 指纹与加权 grammar capacity-deficit 证书

- UTC 对照：开始：10-11 20:42:03；10-11 20:47:15；10-11 20:49；支持模型 1（截至记录，非精确起点）：10-11 20:45:38。

- 开始：2026-10-12 04:42:03；代数与证据 checkpoint04:47:15，04:49 补充 unsafe tree-hash 反例。
- 强度动机：Chinese ByteLevel 初始 pair 支持 15,049，目标需 49,744 fresh merges；即使无义务损耗，D2 初始证书也不可能在前 34,695 条规则内激活。这是支持上限，不是已测的证书值。
- 新结果：对精确合格 fragment 的完整 root 做 occupied-bin 义务，production 按其 snapshot 展开清 root bin；parse-independent concatenation fingerprint 可 O(1)组合，无需展开 token 字符串。直接 hash 两个 child hashes 取决于 parse，可能漏掉应清 bin，不能代替 monoid。
- 具体 monoid：每 token 保存 H 和 P，拼接 H(uv)=H(u)P(v)+H(v)、P(uv)=P(u)P(v)，模 2^64 的 wrap 仅用于 fingerprint；频率计数仍不得 wrap。预计 16bytes/token，仅 metadata 约 800KB/50k IDs，未测实际布局。
- 更强界：k-gram 与不同短 root 长度域按每条 production 最大 credit 消耗联合计账。k3 为 R2+ceil((D3+R3)/2)，k4 为 R2,3+ceil((D4+2R4+R5)/3)。与 whole-root 总数取 max，不能随意相加；每个 root 长度域独立 hash/clear，使用 snapshot atomic length。
- 构造强度：48 个 unit-weight、唯一外 context 词含 16 种 binary length4 core，L3 下 D2 仅 4、root 可 16，实际高频 prefix23 条。证明能绕过 D2 支持天花板，不是 Chinese 分布模拟。
- 支持模型 1：截至 04:45:38，24,000 个 setting、444,136 个 joined state 通过 monoid 一致性、credit 每步下降≤1 及首次低频前义务清零；11,861 个生成 state 强于旧界/root 界。`check_weighted_fragment_roots.py` SHA-256 `35cd63d12c7e…`。
- 独立模型 2：04:46:16 报告 20,020 个 setting、285,680 个 state，k2..6、bin1/3/11/101 通过；用完整 tuple 展开检验代数，不是 fingerprint 集成。`weighted-capacity.py` SHA-256 `eae46a1bd9b6…`。两模型均未归档精确进程起止。
- 状态：阶段研究材料阶段研究；fresh、uncapped、含 AA/zero 模型，无仓库运行、cap 源码集成或性能结论。多 L 共用 census 仍需分别证明各 floor，禁止跨 floor 累加义务。

### 101｜AA snapshot 并行构造的额外工作审查

- UTC 对照：时间：10-11 20:47；10-11 20:49；04:50—04:53:56 新候选（进行中；所列为目标时间）：10-11 20:50；10-11 20:53:56。

- 时间：2026-10-12 04:47—04:49。
- 发现：当前 serial AA filter 会对 overlap 短路而不调用 matcher；并行“两遍完整 validity”增加实际 matcher/decode 工作，不能宣称免费并行化。
- 后续候选：fixed-span segmented parity summary、O(tasks) prefix 及安全 accepted 输出收集；另审查 one-pass leading-run patch。尚在推导/验证。
- 区分：AA 既有重复 validation/search 清理仍是独立较小范围候选，不依赖采用新并行 snapshot 方案。
- 04:50—04:53:56 新候选：单次 decode 的 parallel private prepare 可能复用 local accepted SmallVec 为 Job.starts，修补 leading parity 并用 O(T)neighbor boundary 信息，省 global accepted Vec 与 job 复制。stale-heavy raw task 切分可能增加 fragment 数，另有 O(K)patch 工作；独立模型进行中，不预报通过。

### 102｜weighted-root capacity 独立扩展验证

- UTC 对照：开始：10-11 20:52:32；10-11 20:56:46.399305；10-11 20:56:47.148288；10-11 20:57:25.836619；范围：10-11 20:57:35。

- 开始：2026-10-12 04:52:32；最终执行 04:56:46.399305—04:56:47.148288（0.749 秒）；报告完成 04:57:25.836619。
- 成功：3,048 个 trace setting、19,949 个 frontier、16,901 次 production credit 检查、7,519 次具体未来 horizon 通过；含 1,310 个 mid-run、1,505 个初始 admission 例外、2,145 条 AA、615 个 constant-bin 及 1,830 个退化 monoid base 配置。
- 扩展：覆盖 strict caps、可变 coordinate spans、较小 reserved first activations、zeros；bounded prefix/suffix 与 H/P metadata 匹配完整 expansion oracle，验证真实未来 greedy 可用性而非仅抽象 credit 式。447 个生成 state 强于旧界与 root 界。
- 条件差别：按正确长度域 clear 使每步 credit 下降≤k−1、G−H 不下降；额外乱清其他域仍安全削弱，但会破坏此 gap 单调性。重复 root admission 或把重叠 root/gram 界直接相加不安全。
- 证据：阶段研究材料、`verify_weighted_roots.py`与 results/run log；SHA-256 `e675f22b45ef…`，依赖 SHA-256 `69d6b3696cd2…`。
- 范围：fresh 有限数学模型；不验证 active reuse、partial commit、Rust/并发/分配错误或性能。geometric live-reindex 模型于 04:57:35 开始，另存新材料。

### 103｜按 live 规模减半触发的多级降 floor 恢复

- UTC 对照：初始推导：10-11 20:50；10-11 20:53:56；10-11 20:57:35；最终执行：10-11 21:01:20.302916；10-11 21:01:20.769443；10-11 21:02:41.267873。

- 初始推导：2026-10-12 04:50—04:53:56；独立模型开始 04:57:35。
- 最终执行：05:01:20.302916—05:01:20.769443（0.467 秒）；报告完成 05:02:41.267873。
- 策略：clean fresh underfill 时，只有 S≤上次 build 的 S/2 才允许另一个高于基础 F 的较低 floor；否则只重建一次 F 并停止试探。每次 drop 旧 index、按 live endpoints 完整恢复，保留已提交 prefix 与原身份。
- 成功：1,334 个完整 reference 配置一致，1,067 次 live rebuild（40 次中间 build、1,027 次最终 build）；37 个多次重建 case、1 个多次中间 build 定向 case。4,328 条规则含 1,415AA、983reserved 动作；26,843 次实际物理替换含 526 次零权重替换。
- 体积定义：S=live heads+所有真实词 separator（包括空词）+leading sentinel。成功替换只按已有 accepted physical starts 数减少 S，不能用加权 frequency 或 AA raw count。每次 ledger 和独立 path 数匹配。
- 预算：包括初始 construction 在内，所有 index build 输入体积≤2S0；两遍 count/emission 各自读取的倍率另计。模型有 497 个达到 2S0 的 tight case。original-coordinate gap 跳跃与 compressed weight-region sweep 显式建模，不把物理 holes 逐 slot 扫描免费忽略。
- 证据：阶段研究材料、`verify_geometric_reindex.py`与 results/run log；SHA-256 `66b7257a1671…`，依赖 SHA-256 `b9de7872c688…`。
- 边界：只是成功 fresh 路径的 build 输入体积，不是 wall、peak memory、allocation、所有 per-ID setup 上界。active-reuse 原始重启和真实 partial-error 丢弃/传播均不在该 ledger 内；原材料把 partial errors 合写为重启的说法已于 08:12 更正（阶段 165）。未执行仓库或 benchmark。

### 104｜按真实 matched interval 放宽 fresh batch 冲突

- UTC 对照：开始：10-11 20:56:08；10-11 21:05:47；覆盖模型执行：10-11 21:03:36.100048；10-11 21:03:37.566087。

- 开始：2026-10-12 04:56:08；阶段结束 05:05:47。
- 归类：actual-physical-conflict batch 正确性已在早期阶段研究材料证明，不算本轮新定理。本轮增量为 right-only overlap detector、min later-rank prefix、源码 staging 与 cached Positions bounds 对照。
- 最小控制：selection 时用 cached whole-posting interval bounds 可放宽部分 crossed-endpoint 冲突，无新 prepare join、tentative identity 或 persistent metadata；leaf envelope 细化只扫 descriptor，但需工作预算，selectivity 未知。
- 完整实际 starts 证书：每个相关 validated occurrence 只需 right-overlap query，prefix cutoff 为 later rank 的 min；但现有 take 会删 count、resolve_merge 改 vocab、prepare_identity 改 geometry，必须延迟 commit 并可 restore suffix。新增 join、重复 context 工作及浪费 accepted-start buffers 不能忽略。
- 覆盖模型执行：05:03:36.100048—05:03:37.566087。20,000 个合成 case 中保守 whole/leaf cover 证书无 false positive；start-only 与 endpoint-only region 过滤各有漏 overlap 反例。245/435/26 个不同证书命中数仅属合成分布，不能当真实 selectivity。
- 证据：阶段研究材料 SHA-256 `7fac22c92c58…`；`check_model.py`及`results.json`，SHA-256 `31cc24b87717…`。
- 限制：fresh corpus 已 drop word directory，不能免费依赖整词 id 索引；reserved/AA/reuse barrier 保留，不跳过最大冲突 candidate。无仓库运行或测量性能。

### 105｜PR 与分支复查，保持固定证据基线

- UTC 对照：PR 检查：10-11 20:54:38.780；分支检查：10-11 20:59:29。

- PR 检查：2026-10-12 04:54:38.780。PR2501 仍 open、未合并；requested reviewers/teams/assignees 为空，完整 8-event timeline 没有 assignment 事件。未发送无变化通知。
- 分支检查：2026-10-12 04:59:29。compare 仍确认`a2b20f329d1b4a0d39b5721f4a7a678a80d0d9b2`相同。

### 106｜把 protected fragments 作为精确压缩工作语料

- UTC 对照：开始：10-11 21:03:07；10-11 21:05；10-11 21:06；证据（进行中；所列为目标时间）：10-11 21:07:19。

- 开始：2026-10-12 05:03:07；首份材料 05:05，05:06 继续独立验证。
- 新变体：完整初始 checked pair counts 定义<L 不可跨 cut，对实际 resolved-ID words 切分；精确去重所有长度≥2 的 fragment 并聚合 multiplicity×weight，保留原 alphabet、vocab/IDs、reserved 状态、affix 语义和原始输入。
- 条件推导：L=基础 F 时是整个 fresh-run 精确变换，包含合法 underfill，无需试探 target 成功；L>F 时仍需 target-fill 或恢复。active reuse 回到原始 reuse 路径。不能只保留 weight≥L 的 fragment class，高 pair 可能分布在多个低 weight class 中。
- 工作机会：已有初始≥L counts 仍权威，只需为缩减后语料发射 Positions。只有去重会减少合格 edge occurrence；单纯切分不减少它们。separator 可能把 resident slots 推高到约 1.5N，原 unit-weight fast path 也可能失去。
- 证据：阶段研究材料，基于已证明边界 lineage 的新语料变换推导。05:07:19 另开始只读对抗审查，重点 ties、IDs、零权重、初始 max-length 例外与 reserved 激活；独立模型进行中，没有真实压缩率、仓库实现或速度结果。

### 107｜初始 checked counts 后的 fresh 算术边界

- UTC 对照：材料时间：10-11 21:03；10-11 21:06。

- 材料时间：2026-10-12 05:03；05:06 记录支持模型结果，精确模型执行起止未归档。
- 证明：初始全部 pair checked 成功后，任意 fresh 当前 pair occurrence 注入唯一 initial family，因此 raw count、同 key birth partial/global sum 和合法 old loss 均受该 family≤u64 上界约束，包括 AA/zero/low reserved 首次激活。
- 错误边界：这不能界定跨 key row/sibling/global totals；CABD 权重 u64MAX 生成 CX 与 XD 各合法，却总和 2×MAX。reuse 失去唯一 family。对 malformed/重复 owner/internal corruption 的检测仍是独立要求，未建议移除 checked 算术。
- Positions 界：valid fresh 唯一坐标与 resident-size 可界定单 cohort 长度，但不能证明所有同时 live 叶子、descriptor/maps/jobs 可分配；跳过存储的 resource-error 等价仍未解决。初始全部 checked counts、零合并输入检查及 affix/reuse 限制保留。
- 支持模型：10,000 个向 u64 上界缩放的语料、87,613 次 fresh 转移，含 10,341 次 AA 与 9,525 次 low reserved 首次激活，验证 family 界与合法 old removal；包含 cross-key overflow 反例。
- 证据：阶段研究材料、`model.py`、`results.json`；源 SHA-256 `a1b314a929ba…`。只读固定 mod/index/merge/positions；旧 corpus 仅用于明确条件化 resident-size 论证。不是 Rust fault injection 或 allocation 验证，没有仓库执行。

### 108｜fragment 语料预处理的直接文献/实现先例

- UTC 对照：发现与网页核验：10-11 21:08。

- 发现与网页核验：2026-10-12 05:08。
- 来源：[FFBPE 0.1.8 maintainer release 说明](https://pypi.org/project/ffbpe/0.1.8/)明确描述两遍 frequent Unicode bigram inventory shaping、在未保留 boundary 处切分、重新聚合 word 与 inclusive cutoff ties。
- 修正：protected-fragment dedup 不作新颖性宣称，正在继续审查真实代码与 provenance。该 release 广告的 64MiB Chinese 约 3 倍 inventory/7 倍 training 是作者另一 workflow 结果，不能迁移到当前 Rust 基线或本方案。
- 验证进度：独立 reduced-fragment 模型初步报告 7,904 个 setting、14,977 个 frontier 通过；详细报告、精确时间及哈希待核验。
- 05:09 重要限定：FFBPE BENCHMARKS.md 第 33–35 行明确其 Unicode shaping 对比不是 model-parity；HF 对比使用同一个已 shaped inventory，不能证明原始与 shaped 模型精确相同。当前 actual-ID barrier 定理需独立成立。
- ByteLevel 反例：éa 的 UTF-8 C3 A9 61 与Ωa 的 CE A9 61 各 weight1、F2，byte pair A9,61 总 count2，但各 Unicode character pair 仅 1；按 character 切分会删掉应选 merge，不能直接借用其 Unicode 预处理。
- 源码限制：tie 使用 count 与 Reverse<Pair>可保持，但原 active IDs/span 必须保留，包含 zero-weight singleton IDs，否则会改变 reuse 检测。

### 109｜fragment 语料缩减的源码对抗审查

- UTC 对照：开始：10-11 21:07:19；10-11 21:12:07。

- 开始：2026-10-12 05:07:19；阶段结束 05:12:07。
- 结果：在完整保留 resolved-ID vocabulary、allocator、activation/span 历史、初始 admission 与 reuse 重启前提下未找到模型反例。Candidate comparator 仅(count,Reverse<Pair>)，不读取 Positions；物理坐标、task/first-touch 改变不必改变 fresh 选 key。
- 关键陷阱：不能只保留 alphabet 或 token 字符串。prefix=ab、suffix=c 时，positive ab 解析[a,abbc]，zero-weight xcd 解析[x,abc,abdc]；所选 merge 目标 abc 已被零权重词激活。缩减丢弃 singleton 后若重算 activation，会漏掉原 reuse 重启。该 fixture 是重启分类错误，不宣称已证明最终模型不同。
- 定向执行：05:10:27.261387—05:10:27.261422，24 个 priority 排列、activation 字符串与 AA 原/缩减 raw/accepted weight 算术检查通过。
- 证据：阶段研究材料 SHA-256 `1e741908ca47…`；`check_contracts.py`及`results.json`，SHA-256 `d8537ded2cd4…`。
- 范围：固定源码假设与小 fixture 检查，不是实际 Rust trainer 执行、完整错误等价或性能证明；更丰富变换模型另行记录。

### 110｜初始 rare-boundary 工作语料缩减的独立验证

- UTC 对照：开始：10-11 21:05:09；10-11 21:11:09.270580；10-11 21:11:10.898802；10-11 21:12:37.617771。

- 开始：2026-10-12 05:05:09；最终执行 05:11:09.270580—05:11:10.898802（1.628 秒）；报告完成 05:12:37.617771。
- 成功：7,907 个完整配置、14,984 个 high-index frontier、40,075 次 fragment 同步与 7,077 次 weighted match/六类 delta map 检查通过。3,437 次 target certificate（1,650 次非空成功），112 次正常基础 floor underfill，4,358 次原始输入回退。
- 保留条件：所有长度≥2 的 maximal class 都保留，包括低于 L、零权重与长于 max-token-length 的 fragment；只有 empty/singleton 在原初始化后可丢。初始 span1 pair admission 例外与 newborn strict gate 分开，原 activation 包括零权重/单字符来源不能重算。
- 新表示边界：保持规则、raw counts、output IDs/vocab 与模型，不声称缩减后的物理 Positions 等于原始语料；若最终原始 tokenization 是可观察结果，需把 learned rules 作用回原始输入。
- 反例覆盖：Unicode-character cut 用于 UTF-8 byte BPE、漏低 mass class、按 whole-fragment cap 筛掉长 class、affix 重新 tokenize、丢 active identity、reuse 及初始 checked overflow。2 个预期 overflow 控制检出，无意外有效配置失败。
- 证据：阶段研究材料、`verify_input_reduction.py`、`input_reduction_results.json`、`input_reduction_run.log`；SHA-256 `100bc30cc659…`。
- 范围：自含标准库数学模型，不导入仓库；7,904/14,977 是早期扩展前摘要，由此 7,907/14,984 最终结果取代。物理工作减少计数为生成 fixture 结果，不作真实压缩率、速度或资源失败等价结论。

### 111｜FFBPE hot-pair occurrence window 的可迁移性

- UTC 对照：开始：10-11 21:12:51；10-11 21:23:41；扫描 fixture：10-11 21:20:25.354526；10-11 21:20:25.445892。

- 开始：2026-10-12 05:12:51；阶段结束 05:23:41。
- 固定源码：`tokn-ai/ffbpe@f29ac35d5dfa7799bcf824a4abfa68439c81c9e0`保留 exact scalar table 与 word-ID occurrence sets，支持批量 top-K hydrate；不是精确 coordinate Positions。merge 仍先为 positive delta 构造 affected-word sets，再丢 cold sets，不能声称其自动省掉本方案 prepare birth appends。
- 新对照：count/heap 保持基础 F，仅 posting threshold T>F 控制完整 witness 可捕获的 child Positions。cold winner 不能跳过，需 joined multi-key live hydration，可降低 posting threshold 而非改变模型 floor；增加 ready/missing payload 状态、scalar/payload 解耦与恢复策略。
- 限制：FFBPE positive-delta 过滤遗漏 zero-weight word 的 birth set，不可直接迁移；K2 可在 prune 前突发 100 个 resident newborn，key 数不等于 byte 上界。inline Option<Positions>本身也不自动变紧凑。reuse 保留原始 eager 重启。
- 扫描 fixture：05:20:25.354526—05:20:25.445892，33 步链在 K1/4/16 需 33/9/3 次 hydrate scan，含初始；说明小 window 可反复扫描未显著缩小语料。100-newborn burst 与 zero-filter 反例验证。不是完整 trainer 或 native 测量。
- 纠正：现有 O(P) coarse whole-word checkpoints 足够支持 live 扫描，不必引入 O(word_count)目录；误称“完整 word 目录必需”已修正。
- 证据：阶段研究材料 SHA-256 `c1b383a5d1eb…`；`check_model.py`及`results.json`，SHA-256 `bb6039ea4892…`。
- 性能边界：作者 1GiB Chinese 两 scan benchmark 属于不同表示/流程，不迁移收益。无仓库执行/修改或本方案性能结果。

### 112｜safe cut 细化的全生命周期工作与 slot 界

- UTC 对照：开始：10-11 21:12:55；10-11 21:14:11.055967；10-11 21:14:12.302798；10-11 21:15:04.062215。

- 开始：2026-10-12 05:12:55；执行 05:14:11.055967—05:14:12.302798（1.247 秒）；报告完成 05:15:04.062215。
- 证明/验证：uniform 全局或 per-exact-class 加入安全 cut 后，精确去重所有非 singleton fragment；高 floor 全历史的 physical heads、admitted high Positions、accepted matches 及保留高 family cut 事件均不增加，weighted counts 与六类 delta maps 一致。
- slot 条件：每个 coarse class 保留 q 个非 singleton child、丢 t 个 singleton atom，q≤t+1 可保证所有后续高频 frontier 的 logical resident slots 不增加；差值 q−1−t 不随后续收缩变化。所有同 class occurrence 必须统一切分，不能一部分保留整词另一部分贡献碎片。
- 模型：701 个语料、1,402 条完整高 floor trace、7,852 个 joined frontier、6,450 次 per-rule 检查通过。612 个未加 guard 的 frontier 出现 slot 膨胀，证明需区分 heads 减少与 slot 总数；guard 拒绝 278 次 split，701 条 guarded trace 通过。
- 证据：阶段研究材料、`verify_cut_refinement.py`及 results/run log；SHA-256 `536a0a7c99e0…`，依赖 SHA-256 `100bc30cc659…`。
- 限制：低 cut 事件不要求相同，数字只属有限生成模型；没有仓库执行、native 工作量或速度测量。

### 113｜mid-run sparse-plane fragment 缩减验证

- UTC 对照：开始：10-11 21:15:04；10-11 21:19:22.854437；10-11 21:19:23.731149；10-11 21:20:44.832236。

- 开始：2026-10-12 05:15:04；最终执行 05:19:22.854437—05:19:23.731149（0.877 秒）；报告完成 05:20:44.832236。
- 成功：825 个配置，724 个真正非初始且 nonunit-span snapshot，1,631 条原/缩减 suffix 规则一致；135,930 次 forward/backward head 操作、3,304 次 checkpoint/weight-region 重建对照通过。
- 物理结果：生成 fixture 的 suffix matches 从 2,727 降至 2,549 而 weighted mass 相等；6,562 次 plane 审计通过。823 个完整 logical reference 检查外，两分支仅正确 dispatch 到原始 reuse 重启，不验证 reuse kernel。
- 新构造：ABCD:1/EBCD:1/AB:2/EB:2、F2 初始无低 family cut；AB3、EB3 后 XC/YC 降至 1，可缩成一个 CD:2 class 并保持下一规则，说明 mid-run 可能出现初始没有的机会。
- 必要前提：fully committed 完整 base-F index；缺席可能是 below-F 或不可再变 eligible 的 strict-cap cut。保留 current-ID span 并写正确 head/tail sparse plane，旧 Positions 全部失效，原 vocab/activation/history 仍保留。不能稠密打包后继续用旧 span、重置 span、漏 tail 或丢 zero-only active identity。
- 负控制：dense old-span packing、reset spans/cap admission、missing tail、丢 activation、几何已丢后降低 floor 都检出；不能直接在缩减语料降 floor 恢复已切连接。
- 证据：阶段研究材料、`verify_midrun_sparse.py`与 results/run log；SHA-256 `07815a316cba…`，依赖 SHA-256 `100bc30cc659…`。
- 边界：独立标准库数学模型；没有仓库执行、Rust memory model、真实 allocation failure 或性能测量。826 次 reduction 含一个负控制额外动作，不当成 826 个完整配置。

### 114｜分支复查与 heap bulk publication 小控制

- UTC 对照：分支检查：10-11 21:19:33。

- 分支检查：2026-10-12 05:19:33，compare 仍确认 head 为`a2b20f329d1b4a0d39b5721f4a7a678a80d0d9b2`且相同。
- library 源码审查：固定 dary_heap0.3.9 的 Extend 存在 adaptive bulk heapify，可作为 fresh-only 小范围对照。实际作者 benchmark 二进制解析到的版本尚未验证，不能据 library 事实宣称当前性能改进。
- 状态：heap 候选仅是 API 工作方式发现，未实现或测量。

### 115｜保留旧 token plane 的 representative fragment 方案

- UTC 对照：探索：10-11 21:22；10-11 21:27；10-11 21:23:14；10-11 21:31:43.599819；初轮执行：10-11 21:27:12.773850；10-11 21:27:13.545901。

- 探索：2026-10-12 05:22—05:27；独立模型开始 05:23:14，最终报告 05:31:43.599819。
- 候选：保留原 Slots 与 span/activation，代表 fragment 为旧坐标区间并带聚合 weight，只重建代表 postings。避免第二个 token plane，但不释放原 resident slots；将 gap 当活动范围会错误。
- source 可选方案：显式 range 边界裁剪 immediate/two-hop neighbor，一个预计 24B weighted Range 替代原 16B weight 记录；布局与 weight-region 退化未测。纯 frozen gate 模型在完整不变量下也维持代表区间 confinement，但不因此省掉正常 token/word/sentinel 验证。
- 初轮执行：05:27:12.773850—05:27:13.545901，736 配置/1,859future rules；初轮脚本 SHA `a043aea04384…`。
- 最终执行：05:30:07.602956—05:30:08.503278（0.900 秒）。738 配置、402mid-run、1,861future rules 通过；1,861 次 original 事件 oracle、1,866 次 rebased 事件 oracle 一致，包含 3 个 compatible cohorts。803 次 cross-cut transition 与 15 次 false selected-neighbor prediction 均被冻结旧 witness gate 正确抑制。
- 反例：ABAB:1/F2 代表 AB weight2 若不 gate 会假 birth XX2；原 weight lookup、after-take AA gate、全旧 plane ghost recount 与 physical 整词 completion callback 都可能错误。后续统计/扫描必须分清 logical representatives 与旧物理 plane。
- 证据：相关阶段研究材料、`verify_inplace_fragments.py`、results/run log。最终 SHA-256 `73f56bae6e8b…`取代初轮；依赖 SHA-256 `07815a316cba…`。
- 边界：固定 floor/fresh 数学模型，保留 activation 且 gate 包括 selected AA；没有仓库运行、Rustmemory/并发或速度结论。整个旧 plane 仍分配，不能计作 resident memory 释放。

### 116｜从当前 representatives 恢复原始 live 语料

- UTC 对照：初步推导：10-11 21:25；10-11 21:27；10-11 21:33:26；最终执行：10-11 21:36:01.497960；10-11 21:36:02.599481；10-11 21:37:23.386396。

- 初步推导：2026-10-12 05:25—05:27；独立模型开始 05:33:26。
- 最终执行：05:36:01.497960—05:36:02.599481（1.102 秒）；报告完成 05:37:23.386396。
- 结果：clean high-L underfill 时，把每个原 fragment occurrence 展开为代表当前序列，恢复原词边界/weight 与 head/tail/Positions，重建 F 并继续已提交 prefix，无需重放规则或再分配 IDs。
- 两种状态：retained 旧坐标 plane 只需为实际 duplicate class 保留 map，unique class 留原处；compact relocated plane 则需所有 class→representative map。初始 cut 谓词不可变，mutable live key map 不够。
- 模型：732 配置，525 次精确原 geometry/index 恢复，357 次非空 prefix 恢复，799 条 base suffix 规则；731 完整 base 对照与 1 个 original reuse restart dispatch。206 次 target 成功无需 restore，420 个 zero-weight 原 row 正确恢复。
- 负例：把 weight 当 physical multiplicity、evict map 条目、使用 mutable cut/live key、错误 affix context、缺 unique relocated map 或沿用 aggregate weights 均失败。字节/原 resolver/context 保留是实际寿命成本。
- 证据：阶段研究材料；阶段研究材料、`verify_representative_recovery.py`与 results/run log，SHA-256 `747c35d0740e…`，依赖 SHA-256 `73f56bae6e8b…`。
- 成本边界：map/resolver/cut metadata 活得更久，恢复与重建仍要工作，compact 模式可能双 plane 共存。无仓库运行、allocation-error 或性能结果。
- 05:41 源码恢复成本修正：CorpusPlan 按 weight 并行 unstable 排序后丢原 rows/starts，仅 WordCounts/resolver 不足以重建 retained-plane 原 destination。还需保留原 row order/starts、per-copy destination 或另一个已证明 mapping，并单独保留原 weight regions。compact 新 plane 可选择新顺序再全量重建 Positions；两者成本不能混淆。

### 117｜fragment quotient 与证书组合的全生命周期对抗审查

- UTC 对照：开始：10-11 21:35:53；阶段结束：10-11 21:45:09。

- 开始：2026-10-12 05:35:53。
- 阶段结束：2026-10-12 05:45:09。
- 工作：审查阶段研究材料与 representative 恢复方案，比较 initial/mid-run protected-fragment quotient 的精确性与完整 prepare/commit 成本，构造 ByteLevel 及 weight-one 反例。
- 状态：只读研究，未执行或修改仓库。
- 05:40 成本措辞校正：q 种 root 各重复 m 次的构造证明 physical decode/accepted-start/apply 从 qm 降 q，不能推出 grouped commit 相同比例减少；selected-key retirement 仍 q，受保护的双 atom roots 没有保留 birth。commit 必须另记 local groups、global keys、birth fragment descriptors。
- rare 外 boundary 抑制与独立 frozen-witness gate 收益重叠，不可重复计算。阶段研究材料已明确更正此前过宽 prepare/commit 措辞。

- 有限 model：05:43:02.089792—05:43:07.881847，20,000 个 unique unit/zero case 与 5,000 个 mixed-weight partial case 通过，含 sharp F2/L2 与 F2/L3、binary 无 cut 无收益、少 cut 却长 core 大量工作节省、zero/heavy/alias 例外。
- 新界：C≤S_low≤(F−1)A²；全 unique unit rows 的 full quotient Q_L≤floor(2C/(L−1))，因每 class 可有一份未切整词。affected-only 则 Q_L≤floor(2S_low/L)，不需要全 unit 假设。两界不能互换，且不限制 long-gram 证书或 few-cut long-core 节省。
- 证据：阶段研究材料 SHA-256 `42264e2d701d…`；`check_bounds.py`及`results.json`，SHA-256 `4cdcaec3d5d2…`。无仓库运行、真实语料 selectivity 或 benchmark。

### 118｜重复缩减的 head/range 预算与初始稀有 cut 上限

- UTC 对照：探索更新时间：10-11 21:36；10-11 21:37。

- 探索更新时间：2026-10-12 05:36—05:37。
- 体积细化：设 H 为当前 live heads、D 为非空 active logical ranges，则 D≤H；若只有 H 至少减半才重新做 census，累计 head+range 事件≤4H0。slot zeroing、原 byte hashing、分配与 per-ID setup 另计，不能把这个界扩展为总 wall 或全部内存访问。
- bounded dedup 进展：whole-map 分批去重后，同一 exact class 可安全保留多个加权 representatives，仍由全局 pair count 合并。cache 容量只界定 transient key metadata，persistent representative 数 R 不受该容量约束；不能声称整个转换内存有同样上限。
- 初始支持上限：对固定初始 alphabet 大小 A、正整数 unit rows，稀有 pair family 总 cut 数候选界 C≤(F−1)A²；正在核验 zero weights、max length、alias 与 sharpness 限制。
- 状态：阶段推导，未实施或 benchmark。

### 119｜证书状态释放与 initial barrier bitmap 条件

- UTC 对照：更新时间：10-11 21:37；10-11 21:38。

- 更新时间：2026-10-12 05:37—05:38。
- 简化：一旦 Q≥H 等证书已证明剩余 target 的 floor，可丢弃 discharge tracker 并只保留 certified floor；reuse fallback 与正确 horizon 前提不变。动态 tracker 仅在激活前或争取更高 floor 时有价值。
- bitmap：若确有 256 个紧凑 rank 的初始 alphabet，精确 pair barrier bitmap 可为 8KiB。ByteLevel 原始 numeric ID 可能被 special/reserved 偏移，不能直接假定 ID0..255 或免费 rank mapping。
- 状态：理论/source 维护性细化，无实现/布局测量。

### 120｜缩减 metadata 成本与增量 cut 发现反例

- UTC 对照：分支复查：10-11 21:38:39。

- 分支复查：2026-10-12 05:38:39，head 仍为`a2b20f329d1b4a0d39b5721f4a7a678a80d0d9b2`且相同。
- 内存提醒：显式 reduced interval 约 24D，对比 baseline 约 16R 的 weight regions；原 unit-weight 语料可能只有一个 region。保留旧 plane 并未释放 token slots，新增 descriptor 可能超过所省压缩 Positions 字节，不能概括为省内存。
- 增量 cut 反例：CAB:2 在 CA→X 后，旧 AB 因 retirement 留下的 stale Positions 不能授权切当前 XB；count 键退休不等于其历史物理 cut 低频。正在审查更强 safe-cut 发现条件。
- 初始化 source 事实：WordCounts 已冻结为 Vec，common view 没有 lookup；cut-local dictionary 若需反向关联原 word，可能要额外原 row 扫描，不能免费复用不存在的 map。
- 状态：成本与正确性范围检查，无生产测量。

### 121｜仅改 cut-affected rows 的受限 quotient

- UTC 对照：推导更新时间：10-11 21:42。

- 推导更新时间：2026-10-12 05:42。
- 候选界：J 个受影响正权重 row≤C 个 cut；删除完整受影响 word interval 连同尾 separator 后，普通 pieces≤原 R+J；插入代表 D≤C+J≤2C，故新 region 数≤R+3C。
- 取舍：不动 uncut rows（即使与其他序列重复）及 zero rows，牺牲压缩率，避免整个 row 集合 reverse join 或特定 map lookup。仍需审查 source 和寿命成本，不能直接认为 region 很少。
- key 工作：应直接用原 byte slice 与原 first/last 上下文；反复 fullword.symbols.skip 前缀可能导致 quadratic 工作。
- 恢复成本细化：retained-plane 若保留 sorted row refs/starts，可能增加约 16W+8W 生命周期内存（未测布局）；仅持有 owned WordCounts/resolver 不足。
- 状态：新代数/接口候选，未实现或 benchmark。

### 122｜ByteLevel 场景常数的原始 source 交叉核验

- UTC 对照：读取审计：10-11 21:43:41；10-11 21:45:35；10-11 21:46。

- 读取审计：2026-10-12 05:43:41—05:45:35；证据材料整理至 05:46。
- 直接证据：a2 固定提交 diagnostic job 与最新 read job 同一 zh-bytelevel prepared input，target50,000、min_frequency2、无 prefix/suffix/max-token-length/alphabet 限制；diagnostic stdout 与 read pairs 均 actual_merges49,744、actual_vocab50,000、special_tokens0。不是仅由最终差值推断配置。
- 源码：冻结 runner adapter 显式 initial_alphabet(ByteLevel::alphabet())，该 API 返回 256 个 bijective 字符；未调用 special_tokens，builder 默认空。故实际 active 初始 atoms≤256、无多字符 reserved 零增长输出；初始 vocab256 与 49,744 实际 merge 相容。
- 15,049 来源：同场景`child-inline-fast-final-diagnostic.initial.containers`，root14,695、leaf149,214。是 adopted-child 初始 admitted unique pair support，非 final-read 重新诊断、非全部初始 keys、非实测 D2/fragment 证书。
- 数学推论：对已证明 affected-only root 界，A≤256/F2 给 Q3≤43,690<49,744；结合 admitted support 最多可收紧为 Q3≤33,658。这些是上限推论，实际 S_low 与 Q 未测，不能推出没有 dedup 收益或排除 grammar/后期证书。
- 可复核来源：[diagnostic job](https://github.com/Yikai-Liao/tokenizers/blob/a2b20f329d1b4a0d39b5721f4a7a678a80d0d9b2/experiments/bpe-simplification/evidence/positions-child-20261012/final-storage-multi/zh-bytelevel_regex-w4-r0-child-inline-fast-final-diagnostic/job.json)、[storage summary](https://github.com/Yikai-Liao/tokenizers/blob/a2b20f329d1b4a0d39b5721f4a7a678a80d0d9b2/experiments/bpe-simplification/evidence/positions-child-20261012/final-storage-summary.json)、[latest read pairs](https://github.com/Yikai-Liao/tokenizers/blob/a2b20f329d1b4a0d39b5721f4a7a678a80d0d9b2/experiments/bpe-simplification/evidence/positions-read-20261012/pairs.json)。全部 job/adapter/default/mapping 固定链接与 Git blob SHA 保存在阶段研究材料。
- 方式：GitHub 只读取得原 JSON 与源码，未读取新语料、执行仓库或 benchmark。一次猜测 adapter 文件名 404 后使用 core 明确引用的真实文件完成核查，没有访问阻碍。

### 123｜radix priority queue 与 fresh monotone priority 审查

- UTC 对照：开始：10-11 21:46:57；10-11 21:53:47。

- 开始：2026-10-12 05:46:57；阶段结束 05:53:47。
- 正结果：当前 fresh best()/take()真实 lookahead 支持 scalar 频率单调 watermark；若严格限制所有新 output 追加在所有旧 ID 后，完整(count,Reverse(pair))也单调，radix-heap tuple 加单个 owning cached head 可适配 best/take。
- 反例：首次激活 low reserved ID 可在相同 count 生成字典序更优 child，完整 priority 反向；任意为了组织 equal-count group 而多 pop 的 lookahead 不在原证明内。一般 count-only radix 仍需精确 pair tie 处理。
- library 审查：radix-heap0.4.2 及固定 upstream 源码、rheaps 固定源码均只读。radix bucket Vec drain 保留 capacity，冷 candidate 块跨 level 重分配可令总 header capacity 远超 live 数量；rheaps equal-key refresh 循环可推导 quadratic 比较，未执行或 benchmark。
- 结论：暂保留当前 OctonaryHeap；受限 append-only-ID 控制需独立 queue profile、stale repair/tie 分布、redistribution 与 capacity 证据。它不减少 prepare 事件、count routing、sort 或 concat 工作。
- 证据：阶段研究材料 SHA-256 `417a612a7e0d…`，target 固定 a2b20f32。没有代码运行、安装或仓库修改。
- 关联补充：前述 ByteLevel 原 source 核验已加入阶段研究材料，其报告新 SHA-256 `2485dabef44c…`；模型不变，Q3≤33,658 仍是界而非测量 yield。

### 124｜初始 ByteLevel 剪枝机会的阶段优先级校正

- UTC 对照：更新时间：10-11 21:55。

- 更新时间：2026-10-12 05:55。
- 固定 256 初始 alphabet 时，positive 初始 pair count<F 的物理坐标总数≤(F−1)×65,536；F2≤65,536、F3≤131,072，独立于总语料 N。对很大的 ByteLevel，这是有限的初始低频 append/cut 预算，不能用 Whitespace 大 pair 域直觉预测收益。
- 不适用范围：该上限不限制 few-cut long-core dedup 节省，不适用于 mid-run 扩大后的 ID 域、zero-weight Positions 或任意 decorated alphabet。
- 当前方向：later born rare families 和 pre-group frozen-witness gate 仍更值得用实际 group/event 证据验证。initial count-first/quotient 收益不预设。
- 现有 one-shot 高 T→F live-reindex 已有最小源码集成审查，归类为既有恢复方案落实，不作为新发现；所有方向仍待生产验证，没有最终方案锁定。

### 125｜增量 positive safe-cut 预算与 F2 singleton XOR 定位

- UTC 对照：开始：10-11 21:39:29；10-11 21:56:56。

- 开始：2026-10-12 05:39:29；阶段材料完成 05:56:56。
- 结果：完整 joined 最终 count 可给新增 positive safe-cut 质量而无需 per-key occurrence counter；F2 时余 count1 恰好一个正权重为 1 的 physical cut。当前 source 却在第一次中间 below-floor 就删除 key 并忽略余 loss，所以不能在现有即时退休处加 callback 冒充最终余量。
- 三项分离：发现安全 family、定位当前 cut、形成/去重 fragment 是不同工作；scalar 触发器只可判断何时值得 census，不能免去后两项。
- XOR 候选：每 active pair 另存 u64，把 odd weight occurrence 的固定 physical cut coordinate XOR；完整 count/XOR 一起 join 后 count1 可定位正 cut。posting 左 start 会移动，不能拿它代替 cut 坐标；zero 权重不能无条件参与 XOR。额外至少 8bytes/key 及 per-event parity/XOR、构建/同步均有成本。
- 首模型：05:53:02.151131—05:53:02.780330（0.629 秒），705corpora、4,003commits、1,874 次 F2 精确 positive-cut 检查、365 次 singleton 位置通过。`check_positive_cut_budgets.py` SHA-256 `537c2065223a…`。
- compatible 补充：05:56:50.414429—05:56:50.690265，21,898 配置、18,825 定位及 1,059new/new boundary 通过。`check_xor_batches.py` SHA-256 `451553bbec90…`；这些是有效 compatible transformations，不宣称每批都是 greedy reservation。
- 反例：中间余 1 而最终 0、weighted mass 不能给一般 physical multiplicity、unweighted XOR 受 zero copies 污染、count2 XOR 碰撞、count 和 XOR 混代读均被保留。
- 证据：相关阶段研究材料及各 results/run log。无 Positions 重数方案、仓库执行、Rustmemory 或速度结论。

### 126｜prepare/apply 融合的 interleaving 审查启动

- UTC 对照：开始：10-11 21:57；10-11 22:05:47。

- 开始：2026-10-12 05:57（秒未记录）；阶段报告完成 06:05:47。
- 工作：独立核验混合 prepare/apply 事件交错是否保留边界读、写和 count 语义。
- 06:00:37.855732—06:00:37.861280 模型：1,404 个 SC read schedule 中 naive snapshot 角色解码失败 666 次；8,712 个 tagged old-endpoint 检查通过。脚本`check_interleavings.py` SHA-256 `1060a4152565…`。
- 06:02:50.297244—06:02:50.375183 region 补充：immediate shortcut 仍在 1,404 schedule 中失败 315 次，另 1,500 个 region case 通过；tagged role+ID 涉及更大的表示改动。stage 报告阶段研究材料已完成；不能把 SC 枚举当 Rust memory-model 证明。
- 06:04:18.138899—06:04:18.341999 tagged histories：700 条 history、3,028 条规则、14,348 个 mixed-write state 和 1,143,258 次 virtual cell 检查通过。新增 tagged 表示的性能/生命周期成本未测；没有仓库执行。

- 最终界定：tagged 方案需要 atomic in-cell role+ID 与更改 plane encoding；whole-word region fusion 仍保留 accepted-start 流量并付 pair-region fan-out。region 脚本 SHA-256 `5f27aa94d7a2…`，tagged history 脚本 SHA-256 `df0ac987379f…`；固定 source manifest 另附。

### 127｜fresh count owner 过度拆分的工作/span 审查

- UTC 对照：开始：10-11 21:57:18；10-11 22:01:13。

- 开始：2026-10-12 05:57:18；阶段结束 06:01:13。
- source 结果：attempt 构造时选择一个固定 S 即可，5 处 owner 调用一致取 shards.len；prepare 仍按 P 个 workers 切 task。fresh S=2P/4P 可作为比替换 owner 架构小的 many-key 负载对照；P1 与 reuse 保留原 S。
- 正确性：各 pair 仍完整且 stable 顺序归组，跨 fresh key publication 不同不改变 total priority；S 取 P 的精确倍数形成原 bucket 细分。mask 只等价于非零 power-of-two S，rounding/mask 是另一个改动，不随手捆绑。
- 成本：更多 shard 可能拆开原同 owner 融合的 old/birth action，增加记录；更多 Vec/map 与保留 capacity，serial routing 不消失；并行 router 可能付 O(S×chunk 数)目录成本。一个 hot key 不会被拆，真实 many-key owner tail 尚未测。
- 证据：阶段研究材料 SHA-256 `245b22fabdbc…`；a2 固定 mod/index/merge 只读审查。均为分析，不是 AHash 负载统计或运行速度结论。

### 128｜stable producer-chunk routing 与 header 转置

- UTC 对照：研究区间：10-11 21:56；10-11 22:00。

- 研究区间：2026-10-12 05:56—06:00。
- 结果：把原 ordered Change 流分成连续 chunks，每 chunk 按 owner 稳定路由，再按 chunk 次序拼每 owner bins，可精确保留 fresh leaf 次序与 reuse signed actions，包括同 owner remove-before-birth。无需共享 map atomics 或改变 count owners。
- 成本：O(G)路由工作仅并行化，新增 O(SC) bins 初始化/header 转置及分配；root 计账 O(task 数+SC)，owner skew 不消失。unbudgeted 跨轮保留每 chunk-owner bin capacity 可达原 P 倍。
- 独立模型：10,000 个 case、3,923,544 个 event 检查通过；P16/G1600 构造原 owner capacity1600、chunk-owner capacity25,600。精确运行起止未归档，不能当 native 峰值。
- 证据：阶段研究材料、`model.py`、`results.json`；SHA-256 `1fa01004fbee…`。只读 source 及 descriptor 模型，无 Rust/仓库执行或速度结论。

### 129｜全部 live fresh keys 已选后的 stale heap bulk 清理

- UTC 对照：源码核验进展（截至记录，非精确起点）：10-11 22:05。

- 源码核验进展：截至 2026-10-12 06:05。
- 结果：若已选尽所有 fresh live count keys，剩余 heap entries 可判定全 stale；prepare 前并行 drain 可省逐项 heap-pop 修复，queue 工作由潜在 M log M 变为 M。AA 后续 birth 仍须正常 commit，不能因当前 empty 就提前结束。
- API 事实：Rayon v1.12 par_drain 保留 Vec capacity；dary_heap0.3.9 empty From<Vec>可复用该 capacity。作者 timed binary 确切解析版本尚未核验，来源/library 行为不等于当前测量收益。
- 状态：小范围源码控制，未实现或 benchmark，保留异常/ownership 顺序审查。

### 130｜以现有邻居目录缓存通过的 frozen-witness 判定

- UTC 对照：开始（截至记录，非精确起点）：10-11 22:05；阶段材料：10-11 22:07；10-11 22:08:17。

- 开始：截至 2026-10-12 06:05 已启动；精确开始秒未保存。
- 阶段材料：06:07 完成首份源码/证明与模型材料，06:08:17 核对现有文件。更广的过夜研究继续。
- 工作：审阅 a2b20f32 的 `Neighbors::record/group`，寻找不用新 bool、Change 字段或负缓存的早期过滤接口。先查询现有目录；只有 OLD-role group 缺席时，才在创建前查询真实冻结父 key。拒绝时不插入 group；通过后目录本身缓存判定。不同 BORN-only group 只在父 key 获准后创建。
- 结论：fresh 输出此前不在旧 token 平面，因此 BORN-only ID 与 OLD-role ID 不碰撞；此前未激活的低数值 reserved ID 也符合这个性质。selected AA 必须短路放行，reuse 绕过 gate。这个方案在已读源码契约下省掉此前额外 bool cache 或 born-only provenance 改写；尚未编译实现。
- 工作账本：oracle 调用为 G+ + E−，其中 G+ 是获准的本地旧 group，E− 是被拒绝的 record 次数。被拒绝 record 可跳过旧/新 group 创建、相关累加、压缩 append、Change 和后续路由/归并；matcher、已接受 starts、此前邻居预测和现有目录查询仍需执行。
- 适用边界：对长度原本合格、频次 c<T 的旧 witness，其正权物理出现数最多 T−1，且是跨该 batch 所有 tasks 的合计界。零权重副本和纯长度不合格的缺席 key 不受该界限制；不同被拒绝 key 数也不受该界限制。负判定不缓存，可能重复查询。
- 模型运行：2026-10-12 06:07:00.641804—06:07:01.167284。5,452 个有效结构 batches、13,133 个 owned-cut 事件通过，含 2,131 AA、4,250 低数值 fresh 输出配置；三种 gate 对照合计 300 个零权重 fragment 保留检查。精确 active counts 和完整 newborn coordinate 列表一致。套件中正缓存 oracle 调用 12,651，bool cache12,627；分组数分别 12,209、13,155，基线 13,190。AA 短路也计入 oracle 次数，不等于都进行了 hash lookup。
- 证据：阶段材料 阶段研究材料；独立模型 `check_model.py` 与 `results.json`，SHA-256 `b4ff103ec395…`。源码：[冻结提交 merge.rs](https://github.com/Yikai-Liao/tokenizers/blob/a2b20f329d1b4a0d39b5721f4a7a678a80d0d9b2/tokenizers/tk-train/src/trainers/bpe/merge.rs)。
- 限制：套件包含任意 compatible 结构批次，不全是实际 greedy prefix；未验证 initial-cap、Rust borrow/并发内存模型、故障注入错误顺序或性能。计数不代表真实语料命中率。

- 独立复核：2026-10-12 06:07:07 开始，06:12:18 完成报告；模型 06:10:32.817661—06:10:33.053774 运行。3,005 配置中 2,138 有非空 batch、867 无可选规则；3,640 jobs、9,720 records、740 AA、402 低 ID reserved 单例激活。源形 record-for-record 输出、active birth mass/Positions、stale 目录槽和 initial-cap 例外均通过。正常保留 230 零权重 birth records，过滤 2,777 records；其中 461 零权重、820 已过长度但全球低频、1,513 长度不合格。
- 复核反例：通用 born-group gating 会查 synthetic removed 而错删真实 birth；缺失 AA restore、reuse 角色碰撞、跨左右 side 共享判定均不安全。高频但长度不合格的缺席父 key 再次否定通用<T 次数界。模型 9,190→6,561 group 和 647 次正缓存命中仅为套件统计。
- 复核证据：阶段研究材料、`check_positive_cache_gate.py`、`results.json`；源 SHA-256 `65ec7e65b2de…`。两固定提交 ea1/a2 的 merge.rs 同 blob `f9d6ac21d67038087ab9e96409fd0b9a3a0b6367`。模型选择合法 greedy priority prefix，但任意固定 span 快照不全声称来自完整 production 历史；没有仓库运行、编译或速度结论。

### 131｜fresh 完整训练 prepare 工作摊销证明

- UTC 对照：开始：10-11 22:05:55；结束：10-11 22:19:20。

- 开始：2026-10-12 06:05:55。
- 结束：2026-10-12 06:19:20。
- 工作：只读 a2b20f32 的 merge/index/positions/corpus，按物理 edge version 和唯一 cut owner 分析 selected postings、stale 坐标、AA overlap 及 birth pushes 的全成功前缀账本。
- 证明结论：一个 fresh attempt 内每个成功提交前缀，原始 selected Positions 解码 D≤3R，当前 matcher 调用≤4R，出生 push B≤2R；R 包括零权重词的物理收缩。被拒绝的 stale/AA-overlap edge version 可计入使其失效的收缩至多两个外邻 cut；selected cohort 只消费一次。当前 restart-aligned tasks 没有另加块前缀重复解码。
- 紧性与限制：零权重固定字母表 fixture 及全正权重、strict cap2/增长字母表 fixture 均可逼近 D/R=3、matcher/R=4。后续单 batch 可能有大量 stale 坐标，只能向早先收缩摊销，不能据此否定尾部延迟。AA accepted 数组的二次 matcher、最多两次全数组 binary search、Directories 初始化、weight-region 查询、分配/排序和 task 开销分别记账，不能把坐标线性界改说总 prepare 时间线性。失败末轮和 reuse 重启不纳入成功 R 分母。
- 验证：独立模型 2026-10-12 06:18:58.611734—06:19:02.615071 运行。20,000 随机案例、75,223 个成功前缀检查通过；合计 R161,037、D185,107、matcher210,686、B99,293。覆盖 17,747 AA 批次、9,581 多规则批次、455 个先被 AA overlap guard 跳过的 stale 条目、10,926 个双方 selected 共享 cut 并核对唯一左侧 emitter。
- 证据：阶段研究材料，SHA-256 `f58d3679fe2e…`；模型`check.py`，SHA-256 `db377de7e3bf…`；`result.json` SHA-256 `d4b0b1ac122c…`。
- 状态：阶段证明和独立数学模型完成。未执行/修改仓库，没有 codec、分配、故障/reuse 或原生性能验证。

### 132｜高频 pair 图的动态 incidence 证书

- UTC 对照：开始：10-11 22:08；结束：10-11 22:12。

- 开始：2026-10-12 06:08（阶段记录精度为分钟）。
- 结束：2026-10-12 06:12。
- 工作：从完整当前高频 pair 图建立 induced matching 与 edge-domination 下界，补充普通 matching 每两个 witness 仅保证一次高频行动的局限；核查 loop/AA、length gate、reserved 和动态 discharge。
- 结论：induced matching 中每个未被 selected input 首次触及的 witness 可保证一次≥L 后续选择。先做 matching、再扫全部高频桥边剔除冲突，两个 key passes 和 O(A)状态可构造安全子集；每条规则用两个 input-ID lookup 更新。该“首次触及”不等于 token 永久退休。仅采样 witness 而漏查 bridge 会不安全。
- 强度/成本：256 初始字节 IDs 的 loopless 上限 128、含 loop 上限 256，不能覆盖已核实的 49,744 初始 action horizon。fractional edge-neighborhood credits 可更强，但动态更新需要 O(E)状态和 incidence visits。构造扫描、证书强度、后期剩余 horizon 的收益均未测量。
- 验证：独立模型 2,200 带权配置，含 1,527 mid-run snapshots、880 loop 图、7,684 动态 frontiers；2,092 小图核对精确 edge-domination 下界。保存结果时间 2026-10-12 06:11:06.434978；未单独保存进程起止。包括 AA、zero、cap、初始 admission 例外及低 reserved 输出。验证适用模型条件，没有完整 trainer 性能结论。
- 证据：阶段研究材料、`check.py`、`results.json`；源 SHA-256 `d28ba6ae89f4…`。主来源：[Yannakakis–Gavril, Edge Dominating Sets in Graphs](https://doi.org/10.1137/0138030)；原论文 simple loopless 结论不直接覆盖 AA，AA 依据本阶段的独立 incidence 证明。

### 133｜早期 gate 与稀疏 routing 的组合成本

- UTC 对照：开始：10-11 22:12；结束：10-11 22:16；10-11 22:57。

- 开始：2026-10-12 06:12。
- 结束：2026-10-12 06:16（阶段材料所记分钟精度；06:57 补核保存产物）。
- 工作：核算 positive gate、owner 过分片和 chunk routing 组合；用每 chunk 临时 HashMap<owner,Pending>，在 worker 上转换为只含 occupied bins 的 compact Vec，再由 root 按 chunk 顺序转置。
- 结论：语义可组合但收益不相加。gate 可能删除原本能摊销并行 routing 的体量；更多 S owner 会增加 dense S×C 初始化和拆散 same-owner action fusion。稀疏方案 root O(S+C+Z)，worker 仍付 map entry 与实际 capacity 迭代/compact 成本；任意 bin 顺序不改变每 owner 跨 chunk 的原始 action 顺序。
- 内存/成本：不保留历史空 bin 可避免 S×C 矩阵，但临时 map 分配、action 元数据和 birth descriptors 仍存在；密集 Z≈SC 时稀疏更不利。持久 bin 容量按历史各格最大值累积，可从 O(G)放大到 O(PG)；无免费 buffer pool 假设。
- 支持模型：6,000 descriptor 案例、1,488,759 events 通过 dense/sparse 逐 owner action 顺序及 fusion/Z 账本检查。未保存精确运行起止，未重跑补时间。虚构 P=C64、S256、A512 示例只是 header 算术，不是实际 BPE 选择率或耗时。
- 证据：阶段研究材料、`model.py`、`results.json`；模型 SHA-256 `b6e973ff1f3d…`。源级与独立 descriptor 数学研究，无 Rust/仓库执行或性能胜者。

### 134｜对照 Re-Pair 与 exact BPE 的 occurrence 摊销文献

- UTC 对照：开始：10-11 22:19:36；结束：10-11 22:25:45。

- 开始：2026-10-12 06:19:36。
- 结束：2026-10-12 06:25:45。
- 工作：直接检查 Bille–Gørtz–Prezza 2016 §3.3/Algorithm1、2017 §2/§3.2，Zouhar ACL2023 最新版 v3，Briggs–Torczon sparse sets、Navarro 数组初始化及 Rust/Rayon 官方文档；另交叉检查 ESA2026、BatchBPE 和 Faster Superword Tokenization。
- 结论：局部边界更新和惰性 occurrence 删除的摊销是已有方法；D≤3R 应描述为当前源码的 fresh 唯一 cohort、AA、compatible cut owner 和权重细节核验，不主张新发明。2017 文纠正早期 remove 常数时间说法；weighted frequency 不等于物理 posting 长度，零权重直接阻止照搬 F/L 半满清理阈值。
- 迁移限制：Re-Pair 原地不稳定聚类可破坏 child 空间顺序；ACL formal BPE 的 nonoverlap gain 与当前 AA raw count 不同；BatchBPE 不同 merge 序不能作为 tie-identical trainer 证据。数组初始化文献不允许读取未初始化 Rust 整数。当前目录已是 bounds+key 回查的 sparse-set 式验证，新增 epoch 没有所述收益，生命周期才是继续审计的目标。
- 证据：阶段研究材料 SHA-256 `0a1e6a4bbfe8…`；阶段研究材料 SHA-256 `1632def25ccf…`。主来源：[2016 Space-Efficient Re-Pair](https://arxiv.org/pdf/1611.01479)、[2017 Practical and Effective Re-Pair](https://arxiv.org/pdf/1704.08558)、[Formal Perspective v3](https://arxiv.org/html/2306.16837v3#S4)、[Briggs–Torczon sparse sets](https://www.dcs.gla.ac.uk/~pat/ads2/papers/sparseSets.pdf)。
- 状态：主文献比较阶段完成。没有仓库运行/修改、新的 native 测速或从论文迁移的性能倍数。

### 135｜prepare 目录的 split 局部与 attempt 局部生命周期

- UTC 对照：开始：10-11 22:20；结束：10-11 22:21；10-11 22:22:14。

- 开始：原始开始秒未保存；阶段材料标记 2026-10-12 06:20。
- 结束：06:21 完成源码细化，06:22:14 核对当前报告。
- 工作：阅读 Rayon v1.12 的 map_init、Splitter 和 LengthSplitter，核对当前两份 Vec<u32>目录的 sentinel 填充；比较 with_min_len、per-batch 和 attempt-local ThreadLocal、scratch pool。
- 结论：map_init 状态是 split-folder 局部，不是固定每线程一份；但普通无 steal 路径只产生 O(P)份，不能把 O(TV)最坏界当实测典型放大。with_min_len 可限定每批状态数，却不消除跨轮反复初始化。
- 条件方案：attempt-local ThreadLocal<RefCell<Directories>>可复用已有 stale-index 验证，单调 ID 域下 sentinel 写入总量≤8H×V_final 逻辑字节，H 为实际参与线程 slot 数。类型最小可见性、参数传递、新 direct dependency、非重入契约和输出前 drop 均有成本。无须新增 Corpus/Engine 字段；保留缓冲峰值、Vec 增长 copy、TLS 查询和 RefCell 检查另计。
- 源码限制：enclosing lock 里的 Rayon1.12/thread_local1.1.10 不能证明作者 binary 实际解析版本，也不使 thread_local 自动成为 tk-train 直接依赖。历史 scratch 数据没有隔离当前目录 site。
- 证据：阶段研究材料 SHA-256 `ba3d324c50a6…`；[Rayon v1.12 map_init 源码](https://github.com/rayon-rs/rayon/blob/v1.12.0/src/iter/map_with.rs#L443-L547)、[ThreadLocal 官方 API](https://docs.rs/thread_local/latest/thread_local/struct.ThreadLocal.html)。只读分析，无编译或原生 benchmark。

- 06:28 新推导：即便每 batch 恰好一份状态，重复空目录也有不可忽略的逻辑初始化下界。K 个 append fresh IDs、初始域 A0、batch 上限 C=256，令 n=ceil(K/C)、r=K−C(n−1)，最小域长和为 nA0+nr+C·n(n−1)/2；两份 Vec 对应两倍 u32 元素填充。故出现Ω(A0K/C+K²/C)，singleton 可为Θ(A0K+K²)。源码 prepare_identity 先增长域再 prepare.resize；有效正频 candidate 有至少一个 job。reserved 零增长不计入 K 但增加工作。该推导已获独立数学审阅，材料整理中；不是 CPU 时间或实际 batch 分布测量。

- 06:29 完成精确补充：阶段研究材料 SHA-256 `2478af29b78c…`。下界的抽象最小排布为首 batch 仅 r 个 append、其后每批 C；不能把 full-first 排布的更高和误当普遍下界。singleton 精确为 2KA0+K(K+1)个 u32 初始化；attempt-local 上界 2H V_final。例示的 34.128 MB/8.704 GB 都是虚构参数算术，未观察实际语料。逻辑 sentinel 初始化不等于每条机器 store 或总内存 traffic，allocator copy/容量仍另计。

### 136｜继续核对远端代码基线

- UTC 对照：时间：10-11 22:18；补充核验：10-11 22:37:52。

- 时间：2026-10-12 06:18（未提供精确秒）。
- 工作/结果：支持的分支比较再次确认 head 仍为 a2b20f329d1b4a0d39b5721f4a7a678a80d0d9b2。没有新源码变化可归入本轮结果。

- 补充核验：2026-10-12 06:37:52 远端比较仍为同一 a2b20f32 head，无新提交。

- 补充核验：2026-10-12 06:57:25 支持的远端比较返回同一 head，ahead/behind 均 0、无新增 commit。

- 补充核验：2026-10-12 07:17:09 远端分支仍与 a2b20f32 相同。

- 补充核验：2026-10-12 07:42:26 远端 head 仍为 a2b20f32。

- 补充核验：2026-10-12 08:00:08（UTC 日期已到 10 月 12 日）远端 compare 仍为相同 a2b20f32。

- 补充核验：2026-10-12 08:18:57 分支 compare 仍相同 a2b20f32。

- 补充核验：2026-10-12 08:39:57compare 仍相同 a2b20f32。

- 09:02:47 持续核验发现分支较 a2b20f32 前进 3 个提交，head 为 a583a0b4025495ef0006a6adea84c6139cb7b295；详细初始 ID 适配审查另列阶段 176，较早模型不自动改称新 head 验证。

- 09:22:05 分支较 a583a0b4 又 ahead1，新 head5c7ef814f55da46cd7c820173cc72053fe092d74，新增 Affix 初始化/validation evidence；独立证据核对见阶段 182。

- 09:40:30 又观测一份 evidence-only 提交 890ea1f58e65f8d8503f7cd3fc97ad1651eaf284，parent5c7ef8，207 files 全部 experiments；production 未改，right-token prefetch 作者实验另列阶段 186。

- 09:59:40.841 再次 compare，head 仍 890ea1f58e65f8d8503f7cd3fc97ad1651eaf284，无新增提交。

- 10:17:23.604 branchAPI 重新核验，head 仍 890ea1f58e65f8d8503f7cd3fc97ad1651eaf284。

### 137｜trusted AA range 的最小迭代器接入点

- UTC 对照：时间：10-11 22:26。

- 时间：2026-10-12 06:26（聚焦源码材料完成；未保存独立开始秒）。
- 工作：细化已有 trusted accepted-index 方案在当前 Source、prefetch 和 record_neighbors 中的接入，不另造 preparation 表示。
- 结论：现有 AA Source 载荷可改为 accepted snapshot 中的 Range；共享迭代器临时 yield (ordinal,Match)，保留一个 weight/neighbor/starts 处理 loop。AA 邻居使用完整 accepted 向量中相邻 ordinal，跨 task 边界仍正确，可删除 K 次重复 matcher 和最多 2K 次 binary search 的 logK 因子。
- 边界/成本：保留 corpus.resident 的 checked u64→usize 转换；只重复此前已验证的 fixed-span 端点计算。仍保留 8K 字节 accepted u64 向量和 Job.starts，24 字节瞬时迭代 item 不会收集成 24K 数组。初始 serial AA acceptance 及第二份 starts 流量没有被这个小改动消除。旧 private 任意 Slice fixture 应移到 snapshot 输入验证边界；reuse 不适用。
- 证据：阶段研究材料 SHA-256 `2e690006dace…`；[a2 固定 merge.rs](https://github.com/Yikai-Liao/tokenizers/blob/a2b20f329d1b4a0d39b5721f4a7a678a80d0d9b2/tokenizers/tk-train/src/trainers/bpe/merge.rs)、[itertools0.14 Either](https://docs.rs/itertools/0.14.0/itertools/enum.Either.html)。
- 状态：已有控制的聚焦集成审查，非新算法发现；无仓库编辑、编译、layout 测量或 benchmark。大规模 parallel AA 方案单独记账，不能重复计入这些节省。

### 138｜冻结 key 集合下把分组 atomic losses 移入 prepare 尾部

- UTC 对照：开始：10-11 22:22；首份阶段材料结束：10-11 22:26；10-11 22:26:07；10-11 22:31:38。

- 开始：2026-10-12 06:22。
- 首份阶段材料结束：06:26；独立 source-shaped 复核 06:26:07 开始，06:31:38 完成。
- 工作：组合 positive-directory membership-only gate 与分组 AtomicU64 扣减，检查键集合/数值的不同可见性要求，以及低于 floor 后的唯一 retirement ticket。
- 结论与修正：先前“prepare overlap 必需另一份 numeric snapshot”的笼统表述过强。只要 map 结构及 KEYS 在 join 前不变，gate 仅 contains_key 并保留 AA override，已完成 task 可扣减值而不改变其他 task 判定。corpus 仍须冻结；不得同时 selection、value 阈值判断或 birth publication。
- 退休机制：成功 atomic 扣减唯一跨越 old≥F、new<F 时生成一个 Pair ticket；join 后只删除/路由 W 个 ticket，不需要扫全 map。fresh 整个 attempt 的 W 总量等于初始 keys+admitted birth keys−selected takes−最终 keys。W 可与全部 loss 数相同，并非普遍低成本。worker 过滤 removal-only groups 后才可实际减少 root 路由。
- 成本/边界：组完成后仍可能第二次 map lookup；CAS 重试无固定竞争界，per-job 退休 Vec、atomic64 目标支持、hot cacheline、保留 Vec capacity 和 birth owners 均仍在。全 local loss 与 global birth projection 不能叠加扣减。reuse 保留原有 ordered 算术；故障 attempt 丢弃不意味着复现旧错误顺序。
- 支持模型：15,000 scalar interleaving 案例、264,845 membership reads、17,138 唯一退休通过；value-threshold gate 会误拒 4,789 次读取。两份 ABC 在 F2 的定向例中，首 task 扣 BC2→1 后若按 value 判定，第二份 XC birth 被错误丢弃。结果未保存精确进程起止，不能把材料区间当运行时。
- 证据：阶段研究材料、`model.py`、`results.json`；脚本 SHA-256 `e17679bb210d…`。无仓库执行/改动、Rust 内存模型或性能验证。

- 独立 source-shaped 模型：2026-10-12 06:30:06.169651—06:30:06.543824 运行；4,109 schedules、15,189 jobs、1,451 穷举定向完成顺序、400 CAS retries、5,116 唯一 tickets 和 10,026 有序 birth-fragment 检查通过。17,119 membership reads 包含 415 次 value 已<F 的读取；1,656 AA schedules、2,160 缺失 selected-AA loss groups 按预期忽略。
- 新负例：在 CAS closure 里生成 ticket 可因重试重复；成功后再 reload 也会把别人的跨阈值算成自己的；birth 按完成顺序收集会破坏 concat 的空间排序。标量正确不允许双重 loss 扣减或改动 birth 顺序。
- 独立证据：阶段研究材料、`check_overlap.py`、`results.json`。脚本 SHA-256 `1347e2d0bf69…`，独立 recorder 依赖保持 `65ec7e65b2de…`。任意固定 span 快照不全是完整 greedy 历史可达；无 Rust memory-model 或 native 性能验证。

- 06:41 文档更正：在较早阶段研究材料与阶段研究材料顶部加入有日期的 superseded 说明，完整保留历史文本并指向本阶段独立验证。只修正对 membership-only reader 过强的限制；numeric gate、提前删 key 及未闭合 selection 仍不安全。两份报告更新后 SHA-256 分别为`167b2ac07ed5…`、`724180a1a6a7…`，原数学模型不变。

- 07:45 进一步文档更正：阶段研究材料原 proposed-benefits 第 4 项仍含“所有 gate 先完成才能扣 loss”的旧笼统表述，已加顶部日期说明并保留原文。明确只对 structurally frozen membership 放宽；count-sensitive gate 或 pending-birth 初始数值界仍不能读扣减后的 cell。更新后 SHA-256 `ec55d82306e4…`，原实验数据及其来源不变。

### 139｜单调数据流与可串行化框架的协调边界

- UTC 对照：开始：10-11 22:32:04；结束：10-11 22:40:15。

- 开始：2026-10-12 06:32:04。
- 结束：2026-10-12 06:40:15。
- 工作：对照 CALM、monotone dataflow、conflict serializability、IVM、invariant confluence 及 semantic commutativity 的主文献，审查冻结 membership/atomic prepare 与 positive gate 的协调/发布义务。
- 结论：冻结闭合 key 集合可与 atomic value 变化共存；标量投影只有在唯一 cut 归属和可表示性成立后才可交换。monotone floor 跨越不表示精确 count/Positions 已完成；某个 serial 执行等价也不等于指定 greedy rank/ID 顺序。
- 反例：ABC×2、BC×8、AB×18、DE×9 中未完成 BC loss 会翻转下一 winner；ABC×5、ABD×5、DE×4 中先完成旧 loss 而隐藏 birth 会错选 DE。checked signed reuse 净和归约也不能保留逐步 overflow 行为。均为代数/源码推导，未跑 trainer。
- API/调度限制：当前 concat 只接受 Leaf，故连有序 tree concat 都不是现有 API 的闭合归约。producer-local 完整 birth reduction 可提早私下执行，但不删除 grouping/publication，新增 staging/scheduling 且 singleton AA 没有跨 producer 重叠空间。
- 证据：阶段研究材料 SHA-256 `d7c3632f9e92…`；阶段研究材料 SHA-256 `abc5e9a65e69…`。主来源：[DBSP](https://www.vldb.org/pvldb/vol16/p1601-budiu.pdf)、[CALM formalization](https://arxiv.org/pdf/1012.2858)、[Invariant Confluence](https://arxiv.org/pdf/1402.2237)、[Scalable Commutativity revised proof](https://arxiv.org/pdf/1809.09550)。
- 状态：阶段主理论比较完成；无新 scratch 模型、仓库执行/修改或 benchmark，未把理论迁移为速度保证。

### 140｜prepare 完成后重叠现有 apply 与 commit 的源码接入

- UTC 对照：时间：10-11 22:36。

- 时间：2026-10-12 06:36 完成聚焦源码/API 审查；独立开始秒未保存。
- 工作：重访此前阶段重排设想，核对当前 owned Changes 与 corpus 应用的依赖；按原 Job 顺序 mem::take Changes，用 scoped rayon::join 分别 consume Jobs 执行 apply 及现有 index.commit。
- 结论：现有 commit 不读取 corpus，apply 不读取 index，成功路径可保持数据依赖与 reuse action 顺序，且不要求新 Apply 类型、count 表示或 atomic 协议。下一次 selection 和成功返回必须仍在外层 join 之后。
- 错误/成本：普通 commit Result 错误在 join 等待所有写入后返回，符合当前 fixture；apply panic 却可能已与 index publication 重叠，要求丢弃 attempt，不能声称 panic 观察等价。新增 O(J)header 抽取与 join，共享线程池竞争，starts 和 commit 临时数据的峰值生命周期重叠。理想节省最多 min(apply,commit)，不是 grouping/counting 工作删除。
- 证据：阶段研究材料 SHA-256 `515febbc73a1…`；[Rayon1.12 join 契约](https://docs.rs/rayon/1.12.0/rayon/fn.join.html)。固定 a2 源码只读；未执行、编译、修改仓库或测时。

### 141｜成功 fresh attempt 的完整工作量账本

- UTC 对照：时间：10-11 22:39。

- 时间：2026-10-12 06:39 完成阶段综合；独立开始秒未保存。
- 工作：合并物理收缩/codec/descriptor 摊销、trusted AA、attempt scratch、weight-region、commit/heap 及初始化/输出成本，检查是否能得出整体线性结论。
- 结论：在 expected-hash 和 fixed-width 模型下，两项小控制后的 prepare 可写为 O(R+HV+M log(W+1))，其中 W 是压缩权重 regions、M 是实际 cache miss、M≤min(R,TW)且可达 R。故不能宣称整个 prepare 或 engine 线性。跨词不同 weight、每 job 单次匹配的构造保留 logW 查询项。
- 其余成本：commit stable fragment sorting、每 batch shard 目录扫描、heap logQ 操作仍在。fresh lazy priority correction 次数≤正 count decreases≤G 只是次数界；不消除每次 heap 成本。初始原文/rows 排序/slots 填充、dense pair states、词表字符串构建另计，不能全塞入 physical edge N 或 R。
- 证据：阶段研究材料 SHA-256 `7ab6a21ec668…`；固定 a2 源码与前述审查的综合。无新增仓库执行、实现或性能结论；该材料不是最终报告。

### 142｜使用标准 Vec 提取 API 整理 fresh dead heap

- UTC 对照：阶段材料：10-11 22:42；10-11 22:43。

- 阶段材料：2026-10-12 06:42 开始标记，06:43 完成；未保存精确开始秒。
- 工作：核对 Rust1.87 Vec::extract_if、Rayon1.12 par_bridge 及 dary_heap0.3.9 From<Vec>，研究不分配第二个 Candidate 数组的 dead payload 回收。
- 结论：quiescent fresh 边界可 move heap backing Vec，以冻结 membership 提取 dead 候选，在 bridge 外并行 drop，再用同 Vec heapify。完整过滤、survivor 搬移仍被 iterator mutex 串行化；只有 payload 析构并行。Q 次 expected hash 查询、D+O(P)锁、serial heapify、保留 capacity 均须付费。
- 前提/边界：D=Q−live_key_count 依赖 fresh 唯一 cohort，不直接适用 reuse。D≥L 且 D>0 触发使 Q≤2D，可把线性整理向独有 dead 候选摊销；早 drop 不自动遍历剩余项，panic 只支持丢弃 attempt。当前 ancestor stable toolchain 意图不能验证作者已安装 Rust≥1.87。
- 选择：sequential retain 更简单；全 dead 情况用既有 par_drain 而非 membership 过滤。没有证据证明 bridge 更快，也不宣称 capacity 保留等于 RSS 下降。
- 证据：阶段研究材料 SHA-256 `2ccd1b7cce30…`；[Rust1.87 extract_if 源码](https://github.com/rust-lang/rust/blob/1.87.0/library/alloc/src/vec/extract_if.rs)、[Rayon1.12 par_bridge](https://github.com/rayon-rs/rayon/blob/v1.12.0/src/iter/par_bridge.rs)。只读 API/source 审查，无仓库执行/编译/benchmark。

### 143｜producer 级并行完整 birth 归并

- UTC 对照：源码/可行性阶段：10-11 22:42；10-11 22:46；10-11 22:43:38；10-11 22:48:56。

- 源码/可行性阶段：2026-10-12 06:42—06:46；独立复核 06:43:38 开始，06:48:56 完成。
- 工作：用现有 replacement 与有序 task 分组，producer 局部 pack 全部 fragments，Rayon 稳定 parallel sort 后按 key 做 disjoint reduction；每个完整 key 只 concat 一次原始 leaves，再把 complete Candidate 交结构 owner。
- 结论：已知唯一 canonical emitter 使全部同 key fragments 落在一个 producer；B=1 也可暴露 sort/跨 key 并行。root birth 路由有条件从 F 降至 K_live，但 producer 局部 F_r packing 仍串行、每个 key 归并不可拆，且普通 count-map 结构发布仍需额外 join 或串行插入。并未做到 ownerless 或只处理 K 条输入。
- 内存/边界：stable parallel sort 全长 scratch、input 与 flat 容量重叠、root child descriptors 和 unindexed 输出 collection 都须计入。concat 结果 Root 不得送回既有 Leaf-only concat 路径。numeric-min 新 endpoint 不是正确 producer 归属，普通 append IDs 也有反例。AA/zero、源顺序和 retirement ticket 保持独立。
- 初步独立验证：06:48 收到 1,808 配置、1,350 非空 batches、450 AA、165 双新输出 fragments、91 保留 zero-weight 记录通过的摘要；等待最终报告/哈希，不将初步数值冒充已复核最终结果。
- 证据：阶段研究材料 SHA-256 `9b14e837455a…`。只读 source/API 审查；无 native 编译或性能证明。

- 最终独立结果取代初步摘要：模型 06:47:26.528785—06:47:26.726236 运行；1,809 配置、1,351 非空 batch、458 空配置，3,440 tasks、2,992 birth fragments、166 双新 fragments、450 AA；2,133 complete candidates/concat，其中 408 Roots、91 个 zero fragments 在 live children 保留，384 complete groups 低于 floor。physical-left 唯一 producer 与 Pair→bucket 不变性通过。
- 验证证据：阶段研究材料、`check_producer_groups.py`、`results.json`，脚本 SHA-256 `cb42118e4499…`，依赖`1347e2d0bf69…`。local floor、split key、completion/unstable order、min-ID owner、Root 再 concat、overflow 和 premature publication 均保留负例。无 native API 或性能证明。

### 144｜ordinal 装饰与不稳定 parallel sort 的 scratch 权衡

- UTC 对照：开始：10-11 22:48；结束：10-11 22:50。

- 开始：2026-10-12 06:48。
- 结束：2026-10-12 06:50。
- 工作：审查以(Pair,原始 fragment ordinal)装饰排序 key，再只按 Pair 分组的顺序证明和 Rayon1.12 move-only unstable sort 接口。
- 结论：可复现 stable Pair sort 的同 key 叶顺序，去掉全长 sort scratch；明确的私有 56B 估计 tuple 比把 bucket 偷偷改作 ordinal 更清晰。相比 48B 无 ordinal tuple 多 8F 逻辑字节和 F 次写，conversion/packing 仍付费，所有 layout 数字未编译测量。
- 权衡：唯一装饰 key 不能使用重复相等项 collapse，但不等于所有输入自动 n log n 退化，既有有序 runs 仍可能受益。串行 producer packing、concat、unindexed collection 及结构发布 join 不变；只证明顺序和 scratch 交换，不证明速度改善。
- 证据：阶段研究材料 SHA-256 `68d8cd107fd4…`；[Rayon1.12 unstable sort 源码](https://github.com/rayon-rs/rayon/blob/v1.12.0/src/slice/mod.rs#L697-L741)。无仓库/第三方程序执行、安装或编辑。

### 145｜多项控制组合的对抗性审查

- UTC 对照：开始：10-11 22:48:50；结束：10-11 22:56:49。

- 开始：2026-10-12 06:48:50。
- 结束：2026-10-12 06:56:49。
- 工作：检查 positive gate、目录生命周期、trusted AA、冻结 membership 的 atomic losses、producer grouping、speculative floor/fallback 和 quotient recovery 的联合契约与重复计益。
- 确认：same-batch 精确 K0 在数值跌破 floor 后保留仍安全；TLS 借用须在 nested Rayon 前释放；完整 local loss 与 global projection 不可叠加；birth task 顺序与 atomic 完成顺序分离；target 成功只能在完整 commit 后判断。
- 新反例：strict cap2 的 ABC×2、AB×1 仍需先应用 BC loss，不能先丢 birthless group；若 retirement 使用另一个更高 threshold，可删仍达到实际 floor 的 BC。依赖 gate 表示虚拟 fragment 边界的 reweighted in-place quotient 中，batch 起点就错误的 positive membership 会制造 ghost birth 或过扣；这不推翻同 batch 保留真实 K0 的安全性。
- 独立 tiny 检查：2026-10-12 06:54:04.123671—06:54:04.123738 运行，6 种 gate/decrement interleavings 通过，含 2 次 numeric floor 跨越后的 gate 读取；附定向 scalar/phase algebra checks。它不是 trainer 或完整 BPE oracle。
- 证据：阶段研究材料 SHA-256 `039fb99476b2…`；`check_contracts.py` SHA-256 `bda21f2cf2f7…`，`results.json` SHA-256 `772db03bb7de…`。无仓库执行/编辑/native 验证。

- 06:50 发现早期阶段研究材料 §§5–6 的通用 Result-retry 措辞过宽；06:51 完成有日期的§12 更正并保留历史文本。只允许 coherent committed underfill 按已证明恢复路径处理，或既有 original-input active-reuse restart。算术、codec、allocation/capacity 等真实错误传播并丢弃 attempt；不能从任意 partial mutation live-reindex，也无 panic/OOM 恢复承诺。
- 算术细化：必需初始 checked pair counts 之后，valid fresh 每 key 的 u64 频次受唯一初始 cut family 限制；真实后期 frequency 错误说明前提/实现被破坏，不是 trial 可掩盖错误。cross-key 合计、ID 容量、编码、资源错误和 reuse 另算。§9 family upper bound 在完整 coherent scalar snapshot 下仍兼容 cap 排除和低 reserved 首次激活，无新反例或模型运行。
- 更正证据：阶段研究材料 当前 SHA-256 `7fe5f817e097…`；引用先前 4,209 early-abort、1,670 family traces、168 mid-run 模型，不把旧结果计作本轮新执行。

### 146｜权重 region 查询的剩余对数项

- UTC 对照：开始（截至记录，非精确起点）：10-11 22:49；结束：10-11 22:50。

- 开始：截至 2026-10-12 06:49 已启动；精确开始秒未保存。
- 结束：2026-10-12 06:50。
- 工作：检查现有(weight,end)cache 与 whole-array partition_point，审查 per-job index、successor test 和 galloping bracket 加标准 slice::partition_point。
- 结论：一个额外 job 局部 index 可使后继定位达到 O(log Δ)的 gap-sensitive 比较界，不需新坐标 buffer 或 Positions 重数；保留 one-region/unit-region 原有 O(1)捷径，unit shortcut 索引需要明确更新/存储策略。不得跨不同 job 继承单调下界。
- 反例/边界：每个独立 job 首次仍可付 O(log W)，单匹配 job 保留Θ(R log W)全局族；固定 b 步线性扫描再搜整个后缀在规则间隔 b+1 族仍Θ(m log W)，纯线性 cursor 遇大 gap 更差。galloping 降低分布敏感项，不解决全部 prepare 线性问题。
- 证据：阶段研究材料 SHA-256 `5782ba87b7e4…`；固定 a2 corpus/merge 与 Rust1.87 slice 源码。无仓库运行、实现或测时。

### 147｜AtomicU64 可移植性与上游 target 维护成本

- UTC 对照：开始（截至记录，非精确起点）：10-11 22:57；结束：10-11 22:59。

- 开始：截至 2026-10-12 06:57 已启动；精确开始秒未保存。
- 结束：2026-10-12 06:59。
- 工作：核查 std AtomicU64 与 wasm32/Rust target 条件、当前 manifest 和 CI 范围、portable-atomic1.15 依赖及 fallback 锁实现，比较维护成本。
- 结论：无条件 AtomicU64 引入当前局部源码没有要求的 64-bit atomic 能力，但不能误说所有 32-bit 或 wasm32 都失效。普通 Rust wasm32 配置提供 64 位 atomic 类型；共享线程 runtime 是另一个问题。没有完整依赖/target 实际 build，故只能称潜在支持限制，不能称已证实上游 target 编译回归。
- 成本：portable-atomic 可维持一个 trainer API，但增加 direct dependency，在某些平台用 global lock fallback，绝非免费 alias。现有 owner-grouped u64 仍是最简单统一控制；AtomicUsize 不能保持 32-bit 目标上的 u64 计数域。当前包含 workspace 的 CI/lock 不证明 excluded tk-train 的独立版本/target 覆盖。
- 证据：阶段研究材料 SHA-256 `eae64e0ec6ab…`；[Rust atomic portability](https://doc.rust-lang.org/std/sync/atomic/#portability)、[Rust1.87 wasm target](https://github.com/rust-lang/rust/blob/1.87.0/compiler/rustc_target/src/spec/base/wasm.rs)、[portable-atomic1.15](https://docs.rs/portable-atomic/1.15.0/portable_atomic/)。只读审查，无编译/运行/源码修改或性能结论。

### 148｜以高频 key 死亡计数维护两标量 horizon 下界

- UTC 对照：开始：10-11 22:49:29；结束：10-11 22:59:23。

- 开始：2026-10-12 06:49:29。
- 结束：2026-10-12 06:59:23。
- 工作：独立审查完整 directed high-key 集合大小 U 与保守 death 计数 E，M=max(0,U−E)及固定 incidence 容量 D 的点态 floor 证书；比较更紧 directed closed-star 分母和 induced matching。
- 结论：完整计入原始 key 死亡、允许额外保守 death 时，ceil(M/D)安全下界未来≥L 规则数。更紧 D*按 deg(u)+deg(v)−1−reverse_high（loop 用 deg(u)）计算，保持两个持久标量。此下界指 actions，reserved 零 growth 要求 H=K+Z，不能只比 K。
- 重要否证：证书与 remaining horizon 的差不单调。无重复语料构造 U263,D8 在一次 AB16 后，M228→211、bound29→27，而 horizon 只减 1。已认证完整 horizon 应锁存证明，不因保守统计变弱撤销；未认证前仍可等待更强机会。
- 其他负例：漏记 L>F 的跨 L 下降、用 AA accepted mass 代替 raw selected frequency、directed/undirected 混用、长度不合格 witness 及 unsigned wrap 均不安全。初始 256 字母表仍限制证书覆盖，不能据更紧分母保证 50k horizon。
- 验证运行：2026-10-12 06:58:04.134917—06:58:06.632697。1,463 完整历史、5,872 规则、9,607 snapshots（5,235 mid-run）、45,102 frontiers；150 次已认证 raise-floor continuation 精确匹配 base 规则/counts/IDs/paths/vocab。46,985 high-death 观察含 16,582 post-snapshot key deaths，检查会重复访问同历史规则，不等于独立语料数。
- 证据：阶段研究材料、`check_incidence.py`、`results.json`；源 SHA-256 `ffa1e45553ff…`，依赖`100bc30cc659…`。独立数学模型，无仓库或性能验证。

### 149｜精确 bulk stale-heap repair 的主文献比较

- UTC 对照：开始：10-11 22:59:56；结束：10-11 23:02。

- 开始：2026-10-12 06:59:56。
- 结束：2026-10-12 07:02。
- 工作：阅读 LaB-PQ、selectable heaps、heap selection 和 lazy upper-bound 认证主文献，核对当前 owning OctonaryHeap 的 exact tie、compatible prefix 和 blocked suffix 回插义务。
- 结论：tournament 批量修正 d 个叶的祖先并集有 O(d(1+log(N/d)))真实共享工作界，但它比较的是 eager 修正全部 changed keys；当前 lazy 仅修正遇到的 r 个 stale 候选，d 不能由 r 界定。冷 survivor 构造在目标前无需修正 n 个变动 key，批量 Sync 反而全付费。
- 源码成本：stable pair→leaf handles、renew flags、O(N)额外 metadata、changed-key 发现和动态 relocation 不免费。标量 outside-bound 可认证 exact 已查 prefix，但 owning extraction/回插、排序及首次 compatibility stop 后的过读仍在。没有现有 API 的 drop-in 普遍胜者，不重复把已探索 peek_mut/radix/dead-drain 称新发现。
- 支持模型：12,000 scalar upper-bound 配置、277,408 certified elements 通过 top-q frontier 及完整 tie prefix 检查；不是可达 BPE 语料验证，未保存精确运行起止。`model.py` SHA-256 `ca9c7dcfdb10…`。
- 证据：阶段研究材料、`results.json`；[LaB-PQ SPAA2021](https://arxiv.org/html/2105.06145v3)、[Selectable Heaps](https://arxiv.org/html/2011.11772v2)、[Heap selection SOSA2019](https://drops.dagstuhl.de/entities/document/10.4230/OASIcs.SOSA.2019.5)。只读源码/主文献与独立 oracle 模型，无仓库执行或性能结论。

### 150｜fresh batch cap 256 的语义与策略审查

- UTC 对照：开始：10-11 23:03:12；结束：10-11 23:09:29。

- 开始：2026-10-12 07:03:12。
- 结束：2026-10-12 07:09:29。
- 工作：只读 a2 merge/DESIGN/types，独立获取历史作者 records 及最新 child storage 诊断，核对 cap 的语义、类型限制和实测压力。
- 结论：256 是调度/资源策略；rank usize、bucket0/1、ID 目录 u32 和 compatible-prefix 证明不要求 256。保持 conflict、AA、reserved、reuse、target 规则后可改变 cap。代数构造 128×128 disjoint head/tail 二元词有 16,384 兼容前缀，可出现 64 轮对 1 轮，但不是自然语料观察。
- 关键负证据：旧 c638 作者 ByteLevel1,646 batches/49,744 rules/max169、Whitespace1,390/32,241/max111，均无 256 命中。records 只有 summary，不能据此算ΣV。新 child 的 50,019/32,607 prepare jobs 没有 batch 边界，不能推 cap hit。
- 决定：保留 256，待真实 cap-pressure 证据。更高 cap 扩大 Jobs、old-parent fan-in、hash 及 staging；若已复用 attempt scratch，不能再重复计算跨 batch 目录填充节省。
- 证据：阶段研究材料 SHA-256 `d4da41ca31fd…`；`source-evidence.json` SHA-256 `7138e50aa06f…`；[历史 records](https://github.com/Yikai-Liao/tokenizers/blob/02620e0ad37d8b165b5923902a05184e483cbd88/experiments/bpe-simplification/evidence/trainer-interface-20261011/records.json)，blob `ae53cdc52426f22882932b6cc84ce7ff9139c8e2`。无新 BPE 模拟、仓库运行/修改或 timing。

### 151｜最小因果测量账本设计

- UTC 对照：阶段时间：10-11 23:03；10-11 23:05。

- 阶段时间：2026-10-12 07:03—07:05。
- 工作：为已研究小控制列出既有 join 边界 timers 和现有 Vec 长度计数：B/T/R/G/M/F、owner action/fragment 量、complete K_all/K_live、目录初始化量、gate probes、heap correction/dead pops 和最终 retirement。
- 结论：CPU samples 不能替代 joined wall span 或工作消除证据。先用现有 loop/长度聚合，禁止每 position timer/atomic、坐标日志或 Positions 重数；instrumented 诊断与最终 uninstrumented 比较分开。F/K_live 含低频淘汰，不能都说成 dedup；降低 route 并抬高 prepare 只是转移，须看 total。
- 证据：阶段研究材料 SHA-256 `7ec4f0f22e9f…`。阶段证据设计，不是最终 experiment 排期；没有插桩 patch、仓库执行或 benchmark。

### 152｜token-mass 与惰性 old-loss 记账的组合

- UTC 对照：开始：10-11 23:10；首份阶段完成：10-11 23:17；10-11 23:17:40；10-11 23:25:02；10-11 23:25:31。

- 开始：2026-10-12 07:10。
- 首份阶段完成：07:17；独立验证 07:17:40 开始，07:25:02 完成报告，07:25:31 加入 AA 澄清。
- 工作：扩展既有 U 定理，在 prebatch witness 许可和 postbatch 失活之间保留两个时间视图；省略 postbatch dead endpoint 旧 key 的 loss，允许数值 stale 的 dead cells 延迟清理。
- 结论：coherent frontier 上，只要两端仍 live 则 stored count 精确；被省 loss 的 key 至少一端永久 dead，所有 authoritative lookup/heap/gate 必须用 U 守卫。pending 更新只需 O(batch) touched 条目，但高效 lookup 需计费。raw map length、stored count 总和与 queue-live 差式不再自动代表 logical live 状态。
- 反例：postbatch 已死旧 parent 仍可产生 live newborn，不能把其死亡用于本 batch snapshot gate。combined old/birth group 仍要保留 birth 工作；只有 removal-only 或分离的 old-only group 才可能整组省掉，且省旧 group 会影响现有 positive cache 命中，不能同时声称两项免费收益。
- 组合边界：普通语料 raw stale membership 或仅增加真实工作，但 implicit reweighted quotient 依赖精确边界抑制，错误 positive 可造 ghost。统一 oracle 为 physical membership 且 prebatch endpoint eligible，再加 AA 例外；coherent commit 后才发布 U。
- 支持模型：3,000 顺序 fresh 配置、29,864 committed frontiers 通过 guarded live map 与完整参考比较；省 6,653 old-key loss events，累计跨 frontier 观察 57,678 stale dead cells（不是峰值内存）。含 AA、zero、strict cap、低 reserved 及 postmass-gate 反例。保存结果时间 2026-10-12 07:16:11.793316，未保存独立进程起止；compatible batch/source-shaped 验证待补。
- 证据：阶段研究材料 §10，当前 SHA-256 `7960723146ea…`；`check.py` SHA-256 `17a5fdd82c44…`、`results.json`。无 Rust/仓库执行或性能结论；sticky∞初始 tokens 和 loose AA 界会限制覆盖。

- 07:24—07:25 更小变体：每条 Rule 仅保留两个 own-input death bool，可保守省旧 loss 而不做每事件 post-bound 查询，所有 birth 仍照常处理。初步独立 1,858 histories 摘要：own-input 控制省 4,394 groups，全 endpoint 控制 4,667，保留 273 neighbor-only 更新。报告/哈希待最终核对；不是当前语料覆盖率。
- 源码接入约束：U 归 PairIndex 拥有，Batch 消耗前算 O(B)pending 更新，prepare 全程用 pre-U，正常 owner commit 之后才安装 post-U，不需 AtomicU64 依赖或新 coordinator。coherent fresh frontier 的 map key 可注入 heap 候选，publication 中 pending candidates 另算；这不界定 map capacity/RSS，lazy 占位仍可能使表增长。
- 小 fixture：uncapped PiBCA 有 7 个 live PiX births，必须保留；cap4 没有 birth，可省全部第二 batch old-loss actions。已知 AA selected-key missing-loss 跳过不是新定理，提前抑制也须保留 born payload/目录 cache。

- 最终独立验证：全 endpoint 模型 07:22:04.846022—07:22:07.412968，own-input 模型 07:23:43.258482—07:23:45.576756。各 1,858 histories、15,033 frontiers、13,175 batches、18,283 规则通过，含 3,455 多规则、1,064 shared-head、503 shared-tail、2,130 AA、977 reserved 首次激活。30,040 baseline grouped updates 分别省 4,667/4,394，own-input 留下 273 neighbor-only 更新；此套件相近覆盖率不外推真实语料。
- 新完整性条件：814 次 floor raises 中 2,936 个仍 live-endpoint key 需按新 floor 显式清理，不能仅提高 U 比较 threshold 就假定 logical membership 已正确。4,024 live birth keys 的旧 parent 在 postbatch dead，支持必须保留旧 snapshot birth 的反例约束。
- Quotient 补充 07:21:04.215212—07:21:05.076425 运行：303 cases、227 mid-run 构造、2,422 后续规则、419 protected-cut absence 检查通过，省 660 旧 key 更新。模型在 logical representative mass 而非废弃 physical copies 上检查 U。
- 最终证据：阶段研究材料；`check_two_view.py` SHA-256 `032927b3ada0…`，`check_own_input_subset.py` `5e7cb9f8b841…`，`check_implicit_quotient.py` `2db17f5059bb…`。相应机器结果/运行日志保存；无仓库执行、Rust 验证或性能结论。

### 153｜旧 parent span 预过滤与初始 admission 例外

- UTC 对照：开始：10-11 23:28:08；结束：10-11 23:33:05。

- 开始：2026-10-12 07:28:08。
- 结束：2026-10-12 07:33:05。
- 工作：利用 fresh 已读取几何，在 global membership lookup 前排除 length-ineligible 旧 witness，同时检查初始 span-one 例外、AA、zero 及 cap 算术。
- 状态：只读源码/数学审查，无仓库执行或编辑。

- 07:31 阶段细化：只在 OLD-role 创建 miss、child admit=false 且经过 AA override 后检查 parent span；复用已经读取的 neighbor span 加一个旧 endpoint span，无新字段/cache。可避免纯 length-ineligible 的全局 negative hash probe，但不限制 zero-weight probes；已知 uncapped 场景没有这项查询节省。
- 数学模型 07:31:22.519249—07:31:22.795810 运行：127,944 parent span cases 通过，含 6 initial exceptions、6,930 ineligible、3,075 selected-AA 及 60 overflow cases。`check_spans.py` SHA-256 `99765e10b1a9…`，`results.json`保存。有限 span 代数，不是完整 BPE oracle 或源码执行；完整报告已完成。

- 阶段报告：阶段研究材料 SHA-256 `f6c46758fb7a…`；机器结果 SHA-256 `2ed8c0ffcb82…`。判定为两个 span 都 1，或 checked sum 严格低于 limit；仅是可能 indexed，true 仍需 membership。

### 154｜initial Positions admission 的无假阴性过滤器

- UTC 对照：阶段材料：10-11 23:28；10-11 23:29。

- 阶段材料：2026-10-12 07:28 标记，07:29 完成；独立开始秒未保存。
- 工作：核查在 count-first/materialize-and-emit 中用 Bloom/Xor 筛选初始 append，最后对完整 fragments 做精确 membership，及 bloomfilter3.0.2/xorf0.13.0 源码成本。
- 结论：完整 checked 初始 counts 先做，filter 由精确 admitted keys 构建且无 false negative，再以 exact membership 移除 false-positive fragments，能保持精确 counts 和所有 admitted Positions（含 zero）。这是保守 storage gate，不能用于允许近似 quotient 边界，也不近似计数。
- 成本：同 key 的 false-positive occurrences 完全相关，不能按每位置独立伯努利估计。期望多余工作界不等于固定 seed 确定界；zero rows 可使单 key 额外位置任意大。≤256 dense 域既有≤8KiB exact bitmap 更自然。retained-local-directory 原本也能免 per-edge global probe，必须作为对照。
- 库审查：Xor8 最终约 1.23B/key，源码推导 construction scratch 约 135B/key（64-bit 布局估计），不能只报最终小 filter。Bloom 每查询多个 hash/bit 探测、构造/参数失败与 seed 策略仍需计费。无性能胜者。
- 证据：阶段研究材料 SHA-256 `2faa48ee8404…`。只读源码/API；无安装、运行、实现或 benchmark。

### 155｜zero-weight 位置省略的语义边界

- UTC 对照：开始：10-11 23:33:13；结束：10-11 23:49:46。

- 开始：2026-10-12 07:33:13。
- 结束：2026-10-12 07:49:46。
- 工作：独立审查 full-initialization fresh 路径与历史 reuse cohort 中，零权重记录/位置能否省略及何种观察会改变。
- 状态：只读源码与数学分析，不运行或修改仓库。

- 07:39 独立手工源码复核完成：prefix a、floor2、target9、ab:3/xbc:1/ybc:1/abc:0，在完整初始化不变后删 zero row，会把 public train_vocab Merges 从 q,p,p 变为 q,p；两种同 priority cohort tie 顺序均成立，最后 vocab 同为 8。zero-only 新 cohort 因 global reuse ledger 仍正而被发布/接受，不能按局部 weight0 删掉。
- 关键限定：此 fixture 的 PipelineBPE 内部 p 数值 rank2/1 不同，但最终 serialized merge list 在 duplicate collapse 后相同，未证明 tokenization 差异。不得把公开训练 merge 序差异扩大为已证实 serialized-model 差异。
- 手工复核证据：阶段研究材料 SHA-256 `b6c24b9c7e53…`。固定 a2 trainer/encoder 源码，无 native 执行；另一个有限 cohort 模型等待完整归档核验。

- 07:44 更强 zero-row 候选摘要：720 input-order 检查，涉及 raw merge order 及可能 normalized rank flip；独立 encoder/dependency 源码复核未完成，尚不能宣称已执行 encoding 或最终模型差异验证。

- 强 fixture 及 scratch 归档：prefix=a、F4、target12，ab9/xbbb2/ybbb2/tbc3/zbc3/abbb0，完整 raw 顺序 q,p,r,p,s，对照 q,p,r,s；last-wins normalization 使 p/r 顺序翻转。符号编码模型预测 xbbc 结果不同，但未执行 PipelineBPE。关键 heap 条件是旧唯一 max root 在 equal push 后留在根，已查 upstream0.3.6/0.3.9，未验证作者实际安装依赖。
- 模型运行 07:41:03.108208—07:41:03.196216：48 simple raw fixture 检查、240 full-target 检查、720 stronger input-order 检查通过。`check_cohorts.py` SHA-256 `61dd49ee3097…`；`results.json`保存。阶段报告阶段研究材料当前 SHA-256 `fdbbc96e8623…`，强 encoder seam 独立复核仍待完成。
- fresh-only 限定：完整保留 original 初始化、activation 和数值检查，省原始整行 zero 可在 fresh 成功轨迹下保持 positive 语义；一旦 reuse 必须恢复 original zero rows。此方案新增 fresh/reuse 物理表示政策，未按统一实现目标推荐，也没有实际 zero 占比证据。

- 07:49:08 强 fixture 独立源码复核完成，07:49:46 本轮材料定版。affix 转换、没有 whole-pretoken shortcut、minimum-rank 选择及 equal push 局部 heap 条件均已核对；因此 normalized configuration 顺序差异获源码证明，xbbc 输出[4,7,10]对[4,9,8]为具体源码推导并获符号模型支持。仍未 native 运行，不把它写成 Rust 测试通过。
- 最终报告 SHA-256：阶段研究材料 `1d0af5d052e0…`；独立 addendum 阶段研究材料 `e04c8a57822d…`；model 结果`2ae1a0f94878…`，脚本未变。较简单四行 fixture 的 serialization 限制仍保留，不能混用两个例子。

- 07:50:10 最终引用/scope 补充后，主报告 SHA-256 更新为`d13f719b486b…`。编码比较限定单个 xbbc pretoken、affixed non-byte 模型和 fresh scratch，不暗含任意外部 normalizer/pretokenizer。独立 addendum 及模型哈希不变。

### 156｜atomic old-loss 期间的 pending-birth 选择上界

- UTC 对照：初份阶段：10-11 23:33；10-11 23:38；10-11 23:35:16；10-11 23:45:55.605168。

- 初份阶段：2026-10-12 07:33—07:38；独立验证 07:35:16 开始，07:45:55.605168 完成。
- 工作：扩展既有 publication guard，在 selected raw frequency 开始的 producer 上界中，按完成 task 扣除 accepted mass−最大 local child mass；确保 payload 已保留再发布 credit。
- 反例/结论：ABC×10 经 AB→X 使 BC 计数 10→0 时，XC10 仍 pending，current counter 不能当 birth 上界。producer Pair 也不能当 child tie identity。需要 strict count 或合法 lex envelope，不能仅等旧 loss 完成。
- 更低热循环成本的候选：当前 Corpus 按 weight 降序，job 最后 cached weight 是其 accepted 匹配的最小 weight，k×last_weight 提供保守 mass 下界，避免每 match 再加总。须 checked 乘法，失败可取 zero credit；任意 reweighted in-place quotient 不保持降序，不能套用。
- 07:42 独立 fixture 摘要：500 个 ABAC_i Z_i ABABD_i 词、每词 weight2，加 EF1600，AB3000/BA2000；P4 restart-aligned tasks 为 384/384/384/348。三个完成 task 各 mass768、max-child256 后 producer 上界 1464、BA 上界 464，可在一 task pending 时认证 EF1600 的 priority。不是已运行生产 pipeline。
- 证据：阶段研究材料。physical 读写隔离、AHashMap 结构/span publication 仍是独立未解决接入成本，priority 证书不自动提供可执行快照；无仓库执行或性能结论。

- 07:44 独立滚动证书初步范围：1,956 snapshots（809 AA、207 多 producer）、14,278 completed-job subsets；另 240 histories、1,737 reachable batches、3,507 arrivals。等待最终归档时间/哈希。reuse 反例 XABX 的 AB→已有 X 用 mass1 产两个 XX，否定把 c≤m 推广到 reuse；fresh-only 前提不可删除。

- 最终独立范围：1,956 非空加 360 空配置、2,186 producers、4,560 jobs、6,981 物理 matches；14,278 completion subsets 和 24,832 monotonic edges 通过。另 240 完整 history、1,737 batches、2,680 rules、3,507 arrivals 通过。前述 P4 fixture 已按真实 restart-aligned 任务形态检查；只证明 priority，不声称 reader readiness。
- scope 更正：经过 early witness gate 后，U 界定 potentially selectable children；若要界定所有 length-admitted raw children，包含已省 below-floor children，则用 max(F−1,U)。例如 ABC1+ABD1、F2 会使 U0，但 raw XC/XD 各 1，故不能省略限定。
- 可复验材料均在`producer_pending_bound_model`，报告阶段研究材料，五项实际运行北京时间如下：
  - `check_pending.py`：07:38:38.372220—07:38:38.686766；SHA-256 `202737fa3643…`。
  - `check_reachable_stop.py`：07:40:46.747957—07:40:46.827319；SHA-256 `f9c9b53309e3…`。
  - `check_four_thread_stop.py`：07:41:53.517947—07:41:53.580669；SHA-256 `e4bb2a230741…`。
  - `check_histories.py`：07:43:04.059844—07:43:04.305869；SHA-256 `afce9e6231f2…`。
  - `check_scope_edges.py`：07:44:17.699236—07:44:17.699389；SHA-256 `c3d103e4e202…`。
- 每项保存独立 JSON/run log，前置 boundary recorder hash 保持`65ec7e65b2de…`。无仓库执行、Rust memory-model 或性能验证。

### 157｜conservative lookahead 与 producer guard 的理论对照

- UTC 对照：阶段时间：10-11 23:47；10-11 23:50。

- 阶段时间：2026-10-12 07:47—07:50。
- 工作：阅读 Chandy–Misra–Bryant 原始保守模拟/空消息机制及 ns-3 官方实现，映射 producer U 与 full-priority envelope 的 safe-head 条件。
- 结论：已注册所有 pending producers、ready 候选自身精确且其 priority 严格超过所有 known/producer 上界时，可安全认证下一顺序。只迁移 frontier-local admission lemma，不宣称整个 BPE 等价于 CMB 共享状态/时间模型；低 reserved 输出可使完整 priority 向前跳。
- 未闭合义务：blocker 订阅要 race-free register/recheck 和错误/取消释放；c=m 的 zero-lookahead 不能凭通知产生进展。child 可见后才能删 guard，数字 0 不表示 zero-weight payload 或 reader 已结束。AHashMap 结构、span Vec 发布、corpus 邻居快照和 Positions 完整性仍独立处理。
- 证据：阶段研究材料 SHA-256 `7de9ca95977c…`；[Chandy–Misra 原始论文](https://www.cs.utexas.edu/~misra/scannedPdf.dir/DistrSimulation.pdf)、[Bryant 原报告](https://dspace.mit.edu/entities/publication/579e8de6-49c2-4564-a339-5f3dad1d9450)、[ns-3 CalculateGuaranteeTime 源码](https://www.nsnam.org/docs/doxygen-old/d5/d93/null-message-simulator-impl_8cc_source.html)。只读主来源，无仓库实现或吞吐保证。

### 158｜从初始 admitted pair 端点量构造有限 token 上界

- UTC 对照：开始：10-11 23:51；10-11 23:54；结束：10-11 23:58；10-12 00:13。

- 开始：2026-10-12 07:51—07:54 间完成初步推导并启动独立验证；各验证精确开始秒待记录。
- 结束：首份独立报告 07:58 完成；08:13 核对归档，后续降 floor 反例另见阶段 165。
- 新方向：不把 initial token U 都置∞，在既有 initial candidate flatten 中汇总 admitted 初始 pair 的两端计数，形成一份 8V U。候选证明跟踪接触原始高频 cut 的带权 token occurrences；任何未来 selectable pair 应继承某个初始 admitted family。
- 候选更新：仍对 distinct selected input 各减 f，对 fresh output 取 f，不扫描 Positions/token 也不持久保存 family 元数据。source 整合与完整不变量尚待独立验证。
- 已发现限制：16V directional 分离界不能天真套用同样 AA 更新；AAA raw f2 后仍有 XA，In[A]只少 1。当前优先审查单份 exposure bound，不宣称已完成证明或测得收益。

- 07:55—07:56 独立审查得到关键负结论：该 admitted-degree U 虽安全，但对拟议首阶段 live old-loss 节省是冗余的。initial ID 上界保留原始 marked cut 端点 credit；任何未被选走的 prebatch 高频旧 key 仍保有这些 cut，故 post-U[t]≥其 count≥F，initial-ID 死亡不能新删此类 live 旧 loss。rare-neighbor births 已有早期 witness gate 覆盖。不能把 synthetic retirement 数当新增有用工作节省。
- 后续仅保留有明确成本的替代问题：完整 active initial mass 可由全部 outgoing counts 加 non-singleton 词尾 weights 构成，但需要 per-range terminal 摘要/O(words)更新；尚未作为推荐。该阶段当前成果是排除一个看似廉价的新机会，模型/证明产物仍待归档核验。

- 08:10 另一组合反例：trial T3→base F1，AB3+CD1 在 reindex 后若继续使用 T 阶段 marked-degree U[C,D]=0，会错藏仍应选择的 CD。此前∞初始 true-mass 上界不受该特定错误影响。independent 模型进行中；任何仅相对阈值 cut 的证书在降 floor 后必须重建或废弃。

- 归档补核：独立`admitted-mass-independent`材料标记 07:58 完成，保存模型时间 07:57:55.654388。18,840 配置、79,947 frontiers、61,114 cohorts、80,423 rules、1,089,086 initial-high-witness 检查通过，含 13,961 多 cohort、12,499 AA；支持安全性及 initial-ID 新增节省冗余，不把 retirement 数量算收益。脚本 SHA-256 `05d3fe6bef8d…`。主报告阶段研究材料当前 SHA-256 `3573801582de…`；模型未保存精确进程起止，本次没有补跑。

### 159｜组合 loss 控制与 producer 完整 birth 发布

- UTC 对照：开始（截至记录，非精确起点）：10-11 23:54；结束：10-11 23:56；10-12 00:13。

- 开始：截至 2026-10-12 07:54 已启动；精确开始秒未保存。
- 结束：2026-10-12 07:56 报告完成，08:13 补核归档。
- 结论：frozen positive gate、dead-endpoint loss 省略、producer 完整 birth reduction 可与普通 exclusive count owners 共存，不要求 AtomicU64 或退休 tickets。input-dead 两 bool 是保守子集；声称所有 routed losses 均两端 live 须另付 neighbor pending-U 查找。省 old-only group 会失去 positive cache，birth group/Positions 仍需保留。
- 模型 07:54:47.035145—07:54:49.380702：2,058 histories、16,783 frontiers、14,725 batches、20,965 rules；baseline positive loss groups34,810→retained28,888，allocated groups37,726→35,496。2,740 empty action batches、3,255 birthless 但仍有必需 loss 的 batches 均覆盖。重叠 input/neighbor/AA 计数不可相加当独立节省。
- 证据：阶段研究材料 SHA-256 `33beec0cf3a8…`；`check_composition.py` `0bad848c5eaa…`，依赖 boundary recorder `65ec7e65b2de…`。使用先前∞初始证书，未验证较新 finite 初始化；无 Rust/性能结论。

### 160｜候选自身 priority 与 prepare readiness 审查

- UTC 对照：阶段时间：10-11 23:50；10-11 23:58；模型 07:55:52.007118—07:55:52.836501：10-11 23:55:52.007118；10-11 23:55:52.836501。

- 阶段时间：2026-10-12 07:50—07:58。
- 工作/结论：两 endpoint-incidence 查询可认证旧候选自身 count finality，要求 loss publication 而非 task 沉默。更强 no-overlap 可使 matching/starts 在无关写入时稳定，但 neighbor 处理仍需延后或新 pre-context gate。value-level 无高频 parent–candidate 相邻 witness 可跳过正在变动侧的 context 读取；这不使现有 reader 原封不动安全，也不授权 candidate 早 apply。
- 成本/前提：分清 loss/write/payload/reader seals、output 到旧 endpoint 投影、待写 producer 查询和安全 span 读取。没有 symbol 冲突不自动提供全部物理 read closure；零 weighted delta 也不证明无实体 overlap。源审查覆盖给定 ea1 完整几何，未声称逐行重审所有 a2 corpus。
- 模型 07:55:52.007118—07:55:52.836501：15,783 certified completion states、27,778 endpoint observations 通过，111 个变化但无关 second reads、1,370 changing-neighbor sides 在 context 前跳过。有限 old/final cell 组合，不是 Rust memory-model、length/error 等价或真实选择率。
- 证据：阶段研究材料、`check_certificate.py`、`results.json`；脚本 SHA-256 `5c099be04f5c…`。无仓库执行。

### 161｜按既有 weight-cache runs 归约精确 accepted mass

- UTC 对照：工作时间：10-11 23:57；10-11 23:58。

- 工作时间：2026-10-12 07:57 开始；07:58 收到首份完成材料。
- 工作：为 pending-producer credit 提取精确 accepted mass，使用已有 starts.len 与 weight cache 切换，而非每 accepted match 额外 checked 加法。
- 结论：两个 job 局部 scalar mass/run_start_len，在 cache 换段及 job 结束时折叠 weight×该段 accepted 数量。额外算术 O(visited weight runs)，无 Positions 重扫、无逐匹配额外计数递增；任意权重顺序仍成立，不依赖前述 descending-min shortcut。
- 成本/限制：unit-heavy 单 run job 可能少付很多加法；每 start 换 weight 时可更差，新增状态与 checked 乘法仍计费。此结果不解决 rolling 读写/publication readiness。精确 AA accepted mass 可在已得证据后收紧后续 U，不能回溯省本轮 loss。
- 支持模型：20,000 配置、2,722,949 accepted events、727,862 weight-run folds 通过，覆盖 zero、任意 weight 顺序与 u64 乘法。保存时间 2026-10-12 07:57:48.999926，未保存独立进程起止。
- 证据：阶段研究材料当前 SHA-256 `699ab5f4072f…`；`check.py` SHA-256 `a72948f8dbf1…`、`results.json`。独立数学运行，无仓库执行或性能测量。

### 162｜producer prepare 与纯完整 birth 归并在同一外层 join 内执行

- UTC 对照：开始：10-11 23:59；10-12 00:03；结束：10-12 00:06；10-12 00:13。

- 开始：2026-10-12 07:59—08:03 开展新组合；精确开始秒待记录。
- 结束：2026-10-12 08:06 报告完成，08:13 补核保存材料。
- 工作：在 corpus/count key 保持冻结时，每 producer 并行 prepare 后立即执行其完整纯 birth reduction，外层 join 后才做 exclusive old-loss/插入发布；尝试弱化一条 barrier 而不造完整 rolling engine。
- 初步限制：嵌套每 producer map_init 可强迫 B 份 domain 填充，attempt-local TLS 需作为明确成本/依赖才能避免回归。Rule 拥有的 selected Candidate 可在 inner ReadTask join 后由 worker drop，可能分散原 Batch 调用方的回收尾部；尚无实际 span 或性能结论。
- 状态：本轮源码 lifetime、work/span 和维护阶段审查已完成；08:25—08:34 进一步内存复核见阶段 173。

- 完成结论：唯一 emitter 令每 producer inner prepare join 后 birth 已完整，可纯 pack/sort/sum/floor/concat，外层 join 仍在任何 corpus/count mutation 前。固定 DAG 理想 span 从 max p+max a 变为 max(p+a)，B1 或同 producer 双重重尾无此收益；finite pool 和内存峰值可能变差。
- 独立 schedule 检查覆盖 4,096 span identities 及 7 个 finite scheduling fixtures；opposite skew 模型 18→10、work saturation20→20、late-heavy 最优 110→110 且可行 eager121，均为抽象单元，不是 Rayon 预测。执行精确起止未保存。`check.py` SHA-256 `4e8e3824d64e…`，results.txt 保存；主报告阶段研究材料 `0718cce8524c…`。无仓库执行/benchmark。

### 163｜early witness gate 适用区间审查

- UTC 对照：开始（截至记录，非精确起点）：10-12 00:03；结束：10-12 00:05；10-12 00:13。

- 开始：截至 2026-10-12 08:03 已在研究；精确开始秒待记录。
- 结束：2026-10-12 08:05 材料完成，08:13 补核。
- 阶段结论：F=1 且所有 row weight 为正时，gate 不能省任何 length-admitted birth；即使有 length cap，能额外省的只可能是 removal-only groups。该边界防止把 gate 当所有输入均有收益。
- 状态：适用性阶段证明已保存，无 native 测量。

- 推广：所有 row weight≥实际工作 floor F 时同样没有 admitted-birth selectivity；cap 只能删 removal-only population。stale Positions 在 gate 前已 matcher 排除，gate 不能省那部分 decode。高 weighted 频次不代表多物理匹配，当前 Chinese/English 汇总仍未提供所需 G/F/key fanout/实际 probe 比例。
- 证据：阶段研究材料 SHA-256 `c4920f13b3d7…`。只读源/证据与代数审查，无新增 native 测量。

### 164｜forced-pair contraction 的计数重命名与历史 Positions 迁移

- 初步探索 08:04；报告所记阶段 2026-10-12 08:05—08:15，08:18 核对完成材料。
- 工作：若所有 physical a/b 出现都被 ab 覆盖，研究能否重命名边界 counts 并复用历史右 parent postings，减少逐位置出生构造。
- 否证：当前 matcher 下对历史 right-parent Positions 做固定 offset 移动不安全，即使来源是合法 greedy 轨迹。F2、IDs c0/b1/d2/v3、words db6/ccbv3/ccbdbv2，顺序 db8→y、cc5→a、ab5→x、xv3；旧 BV 坐标[6,13]减 2 得到[4,11]，11 实际是 XYV 中 X 的 tail，matcher 却可读 X11/V14 并接纳 phantom weight2。
- 结论：不是 malformed 输入反例。stable-cut 重设计需真正 live-cut invariant 和任意 heap payload handles，超出小改动前提。没有 native 执行或收益证明，具体源码/模型材料已归档。

- 最终有限检查：08:11:46.304197—08:11:46.424546，5,043 forced-word/weight cases 验证精确 scalar 重命名法则；完整 greedy phantom fixture 与相同 pair marginals 不同后继 counts 例通过。模型`check.py` SHA-256 `63a5fd20be57…`，result.txt 保存；报告阶段研究材料 `957468719b11…`。
- 来源范围：本轮读取继承 ea1 source copies，未另取当前 checkout；pair/trigram 不足反例不构成对 lossless DAG/grammar 的普遍不可能性。保留 scalar 边界法则和 stale-shift 反例，不建议新增 rename-root fast path。

### 165｜降 floor、lazy loss 与完整 live reindex 的组合

- UTC 对照：阶段验证：10-12 00:11:06.291193；10-12 00:11:08.225297；08:12 错误契约更正：10-12 00:12。

- 阶段验证：2026-10-12 08:11:06.291193—08:11:08.225297 实际模型运行；独立开始任务时间未保存。
- 结论：降低 T→F 必须完整替换 count/candidate index；即使保留真实 mass U，跳过的旧 loss 也不会自动恢复精确。T3/F1，PBCA×4+QBCR×1 中 U[X]=1，但 XA stale cell4、真实 0；只降 guard 会复活错误候选。模型观察 36 次此类 revival。相对 T 的 admitted-exposure U 则须额外重建/废弃。
- 验证：4,530 histories、20,310 frontiers、3,149 one-time reindexes，其中 1,509 在非空 prefix 后；8,640 suffix rules、2,051 rebuilt zero Positions。涵盖 AA/multi/shared-input/reserved/low outputs。retained payload 按当前坐标规范化，未建模 historical lazy matcher 和 producer leaf scheduling。
- 证据：`check.py` SHA-256 `2e8301f8f2a3…`，`results.json`。无仓库执行。
- 08:12 错误契约更正：阶段研究材料 Limits 前加有日期说明，保留原“reuse and partial failures”历史文本但废弃后半句。仅 explicit active reuse 原始重启；仅 clean joined underfill live rebuild；真实 partial/arithmetic/codec/capacity/rebuild errors 丢弃并传播。遵循 speculative-frequency-floor §12。更新报告 SHA-256 `ca4e6ac224e9…`，原脚本/结果未改。

### 166｜terminal fresh batch 的工作省略

- UTC 对照：开始：10-12 00:11:53；10-12 00:18:06；结束：10-12 00:21:15；10-12 00:23。

- 开始：2026-10-12 08:11:53；最小契约/保存补充 08:18:06 开始。
- 结束：2026-10-12 08:21:15 报告完成，08:23 核对归档。
- 结论：fresh selection 已经确定精确规则、IDs 和 selected raw-count trace 且实际 vocabulary 达到 target 后，可跳过该末 batch 全部 prepare/apply/index commit，调用方之后只处理 records/progress 和销毁。这是既有末 batch liveness 观察的扩展，不作为新发明或 50k 大收益保证。
- 最小契约：无需新 enum 分支，可把 Committed 命名改为 Accepted；未达 target 时仍须完整 coherent 提交，达到 target 时 corpus/index 仅可销毁，不保证反映末 batch。reuse 保持旧有 ordered 执行，reserved 零 growth 必须比较实际 vocab 长度。
- 等价范围：公开 model 及 accepted 规则/ID/count trace 在 source-reachable fresh 且成功分配前提下相同。initial validation 及 selection 错误仍保留；省去 allocation/codec/debug/fault-injection 工作会改变其错误观察，不能称完全 Result/panic 等价。最终物理 corpus 不再是该契约的输出。
- 模型：原 in-memory smoke 约 08:15:38 完成；为归档于 08:19:12.222262—08:19:14.734329 实际复跑，复现相同计数。4,089 corpora、24,534 target comparisons，24,360 成功、174 初始 pair overflow 拒绝；98,016 rewrites/246,273 birth-group 检查。范围 F1、fresh 首次激活、无 affix/cap，未遇 restart；并非完整 source 契约的全范围 Rust 验证。
- 证据：阶段研究材料 SHA-256 `5f5450434ad3…`；`terminal_fresh_batch_elision_model.py` `04d5d192ca01…`及`.result.json`。无仓库执行/编译或 benchmark。

### 167｜priority 与 own-readiness 组合 fixture 验证

- UTC 对照：开始（截至记录，非精确起点）：10-12 00:17；结束：10-12 00:22；10-12 00:28；10-12 00:44。

- 开始：截至 2026-10-12 08:17 已在独立验证；精确开始秒待记录。
- 结束：08:22 正 fixture、08:28 对称 apply 阶段材料保存；08:44 完成本日志证据核对。
- 工作：把 pending producer priority 上界与 candidate 自身 readiness 条件放进同一个可实现例子，区分数值认证和安全物理执行。
- 初始状态：独立模型进行中；完成结果与保留限制见本节后续日期记录。

- 08:20:54 组合 fixture 通过当前 source 排序/task-cut 模型：AB3001、EF1601，前三 job 各 384 starts，d768/c256，pending U1465、BA464；末尾 ABEF bridge weight1 令 BE 在 F2 缺席，EF 语义上可在投影后跳过变动前驱。
- 未解决：相等的投影值不使 non-atomic 冲突读写合法；正在独立审查实际 data race 和对称 apply/read continuation。此处只记录语义/priority 兼容，不声称 Rust 并发实现已安全。

- 08:28:20 完整 a2 corpus 源码重新取得并核对 Slots422–479：仅 Narrow AtomicU16/Wide AtomicU32，整个 logical ID 以 relaxed atomic 读写。此前 packed24 疑虑只是历史表示核查，不是当前代码 bug，已关闭。多 cell 几何一致性、span/metadata 稳定发布仍是独立义务。
- 固定全文`pinned-corpus-a2b20f32-full.rs` SHA-256 `a488b9727c39…`；与已用完整 ea1 corpus 副本的字节 hash 相同。本次只读文件核验，未运行仓库。

- 08:44 核对已保存 08:22 fixture 及 08:28 对称 apply 材料：精确 production task 尺寸为 384/384/384/349，最后 349（含 bridge），不是原无 bridge fixture 的 348。前三项 d768/c256，U1465、BA464，EF1601；完成标记必须包含 old-loss 真正可见，不能只完成 worker 算术。
- 对称结论：在 immutable ports、结构冻结 membership、pre-context gate 先于 span/two-hop/directories、各自准备先于自身 apply 的条件下，EF 可语义上 apply 而最后 AB prepare 未结束；两侧都把 BE 的 old/new 值投影为同一低频原 cut。已核对现存 Narrow AtomicU16/Wide AtomicU32 整 ID 原子，因此此 fixture 的 endpoint 访问不撕裂；这不提供多 cell 快照或安全 metadata 发布。
- 有限证据：`check_fixture.py` SHA `9ebb0b15cadd…`，08:20 通过 3 种同 weight 排序；`check_symmetric.py` SHA `9a99af4a24de…`，08:27 通过 6 种 bridge write interleavings、AB 准备在 5 个 EF 写入前缀、EF 准备在 3,003 个 AB 写入前缀不变。没有精确进程起止记录，不补造。
- 必需负例：AB3、BC2、ABC1，F2，AB4 的 BC loss 已发布但末端写入未完成时，BC 真实 count2 且胜 birth guard1，旧 matcher 却仍接受将消失的 weight1 BC；因此 numeric readiness 不等于 physical support readiness。`check_support_counterexample.py` SHA `0c5a5f0a32c7…`；原执行精确时间未保存。既有 accepted starts 精确查询可避免特定 neighbor load，但粗首尾 envelope 会错删 1,500 个 side；需要稳定 buffer 和读者启用协议。
- 当前状态更新：本 fixture 的有限语义/source 审查完成；一般 rolling 协议、span Vec 与 hash 结构并发发布、Rust 所有权和真实性能仍未完成。不是现有 merge_next 已并行运行的证明。

### 168｜精确 accepted mass 对 AA token 上界的后置收紧

- UTC 对照：阶段时间：10-12 00:18；10-12 00:24；10-12 00:36；支持模型保存时间 08:23:54.299362：10-12 00:23:54.299362。

- 阶段时间：2026-10-12 08:18—08:24；08:36 核对归档。
- 结论：accepted contraction mass d 已知后，对每 input occurrence 扣 d（AA 两次），output 取 d，sticky∞不动，能在一份 U 及原 pending plan 内安全收紧。non-AA d=f 无新信息；AA δ=d−(f−d)避免直接 2d 溢出。post-prepare 证据只能省后续或下游已 materialized 工作，不能回溯省 prepare。
- 新适用例：aaaaaa、F4 有 raw f5、d3，selected output U3<F 可让全部 incident birth 不活跃，但不能撤销所选 AA 规则。AA 新 output 立即死亡仅可能 F≤f≤2F−2，F1 不发生；F1 仍可能更精确使旧 generated input 退零。原“selected output 总≥F”仅对 raw-f 更新成立。
- 成本：按既有 weight runs 折叠仍加两个 locals、每 run 乘加、每 job 返回 scalar 和 producer 合计。改在 serial AA prepass 提前拿 d 可省 prepare，但要新 weight 查询/串行工作并可能重复 job 查询，不是免费统一改进。
- 支持模型保存时间 08:23:54.299362：12,000 配置、159,810 frontiers、29,888 AA、18,130 shared-input；42,724 更强 retirement 事件仅是套件事件。随机 cap3—19，未覆盖 initial cap≤2，见阶段 171 精确 scope 说明。没有完整 lazy-map/source 运行验证。
- 证据：阶段研究材料 SHA-256 `481d75a0f617…`；`check.py` `01110fbbf0e8…`、`results.json`。精确进程起止未保存，无仓库执行或速度结论。

### 169｜稀疏 mutable counter frontier 的安全借用方案

- UTC 对照：开始（截至记录，非精确起点）：10-12 00:19；结束：10-12 00:32；10-12 00:51。

- 开始：截至 2026-10-12 08:19 已启动；精确开始秒未记录。
- 结束：阶段报告 08:32 完成，08:51 核对。
- 工作：为 K 个 affected keys 取得安全 disjoint mutable values，审查 duplicate 验证、hash/结构冻结、work/span，与 producer grouping 比较，目标是在无 AtomicU64/永久 handles 条件下改善 owner 倾斜。
- 否定结果：当前 ordinary AHashMap 及已查 safe API 不能同时提供任意动态 K、稀疏查找和 subquadratic 一次 borrow 验证。std get_disjoint_mut 使用 const 数组且重复验证二次，hashbrown0.17.1 源对每个已找到 pointer 与前序全部比较。此为已查 API 限制，不是 hash map 理论不可能。
- 替代成本：固定 C 分块使验证 O(KC)，但同 owner 须 ceil(K/C)借用/join 波；remove-owned-staging 需 K removals 与最多 K reinserts 及 O(K)storage；copy-new-value staging 有额外 lookup 和 owner 写回；先 M→K 归并仍须 sorting/hash、temporary 与每 owner K 次结构发布。many-distinct-key hot owner 瓶颈未消失。
- 语义：valid fresh 完整 loss 可归并，非法 floor-retirement streams 的错误次序和 signed reuse 不可由 net delta 取代。抽取后错误丢弃，不能当 clean live-reindex。U-lazy 物理 dead keys 会让 queue.len−physical map.len 漏数，原 GC 触发只能覆盖其检测出的死亡，不能顺带保证该组合内存界。
- 证据：阶段研究材料 SHA `f40485ff6545…`；Rust 标准 HashMap 与 hashbrown primary API/source。无仓库执行、编译、native benchmark 或额外数学模型。

### 170｜edge-mass pigeonhole 频率下界的复核与支配性

- UTC 对照：开始：10-12 00:24:06；结束：10-12 00:28；10-12 00:29；10-12 00:29:56；10-12 00:31:28；10-12 00:31:36。

- 开始：2026-10-12 08:24:06。
- 结束：发现约 08:28—08:29 交付，08:29:56 开始整理，08:31:28 保存，08:31:36 确认产物。
- 归属修正：不仅 uncapped c1，capped c3 也已在阶段 63 的 03:47—03:49 §9 记录，二者都不能作为本轮新发现。新内容为支配证明、绝对 horizon ceiling 和 marked-support 分母细化。
- 结论：capped growing-type potential 被既有 frozen-excess 证书支配，应淘汰该冗余候选；256 active 初始 IDs 下不能达到 49,744 初始 action horizon。uncapped 界独立但已有，marked support 给 N+t(Δout+Δin)+t²−t 的更紧 future-type 分母，仍需 census 且无实用收益证据。
- 数值/语义：实际或 marked 物理边界 mass 递推不能套到被 floor 退休任意丢质量的 map sum。E 为 weighted mass，不能用 256MiB 输入字节数直接代替；unit-weight 上界的数值否证需另立前提。optional proof 算术 overflow 只能放弃新证书，不能抬高 lower bound 或制造训练错误。
- 证据：阶段研究材料 SHA-256 `d8b4a542c517…`。本轮纯数学/源码证据审查，没有独立可执行模型、新 benchmark 或仓库执行。

### 171｜保留历史 stale Positions 的集成对抗模型

- UTC 对照：开始：10-12 00:30:14；10-12 00:30:19；结束：10-12 00:41:51；10-12 00:39:44.318264；10-12 00:40:16.152588。

- 开始：任务 08:30:14 启动，研究报告记录 08:30:19 开始。
- 结束：阶段材料 08:41:51 保存；最终模型运行 08:39:44.318264—08:40:16.152588。
- 工作：联合 frozen witness gate、pre/post U、精确 AA mass 选项、producer 完整归并及 terminal elision，保留原始固定坐标、历史 candidate cohort、孤儿和 lazy-dead cells，而非每轮重建 posting oracle。
- 有限验证：7,773 配置、15,546 差分比较；1,283,493 拒绝 stale 坐标，3,696,216 历史 head 合法性检查，763,944 个 floor 过滤前 physical birth 坐标检查，其中 105,350 零权重 birth；24,327 AA batches、6,206 live-child 零片段、8,408 terminal 禁止续用断言。均为汇总数学计数，不是生产 selectivity 或 benchmark。
- 支持结论：正确组合没有找到 phantom birth。新增关键不变量是每个 posting 必须来自该 candidate 左 ID 的真实历史 head；fresh 首次消费会破坏该 head 上的旧 ID，且不再激活该 ID。任意内部 stale 坐标可以骗过 matcher，因此不能把此结论扩大为“全平面 endpoint 标签足以判断 live”。
- 故意破坏的 proposal 负例：PBCA/QBCA 各 weight2、F2、target8，BC→T 再 TA→X 后，若下一 selector 先于 U[T]=0 发布，stale PT count2 按 tie 胜过真实 PX count2；terminal 快捷返回会接受不存在的 PT，拼写 PBC 而非 PBCA。既定先发布 post-U 再选择的协议排除此例。这是优化组合的回归 fixture，不是当前仓库缺陷。
- 范围：覆盖初始 cap1/2、低 numeric reserved 零增长、256 cap、零权重、接近 u64 上限、原输入 active-reuse restart 边界；不验证压缩 codec、Rust 借用/内存模型、实际并发、完整 affix 初始化或 reuse kernel、allocation/panic 错误等价。
- 证据：阶段研究材料，脚本`check_fixed_history.py` SHA-256 `d79e9cdd2c7f…`；results SHA-256 `8c4408c319a2…`。没有仓库执行或 native timing。

- 08:31 模型 scope 修正：审查发现较早简化 scratch 的 cap 语义不能泛化为 production initial admission。源码初始 pair 仅按 frequency admitted，strict length gate 用于 newborn；新集成模型显式保留 initial 例外和 overflow 检查。既有 reindex 模型已用“两 span 都 1 或 strict cap”谓词。本条不是生产 bug，也不抹去已明确含该例外的后续模型；旧模型计数仍受其各自报告范围约束。

- 08:35 精确定位：`check.py`第 16–20 行 pairs()含全局 strict span-sum 过滤，但实际随机 limits 为 3—19，未覆盖 cap≤2，因此已报告套件不受该差异影响。`check_positive_cache_gate.py`第 51 行显式保留两 span1，composed loss 继承正确例外，不应称其旧结果错误。新固定坐标集成模型另覆盖初始 cap1/2。

### 172｜只选择 prewindow 旧 key 的滚动窗口

- UTC 对照：开始：10-12 00:34:59；结束：10-12 00:52；10-12 00:54。

- 开始：2026-10-12 08:34:59。
- 结束：08:52 主理论阶段报告保存；08:54 独立 family 模型结果完成，异步 event/readiness 审查另列阶段 175。
- 工作：限制窗口内只选 prewindow old-old pairs，全部 final births 推迟到 drain，研究能否消掉临时 Xc→XY birth；区分 earliest-selected 邻居的 old-loss owner 与最终 physical-left birth owner。
- 状态：immutable metadata/registry、atomic old counts 及精确选择/执行义务仍在证明，未接受实现或收益结论。

- 08:40 阶段推导：在窗口不消费新 outputs 前提下，每条最终 affected cut 至多经历 2 条 eager newborn 记录，最终只需 1 条；潜在省略有常数 factor 界，不是无限次 telescoping。与 Zhou/Larson/Elmongui VLDB2007 Condense/Partial Condense 相联系，稳定 original cut 可作键。协议、retained starts 内存仍独立审查，未证明完整实现。

- 08:43 后简化方向：以 W-adjacency 及 prepare-prefix apply fence 换取不保留 prior-start registry，代价是放弃部分 early apply；span metadata 稳定发布仍在源码设计审查，尚未定论。

- 08:48—08:49 证书细化：original-port 粗分组把 XA 与 XX 合并，fixture 前三 job 后 G2233，丢失 AB/EF 早期机会，故否决该弱分组。当前 source-shaped child groups 在完整初始 compatible cohort 已 admitted 的 coherent prefix 可作持久 family：已解析 output IDs 固定，剩余旧邻居仅能被后续不同 output 细分，不能合并到这些固定 IDs 或别的旧 ports。无需额外 tag enum，理论上恢复 c256/G1465；证明及模型尚在整合，不推广到 partial-cohort 语义。

- 08:50:38 初步模型结果已报告：9,152 案例、18,298 规则、376,756 guard states；refined fixed/open-ID families 恢复 384/384/384/349 fixture 的 G1465，而 coarse 为 2233；bridge weight2 被 W 条件正确阻止。完整 proof note 和独立对抗审查尚待完成，此处保留为阶段统计。

- 08:52 主报告完成：阶段研究材料。正式采用统一 earliest-cohort cut ownership，同 cohort 才 physical-left；此细化取代初始“跨 cohort 仍 final physical-left”的设想，避免 old loss/final birth owner 转移。每 final key 仍有唯一 producer，可用 uniform 既有 Positions leaf/concat，未采用新 payload 变体。
- 结构方案：窗口旧 key topology 冻结，selected/retired 保留 zero tombstone；K0 按 immutable key membership 查询，不看 count>0。旧 count 更新需真正 AtomicU64 或已证同步，不能把共享 u64 cast 成 atomic；结构删插在 drain。O(B)固定 OnceLock descriptor registry 及 published-prefix 读取，old spans 冻结，reserved 低 ID 在简化 dense 方案 drain，active reuse 原输入重启。08:54 更正：Directories 长度须 base+published prefix，而非冻结 Corpus.id_count；新 descriptor 内的 activated identity 也需识别 reuse。
- 准入/执行边界：自身 count final 与 physical support stable 分开；crossed-role 或 K0-retained 可能邻接等待较早 W seal，缺席 cut 用前置 port 投影 skip。后 cohort apply 还等所有较早 prepare seal，省 prior-start registry 却放弃较强 early apply。严格 count>所有 G 避免未决 newborn tie。全部 birth/zero payload closure 之后才能撤 guard。
- 主有限模型：9,152 cases、8,703 非空 window、18,298 rules、66,654 accepted、7,535 AA、6,934 zero、253 multi-rule seeds；379,414 coarse/376,756 refined guards，6,220 jobs refined 更紧。eager raw boundary 记录 100,587，final87,312 为合成工作计数，非实际性能。原输出无精确运行起止；脚本 SHA `ba18ebc74761…`，08:55 核对报告 SHA `349bb0dc4baf…`。
- 独立 family 模型 08:52:48.178190—08:53:01.426425：11,036 cases、88,865 producers、188,376 jobs、1,556,463 completed-job envelope subsets、2,086,628 boundary-prefix checks 通过，含 cap1—5。非 coherent 混合值分组的 1+1 最终合为 2 负例保留。该模型仅证明 original-cut/family/ownership 有限性质，不验证异步 scheduler、Rust 并发、codec 或内存。保存于该项验证记录，脚本 SHA `043a9ec6f206…`。

- 08:57 新增 source-reachable 合成工作删除 fixture：n 个不同 unit-weight ABBCD_i rows，另 AB weight2、BC weight1，F2、target≥n+6。AB(n+2)后 BC(n+1)因 crossed role 分 cohort 并等待 W；G_AB=n 使 BC 可在 W 后进入同 window。省掉临时 XB 的 n 个 physical postings，最终 XY n 只产生一次；BC 无 retained side、G_BC=0 可省其 final emission。说明即使 AB/BC prepare 完全不重叠也可删除工作；不是大重复 word-weight 情形，也不是 native 加速证据。

- 09:03 核对独立 final family 审查：原 08:52 初跑保留为历史，最终归档重跑 08:57:15.159462—08:57:28.592325，11,036 案例等计数不变；duplicate generated words 统一后保存，脚本 SHA 更新为`13b48600b6a9…`，取代初步`043a9e…`版本。报告 08:58:20 首次保存。
- second-hop 补充 08:55:38.484165—08:55:39.718402：64,440 bypass-ready cases、317,100 partial-write stage combinations、290,424 retained second-hop 读取检查通过；singleton 后 cohort 条件下可能使预测由 false 变 true 的旧 writer 正好被 crossed-role W 拦住。若扩大后 cohort，必须对全部 selected peers 检查，不能只检查当前 q。脚本`check_second_hop.py` SHA `78cd9bf3a3fd…`。
- 新负例/边界：old heap 空仍可藏 ABAB→XX，须 drain 而非 Finished；不同旧 pair 的拼写可同为 abc，pending identity 也属 active reuse；zero jobs 数值 credit0 不表示 payload 完成。独立报告阶段研究材料 SHA `a2ae776f915c…`。不提供 safe Rust scheduler 证明。

### 173｜fused producer 准备的全生命周期内存界

- UTC 对照：开始／结束：10-12 00:25；10-12 00:34；10-12 00:44。

- 开始／结束：2026-10-12 08:25—08:34（报告所记阶段），08:44 核对保存材料。
- 工作：区分 live encoded payload、容器 capacity、重分配瞬态、RSS，审查提前 concat 释放与其他 producer 源数据、sort/child/result descriptors 共存的峰值。
- 结论：zero-copy concat 不能保证总峰值不升。相同 leaf 构造且旧 Batch 留存至全局 join 时，融合不增加最大“语义拥有的 encoded coordinate data”，但此窄界排除 starts、descriptor/capacity/scratch 等。固定至多 min(B,P)个 producer 一波可约束并发临时 envelope，不保证 baseline 峰值相同；普通 par_iter 不自动给至多 P 个 live producer allocation。
- 排序控制：Pair+原 ordinal 原地 unstable sort 能去 full-array stable scratch，维持 leaf 次序；额外 ordinal record 和 grouping/output 峰值可能抵消节省。若 grouping 本已主导，移除 sort scratch 不能独立证明 reducer 峰值下降。
- 有限验证：15,872 反例不等式、379,776 stable/ordinal 阶段 envelope、9,841 ordinal 等 key 顺序、5,120 wave 上界、2,775 release-credit 不等式通过。抽象单位，不猜 Rust layout；原输出没有精确进程起止，未补跑。
- 证据：阶段研究材料 SHA `4551a31f9b26…`；`check.py` SHA `b434bbf6a606…`。无仓库执行、allocation/RSS 或速度测量。

### 174｜从原压缩 cohort 重放未再消费的新 output heads

- UTC 对照：开始：10-12 00:43:12；结束：10-12 00:49:06；10-12 00:47:06.627853；10-12 00:47:08.977002。

- 开始：2026-10-12 08:43:12。
- 结束：阶段报告 08:49:06 保存；最终模型 08:47:06.627853—08:47:08.977002。
- 条件性成功：窗口内 output 首次激活且不再被消费时，保留原 sorted compressed cohort，按 token==X 与 nonoverlap 可恢复精确新 X heads。每非空 job 保存真实 first/last accepted 坐标，drain 用 iter_from_value 重放该区间，AA 也不需保留全 accepted 向量供最终 replay。
- 反例与成本：AAAA 原 cohort[1,2,3]，任意 raw task 从 2 开始会误收 X 尾部并漏 head3；语义边界 seed 必要。每 job 两坐标约 16 逻辑字节，另计 bookkeeping、seek 约 O(128)重复 prefix、第二次历史 decode、weight-region lookup 及 final grouping。释放 starts 必须等 apply 和所有既有 registry 读者完成，final replay 不自行取代早期 prepare registry。
- 内存否定：F1，CA2、AB1、(CAB)^M 零权重，先 CA→Y 后 AB 仅 1 个 accepted start 却保留 M+1 历史 posting。M=10,000 时保留 10,001 坐标只替代 8 逻辑字节 raw start（该小 starts 实际可 inline）；没有普遍 HWM 下降保证。
- 有限验证：最终 7,034 案例取代初步 7,033；28,714 producers、5,288 AA、16,319 低 output IDs、56,862 heads 含 15,618 零权重、41,843 数学 tasks、79,753 final payload 比较。whole replay 解码 97,653 与 bounded intervals63,279 未计压缩 seek-prefix。随机合法 fresh history 不验证 authoritative priority/window scheduler；memory 定向 fixture 符合 exact priority。无 codec/Rust 并发/真实内存测量。
- 证据：阶段研究材料 SHA `b808240e9a1b…`；`check_replay.py` SHA `eabb7321a9b1…`，results 保存精确运行时间。无仓库执行。

### 175｜冻结窗口的实际 endpoint 事件与 readiness 对抗审查

- UTC 对照：开始：10-12 00:54:33；结束：10-12 01:02:22；10-12 01:13；10-12 01:00:58。

- 开始：2026-10-12 08:54:33。
- 结束：阶段证明 09:02:22 保存；09:13 本日志核对。positive event run 完成时间只知早于 09:00:58 hash 检查，不补造精确起止。
- 条件性结论：初始 compatible cohort 后只用 singleton cohorts，W 含 support 写入与 old-loss ACK，后 apply 遵循全部更早 P-prefix，pre-context K0 gate 先于 geometry；在这些真实 completion happens-before 前提下，matcher support 及 retained event/selected-prediction 稳定。原始 second-hop 值可以变化，不得声称整个 context 冻结。
- 有限验证：29,046 cases、603,030 去重 SC reader/writer states、139,308 probes；每 probe 另查 unsealed cells 独立可见性，共 535,990 states，允许跨 cell 不同于 source 写入顺序。含 13,332 非 unit spans、16,496 AA、18,000 pruned edges、1,632 multi-rule seed、750 较长随机 case；至多 3 个同时未 sealed occurrence writers。
- completion 模型：145 positive states 未早选；撤掉 ACK 依赖后 14 states 出现 observed q4、真实 q2 而 competitor3 的错误。32,481 单调 tombstone order/prefix 检查通过。
- 负例：缺 crossed-role W 会收重叠零权重 support；缺 retained-witness W 会重复 cut owner；缺 P-prefix 会读超 published descriptor 前缀的 new ID；只给每 rule readiness 的后续 multi-rule cohort 可错预测 peer。此扩展必须取全 cohort W 依赖并集。
- 源码/Rust 范围：既有 Relaxed endpoint 操作可在真实 join/Release-Acquire/task handoff 建立的 happens-before 下保留；整数状态碰巧完成不够。没有证明具体 Rust fan-in、借用、Atomic64 目标支持、heap/guard 算法、codec、cap/reuse 或完整 end-to-end 实现。
- 证据：阶段研究材料 SHA `79da3894ecd1…`；`check_events.py` SHA `6fbc25cdc8ec…`；`check_completion.py` SHA `ce736b6bc0e1…`。无仓库执行/性能测试。

### 176｜新 InitialTokenIds 提交及最新证据适配审查

- UTC 对照：开始：10-12 01:02:47；结束：10-12 01:04:40；10-12 01:06:23；10-12 01:08。

- 开始：2026-10-12 09:02:47 观测分支更新后。
- 结束：09:04:40 源码审查完成；09:06:23 作者测量细分补充，09:08 本日志核对。
- 源码：新 head `a583a0b4025495ef0006a6adea84c6139cb7b295`，较 a2b20f32 ahead3；中间包含`62263dbd` InitialTokenIds 与`76f08366` evidence。
- 时间区别：head commit timestamp 为 2026-10-12 08:46:57；中间提交 author 时间为 Oct12 04:12/04:16，但 09:02 才观测其已发布，不能当成早先已读取的新 baseline 或精确 push 时间。
- 工作：只读比较初始 ID 结构、相关路径及作者实验；现有理论/模型仍保留其 a2 固定来源，待确认哪些前提改变。

- 源码结论：两个 production edits 都只改 vocabulary.rs，未发现新正确性缺陷。InitialTokenIds 接管 initial 映射/affix cache/activation；a583 的 Chars+first 状态在过滤前推进 first，以 original remainder 判断 last，保留过滤头尾及 UTF-8 语义。merge/index/corpus 合并循环未改，不需重置 prepare/commit 理论，但 full-train 比较必须固定初始化版本。
- evidence 范围：`76f083668ae28e7fe5ad5230c93149d81ea239c3`共 546 evidence files，aggregate compare 的 300-file 截断经分页审查补齐；未把 frozen source 副本误当 live trainer 变更。实验实际测`62263dbd97f7d6eb6e4d34541cf6441c032b41c1`的 char_indices constructor，不是 a583 最终 loop；overlay build 的 source_commit 字段不能单独识别候选。
- 作者结果：4workers、CPU0—3、Chinese/English prepared Whitespace、both affixes，各 primary/reverse 每 arm 一次，共两次普通候选观察。候选 init 相对 flags Chinese +14.68%/+6.64%，English +12.22%/+2.68%；whole-training 无稳定一致收益。plain init 仅约 2.8—4.0ms，百分比波动不扩大解释。all-normal36 行含 cold/suffix 不同候选，不能当 36 次 ordinary repeats。
- 作者核验：记录的各 arm model_sha 一致，非本研究独立解压比字节；allocation 四对记录相同，rows12B、InitialTokenIds64B、no-affix table0。README 称两 feature 设置各 30 BPE tests 通过但排除已知 JSON-roundtrip 失败，未 run Miri。以上不能证明新 a583 实际测试或性能。
- 数据缺口：所查 summary/init profiles 没有 producer local/global maxima 或一次/两次改变 cut 比例；不称已遍历全部 546 文件内容或全 repo 不存在。证据阶段研究材料（SHA `3677147ca33e…`）含固定源码和 README/primary/reverse/allocation 链接；只读审查无仓库执行。

### 177｜保留 cohort joins 的 old-input window 简化

- UTC 对照：开始：10-12 01:03:52；结束：10-12 01:05；10-12 01:08。

- 开始：2026-10-12 09:03:52。
- 结束：09:05 阶段材料保存，09:08 核对。
- 工作：保留每 cohort prepare/apply/scalar commit joins，仅把 newborn payload/publication 推迟至 window drain；独立核对 ABBCD_i fixture 仍可删除临时 XB 工作，而无需 async scheduler、P/W protocol、AtomicU64 或 OnceLock descriptors。
- 新上界：producer 全部 jobs 完成时 G=各 job 完整 child 最大值之和已安全，不必保留 raw residual f−Σd；因此 joined 设计可完全省 exact accepted-mass instrumentation，并消去 AA raw slack。该简化正在交叉审查。
- 状态：设计和 break-even 账本编写中，没有实现或速度证明；原 async 模型结果不被改称这个更小方案的 native 验证。

- 完成成本结论：ordinary map 原 entries 保留 zero tombstones 供 K0.contains_key，不需 clone K0；exclusive owner 每 cohort 更新，selected 与首次 live→zero 记录 Pair 列表，drain 仅删日志。best 必须明确跳过 present-zero tombstones，old heap 空仍须 drain hidden births。原 Positions 表示及 pair 内并行不变。
- closed guard 支持：G_complete=Σjob c_j，global grouped max≤G_complete≤带 raw slack 的 progressive guard；更紧 global scalar grouping 另付 hash/sort/alloc。无需 accepted-mass 热循环计数；有 G<F 可跳整个 producer final traversal，包括 zero payload，但已启动 prepare/apply/loss 必须完成。
- break-even：省 transient 编码/归并/route/heap/loss/drop，减去 family tallies、额外 final starts/weight/context 遍历、tombstones 和延迟释放。一次改变 cut 也多付 family+final scalar，收益依赖跨 cohort 两次改变比例 C2，不等于省一半 prepare/commit。必须共同 baseline 启用 witness/terminal 控制以免重复计功。
- 有限数学检查：09:05:17.504423 运行 21,297 个 1—3 jobs/families、weights0—2 矩阵，raw slack0/1/4，上述不等式通过；工具记录 0.061507 秒，没有独立结束绝对时间。只验证 scalar 不等式，非 full scheduler 或 trainer。阶段研究材料（SHA `f108f7b8f8da…`）、`check.py`（SHA `1a59e919bf04…`）/result.txt 保存，无 native 测量。

- 独立 joined 后续 09:06:15 开始，09:10:03.423313—09:10:16.714501 执行，09:10:58 保存：13,021 cases、53,700 逐 rule exact greedy 比较、102,236 closed-bound key 检查、11,797 AA、6,192 后续 multi-rule cohorts 通过；511 old-empty/hidden-nonempty drain。脚本`check_joined_window.py` SHA `f94f244e88de…`。
- compatible lift 结论：每个新 cohort member 都须 strictly 高于所有 prior producer G；通过概念上的完整虚拟 queue 沿用原 compatible-prefix 定理，无需真实插入 hidden births。当前 cohort 自己的 G 不能回头阻塞其内部已合法 prefix；只测首 member 不够。model 只到 window drain 与 scalar frontier，不覆盖完整 reuse trainer、codec、真实并发或性能。

### 178｜joined window 延迟 old-loss overlay

- UTC 对照：开始：10-12 01:07:30；结束：10-12 01:09；10-12 01:11；10-12 01:13。

- 开始：2026-10-12 09:07:30。
- 结束：09:09 阶段报告、09:11key-sharing 限制补充完成；09:13 核对。
- 工作：保留完整 grouped old-loss logs，让 optimistic heap 候选通过 base−logs 精确修复，drain 再 bulk scalar fold；研究能否省每 cohort 重复 owner 工作，而不 Positions 重数。
- 待审查：每次 repair 跨 loss-log hash probes、retained map 内存、lazy correction 和 floor/故障契约；尚无正确性完成或性能结论。

- 结论：valid fresh joined 状态可精确延迟 OLD map commits；top optimistic heap 按 base−完整 logs 修复，changed head 必须 pop/update/push 再比较，selected AA 是独立 tombstone 不能从 0 再扣 exterior loss。cached frontier 只能避免重复 sum，不能让无关 log misses output-sensitive；drain 从未改 base 扣 total，不能对已修复 cache 二次扣。
- 代价/否定默认：Mloss records→U affected base writes 仍需保存/index M；cohort map 查询 O(B)probe，run compaction 另付 merge traffic。可实现 crossed-chain B=256、M=U=256 却需 32,640 空 probe 后再 drain256updates，远多于 eager256 次。故数学成立但缺少低重复语料收益证据，不建议默认采用。
- 验证：5,000 valid monotone-loss schedules、88,653 exact head checks、119,476 heap corrections、1,108,545 cached probes。仅 scalar ledger，不是 BPE 语料/overflow/Rust 模型；原输出无精确进程起止。`check.py` SHA `a452feb7e530…`，results SHA `717863838071…`。
- 错误/共享成本：F5 base6 losses2 再 10，baseline 先退休后忽略 10，aggregate12 却 underflow，故 valid-history 证明不等于非法 stream 错误次序等价。family reducer 与 loss overlay 可共用调度/storage 工具，但 family key 和 removed old pair 不同，不能声称一次 sort/hash 自动算两者。

### 179｜完整 coherent family 低频过滤与 unit-context 工作删除

- UTC 对照：开始／阶段完成：10-12 01:10；10-12 01:12。

- 开始／阶段完成：2026-10-12 09:10—09:12。
- 工作：把 producer 完整 family scalar reduction 从更紧 admission guard 推广到永久低频后代过滤，并核对 one-cohort 收益及真实 leaf allocation。
- 条件性成功：完整 family F_k<floor 时，其 window 内所有后代 full key 都不能达 floor。全部 families 低频则 producer apply 后可释放所有 starts、省整个 final traversal，保留 output descriptor 及 old losses；混合 producer 仍须扫 starts、读 final endpoint 并做 side/port membership，不能免费省其匹配访问。
- 合成 fixture：n>2，各 n 个 unit ABc_i 和 DBc_i，F2，AB/DB 共享 tail 兼容先选；旧 Bc_i 每个 2 所以单纯 old-witness gate 均通过，所有 Xc_i/Yc_i 却各 1。n500、P4，每 producer tasks256/244，sum-local-max2 不能证明低频，完整 global max1 可以；省 1,000 temporary encoded coordinates 及相应 fragment route/group/drop，无 newborn heap/root 可省。target 留第三 output 余量，收益不能归 terminal。
- 源码精度：one-position leaf 数据使用 SmallVec<[u8;16]> inline，不声称每 leaf 分配/释放 heap。只证明少做编码和 descriptor 工作，global scalar reduction 与 persistent eligible-key 查询仍计成本。
- 小模型 09:11:23.617722 记录，n=3/4/32/500 通过，工具耗时 0.000849 秒（非 native benchmark）。脚本`check_low_repeat.py` SHA `475beb524ed7…`。
- 证据：阶段研究材料§9。seed-only 不省 transient 历史的早结论仍对，但不能扩大成 seed-only family 预过滤没收益。真正 single-cohort 可复用完整 counts 做 encode-only 第二遍；multi-cohort family 会 split，需新 final scalar tallies，该接口分支不能免费算入统一方案。

### 180｜冻结窗口 mutex driver 的 reservation、错误及 join 契约

- UTC 对照：结束：10-12 01:10:30；10-12 01:09:08。

- 开始：阶段 175 事件审查后续；精确开始秒未保存。
- 结束：2026-10-12 09:10:30 阶段报告。模型完成早于 09:09:08 hash 检查，未单独保存运行起止。
- 工作：单 scope、短 mutex driver 把 reserve、unlock、spawn、finish、acknowledge、callback 返回及 scope join 分成独立事件，检查 lost wake、错误和 terminal 退出。
- 条件性成功：reserved 工作在 unlock 前计 outstanding 且必须有人持有 dispatch/cancel 责任；每 callback 锁内 drive，first error sticky。ordinary drain 完成所有已 admitted phase；error/terminal 可停未 reserved 工作，已 reserved/running 仍 join，真实 error 覆盖 terminal 成功。ready flag 不是 scope join，不能据此早发结果或回收源数据。
- 有限模型：232 配置、82,280 状态，最大单例 13,520；5 mutants 分别暴露未计 reservation、丢最后 drive、error 后丢 reservation、后 success 覆盖 error、join 前 publication。job credit 不是 phase runner 结束或 source lifetime 结束。
- 证据：阶段研究材料 SHA `194ac8d192d5…`；`check_driver.py` SHA `04329bb64381…`；driver 结果 SHA `1cadc98c14f7…`。
- 边界：不证明真实 Rayon 借用、panic/cancellation、memory fan-in、heap/guard 正确性或公平性；mutex 内串行 selection/string/heap 工作另计。没有仓库执行或速度测量。

### 181｜完整 family 的 threshold sketch：安全但成本不利

- UTC 对照：开始／结束：10-12 01:15；10-12 01:17；10-12 01:18。

- 开始／结束：2026-10-12 09:15 阶段材料，09:17 补充筛选方案；精确探索起始秒未保存，09:18 核对。
- 方法：非负 family scalar records 用 floor 饱和 Count-Min upper bound；query<F 只会安全拒绝低频 family，hash 碰撞只损失优化机会。单 producer 流中取每次 update 后的 point-estimate 最大值 U，最后 U<F 可证明全部 families 低频，无需全 key 二次查询。
- 限制：U=F 表示 unknown，不能当 numeric admission upper bound；max 各 job U 不安全，相同 family 跨 job 各 1 可达 F2。只对 max(local c)<F≤Σlocal c 的 ambiguous producers 试探；首次饱和可 early-abort 回 baseline。merging、reset、hash 和串行 producer 尾部仍真实计费。
- 成本否定：H500、F2、ideal 独立 uniform hashes 下，8rows×1000 u64 counters 为 64KB，online whole-producer 优化失败 union bound 约 3.8835%；已有 500 个 16B scalar records 仅 8KB。64KB 不是小于 8KB 的方案；2-bit 等紧凑表示另有维护成本，不能默默假定。只拒绝大多数单 key 不等于能释放整个 producer starts。
- 有限数学检查：09:15:11 执行，20,000 随机 trials、638,644 point checks 通过，含 standard/CU saturated updates、online max、componentwise merge、weight0—7/floor1—11。不是概率正确性证明或 native benchmark。
- 证据：阶段研究材料 SHA `1efa1d8bdf97…`、`check_family_threshold_sketch.py` SHA `04530e6a5c3d…`及 result.txt；Count-Min/CU primary papers 按报告标明全文与仅索引片段来源。决定保留 exact compact scalar sort 或 reusable dense 数组为更简单对照，不推荐 fixed-u64 sketch 默认采用。

### 182｜新 Affix 完整验证和初始化测量证据复核

- UTC 对照：开始：10-12 01:22:05；结束：10-12 01:24:10；10-12 01:25。

- 开始：2026-10-12 09:22:05 观测新 head 后。
- 结束：09:24:10 superseding 审查保存，09:25 核对。
- 固定来源：`5c7ef814f55da46cd7c820173cc72053fe092d74`，较 a583 ahead1，commit authored09:07:06；消息为“Record restored Affix initialization performance and full validation”，路径`affix-activation-20261012`。
- 工作：只读审查新证据是否覆盖 a583 最终初始化 loop，完整验证范围及实际性能；标题本身不当作通过或速度恢复的证明。原阶段 176“最终 loop 未测”只代表当时已查包，待新数据核实后补充。

- 已核实：512 changed files 为 511 份新 evidence 加旧 README 更新，无 production 修改；最终 candidate frozen vocabulary.rs 与 a583 逐字节一致，base 与 a2、flags 与 83630bd0 亦各自一致。三 arm mod.rs instrumentation 相同。build JSON 的 a2 substrate 不能取代 overlay 源码身份。
- 最终作者测量：candidate/flags 算术均值之比，Chinese 两 pairs init wall−2.07170%/CPU−2.18726%；English 四 pairs wall+0.22629%/CPU+0.34349%，保留+11.28809%高样本，wall 均值绝对差约 0.115ms。小样本观察支持相对 flags 恢复，不证明统计等价、普遍无回归或相对 a2 baseline 恢复。
- baseline 只有各语言一个最终 affix 同 suitepair：Chinese init+1.21445%，English+20.07207%；不能与 flags 混称。18 completed final processes，all-normal44 行其余 26 为被弃候选，四个终止 screen 不算模型完成；不是 44 次 retained 候选重复。
- 作者验证：default/no-default 压缩日志按数据读取解压，各 30pass、0fail、8filtered，5.51s/5.20s；仍排除已知 JSON-roundtrip，非未过滤全 suite。作者记录 Clippy/rustfmt 及无 Miri；本研究没有执行 tests。最终 allocation 各 pair delta0，snapshot 与记录的 model hashes 一致；非本研究重跑。
- 状态更新：阶段 176“最终 loop 未测”由本次新证据限定性取代，旧 62263db 回归仍保留其历史版本。阶段研究材料更新 SHA `c0a87a36b61c…`；固定 evidence 链接见报告。prepare/commit 代码及缺少 family 分布统计的结论未变。

### 183｜基于任务坐标 envelope 的局部 phase readiness

- UTC 对照：开始：10-12 01:27:56；10-12 01:34；结束：10-12 01:33:12；10-12 01:40。

- 开始：2026-10-12 09:27:56—09:34 持续探索。
- 结束：主 source/proof 及事件检查 09:33:12 前完成；09:40 核对。
- 工作：用每 job corpus read/write 保守 envelope 与已排序 producer 数组的 partition_point 表达依赖，研究 O(J)counter metadata 替代全 producer fence；overlap report E 可二次，AA 未来 reader 注册须单独处理。
- 文献：09:31—09:32 核对 Legion Depoch2018、visibility2023 及 OpenMP5.1 arbitrary-range 限制，作为任务依赖/可见性框架对照，未迁移性能数字。
- 状态：不增加位图或 Positions 重数的方案仍在推导；未证明 event 协议或成本收益。

- 完成结论：以每 job 读/写保守 coordinate hull 及 phase DAG 保证相交旧 writer 完成、同/后 writer 等待读者；producer 内 bounds 单调，用 partition_point 两次查交集，无需 interval tree。只存 O(J)bounds/counters，completion 重新报告交集，时间 O(BJ log Jmax+E)，E 可Θ(J²)，稀疏 posting 跨度也可令 hulls 几乎全相交。
- AA 限制：canonical-start scan 本身是 reader；后 neighbor tasks 尚未注册时不能放行 writer。可先保守全 hull reader lease 到全部 AA 准备完，或保持 joined AA；late-registration 不是补 dependency 就能修复。halo 用 checked/saturating 后 clip，numeric loss ACK 依旧独立于 spatial readiness。
- 有限模型 09:32:49—09:32:58：1,500 cases、15,303 jobs、163,962 admission/load/store/completion events；61,212 interval queries 与 186,935 overlap reports 对全 scan oracle 一致，86,045 loads/42,313 stores。random SC 调度，复用既有独立 event 语义，不是独立 matcher 实现或全部 schedule 枚举；AA 注册、greedy priority、Rust visibility/借用及最终 Positions 未模型化。
- 证据：阶段研究材料；脚本`check_intervals.py` SHA `0212e440889a…`。保留 joined window 为工作删除对照，本方向属额外 scheduler 工程，无性能保证。

- 09:42:02 实质更正：先前 AA 只注册未来 reader covering lease 不完整，漏掉未来 writer 尚未生成时较晚 reader 对其依赖。须从 admission 起同时公布 aggregate P 和 W：P 覆盖 canonical scan 及全部 future neighbor readers，W 覆盖全部可能 apply stores；P 到全 AA prepare 完才退休，W 到全 AA apply 完才退休。内部 jobs 为 aggregate 子工作，不做中途 promise 替换；牺牲 fine-grained AA overlap。non-AA 证明/原随机模型不受此缺漏影响，修正后 AA 尚待独立模型。更新阶段研究材料 SHA `6b298d894cf2…`。

- 10:00 核对独立 source-shaped interval 验证：非 AA run09:44:04.318706—09:45:12.783527，9,853 cases/133,606 jobs/959,905 events，17342 zero accepted、183875 原 stale 拒绝；改正 aggregate AA P/W 后 run09:47:41.127798—09:48:12.584730，5,000 cases/64,489 jobs/533,982 events、6,621 aggregate AA phases、9,605 zero accepted。另 81 abstract phase graphs/3,117states。全部 SC 事件，独立逻辑 token 参考对 raw endpoints；不证明 Rust 内存模型或真实性能。
- 独立负例包括 AABC 中 AA 未来 W 缺席令 BC 读旧 A、placeholder 拆成两个 writers 未迁 counter、仅等自己 reader 而漏同 cohort peer reader。`check_model.py` SHA `25bee285fb8d…`；`check_aa_model.py` SHA `4de390092146…`，results 保存时间。修正后 AA 有限模型通过取代先前待验证状态，仍保留宽 hull 带来的低并行度。

### 184｜pending child 的 lexicographic full-priority guard

- UTC 对照：开始：10-12 01:33:28；结束：10-12 01:36；10-12 01:38:48；10-12 01:40。

- 开始：2026-10-12 09:33:28。
- 结束：主报告 09:36 保存，独立边界检查 09:38:48 完成；09:40 核对。
- 新方向：在窗口仅允许 append IDs 时，coherent child key 后续不能 lex 下降，尝试把严格 count 准入收紧为完整 priority 认证，允许部分 equal-frequency 旧 rule 继续。
- 候选证书：每 job 保留最大 family mass c_j 及达到该最大值的最小 Pair；G=Σc_j，T 取 positive jobs 的这些 argmin 的最大 Pair。若某 child 达到 G，必须饱和每 job，因此可能由(G,T)界定 tie。
- 状态：主证明与独立对抗检查刚开始，numeric-low reserved 必须 drain，尚不当作成立的通用 guard 或 native 收益。

- 证明完成：append 且全局唯一 output 使 coherent child family key 只能 lex 增。G=Σpositive job 最大值，t_j 为该 job 所有 maximizers 中最小 Pair，T=max t_j。child count=G 才用 tie 且必饱和每 positive job，故 priority(G,T)安全；zero-mass jobs 不能参与 T。对 old-old q 可压缩成 old-left ID 阈值，G 相等时 q.head≤阈值即胜；无需 global family map。
- 主模型是 adapted joined 模型，不另称第二独立实现。09:34:20.048598—09:34:42.544161：21,021 cases、89,230 exact greedy 规则、183,545 full-priority key 检查、27,495 equal-count；09:35:34.982672—09:35:59.529883 实际用更强 saturation 准入：21,021 cases、91,966 规则、204,571 key、31,700 equal-count。生成后续 case 受轨迹影响，两 run 不是 paired 收益比较。directed 六例 09:34:55.022155—.023094。
- 独立负例 09:38:48.020759—.022632：低 reserved producer 破坏 new-head 比较、后续 low output 使 saved key lex 下降；两个输出 alias 可使 families coalesce（快照例，不冒称完整初始化可达）；zero job 入 max、任意选 tied argmax 而非 minPair 均不安全。脚本`check_lexicographic_boundaries.py` SHA `8f98551af02b…`。
- 合成全 unit fixture：n 不同 ABBCD_i、F2，AB/BB/BC 均 n，先 AB，BB 冲突止 seed；AB→X hidden XB=n，full-priority 允许 BC=n 胜 new-head XB 而同 window 继续，最终 XY=n。n500、target≥506 留非 terminal 空间，P4 一 leaf tasks128/128/128/116；省 500 个 transient XB 坐标，final500 仍需，不等于 native 时间减半。
- 证据：阶段研究材料；脚本`check_joined_lexicographic.py` SHA `19867cdc294d…`，`check_joined_saturation.py` SHA `a1fb5276cb2c…`，`check_directed.py` SHA `a1964e6c499e…`。checked G≤raw f，reserved/alias 边界不能删除，无仓库执行。

### 185｜按固定 family 分桶的 local-MAX guard

- UTC 对照：开始／结束：10-12 01:38。

- 开始／结束：2026-10-12 09:38 报告整理；更早推导起点未单独保存，模型完成时间如下。
- 方法：每 job 每 bucket 取完整 family 质量的 MAX，再跨 job SUM，producer 取 bucket MAX。所得 M_exact≤G_bucket≤原 sum-local-max；不同于把 bucket 内 family 质量 SUM 的早期粗 guard。任意碰撞仅变松，一致 seed/canonical key 是安全前提，width1 退化原界。
- 成本与限制：w8 每 job64B 对原 8B 多 56B 逻辑 summary、Jw 初始化/相加及 H 个 family hash；不改 Positions、不加 global family map。许多 unit families 使每 job 所有 bucket occupied 时仍 G=J，无法替代 exact global 低频过滤。nested partition 增宽单调更紧，任意换 seed/模数不保证单次更紧。
- 附加界：unique-argmax 与 second-max gap 可用至多 J winner fingerprint 归并收紧，collision 只变松；额外 second-max/tie 状态仍计费。相同 partition 下 bucket-MAX 至少同样紧，不推广成不同 metadata 预算的统一胜者。
- 有限数学：09:37:02.824597 记录完成 1,321,356 matrix/partition cases，报告耗时 17.411155s；09:37:19.910979 完成 166,866 gap cases，0.581989s。仅 integer inequality，报告未给独立归档脚本路径/hash，未为日志补跑。没有 native family extraction/布局/并发/性能验证。
- 证据：阶段研究材料。frequency refinement 可用严格 count 准入；tie 与 saturation 结合需单独保持全 priority 界，不能任意拿 local 最大 family key 冒充全 bucket tie。

- 09:43 新增可复现归档：独立 checker 实际复跑 09:42:35.262628 开始；1,321,356 bucket matrix cases 于 09:42:52.472231 完成，166,866 argmax-gap 及 same-partition dominance cases 于 09:42:53.679073 完成。原 inline 时间/计数保留为历史，这次明确是真实新运行而非补造原时间。
- 归档`check.py` SHA `99b7c479583d…`；result.txt SHA `84165ae9ecc6…`。标准库 integer matrices，无 repo imports/execution；本日志只核对没有再运行。

### 186｜right-token prefetch 新配对实验审查

- UTC 对照：开始：10-12 01:40:30；结束：10-12 01:40:42；10-12 01:46:36；10-12 02:00。

- 开始：2026-10-12 09:40:30 观测新 head 后。
- 结束：source/data 审查 09:40:42—09:46:36 完成；10:00 本日志核对。
- 固定来源：`890ea1f58e65f8d8503f7cd3fc97ad1651eaf284`，authored09:32:42，parent5c7ef8；207 files 全部 experiments，无 production 变更。
- 工作：核对标题“reject right-token prefetch after paired training runs”的实际配对数据、候选源码和否决依据，不先把标题当结果细节。一次不支持的 branch URL 形式报 INVALID_ARGUMENT，随后标准 commits query 及 compare 成功；没有持续访问阻碍。

- 作者否决获数据支持：extra right-token hint 仅加入 fresh prepare 既有 16-position head-prefetch queue；production 保留 head-only。重算 8cases×3 共 24 摘要指标，112 个 raw 文件 Git blob、28stdout、6 个完整 model gzip 解压 SHA 均核实。未执行 author 脚本/源码；数据核验不等于独立复跑或认证机器历史。
- 配对结果：4worker 两 pairs/scene，candidate 相对 baseline training wall/prepare wall：zh Whitespace +2.07382%/+0.57320%；zh ByteLevel −1.64980%/−0.70783%；en Whitespace +5.23348%/+12.21261%；en ByteLevel −1.01027%/+2.21943%。English Whitespace 两 order 皆退化；zh ByteLevel w1 prepare+1.24672%而 total−0.60544%，不把非 target phase 噪声当 prefetch 收益。
- 新当前基线 phase 证据：4worker prepare/train，zh Whitespace2.7900081295/20.7422091850=13.451%；zh ByteLevel9.2193904875/22.8407544620=40.364%；en Whitespace29.901%、en ByteLevel31.855%。prepare 为 joined batch.prepare elapsed，嵌在 merge 内，不是 worker CPU；merge−prepare 包括 selection/writes/commit 等，不能命名为 commit。此前没有 exact-current prepare share 的资料缺口由本包部分补上，但 removable birth encoding 比例仍未知。
- 范围：实语料全 16-bit、0reuse；真实模型 50k vocab 及 ordered merges tuples，不是完整 serialized tokenizer。作者 targeted tests default/no-default 各 31pass0fail8filtered，known roundtrip 仍排除；wide 仅功能 fixture 无实数据性能。candidate+14production LOC/+34tests 且未采用。HWM 含 load 且 allocator/hostswap 不支持内存节省；no-prefetch/distance/neighbor 或 newbatching 未测。
- 证据：阶段研究材料 SHA `53aabfa7d821…`，固定 890ea source/manifest/runner 与原 numerator/denominator 保存在报告链接。否决仅针对本候选/机器/负载，不推出 prefetch 普遍无效或 endpoint latency 不重要。

### 187｜Fagin–Lotem–Naor threshold aggregation 理论对照

- UTC 对照：开始／结束：10-12 01:41；10-12 01:45。

- 开始／结束：2026-10-12 09:41—09:45。
- 来源：作者 host 全文`Optimal Aggregation Algorithms for Middleware`，读取 algorithm/correctness 印刷 pp6—7、Theorem6.1 access 条件及 p21 起 no-random-access 讨论：[原文](https://www.wisdom.weizmann.ac.il/~naor/PAPERS/middle_agg.pdf)。
- 结论：M=max_family Σjob mass 与 G=Σjob max 解释未 materialized children 的 upper threshold，但 BPE 须先证明 immutable no-coalescence 及固定 tie 顺序。论文 sorted/random-access 输入和 instance-optimality 不能免费迁入；不建议每 job 排序或保留通用 random-access 索引，更不 Positions 重数。
- 有限信息下界：只针对两个独立 unit family 集合、threshold2 的 exact deterministic summary，set-disjointness 区分要求不同子集不同 summary；不称无条件 BPE 下界，source 侧信息、adaptive revisit、已保留 scalar records 都可改变模型。
- 证据：阶段研究材料 SHA `9ea27cf7c678…`。只读主文献/推导，没有新仓库执行或速度证明。

### 188｜作者发布 50k ByteLevel 模型的 old-input window 机会 census

- UTC 对照：开始（截至记录，非精确起点）：10-12 01:45；结束：10-12 01:52:51.423075；10-12 01:53:03.026177；10-12 02:00。

- 开始：截至 2026-10-12 09:45 已开始；精确开始秒待记录。
- 结束：主要 census 执行 09:52:51.423075—09:53:03.026177；独立数据检查已保存，10:00 核对。
- 工作：读取作者发布 model merges，统计 old-input-only 窗口的结构上限，区分真实训练模型资料与 synthetic oracle；不运行 repository trainer 或从 Positions 重数。
- 状态：来源清单/模型文件收集中，尚不报告 window 长度、payload 节省或速度结论。

- 09:49 初步 author-model 结构统计：cap256 old-input-only windows Chinese228/English231；加入 AA singleton boundary 为 495/359；final-rank endpoint-compatible oracle 为 1,474/1,211，实际作者日志 1,646/1,335。来源/脚本完整审查待完成，暂作为初步 census。
- 解释边界：最终 model 缺少中途 conflicting queue heads，oracle 不能重建真实 batch schedule，也不能证明 frequency guards 通过或加速。性能设计修正：每个 window 步骤应复用 compatible cohort 并逐 member 查全部 prior guards；later singleton-only 可能把 prepare/apply/loss joins 推近 49k，只宜正确性/debug 控制。既有 multi-cohort 虚拟 queue 证明及模型已覆盖这种 lift。

- 最终四模型结构计数（old-input/AA-singleton/endpoint-compatible）：zh ByteLevel228/495/1474；en ByteLevel231/359/1211；zh Whitespace149/371/1231；en Whitespace257/410/1368。对应作者 actual batches1646/1335/1390/1491。AA control9/9/9，reuse control 因非 append 输出正确拒绝分析。
- 身份/最优性：accepted 五 stream 连续唯一 vocab IDs、exact string concatenation、output=initial+rank 逐条核实；独立 checker 用实际 output 集合而非仅 N−M 推 initial，ByteLevel 初始均 canonical256。全 stream 动态规划核实 greedy maximal-prefix 在这些 hereditary interval policy 下最少 windows；不是在线 batch 最优性。
- 维护成本证据：ByteLevel old-input outer 内 compatible inner cohorts 为 zh1585/en1338；AA-conservative outer 为 1544/1311。en1338 已大于实际 1335，outer 边界可碎化 cohort。“seed 后全 singleton”是 zh40901/36444、en38598/36348 个 groups，远非 228/231；只能作 count 算术，不能当每组等量工作或真实运行时间。
- 复现：`analyze_models.py` SHA `04617b8e9ee5…`；独立`review_checker.py` SHA `c7d10eed93da…`，无 import 前者或 repo；结果 SHA `8ab584fb81b5…`。数据解压/hash/JSON 运算，无 trainer 执行或 guard 成功率/速度结论。

### 189｜历史有序 BPE_RECORDS batch 轨迹与目录初始化量

- UTC 对照：开始／结束：10-12 01:57:59.243729；10-12 01:57:59.412717；10-12 02:00。

- 开始／结束：历史数据执行 2026-10-12 09:57:59.243729—09:57:59.412717，独立审查随后保存，10:00 核对；源码收集起始秒未另记。
- 新来源范围：历史 Chinese ByteLevel/Whitespace 的完整有序 batch sizes 从压缩数据恢复，逐 blob/hash 及作者 summary 核对；此处补足早期阶段 150 只有 summary 不能算ΣV 的缺口。未运行 author probe。
- 结果：ByteLevel1646batches/49744rules，B<4 为 198(12.03%)、B<8 为 267(16.22%)、B<16 为 478(29.04%)，median26/max169；Whitespace1390/32241，对应 193(13.88%)/297(21.37%)/532(38.27%)，median20/max111。百分比是 batch 数，不是 rules/positions/时间。
- 边界审查：映射同 final merge 顺序时，437 ByteLevel 及 394 Whitespace nonterminal 边界按 final-stream 规则还能延长，直接证实最终模型不能恢复 online blockers。历史/最新 model 和总 batch 数一致，不证明逐 batch 边界相同；identity chain 按作者报告与 hash 一致性核对。
- 条件性目录账本：按当前 resize 位置和 append IDs，Σend-domain 为 37,450,623/44,175,229；每非空 batch 至少一个新 Directories 时，两 u32 数组最少逻辑初始化 299,604,984/353,401,832bytes。与假定 4 个 persistent 实例各至 50k 的 1,600,000bytes 比 187.25/220.88 倍，仅逻辑 scalar fill 账本，非实测 allocation/流量/RSS/time，map_init 实例数未知不能设 P。历史实际 record-vector allocation397,952/257,928bytes 属于另一结构，不能混称目录数据。
- 证据：`analyze_historical_records.py` SHA `62a48ec3c0a1…`；`review_historical_checker.py` SHA `de4e09f7de6d…`；`historical-batch-sizes.json` SHA `5ab8e6e9d220…`，均在该项验证记录。只处理既有作者数据，无仓库执行。

### 190｜exact global family direct-ID reducer 及独立复核

- UTC 对照：开始：10-12 01:54；10-12 02:03；结束：10-12 02:07；10-12 02:09；10-12 02:10；10-12 02:11。

- 开始：主探索 2026-10-12 09:54，独立 physical/source 复核 10:03 开始。
- 结束：主报告 10:07、独立报告 10:09 完成，10:10 交叉核验；10:11 本日志读取。
- 成功修正：exact complete family reduction 不必 hash/sort。固定 producer X，用既有 two-side validated direct-ID 目录读其 count-only Changes、按 born pair 恢复 side/neighbor、checked 累加稀疏 groups，并在线维护 global maximum 及最小 maximizer tie；返回 scalar guard 后 drop groups。O(H)record work/O(K)临时 groups，但每 producer 顺序 O(H)尾部及目录初始化另计。
- 源码条件：XX 按 born.0==X 归 right，不能用 bucket 当 side；born_weight0 的 loss-only placeholder 不计 guard。若合并 side vectors 须验证 side+key，不能仅 neighbor。map_init scratch 默认不跨 join 留存，复用 allocation 需显式 attempt/thread lifetime；不能持 RefCell 借用进入 nested Rayon。当前跨 cohort chronological owner 仍需代码改动。
- 功能边界：M<floor 时 apply/readers 闭合后释放 starts、省整个 final emitter，old losses 和 output descriptor 保留；M≥floor 时最小方案不保留 partial-family eligibility，仍发其全部 otherwise eligible sides。更紧 global guard 不弱于 local saturation，但不是 native 更快证明。
- 主模型最终 10:06:39.033463—10:06:42.521100（初次 09:58 统计保留）：105,195 cases、2,205,267 old-priority、2,217,049 descendant checks、448,477 window aggregates。source-shaped fixture10:00:45.011116：n500/8workers tasks128/128/128/116，local 和 bucket G4、exact M1，F2 可省 1000 临时 birth 坐标并释放 starts。没有编译 task/codec 验证。
- 独立 physical-cut 模型 10:06:28.774428—10:06:30.398044：11,050 cases、24,967 exact eager rules、42,456 descendant/no-coalescence、24,012 whole-rejection 等价，含 10,150 zero accepted/4,137 live zero 坐标；保留 10 类定向条件破坏负例。无 native 执行、layout、Rust 借用或性能验证。
- 证据：相关阶段研究材料；主`check.py` SHA `3a89b8965344…`、source-shape `aebe2930abf3…`；独立 physical `826a440e235f…`、negative `001a5a805c71…`。

### 191｜joined window 源码契约的独立边界审查

- UTC 对照：开始：10-12 01:45:49；结束：10-12 01:57:51；10-12 01:59:05；10-12 02:11。

- 开始：2026-10-12 09:45:49。
- 结束：09:57:51 首次保存，09:59:05 读回核实；10:11 补入本日志。
- 必需修正：guard admission 必须先于会 append token 的 resolve_merge，不能让 blocked candidate 占最后词表 slot；K0 缺席可因高质量但 length-rejected birth，不能一律解释成 subfloor，正确安全前提是 subfloor 或 length-ineligible 加 monotone spans，保留 initial cap 例外。
- 其他澄清：best 显式 pop present-zero tombstone 利于回收/省修复，但非 selection 正确性的必要条件；window cap256 累计算，不能每 cohort 重置；每 compatible member 只查 prior guard；pending reserved 应 drain 不 take/activation，active alias 原输入 restart；scalar MAX 不可混为全 family u64 sum。
- 六个定向模型 09:54:26.839997—.840115：上述 resolve/length 例外、reserved、MAX 溢出、AA/zero、cap arithmetic。脚本`check_boundaries.py` SHA `49bd255c0d5a…`。
- 证据：阶段研究材料含固定 a2 exact-byte source 清单。新增 standard-library TLS 替代也须说明 pool/thread lifetime、source 参数和 reentry/drop 义务；不把目录填充摊还自动称维护简化。没有仓库执行或性能结果。
- 10:09 后续 scope 探索：已找到可实现 affix planned-output alias fixture，并用现有 span activation markers 分析 reserved horizon H=N+R、O(V)无新 registry；完整材料/精确 fixture 待核对，未当作新模型通过。

- 10:12 新 ordering 边界：hidden-priority guard 必须先于 active-alias restart 检查，不只先于 append resolve。prefix=a、0cd2/xcd1/ab2、F2、target11，ac/ad→acd 后 hidden(0,acd)2 应先填满 target，old(a,ab)2 的 alias 在目标之外；若先查 alias 会不必要地原输入重启并改变执行路径。完整可达 source fixture 尚在整理，所需修复是顺序而非新增状态。

- scope 独立审查 10:06:29 启动、10:17 读回完成，定向模型 10:14:34.881876—.884974。planned alias 完整例：prefix a/suffix b，cab3/aabbb4/babbab2，全部 6 种初始 row 顺序均先两条 count6 规则，第三 count5 碰当前 output；target8 先完成不得查第三，target9 应 reuse restart。joined 保留 prepare_identity 顺序即可用已有 span 识别，无 pending registry 必要。
- blocked alias 例确证为不必要提前 restart，而非已证明最终模型不同。H=target−V+R 在完全 drained frontier 用 existing span!=0 activation O(V)扫描，10,660 小参数 checks 支持 H≤target 及每 first-activation 减 1；不扫描 physical support 替代 activation，不跨 active-reuse 保存证书。阶段研究材料及`check_boundaries.py` SHA `a91c11f8945a…`。

### 192｜retained starts 的内存 drain 政策与压缩收益反例

- UTC 对照：开始／当前状态（截至记录，非精确起点）：10-12 02:11。

- 开始／当前状态：截至 2026-10-12 10:11 所读研究稿已存在；原探索起止未标明，待补。
- 工作：为同一 joined deferred kernel 加入基于实际 retained-start capacity 的 drain 条件，区分 used bytes、capacity、Job 容器、AA/候选/scratch、final burst 和 RSS。
- 条件性界：H 为历史最大完成 cohort 的实际 starts/storage charge，下一 cohort 前保留≤H，完成后更新 H 并超限 drain 可给 joined checkpoint≤2H；不是绝对 budget、总 HWM 或相对原 batch 方案保证。超大下一 cohort 仍可能抬高 H，prepare 瞬态另计。
- 重要反例：pure unit ABBCD_i、n500/P4，双 cohort raw starts used8000bytes；省掉的 500 个 XB 压缩坐标四 leaf 仅 560 used encodedbytes（另计 capacity/descriptors）。额外 4000 raw bytes 可超过被删压缩 stream，不能用删除坐标数量证明省内存。
- 证据：阶段研究材料。仅源码/数学账本，不是 Rust allocation 或速度测量；细节与完整范围仍待最终核对。

### 193｜joined window 的 cost controls 与最终 publisher 审查

- UTC 对照：开始：10-12 02:03；10-12 02:10；结束：10-12 02:21；10-12 02:27。

- 开始：2026-10-12 10:03—10:10 新探索。
- 结束：10:21 cost 报告读回完成、direct publisher 阶段材料已保存；10:27 核对。
- 工作：对照 cheap raw guard、exact/local family guard 和最终 owner grouping 成本；合并多个 cohort 统一 owner sort 可多付 N·H(distribution)≤NlogC 比较工作，不能只计少几次 join。新 producer-complete direct-ID publisher 审查已开始。
- 当前模型：raw guard 控制 10:08:11.779621—10:08:16.192344，108,347 cohort、196,805 families，17,146 raw saturation/21,569 frontier saturation，保留 all-old equality 不安全及 AA plateau 负/正例。只独立 list/count 数学，不是完整 trainer 或 native 测量。
- 状态：最终 publisher、排序/descriptor 内存及可删除工作账本仍在审查，未报告胜者。

- 10:17 publisher 审查发现每 key 单独 Vec 可引入大量 allocation；safe 两遍 flat staging 可替代，但两遍 lookup/descriptor 搬运与 memory peak 均须计费，最终对照尚在完成。

- 10:21 完整 cost report 读回完成：阶段研究材料。RAW/FRONTIER 各 15,000 corpora 实际准入模型 10:18:04.846193—10:18:22.571817，每 policy118,075 exact greedy checks；RAW84,452 cohorts/82,071 windows/max3，FRONTIER87,783/69,455/max4。random 比例不作真实语料覆盖率。
- direct publisher 独立模型 10:15:23.817411—10:15:27.501904：38,235 path/fragment traces、344,115 reducer 等价比较、114,705 floor cases、472 live-zero fragments 及 10,000 route arithmetic；只对给定合法 contractions 归并，不验证 greedy 准入。脚本`check.py` SHA `c652d73a97ba…`。
- Publisher 结论：producer 两 side direct 目录稳定归并、完整 sum/floor 后每 key concat 原 leaves 一次，Roots 须走独立 complete-Candidate insertion seam，不能回现有 concat。birth hash/route H→K_live 成立，但原 combined actions=L+H−E，分离后 L+K_live，action 减少 H−E−K_live 可为负；birth-only drain 才无 E。每 keyVec 可新分配，SmallVec 或 safe 两遍 flat staging 都须计容量/两遍 lookup/搬运及 producer 串行 H 尾部。作为与 window 机制独立的对照，不重复计上游未建 leaf 收益。

### 194｜frontier envelope 省去首遍 birth-family tally 的候选

- UTC 对照：开始（截至记录，非精确起点）：10-12 02:17；结束：10-12 02:19；10-12 02:23；10-12 02:27。

- 开始：截至 2026-10-12 10:17 正在研究；精确开始秒待材料补充。
- 结束：10:19 old-loss-only 报告，10:23 deletion 补充完成；10:27 核对。
- 工作：研究只输出 old-loss 的首次 prepare，使用 frontier 上界而非逐 occurrence birth family 统计，是否仍可精确证明 deferred-child priority 并扩展 window。
- 状态：成本与独立证明正在检查；没有预先声明能删除全部 first-pass family 工作或实际更快。

- 10:22—10:23 独立简化检查进展：先移除 optional K0 witness gate，再保留普通 old-count deletion 而不用 tombstones/retire log/frozen map topology，13,271 同 corpora 及含 zero 的 final coordinates 一致。joined 生命周期在 join 之间允许结构 mutation；完整源报告待核对。
- 对 exact-M 的关联推导：noncoalescence 可能只需 old-ID domain、coherent groups 和独占 fresh identities，而不依赖 K0 membership。源码 scope 复核已启动，尚不把该状态精简推到 async freeze/readers 路线。

- 核心结论：ordinary cohort 完整 selection 后、任何 current writes/losses 前，取 exact remaining-old head(count,head)作 selectable-child bound；AA 用 selected raw(count,input head)。首次 prepare 只做 accepted starts 及唯一 owned 旧 loss，无 born family counts/key/length 预测、accepted-mass 统计。right selected-neighbor lookup/AA right binary search 原为 born 预测，可省；left 前驱判定仍须防重复 old loss。final drain 才重读真实邻居并形成 uniform payload。
- 限制/紧性：FRONTIER 首 guard 固定后，postseed 只允许同 count/head plateau 的低 tail、AA、高 tail 三类 cohorts，故总≤4；RAW≤3，均有小非 terminaltight fixtures。不能把 228/231 结构 oracle windows 解释为 cheap guard 可达。empty frontier 只证明无 selectable child，非无 physical edges。
- 独立 operational run10:19:56.707270—10:19:57.832224：13,271 corpora、14,711 eager 规则、13,415 old-map checks、全 final zero-inclusive coordinates 一致，5,857 AA fallback；814 starts 在 375 空 frontiercohort 后释放，2,852 live zero 坐标保留。删除 tombstones 变体 10:22:43.780105—10:22:44.900374 同套通过，count-onlyoldmap 用正常 take/retire 删除。
- 证据：阶段研究材料；`check_frontier_loss_only.py` SHA `0b629bd6b55e…`，delete 版本 SHA `5f12c2075afe…`。无 K0 运行时 gate 则不必保存 frozen topology；若保留 gate 仍须保存其语义视图，不能删 state 留 probe。没有 native 执行或速度结论。

- 10:51 正式正文前提澄清：四 cohort 上限依赖累计 window cap256 且容量截断即 drain。如果每 cohort 重新给 256 额度，同 head plateau 超过 1280 个 keys 可切成六批，不能继续声称≤4；这是窗口定义的边界核对，未另跑模型。

- 10:55:44.462400—10:55:44.463498 新增独立定向数学执行：1,281 个二 token 词(0,i)，每词 weight1，只给每 cohort cap256 而不限制累计 window 且不触发 memory stop 时，所有 prior FRONTIER guards 通过，得到[256,256,256,256,256,1]六 cohorts、1,281 rules、零 birth。该检查验证缺少累计 window cap 会破坏≤4 说法；不否定已经补齐前提的方案，不是仓库执行，也不倒写成 10:48 已运行。

### 195｜只等待影响当前 old head 的 loss ACK

- UTC 对照：开始：10-12 02:20。

- 开始：2026-10-12 10:20。
- 结束：进行中。
- 工作：针对 q=(u,v)，检查只有 pending r=(a,b)满足 v=a 或 u=b 才会减少 q（q=r 已 selected 另处理）。非负 partial losses 使其他 old counts 保留 upper bound，研究 exact-ready heap top 是否可只等影响它的 producer 闭合而不等全部 prior ACK。
- 边界：hidden-birth upper priority、physical support/interval hazards 继续单独要求；本轮为局部 barrier 条件的集中审查，不先宣称完整 pipeline 已无 global fence 或 native 安全。

- 10:26 阶段模型：23,488 cases、32,150 数值 admissions，其中 6,668 有其他 old ACK 未闭合、2,235 有未完成 prepare；3,324,229 upper 及 1,762,214 role-exact checks。只 weighted-token numeric 模型对 serial greedy oracle，未含 physical events/HB/Rust；21.908s 为工具 duration，原 absolute 运行起止未保存。
- 独立 role 枚举 797,040 checks/131,787 被毁 pair checks；缺 right/left loss ACK 可错收旧 count5 而实际 2、competitor4；numeric count 已闭合仍可有待写零 support。脚本`check_selective.py` SHA `43aedf4c27cf…`、role 脚本`5b0bb68bf821…`。完整说明于 10:27 形成，10:31 读回核对；该角色条件此前 interval 说明已提出，本轮补充证明、闭合协议和独立数值事件验证，不计为全新角色定理。

- 10:34 完成说明核验：可选 async 方案对当前旧 heap head 只等待所有满足 `r.tail=q.head` 或 `r.head=q.tail` 的先前 producer 之 loss-closure；O(B)扫描即可，无需新增 O(V)数组。若 guard 仍为 raw selected frequency，则未完成 producer 继续挡住后续候选，选择性 ACK 本身不能越过 prepare；提前 prepare 重叠仍需渐进 guard credit。保持 hidden-birth、物理 P/W、AA 未来任务注册、descriptor 及 scope-join 要求。

### 196｜joined 模型取消 K0 状态的独立双控制验证

- UTC 对照：开始：10-12 02:23；结束：10-12 02:26:26.644642；10-12 02:26:30.321290。

- 开始：2026-10-12 10:23 后源码复核；精确开始秒未记录。
- 结束：已保存模型 10:26:26.644642—10:26:30.321290 通过，说明文档补充待完成。
- 工作：FRONTIER 与 exact-M 都不运行 witness probes，比较 normal deletion 与 zero tombstones，保留 source initial length 例外、后期 length-ineligible missing keys 及完整 final 零坐标。
- 结果：50,200 windows/final checks，25,100 delete/tombstone 等价、74,684 greedy、67,828 old-map、47,604 exact witness-gate-equivalence checks，14,426 live-zero coordinates。joined 各阶段 join 后允许旧 map 结构更新；old-ID domain/coherent group/unique fresh ownership 仍必需，async reader 设计不在此精简范围。
- 证据：`check_unfrozen_counts.py` SHA `7402b309ce80…`及`unfrozen-results.json`。仅独立 weighted-list 数学执行；

- 10:28 独立结论补充：无 K0 的 exact-M 在 floor 以上 guard 及 whole-producer 拒绝判定与保留 witness gate 一致；多出的低频 families 只增加 scalar 工作。该等价不声称低频 raw guard 值字节完全一致，也不推广到带运行时 gate 的 async 结构。

- 10:51 精化 above-floor 等价范围：exact-M 只对唯一 owned 且满足 source birth 长度准入的 coherent families 求最大。将 length-ineligible 物理 family 也算入仍给安全 upper bound，但会变松，不能再声称 no-K0 的 above-floor guard 完全不变。complete publisher 接收完整 Candidate，可含 Leaf 或 Root，不应误称 root-only insertion。

### 197｜利用 word 边界收紧 read interval

- UTC 对照：开始（截至记录，非精确起点）：10-12 02:25；结束：10-12 02:27:48；10-12 02:32:01.210191；10-12 02:34。

- 开始：截至 2026-10-12 10:25 已展开；精确开始秒未记录。
- 结束：报告的首个模型完成时间为 10:27:48；最终保存结果的时间折算为北京时间 10:32:01.210191，未记录精确进程开始时间，两个记录不合并成虚构运行区间；10:34 读取最终说明及结果。
- 工作：审查既有 Corpus.word/start/end 与历史合法 Positions 能否把 read envelope 裁在真实 word 边界内，不新增全 corpus 扫描；核对几何、restart seed、零权重、读取前后两个邻居、错误边界与 overlap query。
- 成功范围：2,000 cases、11,418 snapshots、534,424 candidate probes（218,358 stale）、2,070,407 loads、917,808 stores、315,563 job covers、1,420,686 partition queries，reference accesses 均在裁剪区间。真正同 pair 的 tail/restart upper cover 本已使 W 落在 word 内，因此 W 不缩小；收益仅可能来自 R halo 缩小。
- 失败前提与代价：普通 fresh materialize 把 starts 换成空 Box；需延长原有每 word 一个 usize 目录的生命周期，计 O(words)memory 与每 job 两次 O(log words)查询，而非免费已有 fresh 索引。无需重新构造目录或扫描 Positions。跨很多 word 的 job 仍保留巨大空洞；下一 restart seed 可以安全但宽松。
- 反例：人为把 stale token tail 作为任意 resident posting 可让 matcher 跨 word 读取，故 resident bounds 不足；证明依赖 source 生成的真实历史 cohort。此反例不属于合法 source 产生的 posting，不是生产缺陷。零/末 separator 和 equal restart seed 均单独检查。
- 证据：`check_words.py` SHA `14f441ebb857…`。仅顺序 reference 访问/区间模型，未模拟完整并发 scheduler；AA 仍需原 aggregate P/W 或 joined 边界。

### 198｜最终频率下界假设的反例审查

- UTC 对照：开始：10-12 02:30；结束：10-12 02:34；10-12 02:38。

- 开始：2026-10-12 10:30 前后新展开；精确开始秒未记录。
- 结束：10:34 阶段说明，10:38 核对完成报告与保存结果；精确模型运行起止未记录。
- 工作：对动态 floor、剩余词表 horizon、reserved/reuse、length admission、错误回退假设集中审查。只从充分 drain 的完整 fresh frontier 取证，H=(target−V)+未曾激活的现有 IDs；每次首次激活使 H 降一，不能以当前正权重 support 或 256 字母表代替 activation history。
- 结果：3,000 weighted corpora、524,352 certificate checks、80,182 成立证书通过，涵盖 raw AA、零权重、cap0 及 initial admission 例外、低 numeric reserved 首次激活。修正后充分条件未找到反例；此 PASS 不是 native 实现验收。
- 重要反例：六个 xyab…xyag 词各权重 2，加 uv:1，F1、strict cap4、target14。XY12 后隐藏 XA12，旧可见六个 pair 各 2 会错误证明终频≥2；真实后续 XA12 的孩子因 cap 被拒，UV1 才填满 target。只看精确“可见旧 counts”仍不是完整 frontier，不能在未 drain 窗口内更新该 floor。该定向反例按推导记录；保存 JSON 未含后来脚本的 directed trace 字段，不将其冒称同一次已保存模型输出。
- 生命周期结论：已证明的 floor 在同一 attempt/target 内取历史最大，后来弱证书不要求降低或重建；reuse 重启清空证书。真正降低门槛必须完整重建适当 eligibility 的 counts/候选，不能只改单个 scalar。错误的证书中间算术 overflow 应使 heuristic 无法认证，不能拒绝本来合法训练；实际启动的原训练错误仍传播。
- 证据：当前`final-floor-assumptions-check.py`指纹`f71cf5500d93…`；此为读回当前源指纹，未额外主张精确运行时绑定。充分 drain/完整 key/whole-family upper bound、保留 corpus 写入与零权重 coverage 仍必要。无 K0 的 joined 版本可正常删旧 counts。

- 10:39:38.969274—10:39:38.969512 新增定向模型实际执行并独立读回：完整轨迹频次[12,12,1]填至 vocab14；错误 floor2 仅[12,12]停在 13；不完整快照 D6>5 产生错误认证。只运行这一个新增反例，没有覆盖或重跑上述 3,000 套。此前“仅推导/未在保存 JSON 执行”保留为当时核验状态，本条补充后续实际检查。

### 199｜当前主方案的维护成本与最小接口审查

- UTC 对照：开始：10-12 02:29；结束：10-12 02:37；10-12 02:38。

- 开始：2026-10-12 10:29。
- 结束：10:37 审查完成；10:38 读回核对。
- 工作：独立检查 one-cohort complete publication、无 witness 的 joined FRONTIER 及可选 exact-M，比较真实删去的工作与移动到 producer 的工作、新增 join、顺序尾部、内存和接口成本。
- 阶段结论：分别评估 commit complete publisher 与 prepare deferred births；FRONTIER 不需要把 exact-M 设为基础设施。无 K0 的 joined 方案删去的是此前提议的 snapshot/tombstone/cleanup 机制，不能冒称删掉现有生产结构。保留正常 counts 删除、原 compatible cohorts 及每轮 joins；不据此增加 Engine、async 调度器或公共 Builder 类型。
- 主要限制：complete publisher 仍需 producer grouping/sum/floor/concat，giant producer 可能失去 owner 并行；Roots 不得重新走只接收 Leaves 的 concat。FRONTIER 最多四个 compatible cohorts/window，start retention 与重复最终邻居读取仍需记账。维护审查已完成，没有实现、LOC 或速度测量。

- 10:35 新增接入反例：当前 Neighbors::finish 仅在 removed_weight 非零或 Positions 非空时保留记录。count-only exact-M 若原样复用，会丢掉 born_weight 为正的纯 birth scalar 记录。必须调整私有 finish 契约或使用语义清晰的 count-only 完成路径；这是拟议接入陷阱，当前生产仍有 Positions，不是现有 bug。

- 定向 fixture：ABCD:1、AB:3、CD:2、F1，AB4/CD3 同 cohort 产 X/Y；X 的 C-neighbor 记录 loss1，另一个 Y-neighbor 纯 birth 记录 XY1。停止 Positions::push 却复用旧 finish 会丢 XY1，故 count-only 必须显式保留 scalar-family liveness。
- 维护账本：joined 五项共同契约（old-input/身份边界、跨 cohort cut ownership、starts/descriptor 生命期、完整 drain、资源触发 drain）加一项 guard；exact-M 另有 whole-producer 安全释放条件。独立 complete publisher 四项契约，不因组合就自动简化。
- 生命周期修正：selected Positions 语义上在该 producer prepare join 后可释放，但当前 FreshSnapshot 整体借用 Batch，需 ownership 重构才能早释放；保留空 Job/Vec::retain/mem::take 元素都不自动释放 Change backing capacity。实验应独立比较 eager、每 cohort 强制 drain 的同一 deferred kernel、guarded 多 cohort；publisher/scratch/terminal 政策必须固定以免错归收益。

### 200｜最新提交与已审核心源码的逐 blob 核对

- UTC 对照：开始：10-12 02:32:09；结束：10-12 02:32:09；10-12 02:34。

- 开始：2026-10-12 10:32:09。
- 结束：10:32:09 核验事件；10:34 读回保存证据。
- 工作：取得 890ea1f5 完整 recursive Git tree，确认 truncated=false，逐一比较已审 a2b20f32 的 corpus/index/merge/mod/positions Git blob。
- 结果：这五个核心文件完全一致，可将正式文档链接定位最新 890ea1f5；vocabulary 不同，继续依据 a583a0 的独立审阅。没有把文件名相同当作内容相同，也未执行仓库。
- 证据：[最新固定提交](https://github.com/Yikai-Liao/tokenizers/tree/890ea1f58e65f8d8503f7cd3fc97ad1651eaf284)，完整 tree 记录已保存；此阶段是来源核对，不是新的性能实验。
- 10:34:40.728 再次核验 branch，仍为 890ea1f5，没有新提交。

### 201｜交付材料整理与截至验收前的持续核验

- UTC 对照：开始（截至记录，非精确起点）：10-12 02:41；结束：10-12 03:00。

- 开始：截至 2026-10-12 10:41 已开始主计划与正式附录写作；精确开始秒未单独记录。
- 结束：2026-10-12 11:00，达到本次约定验收时点。
- 工作：把已完成证明、来源审计、数学模型、否定方向与更正整理成可阅读附件，同时继续检查新提交、重要前提及接入反例。完整阶段日志保留真实时间缺口，不把作者实验写成本次亲自执行。
- 状态：文档整理不是生产实现或 benchmark；最终记录在下方列出，未先写尚未完成的检查为通过。

- 10:45:07 开始生成独立用户阅读副本；保留 201 个阶段及全部顺序更正，增加 UTC 对照，移除内部目录与研究报告路径，压缩长模型指纹。首次读回检查确认 201 个阶段均在、81 个公开来源链接保留；未发送或提前封稿，之后继续更新。

- 10:45—10:48 正式正文逐段一致性核验：复核 FRONTIER 四 cohort 上限、无 K0 exact-M 等价、complete publisher 的 same-owner coalescing 费用、retained starts 界；通过 Git tree 确认 prefetch README 与 all-normal 公开证据路径。补回先前已证明的渐进 U=f−Σ(m−c)及 AB3000 四 worker 真实 task-cut 正例，没有新发现或重跑模型。主计划、证明与本日志继续修订，不提前封稿。

- 10:48—10:51 正文 QA 补明累计 window cap、length-eligible exact-M 与完整 Candidate 直插三项边界，未新跑模型。10:49:37.495 再次核验 head 仍 890ea1f5；三条论文公开链接复核可访问且作者/标题对应。仍继续到验收前的核验。

- 10:52—10:54 继续读取当前 index.best/take、merge_next 及 Positions Leaf/cache/seed 公式：fresh take 实际删除 counts，原代码不采用 peek_mut；原 cap 按 cohort，新 window 控制须显式累计。n500/P4 中 560 used encoded bytes=500+4×15，仅编码数据，非 capacity/总内存。正式示例补 500 个 ABc_i、500 个 DBc_i 与 c0A:2、target≥507，保持不靠 terminal 取得收益；核对实际 born_weight 字段。正式文档中的数学前提修订已读回，没有新跑仓库或模型。

- 10:55:44 执行补充的累计 window-cap 反例，10:56 读回保存结果；这是新增独立小模型，与 10:48—10:54 的纯源码/正文核验区分。

- 10:59:35（工具完成时间约至秒）完成最终 branch GET，head 仍为 890ea1f58e65f8d8503f7cd3fc97ad1651eaf284，没有新提交；未补造更细时间。最终主计划核对 heap reclamation 修正后为 329 行、37,324 bytes，SHA256 `7ca68138819c…`。
- 11:00 达到约定验收时点并同步本完整日志。所有阶段保留探索、失败、更正、证据范围与时间缺口；没有把本次理论/独立数学研究记作生产实现或性能验证。

## 本日志核验记录

- 2026-10-12 02:32:30：读取两份模型摘要和对应机器可读结果，核对 PASS 状态、计数、耗时、适用范围与负面对照。
- 2026-10-12 02:32:44：对两份模型源文件重新计算 SHA-256，与已保存结果一致。此次仅核验文件及历史结果，
- 2026-10-12 02:44:13：读取 token-mass 机器结果与报告，并重新计算源文件 SHA-256，一致。AA 规则数按保存结果核对为 23,566；此核验

- 2026-10-12 02:50:17：读取 sealed-row 模型结果，并计算源文件 SHA-256；

- 2026-10-12 02:52:41：读取 frozen-witness 模型报告与机器结果，重新计算源 SHA-256 并核对一致；

- 2026-10-12 02:57:10：核对 AA supplement 的机器结果、报告和两个源文件哈希；有效 AA batch 数为 7,278，不能把全部 45,432 配置称为已执行 AA batches。

- 2026-10-12 03:10:05：读取 ephemeral-slot 保存结果并计算源 SHA-256；

- 2026-10-12 03:10:36：核对 parent-budget 机器结果、报告、开发断言修正说明及源哈希；

- 2026-10-12 03:20:51：核对 bpe_reachable 机器结果、可实现 fixtures、限制及源哈希；明确区分提前预留与执行，

- 2026-10-12 03:21:59：读取最新源码审计及 slot-simplification 数学结果，核对 source hash；

- 2026-10-12 03:24:01：核对 bucket supplement 结果、源码与依赖哈希；区分重复分区配置和不同语料，

- 2026-10-12 03:25:51：核对 DAG 局部公式检查的事后归档、有限范围和源哈希；保留原 inline 执行缺少精确起止/运行时哈希的限制，未重新执行。

- 2026-10-12 03:29:13：核对 DAG 多轮原始输出、执行起止与源哈希；明确未检查 greedy 选择或势能复杂度，没有重新运行。

- 2026-10-12 03:30:38：核对 MG 最终结果及 DAG 独立审查的结果文件、有限范围和当前源哈希；没有重新执行或把抽象复制界称为 trainer 复杂度。

- 2026-10-12 03:36:45：读取 slot work/span 与 AA DAG 独立结果，核对源哈希和统计范围；

- 2026-10-12 03:41:00：核对 dynamic-floor 两份 smoke 脚本、原输出归档及当前源哈希；不把时间上界当精确结束，也不把 reference 条件检查称为 pruned lifecycle 通过。

- 2026-10-12 03:46:07：核对 marked-state β 支持性检查的原始起止、结果和运行时哈希；保留未独立证明的状态，没有重跑。

- 2026-10-12 03:48:20：核对动态 floor 最终报告、机器结果和源哈希，记录 16,988 次完整生命周期比较；

- 2026-10-12 03:53:10：核对 whole-word 证书完整报告、机器结果及源哈希；记录未实现的 B 细化和 reuse 重建边界，没有重新运行。

- 2026-10-12 03:58:23：核对 grouped atomic 报告、数学状态计数、反例及源哈希；未执行仓库或重跑模型。

- 2026-10-12 03:59:23：核对静态 word-summary 模型的结果、quantile 和 batch 审计、反例及源哈希；明确 752 次后期提高中 585 次来自初始 Q0<H0，没有重跑。

- 2026-10-12 04:00:11：核对 DAG 数值域审查与算术原输出；读取 batched-join 报告并核对两份脚本哈希，原执行记录待补。没有重跑。

- 2026-10-12 04:07:52：核对 frozen gate source-shaped 模型的精确运行时间、结果和源哈希；未重跑。

- 2026-10-12 04:08:22：核对 ancestry 物理边界模型结果、完整历史统计、strict-cap 负例与源哈希；未重新运行。

- 2026-10-12 04:12:46：读取 speculative-floor 与 rewrite-frontier 阶段报告、支持模型结果并核对脚本哈希；batched-join 归档仅有脚本，原始执行时间/stdout 标为未保存。没有重跑。

- 2026-10-12 04:21:40：核对 boundary-conservation 模型的精确运行时间、结果、双向工作量 fixture 与源哈希；没有重跑。

- 2026-10-12 04:22:12：核对 speculative 挑战模型、接受 trial 中平凡/非平凡拆分、early-abort oracle 与源哈希；没有重新执行。

- 2026-10-12 04:24:24：核对 barrier fragment 两份支持模型结果和当前脚本哈希；明确模型运行时间缺失及 late-census 不能回溯省 initial append。没有重跑。

- 2026-10-12 04:25:35：核对 boundary-conservation 最终报告、restart-aligned fixture 复跑及新源哈希；保留初轮记录并明确由 21,117 事件最终结果取代。没有自行重跑。

- 2026-10-12 04:28:20：核对 barrier/grammar 最终报告、snapshot 与完整历史范围、初始 admission 例外和源哈希；未重跑。

- 2026-10-12 04:34:15：核对动态 D2 模型结果/哈希及 hash 强度合成材料，保留固定 H 误读的更正与 live-reindex 待验证状态；没有重跑。

- 2026-10-12 04:38:28：核对 atomic/global-birth 组合模型结果、精确时间与源哈希；记录“省 scalar 阶段不等于省总 phase”的结构 mutation 限制。未重跑。

- 2026-10-12 04:40:10：核对 live reindex 最终模型、独立 slot-plane 扫描与负例、空成功拆分及源哈希；未重跑。

- 2026-10-12 04:50:02：核对更强 root/weighted grammar 报告、两份生成模型统计及当前源哈希，明确无精确进程起止和 uncapped 范围；未重跑。

- 2026-10-12 04:52:42：核对 practical census 源码 manifest 范围、模型结果/负例及源哈希；区分无 resolver 加遍与仍需 slot 读取，未重跑。

- 2026-10-12 04:53:33：核对 birth grouping 有限模型、task 界、同 key 顺序反例及报告/脚本哈希；布局账本仅估计，未重跑。

- 2026-10-12 04:59:37：核对 weighted-root 扩展模型、cap/mid-run 范围与源哈希，记录正确 typed clear 和任意额外 clear 的区别；未重跑。

- 2026-10-12 05:03:05：核对 geometric live-reindex 模型的重建次数、物理体积/初始计入方式、时间与源哈希；未重跑。

- 2026-10-12 05:06:26：核对 actual-interval source 审计、保守 cover 模型时间/结果和两份哈希；明确继承定理与本轮增量，没有重跑。

- 2026-10-12 05:12:30：核对 fragment 变换源码审查、tiny fixture 时间/结果与报告/脚本哈希，避免把 activation 陷阱说成最终模型反例；未重跑。

- 2026-10-12 05:12:55：核对 input-fragment reduction 最终结果、初步摘要修订、六 delta 检查与源哈希；未重新运行。

- 2026-10-12 05:15:22：核对 safe-cut refinement lifecycle 的 slot guard、统计和源哈希；没有重跑。

- 2026-10-12 05:21:07：核对 mid-run sparse plane 模型、825 配置与 823 完整参考/2 个 reuse dispatch 区别、时间和源哈希；未重跑。

- 2026-10-12 05:24:08：核对 FFBPE source 审查、hot-window 操作 fixture 与两份哈希；明确 word-ID sets、zero-weight 差异及未迁移 benchmark。未重跑。

- 2026-10-12 05:31:59：核对 in-place 最终扩展、original/rebased oracle 拆分与新源哈希；初轮 736 配置保留为历史，最终 738 配置取代，未自行重跑。

- 2026-10-12 05:37:46：核对 representative 恢复最终模型、retained 与 relocated map 差异、时间及源哈希；未重跑。

- 2026-10-12 06:08:25：读取 prepare/apply 融合完整报告与正目录缓存报告，核对四份模型源哈希及正缓存机器结果；保留 tagged-plane 额外编码、负判定重复查询和有限模型范围。没有重跑或执行仓库。

- 2026-10-12 06:13:02：读取正目录 gate 独立复核及机器结果，核对源哈希、非空/空 batch 拆分、初始 cap 与 stale 目录负例，没有重跑。

- 2026-10-12 06:19:44：读取 fresh prepare 摊销证明、随机/紧性模型结果和三份哈希，明确物理 R、成功前缀及非坐标额外成本；未重跑。

- 2026-10-12 06:32:12：核对 atomic-prepare overlap 独立报告、源/依赖哈希、CAS 重试与 birth 顺序负例、机器计数及精确时间，没有重跑。

- 2026-10-12 06:49:24：核对 producer grouping 最终报告、机器数据、初步摘要更正及源/依赖哈希；没有重跑。

- 2026-10-12 06:57:10：核对组合审查报告、tiny 契约模型精确时间/结果及三份哈希；保留“错误初始 membership”与“同 batch 正确冻结 membership”的区别，没有重跑。

- 2026-10-12 07:00:08：核对 incidence-death 独立结果、点态界及非单调 gap 反例、模型时间和源/依赖哈希，没有重跑。

- 2026-10-12 07:26:03：核对 two-view/own-input/quotient 三份独立模型、执行起止、统计/源哈希与 floor-raise 清理限制，未重跑。

- 2026-10-12 07:46:43：核对 producer pending bound 五项模型的结果、精确时间与 hash；明确 raw-child scope 修正和 priority/readiness 分离，未重跑。

- 2026-10-12 07:50:06：核对 zero-row 强 fixture 最终独立 encoder 源码 addendum、模型/报告哈希，记录由 raw trace 升级为 source-proved normalized 差异及未 native 执行界限；未重跑。

- 2026-10-12 08:13:20：补核 admitted 初始界、组合 loss、candidate readiness、fused schedule 和适用性已保存报告/结果/脚本哈希，并核对 floor-composition 新模型；原时间未保存处明确标注，没有重跑。

- 2026-10-12 08:23:36：核对 terminal elision 源码报告、归档复跑的精确时间/结果与 hash，明确 model 输出和 discard-only corpus、语义及资源失败的区分；本次未再运行。

- 2026-10-12 08:44:00：核对 fixed-coordinate 集成模型最终机器结果/精确时间/哈希、readiness 对称 fixture 与反例、fused memory 抽象结果。没有重跑或执行仓库。

- 2026-10-12 08:51：核对 output replay 最终结果与脚本/报告 hash、精确时间、AA alignment 及 stale-retention 反例，并读取稀疏 mutable frontier API 审查。没有重跑。

- 2026-10-12 08:55：读取 frozen-window 主证明及模型输出，核对独立 family JSON 精确运行时间/计数。异步事件模型另列未完成，不提前合并成 full pipeline 验证。

- 2026-10-12 09:03：核对独立 window 最终报告、二次归档运行时间/源 hash 变化及 second-hop 补充。新 head 更新单独记录，未回写成旧模型已验证新源码。

- 2026-10-12 09:08：读取 InitialTokenIds 新 head 审查及作者两轮数据范围，核对 joined-window break-even 报告与 scalar 模型原输出；不把 62263db 实验归到 a583 最终 loop。

- 2026-10-12 09:13：核对 event/completion/driver 完整材料与源 hash、joined 独立模型精确时间/计数、lazy overlay 反例、完整 family 过滤 fixture 及 SmallVec inline 源码；没有重跑。

- 2026-10-12 09:25：核对新 Affix evidence 的最终 source 身份、raw pairs 均值口径、test 过滤项和报告 hash；旧未测判断保留并指向新限定性证据，没有执行作者测试/benchmark。

- 2026-10-12 09:40：核对 interval 事件 JSON、两套 lexicographic 模型与独立边界负例的原始时间/hash，并读取 partitioned-MAX 报告；保留 adapted model 与 independent 验证的区别，未重跑。

- 2026-10-12 10:00：核对 prefetch 最终报告/当前 phase 分母，独立 interval 及 aggregate-AA 结果，作者 model census 与历史有序 batch 数据、script hashes。没有重跑模型或 author 程序。

- 2026-10-12 10:11：补读 exact direct reducer/独立 physical-cut、joined source-contract 及已保存 memory/cost 稿，核对模型时间/hash；没有重跑，未标明起止的稿件保留该缺口。

- 2026-10-12 10:27：核对 FRONTIER 实际操作与 delete 变体、direct publisher、semantic alias/horizon 及新 unfrozen-count 结果；selective loss 先记数值模型，physical 同步与未完成说明继续分开。未重跑。

## 验收时的证据状态

- 已完成：上列 201 阶段的理论推导、来源审阅、定向反例和有界独立数学检查；已明确前提的结论与被否定方向分别记录。最终主计划与证明附录已汇总当前可用结论，早期“进行中”属于当时时点，不表示这些证明全部仍未完成。
- 已核验：最新固定提交及作者发布的指定测试、模型 hash 和实验数据。作者结果保留其实际配置与范围；未移植为本次亲自运行的结果。
- 仍需生产验收：实际实现后的完整有序模型/error/reuse/AA/零权重/长度/identity/target 等价，Rust 所有权与并发接口，新增工作和消除工作的实际计量，可重复端到端性能及时间对齐的内存峰值。
- 没有宣称：数学模型通过等于 native 实现通过，或理论可省某类工作就意味着实际加速。只有实际完成实现、测试和 benchmark 后，才能关闭这些生产证据缺口。
