# BPE Trainer 简化：消融、收益评测与删除决策执行计划

日期：2026-10-09。目标：显著降低生产代码和维护复杂度，尽量保留训练速度、内存效率及行为兼容性。

## 1. 结论与证据边界

**静态分析只决定实验顺序，不决定删除名单。** 本计划没有执行仓库代码或性能测试。所有预计删除量均来自静态结构估计，不是已实现结果；收益必须通过下述消融得到。

当前分析基线为 [e4f787dc189d9be7192107490d652096cde7480e](https://github.com/Yikai-Liao/tokenizers/tree/e4f787dc189d9be7192107490d652096cde7480e)，也是本次核对的 PR head。历史基准使用 8faaff79；两者 BPE engine Rust 源码未变，但仓库其他部分有变，不能直接混用端到端数字。所有新候选应从同一个固定 SHA 构建并重新测基线。

用户补充：连续 birth 存储在 Byte-level BPE 下曾有约 10%～十几%的收益。当前尚未独立核实其测量口径和原始记录。**据此调整排序：保留连续存储，不再列为首批删除对象；先找历史证据，再单独评估它的自适应策略。** 连续存储、完整 producer 直接编码、自适应选择是三个不同机制，不能绑在一起消融。

最终每项只能得出：**保留、删除、证据不足**。没有统计上明确的差异，不等于已经证明代价可以接受。

## 2. 先设定删除目标与保护范围

保留以下核心能力：兼容规则批处理、局部更新与 occurrence 索引、延迟语料 materialization、压缩位置、宽坐标支持、精确 tie-break、AA 左到右非重叠选择、reuse/重启及 signed ledger 语义。

减少复杂度应体现为至少一种实际变化：

- 一整套实现、状态或表示消失，剩余调用者只面对一种协议。
- 跨模块参数、反馈状态、const 泛型变体或转换路径减少。
- unsafe 操作及其证明义务减少。
- 新特性需要覆盖的交叉组合减少。

不把搬文件、压行、删注释、删测试、泛化成更难理解的框架算作简化收益。不能预先承诺减半；应先看可独立删除的模块能贡献多少净收益。

## 3. 模块实验顺序

这里的优先级表示“先收集证据”，不是“最可能没用”。规模不相加：有些删除相互重叠，必须在实际补丁上计算。

| 顺序 | 候选 | 单项替代方案 | 可验证的简化价值 | 主要风险与判定重点 |
|---|---|---|---|---|
| P0 | 连续 birth 历史证据核对 | 找到原始 A/B、SHA、开关、ByteLevel 语料和计时范围 | 避免误删已有大收益路径、避免重复实验 | 分清整体训练与局部阶段的 10%；是否同时开启直接编码 |
| P1-A | bounded initial collector | 所有输入走现有 keyed collector，保留 compact/wide record 两种布局 | bounded 模块生产部分约 292 行，另有准入和分派代码 | 原来获准输入 raw record 从 4 到 8 B/edge，并增加排序；RSS 与初始化可能明显退化 |
| P1-B | 自定义 radix sorter | 对原有紧凑 record 用标准不稳定排序，比较完整 pair key，再比较 offset | 核心 sorter 约 324 行，减少 scratch/并行排序协议 | 比较排序可能显著慢；不能只按 key 不稳定排序，也不能把 record 换成更大的 padded tuple |
| P1-C | Packed U24 slots | 保留 U16/U32，原 U24 档改 U32 | U24 实现约 65 行，加分派及相关 unsafe 证明 | 此档 slot plane 3→4 B/slot，增加 1/3；不是全进程 RSS 增加 1/3 |
| P1-D | birth 自适应策略 | 对仍满足安全/资源门槛的完整 producer 固定用连续表示，保留其他 linked fallback | 预计约 75～100 行，删除策略反馈、BirthShape 统计及协调器穿透状态 | 不等于所有任务无条件用 vector；可能在短链或小批次退化；先核对既有实验 |
| P2-A | AA 并行选择协议 | 已排序有效位置上串行贪心选择，后续写回保持既有方式 | 预计约 130～180 行及跨块 parity 协议 | 只删并行选择，不能删 AA 语义；长重复串可能暴露退化 |
| P2-B | SelectedRuleIndex 特化 | 完整 pair key → replacement 的简单表 | 预计约 40～65 行及 heads/tails 特殊状态 | 热路径哈希访问增加；保留共享端点和冲突识别语义 |
| P2-C | 4-entry priority prefix cache | 直接走现有 heap exact 校正 | 预计约 30～50 行 | 英文选择成本不能忽略；过小删除量未必值得回归 |
| P3-A | 位置编码 API 收敛 | 考察统一到 measure + replay 的直接编码路径 | 可能删除 encoding scratch、锁和跨模块参数 | 需证明所有 iterator 可重放，可能增加遍历；先做接口设计再决定是否实现 |
| P3-B | reuse 的 word/position 双扫描收敛 | 先证明 cohort 边界，再考虑统一扫描 | 预计约 65～85 行 | whole-word 扫描可能发现 cohort 范围外匹配，不是可以直接启用的等价 fallback |
| 暂缓 | 完整 producer 直接编码 | 强制 buffered → owner commit | 可删部分 completed birth 路由 | 把工作重新塞回 commit，可能损失重要收益；不能与自适应删除混测 |
| 暂缓 | prefetch 小环、router 倒数乘法、事件 scatter | 普通迭代、取模、稳定排序 | 各约十几到几十行 | 代码收益小、可能增加热路径成本，不作为大幅简化主线 |

特别注意：radix 与 bounded collector 互相影响。只在 bounded 快路径开启时测 radix，可能测不到真正的排序负担。

## 4. 第一阶段：整理证据与建立实验基线

### 4.1 建立历史证据清单

对每项记录：历史分支/SHA、是否已进入当前基线、比较对象、语料/预分词、词表规模、worker、CPU、wall/CPU/RSS、重复次数、模型校验、同时改变的其他机制。

连续 birth 优先检索。若历史试验同时改变直接编码和连续存储，不能把全部收益归因给其中一个。如果原始证据充分，只需在当前稳定机器做针对性确认，无须重跑完整探索。

此前 owner 替代、阶段重叠、initial pipeline 等失败实验不应重新实现。未合入 main 的候选也不能算当前可以删除的代码。VPS 波动下的结果可提供线索，不能作为微小性能差异的定论。

### 4.2 固定构建和机器

- 使用用户定频 3 GHz 的 8 核笔记本，核实实际拓扑、SMT、亲和性、频率、温度、功耗限制及是否降频；不把设置值当作实测值。
- 同一 Rust 工具链、Cargo.lock、release profile、features、target-cpu、allocator。保存完整构建命令与二进制哈希。
- benchmark 期间停止编译和无关重任务；所有二进制提前构建。
- 固定语料文件与 SHA-256、顺序、预分词、trainer 配置。每个测量在独立进程运行。
- 基线与候选都从固定 SHA 构建。代码、结果、脚本在独立实验分支/目录，是否推送由用户另行授权。

### 4.3 先跑 A/A

用同一个基线二进制当作 A 与 A'，按正式测量协议交错运行至少 6 对，得到环境与驱动的噪声。

若 A/A 的波动已经足以掩盖设定的回归预算，应先排查环境，或扩大样本并承认不确定性。不能看到候选“没明显变慢”就接受删除。

## 5. 第二阶段：每项消融如何实现

每个候选分两步，不建立需要长期维护的公共消融框架。

1. **隔离版**：用最小私有改动强制现有 fallback 或固定策略，其他行为不变。记录走过的路径，确认实验真正关闭了目标机制。
2. **删除版**：隔离版通过初筛后，实际删除不再需要的类型、字段、分派、统计、const 变体和调用参数。再测性能与正确性。最终评价以删除版为准。

现有 MergeOptions 的部分选项仅在测试配置可用，不应假设已有公共 CLI 开关。用独立候选二进制或临时私有编译配置实现；实验开关不得留在提交版。

### 5.1 初始构建：A × B 的必要交叉消融

| 变体 | bounded collector | radix sorter | 用途 |
|---|---|---|---|
| I0 | 开 | 开 | 基线 |
| I1 | 关 | 开 | collector 独立代价 |
| I2 | 开 | 关 | sorter 在原分派下的代价 |
| I3 | 关 | 关 | 两项都删后的真实代价与可删除范围 |

只测 I1/I2 不足以决定能否同时删除。四种变体必须保留相同 record 布局与完整 key/offset 顺序。

### 5.2 Birth 的有依赖消融

| 变体 | 连续存储 | 自适应反馈 | 完整 producer 直接编码 |
|---|---|---|---|
| B0 | 当前策略决定 | 开 | 开 |
| B1 | 所有满足现有门槛的 producer 使用 | 关 | 开 |
| B2 | 关，使用 linked | 关 | 开 |
| B3 | 关，统一 buffered commit | 关 | 关 |

- 本轮先比较 B0/B1，回答“能否只删策略”。
- B0/B2 只在历史连续存储证据缺失或口径不清时做针对性复核，不能默认进入删除流程。
- B2/B3 才隔离直接编码的收益；目前暂缓，不与 B0/B1 捆绑。
- B1 必须保留 u32 准入、完整任务、node budget 等门槛，以及 AA/reuse/partial/wide fallback；“固定策略”不意味着突破这些约束。两位置组仍保持 linked，第三个位置才提升为 vector；固定开启可能让许多三位置小组增加分配，因此不保证优于自适应。

源码已有 linked-only / always-contiguous / adaptive 的 1/4/16 worker 对照测试，并比较完整结果和 merge trace，可复用其配置与 fixture：[producer_fast_path.rs](https://github.com/Yikai-Liao/tokenizers/blob/e4f787dc189d9be7192107490d652096cde7480e/tokenizers/tk-train/src/trainers/bpe/engine/tests/producer_fast_path.rs#L349-L365)。这证明已有验证入口，不代表本次已执行，也不是性能证据。

### 5.3 U24

强制原 U24 档走 U32，其余不变。通过路径计数确认数据确实落入该档。先测真实常用词表，再用 ID 边界 fixture 验证；不能用始终走 U16 的任务证明 U24 无价值。

### 5.4 第二波小机制

各自从原始基线单独消融。AA 只替换选择算法；priority cache 只换取顶路径；SelectedRuleIndex 只换查表表示。不要同时改调度、分片数量和位置编码。

## 6. 正确性验证：删实现，保留行为覆盖

已有 HF parity、多 worker、错误/unwind 等测试，不能描述成没有测试。按候选补齐针对性覆盖；机制专属路径断言可调整，语义 fixture 应保留。建议恢复适量固定种子的生成式组合对照，避免只靠手写案例。

### 6.1 每个隔离版与删除版都做

- 编译、格式、Clippy；默认及 no-default-features 的 library tests。
- 同一输入比较 baseline 与候选完整 vocab→ID 和完整有序 merges，不能只比 vocab 大小、token 数或一次编码结果。
- 小 fixture 比较逐规则 `(pair, count, replacement ID)` 轨迹，尽早定位偏差。
- 固定输入遍历顺序与 alphabet 选择条件；不要以“让测试稳定”为理由偷偷加入新的 tie-break 语义。
- 1/4/8 worker，并保留现有 16、64/65 等边界测试用途。路由若改变，补 3/6 worker。

### 6.2 所有候选的基础矩阵

| 类别 | 必测内容 |
|---|---|
| 普通输入 | 空输入、零 merge、target 已达到、频率并列、重复词和 weighted counts |
| 文字与装饰 | ASCII/BMP/非 BMP/组合字符；prefix/suffix/两者；过滤和 alphabet 限制 |
| 重复与冲突 | AA 奇偶长度、长连续段、共享端点规则、批次边界；精确左到右选择 |
| fresh/reuse | ID reuse、restart、相同 ID 不同 occurrence span、retained alphabet、signed ledger |
| 长度 | max length 未设、0/1/2、边界值；初始 pair 与新生 pair 的既有规则差异 |
| 数值域 | 真 ID 0/65535/65536；u16/u24 边界与 sentinel；u32 邻界及宽位置；count 溢出和 signed 极值 |
| 生命周期 | 返回错误/panic 时所有任务 join 完毕；owner/arena 生命周期；append 失败保留前缀 |
| 公共接口 | feed/train/reload、进度、parallelism 设置及错误行为不变 |

不能新增“total weighted mass 必须 fit u64”这类原实现不存在的全局限制。受控单故障核对错误；多并发故障不强行要求不存在的固定胜出顺序。也不要求原本没有保证的事务回滚。

### 6.3 候选定向测试

- Collector：准入内/外、alphabet 255/256/257 及真实 ID 域；compact/wide records；通过小测试 wave 触发跨 wave 追加，不必分配巨大语料。
- Radix：完整 64-bit key、相同 pair 的位置顺序、单一 key、重复位置行为、跨 wave 顺序；无序输入不得伪装成已排序。
- U24：表示切换、最大合法 ID 与 separator 区别、尾部/guard 访问；涉及 unsafe 的补丁在支持的环境跑定向 Miri 或 sanitizer。
- Birth：tiny/promoted/complete/partial、floor prune、AA、reuse、wide fallback；证明关闭策略但没有关闭直接编码。
- AA：全 AA、被不同 token 打断、跨旧 chunk 边界、奇偶长度、merge trace 一致。
- Cohort：先证明扫描域等价；证明之前不进入性能实验，更不能因大语料输出偶然一致而放行。

## 7. 性能评测：把真实收益测出来

### 7.1 分层矩阵，避免一开始全排列

**快速筛选**：英文/中文 × Whitespace/ByteLevel，固定 256 MiB、50K vocab、4 worker；每个候选先跑 3 对，仅用于淘汰明显退化或发现问题，不用于批准小回归。再加一个小输入与一个候选定向压力输入。

**决策矩阵**：上述四类语料 × 1/4/8 worker；50K 为共同主点，另选 100K 检验词表和布局变化。默认至少 6 对正式交错样本；接近预算的格子继续补样，仍不能收窄就标记证据不足。不要无限补到出现想要的结论。

**最终组合**：固定四类语料、1/2/4/6/8 worker 核数曲线，以及代表性内存增长曲线。复用既有驱动和语料快照，所有候选用同口径重新测。

定向压力输入不用于宣传普遍加速，但可以否决会伤害合法输入的删除：

- bounded/radix：小 alphabet 和大 alphabet；高重复与高 distinct pair；初始化占比高的低 merge 数任务。
- U24：实际命中 U24 的 ID 域；内存增长测试到安全上限。
- Birth：ByteLevel 必须保留，长 birth 链与短链都测；完整 producer 比例高/低两类。
- AA：长重复串、普通自然文本；不能用某一次 AA 占比不到 1% 推断所有任务。
- 队列/lookup：英文短词、多批次、小 batch 与选择开销较高任务。

### 7.2 配对与记录

- 每种二进制/语料先预热，预热不纳入正式统计。
- 平衡 A→B 与 B→A 的顺序；多候选轮换或随机化区组顺序，保留 seed。每个区组有就近 baseline。
- 每次独立进程记录 feed+train wall、train-only wall、CPU time、进程 peak RSS、swap、温度/频率异常、退出状态、完整模型指纹。
- 诊断 build 的 phase/path counters 与发布性能 build 分开。trace 开销不能混进最终排名。
- 失败、swap 中止、离群值保留原始记录；排除规则在看候选结果前写定。出现环境异常应重跑整个配对，不能只挑较慢一侧重跑。

### 7.3 报告算法

对每个工作负载分别计算：`r = 候选耗时 / 同区组基线耗时`。报告配对比值的中位数、原始散点及按区组 bootstrap 的 95% 区间，并同时保留原始时间。样本少时区间不可靠，应明确标注。

报告每个格子，不用一个跨语料平均值遮住 ByteLevel 或单线程退化。RSS 报告绝对 MiB 差和相对比例；同样区分 slot plane 理论增量与整个进程峰值。阶段时间只解释原因，最终接受依据以用户真正使用的端到端指标和内存约束为准。

## 8. 如何权衡真实删除收益

### 8.1 每个删除版必须交付的复杂度账本

对固定基线运行 diff/统计，分开列：

1. 生产代码净减少行数（新增 fallback/helper 要扣除）。
2. 测试、文档、注释变化，单独列，不能冲抵生产代码。
3. 消失的实现/状态/表示/分派路径，逐项命名。
4. 减少的跨模块参数、const 变体、反馈通道、锁与 unsafe 块及证明义务。
5. 替代实现新增的分配、遍历、排序、缓存访问或容量上限。
6. Review 需要理解的核心不变量是否真的减少；错误路径是否更简单。

“删了 100 行”若只是把问题藏进更抽象的泛型里，不算同等收益。也不建议把这些项目随意加权成一个看似精确的分数。

### 8.2 先定预算，再看成绩

以下是**建议起点，尚未由用户批准**，执行者应在正式决策前确认：

- 无法接受任何模型/错误契约/并发安全回归。
- 主矩阵每个格子的端到端慢化预算可先设 3%；完整组合预算同样先设 3%，不能每项都累加 3%。
- RSS 同时设相对与绝对预算，例如 5% 且不超过 256 MiB；还需满足用户可用内存、不中途 swap 的硬限制。具体数字根据基线规模修订。
- 删除完整模块或跨模块协议、收益明确而代价超过预算时，提交单独 trade-off 供用户决定，不自动合入。
- 仅删除几十行且引入可见退化，默认保留；连续存储若确认有约 10% 级收益，优先保留并考虑只简化外围策略。

统计判断应是“慢化区间上界低于预先约定预算”，而不是“p 值没显著，所以等效”。区间跨预算：证据不足。内存风险、罕见输入严重退化、安全边界不能被平均成绩抵消。

### 8.3 决策表模板

| 候选 / SHA | 真正命中的机制 | 净减生产 LOC / 消失协议 | 最差主矩阵慢化及区间 | RSS 绝对/相对差 | 压力输入 | 完整模型与测试 | 结论 / 理由 |
|---|---|---|---|---|---|---|---|
| 待填 | 路径计数证据 | 不含删测试 | 单项逐格附表 | 峰值口径 | 不隐去失败 | 实际执行日志 | 保留/删除/证据不足 |

## 9. 组合、回退与实际执行顺序

1. **证据准备**：固定 SHA、历史结果、机器与配置；完成 A/A；明确预算。产物：manifest、历史证据表、噪声报告。
2. **第一波隔离**：I0/I1/I2/I3、U24→U32、B0/B1，先正确性再筛选。若历史已有等价实验，先核对并复用方法，不盲目重做。
3. **第一波物理删除**：只对通过初筛且有实质简化空间的项创建删除补丁，清理死代码后重新运行完整测试与决策矩阵。
4. **形成 Pareto 候选**：留下更少复杂度与可接受速度/内存的组合。单项可接受不代表组合可接受，I3 必测，其余候选按共享热点/布局选择交叉测试。
5. **逐项累积组合**：每加一项，既比较上一组合，也比较原始基线；记录累计回归。失败时撤回最近独立补丁，必要时定位相互作用。
6. **第二波按缺口启动**：若第一波代码简化不够，再试 AA 并行协议、lookup、prefix cache；不要为凑行数直接删核心内存结构。
7. **最终审查**：组合版核数与内存曲线、默认/no-default 测试、重要 fixture 与失败路径、文档一致性。检查无实验开关、诊断输出、原始大文件进入提交范围。
8. **交付用户决定**：给出推荐组合、保留机制理由、精确补丁与证据。先不改上游 PR；是否采用简化版或外部 crate，仍需与维护者确认边界。

每个候选独立提交，记录创建时基线，不改动共享 baseline。测试失败立即停止该候选性能排名，先修复或撤销；不要把错误版本跑出的快结果纳入比较。

## 10. 给执行 Agent 的交付清单

- 一个 manifest：基线/候选 SHA、工具链、构建参数、二进制哈希、CPU/worker/亲和性、语料哈希与 trainer 配置。
- 一组最小隔离补丁，以及对应真正删除的补丁；明确哪些路径仍保留。
- 正确性报告：实际执行命令、结果、完整模型指纹、定向路径命中证据；未执行项如实列出。
- 逐次机器可读结果、失败记录、配对顺序、统计脚本；能复算全部结论。
- 模块级复杂度账本和逐工作负载 trade-off 表。
- 最终推荐组合与未采用候选；清楚区分“没有收益”“有收益但不值得复杂度”“噪声太大无法判断”。

**本轮的成功条件是得到可复核的删除决策，而不是一定删掉预先指定的模块。** 若第一波都证明有价值，结论应是这些特化值得保留，并转向接口收敛、职责划分或外部 crate 边界，而非继续硬删。

## 11. 固定版本源码参考

以下链接对应本计划静态判断，不代表已经执行验证。

- [引擎 DESIGN](https://github.com/Yikai-Liao/tokenizers/blob/e4f787dc189d9be7192107490d652096cde7480e/tokenizers/tk-train/src/trainers/bpe/engine/DESIGN.md)
- [Bounded collector](https://github.com/Yikai-Liao/tokenizers/blob/e4f787dc189d9be7192107490d652096cde7480e/tokenizers/tk-train/src/trainers/bpe/engine/initial_pairs/bounded.rs#L1-L292)
- [Initial collector 分派与 record](https://github.com/Yikai-Liao/tokenizers/blob/e4f787dc189d9be7192107490d652096cde7480e/tokenizers/tk-train/src/trainers/bpe/engine/initial_pairs.rs)
- [Radix sorter](https://github.com/Yikai-Liao/tokenizers/blob/e4f787dc189d9be7192107490d652096cde7480e/tokenizers/tk-train/src/trainers/bpe/engine/storage/radix.rs#L126-L449)
- [Slot 表示](https://github.com/Yikai-Liao/tokenizers/blob/e4f787dc189d9be7192107490d652096cde7480e/tokenizers/tk-train/src/trainers/bpe/engine/corpus/slots.rs)
- [Prepare 策略与 SelectedRuleIndex](https://github.com/Yikai-Liao/tokenizers/blob/e4f787dc189d9be7192107490d652096cde7480e/tokenizers/tk-train/src/trainers/bpe/engine/merge/prepare/mod.rs)
- [普通 merge 与 producer 编码](https://github.com/Yikai-Liao/tokenizers/blob/e4f787dc189d9be7192107490d652096cde7480e/tokenizers/tk-train/src/trainers/bpe/engine/merge/prepare/ordinary.rs)
- [AA 选择](https://github.com/Yikai-Liao/tokenizers/blob/e4f787dc189d9be7192107490d652096cde7480e/tokenizers/tk-train/src/trainers/bpe/engine/merge/prepare/aa.rs)
- [Pair index](https://github.com/Yikai-Liao/tokenizers/blob/e4f787dc189d9be7192107490d652096cde7480e/tokenizers/tk-train/src/trainers/bpe/engine/pair_index.rs)
- [测试目录](https://github.com/Yikai-Liao/tokenizers/tree/e4f787dc189d9be7192107490d652096cde7480e/tokenizers/tk-train/src/trainers/bpe/engine/tests)

