# BPE 词频查询优化实验计划

供实现 Agent 执行。先证明有优化空间，再做最小改动；没有稳定收益就保留现状。不预设提速比例。

基线：`Yikai-Liao/tokenizers`，分支 `refactor/bpe-corpus-boundary-20261013`，提交 `e4c35beeb6f8931bb6dc00a3162bca0bdff46a19`。开始前核对 HEAD；若有新提交，先更新事实，不能混用版本数据。

## 1 当前实现与范围

- `WordCounts` 是唯一键、无序的 `Vec<(CompactString, u64)>`；`CorpusPlan` 对借用引用按词频降序排列。每种频率对应一个 `WeightRun`，词间 separator 保留。[布局源码][layout]
- fresh AA/AB reader 已缓存 `WeightLookup { weight, end }`，有效坐标非递减，命中只检查 `p < end`。缓存失效才查询，但仍在整个 run 目录做 `partition_point`；单 run 和 unit-weight 快路径已经存在。[查询源码][lookup]
- reuse 的 `Corpus::scan_word` 已在循环前取一次词频。不要重复实现 hoist，也不要把旧版本逐匹配查询的成本算作新增收益。[reuse 源码][reuse]
- initial edges 已直接携带 weight，不改初始计数、排序、调度或公开接口。

## 2 只保留这些候选

- **A 基线**：当前提交。
- **C 主候选**：fresh 缓存 run 索引；跨区间后仅在 `weight_runs[i + 1..]` 中二分。
- **B 条件候选**：仅当零匹配词很多且提前查询有可测成本时，将 reuse 改为首次成功匹配时查一次，随后按词复用。
- **D 后续候选**：仅当 C 有收益且剩余查找仍值得优化时，根据跳跃分布选择“有界下一 run 检查后回退二分”或“指数搜索后区间二分”。优先复用已有 `slice_group_by::exponential_search_by`；不做无界线性扫描。

各候选先独立对比 A，不展开策略、阈值和布局的大矩阵。仅独立胜出的改动才考虑组合，组合须重新验证。

## 3 先诊断机会

用独立诊断补丁收集 task 本地计数，汇总后移除；正常计时不能带计数器、逐项日志或共享原子开销。

fresh 记录输入坐标、有效匹配、最终接受匹配、reader 数、区间缓存命中、首次查询、跨 run 失效、单 run/unit 快路径及真正二分的次数与比较次数；记录首次查询位置、剩余后缀长度，以及索引跳跃 1、2–4、5–16、更多的分布。为统计而额外做的查找单列。

reuse 记录扫描词数 S、零匹配词数 Z、单匹配和多匹配词数，区分整词与历史坐标扫描。当前查询次数是 S，B 最多降到 S−Z；多匹配的重复查询已经消除。

结合独立采样 profile 和阶段时间判断成本上限。目录小、缓存几乎全中、unit 快路径占主导或 task 通常只查一次时，C 可能没有机会。调用次数减少不能直接换算成端到端收益；机会为零就停止，不先跑完整基准。

## 4 实现必须保持的约束

- 首次有效位置可在任意 run，不能假定从头开始；后续非递减且允许重复。保持 matcher 先过滤 stale 坐标，再查词频；缓存仅属当前 reader/task/快照。
- 命中仍走 `p < end`。失效后利用旧索引搜索后缀，处理末尾、空后缀与索引减法；最后一个 run 的 end 是 corpus 长度。
- 保留单 run 与 unit 快路径。unit 返回时也要拿到正确索引，不能先查 weight 再为索引二次搜索。权重 1 后可能还有 0，不能当作无限尾区间。
- B 用显式未初始化状态，不能拿 0 作哨兵。保持匹配顺序、邻边与长度限制语义。
- 可调整现有私有缓存，只有真能减少重复逻辑才抽私有 cursor；不新增 Engine、DTO、公共 API 或逐词/逐坐标权重数组。
- 测量缓存及其 Option、reader 状态的实际大小。裸缓存常见为 16→24 字节，不能据此断言代价可忽略。检查栈帧、复制、代码体积和寄存器压力，保留 decoder 借用；计入全部准备阶段临时内存及新增元数据。

## 5 分阶段验证

1. **固定环境**：记录 SHA/diff、编译参数、二进制和输入哈希、CPU、runner、训练参数及线程配置。核对真实 Whitespace/ByteLevel 设置与子进程环境；先 smoke run 验证入口和模型比对。
2. **诊断**：先选一个代表性配置，必要时补另一个路径。无机会即停止对应候选。
3. **筛选**：最小实现通过针对性测试后，以正常 release 构建做约 3 对随机顺序试跑，淘汰明显退化并估算完整验证耗时；这一步不能宣称稳定提速。
4. **完整验证**：只让 A 与一个最终候选进入中文/英文各 256 MiB × Whitespace/ByteLevel × 1/6 workers，共 8 格。每格至少两轮，每轮每版本 5 样本；成对随机化 A/候选顺序，保存种子和执行顺序，统一预热且不计入样本。不同基准不并行争抢资源。
5. 两轮仍有噪声时，仅对受影响配置追加成对样本，先说明原因和耗时预算。报告分布、配对差值/比值与不确定性，不挑最快单次。最低完整矩阵是 160 次测量，必须通过前置门槛后再运行。
6. 主矩阵若未触发 reuse，不能证明 B 有效；只有 B 的诊断机会成立时，另选一个真实 reuse 配置验证，不再乘出新矩阵。

正常运行记录总 wall time、user/system CPU、峰值 RSS、完整模型及有序 merges。沿用已有阶段指标；需要额外插桩的阶段分析另跑，不能替代正常构建验收。结合状态大小报告总准备内存，不能只看 RSS。清理诊断或组合补丁改变热路径后，要复验最终版本。

## 6 正确性与测试预算

复用现有 fixtures 和 oracle，以小型表驱动用例补缺、替换等价重复覆盖。当前测试已超 1500 行目标；记录增减，不新建框架、不压行作弊，也不顺带重写测试架构。

必须覆盖：
- 等频词共享 run；重复与非递减坐标；task 任意首位置；跨多 run 跳跃；末尾边界。
- 单 run、权重 0/1、进入及离开 unit 区间，尤其 1 后还有 0；空输入或无有效匹配。
- stale/holes/separator 0 先过滤；reuse 零、单、多匹配。
- AA 重叠、AB 邻边与 task 边界；aliases、occurrence spans、max-token-length；既有溢出检查。
- 1/6 workers 下完整词表及 ID、配置和有序 merges 精确一致。仅对象键顺序不同可做结构化比较；不得排序 merges、忽略 ID 或删除不一致字段。

## 7 验收与交付

出现语义差异、正常构建无法复现收益、重要配置反复退化，或复杂度/状态成本抵消收益，就停止并保留 A。合理追加样本仍无法区分收益与噪声，也不合入。不要用汇总平均值掩盖某个配置回退。

交付：保留/放弃结论、最小 diff、基线及候选 SHA、诊断与停止原因、原始样本及复现命令、各配置 wall/CPU/RSS 与状态变化、精确模型/merges 比对、测试及行数变化、未验证事项。允许结论为“维持当前实现”，不把预期写成结果。

[layout]: https://github.com/Yikai-Liao/tokenizers/blob/e4c35beeb6f8931bb6dc00a3162bca0bdff46a19/tokenizers/tk-train/src/trainers/bpe/corpus.rs#L95-L137
[lookup]: https://github.com/Yikai-Liao/tokenizers/blob/e4c35beeb6f8931bb6dc00a3162bca0bdff46a19/tokenizers/tk-train/src/trainers/bpe/corpus.rs#L296-L398
[reuse]: https://github.com/Yikai-Liao/tokenizers/blob/e4c35beeb6f8931bb6dc00a3162bca0bdff46a19/tokenizers/tk-train/src/trainers/bpe/merge.rs#L200-L214

