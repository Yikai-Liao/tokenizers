# BPE 剩余设计与实施计划

2026 年 10 月 11 日。剩余工作分成两组：**预计运行时影响小的接口与职责整理**，以及**可能明显影响临时内存和运行时间的存储调整**。性能影响与正确性风险分开判断；Merge 收口属于前者，但正确性风险高。

本文只给设计和验收计划，未修改、编译或运行仓库代码。签名是说明边界的草案，尚未编译验证。

## 当前范围与已有进展

- 此前已核对的范围是 [ac32723 → 7cb1bac](https://github.com/Yikai-Liao/tokenizers/compare/ac32723d3dd555d7810d794db86a7a2ae65d7181...7cb1baca6a33efb6cf797c6323a58c0705f8f39d)，这段增量只涉及初始区间指数搜索及其依赖、说明和实验材料。最新已确认 HEAD 为 [2f4803a936619aa575ba4f6cb49fc660057ef5e4](https://github.com/Yikai-Liao/tokenizers/commit/2f4803a936619aa575ba4f6cb49fc660057ef5e4)，其 Feed 增量正在单独复审。本文设计引用的 `mod.rs`、Vocabulary、Corpus、Merge、Index 和 Positions 相对 `7cb1bac` 未变，以下固定链接仍适用。
- `initial_ranges` 指数搜索已落地，Range 接口已修好；不再列为待实现项。[当前实现](https://github.com/Yikai-Liao/tokenizers/blob/7cb1baca6a33efb6cf797c6323a58c0705f8f39d/tokenizers/tk-train/src/trainers/bpe/corpus.rs#L129-L156)
- Feed：用户已确认完成并推送，复审进行中。只保留当前改动的复审与集成验证，不重新分配实现任务，也不把旧建议当成未完成工作。
- Positions 和 Index 按用户确认的已优化状态保持不动；外部 Positions 容器替代实验因性能失败已关闭。不重新做 arena、mimalloc 或 Elias–Fano 替代实验。
- 固定表的 `Vec → Box`：已提交快照仍有 Vec，但其他工作中的实际进展待核对。先查现有工作，不重复发起，也不列为本轮必做项。
- 开工先检查实际 HEAD、工作树和在途任务，保护其他未提交改动，并保留已推送的 Feed。固定 SHA 只是本文的事实锚点，不要求退回旧提交开发。

## 改动分类

| 项目 | 预计运行时影响 | 正确性风险 | 当前处理 |
| --- | --- | --- | --- |
| A1 恢复 Builder 的原始 Config 边界 | 小，主要是构造期常数级变化 | 低 | 可独立实施 |
| A2 完整初始化 Vocabulary 并交出 InitialTokenIds | 同遍历、同释放点时小 | 中高，涉及 ID 与重启 | 独立实施并验证 |
| A3 在 merge.rs 实现 BpeTrainer::merge_next | 沿用算法、记录 Vec 和释放点时小 | 高，涉及提交顺序与失败边界 | 独立实施并验证 |
| A4 注释和阅读顺序 | 通常可忽略，不承诺二进制完全相同 | 低 | 随相应修改完成 |
| B1 Corpus 测量数组原地转为 starts | 临时内存收益可观，时间影响待测 | 中，涉及数值检查 | 单独对照后决定 |
| Feed 集成 | 已推送，不新增设计实验 | 按实际改动核验 | 复审单独进行 |
| 固定表 Vec 转 Box | 尚不能分类，取决于具体转换与寿命 | 待核对 | 不重复启动 |

“小”是设计预期，不是零成本保证。若 A2/A3 实际增加扫描、复制、分配或延长大对象寿命，必须重新归入需要单独性能对照的改动。

## A1 恢复 Builder 与配置的原始边界

建议让 `BpeTrainerBuilder` 重新持有原始 `Config`，以 `f4ed232` 的 10 个配置字段为参照。`build()` 消费配置、构建 trainer，并创建空输入计数；公开 Builder API、默认值和 serde 行为不变。[当前入口与配置](https://github.com/Yikai-Liao/tokenizers/blob/7cb1baca6a33efb6cf797c6323a58c0705f8f39d/tokenizers/tk-train/src/trainers/bpe/mod.rs)

- Builder 内嵌 Trainer 本身合法，但它解决的是与这轮 BPE 设计无关的 DRY 问题；为减少无意义的上游差异，恢复原配置边界更合适。
- 只精确修改配置与构造代码。不能整文件恢复 `mod.rs`，不能 reset 分支，也不能整份替换成上游 trainer；现有必要的 worker、progress 及公共适配逻辑保留。
- 不改 BPE 算法，不宣称节省堆分配：当前空 Vec 本来就不需要元素分配。
- 验收：10 项配置、默认值、Builder 方法、序列化及公开调用行为一致；`build()` 后输入计数为空。

## A2 让 Vocabulary 初始化一次完成

沿用现有 `Vocabulary` 与 `InitialTokenIds`，让初始化直接返回二者。删除调用者继续执行 `initial_ids`、再执行 `take_initial_spans` 的补完协议。[现有初始化边界](https://github.com/Yikai-Liao/tokenizers/blob/7cb1baca6a33efb6cf797c6323a58c0705f8f39d/tokenizers/tk-train/src/trainers/bpe/vocabulary.rs)

```text
Vocabulary::initialize(...) -> Result<(Vocabulary, InitialTokenIds)>
InitialTokenIds::token_count(&self) -> usize
InitialTokenIds::into_spans(self) -> Vec<usize>
CorpusPlan::build(words, ids, trainer, reuse) -> Result<CorpusPlan>
```

- `Vocabulary` 只留下后续训练需要的 token 身份表和 prefix；初始化 suffix 从 trainer 借用，`plain_ids_resolved` 留作初始化局部状态。
- `InitialTokenIds` 拥有初始激活 spans、字符与 decorated ID 查找表，以及解析符号所需标志。`token_count()` 直接取 `spans.len()`，不另存一个需要同步的数量。
- `CorpusPlan` 接收并持有 ids，不再接收 `&mut Vocabulary`、progress 或单独一份 spans。字符与身份规则仍归 vocabulary.rs；位置与词边界计算仍归 corpus.rs。
- materialize 完成符号写入后消费 `into_spans()`，移动 spans 并释放查找表；不复制 spans，不让查找表留到 merge 阶段。保留“初始词表已达到目标时不分配 resident endpoints”的捷径；低于目标但最终没有可合并 pair 的情况，当前仍可能物化，不能混为同一条件。布局验证和初始 pair 的溢出检查仍在捷径之前执行。
- 保持原有遍历次数与释放时点，不为接口统一再扫一次语料。初始化进度交给实际初始化阶段处理，外层 trainer 继续统筹各阶段进度；保持原有启动、更新和结束语义。

必须保留：special/alphabet ID 顺序；decorated ID 按原输入顺序解析后才按词频排序；prefix/suffix 依据原始 UTF-8 字节边界判断；无限 alphabet 快路径；fresh → reuse 共用原来选出的 limited alphabet。保留 ID 未激活与曾激活的区别，也不重新抽取频率并列项。

验收重点：Unicode、多字节 affix、首尾过滤、decorated ID 的完整顺序、limited alphabet cutoff 并列、reserved ID 首次激活、零 merge 与 reuse 重启。cutoff 用例必须真的超过 alphabet limit。

## A3 将完整合并轮次收进 merge_next

在 `merge.rs` 为现有 `BpeTrainer` 实现实质性的 `merge_next`：包含候选选择循环，以及 prepare、apply、index commit。调用者不再接收 `Selection::Ready(Batch)` 后自己续完轮次；模块内部仍可保留有用的私有 Batch。[现有选择与提交](https://github.com/Yikai-Liao/tokenizers/blob/7cb1baca6a33efb6cf797c6323a58c0705f8f39d/tokenizers/tk-train/src/trainers/bpe/merge.rs)

```text
BpeTrainer::merge_next(...) -> Result<MergeOutcome>
MergeOutcome = Finished | RestartForReuse | Committed(Vec<MergeRecord>)
```

- `BpeTrainer` 继续实际执行线程池建立、attempt、fresh → reuse 重试、进度与结果收集，不增加 Engine、Workset、Model 或新执行门面。
- `merge_next` 不是包一层旧 select/commit 协议的转发函数；把完整一轮的控制流放在一起。现有 `Batch::commit` 已返回记录 Vec，直接沿用，不再复制一份结果或增加一份堆分配。
- 保持 consuming `prepare(self)`。选中候选及其 Positions 在 prepare 返回或失败时释放，不能改成借用 prepare 而让它们陪同后续写入存活；不克隆 Candidate 或 Positions。
- 保持严格顺序：全部准备成功并 join → 端点写入全部 join → 索引 owner 提交与原序发布全部成功 → 返回 `Committed(records)` → 外层追加 merges/trace。
- 保持选择顺序、优先级前缀、冲突即停止、AA 与 shared head/tail 规则、reserved ID 首次激活、曾激活身份触发 reuse，以及相同 pair 不同历史 cohort 的记录。测试 count 仍取选择时认证的 count。
- 错误或重启丢弃整个 attempt。选择阶段可以已有局部状态变更；这里不承诺单轮回滚，也不增加回滚机制。Index 与 Positions 的实现不动。

验收重点：AA、reserved ID、reuse、相同优先级与历史 cohort、准备失败前不得写端点、计数溢出、所有阶段的 join 与释放顺序。使用固定输入比较完整词表 ID、ordered merges 和必要 trace，并覆盖 1/4/8 workers。

## A4 按真实职责整理注释和阅读顺序

先展示完整初始化或合并操作，再放辅助实现。把含糊的 geometry 表述换成具体的“位置”“跨度”“词边界”，仅在含义准确的地方修改；不进行全仓自动改名。注释解释顺序、寿命和不变量，不把已有正确性边界写成抽象口号，也不顺手扩大上游差异。

## B1 将 Corpus 测量数组原地变成 starts

保留“并行测量每词符号数 → 串行 checked 前缀计算”的分工。串行循环先取出该槽符号数，再把同一槽改写成起始位置，避免同时持有 `measured` 与 `starts` 两份等长数组。[当前测量与前缀计算](https://github.com/Yikai-Liao/tokenizers/blob/7cb1baca6a33efb6cf797c6323a58c0705f8f39d/tokenizers/tk-train/src/trainers/bpe/corpus.rs#L62-L101)

- 对 W 个输入条目，64 位平台少一份约 `8 × W` 字节的 usize 元素存储。这是该阶段的结构估算，不保证进程 HWM 同量下降，也不预先保证训练提速。
- 保留 length 从 1 开始、每词 separator、空词与完全过滤词、权重区间、checked length、signed mass 和 resident allocation bound 检查；零 merge 路径也照常验证。
- 不改已经落地的 `initial_ranges` 搜索，不碰初始 pair counting、Index 或 Positions。单独提交，测完临时内存和完整训练后再决定保留。
- 不夹带 `InitialPairCounts` freeze、slab 或索引所有权重构；这些不在本轮范围。

## 验证与集成顺序

1. 核对实际工作树与在途进度，保护未提交修改；记录各候选与基线的源码、构建选项和输入哈希。结合已推送 Feed 的复审与集成核验，冻结一致的对照基线。
2. A1 独立提交；A2 与 A3 各自独立提交。A4 随相关切片完成。实现可以在独立 worktree 并行，改到同一入口文件时协调集成；B1 保持独立便于归因和回退。
3. A 类先跑受影响的正确性测试。需要排除意外变化时加一次配对性能 sanity check；若有扫描、分配或寿命变化，升级为 B 类验证，不机械要求所有小整理跑完整矩阵。
4. B1 使用同机、同输入、同编译配置、同 workers/CPU 绑定的 256 MiB 中英文语料，分别覆盖 ByteLevel 与 Whitespace。先排除一对预热，再做两组正式配对；只有噪声、方向不一致或决策仍不清楚时追加，不强制每场景五组。
5. 性能测试串行执行，期间不并行构建或跑其他实验。记录 wall、CPU、HWM、swap 与完整模型哈希；先测孤立候选，再测最终集成版本，保留原始结果和基线已知失败。
6. 接受有解释的小幅性能回退，不虚构统一 5% 门槛。用实际重复结果、内存变化和删掉的调用者负担共同判断；不能用单轮波动证明收益或退步。

Feed 集成只核验已完成实现：callback/error 契约、真实 cache flush 后出错仍不替换旧 counts、counts 等价及与训练入口的衔接。具体实现若改变了并行调度或输入顺序，按其证据检查相应影响。后续所有 A/B 对照两边必须包含同一份 Feed，不能把 Feed 变化混入其他候选的性能差值；固定输入的训练模型比较与各自 Feed 后的模型比较也要区分。

已知 `feed_training_tokenizer_json_and_encoding_roundtrip` 编码断言在基线失败，须继续标注这一验证缺口；不能为了全绿删除断言，也不能把过滤后通过写成全测试通过。[现有验证记录](https://github.com/Yikai-Liao/tokenizers/tree/7cb1baca6a33efb6cf797c6323a58c0705f8f39d/experiments/bpe-simplification/evidence/initial-ranges-exponential-20261011/explicit-one-word)

另有独立的 Feed 验证事项：新归档记录早期 A 版本曾在训练时触发 `coordinate belongs to the resident corpus` panic；同一二进制重试成功，最终规模矩阵 24 次 counts/model 对照通过，但这次失败尚未归因，不能列作旧 encoder 断言失败，也不能认定移除 reserve 已修复它。当前 Feed 复审需单列处理，不通过扩大本计划来掩盖这个缺口。[Feed 实验记录](https://github.com/Yikai-Liao/tokenizers/blob/2f4803a936619aa575ba4f6cb49fc660057ef5e4/experiments/bpe-simplification/evidence/feed-cost-20261011/README.md)

最终验收看三件事：初始化和完整轮次是否减少了调用者必须遵守的协议；ID、顺序、错误与释放边界是否保持；新增成本是否被隔离并如实测量。完成这几项即可收束，不再为减少类型数量另起一套架构。
