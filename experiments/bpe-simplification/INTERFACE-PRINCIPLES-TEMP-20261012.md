# BPE 模块接口整理原则（临时）

本文件用于本轮接口调整与独立审视。范围仅为普通 BPE 训练目录；Parity Trainer、WordPiece 及其他训练器和外部调用方不在本轮修改范围内。基线为 `2f74a832`，保留现有模型结果、错误语义和并行阶段关系。

## 从 Positions 改动得到的标准

原来的 `tasks() -> ReadTask` 要求调用者先接收任务对象，再调用 `.iter()`；merge 还需要 `Source::Slice / Task` 适配两种来源。现在 `chunks()` 直接返回可独立消费的标准迭代器，merge 接受位置流，两个自定义概念随之删除。restart 对齐、起始 leaf 定位、读取边界、目录和 codec 均由 Positions 内部维护。

这种封装的价值是：模块提供完整能力，调用者减少必须理解的规则、维护的状态和配合执行的步骤。代码数量是结果证据；文件搬家、压缩写法或给原有步骤增加包装，不能单独证明简化。

## 调整原则

1. **追踪一条真实调用路径。** 写清调用者现在必须知道什么、哪些改变会要求多个模块同步修改，再决定边界。不能仅根据函数长度、文件大小或方法数量找问题。
2. **实现决定归实现所有者。** Corpus 提供语料与 ID 数量等事实；index 决定计数表示和阈值；Positions 维护位置流的排序、重复值和存储规则。调用者独有的业务选择仍由调用者决定。
3. **提供能直接完成任务的能力。** 优先消除调用者的装配循环、转换、缓存状态和先后协议。接口应能说明输入、输出、所有权及成立条件。
4. **优先标准迭代器与直接所有权转移。** 能以 `impl Iterator`、切片和现有容器表达的能力，不额外引入任务类、服务、trait 层次、`Box<dyn Iterator>` 或解压后的中间 Vec。必要的混合来源可在内部用现有 enum 组合。
5. **一次移动必须有连带收益。** 新接口落地后检查调用侧是否真的删除了概念、状态或内部知识；纯移动可以改善知识归属，但应明确其收益，不声称因此减少了实现复杂性。
6. **保留语义上必要的区别。** 新身份与历史身份复用、AA 与普通规则、整体有序连接与交错流归并、删除与出生事件都有不同契约。相似代码不能成为强行合并的理由。
7. **保留完整的并行阶段边界。** 所有 preparation 读者完成后才能写入；所有写入完成后才能发布 index。错误仍丢弃整个 attempt，不增加回滚承诺，也不让惰性读取越过阶段边界。
8. **保留可验证的性能机制。** 本轮接口调整保留 restart 定位、16 项预取窗口、固定跨度缓存、权重区间缓存、片段顺序和无重编码连接。不得靠封装引入额外分配、排序或全量扫描；性能主张需要实际证据。
9. **检查完整使用与失败路径。** 空流、重复坐标、full-u64、片段交错、过期候选、零权重出生、逐事件溢出、跨任务邻接和身份复用都以现有契约为准。
10. **同步权威说明。** 代码、模块 README 和 DESIGN 应描述同一套当前接口及表示；历史实验报告保持历史事实。生产代码、测试和文档的增减分别看待。
11. **收敛公共接口。** 模块用少量完整入口服务真实调用任务，避免调用者通过多个取值或动作方法重建能力。总是配套使用的参数、输入与状态在所属模块内部绑定；端点、目录、缓存和布局读取优先私有。新增公开入口须说明现有入口为何不足，不能靠把每段内部操作公开来实现封装。接口数量是线索，实际验收看调用义务是否减少；必要生命周期和业务决定仍应可见，测试诊断入口只在测试构建中暴露。

## 本轮已确认的调整

- Corpus 提供带权重的 fresh match 迭代器，内部承担坐标转换、预取、过期坐标过滤和权重区间缓存；merge 保留规则选择、AA 贪心非重叠选择及相邻选中规则的事件归属。
- CorpusPlan 只提供 token 数量，`InitialPairCounts` 自己选择 dense/sparse 表示和阈值。
- Positions 提供有序流归并能力；index 根据 fresh/reuse 语义选择连接或归并，不自行重建位置流。
- 更新 README/DESIGN 中旧的范围读取接口与旧存储描述。

## 独立审视的新发现与处理

独立审视者核对了初始化、候选选择、reuse 准备和索引发布路径，提出将 `CorpusPlan::initial_edges` 从带错误回调改成惰性标准迭代器。该方法自身只生成边，计数及其溢出错误由 index 负责；主线采纳此调整，保持过滤、affix、whole-word range、坐标顺序和消费出错即停止。原测试仍检查消费失败前的精确前缀，并检查剩余边未被提前消费。

最初的审视保留了 reuse 词内扫描，因为单纯返回 Match 无法表达虚拟左邻居。后续重新审视找到完整接口：Corpus 同时返回匹配、权重与逻辑邻居，将 candidate seek、失效位置推进和虚拟 replacement 的几何模拟收为私有实现；merge 保留长度门槛、聚合和 deferred starts。必要的扫描状态仍存在，知识归属变得集中。

候选窥视后再取出、Vocabulary 身份与 Corpus 激活/几何分工、先 count 再 materialize、逐事件删除/出生及 fresh/reuse ledger 的区别均保留。

## Fresh 审视循环

每当独立审视找到有证据且值得调整的新接口问题，处理后启动一个不继承前轮审视上下文的 fresh sub-agent，再从当前代码核对完整调用路径。新一轮没有值得实施的问题时，结束本轮循环；不为继续循环制造候选。

第一位审视者找到 `initial_edges`，该项已采纳并通过其独立改动复审。随后启动新的 fresh 审视者，核对 feed、初始化、count/materialize、候选选择、fresh/reuse preparation、写入、index 发布、模型输出及 WordPiece 消费路径。该轮没有发现值得实施的新接口问题，认为现有必要业务判断和生命周期边界应保留。

用户质疑这一结论后重新打开审视。第三位审视者不继承前轮上下文，也不预读已确认或保留结论，找到 `cohort_matches` 和初始化模式重复传递两项问题。主线实施这两项，删除整个 `CohortPreparation`，收窄 word bounds/weight/matched 等几何入口，同时将 fresh matcher 的 token 读取和邻居目录扩容收回实现内部。上一轮的“未发现”只代表该轮搜索结果，不能作为设计已经完美的结论。

第四位 fresh 审视者确认 fresh 邻接仍泄露端点算术。主线增加 `fresh_neighbors` 及按需的前后 pair 查询，复用 Neighbor 值；AA 使用同宽前一 match 起点及已选起点 membership，不查询额外 pair token。merge 的生产路径不再读取 token 槽位或解释 separator，继续拥有选中映射、共享边界归属和长度门槛。这项收益是知识归属改善，不声称删除了 AA 选择或边界查询机制。

用户进一步要求收敛公共接口后，三个邻接查询改为私有，`fresh_matches` 直接提供匹配、权重及邻接事实；从输入 pair 自行判断 AA，不增加调用者模式参数。第五位 fresh 审视者确认 cohort 的词域装配与 index owner 数透传仍可收敛。`Corpus::cohort` 内部绑定 word Vec、Positions 和快照，Cohort 直接给标准并行 chunks，保留原词块数与借用读取；不新增堆分配或 task 类。`resident/word/token/span/len` 等低层入口成为私有，诊断的 tokens/spans 仅测试构建可见。index 从当前训练 pool 决定 owner 数，协调者删除 workers 透传；失败用例按实际 owner 注入错误，并验证多线程 ambient pool。

测试沿用简短名称和紧凑的字面预期；cohort 回归合并扫描模式与邻居状态，fresh 邻接检查放入现有相邻写入用例，不另建重复 fixture。

第六位 fresh 审视者找到初始 ranges/edges 配对及 alphabet workers 透传。`initial_chunks` 直接给绑定的并行 edge reader 与词数，范围和逐词坐标接口私有；index 只在计数成功后更新进度。alphabet 选择从当前 pool 决定 presence 分块，保留外层 pool 规模选择。InitialTokenIds 的内部 ID 查询、Alphabet 的字符表及 Priority 的 count 字段成为私有；跨模块测试只使用 cfg(test) 的窄诊断入口。已有 Unicode 用例在真正的 1/4 线程 pool 内验证，不增加重复测试。

第七位 fresh 审视者覆盖完整普通 BPE 路径并复核当前 diff，该轮没有确认值得实施的新接口问题。Corpus 本体的生产公开方法由 `399bce65` 的 17 个收为 8 个；整个 corpus 模块还包含 plan、Match 和 cohort chunks 入口。保留 best/take、AA matcher 与长度事实入口，因为目前未找到能减少调用义务且不引入选择策略耦合或额外读取的完整替代。

用户要求扩大非 Corpus 审视覆盖后，第八位 fresh 审视者找到 builder 配置镜像和输出历史装配负担。builder 直接持有 BpeTrainer，删除 Config 和 build 的逐字段转译；默认值只定义一次。Vocabulary 拥有已提交的 ID 规则历史，merge 在整轮 index 发布成功后追加历史和测试 trace，只返回完成量；协调者删除 MergeRecord 装配与历史 Vec，只维护进度计数。输出转换仍在大块 attempt 存储释放后进行；保留 reuse 的重复历史记录及选择时的 count/replacement。记录分配仍存在，不声称这次移动减少了该分配。

后续 fresh 审视严格限于普通 BPE 训练目录，不修改其他训练器或外部调用方。

第九位 fresh 审视者从完整普通 BPE 路径和失败分支独立核对当前代码，该轮未确认新的值得实施问题，结束本轮循环。保留 FreshMatch 的普通规则／AA 辅助字段，因为 merge 仍须决定本批 membership、共享边归属与长度门槛；仅换成 enum 或把选择责任移入 Corpus 尚未显示可删除状态。保留 best/take 的窥视边界及 candidate 在准备后释放的生命周期。未发现只靠接口数量判断即可采纳的新调整。

## 公共接口统计

按 Rust 语法树统计当前普通 BPE 生产声明，包含 pub/pub(crate)/pub(super)，排除 cfg(test)、可选 parity 和隐式 trait 实现方法。字段另计。内部 codec 的 pub(super) 仅供 Positions 使用。

| 模块 | 公开函数／方法 | 公开 Struct | 公开 Enum | 公开字段 |
|---|---:|---:|---:|---:|
| bpe 主模块 | 17 | 2 | 0 | 10 |
| corpus | 16 | 7 | 0 | 15 |
| feed | 1 | 0 | 0 | 0 |
| index | 7 | 3 | 0 | 2 |
| merge | 1 | 1 | 1 | 6 |
| positions | 8 | 1 | 0 | 0 |
| positions::codec | 4 | 1 | 0 | 0 |
| vocabulary | 11 | 3 | 0 | 0 |
| word_counts | 8 | 2 | 0 | 0 |
| 合计 | 73 | 20 | 1 | 33 |

另有 codec 内部公开常量 1 个。对库使用者直接公开的是主模块 17 个函数／方法、2 个 Struct 和 10 个字段。

相对 `399bce65`，全部模块函数／方法从 82 个降至 73 个，Struct 从 17 个增至 20 个，字段从 23 个增至 33 个。corpus 模块自身是 24→16、3→7、2→15。Corpus 本体方法减少不能单独证明整个模块接口收敛；新增值类型与公开字段也是调用者须理解的契约，应按完整调用负担评估。

## 本轮验证

- 默认特性库测试与无默认特性库测试分别通过 42 项，均包含新接口回归与 64 组输入在 1/4/8 worker 下的独立模型/规则对照。最终默认配置使用 4 线程 ambient pool，无默认配置使用 1 线程，覆盖 owner 数内收后的直接初始化与失败用例。
- 沿用此前验证范围，两种配置均跳过既有 `tokenizer_roundtrip`（原名 `feed_training_tokenizer_json_and_encoding_roundtrip`）用例；这不是全部库测试通过的声明。
- `cargo clippy --all-targets` 通过，使用 `-D warnings -A clippy::redundant_field_names`，沿用已有冗余字段名 lint 的例外。
- 只对本轮修改的七个 Rust 文件运行项目 formatter，检查与 `git diff --check` 均通过。
- 存储检查 harness 已补上当前 codec 使用的 leb128fmt 依赖；`cargo check --tests` 通过。本轮未运行 Miri 解释器或性能实验，也不主张整体训练加速。
- README、DESIGN 和 COVERAGE 已同步到压缩 leaf/root、opaque chunks 及本轮接口的实际行为；历史实验报告未改写。

## 独立审视的交付要求

独立审视者继续寻找上述调整之外的候选，核对实际调用代码。每个候选写出位置、现有调用负担、建议接口、能删除的知识或状态、必须保留的契约以及可能新增的成本。允许得出保留现状的结论。

审视者只提供建议，不修改主线代码、不运行重型性能实验、不发布远端变更。重点检查跨模块的完整路径，避免把单行 getter、改名、额外包装或新增模式参数当作深模块改进。
