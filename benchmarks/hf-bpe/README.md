# HF BPE 兼容适配原型

新增热点优化见 [OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md)：当前推荐 J `376363d2`（H+规则邻居聚合），原接口 512 MiB 筛选 train24.613秒、RSS约4.43GiB。H 的历史交错对照、各候选范围与证据保留在报告中。全部组合关系见 [OPTIMIZATION_CATALOG.md](OPTIMIZATION_CATALOG.md)，计划见 [OPTIMIZATION_PLAN.md](OPTIMIZATION_PLAN.md)。

追加 [posting arena 阈值与阶段峰值分析](POSTING_ARENA_THRESHOLD_REPORT.md)：43 次多语言生命周期诊断、12 次同 binary 阈值计时；中文 512 MiB full Bump 没有增峰、train 少约22%，英文小样本的阶段预算与速度不同。阈值按全程资源预算选择，实验尚未迁入生产 J。

追加 [初始化峰值与扩容报告](INITIALIZATION_MEMORY_REPORT.md)：低 scratch block radix、排序/安装分阶段与direct route已核验，512MiB已测组合峰值约3.33–3.46GiB；generic block稀疏计数减少临时表容量，速度混合。数十GiB目标与完整地址/频率边界见 [SCALE_UP_ANALYSIS.md](SCALE_UP_ANALYSIS.md)，全路径论文与实践见 [ALGORITHM_FRONTIER_MAP.md](ALGORITHM_FRONTIER_MAP.md)。新候选在 `bpe/initial-owner-waves`；arena阈值待算法路线确定后再选。

本次交付按实现拆成独立 worktree 和本地分支：HF reference、固定 PR、串行 endpoint、fused、串行初始化并行 merge、并行初始化、原子访问对照。五个新实现都直接接入原始 `BpeTrainer::do_train/train_vocab` 和 `Trainer::train`，公共 Trainer 字段与序列化格式保持一致。完整路径、提交与调用示例见 [WORKTREES.md](WORKTREES.md)。

中央根目录 `bpe/experiments` 保存开发快照、历史实验接口和记录。以下布局与理论说明覆盖这些内部核心；根目录的外挂入口仅用于复现历史测量。当前公平比较固定 **u32 corpus ID、u32 posting**，串行初始化/4线程 merge 是与 PR 的控制项，4线程初始化另列为优化。

开发过程见 [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)，迁移机制审查见 [REVIEW.md](REVIEW.md)，原接口分支复核见 [WORKTREE_REVIEW.md](WORKTREE_REVIEW.md)。最终计时见 [PARALLEL_REPORT.md](PARALLEL_REPORT.md)，历次取舍见 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)。指定初始化时点的源码空间核算见 [MEMORY_LAYOUT.md](MEMORY_LAYOUT.md)。[COMPACT_REPORT.md](COMPACT_REPORT.md) 和 [REPORT.md](REPORT.md) 保存此前版本数据。

固定 affix 和长度限制的进一步理论证书分别见 [AFFIX_PRUNING.md](AFFIX_PRUNING.md)、[LENGTH_PRUNING.md](LENGTH_PRUNING.md)，其中额外优化尚未接入代码。

原型与 HF 共用特殊 token 和 alphabet 初始化；字符装饰及 ID 分配保持相同规则，直接写入扁平语料。普通核心使用 u32 稳定位置、端点编码、单一 posting 所有权、16 字节候选以及共享出生链；普通字符 BPE 按具体位置合并，并按已证明的旧 pair 单调性剪枝，包括有限长度限制。`AA` 按位置排序，保留左到右不重叠选择。非空前后缀配置使用通用 HF 候选词集合及 i64 账本。

## 内存布局

当前布局对照 `efficient_bpe/rust/src/backend.rs` 和 `src/parallel/mod.rs`：

- 一个连续 `Vec<u32>` 保存语料，在活 token 的起止端点保存 ID；合并只改端点，不移动语料位置。
- 一个 `Vec<u32>` 保存每个词表 ID 的字符跨度；独立的 `pivots` 和 `weights` 保存词边界及权重。历史 posting 用二分查询定位所属词，不在每个字符上存词 ID。
- 初始化用栈上的 UTF-8 字符缓冲区和复用的 affix 字符串，直接构建语料；训练路径不构造 `Word` 或 `get_chars()` 数组。
- 普通路径的每 ID 长度表用 0 表示尚未激活。合并命中预留 special ID 时，直接写入首次激活的跨度。通用路径第一次复用 ID 时，从当前活 token 的端点构建 `spans: Vec<u32>`，允许同一个 ID 的不同出现位置具有不同跨度。

串行语料每槽 u32；并行语料在完整可能 ID 域能装入时直接构造 u16，否则使用 u32。HF 的 ID 0 有效，分隔符取对应宽度的最大值。**语料 ID 宽度与 posting 地址宽度独立**：默认 posting 使用 u32 地址块，空间基址和计划位置保留 usize，超出单个 u32 地址块时通过字典分块定位。并行长度表为 usize；串行长度表为 u32。长度限制本身不追加每位置跨度数组，通用路径发生 ID 复用才会触发该分配。

并行规划只读旧语料，AA 用短尾顺查与长尾二分生成摘要，串行传播只遍历摘要数量。有效计划按空间排序后以 `split_at_mut` 分割独占写区，读、写、owner 提交阶段都有完成屏障。默认无语料原子；`atomic_corpus: true` 在相同算法中使用 Relaxed AtomicU16/U32，仅供访问成本对照。当前相对原型新增的全局 16 字节 Plan、混合规则排序和第二次 delta 走访已在独立审查中列明。

## 普通路径的单调性与通用路径的边界

完整数学证明和真实 HF 反例见 [PAIR_MONOTONICITY.md](PAIR_MONOTONICITY.md)。无前后缀的字符 BPE 中，每次合并的字符串身份必定首次出现在语料里。这不要求新的数字 ID：预留 special ID 的首次激活同样满足证明。已经出生的 pair 之后只能失去出现位置，因此可以永久删除低于 `max(1,min_frequency)` 的 pair；新 pair 必须等本轮全部正负变化聚合后才能剪枝。有限长度门控对同一 pair 恒定，仍允许这条路径。

非空 affix 可能让不同原始跨度得到同一个字符串键，例如 suffix `"a"` 下的 `baaba` 会使旧 `(b,aa)` 从 1 增至 2。通用路径保留历史候选及完整账本，不做单调性剪枝；HF 的账本不扣选中边自身、候选携带历史词集合，这些行为均保留。固定 affix 可以扩展的数学证书见 [AFFIX_PRUNING.md](AFFIX_PRUNING.md)，尚未接入实现。

`suffix_alias_really_increases_an_old_pair` 检查上述合法初始化轨迹。`lifecycle_tests::reused_id_resurrects_low_frequency_pair_and_keeps_hf_cohorts` 是人工索引状态的结构测试，不能作为普通 BPE 的可达反例。`monotone_pruning_waits_for_global_birth_count` 检查新 pair 在整轮聚合后才做阈值过滤。

| 参数 | 实现与核验 |
|---|---|
| `special_tokens` | 按 HF 顺序初始化、去重；合并结果与已有特殊 token 同字符串时复用 ID |
| `initial_alphabet` / `limit_alphabet` | 沿用 HF 初始化与裁剪；裁掉字符后邻接的行为也沿用 HF |
| `continuing_subword_prefix` | 按原词字符位置给非首字符加前缀；合并时仅剥离右 token 的前缀 |
| `end_of_word_suffix` | 给原词末字符加后缀；后缀参与 token 字符串身份判定 |
| `vocab_size` / `min_frequency` | 按实际词表大小和 HF 堆中频率停止；复用 ID 不增加词表大小 |
| `max_token_length` | 使用出现位置的字符跨度；保留 HF 初始 pair 全计入及新邻边长度严格 `<` 的行为，不按装饰后的字符串字节数计量 |

字母裁剪的同频边界和前后缀字符的初始化 ID 顺序，继承 HF 本身的哈希遍历行为。线程核验固定同一份词频输入，不宣称重新建立哈希表或重新并行 feed 后所有初始化 ID 都必然相同。

原型的语料位置域（含分隔槽）及词表 ID 小于 `u32::MAX`；权重、初始加权 pair 总量及后续账本更新检查 i64 溢出。HF 原实现使用 i32 频率，超出其范围后的溢出行为不作为兼容契约。

并行路径的全局位置是 usize，u32 是局部 posting 偏移；已经核验跨 2³² 的地址算术，没有为容量验证分配数十 GB 语料。u16 ID 选择检查 target、specials 和初始 alphabet 的完整上界，保留 65535 作分隔符。Halfword/H2.5/H3 是原型另外的串行 ablation，本目录尚未迁入它们。

## 使用

选定实现 worktree 后，使用原始 Trainer 接口，示例与构建步骤见 [WORKTREES.md](WORKTREES.md)。算法由该分支固定，不需要调用 indexed/fused 外挂方法。

训练记录可能包含重复 pair；HF 模型构造器保留同一个 pair 的最后一次 rank。各实现的 `train_vocab()` 返回完整、有序的原始 merge 列表；对照 HF 0.23.2 最终 JSON 时按模型构造器的规则去重。

本轮还修复了 v1 编码器的一个前后缀问题：单字符词表项在存在 affix 时也需要经过编码证明，不能直接认为其自身 ID 就是编码结果。否则输入 `b` 会命中裸 `b`，跳过应输出的 `b</w>`。修正在 [`model.rs`](../../tokenizers/tk-encode/src/models/bpe/model.rs)，回归测试覆盖 ASCII、中文、prefix 与 suffix 同时设置、空 token 和重复缓存调用。

## 已完成的正确性检查

并行初始化分支与原子分支的 `cargo test --offline --locked --no-default-features --lib` 各40项通过；串行初始化分支的原始 `do_train` 测试通过，串行endpoint/fused分支各3项原始BPE及长度限制测试通过。检查包括1,500个逐轮 HF 随机差分、独立 greedy oracle、原始 Trainer wrapper、有限长度限制、跨 worker 出生阈值聚合、原子与非原子的存储布局、跨块 AA、65535/65536 ID 门槛、跨2³²地址的切片写入、权重游标和真实 suffix 增频回归。以下默认配置及稳定版核验是此前版本的记录，保留其原来的测量范围。

- `tk-train` 默认配置的 21 项库测试全部通过，其中 1,500 个随机配置逐轮比较 pair、频率、输出 ID 及最终原始 merge 列表。
- 250 组普通配置通过独立的全量重算 greedy oracle；专门测试覆盖加权重复、同频、`AA`、Unicode、超过 255 的跨度、特殊 token 冲突、空输入、前后缀和长度边界。
- 对未修改的 HF 0.23.2 进行 500 组词表与最终 merge 顺序比较，累计发生 286 次 ID 复用；完整模型 JSON 保存重载及稳定版编码结果均一致。
- 同一份词频输入下，HF 参考路径的 1/2/4 线程与逐条索引路径一致；当前并行版本另外核验逐轮 trace 和最终词表/merge 顺序。
- `tk-encode` 的 27 项 BPE 测试通过，包括新增的单字符 suffix 回归。

稳定版对照脚本检查训练模型和 HF 稳定版读写；不把它解释为整个 v1 推理栈所有配置的兼容认证。扩大随机 v1 推理检查时还遇到了部分小词表的既有 `ptr_hash` 构建失败，本轮未扩大到该加载器问题。

## 性能结果与测量范围

当前关键测量固定512 MiB真实 Wikipedia 中文语料，统一u32布局，PR串行初始化/4线程 merge，我方串行初始化/4线程 merge、4线程初始化/merge、同算法原子访问各一次。仅 MemAvailable ≤1 GiB 时停止，并记录进程 swap 与系统换页。全部算法确定后再运行完整矩阵。各项结果、实际源码与剩余成本见 [PARALLEL_REPORT.md](PARALLEL_REPORT.md)。

以下是首轮串行原型的历史记录。

结果和取舍见 [`REPORT.md`](REPORT.md)，完整逐次记录在 [`results/runs.jsonl`](results/runs.jsonl)，自动汇总在 [`results/runs.summary.md`](results/runs.summary.md)。源码、二进制、依赖锁和语料摘要在 [`results/runs.environment.json`](results/runs.environment.json)。

`REPORT.md` 和 `runs.*` 对应布局调整前的节点版本；当前端点布局的 12 次有限对比记录在 `COMPACT_REPORT.md` 和 `compact-focused.*`，不能混用两个版本的数据。

首轮历史矩阵运行 126 次正式训练，用时约 288 秒。`run.py` 后续默认四组小样本、每个实现一次，共 12 次训练；完整矩阵仅由显式选择 profile 和重复次数触发。新的原接口关键测试由 `run_native_fair.py` 执行；`run_parallel_key.py` 复现旧u16访问对照。

三个实现是当前 HF 主分支 `bbccb051` 的参考 trainer、该提交上的索引原型，以及 PR #2348 的固定 head `6ac0de53`。同一份 runner 源码和同一预处理配置分别构建。`none` 保留每行全文及换行，`whitespace_split` 使用 `split_whitespace()`；`bytelevel` 使用官方 `tokenizers 0.23.2` 的 `ByteLevel(false, true, true)` 和 `PreTokenizedString`。

`total` 从 `feed` 开始到 `train_vocab` 返回为止，包括读取、预分词、词频汇总、初始化、合并和输出；不包括进程启动、结果 JSON 与摘要计算。它是相同 feed/train API 工作负载的总耗时，不是 v1 尚未提供的 Python `Tokenizer.train()` 入口。峰值 RSS 为进程 `VmHWM`。PR阶段探针在临时源码副本插入六个 `Instant` 边界日志；新的分支内部已有阶段统计，临时副本仅追加一个统计输出。

历史小样本复用相邻项目的固定 Wikimedia Wikipedia 样本；本次大语料由 `prepare_gb_corpus.py` 从固定 revision 的前两份中文 shard 独立准备，来源清单与摘要随结果保存，本目录不提交原文。真实语料 benchmark 未设置前后缀、特殊 token 或长度限制，ID 复用次数为零；这些参数的正确性由上述差分检查覆盖，本轮不宣称其性能与普通配置相同。

## 历史实验复现

在本目录执行，先准备固定源码的独立 checkout；不会写入之前的 benchmark 项目。

```bash
cargo build --release --locked
mkdir -p .build
git worktree add --detach .build/pr-head 6ac0de5359d9e0e1ed0608422575a360ef91b908
git worktree add --detach .build/stable 88a4498ad4ea1a9487b0a9b0ff881383fd5a06a3
python3 verify_stable.py .build/stable
python3 build_profiled.py .build/pr-head
python3 run.py /path/to/tokenizers-bpe-benchmark/data/text --output results/smoke.jsonl
```

`build_profiled.py` 检查 PR checkout 的提交和清洁状态，将源码复制到 `.build/profiled-*` 后才加入计时。`run.py` 固定单线程，按固定种子交错调用，每次训练使用独立进程，逐条核对完整模型摘要；遇到差异即停止。首次保存的正式矩阵对应 `--profile representative --repeats 3`；通常无需复跑这一矩阵。


补充内存内分块摘要wave：候选7e794db1通过57项完整tests与同binary对照，13block摘要峰48→16MiB、进程峰479.94→431.57MiB；初始化+12%、全训+0.55%（各n=1）。本轮累计35次正式调用，主报告 [INITIALIZATION_MEMORY_REPORT.md](INITIALIZATION_MEMORY_REPORT.md)。外存路线仅预研，26项一手研究与条件容量模型见 [EXTERNAL_MEMORY_BPE.md](EXTERNAL_MEMORY_BPE.md)。


## 通用有界排序：实测、撤回与自适应候选（2026-10-01）

新增16次正式完整Trainer调用，本轮累计51次；当前候选029ab45b、60lib tests与完整model/工作/source gates通过。始终16B排序和8B临时记录在英文初始化回退25–30%；8B记录按用户要求撤回。最终候选按实际block pair数选择：小字典空间扫描，达到65,536项后才排序后续262,144位置tile，完整u64 key/u32 local/64位base保留。中文单block初始化两次约3.6–4.1%，四块并行约5%；英文不分配排序缓冲，未再出现前述回退。收益有限，未承诺whitespace或数十GiB表现。源码无新增字典库或FFI，生产J376保留。详情及失败版本见 [INITIALIZATION_MEMORY_REPORT.md](INITIALIZATION_MEMORY_REPORT.md)。

正式端到端已明显提速：两对DE→H的train中位32.397→25.577s，feed+train中位36.766→29.838s；后续H→J筛选也继续改善。GPT-6 Luna已完成当前512MiB flat主路径的完整PERF独立审计，按各自binary与DWARF核对旧H00216d91和当前源码。当前主成本是posting校验、邻边统计与owner提交；未发现高占比且明确可删除的重复工作，本轮停止继续优化和追加训练。该结论限于已测flat路径，通用分块与数十GiB仍按单独证据解释。详见 [当前PERF审计](CURRENT_PERF_AUDIT.md)。外存仅预研，arena通用阈值尚未选定。
