# HF BPE 兼容适配原型

当前实现包含串行 [`indexed/compact.rs`](../../tokenizers/tk-train/src/trainers/bpe/indexed/compact.rs) 与并行 [`indexed/parallel.rs`](../../tokenizers/tk-train/src/trainers/bpe/indexed/parallel.rs)，共同入口位于 [`indexed.rs`](../../tokenizers/tk-train/src/trainers/bpe/indexed.rs)。并行入口 `BpeTrainer::train_vocab_indexed_parallel(IndexedParallelConfig)` 已迁入唯一 pair owner、精确规则批次、8 字节出生链、SmallPosting 和持久 Rayon pool。默认 HF trainer 保留为差分参考；`train_vocab_indexed()` 提供紧凑串行核心，`train_vocab_fused()` 保留融合试验。

开发清单见 [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)，两轮独立审查见 [REVIEW.md](REVIEW.md)，关键并行测量见 [PARALLEL_REPORT.md](PARALLEL_REPORT.md)。指定初始化时点的源码内存核算见 [MEMORY_LAYOUT.md](MEMORY_LAYOUT.md)。[COMPACT_REPORT.md](COMPACT_REPORT.md) 和 [REPORT.md](REPORT.md) 保存之前版本的数据，不能套用到当前并行实现。

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

```rust
use tk_train::{BpeTrainer, Trainer};
use tk_encode::models::bpe::{BpeConfig, PipelineBPE};

let mut trainer = BpeTrainer::builder()
    .show_progress(false)
    .vocab_size(8000)
    .min_frequency(2)
    .continuing_subword_prefix("##".into())
    .end_of_word_suffix("</w>".into())
    .build();
trainer.feed(["hello world", "hello"].into_iter(), |line| {
    Ok(line.split_whitespace().map(str::to_owned).collect())
})?;
let trained = trainer.train_vocab_indexed()?;
let model = PipelineBPE::from_config(BpeConfig {
    vocab: trained.vocab,
    merges: trained.merges,
    continuing_subword_prefix: trainer.continuing_subword_prefix.clone(),
    end_of_word_suffix: trainer.end_of_word_suffix.clone(),
    ..Default::default()
})?;
// trained.special_tokens 仍需由调用者加入 tokenizer。
```

普通字符配置可用以下并行调用。非空 affix 仍由入口转入既有串行 cohort 引擎。

```rust
use tk_train::IndexedParallelConfig;
let trained = trainer.train_vocab_indexed_parallel(IndexedParallelConfig {
    workers: 4,
    posting_block_bits: 32,
    narrow_corpus: true,
    ..Default::default()
})?;
```

训练记录可能包含重复 pair；HF 模型构造器保留同一个 pair 的最后一次 rank。`train_vocab_indexed()` 与当前 `train_vocab()` 都返回完整、有序的原始 merge 列表；对照 HF 0.23.2 最终 JSON 时按模型构造器的规则去重。

本轮还修复了 v1 编码器的一个前后缀问题：单字符词表项在存在 affix 时也需要经过编码证明，不能直接认为其自身 ID 就是编码结果。否则输入 `b` 会命中裸 `b`，跳过应输出的 `b</w>`。修正在 [`model.rs`](../../tokenizers/tk-encode/src/models/bpe/model.rs)，回归测试覆盖 ASCII、中文、prefix 与 suffix 同时设置、空 token 和重复缓存调用。

## 已完成的正确性检查

当前并行改动后，`cargo test --manifest-path tokenizers/tk-train/Cargo.toml --no-default-features --lib indexed` 的 26 项索引测试通过，包括 1,500 个逐轮 HF 随机差分、独立 greedy oracle、跨 worker 出生阈值聚合、四种地址/ID 存储布局的原子与非原子对照、跨块 AA、65535/65536 ID 门槛、真实 suffix 增频回归和 SmallPosting 所有权检查。以下默认配置及稳定版核验是此前版本的记录，保留其原来的测量范围。

- `tk-train` 默认配置的 21 项库测试全部通过，其中 1,500 个随机配置逐轮比较 pair、频率、输出 ID 及最终原始 merge 列表。
- 250 组普通配置通过独立的全量重算 greedy oracle；专门测试覆盖加权重复、同频、`AA`、Unicode、超过 255 的跨度、特殊 token 冲突、空输入、前后缀和长度边界。
- 对未修改的 HF 0.23.2 进行 500 组词表与最终 merge 顺序比较，累计发生 286 次 ID 复用；完整模型 JSON 保存重载及稳定版编码结果均一致。
- 同一份词频输入下，HF 参考路径的 1/2/4 线程与逐条索引路径一致；当前并行版本另外核验逐轮 trace 和最终词表/merge 顺序。
- `tk-encode` 的 27 项 BPE 测试通过，包括新增的单字符 suffix 回归。

稳定版对照脚本检查训练模型和 HF 稳定版读写；不把它解释为整个 v1 推理栈所有配置的兼容认证。扩大随机 v1 推理检查时还遇到了部分小词表的既有 `ptr_hash` 构建失败，本轮未扩大到该加载器问题。

## 性能结果与测量范围

当前只做关键测量：固定 Wikipedia 真实语料、同一 u16 语料/u32 posting 布局、非原子 1/4 worker 与原子 4 worker，各一次。可用内存须保持大于 1 GiB，并检查训练进程 VmSwap；系统既有 swap 存量和后台换页另记。全部算法定下后再运行完整矩阵。具体耗时、初始化容量与剩余成本见 [PARALLEL_REPORT.md](PARALLEL_REPORT.md)。

以下是首轮串行原型的历史记录。

结果和取舍见 [`REPORT.md`](REPORT.md)，完整逐次记录在 [`results/runs.jsonl`](results/runs.jsonl)，自动汇总在 [`results/runs.summary.md`](results/runs.summary.md)。源码、二进制、依赖锁和语料摘要在 [`results/runs.environment.json`](results/runs.environment.json)。

`REPORT.md` 和 `runs.*` 对应布局调整前的节点版本；当前端点布局的 12 次有限对比记录在 `COMPACT_REPORT.md` 和 `compact-focused.*`，不能混用两个版本的数据。

首轮历史矩阵运行 126 次正式训练，用时约 288 秒。`run.py` 后续默认四组小样本、每个实现一次，共 12 次训练；完整矩阵仅由显式选择 profile 和重复次数触发。当前关键测试由独立的 `run_parallel_key.py` 执行，固定三次调用。

三个实现是当前 HF 主分支 `bbccb051` 的参考 trainer、该提交上的索引原型，以及 PR #2348 的固定 head `6ac0de53`。同一份 runner 源码和同一预处理配置分别构建。`none` 保留每行全文及换行，`whitespace_split` 使用 `split_whitespace()`；`bytelevel` 使用官方 `tokenizers 0.23.2` 的 `ByteLevel(false, true, true)` 和 `PreTokenizedString`。

`total` 从 `feed` 开始到 `train_vocab` 返回为止，包括读取、预分词、词频汇总、初始化、合并和输出；不包括进程启动、结果 JSON 与摘要计算。它是相同 feed/train API 工作负载的总耗时，不是 v1 尚未提供的 Python `Tokenizer.train()` 入口。峰值 RSS 为进程 `VmHWM`。阶段探针仅在临时源码副本中插入六个 `Instant` 边界日志，不更改算法。

语料复用相邻 `tokenizers-bpe-benchmark` 项目的固定 Wikimedia Wikipedia 样本，其下载脚本和许可说明见该项目 README；本目录不再分发原文。真实语料 benchmark 未设置前后缀、特殊 token 或长度限制，ID 复用次数为零；这些参数的正确性由上述差分检查覆盖，本轮不宣称其性能与普通配置相同。

## 复现

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
