# 固定 worktree 版本的独立审查

## 范围与结论

本轮只读审查五个 worktree 的 Rust 源码、分支差分及 benchmark 构建和运行脚本，未构建、执行测试或运行 benchmark。既有批次、AA、相邻边及独占写入证明见 [REVIEW.md](REVIEW.md)；本轮复核新增权重游标、初始化线程池和原 API 路由。

在原报告的计数范围与 plain 配置条件内，没有发现新增引擎正确性问题。`weight_forward` 与原二分查权重等价；初始化专用池的工作完成后才开始 merge；五个版本通过原 `BpeTrainer` 接口选择固定引擎，普通 corpus slot 和 posting 元素均为 u32。非空 affix 保留原 HF cohort 路径。

首次审查发现一项工具清理问题：五个 worktree 的 benchmark 当时仍声明 `indexed` feature，其条件编译代码引用已私有化的实验接口；显式启用该 feature 会遇到 Rust 可见性错误。测量时的默认 native 构建不启用它，没有影响原 API 路径。此发现及线程参数建议已在末尾的清理复核中关闭；正文保留首次审查和测量版本的依据。

## 一、锁定的分支、提交与源码

五个版本均从 `8c968e10970c4265811b6935917126bfc8c9ba3b` 派生。首次审查核对时，下面五个 worktree 均干净，工作文件与 HEAD 相符。本节保留测量使用的提交与 hash；清理后的当前提交另见末节。

| worktree（位于 `/root/code/tokenizers-worktrees/`） | 分支 | HEAD |
|---|---|---|
| `parallel-count4` | `bpe/parallel-count4` | `c07a6e398b85b875f309adedd853995542f0d024` |
| `parallel-count1` | `bpe/parallel-count1` | `63e384b81786e6d3e0e865a7194acaf8e08e008f` |
| `parallel-atomic` | `bpe/parallel-atomic` | `f2c5415fa21d7c75a09a594d22cca1143e34636b` |
| `indexed-serial` | `bpe/indexed-serial` | `45fb6b95b07c33c280430cc17f45c91f5ebbc14f` |
| `fused` | `bpe/fused` | `0d5093843ee50e2eb650ec262585e583e5d5f24e` |

以下 shared 文件在五个 worktree 中内容相同：

| 文件（相对 worktree） | SHA256 |
|---|---|
| `tokenizers/tk-train/src/trainers/bpe/indexed.rs` | `c11529e6f696e176ce9a504f898d6d316565036a4db3c9b3cd8c341cbd405040` |
| `tokenizers/tk-train/src/trainers/bpe/indexed/parallel.rs` | `8f44311d8f38c91c7480c6b8ffcce191a12868ff266fff81eeb3b8df970c2f6b` |
| `tokenizers/tk-train/src/trainers/bpe/indexed/compact.rs` | `d86ba57e35a87ac6561a92c5e93304702d002e4e6eeb5c3fe26c937ce32577c8` |
| `tokenizers/tk-train/src/trainers/bpe/indexed/aa_parity.rs` | `cf5f6307513f6fe1f4aac905bc42b503099bd76d5c741c6da4e2c905504d68a3` |
| `tokenizers/tk-train/src/trainers/bpe/indexed/small_posting.rs` | `a5d8e66ca0465006c54fc5af197f2f9f2cf7c72c8decef48a1511d474b9ffb7d` |
| `tokenizers/tk-train/src/trainers/bpe/word.rs` | `b81cd9c9819c7be9a76b6556fd58f0c1d04fb00992998e224f360aa7706553bd` |
| `tokenizers/tk-encode/src/utils/parallelism.rs` | `10a4af59e51e3047c80a192c30e1c15eb954da4ad2addf4a57c417c07cadedc9` |

固定路由所在 `tokenizers/tk-train/src/trainers/bpe/mod.rs` 的 SHA256：

| 版本 | SHA256 |
|---|---|
| `parallel-count4` | `8c5ad11abcdb766ff6df1fee14c2df4bf2795c477e64b41d200497567e96c5c4` |
| `parallel-count1` | `0c5db8c92a5f01965fba080dbbc25e5c3f13547a7a32084fd15d2a2c4044a06a` |
| `parallel-atomic` | `cc3f685fe0716263167e3150cfefbf17c821d7a2158f3acff2db29b515dd1efa` |
| `indexed-serial` | `7f5518e9522a7caec09bc1a611f0befa7dc66d318354cee1429118b82d2f785d` |
| `fused` | `9fc39a837bfc31fad0f1f0e35113b333b3afbad9485af3cf8e4a5ff717c4a3f0` |

完整源码摘要覆盖各 worktree 中 194 个已跟踪的 `.rs`、`Cargo.toml`、`Cargo.lock` 文件。计算方式为按相对路径排序，依次向 SHA256 输入 `UTF-8 路径 + NUL + 文件 SHA256 的 32 个原始字节`。

| 版本 | 完整源码摘要 SHA256 |
|---|---|
| `parallel-count4` | `46bad26622ce38c8ebb2e914f08a35621299831cf24c3143c705d79413125b01` |
| `parallel-count1` | `eec17d6cc41de9ecddcc568d5fde4ed8cb65ebfe389d925393686b29ba34e5f2` |
| `parallel-atomic` | `b4cfe83e46410c6197bc93f699f8613134eb63e534a025cf1cdcd983adbf0963` |
| `indexed-serial` | `5ba15a72ab8661a1922976582cb3eb7e138e21a287b2401fbb26cfab570682c9` |
| `fused` | `0b497b0905171225c3c485aba5842fd6795131d1693c29929ae9c5fd30283edc` |

## 二、权重游标的边界与单调性

对应 `parallel.rs:187` 的 `Block::weight_forward` 和 `parallel.rs:963` 的调用。设块内 pivot 数组为递增的 `pivots`，本次位置为 `local`，目标值为：

```text
end = pivots.partition_point(|p| p <= local)
```

游标保存上一次的 `end`。同一块内的 plan 位置不下降，因此旧游标不大于本次 `end`。先向前检查至多 8 个 pivot：如果遇到大于 `local` 的 pivot，已经到达 `end`；如果检查预算用完且下一个 pivot 仍满足条件，在剩余后缀执行 `partition_point`，增加的量恰好是 `end - cursor`。所以返回值与原 `weight` 一致。

边界处理成立：

- `pivot == local` 使用当前词的权重，条件是 `<=`。
- 游标为零使用 `previous_weight`，覆盖跨块长词的前半段；空 pivot 块也返回该值。
- 游标达到数组末尾时不再索引 pivot，只读取最后一个 weight。
- 每个 output chunk 独立初始化游标；地址块改变时把游标重置为零，不能沿用上一块的 pivot 下标。
- `b = position >> bits`，对应 block 的 `base = b << bits`；`bits` 为 16 或 32，因此块内地址转 u32 没有截断。
- uniform 权重分支直接返回常数，整个训练的 uniform 选项不改变，不会留下随后需要使用的错误游标。

调用顺序满足前提：初始化 flat posting 按连续空间 chunk 的顺序追加；新 posting 按 output 的空间顺序追加，并把各局部出生链 reverse；flat 多规则计划全局排序，字典多规则在块内排序，字典块按块号 collect；AA 选择保留输入空间顺序。`plans.par_chunks` 只切分这条有序序列。因此每个游标看到的同块地址始终非降序。

已有 `forward_weight_cursor_handles_near_words_and_sparse_gaps` 测试源码覆盖等于 pivot、重复查询、邻近移动、远距离跳跃、空块、uniform 权重和超过 u32 的全局 base。本轮没有执行该测试。

## 三、初始化专用线程池与阶段屏障

对应 `parallel.rs:398` 的池建立、`:515` 后的初始化 closure、`:733` 的初始化 install，以及 merge 阶段的 `pool.install`。

`workers` 始终决定 owner 数、route 分区数量和 merge 池线程数。`initialization_workers` 只选择执行初始化 closure 的池：该 closure 包含 pair 路由、owner 计数、heap 构建及初始化统计。alphabet 和字符串/语料构建在 closure 之前，仍为单个 coordinator 工作。

当 `Some(1)` 与 merge workers 不同，建立独立一线程初始化池。closure 的可变借用只在同步 `install` 期间交给该池，返回前所有并行 collect、sum、for_each 都已完成；之后 merge 才读取 owners、blocks 和 corpus，并始终使用原 merge 池。不存在初始化与 merge 对共享容器同时读写的阶段。

`None` 使用原 merge 池；`Some(workers)` 也不额外创建池。配置入口拒绝 `workers=0` 与 `Some(0)`。

线程口径需要按阶段解释：4-worker 的 `parallel-count1` 训练会建立 4-worker merge 池和 1-worker 初始化池，后者在 merge 期间保持空闲直到训练返回。它的初始化算法结构仍有四个 owner，改变的是初始化并发度。额外池的建立计入 initialize，总训练还包含池销毁；初始化 route/count/heap 子计时可单独观察实际工作。这个版本不是把 owner 数、route 数或后续 merge 并发一起改成 1。

现有 `serial_initialization_keeps_parallel_greedy_order` 测试源码按四个 merge workers、一个初始化 worker 对照串行 greedy trace、vocab 和 merges。本轮只复核了测试内容。

## 四、原 API 路由与版本公平性

`train_vocab()` 调用公开 `do_train()`，`Trainer::train()` 同样调用它并按原配置构造模型。五个版本的 `do_train` 都返回原 `(Vocab, Merges, Vec<AddedToken>)`，没有新增参数或公开实验结果类型。`IndexedParallelConfig`、`IndexedTraining`、`IndexedTrainingStats` 及实验方法均为 `pub(super)`，原 `pub use` 已取消。

与基础 snapshot 的逐文件差分未修改 `BpeTrainer` 的公开字段、builder 参数、默认值或 serde derive/字段属性，也没有把引擎选择、初始化线程数或原子开关加入 trainer 的序列化表示。模型仍接收同一 vocab/merges tuple 与 `model_options()` 的 affix 配置。因而这次版本选择没有改变 trainer 的公开字段或 serde schema；私有 stats 的序列化仅供构建副本中的 benchmark 探针使用。

| 版本 | 原 API 固定路由 | corpus / posting | 初始化 / merge |
|---|---|---|---|
| `parallel-count4` | parallel | u32 / u32 | 同一 worker 设置 |
| `parallel-count1` | parallel | u32 / u32 | 1 / 原 worker 设置 |
| `parallel-atomic` | parallel | AtomicU32 / u32 | 同一 worker 设置 |
| `indexed-serial` | compact Endpoints | u32 / u32 | 串行 / 串行 |
| `fused` | 按 piece 几何选择 WordArena 或 Endpoints | u32 / u32 | 串行 / 串行 |

三个 parallel 路由显式使用 `posting_block_bits=32`、`narrow_corpus=false`、`batch_size=256`。u16 slot/posting 实现仅通过私有配置供历史测试使用，原 API 不会选择它。32-bit posting 在小于地址范围的语料中走 flat 路径；更大语料仍以 u32 块内地址和目录表示，全局计划地址为 usize。

差分已核实：`parallel-count1` 与 count4 的训练源码唯一差异为 `initialization_workers: Some(1)`；atomic 与 count4 的训练源码唯一差异为 `atomic_corpus: true`。五个版本的 indexed、parallel、compact、AA 和 posting 核心文件相同，indexed-serial/fused 只改变公开入口选择的引擎。

parallel 的 workers 来自 `get_parallelism()` 和 `num_threads()`：禁用并行时使用 1，否则使用 tk_encode 的线程设置。该设置支持 `set_num_threads`；这里的 `num_threads` 自身不读取 `RAYON_NUM_THREADS`。feed 仍使用原 `maybe_par_bridge` 路径。

统一的是 corpus slot 和 posting 元素宽度。parallel 的长度表仍为 usize，serial compact 的长度表为 u32；Plan 仍含两个 usize。这些属于现存布局差异，应继续按照 [MEMORY_LAYOUT.md](MEMORY_LAYOUT.md) 的字段公式说明，不能据 u32 路由声称所有结构宽度都相同。

所有固定引擎在非空 prefix/suffix 时进入相同 generic cohort 路径。有限 `max_token_length` 在 plain 路径继续使用严格 `<` 的新邻边门控，初始化 pair 与已选 merge 的例外未改变。

## 五、benchmark 工具的实际口径

`build_native_fair.py` 从干净提交复制训练源码，仅在原 tuple 返回之前注入一条统计输出。生成 runner 明确执行：

```text
set_num_threads(4)
set_parallelism(false)
feed(...)
set_parallelism(true)
train_vocab()
```

因此当前实验是串行 feed、按固定版本选择的训练引擎。三个 parallel 版本的 merge 并发为 4；count1 初始化并发为 1。外层环境中的 `TOKENIZERS_PARALLELISM=false` 被训练前的 override 覆盖，不会使此次 merge 意外退回单线程。

运行脚本的 `--merge-workers` 当前不改变生成 runner 内固定的 4。新增统计校验会核对实际 `workers`、`initialization_workers`、atomic 开关与 `parallel_u32_flat32` 布局，因此传入不匹配参数会失败，不能静默记录为其它线程数。建议把该固定实验参数限制为 4，或者将线程设置真正传入 runner。当前脚本的布局校验只接受三个 parallel 版本；未来测 indexed-serial/fused 时需使用它们真实的串行统计口径。

新增 provenance 同时记录 worktree 提交与源码、注入探针后的实际构建源码、生成 runner、脚本及 binary 的 hash，能够区分版本源文件和实际编译文件。原 API route、u32 宽度和线程口径来自编译源码及统计校验；benchmark 结果及其性能解释不属于本报告。

### 遗留 indexed feature

在本报告第一节锁定的测量提交中，五个 worktree 的 `benchmarks/hf-bpe/Cargo.toml` 已改成 `default=[]`，但仍声明 `indexed=[]`。`src/main.rs:12` 和 `:69` 后的 feature 分支仍导入 `tk_train::IndexedParallelConfig` 并调用私有实验方法。因此显式 `--features indexed` 会失败。当时建议删除固定版本 benchmark 中这项 feature 和对应条件分支，或把旧实验 runner 保留在历史 snapshot。测量构建未启用这个 feature，不影响已经审查的默认调用链。

## 修改后需要复核的依赖

- plan 或 posting 取消空间排序：重新检查 weight cursor 的单调前提，以及 AA 和独占写入依赖。
- 初始化 pool 改成异步启动、多个 closure 或动态 owner 数：重新检查借用结束屏障与实验唯一变量。
- 修改公开路由、默认配置或 benchmark feature：重新检查原 API、u32 选型、fallback 和线程 override。
- 更新任何锁定文件：更新源码 hash；既有报告不能覆盖不同内容。

## 六、清理复核已关闭

以下为首次审查后的独立只读复核。本轮没有修改 Rust、构建、执行测试或运行 benchmark。五个 worktree 均干净，当前提交为：

| worktree | 清理后的 HEAD |
|---|---|
| `parallel-count4` | `b6a28768feb4af4181f76fc1fb3f78993644f5c9` |
| `parallel-count1` | `4714afd3dec42920a9828b2e61111c4e07f91269` |
| `parallel-atomic` | `65059c4968784657c6426b105eb4f91785b4acf7` |
| `indexed-serial` | `af6aff33aa8e7ab193035463114385f734344a38` |
| `fused` | `7c0e29e5679884e58d80c15f99e08df807981f10` |

逐个对照第一节的旧提交，改动只涉及 `BPE_VARIANT.md`、`benchmarks/hf-bpe/Cargo.toml` 和 `benchmarks/hf-bpe/src/main.rs`；`tokenizers/tk-train` 的 diff 全部为空。训练路由及 indexed、parallel、compact、AA、posting 源码没有变化，第一节的训练文件 hash 和正确性推导继续适用。

### 私有接口 feature 问题已关闭

五个本地 benchmark 均删除 `[features]` 中的 legacy `indexed` 声明，删除对应 import 及所有调用私有实验接口的条件分支。唯一训练分支通过 `trainer.train_vocab()` 进入原 API。旧实验入口留在历史 snapshot，当前固定版本不再声明或使用它们。

清理后的两个 benchmark 文件在五个版本中完全一致：

| 文件 | SHA256 |
|---|---|
| `benchmarks/hf-bpe/Cargo.toml` | `c687052c0fcb1dcef89769e1f44f80e14050bf044931090ab06e8414a7776a8c` |
| `benchmarks/hf-bpe/src/main.rs` | `52773c510e7cf3afec1121cdcfa8169dc8426cea7c214b4a8684a288f95e33d7` |

`BPE_VARIANT.md` 同步明确本地 benchmark 仅使用原 API；serial 和 fused 的说明也明确初始化、合并均为串行。主任务报告 count4 新 runner 的 `cargo check --all-features` 已通过；本次独立复核没有重跑该命令，关闭结论来自已核对的 manifest 和调用代码删除。

### 固定线程参数建议已关闭

中心仓库 `run_native_fair.py` 的 `--merge-workers` 已限制为 `choices=[4]`，与生成 runner 内固定 `set_num_threads(4)` 一致；不匹配值在 argparse 阶段即被拒绝。初始化 worker、atomic 开关与实际布局的运行后统计校验继续保留。

旧测量仍对应第一节的旧 worktree 提交和当时的构建副本、binary、脚本 hash；中心测量脚本版本可从 `1396274f` 回查。清理提交只用于关闭工具可见性与参数说明问题，没有改写旧结果或把旧测量标成当前 HEAD。
