# YTTM Rust 适配基线

此分支从 original Tokenizers 的 `bbccb0513ff9afda385ca5c85c66eddb1318cfc7` 创建，替换 `tk_train::BpeTrainer` 的 feed 和训练核心。公开的 builder、`feed`、`do_train`、`train_vocab`、`Trainer::train` 以及词表/merge 输出格式沿用该提交。目标是保留 YTTM 的训练机制，同时与原始 Tokenizers 输出一致，作为后续时间和内存对照实验的独立分支。

算法来源固定为 [YouTokenToMe f4162d846057a3118222ca04a01b84297eb8a8db](https://github.com/VKCOM/YouTokenToMe/blob/f4162d846057a3118222ca04a01b84297eb8a8db/youtokentome/cpp/bpe.cpp)，MIT 许可见 [YTTM-LICENSE](YTTM-LICENSE)。代码整理自之前的 Rust worker 移植，训练路径没有调用 YTTM C++、FFI 或原始 Word trainer。

## 保留的 YTTM 机制

- 每个 worker 长期持有自己的词、游程链表、pair 位置列表及局部计数。位置列表惰性清理；规则只更新相关位置。
- 以加权输入符号数的平方根划分高、低频候选：高频扫描数组，低频使用按频率分桶的队列。同频时按 Tokenizers 的 pair 顺序选择。
- 协调器最多维持两个在途任务。**每个任务只有一条规则**；下一条规则与在途规则相互影响时等待。worker 可以在其他 worker 尚未完成第一条时执行第二条。完成结果按规则顺序进入全局候选状态。
- 单 worker 使用同一状态机直接执行。多 worker 使用 scoped threads、任务通道和完成通道。

没有 Full engine、批量规则选择、批聚合、posting 压缩、arena/block 优化或 `TK_YTTM_BATCH`、`TK_YTTM_DEPTH`、`TK_YTTM_RUST` 开关。`BpeTrainer` 默认直接运行移植核心。

## 为 Tokenizers 输出保留的适配

| 行为 | 适配 |
| --- | --- |
| 同频 pair | 按原始 Tokenizers 的 `(left_id, right_id)` 升序，而非 YTTM 默认的非确定性桶尾顺序。 |
| 自配对 | 选择分数使用重叠邻接数 `run_length - 1`；实际替换仍从左到右应用 `floor(run_length / 2)` 次。 |
| 词表与前后缀 | 保留 original 的特殊 token、alphabet 选择、初始 ID、continuing prefix、end suffix 与字符串拼接逻辑。 |
| 候选计数 | 保留上游 Word trainer 的邻居事件与被选 pair 的计数行为，而不是把候选计数全部重建为实际邻接数。局部和跨线程 delta 使用 checked `i128`，候选优先级使用 `u64`。 |
| ID 复用 | 第一次复用时先清空在途任务，从已有惰性 postings 建立各候选的词范围。之后使用上游的惰性堆刷新顺序，防止重复候选被提前一起刷新。worker 仍执行移植的链表更新；不切回原始 Trainer。 |
| ID 复用后的游程 | 此时展开游程，保持左至右的 occurrence 更新和每个候选原有的词范围。普通训练在此之前保留游程压缩。 |
| `max_token_length` | 使用逐节点的实际符号长度及上游的严格 `< max_length` 邻居过滤。此模式从一开始使用未压缩的链表。保留上游将负计数转换为 `u64` 刷新优先级的行为，以保持 merge 列表一致。 |

因此，本分支应标注为 **YTTM Rust / Tokenizers-compatible**。它适合与相同输入、参数和输出的 Trainer 比较；其耗时与 RSS 不是原版 YTTM C++ 的测量结果。ID 复用和长度限制的兼容路径会改变游程压缩与候选元数据的内存成本，应与实验参数一起记录。

上游 `limit_alphabet` 在截断频率相同的字符时本来就可能有不同选择；前后缀初始 ID 也可能受输入词表遍历顺序影响。差分测试使用同一份词频表；涉及 alphabet 截断平局时固定相同的保留 alphabet。默认无限 alphabet、无前后缀的实验可以直接对照最终词表 ID 和完整 merge 顺序。

## feed 的等效适配

`feed` 收集输入序列后按连续序列静态分块，每个 worker 累积一张词频表，最后合并到全局词频表。这对应 YTTM 的“线程局部词频 → 全局去重”阶段。它保留 Tokenizers 的 `process` 回调：normalizer、pre-tokenizer、词边界和 ByteLevel 字符映射仍由调用者决定，不额外插入 YTTM 的空白符或词首标记。

与上游每个 sequence 创建小表再 Rayon reduce 相比，此路径按 worker 持有词频表，并在 feed 时保留输入序列。YTTM C++ 按 UTF-8 字节划分整块输入；这里按完整序列数分块，以遵守回调边界。feed 的时间和输入缓冲峰值应单独测量。alphabet 过滤后不再次聚合词，沿用 Tokenizers 的输入语义。

## 验证与复现

常规检查：

```bash
cargo test --manifest-path tokenizers/tk-train/Cargo.toml
cargo test --manifest-path tokenizers/tk-train/Cargo.toml --no-default-features
cargo clippy --manifest-path tokenizers/tk-train/Cargo.toml --all-targets --no-default-features
```

`src/trainers/bpe/reference.rs` 仅在测试构建中包含 original Word trainer，作为差分 oracle；生产构建不包含它。测试比较完整词表 ID、merge 顺序和特殊 token；覆盖 1/2/4 个 worker、普通/长游程、中英文、自配对、同频、前后缀、ID 复用、长度限制、feed 词频与错误传播，以及真正的两槽并发和 worker panic 退出。

可在自己的 UTF-8 语料上运行较重的差分检查：

```bash
YTTM_REFERENCE_CORPUS=/path/to/corpus.txt \
  cargo test --manifest-path tokenizers/tk-train/Cargo.toml \
  yttm_matches_original_real_corpus -- --ignored
```

此检查使用 whitespace 预切分、目标词表 12,000，与同一份词频表的 original oracle 比较 1/2/4 worker 输出。

实验 runner：

```bash
cargo build --manifest-path tokenizers/tk-train/Cargo.toml \
  --release --example bpe_yttm_bench --no-default-features
/usr/bin/time -v tokenizers/tk-train/target/release/examples/bpe_yttm_bench \
  /path/to/corpus.txt 50000 4 /tmp/yttm-model.json
```

输出 JSON 分别记录读入、feed、train 时间、词频表大小、实际词表/merge 数和 Linux `VmHWM` 峰值 RSS。`total_seconds` 包含读取和训练，排除模型导出；整个进程的 `/usr/bin/time -v` 包括导出。输入是完整 UTF-8 文件，预切分为 `split_whitespace`，feed 后释放输入字符串。模型导出按词表字符串排序，可逐字节对比。

对照 original 时，把同一个 example 复制到 `bbccb051` 的独立 checkout，使用相同 Cargo 依赖、构建选项、语料和 worker 数。线程数通过 `tk_encode::parallelism::set_num_threads` 设置；不要把 `RAYON_NUM_THREADS` 当作该 v1 API 的完整线程配置。复现实验依赖见 `yttm-reference.lock`，构建前将其复制为 `tokenizers/tk-train/Cargo.lock`，再使用 `--locked`。

该基线的可选 `parity-aware-bpe` feature 原本仍引用已移除的 `BPE` 模型，不能编译；本分支没有修改这个独立算法。标准 BPE 实验使用默认 features 或 `--no-default-features`。

## 本分支的验证记录

Rust `1.98.1`，依赖使用随分支保存的 lockfile。默认及无默认 feature 各通过 20 项单元测试和 1 项文档测试。两组随机测试共 800 个输入，每个输入对照 1/2/4 worker；此外通过前后缀、特殊 token、长度限制和流水线专项检查。Clippy 通过，测试 oracle 保留两处上游字段写法警告。

英文、中文各约 1 MiB 的真实语料均通过 original oracle 的完整词表/有序 merge 对照（1/2/4 worker）。release runner 的 1/4 worker 导出逐字节一致，模型摘要如下。这些运行用于正确性验收，未作为性能排名。

| 语料 | 输入 SHA-256 | 导出模型 SHA-256 |
| --- | --- | --- |
| `en-1m.txt`，1,048,035 字节 | `830150d2e68195b12a2ef93dd07abae1626453668eb9c09eeb6ca946228426d6` | `07281837ea7be417ed1499e511bbba744831d9e89b615474ac3c7ebd0696a761` |
| `zh-1m.txt`，1,048,535 字节 | `06f507d1c4a4f5183d6eb03ff289588091553f1170aab92bcaee8a7860079c5b` | `681349227589a3b8e55d7842796ea08dd6ada4fbe7229a2f2bca89aa45f7b247` |
