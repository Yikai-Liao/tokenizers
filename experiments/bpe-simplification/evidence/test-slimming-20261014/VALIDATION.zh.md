# BPE 测试精简验收：1,297 行

本次按上传的较新计划精简普通 BPE 测试。格式化后的活动测试为 **1,297 行**，比 1,300 行努力目标少 3 行；完整顺序 oracle、每步 trace、1/4/8 worker 模型对照及列出的失败、溢出、Unicode、codec、feed 交错防线保留。默认与无默认特性配置实际各执行 25 个 BPE 测试：**24 通过、1 个已知失败、0 ignored**。已知失败的 `tokenizer_roundtrip` 恢复为活动测试，断言保持原样。

## 基线与计数范围

独立工作树基于最新提交 `6bbafd1db650842453162b7dcf789f07c7262196`，分支为 `refactor/bpe-test-slimming-20261014`，开始时工作树干净。上传计划的 `e4c35bee` 和 1,729 行属于旧快照，不能直接当作本次实测基线；[原计划](PLAN.zh.md) 完整保留。

沿用 `experiments/bpe-simplification/count_lines.py` 的原范围：排除注释和空行，包含属性、括号、辅助代码、活动测试钩子、完整 oracle、公共 BPE 声明、shared trainer wrapper 及 Miri harness，每处只计一次；Parity Trainer 和仅供它使用的 `word.rs` 不计入。没有迁移测试或压缩排版。只用项目 `format-changed.py` 格式化本次修改的六个 BPE Rust 文件。

| 文件或范围 | 开始时活动基线 | 恢复 roundtrip 后 | 先补高层夹具后 | 最终 |
| --- | ---: | ---: | ---: | ---: |
| `corpus.rs [cfg(test)]` | 142 | 142 | 142 | 59 |
| `index.rs [cfg(test)]` | 196 | 196 | 196 | 128 |
| `merge.rs [cfg(test)]` | 212 | 212 | 212 | 0 |
| `mod.rs [cfg(test)]` | 19 | 19 | 19 | 19 |
| `positions.rs [cfg(test)]` | 173 | 173 | 173 | 140 |
| `tests.rs`，含完整 oracle | 752 | 811 | 841 | 766 |
| `vocabulary.rs [cfg(test)]` | 157 | 157 | 157 | 142 |
| `word_counts.rs [cfg(test)]` | 27 | 27 | 27 | 27 |
| `trainers/mod.rs [cfg(test) BPE wrapper]` | 12 | 12 | 12 | 12 |
| `miri-codec/src/lib.rs [Miri harness]` | 4 | 4 | 4 | 4 |
| **合计** | **1,694** | **1,753** | **1,783** | **1,297** |

恢复 roundtrip 增加 59 行；先新增高层夹具增加 30 行；从增补状态到最终净减 486 行。因此本次实测为 `1694 + 59 + 30 - 486 = 1297`，相对于开始时的活动基线净减 397 行，相对于恢复测试后的基线净减 456 行。没有再叠加旧计划的删减估算。[四个阶段的计数 JSON](baseline-lines.json)、[恢复后](restored-lines.json)、[增补后](expanded-lines.json)、[最终](final-lines.json) 包含全部计数规则及逐文件结果。

普通 BPE 生产计数从 2,565 到 2,559；这 6 行差异来自移除测试观测后 rustfmt 把三个单表达式 closure 去掉花括号并收为一行。生产算法、历史 cohort、优先级及计数修复语义没有修改。

## 先增加高层验证，再删除内部测试

`targeted_models` 复用原 `check` 和独立顺序 oracle，对每份输入比较完整词表及 ID、有序 merges、每步 `(pair, count, merged ID)` trace，并分别使用 1/4/8 worker。

- 共享边界：`abacdbdc×2, ab×10, ac×9, db×8, dc×7`。
- 多词及权重区间：40 个不同词 `"x".repeat(i)+"ababa"`，每 10 词一组取 `[9,1,1,0]`。
- 别名和长度限制：`baaaaaba×2, ba×1`、suffix=`a`，分别使用 `None`、`Some(3)`、`Some(usize::MAX)`。
- 字母表过滤：保留原 Unicode 语料 `qé🙂z×7, é🙂é×6, 🙂é×5, 空串×4, é×3`、limit=2，并补空输入。因为 oracle 共用 alphabet selector，另独立断言 `é/🙂` 的词表 ID 为 `0/1`；原 Unicode、字符存在性与装饰 ID 的独立期望继续保留。

`overlap_and_alias` 的长 AA 输入由 8,195 扩到 20,001 个字符；完整模型和 trace 对照覆盖跨准备任务训练，不固定任务必须分成几块。高层增补阶段先在两种配置实测通过，随后才删除以下 12 个测试。

| 删除位置与测试 | 删除的原因与高层替代位置 |
| --- | --- |
| corpus：`initial_ranges` | 放弃人造布局和精确分片公式；完整训练使用实际输入，不复建旧私有状态 |
| corpus：`fresh_reader` | 放弃预取、缓存及人造 reader 状态；`targeted_models` 的 40 词和权重区间比较完整结果 |
| index：`unified_aggregation` | 放弃事件与载荷的精确排列；保留模型/trace 和 codec 合并结果对照 |
| index：`historical_cohorts` | 放弃相同优先级载荷的出堆身份；顺序 oracle 和完整 trace 保留历史 cohort 语义验证 |
| merge：`shared_boundary_once` | 放弃 prepared job 和 raw slot 的具体形态；`targeted_models` 共享边界输入比较完整结果 |
| merge：`rejected_start_and_aa_tasks` | 放弃精确任务数、首任务长度和注入无效起点；20,001 字符 AA 高层输入比较完整模型/trace |
| merge：`reuse_spans` | 放弃该私有 occurrence span 数组期望；suffix alias 和长度限制高层夹具验证输出，相邻并发写入专用测试仍保留 |
| merge：`directory_recovery` | 放弃注入 scratch 失败后复用同一目录；保留真实 prepare/commit 失败边界测试，不声称它等价覆盖目录复用 |
| merge：`reuse_scan_domain_and_bound_configuration` | 放弃人工截断索引后的私有扫描范围；高层 alias/gate 用例保留无界与 `usize::MAX` 配置 |
| positions：`leaf_format_and_exhausted_cursor` | 放弃私有字节布局、Cursor 大小及重复耗尽；保留 roundtrip、seek、restart、chunks 和截断 decoder 失败 |
| vocabulary：`decoration_row` | 放弃底层槽位 getter/setter 和大小；保留独立装饰及 Unicode ID 期望 |
| tests：`initial_edges` | 放弃私有边坐标及标准迭代器错误恢复；相同 Unicode 语料升为完整训练验证，并保留独立截断 ID 期望 |

同时清理 merge 测试模块、`select` 和失效测试导入；移除 Phase、位图、全局期望状态与生产模块的仅测试观测调用；删除 `sizeof`、SmallVec 内联/spill 布局断言。`public_behavior` 继续检查公开并行设置、feed 的 ambient pool 和回调线程、进度记录、并行使用标记及完整模型。serial feed callback 现在直接与调用者 worker 比较。feed 部分失败的两 worker 交错、成功 tail 和先前 trainer 状态不替换断言原样保留。

九个名称按计划缩短，公共子进程的 `--exact` 路径同步为 `trainers::bpe::tests::public_behavior`。完整重命名映射、删除清单及每个阶段的实际测试名字见 [test-inventory.json](test-inventory.json)；两个最终 `--list` 和实际执行名字完全一致，准确符合“基线减 12、恢复 1、新增 1”的 25 个测试集合。

## 实测结果与失败集合

| 阶段/检查 | 默认配置 | 无默认特性配置 |
| --- | --- | --- |
| 最新基线，roundtrip 仍被注释 | 35 通过 / 0 失败 / 0 ignored | 35 通过 / 0 失败 / 0 ignored |
| 恢复 roundtrip，并先增补高层用例 | 36 通过 / 1 失败 / 0 ignored | 36 通过 / 1 失败 / 0 ignored |
| 最终普通 BPE | 24 通过 / 1 失败 / 0 ignored | 24 通过 / 1 失败 / 0 ignored |
| 最终 `--list` | 25 个名字 | 同样 25 个名字 |
| shared BPE trainer wrapper，`--exact` | 1 通过 | 1 通过 |

`--exact trainers::bpe::tests::public_behavior` 另实际执行 1 个测试并通过；它内部的七个 child setting 已在两种完整 BPE 测试配置中执行。独立 codec native 实际执行 5 个保留测试，全部通过。Miri 使用仓库原锁文件实际执行同样 5 个保留测试，全部通过、退出码 0，测试汇总记录耗时 217.01 秒；[完整 Miri 日志](final-miri.log) 保留了输出和既有 harness manifest 警告。

唯一失败集合为 `trainers::bpe::tests::tokenizer_roundtrip`。原编码断言在输入 `测` 处看到 `left: [PipelineToken(3)]`，而 `right` 是当前模型 `config.vocab["测</w>"]`。增补阶段日志记录默认期望 6、无默认特性期望 4；最终两次日志的期望都是 6。期望 ID 随训练的词表分配而变，但本次已知失败始终是 plain ID 3 与该 affixed ID 不相等。本次没有调查或修复其未知根因，也没有删除断言、添加 ignored 或排除该测试。

必须区分：最新 HEAD 实际启用的基线没有失败，是因为该测试当时被整段注释。恢复后出现的失败不是精简导致的新回归；它在删测试前的增补阶段已同样复现，主线程此前也在旧基线 `d518…` 验证过。该测试是当前工作分支此前新增的验证，不是 upstream 原有测试。即使如此，最终 suite 的真实退出码仍为 101，不能报告为“全部通过”。

日志按阶段保留：[基线默认](baseline-default.log)、[基线无默认特性](baseline-nodefault.log)、[增补默认](expanded-default.log)、[增补无默认特性](expanded-nodefault.log)、[最终默认](final-default.log)、[最终无默认特性](final-nodefault.log)、[公开 exact](final-exact-public.log)、[codec native](final-codec-native.log)。

## 复现命令与格式检查

`tk-train` 不是当前 tokenizer workspace member，使用 manifest path；独立工作树缺少 ignored 的 `tk-train/Cargo.lock` 时，复制主工作树现有锁文件保持依赖一致后再使用 `--locked`。codec harness 使用仓库原锁文件。工具链与锁文件哈希另存于 [环境记录](environment.txt)。

```sh
export PATH=/root/.cargo/bin:/root/.local/bin:$PATH
export CARGO_TARGET_DIR=/dev/shm/bpe-test-slimming-target-20261014
export CARGO_PROFILE_DEV_DEBUG=0
export CARGO_PROFILE_TEST_DEBUG=0
export CARGO_INCREMENTAL=0
cargo test --locked --manifest-path tokenizers/tk-train/Cargo.toml trainers::bpe
cargo test --locked --manifest-path tokenizers/tk-train/Cargo.toml --no-default-features trainers::bpe
cargo test --locked --manifest-path tokenizers/tk-train/Cargo.toml trainers::bpe -- --list
cargo test --locked --manifest-path tokenizers/tk-train/Cargo.toml --no-default-features trainers::bpe -- --list
cargo test --locked --manifest-path tokenizers/tk-train/Cargo.toml trainers::bpe::tests::public_behavior -- --exact
cargo test --locked --manifest-path tokenizers/tk-train/Cargo.toml trainers::tests::trainer_wrapper_train_model_wrapper -- --exact
cargo test --locked --manifest-path tokenizers/tk-train/Cargo.toml --no-default-features trainers::tests::trainer_wrapper_train_model_wrapper -- --exact
cargo test --locked --manifest-path experiments/bpe-simplification/miri-codec/Cargo.toml
cargo +nightly miri test --locked --manifest-path experiments/bpe-simplification/miri-codec/Cargo.toml
python3 experiments/bpe-simplification/count_lines.py --base 6bbafd1
uv run experiments/bpe-simplification/format-changed.py --check \
  tokenizers/tk-train/src/trainers/bpe/corpus.rs \
  tokenizers/tk-train/src/trainers/bpe/index.rs \
  tokenizers/tk-train/src/trainers/bpe/merge.rs \
  tokenizers/tk-train/src/trainers/bpe/positions.rs \
  tokenizers/tk-train/src/trainers/bpe/tests.rs \
  tokenizers/tk-train/src/trainers/bpe/vocabulary.rs
git diff --check
```

修改文件的 formatter `--check` 和 `git diff --check` 均通过。[BPE.diff](BPE.diff) 是相对于最新基线的完整六个 Rust 文件与 `COVERAGE.md` diff；覆盖表已按最终真实测试更新，证据文件另行归档。

## 明确放弃的内部覆盖

高层对照没有等价替代全部旧断言。本次主动放弃精确分片公式、私有预取/cache 状态、事件载荷排列、等优先级 cohort 出堆身份、raw slot/prepared job 形态、精确任务数量/长度、注入的无效起点、私有 occurrence 数组、scratch 失败后的目录复用、人工截断后的扫描域、literal leaf bytes、重复 cursor 耗尽、decoration slot getter/setter、私有 edge 坐标与标准迭代器错误恢复、布局大小与 spill 形态。

移除 Phase 后，不再逐阶段证明 alphabet/count/owner/materialize/prepare/commit/retire 分别在哪个 worker 上执行，也不再以 phase bitmap 判断零合并时内部阶段是否运行。保留的公开设置、feed 回调线程、并行使用标记及模型验证反映可观察契约；它们不构成旧内部阶段监控的逐项替代。
