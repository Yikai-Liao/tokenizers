# BPE 固定长度存储 Vec 转 Box 切片实验计划

仅制定实验，不执行代码修改或 benchmark。目标是让固定长度的所有权更明确，并检查能否去掉多余 capacity；不预设内存、速度或代码行数一定改善。

## 基线与隔离

- 仓库：`Yikai-Liao/tokenizers`，分支：`simplify/bpe-flat-20261010`；只涉及 ordinary BPE trainer。
- 2026-10-11 11:29 UTC 只读核验的远端 HEAD 为 [8f078a8][snapshot]。以下字段与生命周期均按该快照核对。
- 该快照的 `CorpusPlan::build` 已把 measured counts 原地变成 starts。保留这一 B1 行为，不能把它的收益算给 Box。
- 实际开跑时，以执行环境当时确认的当前代码为基线 A，记录完整 SHA、未提交差异及二进制 hash；不强制退回本文快照，也不等待其他优化完成。
- 冻结 A 后复制得到 B，只做本文转换；A/B 的 B1、Index、feed、codec、工具链和 feature 必须一致。中途不 rebase。
- 开始前重查表中字段及其所有写入点；已经是 Box 的跳过，已改变生命周期的重新判断，不为凑候选恢复旧代码。
- `WordCounts::Entries` 连续 entries 方案是独立实验，本实验不改其表示、排序、构建或释放流程；不做两项优化交叉矩阵。

## 候选与冻结边界

表中的“固定”只表示长度固定，元素仍可修改。字段均为当前快照的 Vec；类型改为对应的 `Box<[T]>`，构建期仍可使用 Vec。

| 字段 | 当前生命周期证据 | 转换边界与注意事项 |
| --- | --- | --- |
| `CorpusPlan.words` | collect 后按 weight 排序，此后仅供 counting/materialize 读取 | 排序完成后冻结；只拥有借用条目，不复制文本。[源码][plan] |
| `CorpusPlan.starts` → `Corpus.starts` | counts 原地改写为 starts 后不再增减；reuse 时转移到 Corpus，fresh 时丢弃 | 完成 checked 坐标计算后冻结；直接转移 Box，fresh 使用空切片，不增加 clone。[源码][plan] |
| `CorpusPlan.weight_regions` → `Corpus.weight_regions` | build 中 push，完成后只读并转移 | 最后一次 push 后冻结；这是可能留有明显 spare capacity 的候选。[源码][plan] |
| `Slots::Narrow` / `Slots::Wide` | 一次按 corpus 长度创建 `AtomicU16` / `AtomicU32` 数组；之后仅 load/store | 构造完成后冻结；保留宽度选择、哨兵、ordering 与访问协议。[源码][slots] |
| `Corpus.occurrence_spans` | `Option<Vec<AtomicUsize>>`，alias 几何分歧时按 corpus 长度懒创建；随后仅原子写入 | 保留懒创建与现有填充顺序，转为 `Option<Box<[AtomicUsize]>>`，不提前分配。[源码][occurrences] |
| `InitialTokenIds.characters` | Unicode 域长度固定为 `0x110000`，初始化写元素，随后查询 | 固定长度构造完成后即可冻结；保留初始化写入。[源码][ids] |
| `InitialTokenIds.decorated` | 有 affix 时 resize 一次，然后只更新已有槽位；无 affix 时为空 | 保留该条件和最终长度，用局部 Vec 完成 resize 后冻结；不改 ID 分配顺序。[源码][ids] |
| `SelectedRules.heads` / `tails` | 每个 batch 按 `id_count` 创建，填充后读取，不扩容 | 构造完成后冻结；这是逐批临时表，不是训练全程常驻。[源码][selected] |
| `PairIndex.shards` | build 的 unzip 产生固定 owner 数；后续仅索引、并行遍历及修改 shard 内容 | 只改外层容器；不重新设计已优化的 Index，不改 owner 路由或内层缓冲。[源码][index] |

冻结前记录 `len`、`capacity`、`size_of::<T>()`，区分潜在 payload 节省 `(capacity-len)*size_of::<T>()` 与描述符变化。
这些记录放在诊断构建中，不向正式计时路径添加日志或原子计数。

## 明确排除

- `Corpus.spans` 会随 merge `push`；`InitialTokenIds.spans` 会转移给它，保留 Vec。
- `Directories.indices/touched` 以及执行时可能出现的替代 scratch，依赖跨任务扩容、清空和 capacity 复用，保留现状。
- shard 内的 actions/positions 等 pending buffers、堆、map、事件输出及其他动态集合不改；不把本任务扩成“全部 Vec 改 Box”。
- `Positions` 已有 inline 0/1/2 与压缩 Box 存储，不重新做一次；不改压缩格式、arena、全局 allocator 或释放策略。
- 不新增 unsafe、nightly API、类型状态框架或公共 API 变化；不改变并行 join、借用、原子写入和异常退出语义。

## 收益假设及风险

- 常见 64 位目标上 Vec 描述符为 3 个机器字，Box 切片为 2 个，通常仅差 8 B；Box 切片仍是带长度的胖指针，包含它的 enum/struct 大小不一定缩小。
- 元素数组不会减半。大 Slots 数组若原本 `len == capacity`，payload 节省可能为零；不能用数组很大推导收益很大。
- `into_boxed_slice()` 是稳定 Rust API，会丢弃多余 capacity；存在 spare capacity 时可能触发 allocator shrink、重分配或搬移，需把转换成本算进去。[Rust 文档][rust-vec]
- 即使逻辑分配量变小，allocator 保留页也可能使 RSS 不降；Box 仍需正常 drop/free，不会消除每个缓冲区的释放成本。
- Box 不意味着元素不可变：切片可变借用和原子内部可变性仍可使用，不得借此删除原有同步保障。

## 实现与正确性门槛

1. 先核对执行时的 `AGENTS.md`；只改必要字段声明、构造收尾和相应类型适配，保留读写算法及释放时机。
2. 以 A/B 比较完整 vocabulary IDs、ordered merges、special tokens、序列化结果和已有 per-rule trace；使用仓库现有 oracle。
3. 覆盖空输入、目标词表 0、无 merge、零权重、Unicode、prefix/suffix、窄/宽 Slots、reuse restart 与 alias occurrence-span materialization。
4. 保留 1/4/8 worker 的相关并行正确性用例，以及错误返回、panic/unwind、joined phases 和原子访问保障；不删测试来适配改动。
5. 从仓库根目录执行现有检查：`cargo test --manifest-path tokenizers/tk-train/Cargo.toml --lib`，以及加 `--no-default-features` 的同套检查。
6. 两种 feature 设置都跑 `cargo clippy --manifest-path tokenizers/tk-train/Cargo.toml --all-targets -- -D warnings`；按仓库规则只格式化本次修改的 Rust 文件。
7. 已知 encoder-roundtrip 失败必须在 A/B 分别记录，不能写成全绿；它与 resident-coordinate assertion/panic 是两类问题，不能互相解释或统一豁免。[测试范围][coverage]
8. 新失败、模型不一致、越界或 unwind 行为变化先阻断性能结论；保留输入、日志、配置与失败点，修复后重跑相关检查。

## 最小可判定的性能实验

- 主比较只有 A 当前基线与 B 全部经复核安全的 Box 候选；先回答这个组合是否值得保留。
- 复用现有 256 MiB 中文/英文语料、ByteLevel/Whitespace、1 worker/既有固定多 worker 4，共 8 个场景；同场景语料、参数和预处理入口一致。
- 每场景每臂先做 1 次不计入结果的 warmup，再做两次正式配对，顺序 `A→B`、`B→A`：共 16 次 warmup、32 次正式运行。
- 不强制五次重复；仅当两次方向冲突、噪声接近差异或结果影响取舍时，在受影响场景追加第 3 对并解释原因。
- 使用现有 full-pipeline runner；记录源码 hash、二进制 hash、输入 hash、Rust/feature、allocator、CPU affinity、worker 数和完整命令。[现有 runner][runner]
- 两臂使用相同 release 设置、缓存策略和 CPU 集合；不与另一个 entries 实验、编译、测试或其他 benchmark 同时争用机器。
- entries 实验的 A 仅在二进制、feature、输入、affinity、测量边界完全一致且时间环境可比时复用；否则重跑相邻 A/B，不能只借用旧 baseline 数字。
- 编译在计时外完成；每次独立进程，保留逐次原始结果、执行顺序和异常标记，不只给一个汇总平均数。

## 测量与归因

- 主指标：full-pipeline wall、CPU、进程 HWM。保留完整 feed→train→模型输出的统一边界，并在后续验证/额外 encode 前采集；验证耗时不得混入。
- 主结论使用无插桩正式运行；辅助阶段时间关注 build/freeze、materialize、训练完成及相关释放点。
- 诊断构建在这些边界采集 RSS/HWM 和候选 allocation/reallocation、容量数据；HWM 是累计峰值，不能把相邻 HWM 之差当阶段分配量。
- 发生 alias materialization 的小型正确性 fixture 可验证懒分配路径；不要假称普通真实语料一定覆盖该路径或测出其收益。
- 只有主组合出现可重复收益或回归且需解释时，才按生命周期拆成“常驻/转移存储”与“临时表”两组做定向复测；不展开逐字段组合矩阵。
- 常驻组含转移给 Corpus 的 starts/weight_regions、Slots、occurrence_spans 和 shards；临时组含 Plan.words、InitialTokenIds 表及逐批 SelectedRules，分别标注实际寿命。
- 报告每场景 A/B 原值、配对百分比、两次范围和追加原因；descriptor、payload、RSS、HWM 分开表述，不相互替代。

## 保留与回退

- 正确性全部满足且无可重复的实质性时间或峰值回归，才能考虑保留；实验前沿用现有噪声/回归判据，不在看到结果后挑阈值。
- 若只改善所有权表达、性能在噪声内，明确写“语义清晰度改善，未测得资源收益”，再按改动复杂度决定是否值得保留。
- 若 shrink/搬移成本、逐批转换开销或复杂度超过收益，回退相应逻辑组或整项；允许实验结论为不采用。
- 实验完成后交付单独 diff、复核后的候选清单、A/B manifest、正确性记录、原始 benchmark、简短结论；不要自动推送、合并或覆盖其他进行中的工作。

[snapshot]: https://github.com/Yikai-Liao/tokenizers/commit/8f078a8ca11a62bb6f81fe99a73f028f9cc4a5d2
[plan]: https://github.com/Yikai-Liao/tokenizers/blob/8f078a8ca11a62bb6f81fe99a73f028f9cc4a5d2/tokenizers/tk-train/src/trainers/bpe/corpus.rs#L41-L198
[slots]: https://github.com/Yikai-Liao/tokenizers/blob/8f078a8ca11a62bb6f81fe99a73f028f9cc4a5d2/tokenizers/tk-train/src/trainers/bpe/corpus.rs#L389-L445
[occurrences]: https://github.com/Yikai-Liao/tokenizers/blob/8f078a8ca11a62bb6f81fe99a73f028f9cc4a5d2/tokenizers/tk-train/src/trainers/bpe/corpus.rs#L335-L385
[ids]: https://github.com/Yikai-Liao/tokenizers/blob/8f078a8ca11a62bb6f81fe99a73f028f9cc4a5d2/tokenizers/tk-train/src/trainers/bpe/vocabulary.rs#L125-L176
[selected]: https://github.com/Yikai-Liao/tokenizers/blob/8f078a8ca11a62bb6f81fe99a73f028f9cc4a5d2/tokenizers/tk-train/src/trainers/bpe/merge.rs#L383-L423
[index]: https://github.com/Yikai-Liao/tokenizers/blob/8f078a8ca11a62bb6f81fe99a73f028f9cc4a5d2/tokenizers/tk-train/src/trainers/bpe/index.rs#L65-L140
[coverage]: https://github.com/Yikai-Liao/tokenizers/blob/8f078a8ca11a62bb6f81fe99a73f028f9cc4a5d2/tokenizers/tk-train/src/trainers/bpe/COVERAGE.md
[runner]: https://github.com/Yikai-Liao/tokenizers/blob/8f078a8ca11a62bb6f81fe99a73f028f9cc4a5d2/experiments/bpe-simplification/run_pairs.py
[rust-vec]: https://doc.rust-lang.org/std/vec/struct.Vec.html#method.into_boxed_slice
