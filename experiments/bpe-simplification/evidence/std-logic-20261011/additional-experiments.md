# BPE 剩余标准库与现有依赖替换实验

建议只新增 **3 个独立实验**：字符位图并行归约、restart seed 定长切分、初始语料分段的边界查找。另附 1 个已归属方向的可选调用点增量，不作为第四个独立任务。三项可独立并行实现；B 是小型可读性清理，C 尤其需要实测决定去留。

范围：`Yikai-Liao/tokenizers` 的普通 BPE trainer。源码核对截至 [`a2f1c0f7fda871ad321579a8df690a5f6464043f`](https://github.com/Yikai-Liao/tokenizers/commit/a2f1c0f7fda871ad321579a8df690a5f6464043f)。以下是待执行方案，代码片段尚未编译，未修改或运行仓库。

已排除其他 agent 已发现、已认领或已完成的方向：`Iterator::find` / `Peekable`、`symbols()` 迭代器本体、`VecDeque` prefetch、Index / `BTreeMap` / index-all、parity word 的 `char_indices` / `splice`，以及 Positions `lower_bound` 的 `partition_point` 和 Range 改造。也不新增容器体系、编码格式、varint 抽象或只有语法收益的 FeedEntry 改写。

## A 用 Rayon 归约替换位图收集与串行合并

位置：`vocabulary.rs::observed_characters`。现状先收集 `Vec<FixedBitSet>`，再新建一个空位图，逐个执行 `union_with`。只替换这段收尾；保留输入 `words`、现有 chunk 计算和每个 chunk 的字符扫描。

```rust
fn observed_characters(word_counts: WordCountsView<'_>, workers: usize) -> FixedBitSet {
    let words: Vec<_> = word_counts.keys().collect();
    let chunk = words.len().div_ceil(workers).max(1);
    words
        .par_chunks(chunk)
        .map(|chunk| {
            let mut present = FixedBitSet::with_capacity(0x110000);
            for word in chunk {
                for character in word.chars() {
                    present.insert(character as usize);
                }
            }
            present
        })
        .reduce_with(|mut left, right| {
            left.union_with(&right);
            left
        })
        .unwrap_or_else(|| FixedBitSet::with_capacity(0x110000))
}
```

**实际删掉的负担**：保存全部分块结果的显式 `Vec`、收集完成后的串行合并循环，以及非空输入时额外创建的最终空位图。保留每块位图，不能宣称消除了所有中间分配。`reduce_with` 对空输入返回 `None`，因此补上相同位长度的空结果；位集合并的结合性适合归约。[Rayon API](https://docs.rs/rayon/latest/rayon/iter/trait.ParallelIterator.html#method.reduce_with)

**不变量与边界**：
- 所有输入和输出位图均保持 `0x110000` 位，空输入也如此；不能用零长度默认值替代。
- 保留 `workers >= 1` 的现有调用前提，不同时调整线程或分块策略。
- 不动上层 special token 优先插入、initial alphabet 补入和按 codepoint 分配 ID 的顺序。
- 不动上层 `observed.clone()`：仅由 initial alphabet 补入、从未观察到的字符，其初始激活 span 必须仍为 0。

**验证**：与旧实现比较完整位图和训练输出；覆盖空语料、单词、重复字符、ASCII、中文、非 BMP 字符，以及语料未出现的 initial alphabet 字符。对不同 worker 数量检查相同 vocab ID、initial spans 和 merges。

**成本与去留**：测 unlimited-alphabet 初始化耗时、整体训练耗时和峰值 RSS，覆盖小语料和大语料。归约树可能并行合并，也可能增加调度成本；峰值内存受同时存活的块数影响。只确定显式收集阶段被删掉，不预先承诺更快或更省峰值内存。限定字母表路径应保持不变。

## B 用定长切分读取 restart seed

位置：`positions.rs::Cursor::next` 的 restart 分支。当前将 `bytes[..8]` 转为数组后解码，再手动取 `bytes[8..]`。只替换该分支内部：

```rust
let (seed, rest) = self
    .bytes
    .split_first_chunk::<8>()
    .expect("internal positions restart seed is complete");
self.position = u64::from_le_bytes(*seed);
self.bytes = rest;
```

**实际删掉的负担**：两处独立编码的 8 字节切片边界和切片到数组的转换。固定宽度、返回数组引用、返回剩余输入由一个标准 API 表达；没有新增类型、缓冲或分配。[标准库 API](https://doc.rust-lang.org/std/primitive.slice.html#method.split_first_chunk)

**不变量与边界**：
- 保留 `index == end` 的结束判断，以及 `index`、`position` 的原有更新时机。
- seed 仍是 8 字节 little-endian；delta 解码、restart 间隔和内部编码格式均不动。
- 不足 8 字节仍然 panic，错误信息可以改变。不能改成 `?` 或返回 `None`，否则会把内部数据损坏静默当成 EOF。
- 这是内部编码器生成的数据路径，不借此引入外部数据校验或新的错误协议。

**验证**：空序列、单元素、刚好跨越 restart 边界、多个 restart、重复位置、较大合法位置；比较完整解码结果和迭代终止行为。另在模块测试中构造不足 8 字节的 seed，确认仍 panic；`index == end` 时则不读取字节。

**成本与去留**：定位为低风险清理。原写法也可能被编译器优化得很好，不把它当作可观的性能来源。用已有 Positions 解码基准检查无明显回退即可，不需要为这几行新增通用 codec 框架。沿用仓库工具链，不为此实验升级依赖或工具链。

## C 用 partition_point 查找初始分段末端

位置：`corpus.rs::CorpusPlan::initial_ranges`。这是未归属的语料分段调用点，与已经完成的 Positions `lower_bound` 替换无关。现状内层逐词试探下一边界；可把该搜索交给 `starts` 切片上的 `partition_point`。

```rust
pub(super) fn initial_ranges(&self, slots: usize) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::new();
    let mut begin = 0;
    let n = self.word_count();
    while begin < n {
        let start = self.starts[begin];
        let end = if self.length - start <= slots {
            n
        } else {
            let fitted = self.starts[begin + 1..]
                .partition_point(|&next| next - start <= slots);
            begin + fitted.max(1)
        };
        ranges.push(begin..end);
        begin = end;
    }
    ranges
}
```

**实际删掉的负担**：调用点手写的线性搜索循环，以及搜索中反复表达的 `get(end + 1).unwrap_or(length)`。保留外层分段推进和输出 `Vec<Range<usize>>`。`starts` 递增，使谓词满足先真后假的分区前提。[标准库 API](https://doc.rust-lang.org/std/primitive.slice.html#method.partition_point)

**必须一起保留的语义**：
- `starts` 不包含最终哨兵，故先单独判断 `length - start <= slots`；不能只对 `starts` 搜索后直接收尾。
- 每个词的占用包含末尾 separator，不能用字符数或词数替代 resident slots。
- `fitted.max(1)` 保留“单个超预算词也独占一段”的规则，并保证预算为 0 时仍能前进。
- 空语料不访问 `starts[0]`；不采用 `start + slots`，避免引入新的加法溢出边界。

**验证**：把旧循环留作测试 oracle，生成严格递增的 starts 和最终 length，对多组预算逐项比较所有 ranges。至少覆盖空语料、单词、零预算、正好容纳一个或多个词、超长词、最后一词的 separator，以及 `usize::MAX` 预算。检查无空段、无重叠、无遗漏和原有超预算例外。

**成本与去留**：旧算法整体是 O(n)，不是 O(n²)。新方案输出 k 段时约为 O(k log n)，当 k 接近 n 时最坏可变为 O(n log n)。可能适合词数很多、每段装很多词的情况；小预算和大量超长词可能更差。必须按“大段、小段、超长词”分别测分段耗时，再看端到端占比。若没有值得保留的收益，保留旧循环，不为使用标准 API 而引入自适应阈值和更多分支。

## 可选增量 在现有 symbols 工作上简化 initial_edges

这项依赖其他 agent 正在负责的 `symbols()` 迭代器，不应另开一个“实现 symbols”的实验。只有对方尚未处理此调用点时，才把下面增量交给同一负责人；若已覆盖，整项跳过。

前提是对方已经提供保持现有过滤与装饰语义、`Item = u32` 的无分配迭代器。调用点可使用仓库已有的 `itertools::Itertools`：

```rust
use itertools::Itertools;

// CorpusPlan::initial_edges 内部，保留现有函数签名。
for word in range {
    let (text, &weight) = self.words[word];
    for (offset, pair) in self.ids.symbols(text)
        .tuple_windows::<(u32, u32)>()
        .enumerate()
    {
        emit(pair, (self.starts[word] + offset) as u64, weight)?;
    }
}
Ok(())
```

**增量价值**：移除调用点的 `previous`、手动 `position += 1` 和 `ControlFlow` 错误拆装。`tuple_windows` 负责相邻值配对，`enumerate` 给出保留符号序列中的左端偏移。该适配器复制便宜的 `u32`，不需要 `Vec` 或 `Box`。[Itertools API](https://docs.rs/itertools/latest/itertools/trait.Itertools.html#method.tuple_windows)

**验证边界**：空序列和单符号不发边；过滤后相邻符号才形成 pair；坐标为 resident 符号坐标，不是 UTF-8 字节偏移。用首尾字符被过滤、多字节字符、prefix/suffix 和不同权重比较 `(pair, coordinate, weight)` 的完整有序流。让 emit 在第一条和后续指定边返回错误，确认立即返回同一个错误，且不再 emit 后续边或词。

保留原有 `symbol_count` 的 complete-alphabet 快路径，不改成一律 `symbols().count()`。若旧 `ControlFlow` import 仍被其他调用使用，就保留它。此增量只评估调用点简化，不重复认领 symbols API 本体，也不宣称必然减少机器指令。

## 执行分工与合并标准

1. 先与其他 agent 对齐当前基线和文件归属，记录共同 commit；文中的固定 SHA 仅用于定位，不要求回退到旧提交。已有等价改动就直接跳过。
2. A、B、C 各用独立 worktree，从同一基线开展，可并行编写和检查。只触及各自函数及必要测试；即使不同任务位于同一文件，也不要共享可写工作树。
3. 先跑各自边界测试，再跑仓库已有 BPE 正确性与可用的 parity 对照。已知的 encoder roundtrip 排除项及可选 parity feature 基线问题单独记录，不冒充通过，也不混入本次修复范围。对完全相同输入比较 vocab ID、merge 顺序和最终训练输出；保留仓库现有语义与性能测试。
4. 性能测试串行执行，基线和候选使用相同工具链、构建配置、数据、线程数和机器负载。记录原始命令、输入规模、耗时及适用的 RSS；有噪声再复测，不设机械的固定重复次数。
5. 各项分别报告改动范围、删掉的手写状态、测试结果、性能差异和是否建议保留。正确性必须全过；收益未证实时明确写“未证实”，尤其不把 B 包装成加速项目。
6. 一次只整合一项，再跑组合回归。可选增量随 symbols 负责人的分支验证，避免用未合并的接口另起重复实现。本文件只提供后续执行方案，不代表这些步骤已经执行。

## 源码依据

- [位图归集与串行合并](https://github.com/Yikai-Liao/tokenizers/blob/a2f1c0f7fda871ad321579a8df690a5f6464043f/tokenizers/tk-train/src/trainers/bpe/vocabulary.rs#L391-L411)；[initial alphabet 与 observed spans](https://github.com/Yikai-Liao/tokenizers/blob/a2f1c0f7fda871ad321579a8df690a5f6464043f/tokenizers/tk-train/src/trainers/bpe/vocabulary.rs#L82-L102)。
- [Cursor seed 读取及终止条件](https://github.com/Yikai-Liao/tokenizers/blob/a2f1c0f7fda871ad321579a8df690a5f6464043f/tokenizers/tk-train/src/trainers/bpe/positions.rs#L245-L283)。
- [initial_ranges 原实现](https://github.com/Yikai-Liao/tokenizers/blob/a2f1c0f7fda871ad321579a8df690a5f6464043f/tokenizers/tk-train/src/trainers/bpe/corpus.rs#L131-L148)；[initial_edges 原错误协议和坐标](https://github.com/Yikai-Liao/tokenizers/blob/a2f1c0f7fda871ad321579a8df690a5f6464043f/tokenizers/tk-train/src/trainers/bpe/corpus.rs#L154-L178)。
- [现有 fixedbitset、Rayon 与 itertools 依赖](https://github.com/Yikai-Liao/tokenizers/blob/a2f1c0f7fda871ad321579a8df690a5f6464043f/tokenizers/tk-train/Cargo.toml)。所有候选均不要求新增依赖。
