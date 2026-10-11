# feed 两种实现与对比验证方案

实现两个版本，与当前基线比较：**A 保留局部 AHashMap 聚合后写入 SCC；B 在并行路径直接写入 SCC，不使用局部 map。两个版本的单线程路径都保留 AHashMap。** 使用各 256 MiB 的中文、英文语料，分别验证正确性和性能，再决定保留哪一种。

本文包含 A 的完整设计草图、B 的核心代码及验证计划。尚未修改仓库，也未编译、运行测试或验证性能。AtomicU64 仅作为 B 的后续可选实验，不增加第三个主实现。

## 参照范围

- 已核对的参照分支：`simplify/bpe-flat-20261010`
- 参照提交：`a2f1c0f7fda871ad321579a8df690a5f6464043f`
- 文件：`tokenizers/tk-train/src/trainers/bpe/feed.rs`
- 当前参照文件共 147 个物理行；最初的压缩草图为 80 行，但尚未编译，且需要恢复 `Entry` 分支。
- 该提交是已核对的源码参照，不要求实施时切回旧提交。实验基线应冻结在实际开始实施时的当前版本，记录真实提交和相关工作树差异；A、B 从同一基线出发，不能混用不同阶段的基线。

## 版本 A 保留局部聚合并简化组织

把共同计数逻辑收进一个 `count` 函数和两个有名闭包 `flush`、`accumulate`。保留 `Entry::Occupied` 只递增、`Entry::Vacant` 插入后才检查容量的写法，不为少几行代码增加重复词路径上的判断。

1. 删除 `LocalCounts` 结构体及其 `new`、`add`、`flush`、`finish` 方法，也不新增替代结构体或 trait。局部状态直接使用 `CountMap`。
2. 用 `accumulate` 统一回调调用、错误传播和逐词累加。已有键只加一；只有新键插入后检查 `LOCAL_KEY_LIMIT`。串行路径仍会在新键处检查长度，不能宣称完全没有额外判断。
3. 用 `flush` 汇总加权计数。`shared` 是延迟创建的 `Option<SharedCounts>`：单线程为 `None`，不创建 SCC；并行才建立共享表。串行达到 2048 个键时的 `flush` 调用为空操作，不会清空串行 map。
4. 单线程直接对原始输入 `fold`，最终保留 `WordCounts::Map`；不经过批处理、桥接或共享表，也不复制成另一张哈希表。
5. 并行分支把批处理生成器就地放入 `count`：输入先 `fuse`，前 `workers` 个输入各成一批，之后每批最多 `FEED_BATCH_SIZE = 32` 个。启动单元素批给短流、昂贵回调提供并行机会，但不保证每个 worker 恰好取得一个。
6. 保留 `maybe_par_bridge → flat_map_iter → fold`。回调在桥接之后执行；若先在上游串行迭代器中调用 `process`，回调会落入桥接读取输入的串行区域。
7. 删除收集全部 fold 结果、统一检查成功、再发布尾批的两次中间收集。每个成功 fold 在 `map` 中发布自己的尾批，普通 `reduce` 合并错误状态。
8. 保留 SCC 消费前的容量 reservation，并直接消费键值到 `WordCounts::Entries`。输出不承诺顺序，也不重新分配和 rehash 一张计数字典。

## 两个版本的语义要求

- **普通错误后继续调用回调。** `process(sequence.as_ref())` 必须在 `counts?` 或 B 的 `state?` 之前执行；串行 `fold` 和并行普通 `reduce` 不改成 `try_fold`、`try_reduce` 或其他提前停止形式。这里不承诺 panic 等异常中断后仍处理所有输入。
- **错误选择不增加新保证。** 单个 fold 保留其先遇到的错误；并行分区和归约顺序不稳定，不能承诺返回按输入顺序出现的第一个错误，也不能宣称 fail-fast。
- **A 移除私有的全成功尾批屏障。** 当前参照实现只有在全部 fold 成功后才发布剩余计数；A 允许某个成功 fold 发布尾批，同时另一个 fold 已失败或将失败。满 2048 键的中途发布原本就可能发生。B 逐词写入，不存在局部尾批。
- **失败结果仍不提交给 trainer。** SCC 只在本次 `count` 内部使用，整体失败时随调用丢弃，不返回部分计数。实施时仍须验证调用方仅在成功后替换 trainer 状态；回调自身的外部副作用不会回滚。
- **A 的 2048 限制的是并行局部不同键数。** 它不是字节或整体内存上限，也不是整个串行 map 的键数限制。字符串长度、回调输出、全局不同键数及并行 fold 数量都会影响峰值内存。B 没有这层局部缓存，但也没有整体内存上限。
- 输入仍流式消费，不预载整个语料；不引入新依赖。

## 版本 A 完整代码草图

以下包含导入、别名、常量及完整 `count`。类型推断、借用关系和具体依赖 API 仍需在实际仓库编译确认。

```rust
//! Local preaggregation into a shared table; feed returns unordered counts.
use super::word_counts::WordCounts;
use ahash::{AHashMap, RandomState};
use compact_str::CompactString;
use std::collections::hash_map::Entry;
use tk_encode::{Result, parallelism::*};

type CountMap = AHashMap<CompactString, u64>;
type SharedCounts = scc::HashMap<CompactString, u64, RandomState>;

// Bound distinct local keys in parallel, not bytes or the serial map.
const LOCAL_KEY_LIMIT: usize = 2048;
const FEED_BATCH_SIZE: usize = 32;

pub(super) fn count<I, S, F>(iterator: I, process: &F) -> Result<WordCounts>
where
    I: Iterator<Item = S> + Send,
    S: AsRef<str> + Send,
    F: Fn(&str) -> Result<Vec<String>> + Sync,
{
    let hash = RandomState::default();
    let workers = if get_parallelism() {
        current_num_threads()
    } else {
        1
    };
    let parallel = workers > 1;
    let shared = parallel.then(|| SharedCounts::with_hasher(hash.clone()));
    let flush = |counts: &mut CountMap| {
        if let Some(shared) = &shared {
            for (word, count) in counts.drain() {
                shared
                    .entry_sync(word)
                    .and_modify(|total| *total += count)
                    .or_insert(count);
            }
        }
    };
    let accumulate = |counts: Result<CountMap>, sequence: S| -> Result<CountMap> {
        // Call every input after ordinary errors; retain this fold's first error.
        let words = process(sequence.as_ref());
        let mut counts = counts?;
        for word in words? {
            match counts.entry(CompactString::from(word)) {
                Entry::Occupied(mut entry) => *entry.get_mut() += 1,
                Entry::Vacant(entry) => {
                    entry.insert(1);
                    // Repeated words need no size check.
                    if counts.len() == LOCAL_KEY_LIMIT {
                        flush(&mut counts);
                    }
                }
            }
        }
        Ok(counts)
    };
    if !parallel {
        return iterator
            .fold(Ok(CountMap::with_hasher(hash.clone())), accumulate)
            .map(WordCounts::from_map);
    }

    // Seed with single inputs, then amortize the bridge lock across batches.
    let mut iterator = iterator.fuse();
    let mut singles = workers;
    let batches = std::iter::from_fn(move || {
        let size = if singles > 0 { 1 } else { FEED_BATCH_SIZE };
        singles = singles.saturating_sub(1);
        let batch: Vec<_> = iterator.by_ref().take(size).collect();
        (!batch.is_empty()).then_some(batch)
    });
    batches
        .maybe_par_bridge()
        .flat_map_iter(std::iter::IntoIterator::into_iter)
        .fold(|| Ok(CountMap::with_hasher(hash.clone())), accumulate)
        .map(|counts| -> Result<()> {
            let mut counts = counts?;
            flush(&mut counts);
            Ok(())
        })
        // Ordinary reduction completes all callbacks despite ordinary errors.
        .reduce(|| Ok(()), |left, right| left.and(right))?;

    let shared = shared.unwrap();
    let mut words = Vec::with_capacity(shared.len());
    // Keep table capacity stable while consumption removes its entries.
    let _reservation = shared.reserve(shared.capacity());
    shared.iter_mut_sync(|entry| {
        words.push(entry.consume());
        true
    });
    Ok(WordCounts::from_entries(words))
}
```

上述完整草图共 **93 个物理行**，包括 5 个空行和 7 个纯注释行；相对 147 行参照文件减少 54 行。统计范围都是整个代码文件，不含 Markdown 围栏。减少部分来自注释合并，不能全部解释为逻辑删除；不为压到 85 至 90 行而牺牲可读性。

## 版本 B 直接写入 SCC

当前 `LocalCounts` 本来就用 `AHashMap` 做局部聚合，每到 2048 个不同键就 flush 到 SCC；A 保留的是既有策略。局部 map 不是正确性所必需：重复词较多时能减少共享表更新；低去重场景下也可能引入两次哈希和额外表维护。B 用实测检验能否省掉这层成本，不能事先断言一定更快。

B 不引入类型框架来强行统一单线程与并行状态。单线程用一个短 `fold` 累加 `Result<CountMap>`；并行 `fold` 只保留 `Result<()>`，每个词直接更新 `SharedCounts`。普通错误后继续调用 `process`，但出错 fold 不再提交后续计数。

下面只列出 B 的计数核心，并非第二份完整文件。沿用 A 的导入、类型、worker 判断和哈希初始化；批处理生成器仍在单线程早退之后建立，使用相同的 fuse、启动单元素批和 32 输入批。最终 SCC reservation 及消费为 Entries 的代码也保持一致。

```rust
if workers == 1 {
    return iterator
        .fold(Ok(CountMap::with_hasher(hash.clone())), |counts, sequence| {
            let words = process(sequence.as_ref());
            let mut counts = counts?;
            for word in words? {
                *counts.entry(CompactString::from(word)).or_default() += 1;
            }
            Ok(counts)
        })
        .map(WordCounts::from_map);
}
let shared = SharedCounts::with_hasher(hash.clone());
let accumulate = |state: Result<()>, sequence: S| -> Result<()> {
    let words = process(sequence.as_ref());
    state?;
    for word in words? {
        shared
            .entry_sync(CompactString::from(word))
            .and_modify(|count| *count += 1)
            .or_insert(1);
    }
    Ok(())
};
// Build batches here using the same generator as A.
batches
    .maybe_par_bridge()
    .flat_map_iter(std::iter::IntoIterator::into_iter)
    .fold(|| Ok(()), accumulate)
    .reduce(|| Ok(()), |left, right| left.and(right))?;
```

B 删除并行局部 map、`LOCAL_KEY_LIMIT` 检查、drain 和尾批 flush；每次出现的词都需要访问共享表。并行源读取和回调调度保持与 A 相同，使比较主要反映聚合策略差异。

## B 的可选 AtomicU64 实验

可以进一步尝试 `scc::HashMap<CompactString, AtomicU64, RandomState>`：命中时通过 `read_sync` 取得计数并执行 `fetch_add(1, Ordering::Relaxed)`；未命中时用 `entry_sync` 再次检查，已插入则原子加一，否则插入 `AtomicU64::new(1)`，处理多个线程同时发现缺失的竞争。

这不是无锁方案。SCC 3.8.8 的 `read_sync` 仍涉及桶的读锁；缺失路径多一次查找，热门词仍有原子计数器争用。若所有词仍走 `entry_sync`，就同时承担独占访问与原子 RMW，不能把换成原子类型本身当作优化。API 依据：[read_sync](https://docs.rs/scc/3.8.8/scc/hash_map/struct.HashMap.html#method.read_sync)、[entry_sync](https://docs.rs/scc/3.8.8/scc/hash_map/struct.HashMap.html#method.entry_sync)。

`Relaxed` 只用于计数，不能承担其他状态的发布同步；最终读取在并行工作结束后进行。实现前还须检查目标支持 `target_has_atomic = "64"`，并明确溢出策略：`fetch_add` 会回绕，而普通 `u64 += 1` 在启用溢出检查时可能 panic，不能宣称两者在溢出情况下完全等价。先测普通 `u64` 的 B；只有需要进一步定位共享更新成本时，才单独评估这一变体。

## 正确性验证

实施后先编译，再以按键比较的计数结果做断言，不比较 `Entries` 的遍历顺序。至少覆盖关闭并行、开启并行但只有 1 个 worker，以及多个 worker。

1. **空输入和空回调输出。** 计数为空、无多余回调；检查单线程返回 Map、并行返回 Entries 的既有约定。
2. **输入只处理一次。** 用可记录读取次数和回调次数的迭代器验证不丢失、不重复；记录 ID 集合，不要求并行回调顺序。
3. **耗尽状态。** 使用会在 `None` 后再次返回 `Some` 的非 fused 迭代器，确认并行批处理在第一次耗尽后停止，不复活输入；串行仍保持原始 `fold` 的行为。
4. **批边界。** 对启动单元素批数 `W`，覆盖长度 `0`、`1`、`W−1`、`W`、`W+1`、`W+31`、`W+32`、`W+33`、`W+64`；含最后一个不足 32 的批。单线程取有效长度，避免负数。
5. **局部键边界与重复词。** 在一次回调中返回 2047、2048、2049 个不同词，确保 A 确实穿过同一 fold 的边界；再加入边界前后的重复词、跨 flush 的相同词及多次 flush，验证加权总数。B 跑相同用例检查直接累加结果。另检验两者串行超过 2048 键仍保留完整 map。
6. **普通错误后仍调用。** 在首部、中部、尾部设置一个及多个错误，记录所有输入 ID 的回调次数，确认各一次。并行只断言结果是允许的错误之一，不断言输入序最早错误。
7. **失败不改变 trainer。** 预置可比较的 trainer 状态；A 覆盖中途 flush、成功 fold 尾批发布与另一 fold 失败的组合，B 覆盖已写 SCC 后另一 fold 失败。最终必须返回错误且 trainer 状态不变。必要时用受控同步安排交错，不依赖概率。
8. **成功结果一致。** 对中文、英文及小型可人工核对样本比较基线、A、B 在串行和多线程下的完整计数。对按键排序后的 `(词, 次数)` 生成一致性哈希，同时比较训练后模型、词表及 merges；保留现有训练回归测试，区分纯序列化顺序变化与实际内容差异。

## 256 MiB 中文和英文性能验证

建立基线、A、B 三个独立工作树，全部源于同一个冻结基线。A、B 的实现工作可以并行进行；构建完成后的性能测量必须依次运行，不能同时跑三者或混入其他重负载，避免 CPU、内存、页缓存和 swap 相互干扰。

A 的 `CountMap` 和 SCC 都使用同一次 `count` 创建的 `RandomState` 的克隆，因此本次调用中共享同一组哈希种子；这不承诺不同 `count` 调用的种子相同，也不代表局部 map 已算出的哈希可以直接复用。flush 插入 SCC 时仍会再次计算哈希。

- **语料。** 复用现有的中文、英文各 256 MiB 测试文件，不为凑整字节数任意截断或重新生成。记录来源、原有取样规则、有效 UTF-8 检查、实际字节数和 SHA-256；三者使用完全相同的文件与读取方式，不在某个版本预载全量输入。
- **线程。** 每种语言都测 1 个 worker 和用户平时使用的多 worker 配置，记录实际线程数。关闭并行与开启并行但仅 1 个 worker 的语义均需正确性覆盖，不据此额外扩张主性能矩阵。
- **回调。** 主测试使用当前实际默认预分词器及同一训练参数。若现有基准已经同时覆盖 ByteLevel 和 Whitespace，则沿用；不为了填满矩阵临时扩充组合。补充小型便宜及昂贵回调、低去重中文、重复词密集和短流检查，用于解释差异，不代替两份 256 MiB 实测。
- **重复。** 每个场景先做 2 组包含基线、A、B 的配对运行，轮换执行顺序；固定构建模式、编译器、依赖、硬件、输入顺序及缓存处理方式。仅在波动、结果冲突或差异接近噪声时增加重复，不强制每项跑 5 次。
- **指标。** 分别记录 feed 阶段和端到端的 wall time、CPU time、吞吐、峰值 RSS，以及 swap 活动和主要缺页情况。各阶段峰值若测量工具无法分离，就明确标为整进程峰值，不能把它误记成 feed 峰值；保留原始日志和退出状态。
- **结果校验。** 每次成功运行检查计数一致性哈希及模型内容，不用耗时替代正确性。若基线本身存在非确定性，先复现并限定允许差异，不能直接忽略模型不一致。
- **解释。** 区分吞吐改善与内存代价，中文和英文分别报告。若差异落在测量波动内，就报告“未观察到明确差异”；没有数据前不承诺 A、B 或原子变体提速、无回退或降低峰值内存。

## 实施顺序和验收

1. 确认当前工作树与已核对参照的差异，冻结实际基线，准备三个工作树和两份固定语料。
2. 分别实现 A、B，保留调用接口及 WordCounts 输出形式。修正编译与格式问题，重点检查闭包捕获、`Result` 类型、普通错误契约和 SCC 消费 API。
3. 完成边界、错误、trainer 状态及模型一致性测试，再隔离运行三者的配对性能比较。
4. 根据中文、英文、单线程和多线程的实测结果选择实现；若发生可重复回退，定位批处理、局部聚合或共享更新的具体成本。AtomicU64 仅在需要继续调查 B 的共享更新成本时追加。

验收以实现更清楚、计数及错误契约成立、性能和内存权衡有可复现证据为准，不以压到某个行数为目标，也不预先指定 A 或 B 获胜。
