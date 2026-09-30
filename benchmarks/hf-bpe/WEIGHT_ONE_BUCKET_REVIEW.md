# H：权重为 1 的空间桶独立审查

## 范围与结论

本轮只读审查 H 的静态桶认证、查询捷径及三个统计字段，没有构建、测试、训练或 benchmark，也没有修改 Rust。没有发现阻塞正确性或内存安全问题。原空间目录、radix 初始化、融合和 posting 安装继续沿用 [FUSED_DIRECT_REVIEW.md](FUSED_DIRECT_REVIEW.md)、[INITIAL_RADIX_REVIEW.md](INITIAL_RADIX_REVIEW.md) 与 [POSTING_BULK_REVIEW.md](POSTING_BULK_REVIEW.md)，本轮只复核新增差分。

worktree `/root/code/tokenizers-worktrees/weight-one-buckets` 核对时干净；branch 为 `bpe/weight-one-buckets`，HEAD 为 `00216d914186e42458d45e72276b13c700749c6a`，parent 为 DE `c8702374bd8a3812f8bca34cd53e21afe01632c3`。

以下路径相对 worktree，文件内容与 HEAD 相符：

| 文件 | SHA256 |
|---|---|
| `tokenizers/tk-train/src/trainers/bpe/indexed/parallel/weight_lookup.rs` | `4e974ce81989618c5123a2309d31d912967339b6ee1a58bcab555a3e66b4469e` |
| `tokenizers/tk-train/src/trainers/bpe/indexed/parallel.rs` | `7368db5a0cbf8752c7bb9124a95bc4d4688b377029ce9fbfe0575767c8bb9e67` |
| `tokenizers/tk-train/src/trainers/bpe/indexed.rs` | `0bf990c3390fb563776f7d94035658ffed07457c148dda43b3a90efb917e1802` |

## 一、静态权重与认证证明

原 `Block::weight` 对位置 p 查找所有 `pivot<=p`，没有命中时返回 `previous_weight`，否则返回最后一个命中 pivot 对应的权重（parallel.rs:190–200）。corpus 构造按物理 word 顺序同步追加 pivot 与 weight；两数组等长、pivot 有序（corpus.rs:208–215）。merge 更新 token 槽和 posting，不移动 word 的物理范围或修改这三个权重元数据，因此该位置到 word weight 的映射在初始化和全部 merge 阶段保持静态。

H 的构造按相同顺序遍历 pivot，把 `[0,slots)` 分成半开区间：首个 pivot 前用 previous_weight，每个 pivot 起到下一个 pivot 前用对应 weight，最后一个 pivot 起到 slots 用最后 weight（weight_lookup.rs:32–40）。pivot 被截到 slots，范围外的区间为空；重复 pivot 也产生空区间，并由最后一个重复项决定后续区间的权重，与原 `partition_point(q<=p)` 完全一致。

令 `B=ceil(slots/256)`，第 b 桶的有效域是 `[256b,min(256(b+1),slots))`。bitmap 最初只对这 B 个桶置位。每个非 1 的非空区间 `[begin,end)` 清除从 `floor(begin/256)` 到 `floor((end-1)/256)` 的全部桶（:46–50）；这恰好是与该区间相交的桶。所有权重区间覆盖整个有效域，所以最终置位当且仅当该桶每个有效位置的原权重都是 1。

因此查询置位桶返回常量 1 与原查找相同。未置位桶继续执行原 bounds 加局部 partition_point 查找，表达式没有改变（:68–84）。bitmap 覆盖全部物理位置，包括 separator 和槽零；某个非 1 位置即使不会成为有效 pair 起点，仍会清除桶。这可能减少捷径命中，保证不会误认证。

## 二、边界情况

| 情况 | 行为及等价理由 |
|---|---|
| slots 为 0 | bounds 仍有一个边界；bitmap 长度为 0，不进入尾部 mask 或非空清除。桶数与置位数为 0，没有合法查询位置。 |
| 没有 pivot | 唯一区间 `[0,slots)` 使用 previous_weight；权重为 1 时认证全部有效桶，否则全部清除。 |
| 首 pivot 为 0 | 首区间为空，previous_weight 不影响任何位置，pivot 的 weight 立即生效。 |
| 重复 pivot | 中间空区间不清除任何桶；最后重复项控制接下来的非空区间。临时的 0 或最大权重不会错误污染空区间。 |
| pivot 恰在 256 边界 | 前区间止于该位置，`end-1` 使它不清除下一桶；新权重从新桶开始生效。 |
| 桶内只含一个非 1 位置 | 其所属桶被清除，所有查询继续走精确查找。 |
| 最后不足 256 槽 | 只检查实际有效域；不存在的尾槽不影响认证。 |
| pivot 等于或超过 slots | 先完成直到 slots 的前区间，再更新仅作用于空域的 weight；不改变合法查询。 |
| 权重为 0 或 `u64::MAX` | 都按 `weight!=1` 清除，不进行权重加减乘运算；回退返回原值。真实 corpus 对超出 i64 的权重仍由原预检拒绝，H 没有绕过该检查。 |

若 B 不是 64 的倍数，最后一个 u64 只保留低 `B%64` 位，移位量在 1–63；若 B 为 0 或 64 的倍数，不执行该 mask（:27–31）。所以 padding 位不会被计入置位桶数。所有清除的 bucket 都小于 B，bucket/64 和 bucket%64 分别保证 slice 下标及 0–63 的移位量有效。查询继续要求 `p<slots`，与原 lookup 的合法位置前提相同。

## 三、完整 u32 地址域与内存

lookup 仅为已有 flat 布局构造。32 位 posting block 的 corpus 构造使用 checked shift，并要求 64 位 usize；flat 允许 `slots<=2^32`（parallel.rs:574，corpus.rs:164–166）。在这个既有前提下，bucket 边界的最大乘积为 `2^32`，所有 usize 运算均可表示。

当 slots 恰为 `2^32`，B 为 `2^24`，bitmap 有 `2^18` 个 u64，逻辑 payload 为 2,097,152 字节（2 MiB）。最大合法 p 为 `u32::MAX`，命中第 `2^24-1` 桶、最后一个 u64 的第 63 位；原 bounds 的 b 和 b+1 下标也仍有效。bitmap 不以 u32::MAX 为 sentinel，没有缩小原位置域。

原 bounds 最多有 `2^24+1` 个 u32，逻辑 payload 为 67,108,868 字节（64 MiB 加 4 字节）。真实 word pivot 数不超过 `slots-1`，所以原目录中的 pivot count 转换仍能放入 u32。H 只新增一份 bitmap，空间随 slots 每 256 项增加一位；初始化区间清除的工作量为桶数加 pivot 数的量级，没有按每个 corpus 槽额外扫描或构造另一份权重数组。

`one_bucket_bytes` 使用 Vec 的实际 capacity 乘 8；`bytes` 将其加入原 bounds capacity 乘 4。两者统计 heap payload，均不含 Vec 头、allocator 元数据或 RSS。新增单独 bitmap 字节数已经包含在总 lookup bytes 中，不能再次相加。

## 四、初始化复用、并发与计时

radix eligible 且 nonuniform 时，在原初始化位置构造同一个 WeightLookup，供各 owner 的 group frequency 查询共享使用（parallel.rs:586–600；radix_count.rs:159–163）。随后 `initial_weight_lookup.or_else(...)` 把该实例移交 merge，连同 bitmap 一起复用，没有复制或第二次分配（parallel.rs:859–863）。其它原 eligible 条件及 fallback 均不变；uniform 路径仍直接使用统一权重，未构造 bitmap 时三个新统计值为 0。

lookup 构造完成后只通过共享引用读取；权重元数据也保持静态。捷径不修改 cursor、posting、排序、rule、Output 或频率聚合次序，也没有新的原子操作、锁或任务。初始化及融合查询得到相同 u64 权重，原 overflow 预检、floor、出生链顺序、apply 和 owner commit 不受影响。

`initial_weight_lookup_bytes` 和复用后的 `weight_lookup_bytes` 仍指同一 allocation；二者均包含 H 的 bitmap，不应相加。构造 bitmap 的耗时随原 lookup 构造归入初始化或 merge。三个新统计字段在原 build timer 结束后记录，其中置位数需扫描 bitmap；这次统计扫描包含在整体 merge 耗时，未计入 `weight_lookup_build_ms` 或 batch 阶段计时（parallel.rs:864–868）。已有 timer 边界没有调整。

新增测试源码以逐位置 `Block::weight` 为 oracle，同时核对桶认证位与置位计数，覆盖空域、空 pivot、previous_weight 为 1/非 1、重复和端点 pivot、0、最大权重及跨 bitmap 字的边界。本审查没有执行测试；运行结果及实际收益由主任务记录。若改变静态 pivot 映射、数组对齐关系、lookup 复用的 block，或查询位置域，需要重新证明认证条件。
