# I：出生 posting 直接装配独立审查

## 范围与结论

本轮只读审查 I 的 flat owner commit 聚合与装配，没有构建、运行测试或 benchmark，也没有修改 Rust。没有发现阻塞内存安全或结果等价问题。既有出生唯一生产者、AA 选择、融合顺序及批量写入 API 的证明继续沿用 [FUSED_DIRECT_REVIEW.md](FUSED_DIRECT_REVIEW.md)、[PAIR_MONOTONICITY.md](PAIR_MONOTONICITY.md) 与 [POSTING_BULK_REVIEW.md](POSTING_BULK_REVIEW.md)；本轮复核新的 descriptor 链及一次性填充。

worktree `/root/code/tokenizers-worktrees/commit-direct-assemble` 核对时干净；branch 为 `bpe/commit-direct-assemble`，HEAD 为 `e3a1954ccdce4b80ea2865792088b1fe82fc352c`，parent 为 H `00216d914186e42458d45e72276b13c700749c6a`。

以下路径相对 worktree，文件内容与 HEAD 相符：

| 文件 | SHA256 |
|---|---|
| `tokenizers/tk-train/src/trainers/bpe/indexed/parallel/flat_commit.rs` | `7b07c78b242575ba7e27fc4eeaef714fd8750a2131b529292656f76d45d9a9d3` |
| `tokenizers/tk-train/src/trainers/bpe/indexed/parallel.rs` | `ba48240b17c2aef7f41e6835789d96b22aca41ec9695cb59192ee30420f7fa25` |
| `tokenizers/tk-train/src/trainers/bpe/indexed.rs` | `5a70bd48ba6ff53e817a16055a75738049a2edf8a3a7ecfee10e562cf0ff7ac3` |

## 一、聚合、floor 与 key 身份

新 helper 按原 output 顺序遍历相同 route delta。`occurrences!=0` 的 Group 累加原 weight 和 checked occurrence count，并为该局部出生组添加一个 SourceGroup；零 occurrence 的 Group 执行与 parent 相同的旧频率 checked subtraction、低于 floor 时删除及 retired 记录（flat_commit.rs:20–50）。

全部 outputs 聚合完成后才判断全局 weight 是否低于 floor（:54–58）。每个低于 floor 的 key 仍只记一次 dropped；accepted key 的 frequency 与总 count 均与 parent 相同。零权重出生仍创建 descriptor 并计入 count，所以若其它 output 使同一 key 通过 floor，零权重位置也不会丢失。

删除 key 的两端都属于批次开始时的身份；出生 key 至少含一个本批新激活的 replacement。reserved canonical ID 仍仅在尚未活跃时激活，并单独成批；普通 replacement 各不相同。因此本批 born key 不会别名旧 ledger key或删除 key，同一 delta Group 也不混合两种语义。所有这类选择与出生逻辑均未修改，安装前的 `!ledger.entries.contains_key(k)` debug assertion 保留（:59）。

accepted key 完成 posting 后只执行一次 ledger insert 和一次 heap push，最终候选 key/frequency 与 parent 相同。born map 遍历顺序不定义候选优先级；既有 Candidate::Ord 仍按频率及 key 形成相同全序（parallel.rs:126–140）。

## 二、descriptor 与最终 posting 的顺序

对固定 key，每个 output 的 delta 最多有一个 Group。SourceGroup 在升序 output 遍历中添加，并通过 `next=entry.head` 前插到该 key 的 descriptor 链（flat_commit.rs:34–39）。所以从最终 head 遍历得到该 key 的 output 降序；其它 key 的 map 遍历次序不影响这条链。

设 output i 对该 key 按生产顺序产生的片段为 `S_i`。既有 `Output::birth` 每次把新 Node 插到链头，所以该组的 Node 链遍历顺序为 `reverse(S_i)`。parent 将每段链分别倒序填充，再按 output 升序追加，结果为 `S_0 || S_1 || ...`，跳过没有该组的 output。

I 的 producer 先消费最大 output 的 Node 链，再消费前一个 descriptor，产生 `reverse(S_last) || ... || reverse(S_first)`。对整个流调用一次 `append_reversed_reserved(total_count)`，物理位置按末端到起点写入，最终恰好恢复 `S_first || ... || S_last`（:60–78）。这证明 posting 内容和 parent 完全相同；不依赖不同 key 的哈希遍历次序，也不需要重新排序或去重。

有序性继续由原生产协议保证：

- AA / Plan 路径按升序物理 plans 划分连续 output chunks；AA parity 选择保持这个顺序。左出生来自固定旧左 ID，位置为 `p-len(left)`；右出生位置为 p，相邻已选合并的边界由左侧合并唯一生产。对固定 key，局部及跨 output 的位置均严格递增（parallel.rs:1058–1175）。
- 非 AA 融合路径的 jobs 是 `(rule rank, ordered posting)` 的连续分段，有序 collect 保留 job 次序。固定 born key 有唯一 selected rule 生产者；因此其 posting 片段按 job 顺序严格递增，继续适用原融合证明。

helper 保留最终 posting 全段严格递增的 debug assertion（:80–83）；release 的顺序由上述不变量及两次反转的等价性保证。

## 三、索引、计数与链完整性

每个局部出生 Group 的 occurrences 大于零，head 指向有效 Node；链长度恰好等于 occurrences。Node 的 u32 索引和 NONE 分离继续由原 `Output::birth` 的转换与 sentinel 检查保证（parallel.rs:301–317）。新 SourceGroup 保存原 head，不修改节点或原 Group。

descriptor 下标来自当前 `sources.len()` 的 checked u32 转换，并显式拒绝 NONE；失败发生在该 descriptor 加入 born 链之前（flat_commit.rs:23–27）。每个 next 是此前已安装的同 key descriptor 下标，严格小于当前下标，或为 NONE，所以 descriptor 链无环且只引用有效项。output 下标保持 usize，来自 outputs 枚举；owner 下标来自外层 owners 枚举，所有 Output 的 flat_routes 长度与该 owner 数相同。

每次聚合 count 使用 checked addition。只由正 count 局部组创建的全局 Group，count 也必为正，head 必存在，因此 accepted key 的首次 source/head 读取有效（:60–62）。总 count 等于其全部 descriptor 所引用 Node 链长度之和。

producer 每次读一个有效 Node，更新 head；消费完当前链时转到 next descriptor，若存在则取得其 head（:67–76）。API 恰调用 total_count 次，因此不会在最后 descriptor 耗尽后再读取 NONE，也不会漏消费节点。最后一次正常调用仍返回刚读到的有效 position，并将 source_index 置为 NONE；随后 assertion 检查这个结束状态（:79）。完整 u32 位置域、节点位置转换与最终 posting 的 u32 count 范围均保持 parent 的约束。

## 四、局部 posting 所有权与中断

新 posting 由 `with_capacity(total_count)` 一次创建，len 初始为 0。最多两项为 inline，三项仍使用原四槽最小 heap，更多项保持原 reservation；批量填充没有扩容。parent 同样按全局 count 为每个 accepted key 预留一次，I 改变的是装配和安装次序。

`append_reversed_reserved` 在完整填充前不发布新 len，producer 只读取 immutable outputs/sources，并更新局部两个链游标，不访问目标 allocation。helper 没有新增 unsafe；源的 descriptor、route 和 Node 读取全部使用有界 slice 索引。

若 producer 的源索引因内部不变量损坏而 panic，局部 posting 仍唯一持有其 allocation，len 保持 0；已写的 u32 suffix 无析构义务。unwind 时原 posting Drop 按原 allocation 与 capacity 释放，不读取未完成 suffix。完整填充后 len 发布为 count，debug assertion 或后续操作中断时，posting 已全部初始化，Drop 同样安全。ledger insert 仅在填充和检查之后执行。

这保证局部 posting 的所有权与内存安全，不承诺整个 commit 出错时回滚：旧 pair 删除或此前 key 的安装可能已经发生，和原训练错误路径一样，函数返回错误或 unwind 后不会继续正常训练。

## 五、并发、指标与验证口径

外层仍对 owners 使用 `par_iter_mut`，每个 helper 独占对应 Owner，只读所有 Output；descriptor Vec 和 born map 为 owner 局部状态。apply/commit 的同步 join、owner hash 路由、非 flat 路径和初始化均未改变。非 flat commit 只为返回 tuple 增加一个值为 0 的字段。

在当前 64 位 flat 前提下，SourceGroup 的 usize/u32/u32 payload 为 16 字节。每个局部出生 Group 一个 descriptor，包括最后被全局 floor 丢弃的 key；这是新增的 owner 局部 heap 缓冲区。单 owner 以 `sources.capacity()*size_of::<SourceGroup>()` 记录，在每批各 owner 的记录相加后取跨批最大值（flat_commit.rs:52；parallel.rs:1242–1245）。

`peak_commit_descriptor_bytes` 表示每批各 owner descriptor reservation 的总和，再取最大值；不包含 born map、Nodes、postings、Vec 头或 allocator 元数据。各 owner 的执行和释放时间可能不同，因此它不是实测同时驻留内存峰值或 RSS。descriptor 统计、装配和新的跨 owner 汇总均包含在原 commit timer 内；timer 边界未调整。

新增测试源码覆盖中间空 output、多 source 合并、global floor、零权重来源、旧频率减少与退休，并检查最终 `[2,4,10]`、frequency、heap 与 descriptor 字节数。本审查没有执行测试；主任务另行记录执行结果和正式性能测量。若改变 output 顺序、Node 链方向、descriptor 计数、首次激活条件或 posting len 发布协议，需重新复核本证明。
