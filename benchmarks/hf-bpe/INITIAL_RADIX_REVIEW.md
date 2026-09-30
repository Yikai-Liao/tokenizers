# 候选 D：初始 pair radix 计数独立审查

## 范围与结论

本轮只读审查新增 `radix_count.rs`、共享权重目录提取及初始化集成，没有构建、执行测试或运行 benchmark。没有发现阻塞正确性问题。计数、保留 key、初始 posting 和初始剪枝数量与父版本等价；性能计时与主任务测试按既定并行安排继续。本报告不重复证明旧融合、Atomic 或 corpus 构造协议。

worktree 为 `/root/code/tokenizers-worktrees/initial-radix`，HEAD 为 `d15c18cc07047479ddc2eacd1da1844cb9ac9358`，父提交为 `a0832c488a7ece429630c7d1493da5dd772c87aa`。核对时工作树干净。以下路径相对 worktree，内容与 HEAD 相符：

| 文件 | SHA256 |
|---|---|
| `tokenizers/tk-train/src/trainers/bpe/indexed/parallel/radix_count.rs` | `6280981bb67ae8cb855653b22854b915d2250bda3d43477bccf20f0ca9840de6` |
| `tokenizers/tk-train/src/trainers/bpe/indexed/parallel/weight_lookup.rs` | `43c233b72278e35c83f006b958462262d5512d72dba2813b5d0626d19fa1be10` |
| `tokenizers/tk-train/src/trainers/bpe/indexed/parallel/fused_batch.rs` | `4a8d122f0e4a9252dba226731334424c5cc7fbdb3b4d64415f4c19cd05c8205f` |
| `tokenizers/tk-train/src/trainers/bpe/indexed/parallel.rs` | `3da609fd8c5d6136fb406d1535be4cff9c2b3f63befcd07d47b1a594975b73c1` |
| `tokenizers/tk-train/src/trainers/bpe/indexed.rs` | `425715122f2742a5d2701cca35f938bc51421caef15a17f143b6d3867cf887a8` |

`bpe/mod.rs`、alphabet 和 corpus 与父版本的差分为空。融合模块仅移出 WeightLookup 和对应测试，并增加共享模块 import；born/apply 和 owner commit 协议未变。

## 一、ID 域、记录编码与地址上界

`parallel.rs:578` 只在 flat32 且完整初始 `lengths.len() <= 65536` 时进入新计数。该 length 域由完整 strings 初始化，包含预留 special 与 forced alphabet，而非仅统计活跃字符。所以每个实际初始 ID 均在 0..65535，`code=(a<<16)|b` 完整保留两端所有位。`canonical(code)` 再展开成原 u64 pair key，方向及原 owner hash 不变。

这里的 65535 是合法的临时 pair 分量，不是 u32 corpus 的 NONE。`code=u32::MAX` 也只是合法 pair `(65535,65535)`；记录不使用这个 code 作为哨兵。公开入口的 corpus 与最终 posting 仍为 u32。私有窄 Slot 的原类型选择与 NONE 编码条件未修改。

flat 已保证 slots 不超过 `2^32`，实际 p 小于 slots；有效邻边不会起于词末 separator。将 p 放入记录低 32 位不会污染 code。设各词保留字符数为 k，W 为词数，N 为非空词数，则 `slots=1+sum(k)+W`，初始 live edges `E=sum(k)-N`。当 E>0 时 W、N 均至少为一，所以 `E<=slots-3<=2^32-3`。每个 owner stream 的长度、Group 的 exclusive end 及每个最终 posting 的 count 都在 u32 范围内；转换仍显式检查。

完整初始 ID 域过大、非 flat 或多地址块时直接保留原初始化路径，没有压缩或截断 ID。

## 二、两遍 route 与空间顺序

corpus 按连续位置分 chunk。每个 chunk 两次顺序扫描使用相同条件：排除最后一槽，并排除任一端为 NONE 的边。右端通过全局 `corpus[p+1]` 读取，chunk 最后位置的跨 chunk 邻边也会计入；后一 chunk 从下一个起点开始，不会重复该边。

第一遍计算各 canonical owner 的出现数量，按该数量预留 u64 记录缓冲区；第二遍写入 `(code<<32)|p`。初始化期间 corpus 不变，因此数量一致，记录数组不需要几何扩容。空输入、空词、全裁剪词和 separator 不产生记录。

indexed collect 保留 chunk 顺序，每个局部 owner buffer 内的 p 递增。按 owner 逐个拼接时也按 routes 的 chunk 顺序 append，故拼接后的每个 owner stream 在排序前按全局 p 严格递增。owners、routed、grouped 都恰好有 `config.workers` 项；后续并行 zip 不会少装一个 owner。初始化专用池只控制执行并发，不改变 owner 数。

## 三、稳定 radix、权重与低频剪枝

四个 pass 的 shifts 为 32、40、48、56，按 LSD 顺序覆盖 code 的全部 32 位，不排序 position 的低半部分。每个 pass 的 counts 总和等于记录数，prefix 为每桶精确起点；输入顺序 scatter 后各桶仍稳定。四次稳定排序因而按 code 分组，并保留同 code 原先严格递增的 p。

每组的频率为该 key 所有出现的 word weight 之和。uniform 用出现数量 checked 相乘；nonuniform 用共享目录逐地址查权重。记录低 32 位就是原 p，目录仍查同一个 block 的词起点权重。旧 cursor 与目录查找的等价性见 [FUSED_DIRECT_REVIEW.md](FUSED_DIRECT_REVIEW.md) 第七节；本轮核对提取后的目录算法除可见性与格式外逐字相同，测试内容也相同。

nonuniform 的 `.sum::<u64>()` 虽未逐步 checked，但此前 corpus 构造已 checked 验证全部初始 weighted edges 总和不超过 i64::MAX。当前组仅包含该非负总量的一部分，故不可能溢出 u64。零权重出现仍保留在记录中；floor 继续为 `max(1,min_frequency)`，它们和其它出现先合计，再判断整个 key。

低于 floor 的组在创建 owner entry 和最终 posting 之前丢弃。每个 canonical key 固定归属唯一 owner，且在该 owner 排序后只有一组，因此 `pruned += 1` 正好按低频 key 计数，不按出现数或 worker 分片计数。`stats.pruned_pairs` 先取该值，再执行原 common retain；已安装的全部组均达到 floor，后一次 retain 在 radix 路径不再增加同一批剪枝。fallback 仍由原 retain 计数。

## 四、安装、heap 与 posting 等价

仅保留达到 floor 的 Group，按其数量预留 owner map。各 Group 展开为同一个 canonical key，保存相同 frequency，并按稳定排序后的原 p 写入最终 SmallPosting。每条出现恰好写一次，posting 严格递增，新增 debug assertion 检查该条件。

posting 按已知 count 调用原 `SmallPosting::with_capacity`：最多两项保持 inline，三项沿用原 helper 的四槽最小 heap，四项及以上按请求 count 预留，并记录实际 capacity。因此它避免按每次 push 扩容，但不能把所有 posting 的物理容量都描述成严格等于 count。

common heap 构建和 Candidate 的排序规则未改变。保留的 canonical key、frequency 与 posting 与父版本相同；map 的插入顺序或容量变化不会改变按 frequency/key 比较的 greedy 选择。AA 所依赖的初始位置顺序也保持不变。早过滤只消除原 heap 阶段同样会删除的低频 key。

## 五、缓冲区生命周期与容量统计

临时记录为 8 字节，不含重复的每位置权重：两个初始 u16 ID 在高半部分，u32 position 在低半部分。设 E 为初始 live edges，则两次 route 的初始记录总请求容量为 `8E`。

owner compaction 逐个建立最终 stream，在每个 chunk buffer append 后释放该旧 buffer。最早一刻同时持有当前旧 routes、此前 owner streams 和新 owner stream；`peak_initial_route_buffer_bytes` 在分配新 stream 时更新该和。它没有同时保留全部旧 routes 和全部新 streams。

radix scratch 按各 owner 的记录数分配，四次 swap 后记录 Vec 恢复使用原 allocation，scratch 在 sort 返回时释放。所有并行 sort 完成后才开始 grouping 和最终 posting 分配。`initial_radix_scratch_bytes` 为各 sort 返回的 capacities 之和，是同时 scratch 的容量上界；不是测量到的物理峰值。route peak 再取最终 streams 加该 scratch 的上界。

grouping 只保存保留 key 的 Group。安装时每个 owner 的完整 records 仍与正在建立的 owner map/postings 共存；该 owner 的 groups 和 records 在安装 closure 结束前释放。全部初始化返回时，不留下 route、scratch 或 Group payload。命名统计的范围为：

| 字段 | 覆盖的容量 |
|---|---|
| `initial_route_buffer_bytes` | compact 后全部 u64 owner streams |
| `peak_initial_route_buffer_bytes` | route compaction 和排序阶段的 record/scratch 容量上界 |
| `initial_radix_scratch_bytes` | 各 owner 排序 scratch capacities 之和 |
| `initial_group_buffer_bytes` | retained Group Vec capacities×实际 Group 大小 |

这些字段不包括 Vec 目录、allocator、线程栈、语料、输入、vocabulary、word metadata 或安装期间的 owner map/posting，也没有把各阶段最大值合成全进程峰值。以既有输入的 `E=203,230,114` 计算，记录约 1.51 GiB，记录加完整排序 scratch 上界为 **3,251,681,824 字节，约 3.03 GiB**；这是 named buffers 的容量界，不是 RSS。运行仍按主任务的实际 MemAvailable 与 RSS 监测记录。

## 六、目录复用与阶段计时

nonuniform radix 路径在初始化计数之前建立目录，之后 move 到 merge 的同一个 Option，没有重复分配。`initial_weight_lookup_bytes` 与 `weight_lookup_bytes` 在复用时描述同一个 allocation，不能相加。目录在初始化返回后继续存在，临时 records/groups 则已经释放。uniform 不需要目录；radix 不适用时保持原 merge 按需建立方式。

`initial_weight_lookup_ms` 计入 initialize；merge 的 `weight_lookup_build_ms` 对复用路径只包含 Option 移动/选择，不再代表最初建立成本。对比父版本时应把目录建立移出 merge 的这部分单列，并同时看初始化和总训练。

`initial_route_ms` 包含两次 scan、精确缓冲区分配和 owner compaction，`initial_route_compact_ms` 是它的子阶段。`initial_count_ms` 包含 sort、group 和 install；三个子字段不能再与 count 相加。common heap 仍单列 `initial_heap_ms`。旧融合 prepare/apply/commit 的计时边界未改变。

## 七、测试证据与复核条件

新增排序测试源码对 8192 条记录与稳定标准排序结果逐条比较，覆盖 code 的最高位、全零、全一、重复 key 及全部低位变化，并检查 `(65535,65535)` 的 canonical 展开和空/单记录。共享目录的原逐地址 oracle 测试完整迁移。本轮没有执行这些测试；主任务已报告新增排序测试通过，45 项库测试及 release 构建正在并行推进，完整运行结果由主任务记录。

若更改完整 ID 域门槛、记录低半部分、pass 顺序或稳定 scatter、owner chunk 拼接顺序、初始化 weighted 总量检查、floor 计数或目录复用，需要重新复核相应结论。性能及实际内存收益以锁定源码的测量结果记录。
