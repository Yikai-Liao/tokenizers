# E 与 DE：posting 批量写入独立审查

## 范围与结论

本轮只读审查新增私有 `append_reversed_reserved` API，以及 E/DE 的 owner commit 调用和 DE 的 radix 初始安装调用，没有构建、执行测试或运行 benchmark。没有发现阻塞内存安全或结果等价问题。旧融合及 radix 的完整证明分别沿用 [FUSED_DIRECT_REVIEW.md](FUSED_DIRECT_REVIEW.md) 与 [INITIAL_RADIX_REVIEW.md](INITIAL_RADIX_REVIEW.md)，本轮只复核追加写入的差分。

两个 worktree 核对时均干净：

| 候选 / worktree（位于 `/root/code/tokenizers-worktrees/`） | HEAD | parent |
|---|---|---|
| E / `posting-bulk` | `35eaf03cb59a421662d41a0ba50929fae7e2e2ba` | `a0832c488a7ece429630c7d1493da5dd772c87aa` |
| DE / `radix-posting-bulk` | `c8702374bd8a3812f8bca34cd53e21afe01632c3` | `d15c18cc07047479ddc2eacd1da1844cb9ac9358` |

以下路径相对 worktree，内容与各自 HEAD 相符：

| 候选 | 文件 | SHA256 |
|---|---|---|
| E、DE（完全相同） | `tokenizers/tk-train/src/trainers/bpe/indexed/small_posting.rs` | `6e0da4becf4ea102ea90e76c79c74bd24f652c2910a61ced3b7563317ca5c1a2` |
| E | `tokenizers/tk-train/src/trainers/bpe/indexed/parallel.rs` | `d06bedf6bf228d6896ae00c185ff32df17a9d1ee66d29a8204469a6cd1a908cd` |
| DE | `tokenizers/tk-train/src/trainers/bpe/indexed/parallel.rs` | `36b40c9ef701052c442c00d5ac54b9be6fedfbd50284000b9c0a16e5d0795284` |
| DE | `tokenizers/tk-train/src/trainers/bpe/indexed/parallel/radix_count.rs` | `a2f53ae830dc4edad3a148ee142bccd70fb1ea1b2f45824b56e6a2f27fb2242e` |

## 一、私有 API 的范围检查与初始化

API 以独占 `&mut self` 借用 posting，先 checked 计算 `end=old_len+count`，再按 capacity tag 选择 inline 或 heap。inline 要求 end 不超过 INLINE；heap 要求 end 不超过其记录的实际 allocation capacity。失败在调用 producer 或写入之前返回错误，没有扩容或转移 allocation。

inline union 当前有效成员由 `capacity==0` 的既有表示保证，并且整个数组已由 Default 初始化。所有目标下标都在 `[old_len,end)` 内，检查覆盖其固定数组边界。heap 的 raw pointer 来源于原 Vec allocation，目标 i 满足 `i<end<=capacity`，处于原 allocation 内；既有 Vec 分配也满足指针跨度的 isize 限制。`write` 可以初始化此前未初始化的 suffix，也可以覆盖上一次中断留下的 Copy 值。

循环按目标下标倒序运行，每次取得一个有效 T 后写入唯一槽；全部槽写完才把 len 一次更新为 end。没有读取未初始化 suffix、修改旧 prefix、创建共享可变引用或改变 allocation 的所有权。API 本身没有新增分配。

count 为零时范围为空，producer 不执行，len 保持原值。T 的 `Copy` 约束排除元素 Drop，当前调用类型为 u32，测试另覆盖 u16。

## 二、panic 与恢复

若 `next()` 在填充期间 panic，len 尚未发布，仍为 old_len。已经写入的高位 suffix 不属于公开 slice，旧 prefix 和 capacity tag/raw pointer 均不变。heap drop 仍按旧 len 与原 capacity 重建 Vec；不会读取未完成 suffix，额外写入的 Copy 元素也无需析构。inline 的数组始终已初始化。

捕获 panic 后可再次按原 len 填充 suffix，或执行原 push；这些操作仅覆盖未发布的额外位置。正常完整写入后，suffix 全部有效，再发布的新 len 也满足原 as_slice/drop 不变量。这里的「保持旧 len」针对 producer 调用在填充过程中中断的情形。

## 三、E 与 DE 的 owner 链调用

两版 owner commit 的同一修改把「遍历出生链逐个 push，然后 reverse 新段」替换为直接倒序填充预留 suffix。frequency 聚合、floor、总 occurrence checked addition、owner entry reservation 及 output 顺序均未修改。

每次 `Output::birth` 为相应 key 增加一个 occurrence，并把一个 Node 插到该 Group 的链头；因此该链恰好含 `group.occurrences` 个节点。初始 born 汇总已 checked 计算全部 outputs 的总 count，并据此为最终 posting 预留空间。对任一追加段，旧 len 加该 Group count 不超过总 reservation。

producer 依次访问链头到链尾，返回序列与原 push 循环完全相同。新 API 将这个序列写入目标 suffix 的末端到起点，直接得到原 reverse 后的结果；旧 prefix 保持不变。链尾 debug assertion 检查恰好消费 count 个 Node；原局部段和跨 output 接缝的严格递增检查继续保留。

Node 读取仍使用有界 slice 索引。即使内部链不变量损坏而访问越界，也会在返回值及该次写入之前 panic，API 不会因此发生越界 raw-pointer 写入。release 中 count/链一致性依赖现有 Group 构造协议，debug assertion 是额外诊断。

owner commit 阶段具有独占 posting 借用，producer 只读对应 route Nodes 并更新局部 head；它不访问或修改目标 allocation。两版产生的最终 posting 内容、频率、容量与顺序和各自 parent 相同。

## 四、DE 的初始 radix 安装调用

DE 的每个 retained Group 继续令 `count=end-start`，并按 count 新建 posting。source 正好是 `records[start..end]`，长度等于 count。producer 的 next 从 source.len 开始，在每次调用时先减一、再读取 source[next] 的低 32 位。

API 恰好调用 count 次，所以第 j 次调用前 next 仍大于零，最后降为零；所有 source 下标均在范围内。源按从末到首读取，目标也按从末到首写入，最终 `positions[i]` 等于原 source[i] 的 u32 position。D 已证明每组 source 的位置严格递增，因此 DE 保留相同排序结果。新 debug assertion 检查 next 最后为零，原 posting 严格递增检查仍在。

这项修改保持 code 展开、频率、floor、pruned 计数、map reservation 与 common heap 逻辑不变，没有新的记录缓冲区或分配。批量 API 的 count 检查覆盖最终 posting 容量，不把 radix record 的 code 当作 posting 元素。

## 五、容量与计时口径

原 `SmallPosting::with_capacity` 未修改：u32 posting 最多两项为 inline，三项沿用四槽最小 heap，四项及以上按请求 count 预留并记录实际 capacity。u16 对应四项 inline 及既有最小 heap。预留空间充足是本 API 的调用前提；它一次检查之后直接填充，不会扩容。

E/DE 的 owner 修改属于原 commit 计时；DE 初始安装修改属于原 `initial_posting_install_ms` 及包含它的 `initial_count_ms`。阶段边界不变，容量和 memory layout 也不变。DE 对 D 能评估两处批量写入的整版影响；E 对 B2 只覆盖 owner commit 的影响。性能结果按各自锁定版本的实际配对测量记录。

## 六、测试证据与后续修改条件

新增 API 测试源码覆盖 u32 inline/heap、旧 prefix、inline 容量拒绝、零 count 不调用 producer、容量保持、heap 中途 panic 后的旧 len 及再次填充，并覆盖 u16 的 inline 全范围值。E 和 DE 使用完全相同的测试。主任务正在并行验证 E 的 45 项库测试与 DE 的 46 项库测试及 release 构建；本次只读审查没有重跑。

若改变容量 tag、Copy 约束、producer 调用次数、目标范围或 len 发布时机，需重新证明 API 安全。若改变 Node count、链方向或 radix source 范围，则需重新复核对应调用的顺序及 reservation；旧融合和 radix 的其它证明继续按既有报告维护。
