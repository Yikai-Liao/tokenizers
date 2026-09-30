# 候选 F：局部单例出生表示独立审查

## 范围与结论

本轮只读审查 flat birth Group 的表示、owner 消费和新增统计，没有构建、执行测试或运行 benchmark。没有发现阻塞正确性或内存安全问题。新表示恢复与 D 相同的频率、出现数量和有序 posting。既有融合和 radix 的完整证明分别沿用 [FUSED_DIRECT_REVIEW.md](FUSED_DIRECT_REVIEW.md) 与 [INITIAL_RADIX_REVIEW.md](INITIAL_RADIX_REVIEW.md)。

worktree 为 `/root/code/tokenizers-worktrees/singleton-birth`，分支 `bpe/singleton-birth`，最终格式化 HEAD 为 `4a2f148a4247aa724a72d11426c14f9ac5d8c601`，优化 parent 为 D `d15c18cc07047479ddc2eacd1da1844cb9ac9358`。核对时工作树干净。报告按最终 HEAD 锁定，早期实现与中间格式化提交不作为测量依据。

| 文件（相对 worktree） | SHA256 |
|---|---|
| `tokenizers/tk-train/src/trainers/bpe/indexed/parallel.rs` | `bb64cc67a997dcea060f38647beee63670cd834e4c2e34f7f7fb608221058e3f` |
| `tokenizers/tk-train/src/trainers/bpe/indexed.rs` | `ba41392f5be7e2e9daa9fb06812bb0eac41509129e7e2be62f1c38b5747f0eb3` |

## 一、tag、键角色与状态转换

Group 以 occurrences 作为 tag，head 的含义为：

| occurrences | head | Node payload |
|---|---|---|
| 0 | NONE | 删除计数，不使用链 |
| 1 | 唯一 position | 不创建 Node |
| ≥2 | 链头 Node index | 每条出现一个 Node |

`remove` 仍仅累计 weight，不修改 occurrences/head。原 plain 协议保证删除 key 两端都是批次开始时的语料身份，出生 key 至少含本批首次激活的 replacement；reserved 输出仍为未活跃身份的单规则激活。两种 key 角色在同一批不会混合。因此默认 Group 的 0/NONE 同时正确表示纯删除，owner 仍以 occurrences 是否为零区分删除和出生。

第一次 birth checked 计算新 occurrences，直接保存 position 并增加该 Route 的单例计数。第二次 birth 先验证即将新增的两个节点的末索引，再创建「首位置、next=NONE」的节点，随后创建「当前位置、next=首节点索引」的节点。新的 head 指向当前节点，单例计数减一。第三次及以后只在既有链头前增加一个节点。

归纳可得：单例保存唯一原位置；多例链与 D 相同，按 birth 调用的反序遍历，节点数量恰好等于 occurrences。首次位置不会丢失或误当 Node index；第二次物化以后也不会再次物化。Route 的 singletons 恰好等于该 Route 最终 occurrences=1 的 Group 数，减法不会对一个非单例重复执行。

## 二、NONE、u32 极值与错误处理

position 先 checked 转为 u32，再显式拒绝 NONE，发生在创建 delta entry 之前。合法 flat 出生边含两个正跨度 token，其后仍有词末 separator，故起点最多为 `corpus.len()-3`；即使 corpus 有 `2^32` 槽，最大合法起点也是 NONE-2。新拒绝不排除任何合法出生位置。

Node 模式按 `nodes.len()+extra-1` checked 计算末索引，extra 为一或二；末索引 checked 转 u32 且必须小于 NONE。第二次出生时，首索引小于已验证的末索引，所以 `nodes.len() as u32` 不截断，也不是哨兵。两节点不足以容纳时在 push 之前返回错误。后续单节点同样不会创建 NONE 索引。

node index 零、position 零以及两种含义中相同的数值，均由 occurrences tag 区分。commit 先跳过 head=NONE，再在 occurrences=1 分支把 head 当 position；只有多例分支才把它当 Node index。Node.next 的 NONE 只表示链尾。

每个 Group 的 occurrences 现在显式 checked addition。单例不再消耗 Node，因此不能仅用 Node 数上限间接界定 occurrence 数；该新增检查保持 u32 posting count 约束。跨 outputs 的总 occurrence checked addition 也仍保留。

显式 Result 检查位于节点追加及 count/head 发布之前。后续 Vec 操作保持普通 Rust 的所有权与边界检查，没有新增 unsafe。若准备阶段 panic，局部 Outputs 随训练中断释放；它不承诺捕获任意 panic 后继续复用部分构造的 Route。合法计数范围内的 weight 累加继续采用既有非负加权总量约束。

## 三、owner 消费、排序与 floor

owner 汇总 frequency/count 的循环未改变，仍先聚合全部 producer outputs，再应用 floor 并预留最终 posting。局部单例即使 weight=0，也保留 position 和 occurrences=1；同 key 在其它 worker 的正权重可以使全局频率达标，其全部位置仍会安装。

消费单例时直接 push 唯一 position，等价于 D 的单节点链遍历和长度一的 reverse。多例链遍历及局部 reverse 与 D 相同。outputs 顺序保持原协议，posting 的局部严格递增和跨 output 接缝 debug assertion 继续存在，最终内容与顺序相同。

这项表示用于全部 flat Outputs，包括融合 prepare 和旧 Plan/AA delta 路径；它没有改变 AA 选择的出现集合或输出顺序。dictionary Outputs 的 removed/born/blocks 表示未修改。跨 worker 出生 frequency 在 owner 汇总后决定 floor，局部单例计数不参与过滤，所以没有引入按 worker 提前剪枝。

F 仍使用 D 的原 push/reverse 及 SmallPosting 实现，不包含 E/DE 的批量写入 API。预留空间、count=3 的四槽最小 heap、canonical key 和 heap 规则均沿用 D。

## 四、初始化与测量边界

`bpe/mod.rs`、alphabet、corpus、radix count、共享 weight lookup、fused_batch 和 small_posting 相对 D 的差分全部为空。`train_in_pool` 从函数开始直到 initialize 返回后的 merge 入口逐字相同，该区域 SHA256 为 `e17344d67a4c9326d26b1b35c7eff95aa3f0412cefdbc669c045f75249167b7f`。新增私有统计字段采用原 Default 初始化；初始化算法未修改。

F 同时影响准备和 commit：准备阶段的 `Output::birth` 增加 tag/count 分支并省去局部单例 Node；commit 对单例省去 Node 索引访问和 reverse。融合路径的这些 birth 工作包含在 `fused_prepare_ms`，AA/旧 Plan 路径包含在原 `delta_ms`；消费变化包含在 `commit_ms`。不能把整项收益都归给其中一个阶段。

新增节点统计遍历位于 rewrite 完成后、commit 子计时开始前，因此计入 `merge_ms`，没有计入上述 prepare/commit 子字段。

## 五、单例与节点内存统计的含义

`local_singleton_births` 按每个批次结束准备后的 Route.singletons 累加，是「producer output、canonical owner、本地 key」的单例数量。同一批的一个全局 pair 可以在多个 producer 中各为一个单例，统计又跨不同批次累计，并包含 owner floor 随后丢弃的低频出生 key。它不是全局 pair 数、全局频率为一的 pair 数或最终保留 pair 数。

在同一批内，新表示的 Node 元素数恰好等于旧表示的所有局部出生出现数减去这些局部单例数。Group 仍为 16 字节；Node 仍为 8 字节；Route 新增一个 usize 计数。实际 Vec capacity 受增长策略影响，不能把累计单例数乘八解释为 RSS 节省。

`peak_birth_node_bytes` 取每批全部 Route 的 `nodes.len()*size_of(Node)` 总和最大值；`peak_birth_node_capacity_bytes` 改用各 Vec 的 capacity。准备期间节点只追加，这个时点覆盖该批留存的 Node payload/capacity，但不包括 delta map、Route/Vec 对象、valid positions、postings、输入、corpus 或 allocator。两个字段分别描述逻辑 payload 与命名分配容量，不是进程峰值 RSS。

## 六、测试证据与复核条件

新增测试源码覆盖纯删除的 0/NONE、零权重第一次出生、第二次物化首位置、第三次链追加、链遍历内容、局部 singletons 归零、接近 NONE 的最大合法位置、NONE 拒绝前未创建 entry，以及 Group 大小。原完整差分、AA、相邻合并和跨 worker floor 测试继续保留。本轮没有执行测试，运行结果由主任务另行记录。

若改变 occurrences tag、head 哨兵、第二次物化步骤、Node/position 域、删除与出生的键角色、局部统计口径或 commit 消费分支，需要重新复核本报告。其它融合与 radix 协议按既有报告维护。
