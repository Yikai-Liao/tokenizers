# 融合合并热点独立审查

## 范围与结论

本轮仅只读审查新合并热点：`fused_batch.rs`、其在 `parallel.rs` 的门控与提交连接，以及新增统计和测试源码。没有构建、执行测试或运行 benchmark。未发现阻塞正确性问题；该实现满足 [FUSED_BATCH_PROTOCOL.md](FUSED_BATCH_PROTOCOL.md) 中邻边判定、阶段屏障、出生唯一生产者及 posting 有序的要求。本报告不重复审查此前的初始化实现。

worktree 为 `/root/code/tokenizers-worktrees/fused-direct-atomic`，HEAD 为 `c1ee201978aadd853d032621add6e056d1949a76`，父提交为 `27dc7fd0d1a9d5b0aa617f9505eb9d2843452b7c`。核对时工作树干净。以下文件与 HEAD 相符，路径相对 worktree：

| 文件 | SHA256 |
|---|---|
| `tokenizers/tk-train/src/trainers/bpe/indexed/parallel/fused_batch.rs` | `f0f705987acb187c3fa55daf76daddff5df86009f57f14cdb21c464e1877bbc3` |
| `tokenizers/tk-train/src/trainers/bpe/indexed/parallel.rs` | `03ea52f601dd83b218b68842f28e8d7e43931ee085397bb56e0c070e198bf6f5` |
| `tokenizers/tk-train/src/trainers/bpe/indexed.rs` | `227b34321cfc8d7dbd6ae2b54fd46695286417b53d8894894aa100e6359c6620` |

## 一、进入融合路径的条件

`parallel.rs:912` 只在 flat32、`C::SHARED` 且首规则非 AA 时调用新 prepare/apply。既有选择器把 AA 限为单规则批次，并拒绝随后加入 AA，所以首规则非 AA 意味着整个批次非 AA。普通 slot、多地址块及 AA 继续走原 Plan 路径；该路径的差分是移入 else 分支和格式调整。

既有 heads/tails 证书阻止一个规则的右 ID 等于另一规则的左 ID。共享 head 或共享 tail 可以保留，但在某个真实 token 位置只会匹配一条具体 pair；不同规则不能共享被消费的 token。非 AA 同一规则的两个有效出现也不重叠。因此融合路径的全部有效合并区间互不相交。

预留 canonical ID 规则在加入前、加入后分别停止其它规则的收集，仍为单规则批次。其输出 ID 此前只被预留、尚未活跃。普通输出即时分配不同的新 ID；若另一规则拼接出同一已有字符串，同样会触发 reserved 隔离。故本批 replacement 互不相同，且整批读取前没有这些活跃身份。相关首次激活和有限长度条件继续采用 [PAIR_MONOTONICITY.md](PAIR_MONOTONICITY.md) 的 plain 证明。

## 二、selected 邻居查询与 Plan 邻接等价

prepare 读取整批开始时的 corpus，不在过滤时写入。每条 selected pair 的 posting 覆盖它的全部 live 出现；有限长度门控对同一 plain pair 是常量，已入选规则执行时不按位置跳过。非 AA 无需奇偶择取。因此，当前位置的真实邻居 pair 若命中 selected 表，它的相应出现必定通过同一个 live pair 检查，并被本批选中。

对于当前 `(a,b)` 起点 p，先验证 `corpus[p]=a` 和 `corpus[p+len(a)]=b`，与原 Plan 过滤条件一致。endpoint 表示允许从 `p-1` 读取左邻 L 的末端 ID，再得到其起点 `before=p-len(L)`；`before-1` 是再前一个 token P 的末端。`selected[(P,L)]` 命中，当且仅当原空间有序 Plan 中前一选中区间结束于 p。

- 命中时当前合并跳过左侧 delta；左边合并的右侧 delta 已删除旧 `(L,a)` 并生成两个最终输出之间的边。
- 未命中时删除 `(L,a)`，按原严格长度门控生成 `(L,Z)`，位置为 before。

右邻 R 起点为 after，随后 token S 起点为 `after+len(R)`。`selected[(R,S)]` 命中，当且仅当原下一 Plan 从 after 开始；命中时使用其 replacement 作为 final_R，否则仍使用 R。当前合并删除旧 `(b,R)`，按 `len(Z)+len(final_R)<max_length` 生成 `(Z,final_R)`，位置为 p。

证书保证这些选中邻居不会因与当前区间重叠而失效。NONE 阻断词边界；读取 L/P 或 R/S 时均以真实 endpoint 与词末 separator 为边界，不跨词生成 pair。两条相邻合并间的旧边恰好删除一次，最终边也恰好出生一次。三条及以上连续合并逐边应用同一规则，所以无需全局排序来识别邻接。

权重游标每个 Task 单独从零开始；该 Task 是单规则 posting 的连续 slice，过滤保留位置顺序，所以它看到的 p 非降序。不同规则切换时重置游标，避免 rank 顺序引起地址回退。flat 的全部地址都属于 `blocks[0]`。

## 三、prepare/apply 的读写屏障

jobs 的 indexed `par_iter` collect 同步完成全部 prepare 读取和 route 生成，成功返回后才调用 apply。prepare 错误或 panic 不会进入 apply。selected 表、规则和 lengths 在读取阶段保持不变。

apply 仅使用保存的有效 u32 起点，写当前规则自己的 start、右 token start，以及右 token 长度大于一时的末端。写法与原 `write_plans` 相同。上一节证明区间互不相交，任意 job 顺序都得到相同端点状态；apply 期间没有 corpus 邻居读取。

新增共享写入仅由 Atomic Slot 的 `set_shared` 实现，普通 Slot 的 `C::SHARED=false` 在入口排除该调用。共享引用下的 stores 不创建普通可变引用别名。apply 的 `par_iter().for_each` 和外层 install 返回前完成全部写入，之后才进入 owner commit；commit 完成后才开始下一批。Relaxed slot 访问的正确性依赖这些阶段 join，不依赖某个 worker 的自然执行顺序。

## 四、取消全局排序后 posting 仍有序

jobs 不是动态取队列：输入被视为「规则 rank 顺序、该规则 posting 位置顺序」的连续流，按 total/worker chunk 切分。单个 job 可含多条规则的 slices；每条规则只能按其 posting 顺序跨越连续 jobs。indexed collect 保存 job 顺序。

每个出生 key 只有一个规则生产：

- `(旧 ID,Z)` 只能由输出 Z 的规则的左侧生成；旧 corpus 不含本批新 ID。
- `(Z,旧 ID)` 只能由输出 Z 的规则的右侧生成。
- `(Z1,Z2)` 只能由输出 Z1 的左侧规则在自己的右边生成；输出 Z2 的规则发现选中的左邻后跳过左边生成。

不同 replacement 不能相同；因此共享 head/tail 不会让两个规则产生同一出生 key。单规则 reserved 情况也满足同样条件。

对固定 key，右出生位置 p 随该规则 posting 严格递增。左出生位置 before 为 `p-len(L)`，固定 key 的 L 相同，所以也严格递增。对应旧 posting 的无重复不变量继续保留。由此，每个 route 链的该 key 记录是按递增位置插入链头，遍历链后局部 reverse 恢复递增顺序；按 jobs 顺序追加这些链段得到全局递增 posting。

owner commit 逻辑除新增 debug assertion 外未改变。`parallel.rs:1193` 检查本次追加段及其与上一段的接缝严格递增，覆盖局部顺序、跨 job 接缝和重复位置。这个检查在 release 中不执行；有序性来自上述唯一生产者与连续分段条件。

## 五、计数聚合与临时存储

prepare 继续调用原 `Output::remove/birth`。每条旧边的删除计数与原 Plan delta 相同，出生记录只描述整批结束后的最终邻边。出生 key 含新激活 ID，删除 key 只含整批之前的 ID，所以同一个 Group 不会混用删除和出生语义。

owner 在全部 outputs 上先汇总出生频率和 occurrence 数，再应用 floor 和分配 posting；不会把某个 job 的低频部分提前丢弃。旧 key 只有负变化，扣减到 floor 以下后的缺失记录仍按原逻辑忽略。head/node 索引与 posting 总数的既有检查、非负权重和既有全局计数上界没有改变。

融合路径只保留 u32 有效起点，按 job/rank 分组，用于后续 apply；没有生成全局 16 字节 Plan 数组，也没有进行混合规则的全局位置排序。有效起点容量仍受 Vec 增长策略影响。`peak_valid_start_bytes` 只统计这些 u32 Vec 的实际 capacities，未包括分组目录、Task、selected 表、route map/node 或仍存活的源 posting，不能解释成 prepare 的全部峰值内存。

`fused_prepare_ms` 覆盖过滤和 delta/出生共同遍历，同一个 elapsed 同时加进 `delta_ms`；两字段不能相加。融合批次不记入原 `plan_ms`。`rewrite_ms` 只覆盖 apply，owner commit 与后续 route 的统计继续使用原阶段。

## 六、测试证据与后续改动条件

新增热点测试按 1/4 worker 对比独立 greedy 的逐轮 trace、vocab 和 merges，包含连续合并、共享 head/tail、重复长串、Unicode、reserved 输出以及 None/3/7 的长度限制，并断言实际进入融合批次。1500 case HF 差分的公共 check 新增 Atomic32 路径，比较 trace 和完整模型内容。既有跨 worker 出生 floor 测试同时覆盖 flat32+atomic，继续核对聚合后门控。主任务报告 43 项库测试通过；本轮仅核对源码，没有独立执行。

若改变 selected 邻居判断、允许部分 live 出现执行、取消 reserved 隔离、放宽 head/tail 或 AA 条件、改为无序任务归并、让多个规则共享 replacement，或让 read/apply/commit 重叠，需要重新审查相应推导。当前结论仅覆盖本报告锁定的合并源码；大语料签名与性能结果另按实际测量记录。

## 七、B2 lookup 差分独立复核

本次仅只读复核 B2 相对上述 B 的 lookup 差分，不重证初始化、出生或 apply 协议，没有构建或执行测试。结论：权重目录与 selected 的直接查询等价，没有发现阻塞问题；前六节的 birth/apply/commit 推导继续适用。

worktree 为 `/root/code/tokenizers-worktrees/fused-lookup`，HEAD 为 `a0832c488a7ece429630c7d1493da5dd772c87aa`，父提交为 `c1ee201978aadd853d032621add6e056d1949a76`。核对时工作树干净。以下为 B2 的源码 SHA256：

| 文件（相对 worktree） | SHA256 |
|---|---|
| `tokenizers/tk-train/src/trainers/bpe/indexed/parallel/fused_batch.rs` | `d78c7f5f0d6ab02bc04afac773e13c5531ce58db6394a29bac73c4556107ad21` |
| `tokenizers/tk-train/src/trainers/bpe/indexed/parallel.rs` | `5628bab84261cbe72d966eb3cff3f6f41dac709efe29bbfab4e7463f3da0592e` |
| `tokenizers/tk-train/src/trainers/bpe/indexed.rs` | `cb944820743836f06625b60375f7fb9c5d7f1c034a749f5a72446044bc160518` |

### 权重目录的边界与等价性

`WeightLookup::new` 对每个 256-slot 桶记录「严格小于桶起点的 pivot 数」，末项按 slots 截断。查询 p 所在桶 b 时，start 为小于 `256*b` 的 pivot 数，end 为小于 `min(256*(b+1),slots)` 的 pivot 数。

由于 `256*b <= p < min(256*(b+1),slots)`，全部 `q <= p` 的 pivots 恰好等于 start 个前缀，加上 `pivots[start..end]` 中满足 `q <= p` 的数量。因此局部 `partition_point` 加 start 与原 `Block::weight` 全数组查询相同。等于桶边界的 pivot 属于新桶，等于 p 时计入当前权重；空桶仍从 start 的前一 pivot 取权重；第一个 pivot 之前和空 pivots 均返回 `previous_weight`。

目录只在 flat32、共享 Slot、非 uniform 配置下建立，block base 为零，slots 最多为 `2^32`。任意实际查询 `p < slots` 都有 b 和 b+1 两项。每词最多一个互异 pivot，且起始 sentinel 独占槽零，所以 pivot 数最多 slots-1，u32 转换有效。桶号乘 256 的最大值为 `2^32`，在既有 64-bit usize 前提内安全。目录长度为 `ceil(slots/256)+1`，即使最后一个桶不满也包括右界项。

merge 不改变词起点、pivots 或对应 weights，目录在初始化结束后构建一次即可覆盖全部批次。uniform 直接返回常数；没有目录时仍使用原 weight cursor。

### selected 的唯一与重复 head/tail

head 表的唯一项编码 `(tail,replacement)`，tail 表的唯一项编码 `(head,replacement)`。EMPTY 和 MULTIPLE 的高 32 位都为 NONE；合法规则输入 ID 均不为 NONE，所以合法编码不会碰撞这两个哨兵。

第一次遇到某 head/tail 保存直接项，第二次及以后始终标为 MULTIPLE。第二遍将所有具有重复 head **或**重复 tail 的规则放入 pair map，包含首次出现的规则。因此：

- 唯一 tail 直接比较前一 token 是否等于编码 head，等价于原 `(previous,prior)` membership。
- 唯一 head 直接比较后一 token 是否等于编码 tail，匹配时返回编码 replacement，等价于原 `(next,following)` lookup。
- 重复 tail/head 查询完整 pair map；额外存入的另一方向重复项不影响直接查询。
- EMPTY 可提前返回；previous 或 following 为 NONE 时，直接比较不匹配，fallback map 也没有 separator pair，保持原分隔符行为。

表长使用 `lengths.len()`，包含旧 IDs、本批 replacement 及预留 IDs。prior/next 已由调用者排除 NONE，规则输入与实际 token 均在该 ID 域内；新查询没有引入域外索引。

### 大小、生命周期与测试证据

`weight_lookup_bytes` 为目录 capacity×4；请求容量为 `4*(ceil(slots/256)+1)` 字节，约为 slot 数的 1/64，而不是逐 slot 的完整权重数组。它在整个 merge 阶段保留，不加入此前的 initial core 字节合计。构建位于 merge 起始计时之后，`weight_lookup_build_ms` 已包含在 `merge_ms` 中，不能再重复相加。

selected 每批分配两个 u64 表，容量合计按 8 字节计；fallback pair map 用既有 `table_bytes(...,16)` 估算 bucket/control 大小。`peak_selected_lookup_bytes` 记录两表容量加该 map 估算，不含对象、线程栈或其它 prepare 数据。selected 在 prepare 返回前销毁，apply 仅保存它的字节统计；权重目录则持续到训练函数返回，累计 merge 计时的截点不含返回时释放目录的开销。

新增测试源码逐一检查 p=0..9999 的目录查询与 `Block::weight`，覆盖桶边界前后、等于 pivot、稀疏空桶及最后不满的桶，并另查空 pivots。原 B 的相邻合并、共享 head/tail、reserved、长度限制和 1500 case 差分测试继续保留。主任务报告 44 项库测试通过；本次复核没有重跑。

改变桶界的 `<`/查询的 `<=`、使 pivots 在 merge 中变化、扩大 flat 地址域，或改变哨兵编码与 duplicate fallback 收集条件时，需要重新复核本附节。
