# 普通快速引擎向前后缀训练的完整迁移

## 本轮范围

本轮先迁移现有普通路径中可以复用的机制，再冻结源码做一次统一消融。上一版 `8454618f` 只迁移了通用 cohort 的数据结构与局部执行机制；它的 EN16 prefix / ZH16 suffix 结果分别约为同输入 NONE 的 1.92 / 2.37 倍。当时尚未使用批次、低频淘汰、owner 队列窗口和融合提交，因此那些数字不能用于判断这些机制在 affix 下的收益。

全集在 `438fbca3` 冻结。38 项主矩阵及 3 项 ZH512 对照完成后，选择关闭 `weight_lookup` 的组合，默认值提交为 `fd300ca8`；机制仍保留。选定配置另通过 4 项 guarded 差分测试，包含全开、逐项关闭、全关和选定默认值。完整结果见 [AFFIX_ALL_FAST_V4_SUMMARY.md](AFFIX_ALL_FAST_V4_SUMMARY.md)。

本轮让所有非空 prefix / suffix 使用现有 `parallel` 快速引擎。每次选择 replacement 都检查它是否曾在语料中激活；若发现已激活 ID 复用，在消费候选和该批次改写之前停止，释放试探路径，再用原始加权词完整重建 HF cohorts。这保留了快速引擎的全部机制，也保留了实际 alias 下可观察的历史账本与 cohort。

## 已接入的机制

| 机制 | 受检查保护的快速路径 | 发生 ID 复用后的通用路径 |
|---|---|---|
| 低频 pair 淘汰 | 完整轮次聚合后，永久淘汰低于阈值的旧 pair | 从原始词重建包含历史账本的路径 |
| 多规则认证批次 | 复用头尾冲突、AA、出生优先级和预留 ID 单规则边界 | 保持 HF 单规则及历史 cohort 顺序 |
| owner 分片与队列窗口 | 复用 owner 账本、8 叉堆、Bulk4 预取、未消费项恢复 | 历史 cohort 队列；保持同优先级 cohort 的顺序 |
| 紧凑候选优先级 | ID 和频率范围允许时，使用 8 字节 heap candidate | 使用 24 字节带 SmallPosting 的候选；复用 ID 前恢复 32 字节宽候选 |
| 融合 prepare 和提交 | flat 非 AA 原子槽位复用一遍读、邻居聚合、出生链及直接提交 | 保留中间出生；批次提交的历史由重建恢复 |
| 邻居 dense 聚合 | 复用融合路径目录；宽 ID 域保留 hash 路径 | 并行大 cohort 和串行尾部聚合 |
| scratch 容量复用 | 沿用现有快速引擎的临时数据生命周期 | 串行尾部缓存最多 4 MiB 的目录和出生链 |
| Posting 分配器 | AutoArena / System 与独立 session | 同样接入 AutoArena / System 与独立 session |
| 窄语料与原子改写 | u16 / u32、AtomicU16 / AtomicU32；最终槽位一次分配并按互不重叠词段并行填充 | 泛型 u16 / u32 cohort worker；原始词边界用于历史扫描 |
| 空间 posting 字典 | 复用 u16 block 局部 offset 和 u32 offset 布局，跨 block AA 仍用原协议 | 历史 cohort 使用全局 u32 位置 |
| 初始直接路由及 radix 分组 | 复用原始 owner 直接路由、稳定排序与分组 | 稳定分组保留全部 HF 初始 cohort |
| 有界多 block 初始化 | 复用现有按 wave 处理的 block 初始化 | 通用队列仍按自身所有权表示历史 cohort |
| 权重排序、区间压缩和目录查询 | ID 分配完成后安排物理词段；相同权重区间及直接 weight lookup | 压缩权重查询与完整词边界分别保存 |
| 并行字母表 | 复用 Unicode presence bitmap；有限字母表沿用 HF 原选择器 | 同样复用；单线程本地入口仍用原选择器 |
| 字符装饰缓存与并行容量统计 | 原始遍历顺序分配装饰 ID，随后并行填入最终槽位 | 使用同一份初始化机制 |

u16 语料需要初始实际 ID 数和目标词表大小都不超过 65535；65535 是窄分隔符。排序只改变物理词段位置，装饰 ID 仍按 HF 原始加权词的遍历顺序分配。初始 token 的跨度为一个保留字符，不能拿带 prefix / suffix 的字符串长度替代它。

生产代码使用内部显式 Options。环境变量仅由孤立 benchmark 源码添加。共有 16 个独立开关：`sort_weights`、`narrow_corpus`、`weight_lookup`、`initial_grouped`、`parallel_apply`、`grouped_tail`、`scratch_cache`、`character_cache`、`packed_queue`、`parallel_measure`、`guarded_fast`、`parallel_alphabet`、`arena_allocator`、`batch_execution`、`queue_prefetch`、`fused_batch`。开关的执行范围见表；例如 scratch_cache 用于通用尾部，fused_batch 用于原子非 AA 批次；后续 block 实现扩展到多 block 字典，见 [BLOCK_FUSED_PREPARE_REPORT.md](BLOCK_FUSED_PREPARE_REPORT.md)。

## 为什么运行时检查允许批次和剪枝

初始装饰可能让不同字符位置使用同一身份，但每个保留位置的跨度仍为 1。在遇到第一次已激活 ID 复用之前，每条接受的规则都首次激活 replacement。预留 special ID 的长度为 0，第一次产生它允许激活；它继续使用原来的单规则批次处理。

首次激活意味着本轮新边包含此前不存在的身份。因此，尚未选择的旧 pair 只减少出现；HF 账本等于它的真实加权次数，其候选覆盖所有剩余出现。已选择 pair 的残留 HF 账本不会再次获得出生项。有限长度条件按初始字符跨度计算，检查期间同一身份的跨度固定，所以门控对同一 pair 的所有出现保持一致。初始 pair 的长度例外保留。

批次仍使用现有证书：排除 AA 及头尾冲突，保留出生优先级边界，预留 ID 首次激活独立成批。replacement 激活检查也覆盖同批中较早选择的规则：选择时立即设置其跨度，较后规则如果得到同一 ID，就触发停止。融合提交中的新 pair 仍有唯一生产规则，位置按规则 posting 顺序生成；原 prepare / apply / commit 屏障保持不变。

若第一次复用在批次选择中出现，该规则尚未消费、该批次尚未写入。试探过程只修改它自己的语料副本、目录和队列，原始词与权重始终不变。返回前所有 posting owner 销毁，session 结束，再进入完整通用训练。重建会恢复之前因剪枝和批次融合未保存的每轮中间出生、历史 cohort 与队列初始频率。因此最终模型由完整 HF 路径产生，而不是在丢失历史的试探账本上继续。

这项检查不按 affix 字面值选择白名单。没有实际 ID 复用的输入完整使用快速引擎；实际复用的输入会支付试探及完整重建的成本。复用越晚，额外工作可能越大，必须在完整训练耗时中体现。

## 核验

本轮全库 84 个测试通过，耗时 102.03 秒。新增检查包括：

- 四类 prefix / suffix 配置的完整 HF merge trace、词表和 merges 差分，覆盖全部启用、16 项分别关闭及全部关闭。
- 首条规则触发 alias，以及多轮已提交后才触发 alias 的重建；一和四个 worker 均核对最终完整结果。
- 首次激活预留 ID、有限长度、AA、零权重、空词、窄宽语料、原子及普通槽位、两种 posting 布局。
- 超 u32 频率与检查后的 i64 cohort 账本差分。原 HF 使用 i32 账本，这个扩展范围不能用其溢出结果作 oracle。
- 最终分配的四种槽位、三种词段顺序、过滤和空语料、跨 block 权重、递归并行填充；槽位容量等于有效长度。
- 正常返回和 alias 返回的 heap requested / freed、arena requested / retired 计数相等。两个训练 session 分别检查。

初次试验还修正了关闭权重目录时 radix 分组必须从 block 元数据查询权重的问题；最终开关差分和全库检查均包含该路径。

## 统一消融与计时口径

主矩阵为 EN16 prefix `##`、ZH16 suffix `</w>`，四个 merge worker、四个初始化 worker、词表 30000、最小频率 2。每种输入一次全开、16 项分别关闭及全部关闭，共 36 次 affix 运行，另加同冻结源码的两个 NONE 对照，共 38 次。完整模型 SHA 与已有 HF 对照一致后，以完整训练耗时选择组合，再做代表性 ZH512 suffix 和 NONE512 对照。运行次数按实际完成记录。

关闭 `parallel_apply` 的快速路径控制保留四个初始化 worker，使用一个 merge worker；owner 数也随 merge worker 数减少。它是整段 merge 串行诊断，不能把差值完全归因于 rewrite 内核，也不属于同四 worker 的生产组合选型。请求线程数和实际阶段 worker 数分别记录。

`alias_guarded` 表示执行了运行时检查，`alias_fallback` 表示发生了实际 ID 复用并重建。快速路径中的 `monotone_pairs=true` 描述检查保护的执行阶段，不是对所有 affix 的静态结论。

回退后的 `initialize_ms`、`merge_ms` 描述通用重建；`speculative_initialize_ms`、`speculative_merge_ms`、`speculative_total_ms` 单列试探成本。`speculative_selected_merges` 包含尚未应用的选择，`speculative_applied_merges` 只计完成提交的规则。原生 `train_ms` 包含完整调用，包括两段训练和释放，不能只拿回退后的 merge_ms 排名。

`initial_slot_bytes` 在本版统一为槽位分配总字节，`corpus_slot_bytes` 是每槽 2 / 4 字节；旧版本报表需按其字段口径转换。排序嵌套在布局和初始化阶段。容量统计不能替代 RSS；arena backing 统计记录 session 结束时的分配容量，session 随后释放，不代表训练返回后仍驻留这些 buffer。

用户指定的多 block 融合 prepare 已在本轮选定默认值之后实施，采用版本为 `1959202f`。实现、测量和范围见 [BLOCK_FUSED_PREPARE_REPORT.md](BLOCK_FUSED_PREPARE_REPORT.md)；本表的 affix v4 消融数据保持为 block 实施之前的冻结结果。
