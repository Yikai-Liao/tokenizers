# 位置索引的初始化内存

本文只核算 **初始化 pair 计数、低频过滤和候选堆构造完成，第一次 merge 尚未开始** 的状态。公式来自源码类型和分配容量，运行结果另见并行报告。HF 输入字符串与词表另列，不混入位置数组。

## N 与 E

- **N**：实际 corpus 槽数，包含片段间分隔槽及开头的分隔槽。
- **E**：初始化相邻 pair 的 occurrence 数，未加词频权重，未做低频过滤；不是不同 pair 数。
- W：去重后的片段数，包含空片段；K：保留字符后非空的片段数。
- 因此保留字符数为 `N−W−1`，`E=N−W−1−K`。

例如 `#abc#de#` 有 N=8、E=3：`ab`、`bc`、`de`。同一片段权重 100 不会复制 100 份位置。

Nc、Cv、Cp、Cw 分别是 corpus、ID 长度、词起点、权重的**分配容量**。Nc 当前按 alphabet filtering 前的 Unicode 字符数加 W+1 预留，过滤后可能大于 N。P 是堆分配 posting 的总容量；H 是候选堆容量；B 是哈希表的实际桶数。

## 已有串行 Endpoints

| 组件 | 初始化 payload 字节 |
|---|---:|
| corpus，u32 ID | 4Nc |
| 每 ID 长度，u32 | 4Cv |
| 每词起点，u32 | 4Cp |
| 每词权重，u64 | 8Cw |
| pair 表：key 8 + frequency 8 + SmallPosting 16 | 32B，加 B 控制字节及常数尾部 |
| posting 的堆数组 | 4P |
| 候选堆：key 8 + frequency 8 | 16H |
| 出生表、出生 arena、merge 记录、批次 Plan | 此时没有堆 payload |

所以核心初始化 payload 近似为：

```text
4Nc + 4Cv + 4Cp + 8Cw + 33B + 4P + 16H + 常数头部
```

`4N+4E` 只是忽略容量、过滤、内联位置和元数据的两项直觉，不能当作完整内存公式。

SmallPosting 为 16 字节，已经包含两个内联 u32 位置。0、1、2 项不分配位置数组；3 项以上才计堆数组容量。不能再对全部 E 项各加 4 字节，否则会重复计算内联项。低频过滤释放 posting，但 `HashMap::remove/retain` 不缩小桶数组，B 仍可能对应过滤前的所有 key。

## 新并行内核

新内核默认使用普通整数语料数组；显式 atomic_corpus 对照开关改用同宽的 AtomicU16/U32，仍执行同一规划和切片写入算法。读取规划和独占切片写入分阶段执行，没有常驻每位置 prev/next/word_id 数组。

### 词表与位置采用两个独立宽度

- **语料 ID**：完整可能 ID 域小于 65535 时，每槽 u16，65535 表示分隔符；否则每槽 u32。构造前一次选定，不先创建 u32 再转换。
- **posting 地址**：按空间地址块表示。块基址是 64 位目标上的 usize；局部列表为 u32 或 u16。u16 地址允许 0..65535 全部取值，没有分隔码。
- **长度表**：新内核每 ID 用 usize，在本机为 8 字节。它使全局地址和 token 跨度不受 u32 限制，代价是 O(V)，不是追加 4N 或 4E。
- **词权重**：词起点也存块内 u32 偏移，权重 u64。跨块的长词由块的 previous_weight 延续。所有片段权重相同时，采用一个常量，不分配词起点和权重数组。

### 一个 u32 地址块能装下整个 corpus

此时直接使用 owner 的 `pair → {frequency, SmallPosting<position>}`，没有第二份 pair 字典和块目录。

```text
语料：2Nc 或 4Nc
长度：8Cv
词元数据：4ΣCp + 8ΣCw，统一权重时为 0
owner pair 表：约 33ΣBowner
posting 堆数组：4P
候选堆：16ΣH
另加 owner/块描述结构的固定头部
```

owner 是由 pair key 确定的唯一频率和 posting 所有者；跨 owner 比较候选时保留 HF 的频率和 canonical ID 排序。

### 多地址块的字典方案

每块保存 `base + dict(pair → local posting)`。全局 owner 保存该 pair 的唯一频率和出现块目录。

| 组件 | 初始化 payload 字节 |
|---|---:|
| 语料 ID | 2Nc 或 4Nc |
| 长度表 | 8Cv |
| 词元数据 | 4ΣCp + 8ΣCw；统一权重为 0 |
| 全局 pair 表：key 8 + frequency 8 + 块目录 16 | 约 33ΣBowner |
| 块字典：key 8 + 局部 SmallPosting 16 | 约 25ΣBblock |
| 局部 posting 堆数组 | u32 为 4P，u16 为 2P |
| 全局块目录的堆数组 | 4Pd |
| 全局候选堆 | 16ΣH |
| 块描述数组 | `capacity × sizeof(Block)`，包含基址、map/Vec 头部及跨块权重 |

u16 SmallPosting 同样是 16 字节，但内联四个位置。它不会把每个字典项的 16 字节头缩半；优势来自更窄的堆 payload 和更多内联位置。空间收益必须抵扣重复 pair 字典、块目录及过滤后保留的桶容量。

当前 u16 地址块为 65536 槽，u32 地址块为 2³² 槽。全局块编号是 u32，因此 u16 分块可覆盖接近 2⁴⁸ 个槽；u32 分块可覆盖接近 2⁶⁴ 个槽，实际还受 usize、Vec 分配和系统内存限制。位置通过 `base + offset` 恢复，不截断到 u32。已验证超过 2³² 的地址算术和跨块训练；没有分配数十 GB 语料做端到端容量试验。

## 初始化指标如何核算

`IndexedTrainingStats` 的 `initial_*_bytes` 在这个指定时点记录。核心合计为：

```text
initial_corpus_bytes
+ initial_pair_table_bytes
+ initial_posting_bytes
+ initial_block_table_bytes
+ initial_directory_bytes
+ initial_heap_bytes
```

其中 corpus 项进一步分为 slot、length、weight 三项，避免把长度和词元数据称为每字符载荷。表字节从 usable capacity 反推桶数，加 key/value、控制区及尾组；不包含分配器 header、碎片或线程栈，因此这是源码布局估算，不是 RSS 或进程实测峰值。owner 控制结构的常数头部另计。

## HF 适配层另算

1. 借用的 `BpeTrainer.words` 保留去重 UTF-8 输入：`AHashMap<CompactString,u64>` 每桶约 32 字节，加控制区及长字符串堆容量。短字符串已内联在 key 中，不能再加一份全部输入字节。
2. 正向 canonical 词表每桶约 32 字节，加控制区及长字符串分配。
3. 反向词表 `Vec<CompactString>` 每容量槽 24 字节，加长字符串分配；按目标词表预留的容量不能当作当前 token 数。
4. 此时最终模型尚未构造，不计输出 String map 和字符串 merge 列表。
5. 非空 affix 仍走历史 cohort 引擎；发生跨度别名后可能追加 4Nc 的 occurrence span 表。它尚未迁入本次并行分块路径。

## 与 efficient_bpe 原型的关系

主并行原型的端点布局、16 字节 SmallPosting、唯一 owner、精确批次和分阶段更新已经用于这条路径。迁移后用独占切片替代共享原子语料，增加 HF canonical ID 和预留 ID 处理；初始化先从 UTF-8 构造选定宽度数组，再并行路由位置和按 owner 计数、剪枝、建堆。临时初始化位置路由在本文的指定统计时点已释放，其与 posting 同时存在的峰值需要另估。

原型 ablation 中的 Halfword、H2.5、H3 是另外的串行存储后端，本次没有宣称它们已全部接入。Halfword 可以用两个旧 u16 槽容纳完整 u32 新 ID，并通过位图/跳距得到实际跨度；其约 2.1875N 的载荷不要求最终词表小于 65535。并行迁移还需按物理 bitmap word 分配写所有权，本轮先落地有明确 ID 域证书的直接 u16 路径。

本篇的容量口径只适用于第一次 merge 前。批次 Plan、路由结果、新 posting 与后续 heap 的瞬时占用属于另一个时点，不能拼入这张初始化表。
