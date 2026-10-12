# Positions Buffer 与 SIMD 编码实验方案

日期：2026 年 10 月 12 日

## 结论与实施顺序

建议并行准备四条主线：当前直接 append、统一 Buffer 后冻结且保持 LEB128、同一后端的 D1 bitpacking、同一后端的 FoR bitpacking。四个独立 worktree 中 A 保留原样，B/C/D 并行实现；分析时分别回答“恢复 Buffer 是否值得”和“更换编码是否值得”，不要求按结果逐个串行筛选后才开工。目录只增加一个聚焦对照，论文中的其他结构保留为设计依据，不全部实现，也不展开 codec × 目录 × 容器的交叉实验。

推荐直接恢复历史 `SmallVec<[u64; 2]>` 原始 Buffer → 一次编码 → 不可变叶子的生命周期，需要堆存储的叶子由 Box 持有。Buffer 就是旧代码的小缓冲，不再设计新容器或重新调整 inline 容量。普通新产出片段在 Worker 内完成 Buffer、编码和冻结，Owner 接收已冻结叶子并拼接描述符，不把这条路径重新变成 Owner 收 raw 数组再串行编码。reuse 所需的有序归并保留，实际归并/重编码成本单列。保留非递归 root/leaf 拼接和借用读取任务。完整块采用 128 个位置；短列表和不足 128 的尾块遵循统一规则。32B 是需要验证的目标，不能靠 unsafe 布局、截断值域或隐藏额外分配达成。

本文是实验计划，不是性能结论。本次只阅读源码与资料，没有修改仓库、编译、执行测试或跑 benchmark。此前用户报告严重回退的 scratch、producer-complete publisher、FRONTIER 属于另一组对照，不据此推断本轮编码器的性能。用户确认 FastLanes 已有历史实验；本轮未定位到具体变体与原始指标，不将其列为未经测试的新候选，也不安排复跑。

## 1 基线与已经核对的事实

目标分支为 `simplify/bpe-flat-20261010`，本轮 baseline 固定为 `54c9461e35b43f9735a780b11498780a40aea7bc`。该提交已核对为报告变更，生产代码未变；本文逐项核对的 Positions 源码锚点为其父提交 `890ea1f58e65f8d8503f7cd3fc97ad1651eaf284`。若实施时分支再次前进，先核对差异并重新固定 SHA，不把不同基线混在一张结果表。

该源码使用 `SmallVec<[u8; 16]>` 保存叶子字节，另有长度；叶子开头保存最后值，随后是几何增长的 restart offset 目录、每块绝对 seed 和 LEB128 gap。目录扩容时 `copy_within` 移动已编码数据。root 持有 `Box<[Leaf]>`，concat 移动叶子所有权，不重编码堆 payload。[固定版本源码](https://github.com/Yikai-Liao/tokenizers/blob/890ea1f58e65f8d8503f7cd3fc97ad1651eaf284/tokenizers/tk-train/src/trainers/bpe/positions.rs)

小值情况必须据实描述：源码中的 64 位目标测试断言 `size_of::<Positions>() == 32`；空叶子 0 字节，单值 16 字节且 inline；两个、三个相同值分别 17、18 字节并 spill。当前并不存在“两项仍然 inline”的保证。这些是已读到的实现与测试断言，本次未执行验证。恢复 Buffer 后，最终空/单值的小表示应保留；构建阶段若新增分配，也必须记账。

历史 Buffer 已核对到 `02620e0ad37d8b165b5923902a05184e483cbd88`：初始计数的 dense/sparse `State` 与 Worker 邻居聚合的 `Change` 均使用 `SmallVec<[u64; 2]>`，通过 `SmallVec::new()` 和 `push` 收集原始坐标，再交给 `Positions::from_sorted(&values)` 编码。此次直接取回这些积累与冻结代码，按当前接口做必要衔接；B/C/D 使用相同的两项 inline 容量，不另开容器或容量实验。[历史初始计数](https://github.com/Yikai-Liao/tokenizers/blob/02620e0ad37d8b165b5923902a05184e483cbd88/tokenizers/tk-train/src/trainers/bpe/index.rs#L241-L320) · [历史 Worker 聚合](https://github.com/Yikai-Liao/tokenizers/blob/02620e0ad37d8b165b5923902a05184e483cbd88/tokenizers/tk-train/src/trainers/bpe/merge.rs#L194-L275) · [历史编码入口](https://github.com/Yikai-Liao/tokenizers/blob/02620e0ad37d8b165b5923902a05184e483cbd88/tokenizers/tk-train/src/trainers/bpe/positions.rs#L39-L43)

这里恢复的是 raw `SmallVec<[u64; 2]>` 缓冲路线，不把旧版 Complete/Partial 协议、producer-complete 判定或重编码式 concat 一起恢复。当前 append 叶子的 `SmallVec<[u8; 16]>` 与历史原始 u64 Buffer 是不同层次的存储，不能混用它们的小值分配结论。

所有变体必须支持非递减完整 u64 坐标，包括重复、零、`u64::MAX`、跨越 `2^32` 的绝对值和 gap。禁止用 gap−1 排除重复，禁止坐标哨兵占用有效值。坐标宽度、元素数量、字节 offset 的可表示范围分别检查，不把“语料通常较小”当成格式前提。

## 2 最小实验矩阵

| 标识 | 生命周期与存储 | 完整块编码 | 要回答的问题 |
| --- | --- | --- | --- |
| A 当前基线 | 当前直接 append 与 SmallVec，原目录不改 | 现有 seed + D1 LEB128 | 当前真实成本是多少 |
| B 生命周期控制 | 恢复旧 `SmallVec<[u64; 2]>` → 一次编码 → 冻结，一次写出精确 flat directory | 与 A 相同的 seed + D1 LEB128 | 去掉在线编码与增长目录搬移，收益能否覆盖 Buffer 峰值 |
| C D1 bitpacking | 与 B 相同 | u64 D1 gap → 双 u32 plane → BitPacker4x | D1 固定位宽压缩的实际收益与代价 |
| D FoR bitpacking | 与 B 相同 | u64 块内 offset → 双 u32 plane → BitPacker4x | 用户提出的 128-position FoR 是否改善总耗时 |
| E 目录支线 | 仅在一个固定 codec 上，与 eager flat 对照 | 不再增加 codec | 选中后才建索引是否减少总成本 |

A→B 同时改变构建生命周期和“已知最终长度”的布局，不能仅凭这个对比把收益全部归因于 memmove。通过搬移量、编码时间、分配和峰值内存分项说明原因。C 与 D 使用相同后端、分块、目录、Buffer、冻结方式和尾块规则，尽量隔离 D1 与 FoR 的差别。

E 先以 B 的 LEB codec 为固定载荷开展小规模对照；仅当真实选中率和目录成本支持它时纳入全量训练。StreamVByte、自写 SIMD LEB、FastLanes、Elias–Fano 不自动追加为主变体。已有历史实验先复核，避免重复实现已被实测否定的方向。

## 3 统一生命周期与维护边界

普通新片段的概念流程只有以下三步；名称不是必须新增的公开 API，也不要求再引入一套状态机：

```text
buffer.push(position)                 // Worker 内，普通新片段写原始 u64
leaf = encode_and_freeze(buffer)       // Worker 内完成编码和冻结
positions = concat(frozen_leaves)      // Owner 只移动叶子，不重新编码
```

初始计数和后续 neighbor births 使用恢复后的 `SmallVec<[u64; 2]>` 及同一冻结接口；不恢复 Complete/Partial 分类、不根据生产者身份走压缩捷径。逐一核对初始计数、birth 和 reuse 的调用点与线程归属，不假定所有路径当前都在 Worker。reuse 的片段可能交错，所需 k-way merge 不能直接替换成 concat；沿用其正确性路径并单列重编码成本。B/C/D 都沿用历史 SmallVec 的构建和增长方式，不增加新 Buffer 抽象、allocator、泛型容器框架或容量调参。尾块处理是编码规则，不是生产者特例。

Worker-local 编码相对历史 Owner 编码路线有减少串行瓶颈的潜力，但当前 A 的普通新片段已经在 Worker append 时编码，不能把“移到 Worker”的收益重复算成 B/C/D 相对 A 的新收益。这里固定这条责任边界，比较的是 Worker 内逐条编码与批量冻结的代价。

已知最终个数后，目录一次布局并写出，不再通过扩展前缀搬移既有 payload。先保留 B 与 A 相同的必要 seed、tail cache 和搜索语义，避免同时混入不相关压缩技巧。一次逻辑编码可以包含长度预计算，但若为精确分配多扫一遍，必须计时；`into_boxed_slice` 的 shrink、复制或重分配同样不能藏在计时区间外。

concat 保留每个叶子的真实 count、restart 边界和不足 128 的尾块。两个短尾块不能在“零重编码拼接”名义下合成完整块。零个非空叶子返回空；仅一个非空叶子时直接移动返回该 Leaf，不套 Root，也不分配 Root 数组；至少两个非空叶子才生成 flat Root。保留 root 不可继续 append、当前 concat 输入只接受 leaf 的约束。确实需要 root 时，目录和移动 L 个叶子描述符的 O(L) 成本仍然存在。

32B 只约束最终 handle 的目标；同时报告 Buffer、Leaf、Root、ReadTask、搜索视图的真实大小与额外堆存储。可以接受为了清晰结构而未达 32B 的实验结果，但不能以 unsafe、指针打标、隐含 32 位上限或未计入的 side allocation 宣称达标。

## 4 D1 与 FoR 的公平实现

对一个完整、有序的 128-position 块，保存 `base = x[0]`。D1 定义 `d[0]=0`、`d[i]=x[i]−x[i−1]`；FoR 定义 `r[i]=x[i]−base`。D1 的位宽来自最大 gap，FoR 的位宽直接来自 `x[127]−base`。两者都编码 128 个 code，包括首个已知零，避免一方偷省槽位。

使用 BitPacker4x 的普通 `compress`，统一将 u64 code 拆成低、高 u32 plane：低 plane 位宽 `min(w,32)`，高 plane 位宽 `max(w−32,0)`；零宽 plane 无 payload。解码后先按 `(u64(high)<<32) | low` 合成 u64。FoR 再逐项加 base；D1 在完整 u64 gap 上作正确前缀和，不能分别对两个 plane 求和而丢掉进位。

这是本方案提出的 full-u64 adapter，不是库现成提供的 full-u64 API。`w=0` 和 `w=64` 都有明确路径，避免 shift-by-64；宽度不能小于真实 code 位宽。完整块的 plane payload 为 `16*w` 字节，另计 seed、width、count 与目录。宽度 64 时 payload 已达原始 1024 字节，再加 header 会超过 raw 大小。两个 128×u32 临时 plane 合计 1024 字节，窄值路径可少用一个；临时区、转换和重组都计时。

BitPacker4x 的源码包含 x86 SIMD 和 little-endian AArch64 NEON 编码实现，不能因为旧文档概述而断言 ARM 只有 scalar。其 `compress_sorted` 的确实现 D1；D1 相邻差本身可以 SIMD，D1 解码前缀和也可以向量化。这里只因完整 u64 输入采用预变换后普通 pack，不能把 FoR 描述为“唯一能 SIMD 的差分”。实际 adapter 是否生成理想指令，还需固定编译器/target 后检查。[实现](https://github.com/quickwit-oss/bitpacking/blob/master/src/bitpacker4x.rs) · [宽度分派](https://github.com/quickwit-oss/bitpacking/blob/master/src/macros.rs) · [ARM 更新记录](https://github.com/quickwit-oss/bitpacking/blob/master/CHANGELOG.md)

不足 128 的块在 C/D 中统一沿用 seed + 完整 u64 D1 LEB128，空/单值遵循小表示；首轮不再增加 scalar bitpack tail。不要把 3 个值补成 128 个后当成小列表优化，也不能直接截断 BitPacker4x 的交错 payload。尾块与完整块由 count 确定，不需要每个生产者持有不同模式。真实完整块覆盖率很低时，SIMD 的端到端空间本来就有限。

FoR 的数学独立重建不等于该库已经支持任意单点读取。BitPacker4x 的公开块解码以 128 槽为单位；本轮不能直接宣称获得 MILC 那样的块内免整块解码二分。先按实际块 API 做搜索，额外 accessor 若需要，应另列维护成本。

一个可核对的空间界限：令 M 为 127 个 gap 的最大值，S 为它们的和，则 `M≤S≤127M`。非零时 FoR 位宽最多比 D1 多 7 bit；按本轮 128-code 约定，完整块最多多 112 payload 字节。这只比较同块 D1/FoR fixed-width packing，不是相对 LEB128 或其他自适应 codec 的界限，更不是性能保证。

## 5 目录只验证一个问题

默认先选冻结时一次写出的 eager flat directory：已有 Buffer 提供完整长度，直接消除增长前缀的搬移，结构最容易审查。是否改成 selected-only lazy index，取决于真实有多少 Positions 会被选中、会被反复查找，以及目录在构建和驻留内存中占多少。

E 可选方案是每个块保存固定宽度的 next-block byte span 和绝对 seed，平时不持有全量随机访问目录；选中后建立一个临时搜索视图，借用不可变 leaves。固定 codec、同一 payload 含义，只比较 eager 与 lazy 的完整目录策略。若 lazy 新增 span header，应将这些字节与解析成本明确列为变化，不能宣称只延迟了一次分配。

128 个 u64 LEB 值按最坏每项 10 字节估算，payload 不超过 1280 字节，u16 span 很有余量；仍必须把实际 seed/header、长度是否含自身、最后块约定一起验证。全叶子 offset 和 leaf index 不因此自动变成 u16/u32。若索引记录采用完整 `u64 first + usize rank + usize leaf + usize offset`，64 位下字段预算就是 32 字节/块；先如实计费，再考虑有显式边界的缩窄。

冷启动访问 K 个可跳转块 header 为 O(K)，同时产生索引分配；如果没有 span 或等价的块长信息，可能要解析 O(n) 编码值。建成后的 lower_bound 可以 O(log K + 128)，首次选择必须报告“建索引 + 第一次任务/查询”的总延迟，不能统一标成 O(log n)。多叶子含短尾时 K 为各叶块数之和；按 rank 建扁平索引还需块起始 rank，不能用全局 `rank/128` 越过叶子边界。

重复值安全规则：找第一个 `first[j] >= target` 的块，必要时从前一块重启扫描；若 j=0 从开头开始，若没有这样的 j 则扫描末块。不能取任意一个相等 seed，也不能用 `target+1` 处理上界。跨叶子定位同样保留最早可能含 target 的叶子。

## 6 论文与高级结构如何服务这次实验

以下区分已读到的方法与针对本项目的推导。它们并非同一年代，也不是每篇都应转成一个实现分支。

- **Culpepper–Moffat 2007 抽样目录**：从压缩 postings 抽取绝对样本和 byte/bit offset，先搜索样本再局部解码；文中界限为 O(log(n/p)+p)。适合解释导航元数据与 payload 分离。selected-only 构建、借用 root leaves 和重复值安全处理是本方案适配，不是论文现成容器。[原文](https://jsc.github.io/publications/cm07spire.pdf)
- **MILC 2017 FoR**：块元数据保存起始值、位置、位宽，块内是相对起始值的定宽 offset，可按其布局取值搜索。支持这次 FoR 假设，但论文的字段宽度和具体布局不能直接证明完整 u64 或 BitPacker4x 任意单点读取；其 128 个数据元素加抽取 skip 值的描述也不应与本轮 128-position 格式混同。[原文](https://dbucsd.github.io/paperpdfs/2017_1.pdf)
- **Immediate-Access Dynamic Indexing 2023**：固定字节块、头尾元数据和链接支持直接追加；跨块跳过仍会顺序遍历，较大的块降低链接开销但增加局部解码。其跨块差分和正 gap 假设不直接适用于可独立拼接且允许重复的叶子。它是 append 路线的参考，当前允许 Buffer 后不必照搬。[原文](https://jmmackenzie.io/pdf/mm23-ipm.pdf) · [作者实现](https://github.com/JMMackenzie/immediate-access)
- **Asadi–Lin 2013 增量 segment**：raw buffer 刷成不可变压缩段、通过链接继续增长，说明可以避免反复重建旧 payload；增长 buffer、segment pool 与该检索系统的取舍不能直接换算成 BPE 收益。[原文](https://arxiv.org/pdf/1305.0699)
- **CTZ 向后跳表**：2003 append-only skip-list 的确定性层级和 littlefs 的实际 CTZ 拓扑说明可以把索引随新块追加。按 checkpoint 1…m，64 位 backlink 总字节为 `16m−8·popcount(m)`，再加每点 8 字节绝对 seed，渐近约 24 字节/点，即 B=128 时约 0.1875 字节/值，尚不含 head locator。最新 checkpoint 的地址仍需保存；尾部 locator 会有有界重写，严格字节不可变 append 不能凭空免除定位成本。Buffer 已获允许后，首轮没有必要引入它。[2003 原文](https://arxiv.org/pdf/cs/0302010) · [littlefs 设计](https://github.com/littlefs-project/littlefs/blob/master/DESIGN.md#ctz-skip-lists)
- **Compressed Perfect Embedded Skip Lists**：按二进制层级嵌入 value skip 和 bit-distance skip，与“每块跳长再加几何远跳”最接近。完整对数界依赖足够高的 tower；固定高度仍有额外的跨大块跳转项。压缩预测字段和面向未来的 forward distance 增加维护量，不能直接当成简单在线 append 格式。[作者稿](https://vigna.di.unimi.it/ftp/papers/CompressedPerfectEmbeddedSkipLists.pdf)
- **Partitioned Elias–Fano 2014**：分块和两级定位可作结构参照；用户已有 EF 系列实验反馈，因此不据论文再自动安排重做，也不把 succinct 理论空间直接等同当前工作负载收益。[原文](https://pages.di.unipi.it/rossano/assets/pdf/papers/SIGIR14.pdf)
- **FastLanes 2023**：关注 SIMD 友好的布局和实现技术。已查的 Rust u64 packing 接口以 1024 元素为单位，不能将“1024-bit 虚拟 SIMD”误读成块长；将 128 restart 直接补到 1024 会有明显浪费，拼八块又改变访问和宽度耦合。这些是接口适配分析，不能替代用户已经做过的 FastLanes 实验；具体历史变体和指标尚未定位，本计划不复跑。[CWI 论文记录](https://ir.cwi.nl/pub/33289/) · [Rust 实现](https://github.com/spiraldb/fastlanes/blob/develop/src/bitpacking.rs)

2025 年 FastLanes File Format 的后续论文属于较新资料入口，但本轮未成功读取全文，不据此声称某项新功能或收益，也不让它阻塞上述对照。[2025 论文链接](https://www.vldb.org/pvldb/vol18/p4629-afroozeh.pdf) 2003–2017 年文献应标为基础方法；2023 工作也应标明年份，不能笼统称为“最新前沿”。

通用 indexed skip list、RRB/Conc tree 和碎片化 vector 暂不实现：前者带来链接/跨度元数据与分配，后者改变 flat root 或最终仍需扁平化；快速定位某个字节也不会自动得到压缩序列的 rank/value 定位。只有主实验发现清晰且无法用更简单结构解决的瓶颈，再重开这些方向。

## 7 正确性先于性能

所有变体共用一组与 raw Vec 对照的 roundtrip、搜索、任务切分和 concat 测试。包含长度 0/1/2/3、127/128/129、255/256/257，多块长列表，多叶短尾，全部相同、递增和稀疏值；覆盖零、接近 `u64::MAX` 的 seed、小 gap 的大绝对值、恰好跨 `2^32` 的 gap/span，以及最大合法 gap。

特别覆盖重复段跨多个 restart 和 leaf 边界；所有 lower_bound 返回最早位置。合并所有 ReadTask 输出应与原序列严格一致，不漏、不重、不改序。拒绝降序输入、计数/字节长度溢出；错误路径不发布半成品，不损坏已有数据。冻结后同时借用多个 reader，concat/drop 与提前退出均验证所有权。

C/D 额外覆盖位宽 0/1/31/32/33/63/64、两个 plane 的进位重组、完全空 payload、尾块及精确长度 Box 的读写边界。B 的 LEB 保留 canonical 编码；不要求它与 A 含几何目录空槽的整份字节布局逐字节相同。新增 unsafe 若非必要就不引入；外部内核的安全前提、CPU dispatch 与 padding 合约必须逐项核对。

## 8 Benchmark 组织

A/B/C/D 使用四个独立 worktree，从同一 baseline 并行准备；A 不改生产代码，B/C/D 恢复同一历史 `SmallVec<[u64; 2]>` 路线并统一冻结接口，完成对照前核对公共骨架一致。实现和资料核对可以并行，正式测量在同一硬件上串行执行，不让并行训练、编译或其他 benchmark 抢 CPU/内存带宽。固定 commit、依赖锁、Rust 版本、编译参数、allocator、CPU/NUMA/线程设置和语料 hash。

主矩阵为中文/英文各 256 MiB，分别使用 ByteLevel、Whitespace，分别 1 和 8 workers，共 8 个配置。固定训练参数、预处理结果与输入边界；若“256 MiB”是截取字节，应记录 UTF-8 边界处理及实际字节数。已有 corpus 和 harness 优先复用，不借机改训练算法、任务调度或 corpus pipeline。

每个主变体每个配置先做 2 次完整训练，以轮换或交错的变体顺序减小时间漂移。报告两次原始值和范围；只有排序翻转、重复差异明显，或候选收益落在噪声范围内时加第 3 次。不强制 5 次，也不把两个样本包装成高置信度统计结论。A/B/C/D 共 64 次起步；E 若达到进入全量的理由，再补它的对应 16 次。

先采真实分布：按列表长度统计数量与位置总量，按完整块统计 D1 最大 gap、FoR span、两种位宽、零宽块、尾块比例、每叶块数、每 root 叶数、被选中率及查询次数。长数组 microbenchmark 不能代替大量 1–3 项列表的实际权重；绝对坐标很大也不等于块内 span 很大。

局部测量覆盖 Buffer 填充、预变换、width 计算、两个 plane 的 pack/unpack、分配/增长/复制/冻结、完整扫描、lower_bound、任务生成与 concat。尽量重放相同真实样本；microbenchmark 的内部循环用于稳定短操作计时，不再机械增加整套重复次数。D1 prefix sum 和 FoR add 的成本单独列出，但最终比较必须包含全部步骤。

正式性能以完整训练为主：分别记录初始化/计数与发布、候选读取与 prepare、Worker 编码/冻结、Owner concat/commit、reuse 归并/重编码、最终 retirement/drop，并记录从任务开始到回收完成的 wall time。Worker 累计 CPU 时间和编码阶段的关键路径 wall span 分开记，不能把并行 Worker 时长求和后当成串行总耗时。普通新片段不在 Owner 再编码；其他路径按实际线程记录，不能为宣称“Owner 编码为零”而改变 reuse 语义。阶段命名以实际可用计时点为准；“训练完成”但存储尚未回收的时间不能冒充完整耗时。若改进仅来自延后释放，应在 retirement 和峰值中显现。

## 9 必须报告的成本与决策规则

| 类别 | 最低报告项 |
| --- | --- |
| 时间 | 完整 wall time、阶段时间、首次选择/建索引、稳态查询、retirement/drop |
| 空间 | 最终 payload、全部 header/directory、root 与叶子描述符、SmallVec 对象与 inline/heap capacity、tail/slack、临时 plane、选中视图 |
| 峰值 | 进程峰值 RSS，以及可观测的 live allocation/capacity 峰值；注明采样能否漏掉短暂重叠 |
| 分配 | alloc/realloc/free 次数、申请与释放字节、目录搬移及冻结复制字节；单独说明测量开销 |
| 维护 | 生产代码 LOC 净变化及涉及模块数，排除注释、空行、测试与 benchmark；另列 unsafe、格式分支和不变量 |
| 语义 | 输出词表/merge 等与 baseline 的一致性、测试结果、异常与提前退出回收 |

raw 坐标的逻辑大小为 `8n`，但不能把它全部记成额外堆分配：历史 `SmallVec<[u64; 2]>` 的新建 0/1/2 项缓冲使用对象内 inline 存储，spill 后才有独立堆容量。分别统计 SmallVec 对象占用、inline/spilled 数量及 spilled buffer 的实际 capacity；inline 字节已含在对象内，不重复累计。若有复用，按实际 spilled 状态计费，不仅根据当前 length 推断。编码时 Buffer 仍可能与最终 payload、目录和暂存空间同时存活；`Box` 的稳态更紧凑不意味着峰值更低。按所有存活 fragment 的 Buffer 统计 high-water mark，一个 Worker 可能同时持有多个 fragment。分别报告 1/8-worker 峰值以及 Worker→Owner 队列中已冻结叶子的驻留量。重型 allocator profiling 与无插桩耗时分开跑，避免把 profiler 开销当成算法差异。

四个主变体完成对照后，以端到端耗时、峰值和维护成本共同选择。若 B 已达到主要收益且最清晰，最终就采用 B；若 D 的编码更快但 payload 更大或 reader/训练变慢，不凭 SIMD microbenchmark 选择它；若 C 在真实分布上更合适，保留 C。收益小于噪声时按持平报告，优先简单方案。32B 未达标、内存明显升高或某一 1/8-worker 配置回退，应明确展示，不能藏进总体平均。

交付物应包括各变体 commit、可重复命令、语料与环境指纹、逐次原始结果、汇总与阶段解释、真实生产 LOC diff，以及“采用哪个方案、哪些假设被否定”。若没有一个变体稳定胜出，保留当前基线同样是有效结论。除非新数据指出具体必要性，不继续扩展到更多 codec、树或跳表。
