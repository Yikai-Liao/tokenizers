# BPE 简化实验笔记

日期：2026-10-09。固定基线：Fork main `e4f787dc189d9be7192107490d652096cde7480e`。
工作目录：`/root/code/tokenizers-workspaces/bpe-simplification`。
工作分支：`simplify/bpe-maintenance-20261009`。
原始日志和不可变二进制：`/root/code/tokenizers-simplification-results/`。

## 用户约束与实验口径

- rustfmt 后生产逻辑最多 2000 行，不计空行和注释，不能搬出计数范围或压行。
- 测试含独立 reference 与共享 Word helper，不得超过生产逻辑行数。
- 保留兼容规则批处理、并行准备和聚合；位置索引覆盖完整 U64。
- 优先 4 核中文、英文 ByteLevel；完整词表 IDs 与全部有序 merges 必须一致。
- 先各测一次发现明显问题，结果接近时再补重复。不要每轮展开重矩阵。
- 比较 PR #2501 另外三个实现时，需要核对语料、100K 词表、机器和计时范围；
  本轮当前为 50K、6-vCPU KVM，不能直接混入笔记本旧结果宣称加速。

## 0. 基线准备与 A/A

固定相同 Rust 1.98.1、release opt-level 3 / fat LTO / codegen-units 1，
训练默认特性关闭；输入及二进制 hash 在 `evidence/manifest.json`。
原 main 生产逻辑 5714 行、测试及 oracle/helpers 6192 行。
A/A 展开了 4 格 × 6 对，加预热共 56 个独立进程，完整模型均一致、child swap 为 0。
训练耗时区间可有约 5%–10% 的波动。用户指出过程过重，后续改用少量 smoke 测量，
仅在决策需要时补样。这个 A/A 成本不应在每个候选上重复。

## 1. 首个完整简化版，已独立存档

存档分支：`archive/bpe-lean-1456loc-20261009`。
提交：`95638077`。不可变二进制：`bin/compact`。
生产逻辑 1456 行，测试逻辑 1384 行；6 个实现模块。
默认与 no-default library tests 均为 35 passed；Clippy `-D warnings`、rustfmt 通过。

使用安全 U16/U32 atomic token 槽、完整 U64 restart/delta 位置索引、
兼容 priority prefix 批处理、并行阶段与 count owners。
去掉 arena/linked birth、radix、U24、自适应反馈、多种 producer 协议、
并行 AA parity 和 unsafe corpus pointer。AA 保留串行贪心语义和后续并行执行。

初测（4 核、256 MiB、50K、min_frequency=2，各候选一遍）：

| 数据/计时 | main | 首版 | 首版峰值 RSS |
|---|---:|---:|---:|
| 中文纯训练，紧邻 A/B | 18.80 s / 2.41 GiB | 70.92 s | 3.29 GiB |
| 英文纯训练 | 约 1.1 s / 180 MiB | 3.16 s | 279 MiB |
| 英文含 Feed | 约 5 s | 7.50 s | 281 MiB |
| 中文含 Feed | 约 24 s | 76.47 s | 3.37 GiB |

四格完整词表与 ordered merges 都等于 main，child swap 为 0。
中文含 Feed 运行中附加了 `perf record -F49`；该格是诊断结果，不能当作无采样排名。
首个中文纯训练回归为 3.77×、峰值增加约 0.88 GiB；不需要再跑六对确认明显回归。
重复驱动在第一对完成后停止；下一次未完成的 main 原始目录保留、排除。

### 热点证据

`compact-profile.data` / `compact-annotate.txt`：merge 准备约 35.45% 自身样本，
其中 77.08% 落在 word starts 的二分查找指令上；约三成全部 CPU 样本在这项查找。
matched 约 11.08%，free 约 5.77%，owner commit 约 4.35%。
这是采样占比，不是阶段精确 wall time；采样未覆盖整个初始阶段。

## 2. 权重区间与压缩写回，实施中

生产逻辑暂为 1558 行，测试 1384 行。未有本轮性能结论。

- 单词按权重排序，本轮把同权重的连续 word 合成区间；查找权重不再二分数百万
  word starts。缓存区间末端，权重为 1 的常见区间有直接判断；reuse word 域不变。
- fresh 规则共享 ID span，把写回快照改为完整 U64 的压缩 starts + 一份规则几何，
  避免每个 match 保存 24/32 字节的结构；reuse 保留 occurrence 几何。
- 两种写回都遵守 prepare 全部 join 后才 apply，不在写阶段重新读取匹配几何。

先验证完整模型/逐规则轨迹，再构建不可变 release 候选，优先各测一次纯训练。
存档分支不移动；后续改动在工作分支继续。

### 本轮结果与保留决定

默认 library tests 35 passed，rustfmt/预算检查通过。
不可变 release 二进制 `bin/weight-writes`，源文件 hash 已入 manifest。
每语种一次纯训练，256 MiB、4 worker、50K；全部模型和 ordered merges 等于 main，child swap=0。

| 语种 | 1456 行首版 | 1558 行本轮 | 首版 RSS GiB | 本轮 RSS GiB |
|---|---:|---:|---:|---:|
| 英文 | 3.155 s | 2.345 s | 0.273 | 0.277 |
| 中文 | 70.921 s | 46.982 s | 3.290 | 3.277 |

中文减少约 33.8%，英文减少约 25.7%；内存变化很小，不声称改善。
两项作为本轮组合保留；未独立隔离各自贡献。
main 来源：`corpus/mod.rs::WordWeightCursor`、`corpus/prepare.rs` 的权重区间，
以及 `merge/mod.rs::WritePlan` 的固定规则几何。reuse 仍使用 occurrence 快照。

下一轮从 main 的 producer birth 编码提取精简方案：fresh 部分任务按空间顺序
汇合，可在 owner 上串接已排序位置，减少大 Vec 与重复排序；reuse 保留显式排序。
参考 main 的 small-posting 处理，避免为 singleton 分配整个 block directory。
本轮已有效，按用户要求先提交，再实施下一轮。初版 archive 分支不移动。

## 3. fresh birth 流与 singleton 重启，保留

生产 1576 行（+18），测试 1384 行，library tests 35 passed。
`bin/stream-birth`、source hashes、原始 samples 已保留。
每语种一次纯训练，完整模型和 ordered merges 均等于 main，child swap=0。

| 语种 | 上轮 1558 行 | 本轮 1576 行 | 本轮 RSS GiB |
|---|---:|---:|---:|
| 英文 | 2.345 s | 2.242 s | 0.264 |
| 中文 | 46.982 s | 40.223 s | 3.312 |

中文 wall 减少约 14.4%，CPU 从 136.67 s 到 126.67 s；英文差距小，单样本不声称微小加速。
中文 RSS 未改善。增加 18 行保留了 producer 空间顺序和压缩坐标，值得继续保留。

借鉴 main 的 complete producer 编码：fresh 中每个 born pair 只有一个 producer，
分任务按空间顺序收集后 owner 可串接有序流，不再对 fresh births 再排序。
压缩直接进入局部邻居组；reuse owner 仍先汇集完整 U64 再排序。
借鉴 main small posting 的低分配原则，把首个 restart 放在流对象内，singleton 不分配目录；
不是恢复 arena 或多种 posting 协议。U64 边界、重复位置和跨重启测试继续通过。

下一轮提取 main `CorpusPlan` 的延迟 materialization，暂不恢复分阶段重叠或 wave 协议。

## 4. 延迟 materialization / 小域计数，保留

生产 1656 行（+80），测试 1385 行；library tests 35 passed。
`bin/deferred-dense`、源 hash、inputs 和原始 samples 保留。
每语种一次纯训练，完整模型和 ordered merges 等于 main，child swap=0。

| 语种 | 上轮 1576 行 | 本轮 1656 行 | 上轮 RSS GiB | 本轮 RSS GiB |
|---|---:|---:|---:|---:|
| 英文 | 2.242 s | 2.323 s | 0.264 | 0.275 |
| 中文 | 40.223 s | 41.655 s | 3.312 | 2.998 |

中文峰值减少约 322 MiB（9.5%）。wall/CPU 波动幅度不支持声称加速；以降低内存为保留理由。
英文的十余 MiB 差距包含 map load 的峰值波动，单样本不作精确 RSS 推断。

提取 main `CorpusPlan`：有借用的 plan 先构造压缩初始索引，再消费 plan 分配 token plane，
释放薄 word 引用。fresh 不再保留 word starts；权重只保留连续区间，不保存每词权重。
零 merge 仍先验证初始 pair/count，随后跳过 token plane 分配。
首次测试发现零 merge 少发 `Compute merges` 进度，已补最终空 stage；失败日志保留在原始目录，
复测 35 passed 才进行性能测量。

提取 main bounded collector 的小 ID 域计数原则：初始 ID 域最多 256 时用直接表，
更大域继续完整 pair key hashmap。所有位置始终 U64；没有恢复 radix/wave/compact record 协议。
此实现未覆盖“少量活跃 ID 位于很大实际 ID 域”的 dense 特化，合法输入仍有通用路径。

下一轮提取 main `Execution` / `IdDirectory` 的 worker scratch 复用，减少重复哈希分配和释放。

## 5. worker scratch，组合筛选检查点

生产 1702 行，测试 1392 行，library tests 35 passed。
不可变 `bin/scratch`；中英纯训练完整模型相同，child swap=0。
英文 2.507 s / 0.278 GiB；中文 39.396 s / 2.972 GiB。
相对上一轮中文 wall 低约5.4%、CPU低约4%，英文更慢；差距接近A/A波动，不能认定明确收益。
目录直接索引、双侧有序 Change 和 worker 复用在这一组合中尚未隔离。
保留为可复查的检查点，下一步比较 reviewer 建议的更短 map_init 实现。

用户要求最终依据记录重新筛组合，当前逐轮恢复不是最优组合证明。
用户要求定期 sub-agent review，已启动只读审查，后续每两轮或组合调整后复审。
审查建议：map_init 省约25–35行、u32邻居目录约减半内存、initial第一片直接移入而不重编码。
这些均为待验证假设；U64位置不改，reuse unordered排序不删。

## 6. reviewer 建议的 map_init/u32 简化，保留替代 scratch

生产1681行（-21），测试1385行，默认library tests35 passed。
英文2.302 s、中文39.903 s；CPU分别6.24/123.52 s，RSS分别0.273/3.079 GiB。
完整模型与全部有序merges一致，child swap=0。
中文与worker scratch的39.396 s接近，英文回到上两轮区间。
本轮移除全局Scratch、Mutex、pool-worker-index耦合；接受更少行数和更少隐含约束，
不声称微小速度或RSS差异有统计显著性。RSS增加约110 MiB，应在最后相邻复測核对。
u32仅用于每侧unique-ID目录entry index，U64位置保持不变。

只读review #1固定2b55b7f6：现有六模块边界有封装深度，不恢复Execution/arena框架。
指出scan_symbols零offset参数、initial_spans冗余类型转换可删；initial已直接压缩，
reuse unordered不可删排序；fresh有序串接依赖唯一producer与indexed collect顺序。
建议分别测fresh固定几何缓存、head-read ring、小域dense计数消融，再筛少量竞争组合。
未发现确定数值/U64/并发缺陷；这不是复测或性能保证。下一轮之后安排review #2。

## 7. main固定PairMatcher几何的窄缓存，待最终筛选

生产1703行（+22），测试1385行，library tests35 passed。
英文2.417 s / CPU6.57 s，中文38.386 s / CPU119.02 s；RSS0.274/3.094 GiB。
完整模型及全部merges相同，child swap=0。
相对map_init中文wall低3.8%、CPU低3.6%，英文略慢，仍接近噪声区间。
以Corpus::fresh_matcher窄闭包缓存left/total，reuse occurrence matched路径不变。
暂存以便独立检查，不能据此认定新增22行已获得确定性能收益。
下一轮单独加入16项安全head-read ring，比硬件unsafe prefetch更易维持原phase边界。

## 8. head-read ring与Perf诊断切换

1719生产行，1385测试行；library tests35 passed。
16项ring顺序保留，reuse路径不改；暂未有无采样性能结论。
review #2固定源码，未发现确定问题，但明确同步head-read弱于硬件prefetch。
用户指出中文仍差约2倍，要求按Perf热点/优化潜力排序。
暂停新增微优化，以baseline和head-ring固定二进制各一次中文core4/50K诊断：
perf cycles -F99 / dwarf8192调用栈，并收集cycles/instructions/cache/branch counters。
诊断运行完整模型比较，记录child swap；不用于无采样性能排名。
原始数据 profiles/zh-core-4-{baseline,head-ring}/；下一步由样本与热点指令决定实验优先级。

### Perf结果与用户追加方向

main/head-read采样train CPU74.97/131.85 s，wall24.09/42.93 s（有采样开销，不排名）。
完整模型相同、swap0。owner归属cycles12.76/55.14B、prepare63.22/100.48B，
二者解释约73%归属cycle差；codec push/FlatMap next自样本约23%可识别engine。
见PERF.md，优先真hardware prefetch、typed position cursor、fresh owner完整流发布。
用户指出main prefetch收益明显，应低行数加入；用户要求今后每轮review用fresh agent。
旧reviewer已停止复用，新perf_priority_review_fresh正在只读核对机制/收益潜力。
head-read以诊断检查点提交，不认定保留，不用它的42.93s判断prefetch收益。

## 9. 真正的hardware prefetch，保留进入下一组合

生产1737行，测试1385行；library tests35 passed。
中英文各一次无采样core，完整模型相同、swap0。
英文2.038 s / CPU5.44 s / RSS0.271 GiB；中文35.489 s / CPU108.11 s / RSS3.161 GiB。
相对无ring的geometry38.386s，中文wall少7.5%、CPU少9.2%；英文wall少15.7%。
不能与head-read的有采样42.927s排名；最终组合仍要相邻复测，RSS无改善证据。
从main提取16ahead非阻塞_mm_prefetch；保留安全token操作，只有cache hint使用局部unsafe。
地址由get()给出，其他架构no-op。它与同步head-load不同。

fresh reviewer perf_priority_review_fresh复核了主要潜力：先prefetch，再ownedstream append，再typed cursor。
owner codec自样本仅7.03%识别engine，不把整个23.3% owner当可消除收益。
可变restart+owned append须封装entry/byte偏移；预计+70–120行，metadata/RSS待测。
用户询问Affix coverage/batch：保留prefix/suffix/空值/alias/occurrence spans/strict gate/1-4-8worker的完整模型与逐步trace；
普通Affix可batch，active reuse退到单rule cohort但内部并行/owner聚合仍保留。
用户随后明确不增加显式Affix batch断言；未修改该测试，沿用现有覆盖。
以后review每次fresh agent，不复用旧reviewer。

## 10. owned stream 拼接，保留进入 typed cursor 比较

生产1776行，测试1430行；默认library tests36 passed。
英文2.029s / CPU5.30s / RSS0.272GiB；中文32.940s / CPU96.08s / RSS3.143GiB。
完整词表及merges与main相同，swap0。相对prefetch中文wall低7.2%、CPU低11.1%，
英文接近，RSS不支持改善结论；仍需相邻复测。
按值路由Change，fresh及initial保留片段restart后拼接，不再解码重编码。
Block增加实际entry偏移，短块增加metadata；reuse仍decode/sort。
新增codec覆盖不齐片段、singleton、重复边界、append后push、所有起读及全U64。
新fresh reviewer固定源码未发现阻断问题，见REVIEW-4.md。
用户纠正：停止汇编分析和寻找新方向，先测既定typed cursor，再按总体收益筛选。
用户追加核对Arena：当前未启用；旧负载诊断存在数秒收益，补为下一项独立候选。

## 11. typed spanning cursor，进入组合复测

生产1797行（+21），测试1430行，默认library tests36 passed。
英文1.959s / CPU5.21s / RSS0.272GiB；中文30.873s / CPU88.38s / RSS3.207GiB。
完整词表及merges与main相同，swap0。相对owned stream中文wall低6.3%、CPU低8.0%，
英文低3.5%，单样本不足以区分小差异；RSS增加约66MiB需相邻复测。
以单个Cursor跨可变restart，去每块FlatMap；Source用Either去Box动态分派。
全U64/不齐碎片/任意起读/重复seek的现有codec覆盖通过。
用户提醒fresh review；新typed_arena_review_fresh审查本轮并评估下一Arena候选。

## 12. Arena 第一版，性能退步，待一次窄修正

生产1924行，测试1462行；37默认library tests通过。
英文2.018s / CPU5.52s / RSS0.324GiB；中文36.836s / CPU115.74s / RSS3.548GiB。
完整模型相同、swap0。相比typed cursor中文wall高19.3%、CPU高31.0%，RSS增349MiB。
本轮明确不认定保留。采用worker-scoped Bump及scope-borrowed Bytes，
closure内一次lease，8→256增长，大于256转heap；所有资源joined后释放。
它在编码时增长，与main使用encoding scratch后一次finalize不同。
独立fresh reviewer未发现阻断；建议10-byte栈缓存一次写delta，减少逐字节extend
并避免allocation失败留下部分delta。下一轮只测这一窄修正，再裁决当前Arena方案。

## 13. Arena 单次delta写入，仍未胜出

生产1928行，测试1462行；37默认library tests通过。
英文2.012s / CPU5.55s / RSS0.333GiB；中文33.912s / CPU101.37s / RSS3.504GiB。
完整模型相同、swap0。比逐字节Arena好，但仍比typed的30.873s/3.207GiB差。
不能据增长式builder否定main的最终一次分配策略；下一轮将Arena局限到owner
已完整聚合且过floor后的一次freeze，并恢复main内联小payload原则，避免增长退休
空间与临时Writes进入Arena。随后用相同builder的无Arena控制区分二者收益。

## 14. Inline16 + owner发布时payload freeze，局部提取待控制

生产1909行，测试1460行；37默认library tests通过。
英文2.234s / CPU5.88s / RSS0.285GiB；中文35.871s / CPU101.38s / RSS3.175GiB。
完整模型相同、swap0；仍慢于typed，当前不能保留。
Arena不用于增长和临时Writes，仅在完整owner group过floor后接管17..256bytes。
独立fresh reviewer核对main实际链，明确这只是payload freeze：仍有每列表heap
编码Vec、独立blocks、fragment重启，未恢复main16B本体/scratch/完整producer直接alloc。
前两版增长式原型不是忠实提取main，不能用退化否定main Arena；见REVIEW-7.md。
用户要求最少代码拿主要收益，并明确Inline可能无收益、需要测试；下一轮相同
Inline16 builder NoArena隔离这份inline及freeze，随后重点测main完整producer
前置裁剪及owner直接发布的窄提取，不恢复整个packed codec协议。

## 15. 相同Inline16 builder无Arena控制，待相邻复测

生产1845行，测试1430行；36默认library tests通过。
英文2.181s / CPU5.59s / RSS0.284GiB；中文34.012s / CPU101.35s / RSS3.280GiB。
完整模型相同、swap0。Arena-freeze组中文CPU101.38s，几乎相同；wall35.87 vs34.01
尚不能归因。相比历史typed30.87/CPU88.38，本次builder组合不支持保留，
但须相邻typed/inline复测以排除host频率/负载变化。
这是Inline16加stack-delta builder，不单独将结果归因于inline。
用户要求以最少代码拿主要收益，下一项只提取main完整producer floor裁剪
及owner直接move发布，保留partial/AA/reuse聚合，避免复制全部allocator协议。

## 16. complete producer 前置裁剪与直接发布

生产1838行，测试1439行；36默认library tests通过。
英文2.109s / CPU5.43s / RSS0.272GiB；中文30.957s / CPU85.44s / RSS3.123GiB。
完整模型相同、swap0。相对typed历史样本wall近似、CPU略低，不宣称小差异显著。
恢复typed codec，删除未胜出的Inline/Arena。complete producer在编码后裁剪，
owner直接move发布；partial/AA/reuse继续完整聚合。没有恢复main的裁剪前直编。
新增高floor跨多个partial的oracle场景，防止局部误剪。
此前相邻typed/Inline控制已完成：中文33.301/33.884s，CPU95.25/101.94s；
不支持保留Inline builder组合。用户明确停止为机器噪声追加重复，后续遵循。
下一项按main保留项数<=total/workers的小candidate完整，修正按fragment块数过切；
用一次中英文完整模型测量裁决，不增加汇编/profile。

## 17. 按main项数保留完整ordinary candidate

生产1849行，测试1439行，36默认library tests通过。
英文1.932s / CPU4.96s / RSS0.271GiB；中文31.048s / CPU85.48s / RSS2.990GiB。
完整模型相同、swap0。中文wall/CPU与complete单项近似，RSS低约135MiB。
小candidate按len<=ceil(total/workers)整条调度；大candidate保持整块分段，
AA保持此前grain，未恢复main多小candidate合并job。保留作为结构合理且内存更好的候选，
不声称其填平main速度差距。REVIEW-8包含两份新代理的固定源码审查。
下一项main未压缩写计划仅改数行，净不增长。

## 18. main未压缩写计划窄提取，未取得整体收益

生产1849行，测试1439行；36默认library tests通过。
英文3.263s / CPU5.26s / RSS0.266GiB；中文31.376s / CPU84.87s / RSS3.022GiB。
完整模型相同、swap0。中文CPU几乎相同、wall未降、RSS增约32MiB，
没有证据支持保留。英文wall明显偏高但CPU无相应增幅；不为机器噪声追加重复。
本轮仅改Compact写计划为Vec<u64>，birth postings保持压缩；不否定main较窄
未压缩坐标plane的收益。归档后恢复压缩写计划，再测审查第三项fresh owner目录。

## 19. fresh owner bucket/neighbor目录，未胜出

生产1903行，测试1439行，36默认library tests通过；新代理固定快照未发现阻断。
英文2.418s / CPU6.11s / RSS0.270GiB；中文32.551s / CPU89.53s / RSS3.132GiB。
完整模型相同、swap0。相对whole31.048/CPU85.48/RSS2.990没有优势，归档不保留。
仅替换fresh partial hash groups为stable bucket排序+邻居目录，完整快路及reuse不改。
目录持有usize全局group索引，避免跨bucket总group数误限u32。
用户指出多个局部提取无效，要求明确当前阶段差距。停止新增优化；
先用当前best whole与main相同阶段边界的wall/进程CPU计时定位。
早期Perf owner4.3x/prepare1.6x不代表当前typed/prefetch后的阶段占比。
