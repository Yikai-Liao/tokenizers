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
