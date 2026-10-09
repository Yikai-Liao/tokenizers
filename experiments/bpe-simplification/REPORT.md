# 2100 行预算下的简化引擎

中文 4 workers、ByteLevel、50K 词表、min_frequency=2 的未插桩 core 训练为
**23.879 秒**，满足用户要求的 25 秒以内。生产 **2074 行**，包括独立 oracle、
共享 helpers 和 Miri 引入文件的测试 **1522 行**，均按 rustfmt 后非空非注释行计数。
相对固定 main 的 5714 / 6192 行，分别减少 63.7% / 75.4%；实现模块从 26 个减为 6 个。
没有把生产算法移出 engine 来缩小计数范围。

分支 `simplify/bpe-maintenance-20261009`，实现提交 `d19e5bc6`。
语义基线固定为 Fork main `e4f787dc189d9be7192107490d652096cde7480e`。
用户随后重开 Arena，并把生产上限从 2000 提高到 2100；本报告取代早期 1858 行交付，
其证据和说明保留在 `2c1daf2b` 的历史中。

## 保留的行为与模块边界

完整 vocabulary IDs、全部 ordered merges、兼容优先级前缀批次、并行 prepare/apply/owner
commit、AA 左到右选择、reserved/active 身份与重启、signed reuse cohort 和严格 birth
length 规则均保留。位置 API 与 codec 支持完整 u64 域，包括 `u64::MAX`；resident
分配仍按 usize/isize 边界检查。

上层流程仍是 vocabulary → corpus plan → Arena/index → endpoints → select → prepare →
apply → commit → model。Vocabulary 隐藏身份，Corpus 隐藏几何，PairIndex 隐藏优先级与
计数，Batch/Prepared 隐藏匹配和事件，Positions 隐藏存储与生命周期。所有并行阶段在
下一阶段之前 join；错误丢弃整个 attempt，不承诺回滚已应用的轮次。

本轮恢复的具体机制是：

- 最终列表采用 main 的 16B 描述符和 inline flags。小最终分配借用 attempt Arena，
  大分配单独拥有堆存储；借用生命周期让 Arena 始终晚于列表释放。
- Arena 阈值与 main 相同：`max(256, floor(sqrt(physical_items / 256)))` 字节。
  `physical_items` 来自 `plan.items()`，按含头部、重启目录和数据的完整 layout 判断。
  每 worker 的编码 scratch 跨轮复用，分配 cursor 只在顺序 closure 内持有。
- 临时 Builder 常见情况下用 u32，超过范围时无损提升为 u64。首个聚合片段移交
  所有权；后续排序 append 保持顺序，避免首次复制大列表。
- 完整 ordinary producer 先按 floor 剪枝，再编码。`Birth::Complete` 交给 owner
  直接发布；`Birth::Partial` 仍先聚合全部计数，再编码。AA/reuse 不走完整快路。
- Apply 保留各 job 的事件块，由 commit 消费；紧凑 metadata 引用路由复用容量，
  每个 birth 列表只移交给一个 owner。零权重但有 positions 的 birth 仍保留。

## 未插桩最终对照

一次相邻 main/候选对照覆盖四个 case，共八个独立进程；完整模型全部一致，child swap0。
输入是固定 English/Chinese ByteLevel word map 和约 256MiB 文本。两边相同 feature/release
profile：无默认 features、opt3、fat LTO、一个 codegen unit；4 workers，绑 CPU0–3。
机器为 6-vCPU Xeon KVM。计时前构建、测试和 Miri 均已退出。

| case | main train s | 简化 train s | main pipeline s | 简化 pipeline s | main RSS GiB | 简化 RSS GiB |
|---|---:|---:|---:|---:|---:|---:|
| en-core | 1.080 | 1.425 | — | — | .177 | .191 |
| zh-core | 19.559 | **23.879** | — | — | 2.348 | 2.807 |
| en-pipeline | 1.087 | 2.502 | 5.013 | 6.823 | .178 | .174 |
| zh-pipeline | 17.731 | 24.498 | 23.737 | 29.831 | 2.389 | 2.648 |

Core train 使用公开 do_train 边界；pipeline 总时间包含 feed。RSS 是序列化/校验之前的
进程 HWM，包含输入加载或 feed，并非 engine 的单独存储。中文 core 的 CPU 时间为
76.444s，main 为 64.121s。中文 core 仍约为 main 的 1.22 倍；英文 pipeline 训练段也有
明显差距。达到 25 秒目标不意味着恢复了 main 的全部性能。

这是每 case 一对样本，不能提供稳定百分比或置信区间。按用户要求没有为了噪声追加六对
重复；最终证据保留真实 wall/CPU/RSS、输入和 binary hashes、affinity、swap 与完整模型
比对结果。旧主机、旧词表规模和插桩结果没有混入本表。

## 本轮组合筛选与诊断

| 组合 | 生产行 | zh-core s | CPU s | RSS GiB |
|---|---:|---:|---:|---:|
| 16B descriptor + 分离 raw builder | 1996 | 31.533 | 93.721 | 3.840 |
| scratch 复用 + 首片段移交 | 1998 | 27.235 | 83.992 | 4.227 |
| 再加临时窄缓冲 | 2037 | 24.894 | 78.765 | 3.006 |
| 再加完整 producer 提前编码 | **2074** | **23.879** | 76.444 | 2.807 |

这些是逐次组合的探索样本，不能把相邻差值全部归因于单项机制，也不宣称全局最优。
对应提交依次为 `cd3b1a83`、`3728bdb2`、`e39c8d76`、`d19e5bc6`。

[PHASES.md](PHASES.md) 保留相同 main/1849 行候选的阶段诊断。Apply 的 wall 差 2.103s
中，串行收集/释放差 1.556s，约占 74%，支持去掉大平面 Change 数组。新的 Arena 描述符
诊断显示结束释放已降至 .191s，剩余 raw 聚合和 owner 编码仍值得处理。插桩训练时间仅作
诊断；即使其总时间更短，也没有用来验收 25 秒目标。补丁及逐段数据已归档。

## 验证与复现

最终源码默认/无默认 features 各 38 项 library tests 和 doctest 通过；all-target Clippy
`-D warnings`、rustfmt、行预算及 git whitespace 检查通过。小 fixture 比较完整模型和
每条 `(pair,count,replacement ID)`，覆盖 affix、alias、zero weight、AA、partial floor、
strict length、wide numeric 错误、65536 真实 IDs、feed/reload/progress/线程策略。

Standalone Miri 直接引入实际 `positions.rs`，四项测试在默认借用与泄漏检查下通过，
没有设置忽略检查的 flags。Native 用 Rayon pool 验证并发；Miri 用 scoped threads
及两个 Arena cursor 隔离 Crossbeam 的全局延迟回收。Native 遍历全部 seek 起点；Miri
重点覆盖首尾、重启边界及 full-u64 值。初次 Rayon 运行的第三方 leak 报告保留在原始
results；最终严格检查没有该错误。归档 harness 的相对引入路径已另行编译、运行验证。

最终源码再次 release 构建后的 SHA-256 与计时用的不可变 binary 完全相同：
`8e1e1ee5ed56d153e07a129f8864b5cf5fd3fcf1445fe9108edae27870a681c5`。

```sh
cargo test --manifest-path tokenizers/tk-train/Cargo.toml
cargo test --manifest-path tokenizers/tk-train/Cargo.toml --no-default-features
cargo clippy --manifest-path tokenizers/tk-train/Cargo.toml --all-targets -- -D warnings
python3 experiments/bpe-simplification/count_lines.py
cargo +nightly miri test --manifest-path experiments/bpe-simplification/miri-codec/Cargo.toml
```

完整 input/source/binary hashes 和当前构建口径见 [manifest-final.json](evidence/manifest-final.json)，
最终八进程记录见 [final-runs.json](evidence/final-runs.json)，格式行数见
[lines-final.json](evidence/lines-final.json)。原始 records、stdout/stderr 和不可变 binary
保存在 `/root/code/tokenizers-simplification-results/`。`run_pairs.py --one-pair` 支持在新 label
下只跑一对 × 四 case；需按 manifest 构建 baseline/候选 binary 并提供相同输入。
原始主工作区未改动，没有 push 或新建 PR。
