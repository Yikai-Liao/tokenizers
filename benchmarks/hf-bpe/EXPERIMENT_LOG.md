# BPE 实验记录

## 当前比较规则

- 使用固定来源的真实 Wikipedia 段落，同一输入、预处理、词表与最小频率；核对完整词表及有序 merges 的摘要。
- 新的公平比较统一 u32 token ID 与 u32 posting。PR 的每 Symbol 长度字段属于其算法布局，另列；不拿窄 ID 的内存优势解释算法收益。
- 双方先用 1 线程初始化计数、4 线程 merge；我方 4 线程初始化另列为优化项。
- 只做关键单次测试，完整矩阵等算法确定后再运行。
- 仅 MemAvailable ≤ 1 GiB 时停止；进程与系统 swap 如实记录，少量 swap 可以接受。构建、测试、下载与计时错开。
- 按实现创建独立 worktree、branch、commit，直接使用原 BpeTrainer::train_vocab/train 接口。

## 2026-09-30：旧 u16 并行访问对照

固定 Wikipedia revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`，先准备约 1 GiB 独立原文，再取完整行前缀 `536870289` 字节。输入 SHA256 为 `a0d40d4102cba1e933f25e5ccd17552d2eaebaec4ef770bc39d8e46e740b4d4f`。目标词表 50,000，min_frequency=2，none 预处理，无 affix/special/长度限制。

| 旧版本配置 | Train s | Merge s | RSS GiB | 最低可用 GiB |
|---|---:|---:|---:|---:|
| u16 corpus、u32 posting、非原子1 worker | 176.063 | 83.933 | 3.062 | 3.046 |
| 同布局、非原子4 worker | 56.430 | 26.627 | 3.005 | 3.046 |
| 同布局、Relaxed 原子4 worker | 56.545 | 26.656 | 3.005 | 3.093 |

三个模型摘要均为 `d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd`；alphabet 20,757，实际29,243条规则，1,550批，最大75规则/批。数据在 [parallel-key-512-final.jsonl](results/parallel-key-512-final.jsonl)，来源、源码、依赖锁及二进制摘要在相邻 environment JSON。

收获：同版本4/1 worker训练3.12×、merge3.15×；原子与非原子4 worker差异仅约0.2%，单次共享主机观测无法区分。4 worker的delta/commit约21.2秒、planning约4.35秒、实际写入约0.65秒。串行HF初始化/语料构造约13.48秒仍需优化。源码确定的新增工作是全局16B Plan拼接、混合规则排序、过滤后第二次delta走访；现有planning计时不能把全部成本归给排序。

这些是历史u16访问实验，不与新的u32公平内存对照混用。

## 2026-09-30：未合并 PR 固定版本

PR #2348 head `6ac0de5359d9e0e1ed0608422575a360ef91b908`。WordArena每Symbol为u32 ID+u32 length，共8B；串行pair计数，一轮一条规则，历史word cohort索引，候选词数≥1000才启用4 worker扫描；复用scratch/delta缓冲。feed串行，训练Rayon4；原checkout干净，仅临时副本插入六个Instant阶段探针。

第一次PR512测试因我错误地把少量进程swap设成停止条件而提前结束；该失败记录保留，不作为训练耗时结论。初始化RSS估算4.57GiB也没有覆盖历史cohort、候选、出生记录的训练增长，后续估算必须加训练期峰值。

按用户纠正后的>1GiB余量规则，GPT-6 Luna完成同一512MiB PR4测试：Train约308.111秒，峰值RSS约6.57GiB，最低可用约1.52GiB，完整模型摘要与旧三项一致。最终数值、各阶段与采样记录在 [pr-fair-512.jsonl](results/pr-fair-512.jsonl) 及 environment JSON。此处不将PR与u16版本的内存差直接解释为算法收益；新的公平控制会使用u32。

## 正在进行：原接口、worktree 与公平控制

计划版本：HF reference、固定PR、串行endpoint、fused，以及u32并行串行初始化控制、u32并行初始化优化、u32原子访问对照。每个版本一个worktree/branch，原始BpeTrainer接口保持一致； benchmark驱动不需要外挂trainer方法。

主agent实现新的初始化线程控制与delta权重热点优化，GPT-6 Luna独立负责关键计时与数据。新增权重游标利用空间有序plans，同词复用、近邻最多前进8项、远间隔二分，避免稀疏批次完整线扫。完成后会记录源码commit、原接口验证和新的u32控制项结果。
