# 最终版本阶段开销与内存规模

结论：当前额外 CPU 主要落在 prepare；256MiB 的 initial index 也有明显差距。更大中文语料中，初始收集/索引阶段的内存比例差距明显扩大，但未插桩全程 HWM 没有按同样比例扩大。两种指标必须分开看。

源码固定为 main `e4f787dc` / 最终 `4d181c51`；ByteLevel、4 workers、CPU0–3、50K 词表、min_frequency=2。256/384/512MiB 是同一个真实中文文件的嵌套前缀，原 256MiB 文件已逐字节校验一致；没有通过重复文本制造规模。准备只做一次，每对使用同一 prepared word map。

## 未插桩全程内存

| 中文规模 | 去重词数 | main train s | 最终 train s | main HWM GiB | 最终 HWM GiB | 最终 HWM 超出比例 |
|---|---:|---:|---:|---:|---:|---:|
| 256MiB | 6,859,987 | 16.004 | 21.248 | 2.437 | 2.943 | 20.8% |
| 384MiB | 10,024,214 | 22.249 | 30.601 | 3.756 | 4.599 | 22.4% |
| 512MiB | 13,035,637 | 30.193 | 44.312 | 4.726 | 5.695 | 20.5% |

这是三个规模各一对独立进程的真实观测：全程 HWM 的增幅约 20%–22%，这组数据未显示明显比例放大。HWM 在公开 train 结束、模型序列化/校验前读取，包含输入加载和 feed 之外的进程存储，不能代表初始索引或 Arena 单独占用。所有完整模型一致、child swap0；测量期间没有其他基准/构建。

| 中文规模 | E physical edges | S resident slots | main waves |
|---|---:|---:|---:|
| zh-256MiB | 189,067,447 | 202,787,422 | 1 |
| zh-384MiB | 278,657,154 | 298,705,583 | 2 |
| zh-512MiB | 363,847,704 | 389,918,979 | 2 |

按 prepared JSON 流式独立统计字符数、词数与 E/S，全部非空、plain 字符保留；256/512 的 E/S 与实际诊断日志相同。384 的两 waves 按 main 循环 `(S-1).div_ceil(2^28)` 派生，未把它写成新增实际 route 日志。统计详见 [scaling-geometry.json](evidence/scaling-geometry.json)。

## 初始收集与物化边界

下表来自另外四个完整训练诊断进程。main 的 initial collect 与最终版 PairIndex::build 均在 materialize 之前，期间都提高了进程 HWM，因而这里的 HWM 是该阶段实际达到的峰值。main 额外的 initial publish 在 materialize 后进行，仅约 0.0004–0.0005s，未增加 HWM；粗阶段 CPU/wall 将它合并计入 initial index。

| 边界 / 指标 | 256 main GiB | 256 最终 GiB | 512 main GiB | 512 最终 GiB |
|---|---:|---:|---:|---:|
| corpus plan 结束 RSS | 0.784 | 0.835 | 1.522 | 1.619 |
| initial collect/build 峰值 | 1.848 | 2.417 | 2.868 | 4.713 |
| initial collect/build 结束 RSS | 1.151 | 2.414 | 2.524 | 4.645 |
| materialize 结束 RSS | 1.427 | 2.638 | 3.056 | 5.079 |
| 合并阶段最大观测边界 RSS | 2.384 | 3.124 | 4.697 | 6.129 |
| release 结束 RSS | 1.416 | 1.920 | 2.793 | 3.512 |

256MiB 初始阶段峰值：main 1,938,248KiB / 最终 2,534,760KiB，最终高 30.8%。

512MiB 初始阶段峰值：main 3,007,364KiB / 最终 4,941,808KiB，最终高 64.3%。

实际日志确认 main 在两个规模都采用 **bounded** 收集，分别执行 **1 / 2 waves**，每 wave 上限 **2^28 resident slots**。最终版两次均使用 **16 个词块**，完整 raw pieces 收集后交给 owner append，没有物理 wave 上限。512MiB resident slots 为 389,918,979，已越过 268,435,456 的边界。

这次观测支持“初始阶段比例差距会扩大”：从约 30.8% 增至 64.3%。新旧还同时不同于直接分区/Builder、owner 聚合、提前编码及分配布局，因此不能把 33.5 个百分点全部归因于删除 waves，也不能据此断言所有语料或更多 waves 的固定比例。

RSS 结束值、各轮最大边界值及阶段峰值各有不同含义，表中明确区分。边界 RSS 不是某阶段的新增分配，也不是未采样的阶段峰值。诊断 512MiB 的全程 HWM 为 main 4.737GiB / 最终 6.162GiB（约高 30.1%），与未插桩的 20.5% 不同；独立进程的分配和调度会改变峰值，两组数据不能互换。

## 阶段 wall / CPU

两份隔离副本在 joined 协调器边界使用相同计时器。CPU 是 process clock，计入全部 worker。prepare 包含已选列表释放，commit 包含事件释放；因此 main 的 call 数较多，但每项 wall/CPU 是不重叠子段的合计。

### 256MiB

| 阶段 | main wall s | 最终 wall s | wall 差 s | main CPU s | 最终 CPU s | CPU 差 s |
|---|---:|---:|---:|---:|---:|---:|
| vocabulary | 0.611 | 0.630 | +0.020 | 2.242 | 2.203 | -0.040 |
| corpus plan | 0.645 | 0.668 | +0.023 | 1.764 | 1.832 | +0.068 |
| initial index | 2.569 | 4.139 | +1.570 | 10.025 | 13.141 | +3.116 |
| materialize | 0.699 | 0.774 | +0.075 | 2.625 | 2.673 | +0.047 |
| select | 0.311 | 0.442 | +0.131 | 0.307 | 0.438 | +0.131 |
| prepare | 7.895 | 9.191 | +1.296 | 26.246 | 33.387 | +7.141 |
| apply | 1.048 | 1.271 | +0.223 | 3.579 | 4.655 | +1.076 |
| commit | 2.083 | 2.797 | +0.714 | 5.768 | 6.417 | +0.649 |
| release | 0.087 | 0.157 | +0.070 | 0.087 | 0.154 | +0.067 |
| model output | 0.023 | 0.032 | +0.010 | 0.023 | 0.032 | +0.010 |
| 未归属协调/插桩开销 | 0.933 | 0.739 | -0.194 | 1.311 | 1.152 | -0.159 |
| public train | 16.903 | 20.842 | +3.938 | 53.976 | 66.084 | +12.107 |

### 512MiB

| 阶段 | main wall s | 最终 wall s | wall 差 s | main CPU s | 最终 CPU s | CPU 差 s |
|---|---:|---:|---:|---:|---:|---:|
| vocabulary | 1.331 | 1.229 | -0.102 | 4.814 | 4.436 | -0.378 |
| corpus plan | 1.394 | 1.326 | -0.068 | 3.962 | 3.716 | -0.246 |
| initial index | 7.077 | 7.851 | +0.774 | 26.230 | 24.972 | -1.259 |
| materialize | 1.312 | 1.580 | +0.268 | 4.964 | 5.542 | +0.578 |
| select | 0.350 | 0.560 | +0.210 | 0.346 | 0.555 | +0.208 |
| prepare | 13.720 | 17.872 | +4.153 | 47.404 | 63.459 | +16.055 |
| apply | 1.750 | 2.664 | +0.914 | 6.348 | 9.649 | +3.301 |
| commit | 3.600 | 5.164 | +1.565 | 9.361 | 11.802 | +2.442 |
| release | 0.193 | 0.268 | +0.075 | 0.193 | 0.267 | +0.075 |
| model output | 0.023 | 0.021 | -0.002 | 0.023 | 0.021 | -0.002 |
| 未归属协调/插桩开销 | 1.131 | 0.962 | -0.169 | 1.494 | 1.375 | -0.118 |
| public train | 31.879 | 39.497 | +7.618 | 105.139 | 125.794 | +20.655 |

未归属部分包含计时器之外的 /proc 读取、pool/progress、协调器操作及报告生成，不把它归到某个算法。插桩总时间只用于诊断；25 秒目标使用 REPORT.md 的未插桩 21.248s，不使用这里的更短或更长总时间。

256MiB 的额外 CPU 为 12.107s，其中 prepare +7.141s（约 59%），initial index +3.116s（约 26%）；两项约占差额 85%。initial index 的 CPU/wall 为 main 3.90 / 最终 3.18，说明串行段和执行重叠也有差别，不能仅凭这个比值证明是词块负载不均。

512MiB 的额外 CPU 为 20.655s，prepare +16.055s 占约 78%；initial index CPU 反而少 1.259s，wave 聚合与再编码等工作发生在不同路径。当前 prepare 两个规模均占最终版约一半 process CPU，仍是速度改进的首要位置；初始收集则是内存改进的首要位置。Materialize 当前 wall 差分别仅约 0.075 / 0.268s，不能继续套用早期候选的热点占比。

## 取舍与证据

下一项预算内实验应优先替换当前 raw pieces 的存活/聚合布局，并单独验证工作集上限的收益；直接恢复整个 main collector 会重新引入大量边界和状态。速度侧先检查 prepare 中的匹配/列表处理成本。这里提出的是由阶段数据支持的方向，尚未把未实现方案记作收益。

现有组合的净行数与历史收益、已删除低收益目录及 radsort 原 gate 见 [OPTIMIZATION_ROI.md](OPTIMIZATION_ROI.md)。阈值 E/S 的三语料三规模消融见 [CUTOFF_ABLATION.md](CUTOFF_ABLATION.md)。当前 main 走 bounded，因此这两个 case 的初始差距不能归因于删除 radix/radsort 排序。

八个大规模/诊断进程的完整记录、阶段数据和 main 初始事件见 [scaling-runs.json](evidence/scaling-runs.json)，一致前缀输入与 hashes 见 [scaling-inputs.json](evidence/scaling-inputs.json)。诊断源码、补丁、binary hashes 和编译口径见 [full-review-provenance.json](evidence/full-review-provenance.json)。所有计时器和实验开关只存在于隔离 worktree，交付 Rust 没有插桩。
