# 在线初始压缩与 prepare 优化：逐项证据

本实验选择 **整词分块＋生产者内压缩，不加 wave 屏障；再加入规则索引和有序写入**。规则索引是 prepare 的主要收益来源。有序写入仅增加 3 行，在已有索引时，两种规模的 prepare CPU 均进一步降低约 2–3%，因此一起保留。候选位于独立工作树，生产 Rust 尚未合入原 2100 行分支。

完整候选为 **2185 行生产／797 行测试**，符合 2200／800 预算；比原版本多 85／17 行。诊断观察器和 runner 不进入生产补丁。下列数字有不同计时边界：初始阶段峰值、整个训练峰值、prepare CPU 和完整训练墙钟分别报告。

## 先判断初始阶段哪些组合有价值

四个实验臂共用同一二进制、相同的整词分块、临时压缩片段和最终编码。raw-all 先收集全部块再压缩；online-all 在每个生产者结束时立即压缩并释放原始列表；wave 在每四个块后加入屏障。每个正常块上限为 2²⁴ 个 resident slot；超过上限的单个词独立处理。频率准入仍使用全局完整计数。

表内为两轮观测的中位数；CPU 为进程 CPU，峰值单位 GiB。原 2100 行版本到 raw-all 的差额属于分块与片段表示这一组变更，未进一步拆成单个机制的因果贡献。

| 输入 | 臂 | 初始 wall s | 初始 CPU s | 初始进程峰值 GiB | 训练峰值 GiB |
|---|---|---|---|---|---|
| zh-256MiB | baseline | 4.378 | 13.586 | 2.457 | 2.847 |
| zh-256MiB | raw-all | 4.170 | 14.969 | 1.783 | 2.636 |
| zh-256MiB | online-all | 4.173 | 15.703 | 1.457 | 2.414 |
| zh-256MiB | raw-wave | 4.216 | 15.259 | 1.458 | 2.432 |
| zh-256MiB | online-wave | 4.037 | 14.637 | 1.457 | 2.410 |
| zh-512MiB | baseline | 8.924 | 27.532 | 4.635 | 5.762 |
| zh-512MiB | raw-all | 8.008 | 29.197 | 3.409 | 5.060 |
| zh-512MiB | online-all | 7.606 | 28.715 | 2.862 | 4.638 |
| zh-512MiB | raw-wave | 8.354 | 29.860 | 2.861 | 4.582 |
| zh-512MiB | online-wave | 8.342 | 29.941 | 2.863 | 4.636 |

| 输入 | 条件变化 | 初始 wall 变化 | 初始 CPU 变化 | 初始峰值变化 |
|---|---|---|---|---|
| zh-256MiB | 分块与片段表示组 | -4.7% | +10.2% | -27.4% |
| zh-256MiB | 已有分块，再加在线压缩 | +0.1% | +4.9% | -18.3% |
| zh-256MiB | 已有在线压缩，再加 wave | -3.3% | -6.8% | +0.0% |
| zh-512MiB | 分块与片段表示组 | -10.3% | +6.0% | -26.4% |
| zh-512MiB | 已有分块，再加在线压缩 | -5.0% | -1.6% | -16.0% |
| zh-512MiB | 已有在线压缩，再加 wave | +9.7% | +4.3% | +0.0% |

在线压缩在无 wave 时显著降低初始峰值；wave 对 raw 路径也有效。但 online-all 与 online-wave 的初始峰值几乎一致：在线释放已经控制了同时驻留的原始块。wave 的额外调度屏障因此省略，整个训练峰值的微小变动不作为另加机制的依据。

1 GiB 的原 2100 行诊断臂触发了自身 swap 保护（16 KiB），该次完整训练时间与峰值对比排除；其已完成的初始阶段仅保留为部分日志。1 GiB 旧矩阵只完成部分轮次；旧计划中的无插桩对照被用户要求的阶段分析取代，未完成项不补写为结果。

## 把此前的训练差距拆到阶段

此前无插桩 main 与仅初始段插桩候选的跨轮次墙钟差约 23%，不能直接分摊。下表使用随后同一种十阶段观察器、同一 1 GiB 样本的正序配对；按用户要求取消了反序。main 为复杂实现，初始候选为 2165 行在线压缩版本，尚未加入本轮 prepare 优化。

| 阶段 | main wall | 候选 wall | Δ wall s | main CPU | 候选 CPU | Δ CPU s |
|---|---|---|---|---|---|---|
| vocabulary | 2.558 | 2.576 | +0.018 | 9.426 | 9.352 | -0.074 |
| corpus_plan | 2.625 | 2.957 | +0.332 | 7.427 | 8.256 | +0.829 |
| initial_index | 16.359 | 14.746 | -1.613 | 60.353 | 56.430 | -3.923 |
| materialize | 3.319 | 4.121 | +0.802 | 12.460 | 12.560 | +0.100 |
| select | 0.416 | 0.616 | +0.200 | 0.407 | 0.610 | +0.203 |
| prepare | 28.607 | 34.775 | +6.168 | 99.622 | 126.515 | +26.893 |
| apply | 4.593 | 5.473 | +0.880 | 16.329 | 20.243 | +3.914 |
| commit | 6.705 | 7.493 | +0.789 | 20.154 | 19.336 | -0.818 |
| release | 0.421 | 0.341 | -0.080 | 0.417 | 0.340 | -0.077 |
| model_output | 0.023 | 0.021 | -0.002 | 0.023 | 0.021 | -0.002 |
| 未覆盖间隙 | 1.407 | 1.262 | -0.145 | — | — | — |
| 未覆盖 CPU | — | — | — | 1.745 | 1.675 | -0.071 |

同观察器完整训练 wall 67.031 → 74.380 s（+11.0%），CPU 228.363 → 255.338 s。prepare 是最大增加项；初始索引阶段反而更快。未覆盖间隙包含协调和观察器等，不能全部归给算法。

## prepare 核心差异与细分插桩

main 使用可复用的 `SelectedRuleIndex`：head／tail 直接按 token ID 索引，只有共享端点才回退到精确 pair 哈希。main 还把多个小规则放入同一作业，并使用可复用的窄位置 scratch。初始候选在扫描中查询普通 pair 哈希，并通过 `Builder::push` 重复验证有序性；两者的作业组织和事件构建方式也不同。

这里的主要差距出现在 merge 前准备的扫描／收集，并非 radsort。main 初始索引的通用排序路径与 prepare 是不同阶段；ByteLevel 的有限字母表另有专用索引路径。

细分观察器用 joined 进程 CPU 计整个并行段，用线程 CPU 计每个工作段；没有逐 occurrence 的时钟或原子累加。工作线程墙钟会重叠，不能相加当作阶段墙钟，线程子阶段也已包含于 worker total。

| 细分 | CPU 时钟 | main CPU s | 候选 CPU s | Δ CPU s |
|---|---|---|---|---|
| selected_index | process | 0.009 | 0.007 | -0.003 |
| ordinary_joined | process | 91.932 | 124.403 | +32.470 |
| aa_joined | process | 0.323 | 0.175 | -0.148 |
| worker_total | thread | 90.565 | 123.379 | +32.814 |
| worker_setup | thread | 0.069 | 3.258 | +3.188 |
| scan_collect | thread | 72.192 | 102.050 | +29.858 |
| finish_encode | thread | 17.834 | 17.704 | -0.130 |
| postjoin_gather | process | 0.658 | 0.000 | -0.658 |

这一对 prepare CPU 92.302 → 124.586 s；扫描／收集增加 29.858 s，约占 prepare 增量 92.5%。两者都访问 591,821,980 个位置，匹配 568,585,911 次；差异来自每次访问成本及作业组织，而不是少做了输入工作。main 普通作业 20,261 个，初始候选 49,950 个；本轮没有单独消融小规则分组，因此不把它的贡献写成已测百分比。worker setup 的放置边界也不完全一致，主判断使用 scan 与 joined prepare。

## 各项优化单独及叠加收益

C＝2165 行在线初始候选，O＝仅有序写入，L＝仅规则索引，OL＝二者。每规模每臂一次，顺序 C→O→L→OL；属于探索结果。表中正数表示 CPU 节省。条件收益以被加优化前的那个版本为分母。

| 输入 | C prepare CPU | O | L | OL | C→O | C→L | C→OL | L→OL | O→OL |
|---|---|---|---|---|---|---|---|---|---|
| zh-512MiB | 63.702 | 63.433 | 59.125 | 57.826 | 0.269 | 4.577 | 5.876 | 1.299 | 5.607 |
| zh-1024MiB | 154.402 | 129.464 | 118.674 | 115.237 | 24.937 | 35.727 | 39.165 | 3.437 | 14.228 |

| 输入 | O 单独收益 | L 单独收益 | 组合收益 | 已有 L 再加 O | 已有 O 再加 L | 交互 CPU s |
|---|---|---|---|---|---|---|
| zh-512MiB | 0.4% | 7.2% | 9.2% | 2.2% | 8.8% | +1.030 |
| zh-1024MiB | 16.2% | 23.1% | 25.4% | 2.9% | 11.0% | -21.500 |

交互项定义为 O＋L−C−OL；非零意味着两项收益不能简单相加。本轮 1 GiB 控制臂的 CPU 偏高，O 单独收益从 512 MiB 的 0.4% 跳到 16.2%，因此不能将 16.2% 认定为稳定因果收益。L 的主力判断结合了更小规模、扫描细分与两种条件对比；O 的保留理由是已有 L 后约 2–3% 的同向增量和仅 3 行的代价，不是单独臂的大数字。

L 增加 17 行生产／4 行测试：只对唯一 head 直接查 counterpart／replacement；共享 head 使用原精确哈希，tail gate 先拒绝不可能的左侧命中，保留边界与别名语义。O 仅在单调 snapshot 写入的两个调用处使用有序追加；通用追加与最终 encoder 继续验证。组合没有新增公共模式或配置。

| 输入 | 版本 | 训练 wall s | 训练 CPU s | prepare wall s | 扫描线程 CPU s | 训练峰值 GiB |
|---|---|---|---|---|---|---|
| zh-512MiB | control | 38.069 | 124.810 | 17.771 | 51.112 | 4.642 |
| zh-512MiB | ordered | 39.331 | 129.673 | 17.705 | 50.432 | 4.616 |
| zh-512MiB | lookup | 37.166 | 123.119 | 16.526 | 46.112 | 4.572 |
| zh-512MiB | combined | 37.282 | 122.743 | 16.195 | 44.388 | 4.710 |
| zh-1024MiB | control | 84.784 | 290.711 | 42.497 | 131.274 | 8.169 |
| zh-1024MiB | ordered | 79.876 | 271.713 | 35.937 | 105.909 | 8.157 |
| zh-1024MiB | lookup | 73.069 | 248.335 | 32.885 | 95.612 | 8.125 |
| zh-1024MiB | combined | 71.502 | 244.734 | 32.006 | 91.926 | 8.159 |

512 MiB 完整训练 CPU 的组合节省只有 1.7%，墙钟节省 2.1%；不能把 prepare 的 9.2% 当作完整训练收益。1 GiB 完整训练 CPU 单次节省 15.8%，受到控制臂偏慢影响。O 加到 L 后，512 MiB 完整墙钟略增 0.116 s，1 GiB 减少 1.567 s；这里不声称稳定的完整训练墙钟加速。

## ByteLevel 与 Whitespace：在相同分词器内比较

同一英文／中文 256 MiB 原始语料分别经两种预分词器处理。ByteLevel 使用 regex 和 byte→Unicode 映射、完整 byte 字母表；Whitespace 使用 Unicode 词边界和自然字母表。两种输入的训练难度不同，所以比较候选相对同预分词器 main 的开销，再看二者差异。core 读取预先冻结的 WordCounts；pipeline 包括公共 feed 和 train。表中的 feed CPU 用 pipeline CPU 减 train CPU 推导，包含两段之间的协调间隙；runner 原始输出没有独立 feed CPU。每配置每臂 core／pipeline 各一次，没有反序。

| 语言 | ByteLevel 候选/main CPU | Whitespace 候选/main CPU | Whitespace 初始索引额外 CPU | 占全部额外 CPU |
|---|---|---|---|---|
| en | +22.2% | +14.1% | 0.020 s | 3.9% |
| zh | +21.0% | +110.4% | 20.306 s | 81.7% |

英文 ByteLevel 的相对 CPU 开销更高，中文却是 Whitespace 明显更高；不能推出 ByteLevel 普遍放大差距。四份 main 日志均记录 ByteLevel 走 bounded、Whitespace 走 keyed。中文 Whitespace 的最大差额在初始索引，已不是 prepare；main keyed 路径以 radix::sort_by_key 排序（从 radsort 的 radixsort_permuted.c 移植），简化候选使用块内 pair 累积和临时片段压缩。没有单独关闭 radix 的消融，无法把该阶段的全部差额都算作 radsort 的贡献。

英文 prepared corpus 不到一个 2²⁴-slot 块，候选初始索引仅有一个生产者；其初始墙钟高于 main，而 CPU 差额小得多。固定大块在小型输入上的并行粒度也是实际代价，本轮没有调小块大小。

因此这个候选可以显著改善原简化实现的中文 ByteLevel 内存，但它不是覆盖所有预分词器、可替代 main 的性能方案。特别是中文 Whitespace 的通用初始索引差距依然很大。

### core

| 输入 | 版本 | 训练 wall | 训练 CPU | 初始 wall | 初始 CPU | prepare wall | prepare CPU | 峰值 GiB |
|---|---|---|---|---|---|---|---|---|
| en-256MiB | main | 1.599 | 3.688 | 0.105 | 0.340 | 0.432 | 1.291 | 0.160 |
| en-256MiB | baseline | 1.766 | 4.575 | 0.172 | 0.539 | 0.579 | 1.917 | 0.190 |
| en-256MiB | candidate | 2.083 | 4.509 | 0.451 | 0.452 | 0.583 | 1.867 | 0.182 |
| en-whitespace-256MiB | main | 1.594 | 3.661 | 0.104 | 0.337 | 0.423 | 1.213 | 0.137 |
| en-whitespace-256MiB | baseline | 1.690 | 4.282 | 0.145 | 0.463 | 0.527 | 1.698 | 0.144 |
| en-whitespace-256MiB | candidate | 1.979 | 4.179 | 0.354 | 0.358 | 0.554 | 1.690 | 0.137 |
| zh-256MiB | main | 16.053 | 51.199 | 2.635 | 9.876 | 7.342 | 24.817 | 2.426 |
| zh-256MiB | baseline | 20.989 | 65.999 | 4.410 | 13.931 | 9.290 | 32.961 | 3.072 |
| zh-256MiB | candidate | 18.970 | 61.953 | 3.862 | 14.934 | 8.114 | 28.863 | 2.441 |
| zh-whitespace-256MiB | main | 7.389 | 22.505 | 1.673 | 6.423 | 2.203 | 7.057 | 1.426 |
| zh-whitespace-256MiB | baseline | 15.505 | 46.297 | 8.146 | 25.864 | 2.869 | 9.706 | 2.008 |
| zh-whitespace-256MiB | candidate | 16.555 | 47.356 | 9.033 | 26.729 | 2.926 | 9.697 | 1.814 |

| 输入 | 版本 | 相对 main 训练 wall | 训练 CPU | 初始 CPU | prepare CPU | 峰值 |
|---|---|---|---|---|---|---|
| en-256MiB | baseline | +10.4% | +24.0% | +58.5% | +48.5% | +19.1% |
| en-256MiB | candidate | +30.2% | +22.2% | +32.9% | +44.6% | +13.9% |
| en-whitespace-256MiB | baseline | +6.0% | +17.0% | +37.3% | +40.0% | +5.1% |
| en-whitespace-256MiB | candidate | +24.1% | +14.1% | +6.0% | +39.3% | -0.5% |
| zh-256MiB | baseline | +30.7% | +28.9% | +41.1% | +32.8% | +26.6% |
| zh-256MiB | candidate | +18.2% | +21.0% | +51.2% | +16.3% | +0.6% |
| zh-whitespace-256MiB | baseline | +109.9% | +105.7% | +302.7% | +37.5% | +40.8% |
| zh-whitespace-256MiB | candidate | +124.1% | +110.4% | +316.1% | +37.4% | +27.2% |

### pipeline

| 输入 | 版本 | 训练 wall | 训练 CPU | 初始 wall | 初始 CPU | prepare wall | prepare CPU | 峰值 GiB |
|---|---|---|---|---|---|---|---|---|
| en-256MiB | main | 1.710 | 3.866 | 0.109 | 0.324 | 0.486 | 1.378 | 0.159 |
| en-256MiB | baseline | 1.813 | 4.640 | 0.173 | 0.515 | 0.580 | 1.922 | 0.186 |
| en-256MiB | candidate | 1.933 | 4.202 | 0.382 | 0.385 | 0.560 | 1.749 | 0.178 |
| en-whitespace-256MiB | main | 1.531 | 3.561 | 0.127 | 0.414 | 0.379 | 1.107 | 0.140 |
| en-whitespace-256MiB | baseline | 1.630 | 4.132 | 0.147 | 0.427 | 0.520 | 1.701 | 0.136 |
| en-whitespace-256MiB | candidate | 1.867 | 4.042 | 0.348 | 0.356 | 0.520 | 1.603 | 0.132 |
| zh-256MiB | main | 15.111 | 48.796 | 2.481 | 9.355 | 7.003 | 23.666 | 2.384 |
| zh-256MiB | baseline | 20.113 | 63.633 | 4.167 | 13.167 | 9.030 | 32.187 | 2.773 |
| zh-256MiB | candidate | 19.720 | 63.527 | 3.993 | 15.042 | 8.594 | 30.021 | 2.412 |
| zh-whitespace-256MiB | main | 7.343 | 21.967 | 1.640 | 6.226 | 2.207 | 7.085 | 1.340 |
| zh-whitespace-256MiB | baseline | 15.930 | 47.102 | 8.431 | 26.313 | 2.893 | 9.882 | 2.001 |
| zh-whitespace-256MiB | candidate | 17.262 | 48.684 | 9.282 | 26.970 | 3.225 | 10.443 | 1.793 |

| 输入 | 版本 | feed wall s | feed CPU s | pipeline wall s | pipeline CPU s |
|---|---|---|---|---|---|
| en-256MiB | main | 4.206 | 16.547 | 5.916 | 20.413 |
| en-256MiB | baseline | 3.754 | 14.788 | 5.567 | 19.428 |
| en-256MiB | candidate | 3.604 | 14.115 | 5.537 | 18.317 |
| en-whitespace-256MiB | main | 3.877 | 15.328 | 5.409 | 18.889 |
| en-whitespace-256MiB | baseline | 3.486 | 13.825 | 5.116 | 17.957 |
| en-whitespace-256MiB | candidate | 3.448 | 13.615 | 5.315 | 17.657 |
| zh-256MiB | main | 4.191 | 15.766 | 19.302 | 64.562 |
| zh-256MiB | baseline | 3.726 | 14.048 | 23.839 | 77.682 |
| zh-256MiB | candidate | 3.952 | 14.889 | 23.672 | 78.416 |
| zh-whitespace-256MiB | main | 3.669 | 13.656 | 11.012 | 35.623 |
| zh-whitespace-256MiB | baseline | 3.555 | 13.226 | 19.485 | 60.328 |
| zh-whitespace-256MiB | candidate | 3.434 | 12.627 | 20.696 | 61.310 |

| 输入 | 版本 | 相对 main 训练 wall | 训练 CPU | 初始 CPU | prepare CPU | 峰值 |
|---|---|---|---|---|---|---|
| en-256MiB | baseline | +6.0% | +20.0% | +59.0% | +39.4% | +16.9% |
| en-256MiB | candidate | +13.0% | +8.7% | +19.0% | +26.9% | +11.9% |
| en-whitespace-256MiB | baseline | +6.5% | +16.0% | +3.2% | +53.6% | -3.2% |
| en-whitespace-256MiB | candidate | +21.9% | +13.5% | -14.0% | +44.8% | -5.8% |
| zh-256MiB | baseline | +33.1% | +30.4% | +40.7% | +36.0% | +16.3% |
| zh-256MiB | candidate | +30.5% | +30.2% | +60.8% | +26.9% | +1.2% |
| zh-whitespace-256MiB | baseline | +116.9% | +114.4% | +322.6% | +39.5% | +49.4% |
| zh-whitespace-256MiB | candidate | +135.1% | +121.6% | +333.2% | +47.4% | +33.9% |

| 输入 | 候选/main pipeline wall | 候选/main pipeline CPU |
|---|---|---|
| en-256MiB | -6.4% | -10.3% |
| en-whitespace-256MiB | -1.7% | -6.5% |
| zh-256MiB | +22.6% | +21.5% |
| zh-whitespace-256MiB | +87.9% | +72.1% |

### 候选相对原 2100 行版本的取舍
+
+下面的分母是同组 baseline，而非 main。新机制的内存收益比完整训练速度收益更一致；英文固定大块的初始并行度代价、中文 Whitespace 的时间回退均保留在结论中。feed 的实现相同，其单次差额不归给这些训练优化。

| 输入 | 模式 | 训练 wall 变化 | 训练 CPU 变化 | prepare CPU 变化 | 峰值变化 |
|---|---|---|---|---|---|
| en-256MiB | selected-phases | +17.9% | -1.4% | -2.6% | -4.3% |
| en-256MiB | phases-pipeline | +6.6% | -9.4% | -9.0% | -4.3% |
| en-whitespace-256MiB | selected-phases | +17.1% | -2.4% | -0.5% | -5.4% |
| en-whitespace-256MiB | phases-pipeline | +14.5% | -2.2% | -5.8% | -2.8% |
| zh-256MiB | selected-phases | -9.6% | -6.1% | -12.4% | -20.5% |
| zh-256MiB | phases-pipeline | -2.0% | -0.2% | -6.7% | -13.0% |
| zh-whitespace-256MiB | selected-phases | +6.8% | +2.3% | -0.1% | -9.7% |
| zh-whitespace-256MiB | phases-pipeline | +8.4% | +3.4% | +5.7% | -10.4% |

## 正确性、复现边界与交付

全部有效训练检查完整 vocabulary ID 与有序 merge 列表。实际 vocab／merge 数随初始字母表变化，在各次输出中记录，不能把 ByteLevel 的 49,744 merges 套到 Whitespace。选定对照不得出现 swap 或并行 cargo／rustc／其他 benchmark。观察器固定十阶段，release opt3／fat LTO／codegen-units 1，workers 4，affinity 0–3，min_frequency 2，Cargo.lock 和 source／binary／patch SHA 均已记录。

宿主是共享 VM；检查只排除编译器与 benchmark 并发，不能宣称宿主完全静默。CPU 与墙钟都受时间段影响，单样本百分比不等于置信区间。1 GiB partial 与取消的反序、被取代的无插桩计划均留有原始标记。未运行的计划不会出现在成功计数中。

干净组合候选已通过默认与无默认特性完整 native 测试（各 17＋1 doctest）、Clippy `-D warnings`、严格 provenance Miri（2 tests）及格式／行数检查。索引初版的 separator 越界由小规模语义测试捕获，在构建性能二进制前已修复；失败日志与修正后验证都保留。

生产补丁：[small-combined.patch](evidence/online-initial/small-combined.patch)。补丁基于 `7e77262b`，只修改生产实现和所需语义测试，可在独立树审阅与应用。未自动合入生产分支。当前报告和证据是本轮实验交付。

原始结果：[runs.jsonl](evidence/online-initial/runs.jsonl)；逐项摘要：[small-ablation-summary.json](evidence/online-initial/small-ablation-summary.json)；分词器摘要：[selected-summary.json](evidence/online-initial/selected-summary.json)；候选构建：[selected-full-manifest.json](evidence/online-initial/selected-full-manifest.json)。
