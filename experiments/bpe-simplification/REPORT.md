# 最终简化引擎与取舍

交付分支 `simplify/bpe-maintenance-20261009`，固定Fork main基线 `e4f787dc189d9be7192107490d652096cde7480e`。
引擎生产1858行，相对main5714行减少67.5%；包含独立oracle和共享helpers的测试1439行，
低于生产预算且相对main6192行减少76.8%。均为rustfmt之后非空、非注释逻辑行。
实现模块26→6；没有将生产逻辑移出计数范围。

保留完整词表ID、所有ordered merges、兼容优先级前缀批次、并行prepare/apply/owner commit、
AA左到右选择、reserved/active身份与重启、signed reuse cohort及严格birth length规则。
posting支持完整U64域，包括独立codec中的u64::MAX；resident分配仍受usize/isize限制。
Vocabulary、Corpus、PairIndex、Batch/Prepared、Positions各自隐藏身份、几何、优先级、
快照事件和编码细节；上层训练流程仍可直接阅读。

Arena优化按用户明确决定放弃，接受其退休/结束释放成本；没有引入共享pool替代协议。
最终恢复的是紧凑路由：48B只读metadata、16B动作引用（本机64bit）、每posting一次owner
移交、route容量复用、零removal/空birth过滤。零权重但有positions的birth仍保留，
owner原序及removal-before-birth不变。完整普通producer直接State+queue快路已保留；
partial、AA、reuse仍完整聚合。

一次相邻main/候选验证覆盖四个case。English/Chinese ByteLevel使用256MiB文本与固定
word map，4 workers，50K词表，min frequency2。同样opt3/fatLTO/单codegen unit、
无默认features。机器为6-vCPU Xeon KVM。全部八进程完整model相同、child swap0。

| case | main train s | lean train s | main pipeline s | lean pipeline s | main RSS GiB | lean RSS GiB |
|---|---:|---:|---:|---:|---:|---:|
| en-core | 1.144 | 1.933 | 0.000 | 0.000 | 0.170 | 0.270 |
| zh-core | 19.308 | 29.043 | 0.000 | 0.000 | 2.445 | 3.034 |
| en-pipeline | 1.115 | 1.829 | 5.106 | 5.829 | 0.176 | 0.269 |
| zh-pipeline | 19.564 | 28.586 | 24.207 | 33.819 | 2.375 | 2.973 |

core表的pipeline0表示不执行feed。RSS为serialization/validation之前的process HWM，
包含加载或feed；不是单独engine存储。单对样本不能提供稳定百分比或CI；用户明确要求
不为机器噪声增加重复，未运行额外六对矩阵。中文core约1.50×main，pipeline总耗时约1.40×；
英文core约1.69×，pipeline约1.14×。这些是代码上限下的实际退化，不宣称达到main性能。

阶段定位见[PHASES.md](PHASES.md)。1849行whole候选对main差11.75s中，commit+结束释放
约占61%，prepare仅约17%；初始索引已基本相同。进一步诊断确认serial routing .245→2.753s，
占commit4.697s差距约53%。剩余owner混合更新/发布/partial成本尚未逐项因果归因。
最终1858行组合中文core29.043s，相比whole历史31.048s低约6.5%；
组合还包括30行纯删除整理，不能将全部差值单独归因到路由。

累计结果用于再次筛选组合，而不是沿迭代顺序默认保留所有机制：

| 组合/单项 | 生产行 | zh core s | CPU s | RSS GiB | 决策 |
|---|---:|---:|---:|---:|---|
| first correct archive | 1456 | 70.92 | — | 3.29 | 独立档案95638077，不作为交付 |
| weights/fixed writes | 1558 | 46.98 | 136.67 | 3.277 | 保留基础 |
| stream birth | 1576 | 40.22 | 126.67 | 3.312 | 保留；后续owned append替代转码 |
| deferred+dense/map_init | 1681 | 39.90 | 123.52 | 3.079 | 保留更短map_init，删除Mutex scratch |
| cached geometry | 1703 | 38.39 | 119.02 | 3.094 | 保留组合，不独称小增益显著 |
| hardware prefetch | 1737 | 35.49 | 108.11 | 3.161 | 保留 |
| owned stream append | 1776 | 32.94 | 96.08 | 3.143 | 保留 |
| typed cursor | 1797 | 30.87 | 88.38 | 3.207 | 保留 |
| Arena growth / delta | 1924 / 1928 | 36.84 / 33.91 | 115.74 / 101.37 | 3.548 / 3.504 | 不保留 |
| Inline16 + payload freeze | 1909 | 35.87 | 101.38 | 3.175 | 不保留，不等价main完整Arena链 |
| same Inline16 NoArena | 1845 | 34.01 | 101.35 | 3.280 | 不保留；相邻控制也未胜出 |
| complete / whole candidate | 1838 / 1849 | 30.96 / 31.05 | 85.44 / 85.48 | 3.123 / 2.990 | 保留完整普通快路/whole调度 |
| raw U64 writes | 1849 | 31.38 | 84.87 | 3.022 | 不保留，整体未受益 |
| fresh owner directory | 1903 | 32.55 | 89.53 | 3.132 | 不保留，未解决串行路由 |
| final compact route + cleanup | 1858 | 29.04 | 81.77 | 3.034 | 交付 |

历史表是不同时间的探索样本，多个条目是累计组合，不能做独立因果差值。
这是记录内的组合筛选，未穷举所有开关，也不宣称全局最优。各有效轮次均保留commit，
原始1456行另有独立archive分支。原始主工作区未改动；没有push或新建PR。

默认library tests36通过，无默认features36通过；all-target Clippy `-D warnings`、
rustfmt、格式行预算和git whitespace检查通过。小fixture比较完整模型及每条
(pair,count,replacement ID)；覆盖affix/alias/zero-weight、wide numeric错误、严格length、
跨partial floor聚合、fullU64不齐restart/seek、65536真实ID、feed/reload/progress/线程策略。
独立fresh审查逐轮进行，最终紧凑路由审查见[REVIEW-11.md](REVIEW-11.md)。

[evidence](evidence/)含final runs、源/二进制/输入hashes、格式行数、测试和阶段证据。
全部原始测量和immutable binaries保存在`/root/code/tokenizers-simplification-results/`。
最终源文件已逐个核对与发布runner的hash一致。`run_pairs.py`、`summarize.py`和独立
诊断脚本用于复核。PR #2501的HF main/Fork YTTM/HF PR #2348三基线未重跑；
不同主机/词表规模旧结果不混入本报告。
