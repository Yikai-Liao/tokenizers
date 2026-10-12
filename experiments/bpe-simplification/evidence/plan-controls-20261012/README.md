# BPE 目录复用与完整发布及 FRONTIER 窗口实验

本轮按 [实施计划](references/implementation-plan.md) 对三个接缝分别做独立对照。**三项候选均不进入正式生产实现**：attempt scratch 在英文有可重复的训练收益，中文目标没有稳定净收益；producer 完整发布在英文稳定回退；FRONTIER 的跨批 birth 节省太小，未抵消 starts 重放和额外阶段的成本。报告提交只添加证据，生产实现保持基线 `890ea1f58e65f8d8503f7cd3fc97ad1651eaf284`。

共保存 126 次完整训练：84 次正常计时版本（其中 producer 首轮中文 Whitespace 的两次成对结果剔除性能结论），42 次重诊断版本。同一输入在所有版本间的完整模型内容相同，覆盖六种输入；输入哈希重新核验。模型、失败及剔除记录均保留。原生测试同时核对有序 pair/count/ID trace 和完整训练结果；窗口正例另逐轮核对实际 corpus 与顺序 token-list 改写结果。

## 参数与对照边界

真实场景沿用既有 256 MiB 中英文语料派生的 Whitespace / ByteLevel 准备输入，vocab_size=50000、min_frequency=2，无 affix、长度上限或 alphabet 上限。加载、预分词及模型序列化不计入训练 wall/CPU；计时边界为公开 `do_train` 调用。机器有 6 个可用 vCPU，主矩阵用 4 workers、CPU affinity 0–3。scratch 增加 1 worker 控制和中文反向复测；其余两个明显回退的候选没有扩展 worker 矩阵。按本机情况执行，没有运行 16 workers。

每个版本串行运行；Whitespace 首轮基线先，ByteLevel 首轮候选先，第二轮反向。release opt-level=3、fat LTO、codegen-units=1、locked/offline、CARGO_INCREMENTAL=0、CARGO_BUILD_JOBS=2。正常计时期间不运行编译、测试或 gzip 归档；窗口模型仅在该训练子进程退出并取得计时记录后压缩，再启动下一个进程。正常源仅在每个 joined phase 边界做计时与常数个 Relaxed 更新，没有逐坐标计数。

producer 最早一对中文 Whitespace 的 baseline 与约 9 秒 scratch 模型 gzip 核验重叠，故**整对剔除**，保留完整模型并补跑无竞争的一对。该场景最终比较 reverse 与 replacement 两对；其他场景比较 primary 与 reverse。没有筛掉单独较慢的 candidate。见 [性能纳入记录](performance-inclusions.json) 与 [原始剔除原因](publisher/excluded-runs.json)。

以下百分比均以候选相对对应基线计算，负数表示更快；均值百分比为两次时间算术均值之比。完整秒数、每对结果与阶段时间在 [summary.json](summary.json)、[normal-runs.csv](normal-runs.csv) 和各阶段 `all-normal.json` 中。

## attempt scratch

`ThreadLocal<RefCell<Directories>>` 的生命周期为一次 attempt。每个顺序准备 job 借用当前线程目录并在返回前释放；目录验证既有缓存槽是否仍指向本次同一邻居组，域单调增长时仅初始化新增尾部。错误、restart 和成功均通过 attempt 局部所有权释放。原有 job 切分、publisher 和 join 保持原状。

| 场景 | 训练 wall 变化 | 训练 CPU 变化 | 两对 CPU 变化 |
|---|---:|---:|---:|
| 中文 Whitespace | +0.52% | +0.59% | -0.01% / +1.18% |
| 中文 ByteLevel | -1.37% | -0.76% | +1.07% / -2.54% |
| 英文 Whitespace | -6.45% | -10.40% | -12.73% / -8.08% |
| 英文 ByteLevel | -7.63% | -7.68% | -8.41% / -6.98% |

英文两对 wall、CPU、prepare 均改善。中文 Whitespace 的整轮均值略慢，中文 ByteLevel 的小幅变化没有稳定的 prepare 改善。1 worker 中文 Whitespace CPU 首轮 +1.26%、反向复测 +6.43%；ByteLevel 首轮 +1.94%、反向复测 −3.17%。ByteLevel 增加没有持续出现，Whitespace 的增加重复出现，不能将其统一解释为噪声，也不能直接归因给 TLS 或 RefCell：Whitespace 反向对照中未改算法的 initial index 阶段同样慢了约 2.12 秒。

目录填充确实被删除，但没有因此获得中文训练净收益。下表为逻辑 sentinel 初始化量，GB/MB 使用十进制单位；它不是硬件测得的内存总线流量。四个实际参与线程的 scratch 总填充为 1.6 MB，目录向量峰值 capacity 同样约 1.6 MB。

| 场景 | scratch 基线逻辑填充 GB | attempt scratch MB | publisher 新 reducer 逻辑填充 GB |
|---|---:|---:|---:|
| 中文 Whitespace | 7.271 | 1.600 | 7.442 |
| 中文 ByteLevel | 7.637 | 1.600 | 7.855 |
| 英文 Whitespace | 6.991 | 1.600 | 7.406 |
| 英文 ByteLevel | 6.464 | 1.600 | 6.688 |

本轮不采纳 scratch；保留英文收益证据，不将目标负载的缺口转换为运行时负载选择策略。

## producer 完整发布

每个 producer 按原任务顺序移动 birth 叶子，使用左右 validated direct-ID directory 完整归组与 checked 求和，再做全局 floor 判断和一次 concat。完整 Candidate 的 Positions 可以已有 Root，直接进入 count owner 接缝，随后统一 heap 发布。old losses 与 reuse 的原有算术顺序保持原义。新增一次 producer join；没有为获得更大的 H/K 比加密 job 分片。

| 场景 | 训练 wall 变化 | 训练 CPU 变化 | 两对 CPU 变化 |
|---|---:|---:|---:|
| 中文 Whitespace | -0.24% | -0.44% | +0.76% / -1.63% |
| 中文 ByteLevel | +1.04% | +0.71% | +0.48% / +0.94% |
| 英文 Whitespace | +24.07% | +29.53% | +25.06% / +34.02% |
| 英文 ByteLevel | +12.53% | +17.93% | +19.90% / +16.17% |

英文两对 CPU 均回退。正常英文 complete commit 约从 0.25–0.29 秒增到 0.45–0.49 秒；新的 producer 阶段约 0.26–0.30 秒，owner 节省不足以补偿。中文有效两对变化很小。故不采纳，也没有将该发布器混入窗口实验。

| 场景 | H 碎片 | K 完整键 | K live | owner actions 基线 → 完整发布 |
|---|---:|---:|---:|---:|
| 中文 Whitespace | 13,128,107 | 13,063,335 | 4,769,069 | 22,966,606 → 17,889,559 |
| 中文 ByteLevel | 17,350,748 | 17,325,407 | 7,051,314 | 30,323,397 → 24,346,253 |
| 英文 Whitespace | 961,792 | 913,352 | 565,151 | 1,681,612 → 1,525,381 |
| 英文 ByteLevel | 1,046,588 | 992,829 | 630,851 | 1,830,980 → 1,675,842 |

H/K_total 在中文约 1.001–1.005、英文约 1.053–1.054，同一个完整键并没有很多跨任务碎片。H→K_live 路由与 sort fragments 工作确实下降；完整计数、floor、归组和 concat 被移到 producer，并未消失。空 fragments 的 sort 调用仍存在，因此只声称有载荷的 fragment 排序工作被移除。

新 reducer 的冷目录填充量见上表。诊断还显示英文每次训练新增约 34.5–34.7 万次分配。最大单 producer 线程 CPU 约为英文 0.47–0.48 ms、中文 ByteLevel 2.69 ms、中文 Whitespace 3.63 ms；这些来自重诊断，不能当作正常版尾部耗时。单 producer staging capacity checkpoint 最大约 15.31 MB（中文 ByteLevel）、1.45 MB（中文 Whitespace），是定义好的容量检查点，不能称作并发 staging 总峰值或 RSS。

## FRONTIER 窗口

三组使用同一个原有 publisher 和冷 `map_init` 目录：E 为原 eager 内核，S 为单 cohort deferred/replay 内核，W 为同一个 deferred 内核加 FRONTIER guard，最多四个 cohort。没有组合 scratch 或完整发布器，也没有加入 terminal discard。

窗口固定旧 ID 边界 B，guard 在 identity resolve/take 前检查；接受唯一追加输出，窗口内不消费输出。第一遍只保留 accepted starts 和 old losses，各 cohort 完成 prepare、disjoint writes 和 old-loss commit 后才继续。最终 replay 按较早 cohort、同 cohort 物理左侧的归属规则生成 births。旧 heap 为空、资源和 identity 边界均正常 drain；已激活 alias 从原输入进入 reuse restart。累计规则 cap 为 256，目标使用实际 vocabulary.len()。

| 场景 | S wall | S CPU | W wall | W CPU |
|---|---:|---:|---:|---:|
| 中文 Whitespace | +8.42% | +9.59% | +14.01% | +9.76% |
| 中文 ByteLevel | +29.75% | +33.11% | +28.57% | +31.34% |
| 英文 Whitespace | +18.57% | +31.31% | +20.21% | +32.47% |
| 英文 ByteLevel | +15.16% | +25.84% | +16.09% | +27.80% |

两种顺序中 W 对 E 的整轮 CPU 均回退。W 对 S 的变化方向也不稳定，例如中文 ByteLevel 首轮 CPU 少约 4.3%，反向复测反而多约 1.9%。没有将一次方向有利的样本解释为稳定跨批收益。

### 物理工作

| 场景 | E 与 S append 次数 | W append 次数 | W 跨 cohort owned cuts | E → W 净减少 |
|---|---:|---:|---:|---:|
| 中文 Whitespace | 49,133,128 | 49,130,437 | 2,760 | 2,691 (0.005477%) |
| 中文 ByteLevel | 280,205,796 | 280,205,119 | 587 | 677 (0.000242%) |
| 英文 Whitespace | 5,336,666 | 5,336,626 | 73 | 40 (0.000750%) |
| 英文 ByteLevel | 6,957,536 | 6,957,470 | 62 | 66 (0.000949%) |

E/S 的实际 append 数相同。W 净少的 append 极小；cross-cohort cuts 与 E→W 净差不必相等，因为 guard 会在不同位置截断 compatible prefix，改变 inner cohort 的分组。所有真实场景 accepted starts 数相同。S 的 weight-region 查找次数约为 E 的两倍；中文 ByteLevel 的 corpus token 总读取数由约 9.19 亿增到约 10.87 亿，starts 重放增加了真实读工作。该读取字段包含 matcher、邻居、canonical 和其他 merge 内部读取，不能单称为邻居读取次数。

W 的 outer windows 减少，但 inner cohorts 增加。正常中文 ByteLevel 的 prepare 从约 8.95 秒降到约 5.31 秒，另增加约 10.03 秒 replay，故仅报 prepare 降低会错误描述净效果。这是首轮阶段示例，全部正反记录均在原始 JSON 中。

| 场景 | E / S outer rounds | W outer windows | W inner cohorts | guard stops | 回看 drain 前旧 head 胜过最终 child | rare producer 最终坐标 |
|---|---:|---:|---:|---:|---:|---:|
| 中文 Whitespace | 1390 | 1166 | 1918 | 1165 | 1161 | 242 |
| 中文 ByteLevel | 1646 | 1398 | 2338 | 1397 | 1377 | 1,318 |
| 英文 Whitespace | 1491 | 1323 | 2295 | 1322 | 1284 | 7,107 |
| 英文 ByteLevel | 1335 | 1196 | 2071 | 1195 | 1191 | 5,187 |

W 的真实窗口只出现 1–3 个 cohort，未达到四 cohort 上限；每种场景最后一个窗口均因 target 正常 drain，其余均因 guard 停止。guard-stop 回看时，多数 drain 前可见旧 head 强于最终发布的最佳完整 child。原始 `blocked_priority` 字段取最后 old-loss join 后的 best，不一定是 prepare 前触发 guard 的那个 head；`guard_false_positive_at_drain` 只表示此 joined checkpoint 的回看比较。这说明 conservative frontier 确实留下潜在 admission 空隙；它**不是 exact M 必然放行同一 head 的证明**，因为 exact M 的 coherent-family bound 在较早 snapshot 上建立，最终 child 可能已经分裂或消失。

重诊断另在每个 joined prepare 后扫描 accepted starts，按唯一 owned coherent families、严格 span 门槛与完整 checked 聚合测每个 producer 的 M。M<F 的 producer 可省的最终坐标仅为表中数量；相应编码已用字节为中文 Whitespace 3,872、中文 ByteLevel 21,088、英文 Whitespace 113,712、英文 ByteLevel 82,992。整 producer 剪枝在本矩阵上的物理份额很低。该扫描在诊断版执行，没有把 exact M reducer 放入正常窗口计时。

FRONTIER 不采纳。guard 保守停止频繁且最终 child 留有空隙，符合计划中进一步试验 exact M admission 的触发条件；当前没有 exact M 的计时或收益结论。

## 内存与诊断边界

实验前明确的峰值预算为 candidate 增量 ≤ max(基线 5%, 16 MiB)。正常版两对的最大 process HWM 均通过，具体每场景峰值见 [summary.json](summary.json)。它包括已加载输入和训练临时存储，采样的 VmSwap 峰值均为 0；这不代表整台主机的全局 swap 活动为零。正常版 HWM 是内存采纳门槛，阶段 requested-byte 与目录容量是解释性诊断。

窗口 H 是 attempt 内累计历史最大单 cohort retained capacity，R 计 accepted-start/job Vec 的 capacity、spilled starts 与固定描述。仅 R≤H 才继续，joined checkpoint 保证 R≤2H。中文 ByteLevel 最大 R 与 H 容量检查点约 160 MiB，Whitespace 约 6.10 MiB；它们不是 corpus 总内存或 RSS 的界。第一遍释放 selected Positions 和 loss Changes，保留的 starts 到 drain 才释放。

publisher/window 的诊断目录结构增加 site 字段，header 为 56 B，生产 header 为 48 B。诊断 requested-byte allocator 从进程开始追踪 live bytes，训练开始重置 peak，因此计入输入；不包含 allocator rounding。window 的额外 coherent-family hash scan、逐坐标原子计数与 JSON 分析自身也计入诊断分配及时间，不能冒充生产峰值或性能门槛。prepare/replay 与 prepare/producer 的 directory sites 分开统计，不能相加后归因给一个接缝。

| 场景 | 重诊断 CPU / 正常首轮 CPU E | S | W |
|---|---:|---:|---:|
| 中文 Whitespace | 1.63× | 1.69× | 1.73× |
| 中文 ByteLevel | 3.88× | 3.26× | 3.45× |
| 英文 Whitespace | 2.52× | 2.42× | 2.33× |
| 英文 ByteLevel | 2.59× | 2.60× | 2.78× |

上表显示重诊断相对正常首轮的 CPU 比例，用于说明测量成本；不是同环境成对的纯插桩因果估计，也不用于采纳判断。工作计数与正常计时始终分开。

## 验证与代码预算

scratch 默认/无默认特性各 30 项 BPE 测试通过；publisher 与 W 各 31 项通过。E/S 控制组各通过 31 项默认特性测试。补充逐轮 corpus oracle 后，W 默认/无默认特性与 E/S 的对应单项测试再次通过，最终 W Clippy 通过。Clippy 仅放行基线已有 `redundant_field_names`。原有 JSON roundtrip 的已知“测”问题按此前基线跳过，没有计为通过。日志保存在各阶段 `validation/*.log.gz`。

窗口测试包括 ABBCD_i 的 AB→BC 正例、CDAB 中右侧较早 producer 的 cut 归属、AA、零权重、old-head tie、严格长度门槛、target 停止与无跨 cohort 改写形状；独立 eager 加权 word-list oracle 比较每步 pair/count/ID 和完整 parts，使用 1/4/8 worker 池。补充 corpus oracle 在两个归属正例的每个 complete round 逐条顺序改写 flat token list，再比较 live endpoints 与完整长度。8 workers 仅是小规模正确性检查，并非性能矩阵。完整模型包括 vocab-by-ID、有序 merges 和其余全部输出内容；六种输入跨所有候选与 worker 数均按实际字节匹配，见 [model-manifest.json](model-manifest.json) 和 `models/`。

| 独立候选 | Rust production AST 非空行净增 | 候选测试 AST 非空行 | 正式实现测试行 |
|---|---:|---:|---:|
| scratch | 21（另加直接 thread_local 依赖） | 1435 | 1428 |
| producer 完整发布 | 127 | 1467 | 1428 |
| FRONTIER | 255 | 1500 | 1428 |

计数排除注释，测试包含全部 cfg(test) 项与四行 Miri 入口；每项以同一基线独立计算，不把它们相加成组合版本。窗口最终补丁新增 corpus 验证只修改测试，实际计时冻结源仍保存原始测试版本（1478 行）；生产内核没有随此补充变更。各阶段 `candidate.patch.gz` 为基线上的最终完整差异，`source/` 为实际编译的正常/诊断冻结源。未把增行候选称为代码简化。

## 复现与后续阶段

`environment.json` 保存输入 SHA、工具链与 CPU；build.json 保存各二进制 SHA；原始参数、stdout、stderr gzip 和正常/诊断 JSON 均归档。normal 的阶段数据来自每批 joined phase，重诊断记录 logical hashes/actions、完整键准入/拒绝、实际 append、编码已用 bytes、fragment heap capacity、directory 分配/增长及 requested-byte peak。capacity 累加指标是跨阶段累计，不是同时存活峰值。`summarize.py` 可从原始 JSON 重算摘要。

可移植单任务回放要求干净、一次性的基线 checkout；覆盖实际冻结源、校验准备输入 SHA，再 locked/offline 构建运行：

```sh
python replay.py --checkout /path/to/disposable-base-checkout \
  --stage frontier --arm frontier \
  --job frontier/primary/zh-bytelevel_regex-w4-r0-frontier/job.json \
  --input /path/to/recorded-zh-bytelevel.json \
  --target-dir /path/to/target --output /path/to/run
```

对 E 使用另一个干净基线 checkout 和相应 job，同样运行并比较完整 model.json。AA/reuse 输入已归档在 `control-inputs/`。依赖需事先缓存；不同工具链/硬件的二进制 SHA 和耗时不能与本机结果混用。各阶段 builders 保留原机器路径作为过程证据；回放用相对 artifact 路径的 replay.py。候选源码预算在基线应用相应 patch 后，用 `uv run count-tests.py --base 890ea1f5 --output /tmp/count.json` 复核。

当前三项独立控制已完成。**初始 index 的 count-first/materialize 尚未实施，exact M 正常候选尚未实施**；下一阶段单独验证这两个方向，沿用完整模型、数值错误、正常计时与内存预算门槛，不将其收益归给当前 FRONTIER。动态 floor、terminal elision 与 async 仍是计划中的后续研究，本轮没有相关性能声明。
