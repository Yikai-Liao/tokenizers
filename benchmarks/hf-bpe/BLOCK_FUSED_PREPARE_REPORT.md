# 多 block 融合 prepare：实现与测量

## 采用的版本

采用 `1959202f30673fc21e68ab2868ba40213062c409`。它从 affix 最终默认版本 `fd300ca8` 继续实现，包含第一版片段方案 `d4723fc2`，再将有效偏移数组合并到每个 job。ROOT 的整个 `tokenizers` 子树已恢复为采用版本，避免将增量补丁套在归档源码上。

本次修改普通快速引擎的多 block 字典路径。公共默认 posting block 位数仍为 32；下表通过隔离 benchmark 显式设置 16，实际路线均为 `parallel_u32_dict16`。小语料公共默认会选择 flat32，因此不能用本表替代普通 flat 路径的训练时间。前后缀通过既有身份激活检查进入该引擎；实际 ID 复用仍完整重建 HF cohorts。

## 实际机制

实现位于 [fused_batch/block.rs](../../tokenizers/tk-train/src/trainers/bpe/indexed/parallel/fused_batch/block.rs)。

1. 任务只保存一次源 block、规则 rank、posting 切片范围。按源 block、rank、切片起点建立任务，Rayon 可以乱序执行，结果按逻辑顺序收集。大 posting 切片，多个小 posting 共用 job；单个 job 的选中 posting 数量限制在 4096 到 1048576 之间。
2. 每个 job 共用连续 `Vec<u32>` 保存有效局部偏移。切片描述保存该数组的范围，避免每个小 posting 再分配一个数组。访问语料时才还原全局位置。
3. 复用 `Selected`，从改写前的全局语料判断相邻选中规则。block 只划分索引；跨 block 的合并与相邻合并不需要边界消息或位置排序。权重游标从各源 block 的 pivots 和 previous_weight 查询。
4. 出生链节点仍为 8 字节：局部偏移与 next。32 字节片段描述保存目标 block、pair、链头、位置数和权重和。右出生属于合并起点所在 block；左出生按真实左 token 起点计算，允许跨多个 block。邻居目录按方向复用，宽 ID 使用稀疏哈希。
5. 源 posting 由切片持有共享所有权；最后一个切片完成后释放，不统一保留到最终 posting 分配之后。prepare 全部 join 后才并行改写；全部改写 join 后才提交和开始下一批。
6. 全局 owner 账本先汇总精确频率并应用门槛。接受的片段只路由一次，按目标 block 统计每个 pair 的总位置数，最终 posting 一次分配，按逻辑任务顺序反向填链。局部零权重片段仍保留物理位置。

首次激活 replacement 与现有批次证书保证新 pair 的唯一规则和方向生产者。右出生位置随输入起点递增；固定左邻居的跨度不变，左出生也递增。按源 block 和切片顺序拼接即可得到有序 posting。此分支不建立逐位置 `Plan`，不混合排序位置，也不拼接全局 Plan。AA 和非共享槽位保留原路径，因此总 `plan_ms` 仍可能非零。

## 测量结果

每个案例只有一次正式运行。四个 merge worker、四个初始化 worker，最小频率 2，无 affix。英文词表 30000，中文词表 50000。训练时间不含 feed；阶段时间存在嵌套，fused prepare 包含在 delta 和 merge 内。

| 数据 | 基线 train s | 采用版 train s | train 变化 | merge 变化 | 基线 / 采用版 RSS GiB |
|---|---:|---:|---:|---:|---:|
| EN16，255 blocks | 6.522 | 6.210 | −4.78% | −4.47% | 0.582 / 0.543 |
| ZH256，1555 blocks | 48.819 | 47.689 | −2.32% | −2.84% | 4.905 / 4.896 |

两组完整模型 SHA、实际 merge 数、初始符号/边数和 posting visits 与各自基线一致。实际 OS 线程峰值均为 5，VmSwap 为 0，最低 MemAvailable 超过 2.15 GiB。主分配 session 的 heap requested/freed、arena requested/retired 一致；推测 session 均为零。

第一版独立切片数组的英文 train 为 5.977 s、中文为 48.316 s；中文 merge 相对基线增加 0.26%，不能将总训练下降解释成 merge 收益。采用版减少小数组分配，并在两组中均测得 merge 下降。它的英文单次时间比第一版高，故不宣称它在所有案例中最快，也不把单次差异当成稳定收益。

### 时间如何转移

| 阶段，ms | EN 基线 | EN 采用版 | ZH256 基线 | ZH256 采用版 |
|---|---:|---:|---:|---:|
| initialize | 302.2 | 249.9 | 8006.4 | 8044.0 |
| merge | 6133.7 | 5859.3 | 40031.4 | 38892.9 |
| Plan | 1678.2 | 12.5 | 8401.5 | 82.0 |
| delta，含融合 prepare | 1499.9 | 2449.3 | 6330.3 | 12848.3 |
| rewrite | 81.0 | 116.6 | 305.5 | 384.1 |
| commit | 302.2 | 374.7 | 2414.7 | 2713.4 |
| route | 1836.2 | 2161.3 | 14974.3 | 16257.1 |

Plan 的时间大幅减少，旧邻边遍历的工作移入融合 prepare；新增片段汇总、最终 posting 安装与改写调度抵消了部分节省。中文 route 仍超过 16 秒，是本次结果中明确保留的成本，不能仅根据 Plan 下降宣称同等比例的训练加速。表中阶段均为墙钟累加；相邻阶段的细小波动没有单独的因果估计。

### 临时内存与测量边界

| 采用版容量指标，MiB | EN16 | ZH256 |
|---|---:|---:|
| 有效 u32 偏移 | 8 | 8 |
| 出生链节点 | 32 | 32 |
| 出生片段描述 | 8 | 64 |
| 输入任务描述 | 0.625 | 2.5 |
| Selected | 0.458 | 0.763 |
| worker scratch 容量上界 | 0.842 | 1.526 |

各指标是跨批次容量最大值，不能相加作为同时峰值；scratch 是最多四个执行 job 的容量上界。有效范围描述、外层 Vec、输出账本、片段路由索引和最终安装临时哈希表没有被上述字段全部单列，它们包含在进程 RSS 中。出生链与源 posting 的存活重叠也包含在 RSS 中。取消 `16M` 字节 Plan 逻辑载荷不意味着 RSS 净减少 `16M`。

## 512 MiB 范围调整

初始计划为 EN16、ZH512 各基线和候选。ZH512 的旧字典基线在 20.395 秒触发 `MemAvailable <= 1 GiB` 停止：已观察 RSS 5.87 GiB、VmSwap 424.3 MiB，未产出有效模型，候选未启动。它不是成功的零 swap 测量。

随后从同一中文文件取不超过 256 MiB 的完整行前缀：268435162 字节、666896 行，SHA256 为 `61537f422d250f93ed669193faec992a516ce049b41ad33fb02b52064780ed98`。记录见 [输入 manifest](results/block-fused-prepare/zh-256m-input-manifest.json)。两种片段版本在此输入完成比较。

实际 native 调用 **7 次：6 次成功、1 次资源停止**。保留 [原计划](results/block-fused-prepare/run-plan.json)、失败结果、后续案例及 [汇总](results/block-fused-prepare/analysis.json)。未覆盖 512 MiB 字典完整训练，也未性能测量超过 4 GiB 槽位的宽 offset 字典；宽 offset 的小 fixture 有语义验证。

## 验证与来源

- 第一版整个 tk-train 库 **86/86** 测试通过，见 [完整日志](results/block-fused-prepare/full-86-tests.log)。
- 采用版追加完整 trace/vocab/merges 差分矩阵：一/四 worker、16/32 位原子语料、u16/u32 offset、AA、零权重、special ID、有限长度，与 HF 和旧非共享路径一致，见 [trace 日志](results/block-fused-prepare/pooled-offset-trace-tests.log)。
- 采用版追加直接跨块 fixture：左 token 跨三个以上 block、合并本身跨块、相邻两规则跨块、零权重位置、wide ID 稀疏目录，见 [fixture 日志](results/block-fused-prepare/pooled-fragment-tests.log)。
- 孤立构建没有给生产 API 增加环境开关。`build_affix_analysis.py --posting-block-bits 16` 只修改复制的 benchmark 配置；运行器强制核对实际 dict16 路线及 `fused_block_batches > 0`。
- 三个源码 freeze、逐文件 SHA、二进制 SHA、构建命令与完整隔离 patch 保存在 [provenance](results/block-fused-prepare/provenance)。选定 binary SHA 为 `f0bb2e20e1520e409fa46a54232a11ec579d2748a3fc98c7a55a0552c7e94717`。

运行记录和资源检查可由 `python3 benchmarks/hf-bpe/analyze_block_fused_prepare.py` 汇总。原首轮运行脚本拒绝覆盖既有结果；256 MiB 调整与偏移数组迭代的命令分别保存在对应 environment JSON 中。
