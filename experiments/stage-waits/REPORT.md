# BPE 阶段等待实验：本轮保留 baseline

在中英文 256 MiB、非 Byte-level、50K、4 线程的对照中，corpus 写回／commit 重叠没有显示稳定收益；初始索引的两个流水线版本也没有优势。因此本轮保留 `8faaff79` baseline，三个候选只保留在实验分支。AA 准备在单次诊断中仅占总训练的 0.38%／0.65%，本轮没有实现第 3 项。

## 1. 固定对照与统计口径

先独立测试第 1 项，再把第 2 项的最小版本与完整版本放在同一轮中测试。两轮各自使用本轮 baseline；下表不能用于跨轮比较绝对耗时。每个单元预热一次，正式执行三轮交错对照。指标为逐轮 candidate / baseline 比例的中位数；负百分比表示耗时下降。观测范围是三轮最小／最大比例，不是置信区间。

公开 `do_train` 调用计时包括 corpus plan、初始索引、corpus 物化、merge 循环和输出词表构造，排除输入 JSON 加载、预分词、输出序列化与模型验证。CPU 时间使用进程 CPU 时间；VmHWM 在输出序列化前读取，包含输入和训练内存。

| 对照组／候选 | 中文耗时变化 | 中文观测范围 | 英文耗时变化 | 英文观测范围 |
| --- | ---: | ---: | ---: | ---: |
| 1：写回与 commit 重叠 | -0.50% | -6.79%～+0.81% | -2.66% | -9.36%～+5.80% |
| 2a：每个分区排序＋计数 | +0.42% | +0.33%～+2.47% | +4.23% | -22.74%～+10.72% |
| 2b：每个分区排序＋计数＋编码 | +2.26% | -2.49%～+2.87% | +11.69% | -21.23%～+26.67% |

第 1 项的两个语料均出现变慢的一轮；第 2 项中，2a 的中文三轮均稍慢，2b 的中文三轮有两轮稍慢。英文第 2 组的墙钟与 CPU 时间都有较大波动，保留全部样本。不能根据这组结果把差异归因为内存带宽争抢，也不能认定这些方案在其他负载上必然无效。

沿用此前条件：至少一个语料三轮均更快，且配对中位数降低至少 5%，才进入 1 线程退化检查。这是实验筛选条件，不是统计显著性判据。本轮三个候选均未通过，故没有运行 1 线程，也没有将两个方向组合。

## 2. 原值、CPU 与内存

绝对耗时为各 arm 三次正式运行的中位数；配对比例不能用两个绝对中位数相除替代。内存按 KiB / 1024 转为 MiB。

| 对照组 | 语料 | arm | 训练秒 | CPU 秒 | VmHWM MiB | CPU 配对变化 | VmHWM 配对变化 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| overlap-t4 | zh | baseline | 6.918879 | 22.134797 | 1453.49 | — | — |
| overlap-t4 | zh | overlap | 6.801390 | 22.198497 | 1452.82 | +1.31% | -0.05% |
| overlap-t4 | en | baseline | 0.795187 | 2.330349 | 146.16 | — | — |
| overlap-t4 | en | overlap | 0.769113 | 2.265621 | 142.62 | -3.30% | -2.58% |
| initial-t4 | zh | baseline | 6.732779 | 21.561235 | 1453.57 | — | — |
| initial-t4 | zh | pipeline | 6.885247 | 21.630799 | 1460.16 | -0.06% | +0.45% |
| initial-t4 | zh | sortcount | 6.869563 | 21.668165 | 1451.70 | +0.34% | -0.05% |
| initial-t4 | en | baseline | 0.768467 | 2.274649 | 143.91 | — | — |
| initial-t4 | en | pipeline | 0.858319 | 2.375694 | 145.09 | +4.44% | +0.81% |
| initial-t4 | en | sortcount | 0.800975 | 2.340779 | 146.16 | +2.91% | +1.52% |

## 3. 改动与失败处理

**第 1 项**：准备完成并释放候选位置列表后，`PreparedMerges::apply_with_commit` 在同一训练池中用 `rayon::join` 执行写回与 commit。commit 只读准备好的 events，并更新原 pair index；不读取 corpus。两侧全部结束后才释放 events、进入下一轮或返回错误。没有增加位置副本或检查点；写计划的存活期延长到 commit 结束，实际进程峰值见上表。原 owner 的计数、路由、编码和错误清理代码保持原实现。

新增测试在 1／4 worker、endpoint／occurrence-span 两种写回路径下，注入 commit 错误和 panic，并验证返回或展开栈前所有计划写入均已完成。原有 HF 的有序 `(pair, count, id)` 轨迹、身份复用、AA 奇偶、溢出和完整模型测试继续通过。

**第 2 项**：2a 只将计数移到每个分区排序后，保留编码屏障。2b 继续使用原编码算法，让同一分区完成排序、计数、编码后释放原始记录；所有分区结束后再检查错误和发布。计数错误仍优先于编码错误，波次间的检查、append、过滤顺序保持原实现。worker 的目录／codec／arena lease 只覆盖顺序工作，不跨越嵌套并行任务。

两者均保持原 record、radix 排序、owner 路由和 SortedPositions 编码。bounded collector 保持原路径；本次中英文初始字母表分别为 17,759 和 5,083，均超过其 256 字符门槛，实际覆盖修改的 keyed collector。

| 候选 | 无默认 feature 的完整 lib 测试 | Clippy `--all-targets --no-default-features -- -D warnings` |
| --- | ---: | --- |
| overlap | 102 通过 | 通过 |
| sortcount | 101 通过 | 通过 |
| pipeline | 101 通过 | 通过 |

## 4. AA 诊断与第 3 项决定

单独从 baseline 构建诊断 binary，只给 `aa::prepare` 加墙钟计时。该阶段包含验证位置、计算跨块奇偶、chosen 筛选和写计划准备，因此其全部耗时可作为本方向的理想可消除成本上限。诊断使用相同语料和参数，每个语料只跑一次诊断与一次 baseline，并检查完整模型；这些运行不加入前两组候选排名。

| 语料 | AA 准备次数 | 累计候选位置 | AA 准备秒 | 诊断总训练秒 | 诊断占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| zh | 137 | 391,377 | 0.026360 | 6.976660 | 0.378% |
| en | 92 | 137,331 | 0.008449 | 1.290329 | 0.655% |

占比来自各自的单次诊断运行，不是 baseline 的精确阶段占比；计时与输出诊断信息会影响执行。本组英文总训练明显慢于前两组，也保留其原值。即使将诊断中的整个 AA 准备阶段消除，节省也不到该次总训练的 1%。按照“只有重复字符 merge 占时明显才值得做”的条件，本轮暂不实施 AA 筛选／准备融合；重复字符密集负载尚未测。

## 5. 输入、构建与证据

两种输入均为固定 Wikipedia snapshot 的完整行前缀，实际中文 268,435,162 字节，英文 268,434,441 字节；Whitespace，50,000 目标词表，min_frequency=2，无 affix／max length。CPU 0–3，4 worker，Xeon Gold 6140 KVM 主机（可用 6 vCPU、约 16 GiB RAM）。所有正式 binary 使用同一 runner 锁文件、release fat LTO、codegen-units=1、关闭 tk-train 默认 features。

共 44 次成功运行：30 次正式计时、10 次预热、4 次 AA 诊断／控制。两组正式对照各完成 6／6 配对块；诊断 2／2 块。没有模型失败、计时失败、排除块或资源门中断，采样 Swap 最大值为 0。逐次完整词表 ID 和有序 merges 均一致；正式语料没有导出逐步频率 trace，逐步 `(pair, count, id)` 对照由测试夹具覆盖。

| arm | 冻结源码 commit | 固定 binary SHA-256 |
| --- | --- | --- |
| baseline | [8faaff79](https://github.com/Yikai-Liao/tokenizers/commit/8faaff79d859bfd6b2417cfe8c93ea2851c3aaca) | `070717e030355fb20d7ac13e9a39af744d94afd66179d2ffaa2ab8f7a3da2901` |
| overlap | [90e85812](https://github.com/Yikai-Liao/tokenizers/commit/90e85812ff3a5feb5f19cf61c63ddab63d0c3934) | `419b3a0155ebcff419505c89e2f53463806b75a8b3e4e3633cdccd39b327ee33` |
| sortcount | [09e9efa4](https://github.com/Yikai-Liao/tokenizers/commit/09e9efa40ae3156f66918a7d80bd96266daab167) | `5e47ca62d9d77558289bc67878290c13bd8eca5d4db35466fb985e3beb486976` |
| pipeline | [459d42d8](https://github.com/Yikai-Liao/tokenizers/commit/459d42d8642704477f372d2cd3b5014510d56d3b) | `e3bda9bcbe903dafd2ab5e22b477570608b668f1ced6bbae76ef33da0bc20792` |
| diagnostic | [3465a942](https://github.com/Yikai-Liao/tokenizers/commit/3465a942645ab82f700e7075d2536bf922ff17c5) | `a1b700f48febc9164e04113326c5ebb60d08e3fe636a37751760e1377353b169` |

源码分支：[overlap](https://github.com/Yikai-Liao/tokenizers/tree/experiment/stage-waits)、[sortcount](https://github.com/Yikai-Liao/tokenizers/tree/experiment/initial-sort-count)、[pipeline](https://github.com/Yikai-Liao/tokenizers/tree/experiment/initial-pipeline)、[diagnostic](https://github.com/Yikai-Liao/tokenizers/tree/experiment/stage-waits-diagnostic)。构建后增加的报告文件不改变冻结源码或已测 binary。

可复算的证据：

- [overlap summary](evidence/overlap-t4/summary.json) 与 [原始归档](evidence/overlap-t4/raw-metadata-and-models.tar.gz)。
- [initial summary](evidence/initial-t4/summary.json) 与 [原始归档](evidence/initial-t4/raw-metadata-and-models.tar.gz)。
- [AA 累计计时和每次原值](evidence/aa-timing.json)、[诊断 summary](evidence/aa-diagnostic/summary.json) 与 [原始归档](evidence/aa-diagnostic/raw-metadata-and-models.tar.gz)。
- [继续条件与逐轮比例](evidence/decision.json)、[构建来源](evidence/builds.json)、[构建记录摘要校验](evidence/provenance.json) 和 [验证日志摘要](evidence/verification.json)。
- [复现命令](README.md)、[对照脚本](run.py) 与 [证据导出脚本](export_evidence.py)。

归档保留运行配置、host、job、结果、stdout／stderr、memory 采样、harness 源码和 runner 锁，以及输入 manifest。模型先逐次验证，再仅合并相同的模型值：按词面排序 vocabulary 以消除 JSON 对象顺序，保留每个 token ID 和 merges 原顺序；每次原始模型文件的 SHA-256 与规范化模型指纹在 `models.json` 中。完整输入和 binary 未上传；来源及 SHA 均可核验。
