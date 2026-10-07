# Owner 优化候选实验

比较对象是 fork main `8faaff79d859bfd6b2417cfe8c93ea2851c3aaca`。四个独立分支从该版本开始，共用生产阶段的长链检查点和连续压缩流的协作编码器。

| 候选 | 分支 | 本轮数据处理和状态发布 |
| --- | --- | --- |
| stable，旧路线 A | `experiment/owner-stable-group` | 按完整 pair key 稳定 radix 分组；数据任务按 key 检查删除状态机、聚合出生列表、协作编码，最后由唯一状态 Owner 发布。 |
| epoch，旧路线 B | `experiment/owner-epoch-index` | 对不可变 AVL epoch 计算完整 key replacements；按更新 key 递归分裂、合并并共享未修改的子树，join 后切换根节点。根维护精确 count／pair argmax，压缩列表通过只读 Arc handle 共享。 |
| bucket，新路线 A | `experiment/owner-independent` | 按规范 rule／direction bucket 稳定汇集，复用 dense neighbor accumulator；按本轮工作量调度编码任务，Owner 保留有序删除和完整结果发布。 |
| blocks，新路线 B | `experiment/owner-block-positions` | 沿用 bucket 聚合，把大列表交付为独立压缩块和 ordinal 目录；小片段合并到有界任务，发布连接目录，后续 cursor／seek 直接访问块。 |

完整 producer 的已编码 births 直接发布。active-ID reuse 沿用原兼容路径。选择顺序、同频 pair tie、weighted counts、floor 退休和 feed 契约仍由现有语义测试对照原训练器验证。所有候选的长链检查点按 4096 个 positions 建立，每个 chain handle 从 12B 增加到 16B，`NeighborChanges` 从 32B 增加到 40B；这些代价计入候选。独立块只用于至少 16,384 个 positions 的 residual 列表，保留 `SortedPositions` 的 16B handle，额外块目录和 16B allocation alignment 计入内存和训练时间。

实验按用户确认的配置执行：中文和英文各取固定 512 MiB 语料的完整行前缀，目标 256 MiB；普通 BPE，`Whitespace` 预分词，50K vocabulary，min frequency 2，4 workers。各候选先完成实现、测试和 immutable release 构建，然后与 Baseline 在同一批次顺序执行。每个 case／arm 有一次 warmup 和三个交错 paired blocks。

主指标是公开 `do_train` 的完整训练时间，涵盖训练表示、初始索引、所有 merge 和 vocabulary 输出；预分词、prepared JSON 加载和模型验证不在计时范围内。每个输出对照 Baseline 比较 vocabulary IDs 和完整 ordered merges。内存同时保留 validation 前的进程 VmHWM 和监督进程采样的峰值 RSS；它们与精确分配字节数分别解释。

这一组用于初筛。只有某候选在至少一个语种的三个 paired blocks 都更快、且 paired ratio 中位数至少改善 5%，才补该候选的 1-worker 中文和英文比较。这个门槛用于决定是否继续测量，统计显著性需要结合实际波动判断。

使用 [run.py](run.py) 时，将 `tokenizers-bpe-benchmarks` 放到 `PYTHONPATH`，传入五个已冻结的 `build.json`、两份 prepared manifests 和输出目录。默认一次运行全部候选；补测可使用 `--workers 1 --arms bucket` 等选项。各分支设置 `BPE_OWNER_MODE=baseline` 可运行该分支内的原 Owner 路径；blocks 分支还支持 `BPE_OWNER_MODE=monolithic` 检查连续流编码。正式 Baseline 使用独立的 main binary。

候选来自两份用户提供的研究材料。稳定分组参考 [SPAA’23 Semisort](https://www.cs.ucr.edu/~ygu/papers/SPAA23/semisort.pdf)，epoch 的 join／bulk 思路参考 [Joinable Parallel Balanced Binary Trees](https://www.cs.cmu.edu/~blelloch/papers/3512769.pdf)；bucket 协作任务的研究参照包括 [DuckDB Window 执行](https://duckdb.org/2025/02/14/window-flying)，独立压缩块参考 [PaC-trees](https://arxiv.org/abs/2204.06077v1)。此处 Rust 实现独立编写，完整训练收益由本实验决定。

实测数据、各分支 code commit 和 binary hash 见 [REPORT.md](REPORT.md)。[evidence](evidence) 保留完整 summary、每轮原值、测试日志，以及压缩的构建记录、主机信息、进程输出和模型参照。模型按语种去重，只按拼写排序词表来规范化表示，保留原 token ID 和完整 merge 顺序；每个 attempt 的原始文件 hash 和规范模型 hash 见 `evidence/models.json`。

`export_evidence.py --comparison <run.py 的输出目录> --out <证据目录>` 从已完成且校验通过的比较导出证据，并计算是否满足补测单线程的条件。它和 `run.py` 使用本次归档的 benchmark harness；将 harness 所在目录放到 `PYTHONPATH`。`plot.py --summary evidence/summary.json --out training-ratios` 用 matplotlib 重建图表。
