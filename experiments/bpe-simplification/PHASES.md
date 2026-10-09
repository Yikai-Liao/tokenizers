# 历史阶段诊断与优化依据

当前 2100 行源码的测量见 [PERFORMANCE_REVIEW.md](PERFORMANCE_REVIEW.md)。

固定main `e4f787dc`，候选whole `df2e71a1`（1849生产行）；中文ByteLevel固定word map，
4 workers，50K词表，min frequency2。两边同feature/release profile，一次新诊断，
不是为噪声追加重复。完整vocab IDs/ordered merges相同，child swap0。

## 1849 行候选：同口径粗阶段

| 阶段 | main wall s | whole wall s | 差值 s | main CPU s | whole CPU s |
|---|---:|---:|---:|---:|---:|
| vocabulary | .747 | .642 | -.105 | 2.287 | 2.098 |
| corpus plan | .722 | .771 | .048 | 1.765 | 1.842 |
| initial index | 3.484 | 3.520 | .035 | 12.463 | 12.092 |
| materialize | .756 | .944 | .187 | 2.592 | 2.709 |
| select | .355 | .670 | .315 | .445 | .812 |
| prepare | 8.889 | 10.869 | 1.979 | 30.219 | 38.761 |
| apply | 1.313 | 3.419 | 2.105 | 4.564 | 8.324 |
| commit | 2.471 | 7.069 | 4.597 | 7.075 | 15.348 |
| end release | .183 | 2.773 | 2.590 | .182 | 2.771 |
| model output | .021 | .023 | .002 | .021 | .023 |
| public train | 18.947 | 30.702 | 11.755 | 61.664 | 84.819 |

计时和总训练时间闭合至约.01秒。两边1646批次。prepare包含已选列表释放；
commit包含事件释放。CPU使用process clock计入全部worker。core跳过feed，
初始索引差距目前已经消失；旧1719行Perf的阶段比例不能替代本表。

## Commit进一步细分

新的子段诊断保持算法和动作顺序，串行路由精确计时，owner使用thread CPU并累加。
并行owner的wall相互重叠，不能把子段wall直接相加成协调器wall。

| 子段 | main wall s | whole wall s | main CPU s | whole CPU s |
|---|---:|---:|---:|---:|
| serial routing | .245 | 2.753 | .241 | 2.735 |
| owner grouped/count/complete/partial，合计CPU | — | — | 5.973 | 11.766 |
| main ordered counts | — | — | 3.108 | — |
| main completed publish | — | — | 1.857 | — |
| main partial grouping | — | — | .017 | — |
| main partial reduce/encode/publish | — | — | .991 | — |
| whole mixed owner loop | — | — | — | 11.687 |
| whole partial sort/prune/publish | — | — | — | .078 |
| full commit coordinator | 2.299 | 6.996 | 6.891 | 15.061 |

本轮commit差4.697秒，serial routing差2.508秒（约53%）；其余约2.19秒
落在并行owner工作。当前owner循环混合更新、直接发布、partial分组和append，
本次不能把这部分分别归因到单一优化。不能把main .991与whole .078直接当
整个partial成本比较，因为whole分组/append仍在11.687秒混合循环中。

## 对应机制与取舍

main的路由使用16B数字引用，复用route capacity，并跳过零removal/空birth。
whole将整个Change移动到owner，跨owner创建第二份大记录，空动作也被路由，
每批重新分配route。这与phase_route的串行差距对应，是下一项窄提取的依据。
末端partial排序本身很短，不能拿邻居目录实验填补整段差距。

complete普通producer直接State+queue快路已保留。main编码前floor剪枝，
whole编码后剪枝的差异发生在prepare，不能解释commit差4.6秒。

这轮诊断后，用户重新开放 Arena，并将生产预算提高到 2100 行。
下面的新诊断与实现取代了早期的无 Arena 取舍。

诊断使用隔离worktree，不向交付engine加入计时或实验开关。脚本
`phase_timing.py`、`phase_commit_detail.py`生成粗阶段/细分标记。
原始stdout/stderr、source/binary/patch hashes保存在results；紧凑证据见
`evidence/phase-summary-current.json`、`evidence/commit-summary-current.json`。

## Apply：收集和释放的成本

对相同 main 和 1849 行候选另做一次细分诊断。main 完整训练 19.508s，
候选 31.642s；完整模型相同，swap0。线程 CPU 用于 job 内部，process CPU
用于并行协调器。job wall 互相重叠，不能相加为协调器 wall。

| 子段 | main wall s | 候选 wall s | main CPU s | 候选 CPU s |
|---|---:|---:|---:|---:|
| apply 总段 | 1.374 | 3.476 | 4.762 | 8.552 |
| 并行执行 | 1.337 | 1.883 | 4.646 | 6.776 |
| 串行收集和释放 | .032 | 1.588 | .092 | 1.756 |
| job 内部 CPU，main 12207 / 候选 50028 jobs | — | — | 4.297 | 6.441 |
| 候选 endpoint loop | — | — | — | 6.261 |

收集和释放差 1.556s，占 apply wall 差约 74%。主要整改是保留 job 的
`Vec<Change>` 分块，交给 commit 消费，不再在 apply 协调器构造并释放一份
大平面 Change 数组。endpoint loop 的剩余成本不能全部归因于位置列表释放。
诊断补丁和逐段记录见 `evidence/apply-*-diagnostic.patch`、
`evidence/apply-diagnostic-provenance.json` 和 `evidence/apply-and-descriptor-phases.json`。
在指定 base 的独立 worktree 上 `git apply` 对应补丁即可复现插桩；诊断二进制
使用相同 runner、features 和 release profile。

## 1996 行 Arena 描述符：剩余成本

该版使用 main 的 16B 最终列表描述符，并把可变 raw builder 分离。未插桩训练
为 31.533s，不能作为达标交付；另一次插桩训练为 25.829s，二者不能混用。
插桩各段初始索引 4.990s、prepare 10.452s、apply 1.901s、commit 5.285s、
结束释放 .191s。它显示 Arena 已减少结束逐列表释放，但原始列表聚合和 owner
编码仍有可处理成本。没有用插桩总时间声称达到性能目标。

对应后续改动是首片段移交、跨轮编码 scratch 复用、可变缓冲 u32 自动提升到 u64，
以及完整 producer 在 floor 剪枝之后提前编码，owner 直接发布。最终是否达标只按
未插桩 runner 的完整公开训练边界判断，结果见 REPORT.md。
