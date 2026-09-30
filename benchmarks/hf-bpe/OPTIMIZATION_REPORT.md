# BPE 热点优化：并行构造与融合查询

## 本轮结果与选择

固定512MiB中文 Wikipedia、none预处理、目标50,000/min2、串行feed、4线程初始化与merge，u32 corpus/posting；全部通过原Trainer接口。模型完整摘要、50,000词表、29,243规则和1,429,915个唯一片段一致。每候选只做关键计时；A的追加同期诊断用于解决实际merge回退，未展开矩阵。

| 版本 | 初始化 s | Merge s | Train s | RSS GiB | 最低可用 GiB | commit |
|---|---:|---:|---:|---:|---:|---|
| [原并行版](results/native-fair-512.count4.jsonl) | 22.975 | 27.457 | 54.760 | 3.390 | 4.424 | `b6a28768` |
| [同期诊断原版](results/optimization-a-diagnostic.baseline.jsonl) | 23.760 | 29.810 | 57.982 | 3.395 | 4.390 | `b6a28768` |
| [A并行构造](results/optimization-a-diagnostic.candidate.jsonl) | 15.421 | 29.756 | 49.737 | 3.417 | 4.337 | `98ca7fc1` |
| [C直接构造](results/optimization-c-512.jsonl) | 9.767 | 26.092 | 39.805 | 3.425 | 4.383 | `fbdc0b2b` |
| [B融合首版](results/optimization-b-512.jsonl) | 13.251 | 31.887 | 49.697 | 3.418 | 4.398 | `c1ee2019` |
| [B2查询优化](results/optimization-b2-512.jsonl) | 14.399 | 21.386 | 40.265 | 3.386 | 4.415 | `a0832c48` |

当前实测训练最快为 **C：39.805秒**，相比初始并行版54.760秒快1.38倍；初始化从22.975降到9.767秒，快2.35倍。C分支作为当前训练入口推荐，保留简单的原merge协议。

**B2是有效的merge热点候选**：相对C merge从26.092降到21.386秒，耗时减少18.04%；相对B首版31.887秒减少32.93%。B2整次训练40.265秒，较C多1.16%，尚未证明全训收益，因此不凭merge局部结果取代当前训练推荐。B2保留作后续merge起点。

所有采样进程VmSwap峰值为0；停止条件仅MemAvailable≤1GiB。系统换页属于主机范围，完整记录保留在jsonl。计时与静态正确性复核并行；本轮B2计时时CPU测试已经结束，没有并发构建/测试/下载。按用户最新要求，后续构建完成后性能与正确性检查同步推进，失败的正确性结果会使该候选性能记录作废。

## 热点变化

| 阶段 s | A同期诊断 | C | B首版 | B2 |
|---|---:|---:|---:|---:|
| alphabet | 2.563 | 0.347 | 0.413 | 0.345 |
| corpus measure | 0.199 | 0.187 | 0.184 | 0.189 |
| corpus allocate | 0.265864 | 0.000046 | 0.000025 | 0.000035 |
| corpus fill | 2.407 | 0.555 | 0.512 | 0.681 |
| tokenize合计 | 5.436 | 1.090 | 1.110 | 1.216 |
| 初始route | 0.460 | 0.573 | 0.664 | 0.867 |
| 初始count | 9.445 | 8.015 | 11.370 | 12.162 |
| Plan/AA fallback | 4.261 | 3.877 | 0.040 | 0.039 |
| delta（含fused prepare） | 15.979 | 13.888 | 23.142 | 12.196 |
| fused prepare（不可再与delta相加） | — | — | 22.858 | 11.889 |
| rewrite | 0.727 | 0.621 | 0.787 | 0.765 |
| owner commit | 8.431 | 7.368 | 7.481 | 7.952 |

A先把词划成连续独占区域并行写最终数组；C进一步移除逐字符UTF-8字符串hash查询和最终数组串行清零。None alphabet改为每worker存在位图，Some沿用原HF频次裁剪。C tokenize1.090秒，热点转移到初始pair计数和merge索引管理。

B首版虽然省掉大部分Plan排序，却把prepare推到22.858秒。按规则访问丢失了全局空间有序访问的局部性，并新增selected邻居查询；32MiB perf诊断中prepare worker约占30% CPU samples，但内联使单个查询成本未能独立量化。没有把全部回退归给某一种操作。

B2仅替换上述查询：每256位置一个pivot下界目录，桶内精确搜索；selected head/tail用ID直接表，重复head/tail通过小hash表回退。prepare降到11.889秒；这两个改动一起测量，不单独量化各自贡献。owner commit仍7.952秒，是剩余merge热点。

## 改动范围与依赖

| 改动 | 直接影响 | 本轮保留的算法 |
|---|---|---|
| A/C构造 | 字母集合收集、字符ID查询、最终数组初始化、词权重元数据 | merge过滤/排序、delta、write、owner commit |
| B融合 | flat非AA的过滤、取消全局Plan排序、邻边delta、写入调度、出生访问顺序 | 初始化构造/计数算法、选择证书、ID分配、AA/多块/非空affix路径 |
| B2查询 | 融合prepare的权重查询与selected邻居查询，merge起点一次目录构建 | B的出生协议、apply屏障、owner提交算法及全部初始化模块 |

B/B2入口实例化AtomicU32槽位；历史对照已验证相同槽位大小及较小的实测时间差异，用户明确取消本轮重复Atomic计时。初始化算法源码沿用C，实际slot实例化与C不同；B2相对B没有再次改变slot类型。B2新目录是在initialization返回后才建立，初始计数不消费它。C/B/B2初始count测到8.015/11.370/12.162秒，数据结构规模与输入一致；这段差异的原因仍未确定，不用merge查询改动解释它。

A最初同样有一次merge变慢，而原版/A同期诊断merge29.810/29.756秒，没有复现6.5秒回退。源码未改不能代替性能证据；记录全部测量及CPU/fault/context-switch/主机负载，不指定未经证实的NUMA、allocator、编译器或后台负载原因。

## 内存与正确性

全部native的初始化核心容量仍相同：slots824,359,780字节、length166,056字节、词权重25,165,824字节、posting1,147,872,496字节，另有owner哈希表/heap。它不是进程RSS，空间口径沿用 [MEMORY_LAYOUT.md](MEMORY_LAYOUT.md)。

C的alphabet临时数组11.57MiB、字符直接表4.25MiB，构造返回后释放；无第二份完整corpus。B2在merge常驻权重目录3,220,160字节（3.07MiB），建立11.784ms；每批selected表峰值800,128字节（0.76MiB），有效起点缓冲峰值17,301,504字节（16.50MiB）。后两项按分配capacity计，不包含allocator开销。RSS实测仍约3.39GiB。

C完整42项库测试通过；B把既有1500-case逐轮HF差分扩到新flat Atomic路径并增加相邻批次/共享头尾/长度/跨worker排序案例，完整43项通过；B2增加目录与原全量pivot查询的逐位置等价测试，完整44项通过。模型签名gate包含10项运行：`d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd`。

独立静态审查分别在 [C审查](CORPUS_DIRECT_REVIEW.md)、[融合及查询审查](FUSED_DIRECT_REVIEW.md)。unsafe数组转换、安全屏障、出生key唯一规则生产者、连续任务顺序、重复head/tail与有限长度协议都有对应推导，审查与计时并行。

## 源码与复现

各候选是独立干净worktree与本地commit，中央分支只保存实验账本。路径与branch在 [WORKTREES.md](WORKTREES.md)，过程及用户要求在 [OPTIMIZATION_PLAN.md](OPTIMIZATION_PLAN.md)、[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)。

在 `benchmarks/hf-bpe`：

```bash
python3 build_native_fair.py /root/code/tokenizers-worktrees/fused-lookup --label fused-lookup
python3 run_native_fair.py --case fused-lookup-reproduction \
  --worktree /root/code/tokenizers-worktrees/fused-lookup \
  --build-root .build/native-fused-lookup --binary target/release/hf-bpe-native-fused-lookup \
  --corpus .build/gb-corpus/zh-512m.txt --output results/fused-lookup-reproduction.jsonl \
  --initialization-workers 4 --merge-workers 4 --atomic-corpus \
  --require-stats fused_prepare_ms --require-stats weight_lookup_bytes
```

每次environment JSON锁定实际worktree源码、临时探针副本、runner/Cargo.lock、二进制、输入和脚本hash；probe只在原do_train返回前输出私有统计。实际调用原train_vocab，未增加公有Trainer选算法API。计时原始数据与阶段解释见 [C结果](results/optimization-c-512.summary.md)、[B结果](results/optimization-b-512.summary.md)、[B2结果](results/optimization-b2-512.summary.md)。旧PR/native公平四项见 [PARALLEL_REPORT.md](PARALLEL_REPORT.md)。

当前以C作为总训练推荐、B2作为merge后续起点。用户追加模块公平性与方差要求后，正在补三组B/B2查询模块交错测量；不追加Atomic对照或宽度/线程矩阵。下一轮若继续，应集中在初始pair计数或owner commit，先按源码确定访问/分配热点再改；当前没有证据选择更具体的实现。
