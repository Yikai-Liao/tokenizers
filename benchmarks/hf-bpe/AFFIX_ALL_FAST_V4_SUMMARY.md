# Affix all-fast v4 统一消融结果

## 构建与范围

- 候选冻结提交：`438fbca3fb49f7604eeff76e43ebbe7ed7c8604e`；构建前 worktree clean。
- 孤立构建：`.build/native-affix-all-fast-v4/`；二进制 SHA256：`38308624a5a621dd3b921d6193f3fe08fbec6ffd257a36f7368d84c829301d8f`。可审查 [build manifest](.build/native-affix-all-fast-v4/build_manifest.json) 和 [结果目录](results/affix-all-fast-v4/)。
- 构建 harness 从冻结源码动态读出 `Options` 的 16 个 bool 字段；单项移除和 `all` 均由实际字段集生成。旧 10 字段与新 16 字段的临时副本开关生成检查通过。
- 主矩阵共 38 次串行运行：EN16M prefix `##` 和 ZH16M suffix `</w>` 各 18 次（FULL、16 个单项移除、ALL_OFF），加 EN/ZH 无 affix 各 1 个同源对照。训练参数均为 workers 请求 4、vocab 30000、min_frequency 2；每次都以 v2 的 exact corpus path/SHA、训练参数和模型 SHA 配对。
- 另按最终预算完成 3 次 ZH512M：FULL、minus `weight_lookup`、NONE，vocab 50000/min_frequency 2/workers 请求 4。三次使用同一 v4 二进制，逐次配对 v2 对照。除此之外没有运行额外性能案例。

## 16M 主要结果

训练时间为 native 训练段 `train_ms`，不含输入 feed；RSS 为进程 VmHWM。每个模型 SHA 都等于配对的 v2 完整模型 SHA。

| 案例 | 训练 ms | 相对同源 NONE | RSS MiB | 实际 workers/init | layout | alias fallback | corpus 槽宽 |
|---|---:|---:|---:|---:|---|---|---:|
| `en16m-prefix-full` | 2215.6 | +6.1% | 236 | 4/4 | `parallel_u16_flat32` | false | 2 B |
| `en16m-none4` | 2087.7 | +0.0% | 275 | 4/4 | `parallel_u32_flat32` | false | 4 B |
| `zh16m-suffix-full` | 1070.5 | +4.1% | 162 | 4/4 | `parallel_u16_flat32` | false | 2 B |
| `zh16m-none4` | 1028.2 | +0.0% | 144 | 4/4 | `parallel_u32_flat32` | false | 4 B |

FULL 的同源 NONE 开销为 EN +6.1%、ZH +4.1%。相对旧 v2 affix 路径，v4 FULL 约快 EN 2.02×、ZH 2.23×；旧 v2 NONE 与 v4 NONE 接近（EN v4 快 2.5%，ZH v4 慢 1.1%）。所以先前多倍 affix 差距不能只归因于不能剪枝。

FULL affix 在两组输入上都启用了 `alias_guarded=true`，没有触发重建回退（`alias_fallback=false`）；`reused_ids=0`，没有激活 cohort 全词扫描。FULL 记录 `monotone_pairs=true`，对应受别名碰撞守卫保护的 epoch。`pruned_pairs` 为 EN 2,136,774、ZH 2,435,319。

posting 访问次数几乎相同：EN FULL 24.268M 对 NONE 24.346M；ZH FULL 3.660254M 对 NONE 3.660298M。也就是说当前大头差距已不来自额外扫描这批 posting。FULL 的 `corpus_slot_bytes=2`，NONE 为 4；报告统一用这个明确的逐槽字段，旧 `initial_slot_bytes` 的历史值不混作逐槽宽度。

### 单项移除

百分比以对应语言的 FULL 为基准。`parallel_apply` 移除会显式将实际 merge worker 降为 1（init 请求仍为 4），因此这两行不是 N4 merge 并行配置下的公平选型比较。其余单项移除实际 workers 均为 4；init 统计见列。小幅负百分比是单次样本，不能据此判断应移除该项。

| 选项移除 | EN ms (Δ%) | EN RSS MiB | EN workers/init | ZH ms (Δ%) | ZH RSS MiB | ZH workers/init |
|---|---:|---:|---:|---:|---:|---:|
| `sort_weights` | 2120.0 (-4.3%) | 244 | 4/4 | 988.8 (-7.6%) | 161 | 4/4 |
| `narrow_corpus` | 2230.6 (+0.7%) | 280 | 4/4 | 1007.9 (-5.9%) | 173 | 4/4 |
| `weight_lookup` | 2097.6 (-5.3%) | 246 | 4/4 | 989.1 (-7.6%) | 160 | 4/4 |
| `initial_grouped` | 2039.7 (-7.9%) | 276 | 4/4 | 1153.9 (+7.8%) | 167 | 4/4 |
| `parallel_apply` | 4101.6 (+85.1%) | 260 | 1/4 | 1680.9 (+57.0%) | 137 | 1/4 |
| `grouped_tail` | 2111.1 (-4.7%) | 251 | 4/4 | 1066.5 (-0.4%) | 163 | 4/4 |
| `scratch_cache` | 2281.1 (+3.0%) | 245 | 4/4 | 1202.9 (+12.4%) | 158 | 4/4 |
| `character_cache` | 3547.7 (+60.1%) | 244 | 4/4 | 1292.2 (+20.7%) | 163 | 4/4 |
| `packed_queue` | 2114.1 (-4.6%) | 263 | 4/4 | 1102.4 (+3.0%) | 173 | 4/4 |
| `parallel_measure` | 2135.2 (-3.6%) | 241 | 4/4 | 1050.1 (-1.9%) | 159 | 4/4 |
| `guarded_fast` | 3697.0 (+66.9%) | 372 | 4/4 | 1945.7 (+81.8%) | 302 | 4/4 |
| `parallel_alphabet` | 2255.6 (+1.8%) | 246 | 4/4 | 1189.7 (+11.1%) | 162 | 4/4 |
| `arena_allocator` | 2287.9 (+3.3%) | 220 | 4/4 | 1108.4 (+3.5%) | 156 | 4/4 |
| `batch_execution` | 3824.9 (+72.6%) | 234 | 4/4 | 2479.4 (+131.6%) | 158 | 4/4 |
| `queue_prefetch` | 2261.8 (+2.1%) | 244 | 4/4 | 991.5 (-7.4%) | 163 | 4/4 |
| `fused_batch` | 2782.2 (+25.6%) | 239 | 4/4 | 1344.4 (+25.6%) | 159 | 4/4 |
| `ALL_OFF` | 8093.6 (+265.3%) | 369 | 4/1 | 3255.4 (+204.1%) | 258 | 4/1 |

高影响项是 `guarded_fast`（EN +66.9%、ZH +81.8%；切回 cohort 路径）、`batch_execution`（+72.6%、+131.6%；`max_batch_rules` 降到 1）、`fused_batch`（两语料均 +25.6%）和 `character_cache`（+60.1%、+20.7%）。ALL_OFF 为 EN +265.3%、ZH +204.1%。`weight_lookup` 单项在两组 16M 样本都测得较快。构建时间接近零、`weight_lookup_bytes=0` 是因为这些输入走 `WeightLookup::from_parts` 的单区间表示：`interval_one` 内联保存范围，heap-backed 的 `bounds` 和 `one_buckets` 保持为空；该计数字段不包含内联字段本身。这些计数说明没有分配堆目录，并不表示 lookup 未启用。

## ZH512M 三项预算

计划顺序是同一 438 冻结二进制：`ZH512 suffix FULL → ZH512 suffix minus weight_lookup → ZH512 NONE`。语料 SHA 为 `a0d40d4102cba1e933f25e5ccd17552d2eaebaec4ef770bc39d8e46e740b4d4f`；参数 vocab 50000/min_frequency 2/workers 请求 4。下表训练时间不含 feed，完整 SHA 与 v2 对照一致。

| 案例 | train ms | feed ms | wall s | RSS GiB | min MemAvailable GiB | OS 线程峰值 | workers/init | init ms | merge ms | lookup build ms / bytes | model SHA 前 12 位 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `zh512m-suffix-full` | 24730.9 | 3834.1 | 29.911 | 2.97 | 4.48 | 5 | 4/4 | 9344.7 | 14892.8 | 0.0001 / 0 | `9e7dd5f1d28b` |
| `zh512m-suffix-minus-weight_lookup` | 22465.8 | 3901.0 | 27.884 | 2.98 | 4.50 | 5 | 4/4 | 9341.6 | 12746.9 | 0.0002 / 0 | `9e7dd5f1d28b` |
| `zh512m-none4` | 17843.0 | 3778.1 | 22.859 | 3.34 | 4.15 | 5 | 4/4 | 4582.9 | 12870.2 | 0.0001 / 0 | `d50fb836e342` |

FULL 比同源 NONE 慢 38.7%（24.731 s 对 17.843 s）；minus `weight_lookup` 比 FULL 快 9.2%（22.466 s）。源码中 `WeightLookup::from_parts` 对单个权重为 1 的连续区间设置内联 `interval_one`，`weight_parts` 会检查该范围并直接返回权重 1；因此 lookup 在 FULL 热路径上实际启用。它参与初始 radix 计数和 fused batch 准备；关闭时前者使用 `Block::weight`，后者使用有序游标 `Block::weight_forward`。FULL 与 minus 的 `weight_lookup_build_ms` 都约 0.0001 ms、`weight_lookup_bytes=0`，只反映这个输入无需堆目录分配，不能用来否定 lookup 的运行时工作。速度差落在 merge 计时（14.893 s 对 12.747 s），其中 `fused_prepare_ms`（8.873 s 对 7.635 s，已包含在 merge 内）和 `commit_ms`（4.490 s 对 3.826 s）也有变化。两种实现实际计算权重的路径不同，但每种只有单次测量；残余计时波动及不同阶段的成本都可能影响差异，现有数据不能精确量化 lookup 自身的因果贡献。

结合 EN/ZH 16M 与 ZH512M 的单项对照，关闭 `weight_lookup` 的配置在这三组测试中都比 FULL 快（分别快 5.3%、7.6%、9.2%），且 512M 模型 SHA 与 FULL 一致。基于这组一致的方向，后续默认值选择为关闭 `weight_lookup`，同时保留该机制供其他布局或配置使用。这里的单次样本支持默认选择，但不构成稳定性能收益的精确估计。

三次均为实际 workers/init=4/4、OS 线程峰值 5、VmSwap=0；最低 MemAvailable 4.15 GiB 以上。按 VmHWM 统计，FULL RSS 2.97 GiB，minus lookup 2.98 GiB，NONE 3.34 GiB。主 allocation session 的 arena requested/retired 分别约 526.8 MB/526.8 MB，heap requested/freed 约 877.0 MB/877.0 MB；minus 计数相同。推测 session 三项都为零，且 requested/released 一致。

阶段计时存在嵌套：`fused_prepare_ms` 包含在 `merge_ms` 中，不可相加；`commit_ms` 是全局 delta 与 birth assembly。`initial_count_ms` 包含在 initialization 阶段。三次结果以完整训练计时、原始阶段计时和 allocation counters 一并留存在相应 `.jsonl` / `.environment.json`；只有一轮大集测量，选项级因果结论需以实际计数为据。

## 验收

- 38/38 主矩阵结果文件存在，所有候选模型 SHA 与各自 v2 配对行完全相同。
- 3/3 ZH512 结果均匹配各自 v2 完整模型 SHA。
- 主矩阵和 512 三项均通过 MemAvailable > 1 GiB、VmSwap=0、OS 线程观测、逐槽宽度归一化，以及主/推测两个 allocation session 的 arena requested=retired、heap requested=freed、buffer count=frees 检查。
- 38 个主矩阵条目及 3 个 512 条目各有独立 JSONL、环境元数据、stdout、stderr；build manifest 保留二进制与 isolated instrumentation provenance。

## 可复现基线

EN16 baseline：`benchmarks/hf-bpe/results/affix-parallel-v2/en16m-prefix-t4.jsonl` 和 `en16m-none-t4.jsonl`；ZH16：`zh16m-suffix-t4.jsonl` 和 `zh16m-none-t4.jsonl`。ZH512：`zh512m-suffix-t4-vocab50k.jsonl` 和 `zh512m-none-t4-vocab50k.jsonl`。所有候选环境文件记录准确 corpus path/SHA、vocab/min_frequency、请求 threads、binary SHA、baseline pair 和资源采样。
