# BPE Exact M coherent-family 窗口实验

**不采纳 Exact M，生产实现保持原 eager 基线。** 更精确的 family 上界确实放宽了窗口，但所有真实场景的两轮正常训练 CPU 都回退；跨 cohort 少写的坐标未覆盖新 reducer、starts replay 与目录成本。本提交仅增加实验报告和可复核证据。

生产源码基线为 `890ea1f58e65f8d8503f7cd3fc97ad1651eaf284`，实施 checkout 的报告提交为 `54c9461e35b43f9735a780b11498780a40aea7bc`，二者生产源码相同。E 为原 eager 内核；F 为此前 FRONTIER 窗口；M 只在同一个 deferred/replay 窗口中改为 coherent-family 的 Exact M 上界和 `M<F` starts 回收。三者共用原 publisher、冷 direct-ID directory、相同 job 切分和最多四个 cohort 的上限；没有混入 scratch、complete publisher 或 terminal discard。

## 实现与语义

同一个 producer 的所有 job 先完整准备，随后一个 reducer 按顺序扫描全部 accepted starts，在左右两个 direct-ID directory 中用邻居 ID 定位稀疏 `(neighbor, mass)` 聚合项，使用 checked 求和。coherent family 按 producer、side、neighbor 区分。左侧仅归属不会同时被选中改写的旧邻居，右侧预测本 cohort 邻居的替换 ID；self-pair 使用非重叠 starts 的成员关系。严格长度条件仍为 child span `< limit`。

producer 的 `M` 为最大 family mass；达到最大 mass 的左 family 取最小旧 left ID 为 ell，没有此左 family 时 ell=B。窗口 guard 按最大 M / 同 M 最小 ell 累积，在下一候选 identity resolve/take 前比较完整优先级。窗口内 output ID 唯一且不被消费，后续写入只能分裂既有 family 的质量。

当 producer 的 `M < max(min_frequency,1)`，在 apply 与 old-loss commit、读者 join 完成后释放其 accepted starts，保留 output/cohort 描述以供其他 producer 的 cuts 判定。正权重而暂时没有位置的 birth 状态仍保留。最后 replay 的较早 cohort / 同 cohort 物理左侧归属规则与 F 相同。

[候选补丁](candidate.patch.gz) 可以直接应用到生产基线。按 Rust AST 去除注释、空行及 cfg(test) 后，生产净增加 386 行，涉及 merge 与 mod 两个生产模块；index 的变更仅属于测试调用。测试计数为 1,498 行，包含 Miri 入口，未超过 1,500。没有新增 unsafe 或公开 API。详见 [生产计数](source-counts.json) 与 [测试计数](test-count.json)。

## 正常训练

沿用既有中英文 256 MiB 源语料派生的 Whitespace / ByteLevel 准备输入，vocab_size=50000、min_frequency=2，无 affix、长度或 alphabet 上限。机器有 6 个 vCPU，本阶段用 4 workers、affinity 0–3。首轮 Whitespace 顺序 E→F→M，ByteLevel 顺序 M→F→E，第二轮完全反向。正常版仅在 joined phase 边界计时，没有逐坐标计数；编译、测试、重诊断与正常训练错开，模型在计时子进程退出后压缩，再开始下一项。

保存 30 次正常训练（24 次真实训练、6 次 AA/reuse 控制）与 18 次独立重诊断，共 48 个完整模型。全部六种输入的模型在所有版本、两种顺序及正常/诊断间严格逐字节一致。没有剔除正常样本。输入重新核对 SHA-256，完整模型去重归档并保留逐次 manifest。

百分比为 M 相对同轮 E；负数表示更快。两轮原始值、F 对照及全部 phase 见 [normal-runs.csv](normal-runs.csv)、[summary.json](summary.json)、[primary.json](primary.json) 与 [reverse.json](reverse.json)。

| 场景 | 首轮 wall | 反向 wall | 首轮 CPU | 反向 CPU |
|---|---:|---:|---:|---:|
| 中文 Whitespace | +21.27% | +18.69% | +23.90% | +21.03% |
| 中文 ByteLevel | +52.68% | +54.24% | +59.21% | +63.73% |
| 英文 Whitespace | +35.54% | +28.68% | +46.56% | +38.29% |
| 英文 ByteLevel | +34.63% | +21.27% | +48.70% | +35.03% |

M 相对 F 的 CPU 也在八对中全部增加：中文 Whitespace +10.90% / +11.69%，中文 ByteLevel +23.99% / +27.92%，英文 Whitespace +15.64% / +0.18%，英文 ByteLevel +13.33% / +10.34%。英文 Whitespace 反向 wall 单独降低 5.79%，没有转化为 CPU 或对 eager 的净收益。结果已明确回退，按候选收益门槛未扩展 1-worker 矩阵。

所有正常子进程 VmSwap 为 0，HWM 增量均低于 `max(5% baseline,16 MiB)` 的预算。进程 sampled RSS 是 50 ms 采样，可漏掉短峰；训练 HWM 是 runner 在模型验证前读取的进程历史峰值。系统级 swap 页变化单独保存在逐次记录，不等同本子进程换出。不能将满足内存预算解释为性能可接受。

## 物理收益与新增成本

以下来自重诊断；它带逐坐标统计、目录及 requested-allocation 计数，用于解释工作量，不能混入正常训练耗时。

| 场景 | E append | F append | M append | M 相对 E 净减少 | F → M outer windows | M pruned starts |
|---|---:|---:|---:|---:|---:|---:|
| 中文 Whitespace | 49,133,128 | 49,130,437 | 49,014,738 | 118,390 (0.2410%) | 1166 → 360 | 309 |
| 中文 ByteLevel | 280,205,796 | 280,205,119 | 277,247,800 | 2,957,996 (1.0557%) | 1398 → 436 | 1,885 |
| 英文 Whitespace | 5,336,666 | 5,336,626 | 5,277,644 | 59,022 (1.1060%) | 1323 → 426 | 21,432 |
| 英文 ByteLevel | 6,957,536 | 6,957,470 | 6,902,216 | 55,320 (0.7951%) | 1196 → 357 | 23,733 |

M 放宽 guard 后减少了 outer windows；inner cohorts、cuts 和 append 的净差是各自实际统计，不能把 cuts 全部算成净少写。min_frequency=2 下罕见 producer 的 starts 回收量仍很小。

| 场景 | reducer 扫描 starts | family tallies / K | reducer CPU 合计 | 最慢单 producer CPU | 新 coherent 目录逻辑填充 GB |
|---|---:|---:|---:|---:|---:|
| 中文 Whitespace | 29,382,012 | 49,014,980 / 13,045,123 | 11.954 s | 43.307 ms | 7.508 |
| 中文 ByteLevel | 156,321,947 | 277,249,118 / 17,231,495 | 61.605 s | 817.578 ms | 7.953 |
| 英文 Whitespace | 3,565,676 | 5,284,752 / 910,506 | 2.229 s | 53.950 ms | 7.350 |
| 英文 ByteLevel | 4,790,867 | 6,907,403 / 990,143 | 2.830 s | 48.781 ms | 7.010 |

这些 reducer CPU 数值包含重诊断开销，不能直接从正常版总 CPU 中相减。coherent 目录的 GB 是 sentinel 元素数×4 的逻辑写入量，并非硬件带宽计数。完整记录另列 prepare/replay/coherent 三处目录的实际线程数、容量峰值、初始化分配和旧元素搬移。额外扫描还增加了 corpus token reads 和 weight-region 查找；各项在 summary 的 physical 表中完整保存。

重诊断 allocator 从进程启动累计 live requested bytes，在训练入口重置峰值但保留当时已有的输入驻留量；它不包含 allocator 元数据与 RSS。输出包含 allocations/reallocations/frees、requested/freed bytes 与训练期 requested-byte 峰值。窗口 retained checkpoint 和 attempt high-water 是已定义检查点的 starts/cohort 容量估计，不能冒充所有线程并发 RSS 峰值。

## 验证、失败记录与复现

默认 features 和 no-default-features 的 BPE 原生检查均为 31 passed；Clippy all-targets（`-D warnings`）通过。测试包含 AA、自重叠、alias/reuse、正权重空坐标、低于 floor 的 producer、多线程池，以及每轮有序 pair/count/ID trace 与原生 corpus 的顺序 token-list oracle。正常与诊断完整训练覆盖 AA/reuse 控制及四个真实输入。

现有 `feed_training_tokenizer_json_and_encoding_roundtrip` 在基线就有中文“测”的未知 token 问题，本轮原生检查按既有边界跳过它；第一次使用错误的 skip 名称导致它运行失败，日志保留。no-default-features 首次链接因临时盘容量不足失败，释放本任务已完成的旧测试可执行文件后重试通过。重诊断第一次生成源码因 json! 中索引表达式缺括号失败，修正生成器后六个最终 frozen arms 正常构建；失败源码在 failed，所有失败/重试日志均在 validation，未作性能样本。

复现单次训练，使用干净 disposable checkout（HEAD 为生产基线），将 --input 指向 environment.json 中对应已准备输入：

```bash
python3 replay.py --checkout /path/to/clean-checkout --arm exact   --job primary/zh-whitespace-w4-r0-exact/job.json   --input /path/to/prepared-words.json --output /path/to/output   --target-dir /path/to/dedicated-target
python3 summarize.py
```

arm 可取 eager/frontier/exact 及三个 `-diagnostic` frozen 版本；正常结果由 source 目录完整恢复，不依赖本机被回收的二进制。replay 检查输入哈希、基线和干净 checkout，使用锁文件、offline release 构建与记录 affinity。重建原始正常/诊断 suites 可参考 measure.py；builders 保存插桩来源与构建步骤，本机路径需按新环境映射。native checks 的命令边界如下：

```bash
cargo test --locked --offline --manifest-path tokenizers/tk-train/Cargo.toml   trainers::bpe -- --skip feed_training_tokenizer_json_and_encoding_roundtrip
cargo test --locked --offline --no-default-features   --manifest-path tokenizers/tk-train/Cargo.toml trainers::bpe   -- --skip feed_training_tokenizer_json_and_encoding_roundtrip
cargo clippy --locked --offline --all-targets   --manifest-path tokenizers/tk-train/Cargo.toml -- -D warnings
```

下一轮按用户提供的 [Positions Buffer / SIMD 方案](references/next-positions-plan.md) 准备 A/B/C/D 独立 worktree，固定本报告提交作为新 baseline，并先确认生产代码仍相同。该轮结果独立报告，本次 M 结论不代替编码实验。
