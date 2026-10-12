# Fresh 右侧 token 预取：不采用

基线为 `5c7ef814f55da46cd7c820173cc72053fe092d74`，包含已采用的 Positions 读取简化与 Affix/InitialTokenIds 优化。本轮直接比较现有 head-only 窗口16与 head + right 窗口16。**候选未达到纯性能改动的采用标准，正式分支保留 head-only，生产源码没有变化。**

多线程中文 ByteLevel 的训练 wall/CPU 两次略快，但 prepare 的改善仅约 1.23% / 0.18%，单线程该阶段反而慢约 1.25%。英文 Whitespace 两次多线程 CPU 均变慢（约 10.75% / 3.45%），prepare 也均变慢（约 19.59% / 4.60%）；其 wall 分别慢约 9.84% / 0.39%。其他场景没有稳定净收益。保留全部样本，没有删除较慢的一次。未触发距离、无提示、邻居或批次框架的后续实验，也未加入运行时开关。

## 候选差异与正确性

[`candidate.patch.gz`](candidate.patch.gz) 是基线上的完整最小候选差异，含必要短测试。固定左跨度由已有 `fresh_matcher` 每个准备任务计算一次，并作为返回值供窗口使用；没有逐位置查询跨度。初始填充与每次补入都调用同一个私有闭包：保留 head 提示，`checked_add(left_span)` 成功时通过原 `Corpus::prefetch` 提示 right。该封装继续按当前 AtomicU16/AtomicU32 平面做 `get` 边界检查，经已有 branches 发出读提示。加法溢出与逻辑边界外的地址跳过，不构造非法指针。

VecDeque、16位置窗口、候选顺序、resident 单位、AA 非重叠筛选、邻居事件、原子读取及 prepare/write/commit 与 join 均保持原语义。Source 窗口仅用于 fresh；reuse 的位置相关跨度仍由独立 cohort 遍历处理。预取不提供读取结果或同步保证。没有新增依赖、unsafe、汇编、SIMD、缓存行去重、Positions 或 owner/Write 改造。

28 次完整训练（16 次多线程、8 次单线程、4 次 AA/reuse 控制）的完整模型文件均按字节匹配：vocab-by-ID、有序 merges 及其余输出内容一致。各 worker 数之间同一输入的模型也一致；[`model-manifest.json`](model-manifest.json) 指向经解压 SHA-256 验证的完整 gzip 内容，不只保存模型摘要。

## 参数与测量边界

沿用已有 256 MiB 中文/英文输入的 Whitespace 与 ByteLevel 准备结果。所有真实场景使用相同配置：vocab_size=50000、min_frequency=2、prefix/suffix=None、max_token_length/limit_alphabet=None。4 workers 为主、1 worker 为控制；均固定 CPU 0–3。每个训练进程串行启动；Whitespace 第一轮基线先、ByteLevel 候选先，反向复测交换顺序。测量期间没有编译、测试或 perf 任务。

release opt-level=3、fat LTO、codegen-units=1，locked/offline、CARGO_INCREMENTAL=0；相同 runner 与计时插桩用于两个版本。[`environment.json`](environment.json)、两个 build.json、[`cpu.txt`](cpu.txt) 保存基础 SHA、输入哈希、工具链、CPU 和二进制哈希。`source/` 是实际编译的冻结源码。

训练 wall/CPU 来自 runner 训练边界；加载另记 `load_seconds`；进程 external wall/CPU 还包含加载、模型验证和序列化，这些不计入训练边界。prepare 指标是每次 `batch.prepare` 的 wall 总和，包括准备快照及 Rayon join，**不是各 worker CPU 的累加**。两个版本仅在每个批次边界额外做一次 Instant 计时和常数数量的 Relaxed 计数更新，结束后打印；没有逐候选计时或日志。`phases` 同时保留 selection、vocabulary、plan、index、materialize、merge 与 drop。该插桩只存在于冻结测量源码。

## 多线程结果

下表取正反两对的算术均值；原始每次记录见 [`primary.json`](primary.json)、[`reverse.json`](reverse.json)。百分比为候选相对基线，负数表示更快。

| 场景 | 训练 wall 秒：基线 → 候选 | 训练 CPU 秒：基线 → 候选 | prepare wall 秒：基线 → 候选 | prepare / 训练：基线 → 候选 |
|---|---:|---:|---:|---:|
| 中文 Whitespace | 20.742 → 21.172 (+2.07%) | 59.432 → 60.304 (+1.47%) | 2.790 → 2.806 (+0.57%) | 13.5% → 13.3% |
| 中文 ByteLevel | 22.841 → 22.464 (-1.65%) | 73.243 → 71.768 (-2.01%) | 9.219 → 9.154 (-0.71%) | 40.4% → 40.8% |
| 英文 Whitespace | 1.591 → 1.675 (+5.23%) | 3.747 → 4.018 (+7.21%) | 0.476 → 0.534 (+12.21%) | 29.9% → 31.9% |
| 英文 ByteLevel | 1.871 → 1.853 (-1.01%) | 4.400 → 4.450 (+1.15%) | 0.596 → 0.609 (+2.22%) | 31.9% → 32.9% |

## 单线程控制

每个场景一对；原始记录见 [`single.json`](single.json)。这组用于控制，不能将单次小幅变化解释为稳定收益。

| 场景 | 训练 wall 秒：基线 → 候选 | 训练 CPU 秒：基线 → 候选 | prepare wall 秒：基线 → 候选 | prepare / 训练：基线 → 候选 |
|---|---:|---:|---:|---:|
| 中文 Whitespace | 60.006 → 61.315 (+2.18%) | 59.880 → 61.118 (+2.07%) | 9.771 → 9.665 (-1.08%) | 16.3% → 15.8% |
| 中文 ByteLevel | 72.086 → 71.649 (-0.61%) | 71.969 → 71.520 (-0.62%) | 34.134 → 34.560 (+1.25%) | 47.4% → 48.2% |
| 英文 Whitespace | 2.335 → 2.351 (+0.67%) | 2.326 → 2.349 (+0.98%) | 0.876 → 0.886 (+1.19%) | 37.5% → 37.7% |
| 英文 ByteLevel | 2.891 → 2.828 (-2.18%) | 2.888 → 2.826 (-2.13%) | 1.143 → 1.135 (-0.78%) | 39.5% → 40.1% |

## 内存、swap 与平面范围

峰值取 runner 的训练完成、模型验证前 process HWM；多线程取两次的最大值。

| 场景 | 4 workers 峰值 MiB：基线 → 候选 | 1 worker 峰值 MiB：基线 → 候选 |
|---|---:|---:|
| 中文 Whitespace | 1739.2 → 1735.1 | 1745.9 → 1746.1 |
| 中文 ByteLevel | 2537.8 → 2535.2 | 2518.2 → 2519.3 |
| 英文 Whitespace | 133.1 → 132.2 | 120.6 → 120.6 |
| 英文 ByteLevel | 159.0 → 158.6 | 137.7 → 138.1 |

所有进程采样到的 VmSwap 峰值为0（采样间隔约50ms）。主机全局 pswpin 在少数中文运行发生小量变化；单线程中文 ByteLevel 候选时还记录到78页 pswpout。全局计数包括其他进程，不能据此归因给候选；完整原始计数保存在每次记录中。没有将“子进程未观测到 swap”写成“整台主机没有 swap”。

所有真实场景实际平面宽度均为16位；fresh 批次数分别为中文 Whitespace 1390、中文 ByteLevel 1646、英文 Whitespace 1491、英文 ByteLevel 1335，reuse批次均为0，两个版本相同。AA 控制记录9个 fresh 批次；reuse 控制经历一次 fresh 尝试与5个 reuse 批次，两次 materialization 都为16位。控制耗时很短，只用于完整模型与路径验证，不作为性能结论。

32位宽平面由新增短测试以 vocab_size=70000 和已有 wide_ids 测试验证功能。**没有宽存储真实训练的性能证据**。本轮不申请更大机器，不声称宽平面收益。

## 验证与代码预算

候选默认与无默认特性各31项 BPE 测试通过，Clippy通过（只放行既有 `redundant_field_names`），两个变更文件 rustfmt/check 与 git diff/check 通过。原有 JSON roundtrip 的已知“测”问题仍按此前基线跳过；没有将其计为通过。日志见 `validation/*.log.gz`。

新增测试覆盖空输入、短于窗口、超过窗口后的补入顺序/数量、Slice/ReadTask、逻辑末尾、偏移溢出、窄/宽平面。没有测试“提示一定进入缓存”。未实施无提示控制，因此没有关于关闭提示的性能或遍历实测声明。

AST 排除注释与 cfg(test) 后，候选 production 非空行净增14（corpus +4、merge +10）；测试预算1428→1462（+34），仍低于1500。放弃候选后，正式生产代码与测试预算均保持基线1428。候选计数记录为 [`source-counts.json`](source-counts.json)、[`test-counts.json`](test-counts.json)，最终预算复核见 [`test-counts-final.json`](test-counts-final.json)，没有把候选称为代码简化。

## 复核

`all-normal.json` 与各运行目录保存参数、stdout、stderr；`models/` 提供全部完整模型。`measure.py` / `build-normal.py` 保留原机器路径，供方法复核。可移植单任务回放用 `replay.py`：使用干净、一次性的基线 detached checkout，覆盖相应冻结源码，验证准备输入哈希后构建与运行。

```sh
python replay.py --checkout /path/to/disposable-base-checkout \
  --arm head-right --job primary/zh-whitespace-w4-r0-head-right/job.json \
  --input /path/to/recorded-zh-words.json \
  --target-dir /path/to/target --output /path/to/run
```

对 baseline 另用一个干净 checkout 同样运行，并比较两份完整 model.json。AA/reuse 的小输入也已归档在 `control-inputs/`，可以使用相应 controls/job.json 回放。依赖须已经缓存才能 `--offline` 构建；可先按项目环境准备依赖。不同编译器/硬件的二进制哈希与耗时不应与本机数据混用。源码补丁的预算复核在应用候选补丁的基线 checkout 中执行：

```sh
gzip -dc candidate.patch.gz | git apply
uv run count-tests.py --base 5c7ef814 --output /tmp/prefetch-test-counts.json
uv run count-source.py --base 5c7ef814
```

本次结论只覆盖上述输入、编译器和CPU：右侧地址提示没有显示足以抵消新增工作与维护成本的稳定训练收益，故放弃。现有 head-only 机制本身的收益没有在本轮重新验证。
