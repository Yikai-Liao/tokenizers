# 词频查询实验：保留 A，停止 C 验收

实际基线是开始时的最新提交 `6bbafd1db650842453162b7dcf789f07c7262196`，而不是原计划的 e4c35bee。C 独立提交为 `887aadb16b13f1ccca154bb7ac2d09568ecc2bec`；仅增加 fresh reader 的 run 索引及后缀二分，改动见 [C.patch](C.patch)。C 已从主分支生产代码恢复，不合入。B、D 没有实现。

## 停止证据

使用同一 A release 二进制、同一输入和参数、1 worker，真实英文 ByteLevel 配置加 `end_of_word_suffix="a"` 触发 reuse。三次 A 完整模型哈希各不相同，raw training merges 分别为 29766、29763、29765；原生加载后 merges 均为 29641，但有序数组也不相同。vocab、ID、其余配置和 special tokens 相同。首次 A/C 对比也出现差异，但 C 的完整模型与第三次 A 完全相同。因此，当前记录不能把差异归因于 C，也不能让 C 越过精确一致性门槛。

[model-comparison.json](model-comparison.json) 给出全部差异字段、长度及首个不同索引；[runs/full-smoke](runs/full-smoke) 与 [runs/baseline-repeat](runs/baseline-repeat) 保存每次原始模型、job、stdout、stderr、time 和结果。没有排序 merges、修改 ID、忽略字段或删除失败样本。早期 fail-fast 控制器没有把最后一次失败比较写进 `runs.json`，但该次 `result.json` 和模型已保存；[samples.json](samples.json) 遍历实际 attempt 目录，包含全部 66 次进程。最终 run.py 已修正为先保存记录及 failure.json 再抛错。

这是基线自重复不一致造成的验收阻塞，不是 C 已被证明错误。尚未定位根因；停止实验并保留 A。没有执行约三对性能筛选、160 次完整矩阵或稳定收益统计。

## 诊断机会及限制

诊断构建计数器属于 task/reader，正常 A/C release 没有计数器。以下数据来自独立诊断运行；[diagnostics-normalized.json](diagnostics-normalized.json) 为汇总，逐 reader 及 stderr 原始记录在 runs 中。

|配置|真实 reader|目录 runs|有效匹配|缓存命中|真正二分|全目录比较|后缀查询/比较|
|---|---:|---:|---:|---:|---:|---:|---:|
|zh256 Whitespace|32246|1172|29382012|29140629 (99.178%)|209181|2510172|177048 / 819088|
|en 8 MiB ByteLevel|29744|681|552371|444509 (80.473%)|93742|1031162|64073 / 346532|

跨 run 跳跃 1、2–4、5–16、更多分别为中文 131089/38355/23051/16642，英文 35674/18683/14118/9643。首次查询平均索引为中文 1105.7/1172、英文 635.5/681，不能假定 task 从首 run 开始。诊断 unit 路径为了拿到索引而额外二分，比较数为中文 386424、英文 155320，已单列，不能当作原实现开销。后缀比较也属于诊断额外工作。

早期 diagnostic 构建用带 Drop 的结构体更新语法产生 `runs=0` 临时行，导致原始汇总 reader 数重复；查询计数不受影响。规范汇总明确排除这些零目录行，原始日志及旧 result 保持原样，prepare.py 已修正初始化写法。未用诊断构建的时间判断收益。

真实英文 reuse 诊断：历史扫描 S=226427、Z=13796、单匹配=210932、多匹配=1699；整词扫描 S=692816、Z=367212、单匹配=325203、多匹配=401。总计 919243 扫描词、381008 零匹配词（41.45%），B 有潜在减少查询次数的机会，但零匹配词各自的查询快路径成本尚未单列，且同一真实配置的基线精确一致性未通过。B 停在诊断阶段。

独立 perf 采样使用原始 tuple 输出的正常 A release，不带诊断计数器：zh256 的 AB closure 查询 miss 指令范围占全进程采样周期约 0.0338%，英文约 0.9948%；binary/return 样本仅 21/7 个。输入加载包括在分母中，AA 未单独量化；这些值不能当作整体训练收益上限。原始 IP/period、flat report、反汇编及范围见 [runs/profile](runs/profile)。perf annotate 本机崩溃，使用 perf script 和 objdump 分析；未声称完成源码 annotate。

## 状态与准备内存

|实际类型（字节）|A|C|
|---|---:|---:|
|WeightLookup / Option|16 / 24|24 / 32|
|AA、AB inferred reader|120|128|
|Corpus / CorpusPlan|144 / 176|152 / 184|
|PreparedChunk|56|56|
|借用 decoder，AA / AB|16 / 96|16 / 96|

C 的 unit 区间额外保存一个 usize 索引，没有新增逐词或逐坐标数组，decoder 保持借用。[layout-summary.json](layout-summary.json) 汇总独立布局构建的全部逐准备阶段输出，包括 chunks 容量、spilled starts、events、Positions 的 leaf heap。该口径是已完成 prepare 输出的已知堆容量；不包含全部同时存活的输入/语料/索引、排序 scratch、Rayon task 栈或 allocator 元数据，不能声称是完整准备内存上界。没有测得完整准备内存总量，属于未验证项。

正常 FULL release 的 AB closure 栈 reservation 都为 648 字节、相同六个寄存器 push；函数大小 5753→5673 字节。整个二进制 text 1420917→1420841，data +24，bss −24。AA 被内联，未单独确认其栈帧；寄存器 spill/copy 成本未完整量化。见 [provenance/codegen.json](provenance/codegen.json) 和对应 AB.asm。状态增长不能据此当作无成本。

## 正确性、行数和测量边界

C 默认/无默认特性下原有 35 项 BPE 测试均通过；该实验基线的 tokenizer_roundtrip 仍按前一提交被注释，不计作通过。C 补充表驱动边界覆盖，使测试口径 1694→1719（+25），生产计数为 2570。clippy no-default/all-targets 通过，保留两处基线已有 redundant-field 警告。相关日志和计数在 provenance。并行测试精简是独立任务，其恢复的 roundtrip 失败及最终行数另行记录，不能把其结果混作此 C 构建结果。

五个小 fixture（AA、AB、reserved、alias、length-reuse）在 A/C × 1/6 workers 的 20 次 FULL smoke 中完全一致。另有 A/diagnostic smoke、layout smoke、独立诊断及 profile。完整输出保留原始 training vocab/有序 merges，并通过 PipelineBPE 加载后导出 BpeConfig 全部 11 字段及六项 AddedToken 属性；仅规范对象键顺序，数组与 ID 严格比较。正常核心时间是 public do_train，构造模型/导出均在其后；训练 HWM 在 validation 前读取。/usr/bin/time 的 wall、CPU、RSS 包含加载和后置验证，因此与 train_seconds/train_cpu_seconds 不等价。

[samples.json](samples.json) 列出全部 66 次各配置 wall、user/system CPU、RSS 和阶段指标；诊断/布局/profile 样本不可作为性能验收，FULL smoke 仅为正确性检查。未进行跨版本分布或提速结论。

输入设置固定在 [inputs.json](inputs.json)。中文 256 MiB 使用真实 Whitespace；旧英文 256 MiB 输入是 WhitespaceSplit，不能充当计划要求的 Whitespace，且此次没有运行该英文配置。英文 ByteLevel 使用冻结的 8 MiB 语料生成的 GPT-2 regex/字节映射加权词表，未加 prefix space，initial alphabet 为 256 个字节字符；不能充当完整 256 MiB ByteLevel 矩阵。复现实验应先核对输入哈希；如果以后恢复矩阵，必须另备准确的 256 MiB 输入设置。

## 复现和文件

基线、候选、实际冻结源码哈希、工具链、CPU、构建命令、环境和二进制哈希见 [provenance/environment.json](provenance/environment.json)。runner release 使用 opt-level=3、fat LTO、codegen-units=1，无显式 RUSTFLAGS。每个 job 明确设置 TOKENIZERS_PARALLELISM=true、RAYON_NUM_THREADS=1/6，清除 TK_/TOKENIZERS_/RAYON_ 继承变量，并绑定 CPU 0 或 0–5；独立采样不与其他基准并行。

下面以已冻结源码为例重建 FULL 版本；若从 Git checkout 重建，先按 SHA checkout A/C，运行 prepare.py 复制源码，再各运行 full_model.py。不要对已经应用过的 snapshot 再应用补丁。

```sh
cargo build --locked --release --manifest-path /dev/shm/bpe-frequency-20261014/A/experiments/bpe-simplification/runner/Cargo.toml --target-dir /tmp/bpe-frequency-rebuild-A
cargo build --locked --release --manifest-path /dev/shm/bpe-frequency-20261014/C/experiments/bpe-simplification/runner/Cargo.toml --target-dir /tmp/bpe-frequency-rebuild-C
python3 run.py --definitions inputs.json --arms A=/tmp/bpe-frequency-rebuild-A/release/bpe-bench-runner --cases en-bytelevel-reuse --workers 1 --pairs 3 --out /tmp/bpe-frequency-repeat-new
python3 run.py --definitions inputs.json --arms A=/tmp/bpe-frequency-rebuild-A/release/bpe-bench-runner C=/tmp/bpe-frequency-rebuild-C/release/bpe-bench-runner --cases aa ab reserved alias length-reuse --workers 1 6 --out /tmp/bpe-frequency-smoke-new
```

复跑路径由执行者替换，输入内容与参数保持不变。模型哈希不一致会先保存实际记录再停止；这正是本实验门槛。source-hashes.json/baseline.json 是复制时清单，后置 FULL 修改后的实际源码以 environment.json 的 source_hashes 为准。diagnostic 与 layout 的冻结源码也保留在本机完整归档。

Git 中保存所有 attempt 原始模型/结果/日志（stderr 和采样文本无损 gzip），以及脚本、候选 patch、汇总和 provenance。较大的 perf.data.gz 与完整冻结源码/二进制在本机 [完整原始归档](/tmp/bpe-frequency-20261014-raw.tar.gz)，校验见 archive.json；该本机附件不是远端持久存储。Git 内文件校验见 artifact-hashes.json。
