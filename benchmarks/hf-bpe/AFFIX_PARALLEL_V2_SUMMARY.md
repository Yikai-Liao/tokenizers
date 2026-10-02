# Affix 通用并行路径 v2 测量摘要

## 测试范围

本轮测量使用冻结源码 `65524228234acdc44ce870848632719ad20220c5`，build label 为 `affix-parallel-v2`，binary SHA256 `a5186f76f35f2c5e08cd294d34e35ab88d50ef14bafa268854d183703dec86e7`。只在隔离 build 副本里加入 runner 和统计输出；源 worktree 保持 clean。每次训练前从 baseline environment 读取绝对输入路径、SHA256、词表大小、最小词频、affix 和请求线程，并在启动 native 前核对路径与数据哈希。EN16 输入 SHA 为 `41b20e33253a4db8c66b95536cbadc86c79072e81eb9e5cd59ef136490abdec9`；ZH16 输入使用 `zh-16m-from-512m.txt`，SHA 为 `ebd64699a917bb0982e9b64d302d94cc08057f43d16bcb5efc041594684c12a6`；所有 16 MiB 案例词表 30,000、最小词频 2。

`train` 是扣除 feed 后的训练时间，`wall` 含 feed 与 runner 开销。时间来自单次测量，主要用于方向判断。输入、环境、完整模型 SHA、build manifest 与原始统计保存在 `results/affix-parallel-v2/`。

## EN16 / ZH16 主矩阵

主指标是同一冻结源码、同一输入的 affix4 / none4 比值。候选 affix4 请求四线程；所有候选完整词表和有序 merges 的 SHA 都与各自最旧基线和 v1 候选完全相同。

| 语言 / affix | v2 affix4 train | 同源码 none4 train | affix / none4 | v1 affix1 train | 最旧基线 affix1 train | 峰值 RSS |
|---|---:|---:|---:|---:|---:|---:|
| EN `##` | 4.472 s | 2.142 s | 2.09× | 6.285 s | 8.672 s | 0.403 GiB |
| EN `</w>` | 4.536 s | 2.142 s | 2.12× | 6.500 s | 7.660 s | 0.416 GiB |
| EN both | 4.300 s | 2.142 s | 2.01× | 6.219 s | 8.462 s | 0.400 GiB |
| ZH `##` | 2.083 s | 1.017 s | 2.05× | 2.524 s | 4.614 s | 0.255 GiB |
| ZH `</w>` | 2.383 s | 1.017 s | 2.34× | 2.850 s | 4.536 s | 0.265 GiB |
| ZH both | 2.082 s | 1.017 s | 2.05× | 2.647 s | 4.639 s | 0.253 GiB |

每个四线程 affix case 的统计均为 `workers=4`、`initialization_workers=4`、`layout=hf_cohorts`、`monotone_pairs=false`、`initial_count_backend=cohort_stable_radix16`；都观测到 5 个进程线程。`cohort_parallel_rounds` 在 EN 为 816–821，在 ZH 为 38–42，且 `cohort_parallel_jobs` 大于零。所有案例 `VmSwap=0`、System arena/backing 为零；heap requested bytes 与 freed bytes相等，buffer 数与 free 数相等。

v2 的训练时间明显低于 v1 的单线程 affix 时间，但 affix 与同源码 none4 仍相差约 2.0–2.34 倍。EN 的 parallel apply 局部 rewrite 与 commit 合计约 1.04–1.18 秒；ZH 合计约 0.04 秒。EN prefix 的 merge 为 3.638 秒，其中 `fused_prepare_ms=0.756s`，`commit_ms=0.398s`。ZH suffix merge 为 1.818 秒，其中 `fused_prepare_ms=0.019s`，`commit_ms=0.018s`。`fused_prepare_ms` 统计 word 内改写和局部目录，已经包含在 merge 中；`commit_ms` 统计全局 delta 与出生 cohort 组装。未赋值的诊断计时零值不能解释为没有对应开销。结果显示 ZH 剩余时间主要不在 parallel rewrite/commit，EN 仍有显著成本落在小 posting 串行处理、选择队列等阶段。

## Alias 正确性与并行扫描

5 字节输入 `baaba`（无换行），suffix `a`、vocab 10、min frequency 1，v2 SHA 为 `6049fd68ea6a27340933f0fe86631118875fda81e5b59e5c8ddddfdbd1704bb8`，与旧 baseline 和 HF greedy oracle 一致。v2 记录 `reused_ids=2`、`cohort_scan_activations=4`、`cohort_words_scanned=4`、`word_scan_steps=9`。这个极小案例是串行验证，所以不要求 parallel rounds。

另用 1,024 个唯一行测试活跃 alias 的并行扫描：每行以 `abaa` 重复 64 次开头，再接十进制 ID 与换行，总输入 266,154 字节；prefix `ab`、vocab 128、min frequency 2。v2 训练 38.9 ms，`reused_ids=1`、扫描激活 95 次、扫描 9,984 个 words、`word_scan_steps=65,308`，并发生 8 轮 / 32 个 parallel jobs。v2、冻结 482 baseline 与 HF oracle 的模型 SHA 均为 `a6579a913bc0055fcc421d58f76a97340c4a25b272facbb4b5c4f3b00494d8cd`。同源码 none4 control 训练 30.0 ms，扫描案例自己的 SHA 为 `9351878fd3a69da59881dca103969b9110e858db3f05fe3063ee4ac3451ce35f`。该 alias 案例 v2 比旧 baseline 的 81.2 ms 快约 2.09 倍，比 HF oracle 的 253.7 ms 快约 6.5 倍；相对 none4 control 慢约 1.30 倍。该进程短于 OS 线程采样间隔，记录到的 OS 线程数为 1，不能据此声称观察到 5 个 OS 线程；统计中的 pool 与 initialization workers 均为 4。

## ZH512 MiB

ZH512 输入是同一份 536,870,289 字节语料，SHA 为 `a0d40d4102cba1e933f25e5ccd17552d2eaebaec4ef770bc39d8e46e740b4d4f`；vocab 50,000、min frequency 2。

v2 suffix4 完成于 91.40 秒 wall、86.29 秒 train，模型 SHA `9e7dd5f1d28beaabcbae59df61b574595e7bb6e38facaeb76d77bab95b68c056`。峰值 RSS 为 5.10 GiB，最低 MemAvailable 为 2.28 GiB，进程 VmSwap 为零，OS 线程数为 5。none4 同源码 control 为 24.62 秒 wall、18.79 秒 train、3.31 GiB RSS，SHA `d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd`。因此 v2 affix / none4 train 比为 4.59×。v2 affix 虽在 180 秒 cap 内完成，但 RSS 没有改善：它比 v1 suffix1 超时记录的 4.91 GiB 高约 0.19 GiB，也比最旧 suffix1 超时记录的 3.36 GiB 高。此处请求线程分别为 v2=4、旧记录=1，内存值不是同线程严格对照。

v2 的 `cohort_parallel_rounds=5,982`、jobs=23,928、delta groups=20,029,899；merge 70.82 秒，含 fused_prepare 11.59 秒和 commit 15.32 秒。System arena/backing 为零，heap requested/freed 均为 1,600,995,336 字节。最旧 affix 512 MiB 运行和 v1 suffix1 运行均超时，没有可供比较的完整模型 SHA；本次 512 MiB 结果只能证明 v2 在当前设置下完成，不能证明完整模型语义与旧 affix 版本完全一致。原有 none4 基线 24.375 秒、3.27 GiB RSS；新源码 none4 control 是当前源码内的直接参照。

## 历史记录排除项

v1 测量中有一次 ZH16 错误地读取了 `zh-16m.txt`，而配对 baseline 使用 `zh-16m-from-512m.txt`。该次模型 SHA 不同且 unique-word 数不同，记录保留在 `results/affix-general-v1/failed-wrong-input/`，未计入比较。v1 有一条旧 baseline artifact 名为 `en16m-none-t4`，environment 请求线程为 4，但 JSONL 实际统计 `workers=1`、`initialization_workers=1`；该条不作为四线程基线。有效 EN none4 记录为 `en16m-none-workers4`。

6b67536 的统一消融 build 在 native 计时开始前被取消：复核发现公共入口仍强制 `narrow_corpus=false`，导致测到的不是目标默认路径。该 build 状态标记为 `cancelled_no_native`，没有任何 benchmark case 使用它；后续测量应以修正该入口后的新冻结 commit 为准。
