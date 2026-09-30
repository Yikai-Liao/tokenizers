# BPE 实现与 worktree

每种实现有独立本地分支和提交，训练均走原有 `BpeTrainer::do_train`、`train_vocab`、`Trainer::train`。新实现没有新增公有 Trainer 字段、serde 字段或选择算法的公有方法；内部差分参考和实验统计保留在私有模块。

## 固定源码

worktree 均位于 `/root/code/tokenizers-worktrees/`。以下提交可直接 checkout；中央实验记录在 `/root/code/tokenizers` 的 `bpe/experiments` 分支，迁移代码共同基点为 `bpe/migration-base` 的 `8c968e10`。

| worktree | 分支 | 当前提交 | 原接口使用的算法 | 本次计时 |
|---|---|---|---|---|
| `hf-reference` | `bpe/hf-reference` | `bbccb051` | 未修改 HF 分支参考，Word/cohort | 未新增大语料计时 |
| `pr2348` | `bpe/pr2348` | `6ac0de53` | 固定 PR #2348，WordArena/cohort | 512 MiB，串行计数/4线程 merge |
| `indexed-serial` | `bpe/indexed-serial` | `af6aff33` | u32 endpoint，串行 posting 更新 | 本次未计时 |
| `fused` | `bpe/fused` | `7c0e29e5` | 按piece几何选择WordArena或endpoint，融合检查/更新 | 本次未计时 |
| `parallel-count1` | `bpe/parallel-count1` | `4714afd3` | u32 endpoint，1线程初始化、4线程 merge | 公平初始化控制 |
| `parallel-count4` | `bpe/parallel-count4` | `b6a28768` | u32 endpoint，4线程初始化与 merge | 初始化并行优化 |
| `parallel-atomic` | `bpe/parallel-atomic` | `65059c49` | 同 count4，语料 Relaxed AtomicU32 | 访问成本对照 |
| `corpus-parallel` | `bpe/corpus-parallel` | `98ca7fc1` | A：并行规划、独占区域填充，原字母表/哈希字符查询 | 512 MiB及具体回退诊断 |
| `corpus-direct` | `bpe/corpus-direct` | `fbdc0b2b` | C：并行存在位图、字符直接表、最终数组直接初始化 | 512 MiB，39.805秒训练 |
| `corpus-direct-atomic` | `bpe/corpus-direct-atomic` | `27dc7fd0` | C初始化，原Plan算法，只改Atomic入口 | 按用户要求取消计时 |
| `fused-direct-atomic` | `bpe/fused-direct-atomic` | `c1ee2019` | B：C初始化，flat非AA过滤/delta融合、有效起点写入 | 回退：49.697秒训练、31.887秒merge |
| `fused-lookup` | `bpe/fused-lookup` | `a0832c48` | B2：融合B增加稀疏权重目录与selected直接表 | 44项测试通过，prepare11.889秒/merge21.386秒；追加模块稳定性检查 |
| `corpus-atomic` | `bpe/corpus-atomic` | `082d2811` | 早先A初始化Atomic控制起点 | 未计时，保留历史候选 |
| `fused-batch-atomic` | `bpe/fused-batch-atomic` | `082d2811` | 早先A融合候选空起点 | 未实现，后续改用C parent |

测量时的原接口提交为 count1 `63e384b8`、count4 `c07a6e39`、atomic `f2c5415f`；serial/fused首次接口提交为 `45fb6b95`/`0d509384`。表中当前提交只清理benchmark遗留indexed feature和版本说明，训练源码与测量提交逐字相同。测量脚本及原始数据固定在中央提交 `1396274f`；最终脚本还将固定4线程参数显式限制为4。

表中4线程是本次 runner 显式设置的运行条件。三个并行分支的 merge 线程数服从现有 `tk_encode::parallelism` 控制；关闭并行时使用1线程。count4/atomic 初始化线程数随 merge，count1 固定初始化1线程。串行 endpoint/fused 核心没有并行 merge。

三个并行分支固定 `narrow_corpus=false`、`posting_block_bits=32`、`batch_size=256`；超过单个 u32 地址块时，posting 保存局部 u32 偏移、块基址及全局计划位置使用 usize。普通字符配置使用精确规则批次；非空 affix 转入保留 HF 账本/cohort 的串行兼容路径。串行/fused 同样保留该兼容路径。

## 原接口调用

在选定 worktree 构建 `tk-train`，所有版本使用同一段代码：

```rust
use tk_train::{BpeTrainer, Trainer};

let mut trainer = BpeTrainer::builder()
    .show_progress(false)
    .vocab_size(50_000)
    .min_frequency(2)
    .build();
trainer.feed(["测试文本", "另一段文本"].into_iter(), |line| {
    Ok(vec![line.to_owned()])
})?;
let (vocab, merges, special_tokens) = trainer.train_vocab()?;
```

中央根目录保留历史外挂接口以复现旧实验；它是实验账本，五个实现分支才是本次原接口交付。根目录旧 runner 的 `indexed` 默认 feature 也只用于历史实验。

## 复现新的公平比较

先构建全部二进制，再串行计时。`build_native_fair.py` 要求干净 worktree，仅在临时副本的原始 `do_train` 返回处加一个统计输出探针；runner 编译时禁用 `indexed` feature，选择 `reference` backend。因此 runner 实际调用选定分支的原始 `train_vocab`，不调用实验外挂接口。

```bash
python3 build_native_fair.py /root/code/tokenizers-worktrees/parallel-count1 --label count1
python3 build_native_fair.py /root/code/tokenizers-worktrees/parallel-count4 --label count4
python3 build_native_fair.py /root/code/tokenizers-worktrees/parallel-atomic --label atomic

python3 run_native_fair.py \
  --case native-count1 --worktree /root/code/tokenizers-worktrees/parallel-count1 \
  --build-root .build/native-count1 --binary target/release/hf-bpe-native-count1 \
  --corpus .build/gb-corpus/zh-512m.txt --output results/native-count1-reproduction.jsonl \
  --initialization-workers 1 --merge-workers 4
```

count4 对应初始化4线程；atomic 还需 `--atomic-corpus`。输入、工作树、实际临时源码、runner、Cargo.lock、脚本与二进制摘要记录在各项 environment JSON。runner 在 feed 前显式关闭并行，feed 后显式开启4线程训练，避免环境变量被误当作实际算法配置。脚本检查统计中的初始化/merge线程数、宽度与原子标记，并保留异常记录。

结果与限制见 [PARALLEL_REPORT.md](PARALLEL_REPORT.md)，演变过程见 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)。

## 本轮热点优化

新增候选保持统一u32/32、4线程初始化与merge、batch256，原接口服从相同parallelism控制。A/C没有修改merge源码；C的字母表None按存在集合并行统计、Some沿用原裁剪。B修改flat非AA的filter/plan/delta/rewrite组织；AA、多块及非空affix保留原路径，owner提交接受按唯一规则生产者生成的有序posting。融合源码没有修改初始化模块。

本轮计划与变更范围见 [OPTIMIZATION_PLAN.md](OPTIMIZATION_PLAN.md)，原始数据位于 `results/optimization-*`，独立审查见 [CORPUS_DIRECT_REVIEW.md](CORPUS_DIRECT_REVIEW.md)。新增Atomic对照已取消，不展开矩阵。
