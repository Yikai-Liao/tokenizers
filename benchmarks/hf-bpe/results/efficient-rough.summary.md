# J与原efficient_bpe Rust粗略速度对照

## 结果

同一约16MiB中文语料、四线程，J训练API 1.477984秒，原Rust `ebpe`训练API 2.155127秒，**J约快1.46×**。各一次，无额外重复；按用户要求不检查模型结果一致。

| 指标 | 当前J | 原Rust ebpe |
|---|---:|---:|
| 初始化 | 0.235228s | 0.246095s |
| 完整训练API | 1.477984s | 2.155127s |
| HF串行feed | 0.116270s | 输入已预先准备 |
| HF feed+train返回 | 1.594255s | 此口径未测 |
| 训练阶段进程HWM | 154.66MiB | 160.32MiB |

原Rust整进程墙钟3.066秒，包含Prepared JSON读取/解码、训练及输出fingerprint；输入准备脚本耗时不在内。HF整进程也包含训练后fingerprint，其原始墙钟保留在JSON。二者进程输入格式和产物不同，表中不构造端到端加速比。这里只比较有范围差异的训练API，属于粗略速度参考。

## 输入与工作量

从当前512MiB中文Wikipedia文本取不超过16MiB的完整行前缀，实际16,776,351B，SHA-256 `ebd64699a917bb0982e9b64d302d94cc08057f43d16bcb5efc041594684c12a6`，无复制。J使用原feed逐行保留换行、none切分、重复行累计权重；Prepared桥接脚本按同样的完整行和权重生成整数语料，字符按Unicode排序映射1..alphabet，0为边界，权重u64。

两边均min_frequency2、四线程。HF vocab_size50,000，初始alphabet 9,241，因此原Rust max_rules 40,759，两边实际规则数也都为40,759。两边unique words 34,588、N 6,204,526、初始边数6,169,938、posting visits 4,257,889一致；这些只用于确认相近工作量，没有验证模型或逐轮规则一致。Prepared corpus含边界共6,239,115个位置。

HF默认AHash物理顺序未固定；桥接输入按首次出现顺序排列，初始字符ID与tie-breaking也可能不同。不能把速度差额归因于单个操作，也不将小语料比例外推到512MiB。

## 实现与计时边界

- J：`376363d25b6b1de917b9b8f68c11c0394ca37bb2`，原Trainer API，u32/AtomicU32、flat32 posting、batch256，使用同一J线程扩展binary。train包含字符编码、初始化、merge、词表/merge字符串构造以及训练内部清理；feed单列且串行。
- 原Rust：`/root/code/efficient_bpe` commit `8eb3cc6c3777e940988321571ec8cecaeae30498`，仓库当前README/default-run所选的 `ebpe`（`owned_grouped_inline`），使用现有实现并执行offline release build确认最新。默认chunk4096/lazy heap/ahash/Checked bounds。call从Prepared输入进入train到返回，包括validation、pool、初始化、merge、final及内部清理，排除JSON解码与CLI fingerprint。
- 编译：J使用既有默认Cargo release；原Rust沿用仓库release `debug=1/lto=thin/codegen-units=1`。这是现有实现之间的粗略比较，未统一编译profile。

原Rust CLI的`rules`是规则条数；`actual_merges`=3,060,366指位置替换次数，不能拿它与HF的`actual_merges`规则条数比较。

两次进程VmSwap采样峰均0B；最低MemAvailable HF 7.25GiB、Rust 7.22GiB。计时期间没有运行其它构建、测试或训练。

## Feed解释

HF feed是读取文本、调用切分函数、统计字符串片段频次并存入trainer；本次none时整行是一个片段。整数编码、初始pair计数和BPE merge在train内完成。

原实现已有`maybe_par_bridge()`+`map`+`reduce`并行路径；本系列benchmark runner在feed前显式设置`set_parallelism(false)`，训练前开启，并非HF实现只能串行。该配置用于沿用各版本前端与计时口径；本轮未测并行feed收益。

## 复现与证据

在`benchmarks/hf-bpe`目录运行，输出使用新名字，避免覆盖：

```bash
python3 prepare_efficient_fixture.py --source .build/gb-corpus/zh-512m.txt \
  --directory .build/efficient-rough-reproduction
/root/.cargo/bin/cargo build --offline --release --bin ebpe \
  --manifest-path /root/code/efficient_bpe/rust/Cargo.toml
# 完成构建后再计时，两个训练串行运行。
python3 run_native_fair.py --case j-efficient-rough-reproduction \
  --worktree /root/code/tokenizers-worktrees/prepare-rule-aggregate \
  --build-root .build/native-rule-aggregate-scaling \
  --binary target/release/hf-bpe-native-rule-aggregate-scaling \
  --corpus .build/efficient-rough-reproduction/zh-prefix.txt \
  --output results/j-efficient-rough-reproduction.jsonl \
  --initialization-workers 4 --merge-workers 4 --atomic-corpus
python3 run_efficient_rough.py --repository /root/code/efficient_bpe \
  --directory .build/efficient-rough-reproduction \
  --output results/original-efficient-rough-reproduction.jsonl
```

原始结果：[J](j-efficient-rough.jsonl)、[Rust](original-efficient-rough.jsonl)；各自environment记录source/lock/binary hashes；输入来源与参数见[input manifest](efficient-rough.input-manifest.json)，对齐检查与比例见[signature](efficient-rough.signature.json)。
