# 对齐原型布局后的有限性能对比

实现入口为 `train_vocab_indexed()`，融合试验入口为 `train_vocab_fused()`。非空 prefix/suffix 自动保留通用 HF 路径；空 affix 的位置索引使用已证明的旧 pair 单调性，包括有限长度门控。

## 改动

- 扁平 u32 端点语料、每 ID 长度、每词边界及权重。
- 24 字节频率/posting 条目，16 字节 heap item，内联两个位置的 SmallPosting。
- 出生 pair 共享 8 字节链节点数组，不再每个 key 分配一个临时 Vec。
- 使用打包 u64 pair key；初始化保留 UTF-8 栈缓冲，普通字符只查一次 ID；canonical 字符串仍唯一映射 ID。
- 低频旧 pair 永久退役；本轮新 pair 等全部出生/扣减完成后再剪枝。

理论字节核算及与原型的逐项对照见 [MEMORY_LAYOUT.md](MEMORY_LAYOUT.md)。

## 三个版本

1. 固定 PR #2348 head `6ac0de53`：源码保持原样，只在临时副本加边界计时。
2. 位置索引：上述对齐布局。它已经包含 PR 可叠加的字符初始化技巧。
3. 融合试验：长片段仍走同一位置索引；平均初始长度不超过 32 个字符时，试用 PR 的连续 8 字节 symbol arena 和 read/write 顺序压缩。它与索引版共用计数、剪枝、候选选择和字符串身份处理。

## 范围与结果

只运行中文整行、英文空格切分两组，每组三个实现、各两次，合计 12 次，约 7 秒。4 MiB、目标词表 8000、min_frequency=2，同机单线程、独立进程、固定种子交错顺序。训练包括初始化、索引、合并和输出；输入处理另计。

| 输入 | 版本 | 初始化 | 合并 | 完整 BPE | 输入处理 | 总耗时 | 峰值 RSS |
|---|---|---:|---:|---:|---:|---:|---:|
| 中文整行 | PR | 0.460 s | 0.277 s | 0.829 s | 0.043 s | 0.872 s | 79.8 MiB |
| 中文整行 | 位置索引 | 0.302 s | 0.113 s | 0.446 s | 0.035 s | 0.481 s | 45.8 MiB |
| 中文整行 | 融合试验 | 0.287 s | 0.108 s | 0.417 s | 0.038 s | 0.454 s | 45.8 MiB |
| 英文空格切分 | PR | 0.097 s | 0.244 s | 0.364 s | 0.251 s | 0.614 s | 35.6 MiB |
| 英文空格切分 | 位置索引 | 0.069 s | 0.156 s | 0.233 s | 0.238 s | 0.471 s | 25.9 MiB |
| 英文空格切分 | 融合试验 | 0.084 s | 0.192 s | 0.285 s | 0.248 s | 0.534 s | 29.0 MiB |

各列独立取两次记录的中位数，不依此证明跨机器或所有配置的稳定收益。12 次完整词表与有序 merge 摘要均匹配。

位置索引的中文 BPE 约为 PR 的 1/1.86 耗时，英文为 1/1.56；本次英文没有相对 PR 倒退。

中文样本初始 alphabet 加预留身份共 7308 个，只产生 692 条 merge；英文产生 7002 条。因此这个有限对比验证了相应参数下的初始化和合并收益，没有扩展到更大的中文 merge 预算。

## 融合试验为何不推荐

英文的 arena 版本比位置索引慢约 22%。省掉权重二分查询的收益没有抵消整词扫描和初始化转换；它是存储/访问方式的替代方案，不能把 PR 的速度收益直接叠加在索引上。

中文两者都走 `endpoints_owned` 同一核心，计数器完全一致。两次中位数的差异不能视为融合收益。

因此位置索引版作为推荐的实验入口。保留 `fused` 的代码和记录用于审查，不将 arena 试验作为默认训练器。现有参考 `train_vocab()` 仍保持原行为。

## 核验与复现

28 项无默认 feature 的库测试通过，位置索引和融合路径都参加既有 1500 个随机配置的逐轮 HF 差分；另有全量 greedy oracle、有限长度和 SmallPosting 所有权/增长测试。未使用 suppress-warning；删除了未使用 import 和无调用的旧 Word 方法，本轮 release 编译无警告。

```bash
cargo build --release --locked --manifest-path benchmarks/hf-bpe/Cargo.toml
python3 benchmarks/hf-bpe/run.py /path/to/corpus \
  --profile focused --repeats 2 \
  --binary benchmarks/hf-bpe/target/release/hf-bpe-indexed-bench \
  --output benchmarks/hf-bpe/results/new-focused.jsonl
```

原始逐次记录：[compact-focused.jsonl](results/compact-focused.jsonl)。源码/二进制/语料摘要：[compact-focused.environment.json](results/compact-focused.environment.json)。自动汇总：[compact-focused.summary.md](results/compact-focused.summary.md)。源码记录对应计时时的版本，后续仅文档与注释调整不影响这些二进制的行为。
