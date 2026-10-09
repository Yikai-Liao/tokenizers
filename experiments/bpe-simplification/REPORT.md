# 最终简化 BPE engine

已完成结构、测试及完整计数范围的 fresh 独立审查。源码 `4d181c51` 的生产逻辑 **2100/2100 行**，默认 BPE 测试、oracle、公共 helpers 及 Miri harness **780/800 行**；按 rustfmt 后非空非注释行统计。实现集中在六个 engine 模块，未把生产算法移到统计范围外。

语义基线固定为 Fork main `e4f787dc189d9be7192107490d652096cde7480e`，工作分支 `simplify/bpe-maintenance-20261009`。选择状态、CorpusPlan 到 Corpus 的构建边界、owner 路由和 codec 可信输入已收紧；最终 whole-crate fresh 审查没有新的实质、可落实发现。见 [REVIEW-12.md](REVIEW-12.md)。

## 未插桩最终对照

中文 4 workers、ByteLevel、50K 词表、min_frequency=2 的公开 core 训练为 **21.248 秒**，满足 25 秒目标。一次相邻 main/最终版对照共八个独立进程，完整 vocabulary IDs 和全部 ordered merges 均一致，child swap0。

| case | main train s | 最终 train s | main pipeline s | 最终 pipeline s | main RSS GiB | 最终 RSS GiB |
|---|---:|---:|---:|---:|---:|---:|
| en-core | 1.069 | 1.541 | — | — | 0.174 | 0.188 |
| zh-core | 16.004 | 21.248 | — | — | 2.437 | 2.943 |
| en-pipeline | 0.965 | 1.379 | 5.016 | 5.049 | 0.175 | 0.184 |
| zh-pipeline | 15.727 | 21.083 | 19.813 | 25.029 | 2.397 | 2.694 |

两边同一输入、release profile（opt3、fat LTO、一个 codegen unit、无默认 features），绑 CPU0–3。主机是 6-vCPU Xeon KVM、约 16GiB RAM。Core 使用同一个预处理 word map；pipeline 包含 feed。RSS 是序列化/校验之前的进程 HWM，包含输入加载，不是 engine 专属占用。

中文 core CPU 为 main 52.794s / 最终 68.621s；最终 wall 高 32.8%、RSS 高 20.8%。达到 25 秒目标后仍保留与 main 的差距。这是每 case 一对样本，不能给出稳定百分比或置信区间；旧版本、旧主机与插桩结果没有混入本表。最初发现并发 rdst 基准后已停止并排除该次记录，正式对照在空闲主机重新执行。

当前最终 binary SHA-256：`1ab2fbb529c05f13119f46cd77a553099ecd65e7ea5d12a080b4cf0b10db1c97`。完整 source/input/binary hashes 见 [manifest-review-final.json](evidence/manifest-review-final.json)，八进程数据见 [review-final-runs.json](evidence/review-final-runs.json)，计数见 [lines-review-final.json](evidence/lines-review-final.json)。

## 行为与存储边界

保留完整 IDs/merges、兼容优先级前缀批次、并行 prepare/apply/owner commit、AA 左到右选择、reserved/active 身份、重启、signed reuse cohort 及严格 birth length。Positions 支持完整 u64 域；resident 分配仍检查 usize/isize 边界。每个并行阶段在进入下一阶段前 join，失败丢弃整个 attempt。

Positions 的最终描述符为 16B。小最终分配借用 attempt Arena，大分配单独拥有堆存储；临时 Builder 用 u32，并在需要时无损提升为 u64。Worker lease 中复用编码 scratch。完整 ordinary producer 先按 floor 剪枝再编码，owner 直接发布；partial/AA/reuse 保留聚合路径。Apply 的事件块直接交给 commit，复用紧凑路由容量。

Arena 阈值公式为 `max(256, floor(sqrt(N / 256)))` 字节；当前 N 是 resident slots，main N 是初始 physical edges。九个三语料/三规模 case 的同 binary 消融支持保留现状，见 [CUTOFF_ABLATION.md](CUTOFF_ABLATION.md)。当前初始索引按约 4×workers 个词块并行收集后统一 owner 聚合，没有 main 的 2^28-slot waves。

## 阶段、规模与代码收益

最终源码的阶段 wall/CPU/RSS 归因与 384/512MiB 中文内存对照见 [PERFORMANCE_REVIEW.md](PERFORMANCE_REVIEW.md)。未插桩全程 HWM 增幅约 20%–22%，初始阶段诊断峰值增幅从约 30.8% 升至 64.3%；main 实际从 1 wave 升至 2 waves。净代码行数收益及已删除低收益目录见 [OPTIMIZATION_ROI.md](OPTIMIZATION_ROI.md)。这些单次、不同边界的指标不能互换。

## 验证与复现

默认/无默认 features 各 17 项 library tests 通过；无默认 features 的 1 项 doctest 通过；all-target Clippy `-D warnings`、rustfmt、行预算、git whitespace 检查通过。Standalone Miri 直接引入实际 positions.rs，2 项组合测试在默认借用/泄漏检查下通过。Native 覆盖全部 seek 起点，Miri 覆盖关键重启边界、双 worker lease 和完整 u64 范围。

Rust 源码与完整验证的 `e26115c2` 相同，之后两个提交修正测试计数范围。公开测试覆盖 feed flush/错误/非 fused iterator、真实线程池、JSON/encoding/special-token roundtrip、严格 progress schema；语义测试与独立 oracle 比较完整模型及逐条规则。覆盖清单见 engine/tests/COVERAGE.md，原始验证日志见 evidence/*review-round2*.log。

```sh
cargo test --manifest-path tokenizers/tk-train/Cargo.toml
cargo test --manifest-path tokenizers/tk-train/Cargo.toml --no-default-features
cargo clippy --manifest-path tokenizers/tk-train/Cargo.toml --all-targets -- -D warnings
python3 experiments/bpe-simplification/count_lines.py
cargo +nightly miri test --manifest-path experiments/bpe-simplification/miri-codec/Cargo.toml
```

原始 stdout/stderr、records、不可变 binary 和输入保存在 `/root/code/tokenizers-simplification-results/`。没有 push 或新建 PR。早期交付及探索数据保留在 git 历史和现有 evidence 中；不能替代这里的最终源码验证。
