# BPE 阶段等待实验

本轮从 `8faaff79d859bfd6b2417cfe8c93ea2851c3aaca` 独立开始，实验 corpus 写回与 owner commit 的重叠执行，以及初始索引每个分区连续完成排序、计数和编码。owner 路由、索引结构、位置编码及 corpus 布局保持原实现。

实验结论、原始指标、失败处理与 AA 耗时诊断见 [REPORT.md](REPORT.md)。候选没有继承上一轮 Owner／Prezza 的修改。

## 对照方法

沿用中英文 Wikipedia 各 256 MiB 完整行前缀、Whitespace、50K 词表、最低频率 2、4 个 worker 与 CPU 0–3。每个单元预热一次，再进行三轮交错对照，逐次校验完整词表 ID 和有序 merges。记录公开 `do_train` 调用耗时、CPU 时间和输出序列化前的进程 VmHWM。输入读取与预分词耗时不计入训练耗时。

先单独比较 `overlap` 与 baseline，再在一个对照组中比较 `sortcount`、`pipeline` 与 baseline。各轮只与本轮 baseline 比较，不跨轮拼接绝对耗时排名。至少一个语料的三轮配对耗时均下降且中位数下降至少 5%，才进入此前约定的 1 线程退化检查。这是继续实验的筛选规则，不是统计显著性判据。

## 复现

在 [tokenizers-bpe-benchmarks](https://github.com/Yikai-Liao/tokenizers-bpe-benchmarks) checkout 中使用其虚拟环境，按报告的源码 commit、相同 runner 锁文件和 release 配置构建。`bench.builds.build(source, revision, lockfile, cache)` 返回固定 binary 的 build record；源码、工具链和依赖来源随 record 固定。证据归档内包含本轮使用的 harness 与 `runner.lock`。

```bash
PYTHONPATH=. .venv/bin/python /path/to/tokenizers/experiments/stage-waits/run.py \
  --baseline /path/to/baseline/build.json \
  --arms overlap --overlap /path/to/overlap/build.json \
  --zh /path/to/zh256-words/manifest.json \
  --en /path/to/en256-words/manifest.json \
  --out .bench/stage-waits/overlap-t4

PYTHONPATH=. .venv/bin/python /path/to/tokenizers/experiments/stage-waits/run.py \
  --baseline /path/to/baseline/build.json \
  --arms sortcount,pipeline \
  --sortcount /path/to/sortcount/build.json \
  --pipeline /path/to/pipeline/build.json \
  --zh /path/to/zh256-words/manifest.json \
  --en /path/to/en256-words/manifest.json \
  --out .bench/stage-waits/initial-t4
```

`export_evidence.py --comparison <目录> --out <目录>` 重新验证每次运行的完整模型，并导出配置、计时、资源记录和模型指纹。只有完全相同的模型被合并为引用；词表 ID 和 merges 顺序保持原值。归档中的语料 manifest 记录来源、实际字节数、SHA-256 和预处理 recipe；完整语料与 binary 留在本机。
