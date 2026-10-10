# 最终恢复对象的独立复核

Reviewer: `writes_only_final_review`，独立上下文、只读，2026-10-10。
继承固定纯 Writes 的六模块和公共边界完整审查，复核被拒绝候选退出后的
最终恢复对象。

新增缺陷 0；值得下一轮实验的强全局简化候选 0。现有深模块边界仍成立。

对象为 bpe-heap-experiment 的 simplify/bpe-iterations-20261010，HEAD
`3a298346ffc47ada74eb800714644bf564b9e5c0`。唯一 tracked 源码差异是四行 Birth
契约注释；实际 diff 与指定 patch 的 SHA256 均为
`5c56edf4f17154dd937f7a2e16b3f214b3726e6ab8607f63e0c868a14d5061d0`。

已确认：

- Writes::Compact/Occurrences、Birth::Partial(Builder)/Complete(Positions) 及原
  性能路径恢复；隔离 Writes 工作树干净，其 merge.rs 哈希等于 HEAD 原文件。
- 四行注释准确描述 partial 的 owner 聚合／signed ledger 和 complete 的提前
  floor 判定／直接发布。
- 六模块及公共边界的既有审查依据继续有效：生产逻辑没有其他变化。语料几何、
  字符串身份、计数/cohort、准备应用和存储生命周期各有明确归属；未找到可
  一起退出真实状态、表示或协调协议的新方向。两项用户拒绝的候选均不再提出。

纯 Writes 的固定 fa86 静态报告保留为 0 新增缺陷、0 强候选，它不代表成本采用。
归档真实测量中，4-worker ByteLevel 三对正式样本的成对中位变化为 wall
+3.935%、CPU +1.124%、HWM +5.299%；Whitespace 仅一对，分别为
+2.352%／+1.136%／+0.0186%。12 个完整进程均记录完整模型相等、swap 0；
中断样本排除，6-worker 未启动。用户拒绝决定已落实为恢复。

既有等优先 reuse cohort 次序证明缺口仍未关闭，因此不宣称全语义等价。
本次没有运行、编译、测试或修改目标。
