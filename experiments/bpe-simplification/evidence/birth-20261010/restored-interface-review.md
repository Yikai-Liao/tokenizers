# 回退 birth 后的独立接口审查

Reviewer: `deep_module_review`，独立上下文、全程只读，2026-10-10 06:21 UTC。
对象：恢复到 `3a298346` 的原引擎；复查 Birth 生产、路由、归并、发布路径，
继承其先前六模块中未变化的公共入口、输出、错误和生命周期审查。
这个结论与同目录中针对被拒绝 prototype 的旧报告分别保存。

恢复后的 `3a298346` 仍符合深模块原则。保留现有数据布局、提前剪枝、Complete
直接发布和 Partial raw 聚合后，本轮没有找到新的强结构调整候选。

`Birth::Partial(Builder)` 和 `Birth::Complete(Positions)` 是合理的内部能力契约：

- Partial 表达局部结果，总频率尚待 owner 归并。Producer 不能按局部 floor 丢弃；
  owner 汇总后才决定准入、排序和编码。
- Complete 表达这个 ordinary producer 覆盖完整候选，可以提前剪枝并冻结列表。
  Owner 因而可以直接发布，跳过 raw 聚合和再次编码。

完整性由 merge 内部的 `Source::complete` 判断，只有 `Neighbors::finish` 生产
这两个结果，只有 `PairIndex::commit` 消费它们。Owner 不需要理解任务分块、
AA 扫描或 cohort 扫描怎样推导完整性；引擎 coordinator 和公共调用者也不需要
匹配 Birth 类型。跨边界传递的是必要结论，完整性推导没有被重复实现。

上一轮“压缩 birth 消除了真实双表示”需要限定理解：prototype 确实删除了一套
表示和分派，但原双表示承担明确的计算与资源职责；它的存在没有证明原接口
过浅。深模块可以保留有用的内部特化。

本轮排除了把 `match Birth` 藏进新 wrapper、把 Complete 包成 Candidate，以及
移动 `prepare→apply→commit` 的建议：这些调整没有让完整性、频率、顺序或
所有权协议一起消失，部分方案还会增加元数据配对。

纯接口层面可以把两种 Birth 的契约就近说明得更明确，尤其说明 Complete 的
fresh/完整 producer 条件，以及 Partial 的 floor 归并责任。这属于文档澄清；
目前没有足够证据把它升级为结构重构。

已拒绝的压缩 birth prototype 及原报告事实保持不变；Writes 仍是独立既有候选。
此次结构结论不关闭等优先 reuse cohort 次序的既有证明缺口。
