# 起点类型与邻接参数收尾：独立审查

建议接受当前 diff。未发现 blocking 或需要追加修改的 nonblocking 问题。审查仅限相对已推送 `30f06c04` 的 `corpus.rs`、`merge.rs` 未提交变化；没有编辑源代码、运行编译或性能测试。

`merge.rs:189` 的 Job starts、`:618` 的 CohortPreparation starts 改为 `SmallVec<[usize; 2]>`，fresh/cohort 在 `:525`、`:704` 直接保存 `matched.start`；`corpus.rs:444` 接受 `&[usize]`。这让 preparation 接受的 resident 坐标直接贯穿 application，删除了 `usize → u64 → resident` 的转换及重复校验。起点不再看起来像需要重新解释的 Positions 编码坐标，理解这条数据流的成本更低。

所有生产和测试 `apply_merges` 调用已适配。测试反序列表由 Job 元素推导为 usize，alias 测试也直接使用 FrozenMatch.start。剩余 `u64` 转换服务 Positions/self-pair membership 和邻接 birth 的既有边界；这些没有理由随 Job 改成 usize，不构成漏改。SmallVec 内联容量仍为两项；64 位目标的元素宽度不变，32 位目标与 resident 索引宽度一致。

删除 application 中的 resident 检查没有让正常生产输入失去验证：fresh 的 Positions 坐标在 prefetched 入口已转为 resident，随后 matcher 产生 Match；cohort 从 resident word scan 接受 Match。Job 中保存的是这些已经接受的 start。非法内部 starts 仍会由 token/span slice 索引检查拒绝，panic 文案可能不同，但该非法输入不属于接口的已接受起点契约。空列表、缓存时机、两个 application arm、端点 store 顺序和整轮不重叠要求均未改变。

`merge.rs:543` 的 record_neighbors 删除 p 参数，在 `:549` 使用 `matched.start`。此前唯一调用处的 p 正是 fresh matcher 构造 Match.start 的位置，行为等价；现在调用者无需重复提供同一坐标或维护 `p == matched.start` 的隐含要求。邻接逻辑及事件顺序没有变化。

源码 diff 包含这两个收尾与必要的测试类型适配，未扩展到 Positions、index、身份激活或 commit。编译及定向验证结果由主 agent 记录，本审查不作性能结论。
