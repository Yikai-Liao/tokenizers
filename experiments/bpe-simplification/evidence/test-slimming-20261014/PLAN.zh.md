# BPE 测试精简计划：以高层回归为主

## 目标与口径

基线：Yikai-Liao/tokenizers，refactor/bpe-corpus-boundary-20261013，
e4c35beeb6f8931bb6dc00a3162bca0bdff46a19。
普通 BPE 测试 1,729 行。
努力目标：≤1,300 行；原定上限：1,500 行。

排除空行和注释，计入测试辅助代码和属性、括号。保留原计数范围，不迁移文件规避统计。Parity Trainer 及仅供其使用的 word.rs 不计入。

原则：验证完整训练行为；只有高层难以可靠触发的风险，才保留专门的底层测试。不逐条搬运旧断言。

## 一、删除整组低价值测试

| 位置 | 删除的测试 | 不再单独验证的内部细节 |
|---|---|---|
| corpus | initial_ranges | 人造布局与精确分片公式 |
| corpus | fresh_reader | 预取窗口、缓存与人造 reader 状态 |
| index | unified_aggregation | 分片事件及载荷的具体排列 |
| index | historical_cohorts | 相同优先级下内部载荷的出堆身份 |
| merge | shared_boundary_once | prepared job 与原始槽位的具体形态 |
| merge | rejected_start_and_aa_tasks | 精确任务数、首任务长度、人造无效起点 |
| merge | reuse_spans | 内部 occurrence span 数组 |
| merge | directory_recovery | 注入失败后继续复用同一 scratch 的行为 |
| merge | reuse_scan_domain_and_bound_configuration | 人工截断索引后的私有扫描范围 |
| positions | leaf_format_and_exhausted_cursor | 私有字节布局、Cursor 大小、重复耗尽行为 |
| vocabulary | decoration_row | 底层槽位 getter/setter 与大小 |
| tests | initial_edges | 私有边坐标及标准迭代器错误恢复 |

这些用例当前合计 400 行。删除后清理 merge 测试模块、导入和 select 辅助函数，另约 27 行。

这不是声称每个旧断言都被等价替代，而是明确放弃绑定内部实现的测试，将预算投向可观察的训练结果。

## 二、用少量确定性训练用例补齐关键场景

直接复用现有 check 和独立 oracle，比较完整词表及 ID、有序 merges、每步合并 trace，并跑现有 1/4/8 worker 对照。

新增一个短测试，例如 targeted_models，包含：
1. 共享边界：abacdbdc×2、ab×10、ac×9、db×8、dc×7。
2. 多词与权重区间：40 个不同词，具体用 `"x".repeat(i)+"ababa"`，i=0..39；权重按每 10 词一组取 [9,1,1,0]。
3. 别名与长度限制：baaaaaba×2、ba×1，suffix 为 a，分别用 None、Some(3)、Some(usize::MAX)。
4. 字母表过滤：复用原 initial_edges 的 Unicode 语料，limit_alphabet=2；补空输入。
   原语料：qé🙂z×7、é🙂é×6、🙂é×5、空串×4、é×3。

把已有长 AA 用例从 8195 个字符扩到 20001，继续覆盖跨任务训练，不断言任务必须恰好切成几块。

新增夹具预算约 40–45 行。不新建测试框架，不在高层测试里重建旧的私有状态检查。

## 三、移除低价值监控与布局断言

- 删除逐阶段 Phase、位图和生产模块中的仅测试观测钩子，约 50 行。保留公开并行设置、feed 回调线程检查、进度输出、并行使用标记和模型对照。
- 删除 Positions 测试里仅检查 sizeof、SmallVec 内联容量和 spill 形态的断言，约 11 行。保留 roundtrip、seek、restart 和 chunks 结果检查。

明确取舍：移除 Phase 后，不再逐阶段证明训练代码在哪个 worker 上执行。公开行为测试不等价于原来所有内部监控。

## 四、保留高价值防线

- 独立顺序 oracle 和 trace；不修改历史 cohort、计数修复及优先级语义。
- 独立 Unicode 与字母表期望，因为 oracle 共用了字母表选择。
- u64 计数边界、宽 token ID、重试身份与字母表一致性。
- 相邻并发写入、中间溢出、prepare 与 commit 失败边界。
- codec 完整 u64、重复值、重启边界、拼接、查找及追加失败。
- 公开模型构建、序列化、输入可复用和失败不替换旧模型。
- feed 并发与部分失败测试，包括现有交错用例。

本版不再靠削弱 feed 并发交错测试省行；也不删除或忽略已知失败的 tokenizer_roundtrip。

## 五、短命名

删除的函数无需改名。保留的长名称收短：
- models_match_reference → parity
- failed_training_keeps_model → failed_train
- inputs_remain_reusable → reusable_input
- public_feed_pools_and_progress → public_behavior
- failures_before_and_after_writes → failed_rounds
- append_search_and_chunks → leaf_roundtrip
- concat_preserves_boundaries_and_chunks → concat_boundaries
- append_errors_preserve_leaf → append_errors
- gap_width_matches_appended_bytes → gap_width

同步更新 --exact 路径及过滤器，核实实际执行数量。

## 六、预算与验收

已统计删除区域：400+27+50+11=488 行。
预留新增夹具：40–45 行。
规划结果约 1281–1286 行。
这是现有区域的预算，不是已经修改、格式化后的实测结果；不得再叠加上一版删减估算。

实施前确认 HEAD；如果有新提交，先更新统计基线。
先加入确定性高层用例，再删除对应底层测试，最后清理失去用途的辅助代码。
每一步检查默认与 no-default-features 配置，比较失败集合，单列已有基线失败。

最终交付完整 diff、删除理由与高层替代位置、主动放弃的内部覆盖、实际测试结果、格式化后逐文件行数。
超过 1300 时报告差额，不通过挤排版或删关键防线伪造达标。

## 源码参考

- https://github.com/Yikai-Liao/tokenizers/blob/e4c35beeb6f8931bb6dc00a3162bca0bdff46a19/tokenizers/tk-train/src/trainers/bpe/merge.rs#L467-L683
- https://github.com/Yikai-Liao/tokenizers/blob/e4c35beeb6f8931bb6dc00a3162bca0bdff46a19/tokenizers/tk-train/src/trainers/bpe/tests.rs#L37-L144
- https://github.com/Yikai-Liao/tokenizers/blob/e4c35beeb6f8931bb6dc00a3162bca0bdff46a19/tokenizers/tk-train/src/trainers/bpe/tests.rs#L522-L714
