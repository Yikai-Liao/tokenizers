# 后续简化点：独立只读审计

用户要求独立 subagent 再检查整个 BPE 引擎，寻找与统一堆类似的潜在简化。
目标为 `e2d74e79` 加当前 index 补丁，index SHA256 为
`191de8af497430a99c10a43404933ec051d5e4c3231d318e0e538a0d8cca4cc8`。
CodeGraph 返回主工作树旧布局，结论逐一用目标工作树源码回证。没有修改源码、
执行目标代码、测试或测量。全流程六模块审计只找到一项值得实验的结构候选。

## 统一准备阶段的 birth 交付表示

同一条 birth 流程在两个阶段解释 producer 完整性：

- `merge.rs:36`：`Birth::Partial(Builder)`／`Birth::Complete(Positions)`。
- `merge.rs:141`：完整 producer 提前 floor 剪枝，再选择 raw／compressed 交付。
- `index.rs:310`：Complete 立即发布；Partial 分组、拼接、汇总剪枝和编码。
- `positions.rs:97`：`Builder::append` 的生产调用仅服务 Partial 聚合。

建议准备阶段统一交付 `Positions`。完整 ordinary producer 仍提前剪枝并按原
Arena 阈值编码；Partial、AA、reuse 通过现有 `from_sorted_owned` 压缩成可
单独释放的片段。Owner 统一聚合 count 和压缩片段；fresh 单片段原样移交，
多片段使用已有 `Input::Fragments` 合并；reuse 仍解码、排序、生成历史 cohort。
初始化已经使用 producer 压缩片段、owner 归并、单片段移交，见
`index.rs:147` 与 `index.rs:180`。

这样可能连带退出 Birth 双表示、空值分派、Complete 独立发布分支、raw
`Builder::append`。预计净减少数十行，须实现后确认。主要收益是提交阶段不
再依赖准备阶段传来的完整性分类。

## 必须保持的边界与待测成本

fresh newborn key 含本批新 replacement ID；本批后续 removal 来自快照旧
边界，静态未发现把 Complete 插入延后到分组结束会改变 removal 的路径。
Partial 仍必须在总 count 汇总后剪枝，不能按片段 floor 丢弃。reuse 继续按
逐 action 的 signed ledger、remove-before-birth、bucket、左右顺序、重复
坐标和 cohort 发布条件处理。Arena、动态阈值、可信 Input、在线初始压缩、
scoped 生命周期保持。此建议不解决当前等优先 reuse cohort 次序限制。

Partial 多一次片段编码和后续解码，完整 birth 多一次分组；压缩可能降低 raw
驻留，但归并期间片段与最终列表并存。这些是需要实测的取舍，不能据静态
成本直接否决，也没有证据可预先宣称提速或降峰值。

最小验证应覆盖：各 Partial 低于 floor 而合计达 floor、Complete 与 Partial
同批、AA、alias／零权重、严格 length gate、signed 与 full-u64 边界，再对
冻结中文词频做完整模型／trace 和 whole-training wall、CPU、HWM 对照。
现有入口为 `engine/tests/mod.rs:83`；codec 已覆盖 owned fragments、重复值
和全 u64。

没有列第二项凑数。按值路由虽可退出共享 metadata 与双数组对应关系，但会
增加大 payload 和重复 scalar 搬运，未证明净复杂度更低。历史 owner-directory
与 wave barrier 没有新证据，未重新提出。
