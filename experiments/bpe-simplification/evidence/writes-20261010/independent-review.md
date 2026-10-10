# 固定纯 Writes 快照的独立静态审查

Reviewer: `writes_only_final_review`，全新独立上下文、只读，2026-10-10。
本报告保留候选被拒绝前的固定对象事实；最终恢复对象另见
`../iterations-20261010/final-independent-review.md`。作者注：完整成本矩阵已按
用户拒绝决定停止，本报告中的成本待验证结论不代表当前仍需继续测量。

固定纯 Writes 快照的结构审查完成：新增具体缺陷 0；值得下一轮真实实验的
强候选 0。

已核对 HEAD `3a298346ffc47ada74eb800714644bf564b9e5c0`；工作区唯一差异为
merge.rs，git diff --binary 与指定补丁 SHA256 均为
`fa86dced33ea3220a0346491e3114de7bf3e64bf03a56d4fa5e37dd26a4db956`。

通读六个 engine 模块、公共 BPE／WordPiece、feed／WordCounts、顺序 reference、
现有测试及相关历史报告，按输入、初始化、选择、准备、应用、计数提交、
重启、输出、错误和生命周期追踪。

纯 Writes 的静态依据成立：

- fresh 保存 fresh_matcher 返回的完整 Match。旧 Compact 用相同规则跨度重建的
  start/right/after 与它一致；准备阶段 ID 几何固定。
- 全部准备读者 join 后才应用；Match 仍由 Job 持有，应用完成后才提交事件。
- reuse 已经保存完整 Match，本次统一没有改变其扫描域、历史 cohort 或
  occurrence 几何。
- record 原来的 push_ordered 分支始终成功，改为 unit 没有移除实际错误。
  birth／removal 算术及存储错误仍原样传播。

深模块边界也有实际用途：

| 边界 | 已核对的封装责任 |
| --- | --- |
| engine／公共 trainer | 一个训练入口隐藏专用池、Arena、阶段 join 和整次重启；调用方取得完整模型部件。 |
| Vocabulary／Corpus | 字符串身份、装饰和输出归词表；坐标、权重、激活与真实跨度归语料，初始元数据只移交一次。 |
| Batch／Prepared | 协调者不用解释 AA 起点、兼容规则、邻居事件及端点写入几何。 |
| PairIndex | 计数是权威，队列优先级可修正；路由与历史 cohort 的拥有权和顺序集中管理。 |
| Positions | 压缩、分块读取、Arena／owned 回收及 unsafe 生命周期隐藏在可信 Input 和借用接口内。 |

全局搜索中，计划／驻留语料的分离承载“计数压缩后再分配端点”；ID／occurrence
两种几何承载 alias 差异；Map／Entries 避免 feed 的额外转换。合并 best/take
会妨碍冲突判断后再消费候选；合并准备／应用会隐藏必要 join 边界。这些方向
未形成足够净收益的下一轮候选。用户拒绝的压缩 Birth 没有重新提出。

已读取 default／no-default 原始日志，各为 17 native＋1 doctest，以及 Clippy
成功日志。现有测试覆盖 AA、分裂 producer、reserved／alias、宽 ID、数值边界、
1／4／8-worker 完整模型与每步 trace，以及公共 feed／序列化／进度契约。
本审查没有执行目标、测试、编译或测量。

既有等优先 reuse cohort 次序证明缺口仍未关闭。本结论不宣称全语义等价。
fresh 写入缓存由窄坐标变为完整 Match 的资源代价真实存在；取舍不能由行数
减少或单长词压力输入代替。
