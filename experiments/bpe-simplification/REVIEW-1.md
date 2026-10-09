# 只读简化审查 1

审查者：lean_engine_review_1。固定提交2b55b7f6，1702生产行、1392测试行。
阅读software-design-philosophy与Rust规则；CodeGraph未索引该工作树，直接阅读源码。
不编辑、不编译、不跑测试或benchmark。

- 保留Vocabulary/Corpus/PairIndex/Batch-Prepared/Positions边界；上层流程不泄露endpoint和目录知识。
- Scratch组合收益未隔离，中文约5.4%改善接近波动、英文更慢。建议map_init省共享锁/worker耦合，u32目录减驻留；实际实施后1681行，比估计减少21行。
- initial已直接压缩并按chunk顺序owner串接；reuse unordered仍须排序。
- 可删scan_symbols始终为0的offset参数及initial_spans转换；可试first State移入，避免首次重编码。
- 小域dense初始State方阵可能昂贵，需要独立消融，不能归因于延迟materialization。
- fresh固定几何缓存与bounded head-read应分别试；原profile不足以证明当前热点。
- 未发现确定U64/计数/并发缺陷。prepare/apply join、disjoint endpoints、fresh唯一producer、signed cohort bits排序必须保留。
- 文档的fresh“先排序后编码”已过时，需更正。
- 最终对少量竞争组合补相邻复测；只称已测候选中的选择，不称全局最优。
