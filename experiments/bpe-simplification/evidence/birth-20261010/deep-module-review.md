# 独立深模块结构审查

用户要求再次让 subagent 判断当前结构是否符合深模块原则，并寻找进一步结构
调整。本轮为新上下文只读审查，固定对象为 `3a298346ffc47ada74eb800714644bf564b9e5c0`
加纯 Rust patch `6e9488a57ecc3133850e6bd830d0906dbdb1ba071cbadd5371582c0b8aeff377`。
已通读六模块、BPE／WordPiece 公共入口、feed、WordCounts、输出、progress 与
错误清理。CodeGraph 无此 worktree 索引，结论由目标源码回证。没有编译、运行、
测试或修改源码。SmallVec 首片段优化在本轮结束时尚未验证。

**结论：当前结构基本符合深模块原则，没有新的强结构候选。** 该判断限于接口
与职责，不代表完整语义等价证明。

## 已经较深的边界

- 公共调用者只提交配置和词频，取得模型部件或错误；线程池、初始化、重启、
  Arena 生命周期与恢复由引擎完成。feed 和模型替换在成功后才更新原对象，
  特殊 token 注册仍属于拥有 tokenizer 的调用方。
- Vocabulary 管理字符串身份与规范输出；初始 spans 移交后，Corpus 独占
  activation 和 occurrence geometry。Batch 集中选择 reserved singleton、
  兼容 prefix 与 restart，外部调用者不需配对更新状态镜像。
- PairIndex 封装路由、逐 action signed 更新、remove-before-birth、floor、
  historical cohort、过时 priority 修正与列表所有权。当前压缩 birth 移除
  producer Partial／Complete 分类在提交阶段的再解释，是真实边界深化。
- Positions 封装完整 u64 codec、inline、Arena／owned、restart directory、
  解码及 Drop。可信 Input 和 owned 临时片段选择承载真实资源契约，指针
  标签与释放 layout 没有交给调用方维护。

源码位置：`trainers/bpe/mod.rs:355`、`engine/mod.rs:35`、`merge.rs:162`、
`index.rs:238`、`positions.rs:186`，均对应上述固定版本。

## 核对后没有成立的候选

把 prepare→apply→commit 收进 run_round 只移动调用，快照读取、joined endpoint
writes 和提交失败后丢弃 attempt 的机制仍须保留。没有证明新增 facade 会
减少整体协议负担。Fresh／reuse 统一扫描仍需保留 fixed／occurrence span、
greedy AA、whole-word 和历史 cohort 的差异；没有找到可同时退出的状态或
表示。Restart 和全部资源装进 session 会增加 Arena 借用及所有权装配，当前
唯一 coordinator 没有重复恢复代码需要收回。

Writes 统一是已经启动、等待成本验证的候选，不重复列为新发现。已有反证的
owner directory、wave barrier、提前 materialize 也未机械重新提出。
等优先 reuse cohort 次序的既有证明缺口未关闭；结构审计不关闭这个限制。
