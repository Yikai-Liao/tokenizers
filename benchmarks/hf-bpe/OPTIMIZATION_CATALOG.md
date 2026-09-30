# BPE 优化清单与组合关系

## 目标与选型方式

最终目标是在同一原Trainer接口、语料和参数下选择**稳定的最快端到端组合**。主指标是feed到train_vocab返回的elapsed，以及其中完整train；模块计时用于解释组合为什么变快或变慢。已失败版本的前后比较只保留作诊断，胜出组合必须优于当前最快有效组合。

当前候选为C `fbdc0b2b` 与B2 `a0832c48`，三组交错端到端稳定性比较已完成：B2每组train/elapsed/merge更快，选B2。B首版 `c1ee2019` 已回退，不列为竞争基线。C单点39.805秒与B2单点40.265秒曾受不同初始化计时影响，不能据此选型；以此次配对完整运行判断。

兼容优化组合仍须实测：CPU缓存、带宽、分配与任务调度共享，使各项收益不能直接相加。表中“可叠加”表示算法与表示可以共存，“依赖”表示须一起落实协议，“替代”表示同一操作选择不同策略。

## 全部已实现与在推进的优化

| ID | 优化点 | 热点/模块 | 状态与来源 | 组合关系 |
|---|---|---|---|---|
| S1 | 稳定u32位置、token端点编码，合并不移整个词 | corpus/merge | 各endpoint分支已有 | 后续P/I/M优化的共同基础；WordArena词压缩是替代表示 |
| S2 | 打包u64 pair键；16字节候选、8叉堆 | owner/选择 | 已有 | 与语料构造、批次和计数策略可叠加 |
| S3 | 16字节SmallPosting，内联2个u32或4个u16局部位置 | posting分配 | 已有 | 所有endpoint策略共用；posting宽度改变内联容量 |
| S4 | 唯一pair owner与posting所有权 | pair频率/提交 | 已有 | 并行计数、批次与出生安装的共同前提 |
| S5 | 普通字符旧pair单调，低频永久退役；新pair全批聚合后剪枝 | 频率/候选/posting | 已证并实现 | 与有限长度兼容；非空affix须其它证书或generic fallback |
| P1 | 一次创建持久Rayon池，协调器在merge池内 | 调度 | 已有 | 与所有并行内核共用；初始化与merge线程不同才另建池 |
| P2 | 具体规则的精确批次证书；预留ID/AA独立批次 | greedy规则选择 | 已有 | 允许安全并行写；不能跳过canonical tie/身份条件 |
| P3 | 每worker一份route，8字节共享出生链，聚合后直接预留posting | delta/commit | 已有 | M2/M3继续使用；逐小任务独立map已淘汰 |
| P4 | AA摘要与奇偶传播，跨块保持左到右选择 | AA规划 | 已有 | 原Plan与融合组合均共用AA fallback |
| I1 | 空间路由位置→owner并行哈希计数/建堆 | 初始化pair计数 | 原并行版已有 | 与I2–I5可叠加；I6是其计数阶段的替代方案 |
| I2 | 连续词区间规划、并行独占填充最终corpus | 语料构造 | A `98ca7fc1`，继承到C/B2 | I3–I5依赖最终区域与冻结ID；A本身不是另一个可重复叠加的收益 |
| I3 | None alphabet每worker字符存在位图，Some沿用HF频率裁剪 | alphabet | C `fbdc0b2b` | 与I2/I4/I5兼容，ID集合/顺序保持一致 |
| I4 | Unicode→canonical ID直接表，省逐字符UTF-8字符串hash | decode/fill | C，表4.25MiB、构造后释放 | 不要求I3；固定ID后可独立使用，与I5兼容 |
| I5 | MaybeUninit最终数组由worker直接写，完整join后转换 | 最终分配/first-touch | C | 依赖I2覆盖证明；与I3/I4及所有merge策略兼容 |
| M1 | 空间有序Plan下同词复用权重cursor；近邻8项后远跳二分 | delta权重 | 原并行版/C | 与排序Plan局部性匹配；M2改变访问顺序后不能直接期待同样收益 |
| M2 | flat非AA过滤/delta一遍融合，无全局16字节Plan排序 | merge prepare | B首版失败；B2保留融合但修正查询 | 依赖P2证书、共享Atomic端点写与出生顺序；M3/M4使组合有效 |
| M3 | 每256位置一个pivot目录，桶内权重精确查询 | 稀疏weight查询 | B2 `a0832c48`，3.07MiB | 与M2可叠加；可复用到I6；也可适配M1远跳，但该组合未测 |
| M4 | selected head/tail按ID直接表，重复符号小hash回退 | 同批最终邻居查询 | B2 | 与M2耦合；原有序Plan可直接看相邻Plan，无需叠加此表 |
| I6 | 初始pair/位置按owner稳定radix排序，连续分组频率，精确预留posting；低频先过滤再建map | 初始count及posting分配 | D `d15c18cc`，45tests/复核通过；screen train33.265s、RSS4.43GiB，稳定收益待组合确认 | 替代I1的逐位置哈希计数；复用I2–I5和M3，与C或B2 merge理论兼容 |
| M5 | 预留posting批量逆序直填，省push检查与后续reverse | owner commit；组合D时也影响初始posting install | E `35eaf03c`，DE `c8702374` 已实现，测试/构建中 | 与I6正交；共同SmallPosting接口，组合仍须端到端测量 |

AtomicU32是M2共享引用写入的实现条件，本轮不再当作一个独立速度优化点重复计时。u32与AtomicU32槽位大小相同已核对。原Plan使用split_at_mut独占区间；融合使用prepare join→独立Atomic writes join→commit的屏障，两种调度策略互为替代。

## 尚未融合、待实现或只具理论证书的点

| 优化点 | 适用范围/关系 | 状态与优先级 |
|---|---|---|
| owner commit减少临时born map/二次hash，复用route/group结构 | 与I6及C/B2初始化兼容；须保留全worker阈值聚合和posting有序 | commit约6–8秒。下一具体候选：预留posting按出生链逆序直接填最终区间，省逐node push的标签/容量检查及填完再reverse；D安装初始posting也可复用批量填充，E/DE已实现，正在验证。 |
| route/node/有效起点容量复用 | 可与M2/M3/M4共存；保留容量可能抬高常驻RSS | 待profiling确认实际分配热点，不先造框架 |
| M1远跳使用M3目录 | C排序Plan查询策略的局部替换 | 尚未实现/测量；若B2端到端胜出暂不展开旧路径新组合 |
| u16 corpus完整ID域证明 | 与u32 posting地址独立，限制完整可能ID域及separator | 已有原型和测试；当前公平工作集固定u32，不重开宽度矩阵 |
| posting u16局部偏移/多块字典 | 替代flat32目录，可能减少payload但增加字典访问 | 已实现/边界测试，尚无当前组合的大语料收益证据 |
| Halfword：初始u16 code、合并复用旧槽保存完整u32 ID | 替代S1语料表示；并行bitmap RMW/跨度覆盖需另证 | 已有外部串行原型，尚未迁入HF；不能与端点布局字节收益相加 |
| H2.5/H3、u8初始码+短序列映射 | 替代语料表示，收益/访问成本不同；别名跨度有约束 | 原型或推导，未迁入当前并行组合，暂后排 |
| affix可逆/单调证书通过后进入compact；局部输出角色剪枝 | 配置条件优化，可与其它compact项共存；不满足条件保持generic | 证明在AFFIX_PRUNING.md，未实现；当前none工作集不受益 |
| 有限max_length旧边提前过滤、无出生内核、规则级NONE屏障 | 配置条件优化，需保留初始pair例外；generic不能照搬 | 证明在LENGTH_PRUNING.md，未实现；当前无限长度工作集不受益 |
| 有限max_length的u8/u16跨度表 | 长度表示的条件替代，不改变corpus ID宽度 | 理论证书已有；长度表当前远小于posting，优先级低 |
| PR WordArena/整词压缩的短piece几何路由 | 与端点表示互斥的分片策略，不能同时保存两种完整核心 | 旧fused分支已测英文回退，不作为当前组合方向 |
| 动态posting任务抢占 | 替代连续任务划分；须恢复输出序号或出生排序 | 未实现；会改变M2有序生产者证明，先不做 |

[MEMORY_LAYOUT.md](MEMORY_LAYOUT.md)保存空间口径，[AFFIX_PRUNING.md](AFFIX_PRUNING.md)、[LENGTH_PRUNING.md](LENGTH_PRUNING.md)保存条件证书。未实现项不计入当前性能收益。

## 当前组合图

```mermaid
flowchart LR
  Core[endpoint / owner / posting / batch / pool] --> A[I2 并行构造]
  A --> C[I3 位图字母表 + I4 直接ID + I5 直接初始化]
  C --> Old[M1 排序Plan与权重cursor：C]
  C --> Fused[M2 融合 + Atomic共享写]
  Fused --> Lookup[M3 权重目录 + M4 selected直接表：B2]
  C --> Count[I6 radix初始计数：D]
  Lookup --> Count
```

I6分支实现已从B2 `a0832c48` 派生，提交 `d15c18cc`，初始pair key可用两个u16 canonical ID编码且flat位置可用u32时启用；其余配置沿用旧计数。入口仍公开原Trainer接口、u32 corpus和u32 posting，不把内部临时pair code误写成语料ID宽度改变。

新route记录8字节/pair，radix scratch也8字节/pair，较旧4字节位置路由增加临时内存。先按当前203,230,114初始边核算：两份精确record约3.03GiB，另有corpus、输入与metadata；按实际MemAvailable≤1GiB停止。owner路由逐块精确预留，合并旧路由时及时释放原缓冲，sort scratch在posting分配前释放。记录临时峰值及最终posting/owner容量，收益必须包含route/sort/group/安装整段初始化和端到端，不能只挑hash被省掉的部分。

## 推进队列与完成标准

1. **已完成：C/B2三组配对端到端选型**。train中位C42.887/B2 34.942秒，elapsed47.701/38.965秒；B2每组更快，当前选择B2。
2. **同步推进：I6初始pair计数候选D**。父组合保持固定，只换initial count；差分与性能并行处理，测完整初始化（新增route/sort不漏计）和端到端。
3. **随后集中：owner commit**。先确认剩余热点和明确访问成本，选一个具体候选；不平铺实现理论上所有方案。
4. 最后汇总最优有效组合及固定commit。组合兼容关系与收益证据分开记录；端到端稳定收益优先，慢组合保留原因但不推荐。

## 当前具体组合与尚未确认的关系

| 组合 | 组成差异 | 当前证据 |
|---|---|---|
| B2 `a0832c48` | 基础优化+I2–I5+M2–M4 | C/B2三对端到端胜出，train median34.942s |
| D `d15c18cc` | B2把I1替换为I6 | 单次33.265s，内存4.43GiB；待稳定选型 |
| E `35eaf03c` | B2+M5，应用在commit | 源码独立，测试/计时待完成 |
| DE `c8702374` | D+M5，应用在install及commit | 源码组合，测试/计时待完成 |

D与E从共同数据接口看可组合，DE明确应用同一个bulk方法于两个调用点；实际收益可能共享cache/分配效应，不把D筛选的差值与E差值简单相加。选择依据是组合完整端到端及资源约束。
