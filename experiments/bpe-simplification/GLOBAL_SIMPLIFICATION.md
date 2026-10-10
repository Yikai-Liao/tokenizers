# BPE 引擎整体简化：2026-10-10

在 `simplify/bpe-maintenance-20261009` 的 `727fa3e6` 基线上，完成整个 BPE
训练引擎的简化审计，采用一项状态归属调整。实现提交为 `177302e2`，工作分支为
`simplify/bpe-global-20261010`。本轮覆盖六个 engine 模块及其端到端连接，公共
trainer 和 reference 仅用于核对契约。

## 已采用的完整切片

原来选择一个 merge 时，Vocabulary 更新 `active`，返回含激活布尔值的
`MergeIdentity`；Corpus 再据此更新扫描域和跨度。这要求字符串身份与语料几何
分别维护同一个“ID 曾经激活”事实。

现在 Vocabulary 构造初始跨度表，CorpusPlan 一次性取得所有权；此后 Corpus
统一判断激活、准备几何。Vocabulary 只解析字符串并返回 ID。初始化表移交后
不会随着 merge 再增长，运行期 `active` 镜像、`MergeIdentity` 和跨模块布尔
传递一起退出。初始布尔表转成跨度表的分配和映射也消失了。

保留的边界有具体用途：ID 表非零表示历史激活，最后一次 occurrence 被消费也
不清除；没有 alias 分歧时，它同时提供固定跨度。出现不同跨度的 alias 后，
真实几何由 occurrence plane 管理，后续 ID 元数据只需激活标记。这避免继续
累加已经不能代表真实几何的 ID 跨度。plane 建立后才首次激活的 reserved ID
也得到标记。初始 ID 顺序、过滤、UTF-8 装饰、重启、reserved singleton 和
严格长度 gate 保持原路径。

格式化后，生产实现 **2185 → 2176** 行，默认测试及 helper **797 → 799** 行，
分别符合 2200／800 预算。新增两行测试为既有长 alias 对照加入 reserved token
并在后续 alphabet 判定前清除它；没有删除测试或搬移生产代码。

## 整体审计与取舍

| 工作流与模块 | 已检查的机制及决定 |
| --- | --- |
| 初始化／输出：vocabulary、mod | 初始身份与运行期激活可以分开交接；采用上述调整。保留字符串单份存储、原遍历顺序和重启时 retained alphabet。 |
| 初始计数／驻留：corpus、index、positions | 借用计划、整词分块、生产者内压缩、最终 fragments 聚合共同控制驻留。已有在线压缩证据，不改成提前 materialize 或全宽 raw 列表。 |
| 选择／准备／应用：merge、corpus | fresh 固定几何与 reuse occurrence 几何、AA 贪心起点和兼容 prefix batching 承担不同语义，保留。select 与 prepare 共用目录是较弱候选；需要额外生命周期和 lazy 初始化，未证明净收益，未实施。 |
| 计数／birth 发布：index、merge | 保留 signed action 次序、历史 cohorts、complete producer 提前发布及 partial 聚合。紧凑路由的双数组保证 payload 单次移动和错误时 drain 清理。 |
| 存储／回收：positions、index | 保留可信 Input、全 u64 codec、两字描述符、Arena／owned 分配及 scoped lease。没有改变 unsafe 或其证明义务。 |

另探索了“全局 Candidate 堆＋owner count map”，它可以减少普通队列与 reuse
队列的并行表示，但会失去 fresh count 低于 floor 时的大 owned Positions
立即释放：删除 count 后，旧 payload 可能留到堆顶或 attempt 结束。还会把
普通 owner 内的并行 birth 发布改成汇总后串行 push，并增加 heap sift 的记录
大小。额外的 handle、清扫或撤销索引会重新引入协调，因此这次状态归属调整
没有一并采用它。随后按用户要求在 `simplify/bpe-heap-20261010` 单独实现和
测量；对照是本次实现 `177302e2`，结果见该分支的 `HEAP_EXPERIMENT.md`。
历史已排除的 owner 目录和 wave 屏障没有重复加入。

## 验证与复核

最终默认／无默认特性检查各通过 **17 native tests＋1 doctest**；all-target
Clippy `-D warnings`、fmt、预算和 `git diff --check` 通过。完整 model／每步
trace 的参考对照覆盖 1／4／8 worker、affix、alias、AA、零权重、length gate、
重启、宽 ID 和数值边界。

临时探针在 `prepare_identity` 入口检查 plane 已存在、ID 尚未激活且属于
既有 reserved 范围；在 1／4／8 worker 各观察到 reserved ID 0、pair `(19,20)`。
探针已撤掉，最终测试和二进制使用无插桩源码。首次 probe 的失败来自 fixture
忘记清除 special token，影响后续字母表 literal 判定；修正前日志也保留。
Miri 未重复运行，positions／unsafe 实现没有变化；本轮 native suite 仍执行
原 codec 测试。

另一独立上下文复核完整引擎、补丁、相关测试及验证日志，未发现 actionable
finding；没有已证明值得继续实施的全局遗漏。它仅静态取证和读取作者结果，
没有独立执行目标代码。记录见
[independent-review.md](evidence/global-20261010/independent-review.md)。

验证命令、源码／补丁哈希与日志入口见
[validation.json](evidence/global-20261010/validation.json)。运行成本检查单独记录，
不把代码减少解释为训练提速。

## 运行成本检查

对同一份 256 MiB 英文／中文语料的 ByteLevel 与 Whitespace 冻结词频分别执行
一组预热、三组交替顺序的成对测量。4 workers，CPU affinity 0–3，vocab 50k，
min frequency 2，无 affix。计时包含 public `do_train`，排除输入加载和模型
序列化；进程 HWM 包含输入加载。32 次完整模型全部一致，child swap 均为 0。

| 输入 | 训练 wall 成对中位变化 | CPU 变化 | 进程 HWM 变化 |
| --- | ---: | ---: | ---: |
| 英文 ByteLevel | -10.35% | -6.95% | +1.44% |
| 英文 Whitespace | -6.44% | -0.87% | -0.77% |
| 中文 ByteLevel | +0.62% | +1.83% | -1.53% |
| 中文 Whitespace | +0.46% | +1.63% | -0.39% |

这是共享 VM 上三组成对样本的观察，英文 wall 变化幅度也不稳定；不据此声称
稳定提速或统计等价。输入、二进制和源码哈希见
[performance-manifest.json](evidence/global-20261010/performance-manifest.json)，
逐次测量与模型判定见 [performance-runs.json](evidence/global-20261010/performance-runs.json)。
