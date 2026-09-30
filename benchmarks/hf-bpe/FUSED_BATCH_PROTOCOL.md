# 融合批次候选的协议

## 改动目标与适用范围

当前flat普通批次先保存16字节Plan、按位置全局排序，再生成邻边delta与出生记录。候选把有效位置检查与delta生成放到同一遍历，只保留各任务4字节有效起点用于写入。读取阶段全部结束后，AtomicU32执行无序的独立区间写入，然后提交owner。

适用范围是一个u32地址块、普通非AA批次、原子语料；多块与AA保留原路径。canonical ID分配、批次证书、长度限制、预留ID单规则处理、非空affix fallback不变。候选同时改变遍历与写区组织，收益不能称为Atomic load/store本身的收益。

## 最终邻居判定

读取阶段语料保持整批之前的状态。建立 `selected[(a,b)] = replacement` 的只读小表。对有效出现 `a,b`，令起点为p，右邻起点after=`p+len(a)+len(b)`。

- 左邻L的起点before=`p-len(L)`；读取before之前的token P。如果 `(P,L)` 在selected中，左邻已被同批规则选中，当前出现跳过左侧边界delta，由左边合并负责这条边界。否则删除旧 `(L,a)`，出生 `(L,replacement)`，起点before。
- 右邻R之后的token为S。如果 `(R,S)` 在selected中，最终右邻是相应replacement；否则仍为R。删除旧 `(b,R)`，出生 `(replacement,final_R)`，起点p。
- 分隔符不产生跨词边。出生邻边仍使用严格 `< max_token_length` 门控；选中初始pair的原有例外保留。

批次排除AA和头尾冲突，因此selected邻居的规则有效且不会与当前合并重叠。每个活pair的位置posting完整；同一pair的长度门控在plain路径为常量。必须以测试和静态审查确认以上前提，不能仅凭key匹配宣称正确。

## 写入与屏障

prepare阶段只读corpus，同时生成delta、出生链和有效起点；同步join/collect完成后才apply。apply仅写被选择合并自己的端点，非AA批次区间互不重叠；Atomic表示Rust共享引用下的写能力，使用Relaxed load/store，不含每槽锁。apply完成后才commit owner，新批次才读取语料。

AA继续使用原跨块奇偶协议与有序计划，预留canonical输出ID继续独立批次。shared写入接口仅对Atomic slot开放，普通slot保留独占切片写入。

## 无全局排序时出生posting如何保持有序

将输入看作“规则rank顺序、每条规则posting位置顺序”的连续流。按总posting数量均分为连续worker区间，每个worker持有一份route，输出按区间顺序归并。

每个出生pair至少包含一个本批新激活的replacement。批次内replacement各不相同；若canonical字符串已存在，选择器把该规则留到单规则批次。只含一个新ID的出生pair由对应唯一规则生成；同时含两个新ID的边界，由左侧规则生成，右侧规则根据selected表跳过左边。因而每个出生key只有一条规则产生，其出现按该规则posting顺序访问。出生左边界的before也随p严格递增；所以各worker的局部链reverse并按worker顺序追加后，posting全局仍递增。

这是省掉排序后的必要不变量。实现中应检查出生posting有序，覆盖相邻合并、重复头/尾、预留字符串以及跨worker切分的情况。若任务改成动态抢占，不能沿用上述输出顺序证明，须增加任务序恢复或出生排序。

## 验证与测量

逐轮HF/独立greedy差分、相邻同批规则、有限长度限制、AA、预留ID、跨worker阈值与最终posting有序必须通过。独立审查锁定源码commit/hash。原计划的新增Atomic控制计时已按用户要求取消；已有Atomic结果只说明历史测试差别不大，不是本轮融合的严格隔离对照。融合候选直接与已测C比较，只做一次固定512MiB关键测量，记录训练/merge/阶段时间、RSS、余量、swap和完整签名。
