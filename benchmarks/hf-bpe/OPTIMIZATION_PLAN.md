# BPE 下一轮优化计划

## 基线与约束

基线为原接口u32初始化4/merge4版本 `bpe/parallel-count4`，训练源码固定在 `c07a6e39`，当前仅benchmark清理后的HEAD为 `b6a28768`。固定512MiB真实中文语料上训练54.760秒、merge27.457秒、峰值RSS3.390GiB；PR基准308.111秒/6.569GiB。原子对照训练52.240秒，单次差异不作为选型证据。

继续保持原 `BpeTrainer::do_train/train_vocab/Trainer::train`、公共字段和serde格式。语料与posting统一u32，目标词表50,000、min2、none、串行feed/4线程训练。仅系统MemAvailable≤1GiB停止，进程与系统swap记录；构建、测试、下载与正式计时错开。保存原始计时、RSS、最低余量、换页、源码/二进制/输入hash和模型签名。

每项改动先写本文件，再创建独立worktree/branch并提交。候选实现通过相称正确性检查后才测量；收益、代价和放弃原因同步到 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)。固定源文件与临时探针构建区分记录，历史基线不覆写。

## 候选A：并行语料构造

计划分支 `bpe/corpus-parallel`，worktree `/root/code/tokenizers-worktrees/corpus-parallel`，从非原子初始化4版本派生。

- [x] 拆分alphabet、容量/区间规划、最终数组分配、语料填充阶段计时；13.068秒现有总计不能全部算作填充。
- [x] 固定现有词遍历顺序，分成连续词块；并行统计每块保留字符数和边数。
- [x] 前缀和确定独占区间，一次分配最终corpus，各worker并行解码并写入最终布局，归并词边界/权重。
- [x] 初始lengths活跃标记用局部结果归并，避免多个worker写同一ID槽；保持special预留ID尚未活跃的0长度状态。
- [x] 保留alphabet与canonical ID分配行为、有限limit_alphabet同频边界；本候选先沿用原alphabet实现。
- [x] 覆盖空输入、全裁剪、重复权重、强制alphabet、预留special ID、原子/非原子slot和跨块边界；非空affix仍走原generic路径。
- [x] 通过差分与原接口测试，固定commit，构建再进行一次512MiB关键比较，报告分阶段耗时与RSS。

初始化执行并发由现有私有initialization_workers控制，merge workers保持原值。尽量只保存词引用及每块规划元数据，不构造每worker完整ID语料再复制，不能用新增4N临时数组换取时间。

## 候选B：减少Plan与delta重复走访

在候选A测量完成后记录其局部与整体结果，再选新分叉起点，独立命名worktree/branch。同期复测确认A初始化与整次训练收益；B待C测量后选择构造起点，以相同构造方式的Atomic控制项判断融合收益。

- [x] 逐段核对原型的有效位置检查、邻边delta、出生链和最终邻居判定，列出融合需要保持的协议。
- [ ] 细分delta中的权重查找、哈希/出生记录与提交工作，避免把14.668秒全部归给一个操作。
- [ ] 实现一个可证明的融合候选，优先减少过滤后再次走访和全局16字节Plan存储；保留AA与预留ID规则的特殊路径。
- [ ] 若共享语料融合需要AtomicU32，独立从对应基线分叉；Relaxed访问不增加槽位大小，但算法正确性仍须另证。
- [ ] 测试逐轮greedy顺序、相邻同批合并、有限长度门控、AA、阈值聚合与最终posting位置；复用独立审查。
- [ ] 通过门槛后仅做一次关键大语料比较；若原子/布局/其它改动同时发生，明确收益不能单独归因。

混合规则全局排序目前支持selected邻居识别、权重游标与独占写区；不能单独删除。候选失败或无实用收益时保留branch/commit和原因，胜出版本才进入推荐入口。

候选B协议已写入 [FUSED_BATCH_PROTOCOL.md](FUSED_BATCH_PROTOCOL.md)。为隔离融合收益，另从候选A派生Atomic控制分支，保留原Plan算法；控制项与融合项各一次关键测量，与A合计三项，不展开线程/宽度矩阵。

## 记录与本轮结束条件

- 每个候选有干净worktree和独立commit，公共接口保持一致。
- 原接口、完整模型签名及关键边界验证通过；独立审查结论持久化。
- 汇总训练/merge/初始化加速比、RSS与余量、Atomic内存事实、热点变化、下一步建议和仍未验证的方案。
- 完成本轮A及一个经过源码推导的B候选比较，选择有收益且可维护的实现；不展开完整benchmark矩阵。

## 追加：同期诊断与初始化候选C

用户指出A仅改变初始化却有merge回退，需要先查清。已核对：A与基线从merge主循环开始的源码逐字相同；Trainer公开路由、compact/AA/posting文件相同；实际线程、输入摘要、N/E/pairs、批次数、posting访问数及pruned数全部相同。原因尚未确定，不能把差异直接归因于波动、NUMA或编译器。

- [x] 同环境连续测旧非原子4线程版与A各一次，补记child user/system CPU、wall time、缺页与上下文切换、主机CPU ticks和loadavg。只为解释具体未决回退，不展开矩阵。
- [x] 比较同期merge阶段与CPU/缺页数据，记录能确认的结论及仍未确定的原因；缺少证据时不宣称整体提速。

初始化候选C优先于B，分支 `bpe/corpus-direct` 从A `98ca7fc1` 派生：

- [x] 无limit_alphabet配置按worker统计字符出现位图，再按Unicode顺序分配canonical ID；频次在该配置不参与裁剪，因此无需保存完整频率。limit_alphabet继续原路径，保留同频裁剪行为。
- [x] 字符ID构造一次只读直接查询表，替代每位置UTF-8编码与字符串哈希；过滤计数与填充使用同一表。
- [x] 最终数组使用MaybeUninit分配，独占区域直接写入每个槽；完成join和精确覆盖检查后转成最终Vec。局部封装初始化安全证明，不保留第二份完整语料；每个字符/分隔槽必须初始化，异常路径不读取未初始化值。
- [x] 保留初始化线程控制和阶段统计，补记直接查询表容量；独立审查未初始化内存转换与unicode/特殊ID/裁剪边界。
- [x] 完成正确性与构建后只测一次关键512MiB，报告初始化、整体、RSS及签名。选择C或A的具体parent后，再推进B及其匹配Atomic控制。

## 进展记录

- 候选A已提交 `98ca7fc1`：新私有corpus模块按原词顺序建立连续区域，直接写最终分配，局部ID活跃bitset归并；无额外完整语料副本。alpha/区间测量/最终分配/填充阶段独立计时。
- 40项既有库测试及1项新增构造边界测试通过；release原接口二进制 `hf-bpe-native-corpus-a` 已构建。独立审查和一次512MiB计时待完成。
- runner新增可重复的 `--require-stats`，防止候选阶段字段缺失时静默继续。
- 候选A独立审查通过；512MiB实测tokenize 5.799秒（alpha2.322、measure0.228、allocate0.876、fill2.371）、init16.090秒、merge33.964秒、train55.207秒、RSS3.397GiB、最低可用4.257GiB，VmSwap采样0。旧版tokenize13.068秒、init22.975秒、merge27.457秒、train54.760秒。因此初始化已加快，整次训练单次结果没有加速，保留候选而不宣称整体收益。完整签名一致，来源见 `results/optimization-a-512.*`。
- B的分叉parent为A `98ca7fc1`。新控制分支 `bpe/corpus-atomic` 只切Atomic；融合分支从该控制提交派生，计划名 `bpe/fused-batch-atomic`。读取/写入仍分阶段，flat非AA融合；每worker一份route，任务按posting数均分且连续有序，保持出生posting顺序。

- 同期诊断：baseline/A init23.760/15.421秒、merge29.810/29.756秒、train57.982/49.737秒、RSS3.395/3.417GiB，签名一致。旧6.5秒差异未复现、成因仍未知；记录全部观察而不归因，结果在 `optimization-a-diagnostic.*`。

- 候选C已提交 `fbdc0b2b`，完整42项测试通过，原接口release构建完成。独立复核与一次关键benchmark按用户要求并行进行；计时期间不运行构建、测试或下载，复核只读源码。候选C选择完成后再创建B及匹配Atomic控制的新分支，保留此前A控制分支。

- C一次512MiB完成：alphabet0.347、measure0.187、allocate0.000046、fill0.555、tokenize1.090、init9.767、merge26.092、train39.805秒；RSS3.43GiB、最低可用4.38GiB、VmSwap0。完整模型签名与此前七项一致，独立审查通过。字段与实际资源见 `optimization-c-512.*`。与同期旧版23.760/57.982秒对比初始化2.43倍、训练1.46倍；不同时间的一次测量不隔离环境影响。
- B改用C parent，控制分支 `bpe/corpus-direct-atomic` 提交 `27dc7fd0`，融合分支 `bpe/fused-direct-atomic`。早先A控制/融合起点保留且未计时。新融合实现已完成并通过原42项测试，正在补充Atomic逐轮1500-case差分与相邻批次/跨worker出生顺序测试，再锁定提交复核。

## 用户收窄范围：集中 hot path

用户明确取消重复Atomic对照。此前已测过Atomic与普通slot差别不大，不再为每个初始化候选重复验证。`corpus-direct-atomic/27dc7fd0` 已保存源码与构建，但不计时；`fused-direct-atomic` 直接与已测C比较。只进行融合候选一次关键计时。

改动影响：融合只用于flat非AA原子批次，替换filter、Plan全局排序、delta遍历与rewrite调度；owner提交代码不变，但出生顺序从全局空间顺序变为规则生产者/连续worker顺序，因此必须验证posting有序。初始化alphabet/直接构造、初始route/count、candidate选择、ID分配、AA与多块fallback、非空affix generic路径源码未改。无需重复初始化/Atomic对照。完整43项测试通过，包括把既有1500-case逐轮差分扩到新flat Atomic路径，以及相邻批次、共享头尾、有限长度、跨worker出生顺序的定向案例。后续复核集中于以上变化。
