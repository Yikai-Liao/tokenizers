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

在候选A测量完成后记录其局部与整体结果，再选新分叉起点，独立命名worktree/branch。同期复测确认A初始化与整次训练收益；B待C测量后选择构造起点，直接与已测C比较融合收益，按用户要求取消新增Atomic对照计时。

- [x] 逐段核对原型的有效位置检查、邻边delta、出生链和最终邻居判定，列出融合需要保持的协议。
- [ ] 细分delta中的权重查找、哈希/出生记录与提交工作，避免把14.668秒全部归给一个操作。
- [ ] 实现一个可证明的融合候选，优先减少过滤后再次走访和全局16字节Plan存储；保留AA与预留ID规则的特殊路径。
- [ ] 若共享语料融合需要AtomicU32，独立从对应基线分叉；Relaxed访问不增加槽位大小，但算法正确性仍须另证。
- [ ] 测试逐轮greedy顺序、相邻同批合并、有限长度门控、AA、阈值聚合与最终posting位置；复用独立审查。
- [ ] 通过门槛后仅做一次关键大语料比较；若原子/布局/其它改动同时发生，明确收益不能单独归因。

混合规则全局排序目前支持selected邻居识别、权重游标与独占写区；不能单独删除。候选失败或无实用收益时保留branch/commit和原因，胜出版本才进入推荐入口。

候选B协议已写入 [FUSED_BATCH_PROTOCOL.md](FUSED_BATCH_PROTOCOL.md)。为隔离融合收益，早先从A派生Atomic控制分支作为保存的候选起点；用户已取消其计时。融合候选改用C parent且仅计时一次，不展开线程/宽度矩阵。

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
- [x] 完成正确性与构建后只测一次关键512MiB，报告初始化、整体、RSS及签名。选择C或A的具体parent后，再推进B；匹配Atomic控制计时已取消。

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

## 热点候选 B2：按规则访问的查询成本

B `c1ee2019` 正确性通过，但512MiB实测prepare22.858、merge31.887、train49.697秒，较C更慢。Plan排序确实省掉，新增成本在prepare；C的plan+delta为17.765秒。初始化代码不变，此次pair count从8.015变11.370秒也不能归因merge改动。保留B失败结果，不推广。

只围绕prepare继续一个候选，分支 `bpe/fused-lookup` 从B派生：

- 对B做32MiB `perf` 热点采样用于定位，不作为性能对比或新增矩阵；若函数内联，只报告能定位的范围。
- 当前按rule排序访问让weight游标频繁跳过大量词，原全局空间排序的近邻性丢失。新增稀疏桶目录：每256个corpus位置一个u32 pivot下界，桶内二分恢复精确词索引；额外约3.1MiB，避免大范围二分。仅flat融合路径消费，uniform直接返回。
- selected邻居以ID域直接表表示唯一head/tail映射；重复head/tail通过原pair hash表回退。避免每个边界都做hash，并可在ID未选择时跳过额外corpus读取。
- 不改初始化算法、候选选择、出生协议或owner commit；测试沿用B差分，仅为新查询等价性加定向检查。候选通过后只做一次512MiB关键比较，不重复Atomic控制。

- B独立审查通过；热点采样与限制已保存 `results/optimization-b-hot-profile.md`。B2已提交 `a0832c48`，44项测试通过，release构建中；复核只检查新查询，复用已完成的融合协议证明。

## 执行顺序修正

用户进一步明确要求性能测试与正确性检查并行。候选二进制构建完成即可启动关键测量，同时进行新改动的差分/边界检查和证明复核；不等待全部正确性检查结束。正确性失败则作废该候选性能结果。保存计时期间并发活动，CPU密集测试可能影响性能数值；静态复核也可同步完成。当前B2在GO前44项测试已结束，计时与查询静态复核并行；没有额外CPU测试负载。

## 查询模块公平性与稳定性补检

用户要求“改哪个模块，评估哪个模块”，并指出单次运行可能方差很高。此前B2 vs C merge及全训只描述整版观察，不作为查询模块的隔离证据。本次查询模块公平控制为B `c1ee2019` 与B2 `a0832c48`：相同C初始化、AtomicU32、filter/delta融合、写入协议、owner commit，只改变权重/selected查询和一次目录建立。

追加三组交错关键比较，顺序B→B2、B2→B、B→B2，固定同一512MiB和全部配置。只用prepare中位数、范围、sample CV及逐组差值评估query改动；B2目录建立时间/内存另计。初始化与whole-train作为观察数据保存，不解释为查询影响。每项记录并发活动、CPU/fault/RSS/换页与来源，签名一致方能纳入。三个样本用于检查明显不稳定，不声称刻画尾延迟或建立可靠总体分布。完成后若模块收益各组都明确，不继续增加运行或矩阵。

## 最终目标修正：最佳端到端组合

用户指出B是失败方案，B/B2的改善无法证明最佳组合；已完成记录仅作诊断。正在进行C/B2三组交错端到端比较，主指标完整train与feed+train elapsed，模块等价prepare用于解释。用户要求不干等，已整理 [OPTIMIZATION_CATALOG.md](OPTIMIZATION_CATALOG.md)，列出已合入、未组合、正交/依赖/替代关系和下一队列。

下一初始计数候选D从B2 `a0832c48` 分叉 `bpe/initial-radix`。当前owner初始化逐边hash并增长posting，约8–12秒。计划在flat且初始canonical ID可编码为两个u16时，用8字节(pair code, u32 position)记录稳定radix分组，先精确计数/剪枝，再一次预留最终posting和owner表；记录增加的route/sort成本与临时内存，不单独挑计数子阶段。其它配置fallback旧count，公开语料/地址仍u32。与构造和merge算法可叠加；构建与计时分别安排，正确性复核与计时并行。代码推进与C/B2测量同步，本段计时期间仅编辑源码，不编译。

## 候选E：posting批量填充，与D可组合

D计时期间同步推进E，分支 `bpe/posting-bulk` 从已选B2派生。当前flat owner commit逐node调用SmallPosting::push，再反转刚写区间；已精确知道每个group的occurrences和最终reserved capacity。新增私有bulk接口，先一次验证len/capacity，将链从head读取并倒序直接填入最终区间，全部初始化后才发布新len。scope仅已预留posting，保留原push供普通增量/初始哈希计数。

E影响owner commit，不改变规则选择、频率聚合、出生准备、语料写入或初始化计数。与D兼容；若D胜出，在D安装初始posting时也用同一bulk接口从record尾部读，省逐位置push，组合DE影响posting安装及commit两个模块。heap/inline边界、追加、部分panic后len与allocation有效性必须验证。只做有希望组合的关键测量，最终与当前最快端到端组合配对，不展开2^N组合枚举。

- D `d15c18cc` 完整45项测试通过，独立复核通过，原接口512MiB screening train33.265/init9.617/merge19.267秒。新增route2.563秒（compact1.327），sort2.045/group2.683/install1.037秒，初始count整段5.764秒；这些成本已包含在完整初始化，非只挑hash子项。RSS4.43GiB、最低可用3.39GiB、VmSwap0，完整模型签名一致。最终posting802,967,868B、owner表138,412,096B，corpus849,691,660B不变；count3仍heap最小4槽。单次约5%全训改善仅作候选筛选，尚不替换稳定B2。
- E已独立提交 `35eaf03c`（B2+bulk commit），DE组合提交 `c8702374`（D+同接口bulk commit/install），分别worktree `posting-bulk`/`radix-posting-bulk`。构建和测试中，独立review只检查bulk内存发布、panic、反向顺序及两个调用；复用已有D/B2协议。固定当前组合B2，比较E与DE的完整工作后，只给有希望的组合做稳定端到端确认。

## 候选F：单次出生直接保存在group

E/DE分别通过45/46项库测试，独立bulk审查通过。512MiB单次筛选E train37.082/init13.173/merge19.842/commit6.687秒；DE train33.053/init8.589/merge19.984/commit6.385秒。E没有显示commit收益，DE与D的全训差距小且sort/compact也波动，不把差异全部归因bulk。正在以三个循环块B2/D/DE交错复测选择真实组合。

计时同期仅编辑下一候选源码，F从D `d15c18cc`派生 `bpe/singleton-birth`，不预先加入尚未确认有收益的E。flat route每个key只有一次出生时，将position直接保存在既有Group.head，occurrences=1标识，不分配Node。第二次出生才把第一次及当前position物化为两节点，后续沿用链；occurrences=0仍表示remove。这不改变Group大小、频率、global floor聚合、posting顺序、owner路由或初始化。commit读取singleton直接push，多次组沿用旧链及reverse。AA也覆盖，非flat仍旧路径。

影响模块为delta/融合prepare中的birth以及owner commit读取出生记录；full init/radix/source布局不改。预期减少局部单例Node分配与读取，是否足够占比须实测。新增统计记录累计局部singleton数量、nodes实际长度/容量峰值，不把它解释成全局唯一pair数。测试覆盖remove→singleton→两次/多次、position/sentinel、跨worker floor；已有1500-case逐轮差分继续执行。review和关键性能并行；正式计时开始前结束CPU构建/测试。若D/DE选型相近，不将兼容性当成收益相加。

用户进一步指出E没有实用收益。E不进入推荐组合，不再新增E/DE计时；已经开始的三组合复测完成留档，用于D/B2选型和公平模块分析。E影响commit，DE额外影响install：其余阶段波动不记作E的收益。F明确以D为parent。F源码完成于 `4a2f148a`，直接头/延迟建链，独立审查与九次比较并行；测试/构建待计时结束后执行。

## 数据选型与自动向量化方向

用户进一步纠正：应相信完整数据推进DE。九次完成后DE train中位31.332、elapsed35.493秒，每组三次均胜B2，中位配对train比0.8715/elapsed0.8840；当前最快已测组合为DE。D与DE之间会换号不足以证明E没有用，归因不明确也不能据此丢掉整版数据。此前排除E/DE的决定撤销；完整结果和sample CV见radix-combination-stability.summary。

F已完成46项测试、release与并行审查，现在按D→F各一次筛选，衡量prepare/delta及commit；不因继续设计而搁置已完成候选。同步在DE上移植同一singleton表示改动为DEF（新worktree singleton-birth-bulk），保留bulk append调用，多次出生消费不变；先保存源码，D/F计时结束后再构建。之后有收益才做针对当前最好组合的稳定比较。

用户提出自动SIMD，不希望显式SIMD代码。当前先以同release opt-level3/default target生成LLVM vectorizer remarks、IR和assembly，定位实际热点的未向量化原因，不把查到几条SIMD指令当作整循环成功。普通Rust重构方向优先考察prepare的只读阶段能否安全使用普通slice（当前Atomic Relaxed仍产生atomic LLVM load），以及把固定块的纯映射/比较与hash插入、Vec增长、间接累计分开。初始化radix histogram/scatter有真实同桶写依赖，不能假设简单改成iterator就会SIMD。仅CPU诊断编译与正式计时错开；计时期间只读/编辑；公开接口、target flags和算法语义保持公平对照，不写intrinsics。

F screen D/F完整签名一致。D prepare11.472/commit6.362/merge19.276/init9.479/train33.125秒；F12.062/6.543/20.135/7.907/32.468秒。直接修改的prepare与commit都回退，未改init变快不能归因F。停止F推进；DE+F源码 `7504cbfd`保留但不构建/计时。F的17,124,772局部singletons不是全局pair计数，node容量峰46,137,344B，不影响初始化RSS主峰。

自动向量化候选G1（branch bpe/read-phase, parent DEc8702374）：只为融合prepare的读阶段通过 &mut [C] 建立普通 &[u16/u32]，Slot私有associated Read类型和read_phase方法；Atomic实现使用当前Rust1.98安全标准库get_mut_slice，不写unsafe cast。pool.prepare join完成后普通view已释放，后续apply仍Atomic共享写/同屏障。初始化、AA fallback、delta/出生/commit算法不改；单独评估prepare、其余模块只描述实际观测。此候选要求Rust>=1.98；当前工具链满足，仓库没有显式更低MSRV声明。

G2（branch bpe/prepare-blocks, parent G1）只重组融合prepare过滤：每worker复用固定128位置的端点/valid数组，逐块收集端点（保留left失败时跳过right读取），用普通slice zip循环计算valid mask，再按原posting顺序查询权重/selected并更新输出。纯比较循环不含atomic/hash/Vec.push/错误退出，可由LLVM成本模型自动向量化；新增短暂固定栈缓冲与第二遍mask消费必须全部计入prepare，是否更快由实测决定。不写intrinsics、不强制vector-width，不改变target-cpu。完整差分与静态阶段借用/顺序审查并行处理；构建完立即安排DE/G1/G2各一次筛选，在计时中只做静态检查。

用户再次收窄：真正SIMD小kernel不是优先级。已取消拟议mask标量/向量微基准（未实现、未运行），停止G1/G2推进。三对G2/DE prepare中位比1.0195、elapsed1.0018、方向混合，未有稳定收益。选择DE为当前基点。正在做一次DE32MiB perf定位，绝不从出现几条SIMD指令猜整个12秒热块的收益。

候选H仅在profiling支持后实施：WeightLookup新增每256位置桶的保守'全部weight=1'位图，额外约100KiB。默认所有桶为one，再按previous_weight和每个pivot interval将weight!=1覆盖的桶清除；混合桶沿用原目录精确搜索。有效位置weight1的桶直接return1，省pivot/weights随机访问。这是同一权重查询模块，两个读者为初始radix group和融合prepare；build时间/bytes、桶覆盖计入，不改变频率、语料/selected、批次/出生/commit。源码与证书包括empty、zero、gapped、重复pivot、256边界、末桶与超大weight；新增stats one_buckets/total_buckets只是空间覆盖，不误称实际query命中率。若DE profile显示它不是大头，不机械实现此候选。


## H已实现及筛选；稳定性确认进行中

DE32MiB实际perf中WeightLookup占8.16% top-IP samples（1776 samples，lost0），支持实施H。源码00216d91从DE派生，新增bitmap认证与3项覆盖/容量统计；全47tests及 [独立审查](WEIGHT_ONE_BUCKET_REVIEW.md) 通过。无显式SIMD、无target flags变化。

同期DE→H一次512MiB：group2.640→0.328、prepare11.320→8.548、完整train32.860→26.538、elapsed36.785→30.574秒；H bitmap100,632B（约98KiB），797,827/805,039桶认证为1。增加少量内存，语料/posting载荷保持相同，RSS均约4.43GiB。空间桶比例不等同实际查询命中率；sort/install源码未改但墙钟也变动，不把所有变化归因H。见 [筛选报告](results/optimization-weight-screen.summary.md)。

筛选显示两个直接模块及完整训练同时改善，继续DE→H、H→DE、DE→H三个交错pair；screen不混入稳定性统计。要求完整签名、N/E/pairs与语料/posting容量相同，报告直接模块及train/elapsed配对比、中位、范围、sampleCV。候选选型保持原API及default AHash种子；不开展新矩阵或仅比较小kernel。测试和构建已经结束，正式计时期间仅编辑文档。


## 用户停止额外复测；下一候选I：直接组装新posting后一次安装

用户认为H直接模块及完整训练的改善足够明显，要求停止多组重复测试、推进后续优化。已通知计时agent不再启动下一call；当时p2.de已启动，完成该call后停止，共保留两个完整pair（n=2），不宣称执行n=3。H选作新基点。

I从H 00216d91派生，只改变flat owner commit。原路径先在临时born map聚合，再插入owner ledger，再重走所有worker route、按key查ledger并逐group追加posting。候选在born map聚合时同时建立紧凑source-group描述链（output index/head/next，16B），按output顺序prepend，因此链按来源逆序；最终accepted key一次精确分配，并由group逆序+各group自身逆序的node链直接bulk反向填完整posting，最后一次插入ledger/heap。减少每个local birth group的owner hash查询，bulk检查从逐group降为逐accepted key；floor仍全来源聚合后判断，低频不分配posting。

每key聚合仍16B（复用Group），新增临时descriptor Vec的容量必须记录；无unsafe、没有修改producer或频率/出生顺序。泛型非flat保持旧路径。保留连续来源及唯一规则producer的顺序证书，新增跨output、空output、同组多位置和全来源floor测试；完整逐轮差分/独立审查完成后，只做H→I各一次关键512MiB，直接commit及整体同时决定是否保留，不展开稳定性矩阵。


## I筛选回退；候选J按规则邻居ID缓存聚合

I全48tests与独立审查通过，H→I一次screen完整模型gate通过。直接commit5.971→6.876秒（+15.2%），train26.187→28.538秒，新增descriptor容量峰2,097,152B；未改prepare也波动，不据此推导唯一回退原因。按用户少重复原则停止I，保留H作为基点。

J从H派生，只改变flat非AA融合prepare的局部输出累计。每条规则固定左/右侧token和replacement；worker用两张neighbor ID→局部group index的u32直接目录，将remove weight和birth weight/head/count暂存于紧凑group，born位置仍写入既有每owner Node数组。每task结束后按触及的group一次flush到原route hash表，拼接新旧逆链，不逐位置hash。floor与跨worker聚合仍由原commit完成，不进行提前低频裁剪，零weight出生仍保留位置。

目录只在lengths.len<=65,536时启用（该阈值是scratch容量限制，不是语料ID/地址宽度），每worker两目录上界512KiB；更大域沿用原prepare。每task只重置触及的索引，不清整张表；目录第一次初始化、group容量、flush及每owner routing成本全部计入prepare。常量generic分派隔离fallback循环，不添加数据域宽度假设。新增统计peak_prepare_aggregate_bytes为各job scratch capacity总和再取跨batch最大；无需unsafe。顺序证书、左右cache键的身份、selected邻居与Node尾链拼接由独立审查和逐轮差分检验。全测试/构建完成后只H→J各一次512MiB关键对照，若明显改善按数据保留，不重复多组。


## 用户调整范式：先详查H成本，再推进实现

用户要求从具体性能分析得出优化，避免只见大函数占比后猜方案；授权自主迭代至没有明显优化空间时停止。当前I已screen回退，停止。J只保存原型源码，初次编译的right名称遮蔽已修正，尚未完成测试/构建/计时。暂停J筛选，不把它算作已验证实现。

先对当前最快H的实际512MiB工作集做一次cycles/cache-misses双事件、call graph与top-IP/实际binary disassembly映射的诊断。硬件事件可用性用perf stat true已确认，J构建已结束，采样期间无CPU重任务。分开记录CPU周期样本与cache-miss样本的损失/未解析/归因限制；不要将宽泛cache事件或有skid的指令点当作精确load归因。查询hash、节点遍历、corpus/length随机读取、分配/resize应有实际证据与成本占比；必要时补针对性操作计数，不能把静态源码语句数量当动态占比。

只有分析支持的大项才继续J或其它候选；新增缓存/分组/数组清零成本要一起计入。保留一次关键对照和完整语义gate，明显结果不重复多组。最终停在最快有效组合及明确剩余成本，而不宣称全局最优或无证据的翻倍空间。


## 诊断前补齐实际debug信息

用户指出perf输出Rust编码符号并询问debug信息。readelf核对H正式binary97d890...：只有.symtab/.strtab，无.debug_info/.debug_line。编码名称可由c++filt Rust demangle还原为Output<u32,2>::birth，不能据此证明行号信息存在；perf --call-graph dwarf只采集用户栈，也不会补源码DWARF。已有15775样本H512诊断仅作为函数/符号偏移证据，暂停以它推断精确源码操作成本。

单独构建相同H probe源码的release opt3、debug2、strip none到.build/h-debug-target；正式baseline binary不覆盖。先readelf验证debug段、addr2line验证实际新binary地址/inline链，再对该诊断版本执行一次H512周期/cache采样。记录新hash/Build ID和flags，不把旧地址映射到新binary，也不把诊断耗时参与排名。

## 候选K：并行释放owner账本

独立H cost/count诊断定位到合并后的4.780秒缺口：post-merge实测4.777秒，其中owner销毁4.752秒，占99.5%；词表/merge字符串转换合计21.3毫秒。销毁前尚有10,572,128条ledger entry与14,225,344个heap item。计数另显示173,155,595次birth和同量remove，理论按output去重后约57.84M组，但该下限不是J实际flush次数。

先从H00216d91派生`bpe/owner-parallel-drop`，仅在训练结束时用当前Rayon池并行消费和销毁各owner。可省的是串行析构路径的等待；所有分配仍须正常释放，实际allocator竞争和worker分配决定是否更快。无新增索引、清零或堆内存；调度一次join，owner之间的条目和posting所有权独立。最后一次commit及所有读者已join，结果只依赖ids/strings/merges，不借用owner存储；公共API和阶段统计边界不变。

预算为一次H→K512MiB关键对照，核对完整模型与资源gate。直接指标用`train-initialize-merge`区间，诊断已确定该区间主要为owner释放；同时比较train和elapsed。若直接区间与全训明显改善则选K；若无收益则停止该简单方案，不扩大线程或allocator矩阵。J暂不构建/测量，待K决策后再用已取得的hash成本与重复计数判断。

K288ad858原47tests通过，native release完成。首对H→K清理区间3.835→2.727秒（-28.9%），但未改merge15.439→17.232秒，完整train26.063→26.862秒（+3.1%）。直接改善与整体回退相冲突，预算仅追加一对反向K→H，区分当前完整收益；若仍混合则保留H并停止K，不展开矩阵。首对结果独立保存，不覆盖。

反向pair完成；两对K清理均缩短（-28.9%/-35.1%），n2平均少1.280秒，但完整train平均+1.2%、elapsed+1.1%，方向未支持完整收益。停止K，保留源码和两对记录，不纳入推荐，也不改变J的H parent。

## J恢复验证：重复route查询已有实际依据

H调试采样确认birth/remove中实际inline桶探测，独立计数得到逐位置调用各173.16M、output去重delta组57.84M。前者支持重复工作存在，后者只给理论下限；J的Task×侧×neighbor聚合及flush可能多于该下限。J376363d2独立静态审查通过：左右旧/新邻居身份、unique producer/跨task逆链、零weight、有限长度、目录上界fallback和错误路径安全。新增8×ID域的目录初始化、LocalGroup容量、flush/reset与原Node追加都计入prepare。

恢复原逐轮差分和定向测试、native release构建；通过后只H→J一次512MiB关键比较。直接prepare与完整train/elapsed决定去留；若直接模块无改善或完整结果混合且仅小差异，则放弃J，不继续低价值复测。来源继续H00216d91，不叠加未获完整收益的K。

J全48tests/独立审查通过，一次H→J完整签名与资源gate通过：prepare8.400→7.723秒，train27.790→24.613秒，elapsed31.762→28.701秒；新增scratch capacity汇总峰2.01MiB。直接及整体同向，当前选择J376363d2。初始化源码未改却少2.011秒，不把全训差額都归因J；n=1不作稳定性保证，不追加复测。

## 本轮停止

最后只复用已有H debug的15,897条完整事件IP，对owner commit实际符号的热点做有界批量行号与反汇编核验，无新训练。主要操作是ledger control-byte探测、entry frequency载荷读取/更新、posting长度读取与Node链读取/逆序填充；具体IP不等于精确load或算术成本，cache事件与skid限制保留。临时born聚合未独立定量，不从宽闭包份额外推。

剩余大项的简单方案I/K已筛掉；J减少重复route查询后，此固定u32/原接口工作负载内尚未找到代价清楚、能大幅减少剩余工作的局部方案。停止本轮，保留J为当前选择与H已验证基点，所有失败源码/原始结果可复查。语料表示、allocator或owner数据结构的大改均尚未验证，不能作为已兑现收益。停止不意味着全局最优。

临时skill performance-optimization-draft已创建并验证，指导成本证据、动态计数、完整收益、预算和自主停止；已把实际debug段与地址解析的预检加入。独立轻量行为审查支持按有价值的定位→最小关键对照推进，并建议每轮明确预算和决策条件，已纳入。

## 追加报告：J单线程/四线程加速比

用户要求汇报当前J各阶段与端到端相对单线程的四线程加速比。现有J只有四线程screen，历史单线程属于较早版本，不能拼成J线程对照。预算为同一J376363d2、同一release binary、同一512MiB/none/50k/min2/u32输入，初始化和merge分别1/1与4/4各一次，feed均串行；不测试2线程或重复矩阵。

native benchmark runner新增仅测量用HF_BPE_BENCH_WORKERS配置，默认4，允许1/4；run_native_fair记录该变量并以实际workers/initialization_workers统计核验，生产J源码不变。新label/build-root保留原Jscreen binary及源码副本。完整模型、工作集和载荷门槛保持；内存政策沿用MemAvailable≤1GiB停止。阶段加速比为T1/T4，嵌套计时不相加，零/极短区间不推导性能机制。

线程对照已完成，完整模型/工作量相同，初始化3.43×、merge3.45×、完整train2.99×、含串行feed2.70×；各一次，不追加重复。

## 追加：收尾对象数量与原Rust粗略速度

用户追问结果构造/清理是否由小对象引起。预算一次独立J诊断，在merge后扫描owner对象分布，不改循环或模型，额外扫描191ms单列。实际7.275M堆posting、3.297M inline posting；小容量≤8个u32占堆分配61.13%。结合H精确drop区间说明清理路径，同时保留map扫描与allocator成本未分离的限制。

用户要求与原efficient_bpe Rust只粗略比速度，不要求模型一致。预算同16MiB完整行前缀、逐行去重/权重、4线程各一次；HF50k vocab对应原Rust40,759 rules。训练API J1.478s/Rust2.155s；输入Prepared和字符前端、编译profile及输出范围差异保留。无额外benchmark矩阵。

用户追问feed是否并行：核对生产maybe_par_bridge/map/reduce路径，说明benchmark显式关闭feed并行、train前开启；本轮扩展结果仅代表此配置。

## 用户要求：实际测量全生命周期Bump保留的内存与总体影响

用户要求先测全部中途不释放、最后统一释放时的峰值与总体影响。预算为独立诊断副本J、实际bumpalo3.20.3替换posting backing分配、保留所有退休buffer，每个专用训练worker一个TLS arena；所有posting owner销毁后broadcast释放整块arena。pool在全部指针访问期间存活；不共享Bump，不将arena指针交给std Vec释放。实验限定u32/AtomicU32/flat32/init4/merge4，生产源码不变。

每次heap分配仅在线程局部累计请求capacity字节/次数/growth；末尾读取arena backing/chunk数与live posting inventory。初始与birth分配全部保留到结束，不假设小对象阈值。额外统计开销明确作为诊断。验证一份16MiB模型/工作量smoke；通过后512MiB同期标准分配J→Bump各一次，沿用模型、N/E/pair/work量与MemAvailable1GiB门槛。峰值RSS/HWM、实际arena backing与完整train回答用户问题；不给尚未实现的pool虚构收益。

生命周期静态核对：train_typed的pool.install在返回前完成全部Rayon任务，函数局部owner/block posting已销毁；专用pool的各worker arena仅在此后broadcast释放。空Drop不释放缓冲区，元素为Copy；分配失败/producer panic仍保持已有初始化前缀。初始和merge均同一4worker pool，无第二初始化pool。临时RAW pointer所有权来自arena，不调用旧Vec::from_raw_parts路径。

## 用户追加：N/M、多语言、寿命与全程峰值阈值（已完成）

- [x] 一个独立agent推导理论并审视拟合，主线程执行真实J诊断；不预设word Zipf可直接用于物理posting。
- [x] 记录unique/weighted N/E、片段长度/权重、实际M/S进度；校正none保留LF/CR的语义，36拟合输入N/E/U精确对齐。
- [x] 36拟合+6规模留出+1超大中文留出，全部growth0且资源不变量通过；静态初始与终点存活分布分别记录，survivor寿命标右删失。
- [x] 局部常数/线性/幂律与跨语言预测比较；512MiB外推失败，不采用通用规模指数或语言魔法系数。
- [x] 每allocation/retire维护分桶逻辑capacity peak与coverage，结合阶段RSS/HWM筛选资源候选；不把payload直接当RSS。
- [x] 同binary 0/32/256/all，中文512MiB与英文16MiB none/whitespace各一次，共12计时；另3个allocator smoke。英文all增峰后只补测会改变选择的32/256，不加重复排名。
- [x] 模型/工作集/分配来源与resource gates、源码/二进制/输入hash、原始数据、standalone PNG/SVG、helper和报告保存。

结论：中文512MiB all不增峰、train少21.90%；英文none256B可作为原峰值预算候选，whitespace保留heap。T以整次峰值预算筛选，分配覆盖单调不等于时间必然单调。当前只完成限定配置诊断选型，生产J376363d2保持不变。详情 [POSTING_ARENA_THRESHOLD_REPORT.md](POSTING_ARENA_THRESHOLD_REPORT.md)。

## 用户新约束：降低初始化峰值，寻找时间/空间甜点

用户认为 radix 初始化把峰值从 B2 约3.44GiB抬至4.43GiB不可接受，要求权衡速度和内存；另明确授权一个独立 sub-agent 抽象算法/数学问题、调研论文和实践并做更激进原型。主线程保留 J 基线，从376363d2派生 `bpe/initial-owner-waves`；独立研究单独保留代码/报告，不修改基线。正式计时与其它CPU作业协调错开。

第一假设：现有四owner同时排序，scratch容量合计1.514GiB；全部初始pair records另1.514GiB。按1/2/4个owner分波完成sort/group/install，每波结束释放对应records，减少同时存活scratch及后续records/postings重叠。保留每owner原稳定radix、权重查询、floor筛选、精确一次预留和严格递增positions，不增加逐位置hash。代价是sort/group/install并行度下降；route仍4worker，merge不改，性能决定是否采用。

核验覆盖1/4workers、1/2/3/4owner宽度、非均匀/均匀权重、floor、分隔符、完整u16初始ID与位置顺序；复用现有逐轮HF差分及AA边界测试。初步预算为三种owner宽度各一次512MiB完整训练，使用同binary和同arena策略；确认具体峰值后只补会改变甜点选择的allocator阈值，不默认重复矩阵。新初始化低峰下重新评估Arena预算，不再以旧4.43GiB峰值提供余量。论文原型只有核对真实完整工作/内存后才能称Trainer收益。


### 初始化/通用分块阶段完成

- [x] owner wave、Radsort Rust port、安全/稳定性审查，排序并发与安装wave分别配置。
- [x] direct prefix scatter；8B heap证书、wide与affix fallback，但按大规模目标降优先级。
- [x] 33次正式模型/工作/资源/source gates；56lib tests；实际patch、raw与图表归档。
- [x] generic block的posting.len+signed sparse delta与固定物理词序对照；容量改善、时间混合如实保留。
- [x] 全路径算法抽象、最新一手研究与有条件数十GiB容量分析；没有把flat收益写成wide实测。
- [x] 新独立代理：磁盘/流式精确训练的外存抽象、26项一手实践/研究、I/O模型完成；按用户要求仅预研，见EXTERNAL_MEMORY_BPE.md。

后续arena阈值按用户要求延后。当前算法阶段结果见 [INITIALIZATION_MEMORY_REPORT.md](INITIALIZATION_MEMORY_REPORT.md)。有界初始化/summary waves和全外存Trainer仍是待实现方向，不因本机proxy通过而标成数十GiB已验证。


## 继续内存内算法：按wave归并block摘要

实际源码generic路径把全体Q的 `(key,frequency)` Vec collect后才建owner ledger。假设按初始化worker数W逐wave扫描并立即归并，释放已消费Vec，可把这份临时存储从全体Q降为Q_wave，保持每波block并行度W。代价是wave屏障、owner ledger提前与后续block构建重叠；RSS与时间不由摘要容量单独保证。global floor仍在所有wave之后，directory按原递增block顺序，跨块AA与权重语义保持。

新增57项完整lib测试通过，包括跨3个wave、0/非unit权重、AA、1/2/4merge workers、2init workers的逐轮wide serial oracle。最小测量预算：固定STD allocator/lexical词序、u32local+base、zh32MiB none强制2²⁰槽block，同binary all summaries→4-block waves各一次。直接初始化、完整train、summary容量与RSS决定是否采用；若收益只在容量且性能混合，如实保留候选，不展开默认矩阵。


分wave源码7e794db1、57tests与2个模型smoke、2个正式对照完成：13block摘要Vec48.001→16.000MiB，进程峰值479.94→431.57MiB，init.704→.790s，train4.266→4.289s。按大规模内存目标保留候选，速度损失明确单列；不展开矩阵。源码/overlay/binary/输入hash和模型/工作gate全部通过。当前汇总35次正式调用、8份overlay与源码总补丁；完整bounded字节预算与可回收pool仍是下一轮结构性候选，当前不宣布全局最优。


## 通用有界排序：实测、撤回与自适应候选（2026-10-01）

新增16次正式完整Trainer调用，本轮累计51次；当前候选029ab45b、60lib tests与完整model/工作/source gates通过。始终16B排序和8B临时记录在英文初始化回退25–30%；8B记录按用户要求撤回。最终候选按实际block pair数选择：小字典空间扫描，达到65,536项后才排序后续262,144位置tile，完整u64 key/u32 local/64位base保留。中文单block初始化两次约3.6–4.1%，四块并行约5%；英文不分配排序缓冲，未再出现前述回退。收益有限，未承诺whitespace或数十GiB表现。源码无新增字典库或FFI，生产J376保留。详情及失败版本见 [INITIALIZATION_MEMORY_REPORT.md](INITIALIZATION_MEMORY_REPORT.md)。

早期DE→H的train中位32.397→25.577s、feed+train中位36.766→29.838s；随后J全量arena已测train19.126s、feed+train23.105s。后续低峰值候选的arena配置已测train16.470–20.954s、峰值约3.33–3.46GiB；历史25.577s不代表当前成绩。GPT-6 Luna已完成当前512MiB flat主路径的完整PERF独立审计，按各自binary与DWARF核对旧H00216d91和当前源码。当前主成本是posting校验、邻边统计与owner提交；未发现高占比且明确可删除的重复工作，本轮停止继续优化和追加训练。该结论限于已测flat路径，通用分块与数十GiB仍按单独证据解释。详见 [当前PERF审计](CURRENT_PERF_AUDIT.md)。外存仅预研，arena通用阈值尚未选定。
