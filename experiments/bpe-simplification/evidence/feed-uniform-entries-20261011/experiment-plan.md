# Feed 统一 owned entries 实验计划

## 目标和交付边界

本轮只交付计划，供另一位 agent 后续执行；此处未修改仓库、未构建，也未运行实验。
目标是在 `Yikai-Liao/tokenizers` 的 `simplify/bpe-flat-20261010` 工作线上，检验将 Feed 后的持有状态统一为 `WordCounts(Vec<(CompactString, u64)>)` 是否值得保留。
重点看串行 Feed、串行全流水线，以及 map 转 Vec 的内存代价；多 worker 场景用于检查回归。
不要先做独立 microbenchmark 筛选方案。先完成这个单一候选的最小集成，再与同版本 baseline 比较。

## 一 选择和冻结可比基线

执行开始时检查当前 HEAD、工作区差异、正在进行的 B1 工作及已有实验工具，选择届时可用的工作版本。
不硬编码历史 SHA，不强制 checkout、reset 或清理工作区，不覆盖未提交修改；B1 仍在进行时捕获一致快照，无需等待其完成。
将选定版本及应包含的未提交差异捕获为可复现快照，记录 HEAD、tree hash、完整 patch 和未跟踪相关文件清单。
从同一快照建立 baseline 与 candidate 两个隔离 worktree；不要把原工作区当作可随意修改的实验区。
baseline 保持生产源码不变，仅 candidate 实现本计划；必要的测量 harness 保持两侧一致，并与生产改动分开记录。
记录 Rust/toolchain、构建 profile、features、依赖锁文件、CPU/核绑定、线程设置、哈希种子策略及运行环境。
若当前结构已与本计划不符，先说明差异及最小适配方案，不把其他重构顺带纳入。

## 二 唯一生产改动

将 Feed 后串行持有的 `AHashMap` 与并行持有的 entries，统一为薄包装 `WordCounts(Vec<(CompactString, u64)>)`。
串行路径在原有聚合完成后使用 `map.into_iter().collect()` 移动键和值；不得通过借用再 clone 所有字符串来转换。
该转换新增 O(unique words) 遍历和 Vec 分配；移动 `CompactString` 不复制其堆字符串内容。
并行路径复用其现有 owned entries Vec，不新增一次完整复制，也不改变并行聚合过程。
转换保持来源迭代器给出的顺序；不得在初始 decorated IDs 分配前新增排序，以免改变并列情况的结果。
训练继续借用所持有的 counts；不得 drain、take 或消费持有状态，重复调用训练必须仍然有效。
保留内部借用 `WordCountsView` 对 map 的兼容入口，以及公开 `do_train` 接受 map 的现有接口和行为。
保持 Feed 聚合算法、哈希种子策略、局部表 2048 个不同 key 的刷新阈值、32 个输入的 batch、启动路径及普通 callback 错误语义不变。

## 三 类型和兼容语义

newtype 只 derive `Clone`、`Default`；不要 derive 依赖 Vec 顺序的 `PartialEq`、`Eq`、serde 或 Debug。
自定义 `PartialEq`/`Eq` 按完整 key/count 内容比较：先比较长度，可加相同顺序的快速路径，乱序时用借用 key 的 lookup。
lookup 不复制 owned 字符串；由现有聚合和反序列化路径维持“一词一项”的内部不变量，不额外添加生产路径的全量去重检查。
自定义序列化保持现有扁平 map 形状，不能输出 entries 数组，也不能多加一层包装字段。
反序列化先进入现有 `CountMap`，再移动为 Vec，保留重复 key 的 last-wins 行为及已有校验。
自定义 Debug 保持 map 风格；检查上层类型 derive 对这些 trait 的依赖。
尽量复用现有测试与 helper，不为本次改动另建测试框架或暴露新的公共兼容层。

## 四 明确排除的改动

不同时改成另一种排序引用表示，不提前排序，不删除训练阶段现有的第二个 borrowed Vec。
本计划不包含 Vec→Box 转换，也不修改 B1；两者如需实验，使用各自独立的方案和证据。
不引入 corpus plan 或生命周期重构，不调整 index、positions、allocator 或哈希实现。
不改变词频聚合、分片数、批大小、错误传播、训练选项、初始 ID 规则或 merge 的排序规则。
不借机处理独立的 encoder 问题或语料坐标 panic；发现后分别归因、记录，不宣称已修复。

## 五 先建立正确性证据

基于执行时的代码确定实际 package、target 和命令，先确认 baseline 可构建及已有失败，再验证 candidate。
覆盖默认 features、相关的 `--no-default-features` 构建/测试与适用 Clippy 检查；逐项保留命令、退出码和日志。
不要把历史 SHA 的测试结果当作当前通过，也不要为隐藏基线失败而修改无关代码。
复用内容 Eq、serde、训练 options、重复 train、公开 map `do_train` 及普通 callback errors 的现有覆盖。
补齐同内容不同顺序相等、长度或 count 不同不等、扁平 map round-trip，以及重复 key 反序列化 last-wins 的针对性用例。
检查空输入、串行与并行结果及 Feed 后重复训练；按当前测试组织补缺，不盲目复制已有测试。
已知 encoder 测试失败与尚未查明的 corpus-coordinate panic 分开列出，注明 baseline/candidate 是否各自可复现。
新出现的失败或任一侧无法完成同一场景时，先定位；不得把缺失结果当作性能胜出或对等通过。

## 六 性能矩阵和运行次序

使用已有、精确相同的完整 256 MiB 中文和英文语料；记录路径、字节数、内容 hash、读取方式及所有训练参数。
保持现有流式输入和预处理路径一致，不为测量额外引入全文预加载。
矩阵为 2 种语言 × ByteLevel/Whitespace × 1/4 workers，共 8 个场景；若项目已有固定多 worker 配置，可沿用并明确写出。
每个场景均比较 A=baseline、B=candidate，完整语料正式跑两轮：第一轮 A→B，第二轮 B→A。
总计 8 × 2 × 2 = 32 个正式进程；不默认要求五轮。
每个场景每一侧各做一次不计入统计的 warmup，共 16 次；可用固定小样本运行相同路径，两侧输入和配置一致。
已有 warmup 只有在二进制、features、配置及预热条件一致且有记录时才可复用；历史正式 baseline 不能替代本轮配对运行。
先完成全部构建，再顺序运行测量；不同实验进程之间、实验与构建之间不得重叠。
固定场景顺序和机器条件，记录其他负载、温度/降频迹象、可用内存及缓存策略，不为候选单独制造有利状态。
只有结果噪声明显或两轮方向矛盾时，才追加相关场景的第三轮配对，并保留追加原因及运行次序。

## 七 时间 内存和等价性测量

Feed wall/CPU：使用同一进程中相同的边界，candidate 的串行 map→Vec 转换计入 Feed 完成之前。
全流水线 wall/CPU：使用相同起止定义，涵盖同样的读取、Feed 和训练阶段；说明 CPU 是 user+system 还是其他口径。
不得用外层 child-process wall 冒充 Feed wall，也不得将某一侧不同边界的数字直接相除。
Feed 后分别记录当前 RSS 与累计 HWM；后者是进程启动至此的峰值，不能称为 Feed 独占峰值。
另记全流水线 HWM、swap 活动与 major faults；在重型导出/校验前捕获流水线 HWM，若另报整个进程峰值则明确单列。
保留测量来源和单位，不把瞬时 RSS 与峰值混为一谈，也不把校验分配造成的峰值归给训练。
每次记录 unique words、输入规模、线程数及成功/失败状态，确保没有以少处理数据换取速度。
在计时区间之外比较完整 key/count 集合；不要只比总频数、unique 数或抽样。
完整模型按语义比较，规范化仅消除无语义的 map 键展示顺序；完整词表到 ID 的映射、配置及 ordered merges 必须一致，不要求 JSON 字节顺序一致。
不得排序 merges、重编号词表 ID 或省略模型字段来掩盖差异。
正确性导出、额外校验或诊断若会污染全程 wall/HWM，另开非正式运行，不混入性能样本。
只有现有指标无法解释结果时，另做 instrumented 诊断测量 map→Vec 阶段；诊断数字单列，不给生产路径常驻加 timer。

## 八 解释内存结果

当前串行状态是 owned map，加训练排序使用的 borrowed references Vec；在当前 64 位布局下后者约 16 bytes/word，执行时核实元素大小。
候选状态是 owned entries Vec，加同一个 borrowed references Vec；这次并未删除那份引用数组。
转换期间可能同时保留 map bucket allocation 与新 Vec backing allocation，因此瞬时峰值可能增加。
转换完成后 bucket allocation 可释放，稳态持有成本可能降低；allocator 是否归还页及 RSS 是否下降必须实测。
分别报告 Feed 后 RSS、累计 HWM、全程 HWM 和时间变化，不预设“改成 Vec 就省内存”或“移动就是零成本”。

## 九 报告和最终判断

交付最小 candidate patch、版本与环境清单、全部命令和原始日志、每场景逐次数据，以及完整等价性检查结果。
每场景报告两轮配对的绝对差值、candidate/baseline 比值及离散情况；主要结论区分串行收益与多 worker 回归控制。
明确列出新增 O(unique) 转换是否影响 Feed/全程时间，及稳态和峰值内存分别如何变化。
报告生产代码净增减行数、真正消失的状态类型/分支/重复逻辑，以及仍需保留的兼容代码；文件搬家不算简化收益。
正确性和兼容语义必须满足；性能不预设 5% 或其他硬门槛，小幅速度与内存取舍交由用户决定。
结论写清保留、放弃或证据不足的依据，分开标注已有失败、新回归和无法完成的场景；未经确认不合并或覆盖 B1 工作。
