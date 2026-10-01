# 全量Bump保留posting的实际内存与总体影响

## 回答

已实际使用bumpalo3.20.3跑完512MiB中文/none/50k/min2、初始化与merge均4线程。**全部posting堆载荷（包括中途退休的buffer）保留到训练结束，再按arena整块释放**。同期标准分配J→Bump各一次，结果：

- 全程进程HWM标准J 4.433716GiB、Bump 4.434608GiB，只多0.914MiB（0.0201%），没有明显峰值上升。
- 训练API 24.350969→19.126216秒，少21.46%；feed+train 28.956902→23.105492秒，少20.21%。
- 实际51个arena backing chunks最终统一释放65.341ms。模型与工作量全部一致；进程VmSwap0B，最低MemAvailable两次均>3GiB。

这组实测支持全量arena作为正式候选，修正了此前只根据可能保留退休空间而优先推荐pool的判断。结果仍是n=1诊断；没有把小对象池或自动阈值的未测收益当成结论。生产J worktree未修改。

## 容量究竟增加多少

| 容量或对象 | 实测 |
|---|---:|
| 训练累计posting堆分配请求 | 9,323,712次 |
| 累计请求payload capacity | 1,403,805,436B / 1.307396GiB |
| 结束时仍存活的posting payload capacity | 809,100,424B / 0.753533GiB |
| 退休buffer保留下来的payload capacity | 594,705,012B / 567.155MiB |
| Bump全部chunk容量（不含footer） | 1,996,817,216B / 1.859681GiB |
| Bump向系统allocator请求容量（含footer） | 1,996,819,664B |
| Chunk数量 | 51，分布在4个worker arena |
| posting扩容次数 | 0 |

累计请求比终点live capacity大73.50%，这是真实的退休buffer保留代价。Bump还存在chunk增长余量、对齐等容量；backing减累计payload请求约565.540MiB。这些是capacity，不等于已触及或实际驻留RSS；不能将它们直接加到标准版本RSS上。

全程RSS最大值还受初始化route/radix大临时工作集影响：相同源码统计初始route buffer峰值3,251,681,824B。全程HWM只取各阶段最大值，增加merge期间的保留容量不必改变全程最大值。本次没有记录两版每个阶段的RSS时间序列，因此不把具体峰值时刻或allocator碎片减少程度写成已验证原因。

标准allocator的size-class取整、每块管理开销也不在809,100,424B live payload统计内；Bump backing容量与该数值不能充当精确的RSS差额。**峰值变化直接以实测HWM差936KiB回答。**

## 时间

单位秒，两个调用串行，计时期间无编译/测试或其它训练。

| 阶段 | 标准J | Bump J | 变化 |
|---|---:|---:|---:|
| 初始化 | 5.959652 | 5.324044 | -10.67% |
| Merge | 14.131368 | 13.438598 | -4.90% |
| 完整训练API | 24.350969 | 19.126216 | -21.46% |
| Feed+train返回 | 28.956902 | 23.105492 | -20.21% |
| Post-merge区间（train-init-merge） | 4.259950 | 0.363574 | -91.47% |
| 其中新增inventory扫描 | 0.187714 | 0.256945 | 诊断开销 |
| Post-merge扣除inventory扫描 | 4.072236 | 0.106628 | 包含构造、释放与返回 |

Bump的post-merge扣除扫描后约106.628ms，其中arena release65.341ms、arena计数collect0.111ms，其余约41.176ms。标准J对应区间约4.072s。该残差仍包含词表/merge构造及其他退出成本，不是精确的owner Drop单项。

直接改动是posting分配及释放；初始化posting安装也受到影响。完整train少5.225秒，其中收尾区间少3.896秒。未改的阶段、feed和物理访问仍有波动，不能把全部差額当成allocator可稳定消除的工作。两版均保留相同末尾inventory扫描；Bump还累计线程局部分配计数，观测开销未独立隔离，因此将此次结果标成诊断对照。

## 阈值与数据规模

用户指出不同输入规模和merge目标会改变长期posting的最大长度、中位数和整体分布，使固定cutoff的覆盖率和收益改变。这一点成立。这里不选择一个未经验证的全局cutoff；实际把所有堆posting放入Bump，先测无阈值方案的上限代价与总体收益。

现有终点容量分布中，同一≤32B cutoff覆盖16MiB那组堆posting的76.01%，512MiB这组的61.13%。前者40,759条规则、后者29,243条规则，输入及merge目标均变化；该差异不能只归因于数据集大小。当前分桶是capacity而非精确长度，未求精确最大值或中位数，也没有测生命周期分布。

若再研究混合pool/普通heap，应按当前workload的容量分布、退休与复用量、同时存活容量及内存预算决定候选cutoff。固定字节size classes可作为实现机制，策略阈值可以配置或根据初始化分布选取；仅按数据集字节数等比例缩放或直接取中位数都缺少分配成本与复用收益依据。此次512MiB全量Bump已能满足资源门槛，不需要先靠cutoff才能运行。

## 实现、生命周期与正确性

独立副本`.build/native-j-bump-retain`从J posting-inventory副本派生，源码基点仍376363d2，Cargo release profile不变。添加纯Rust bumpalo3.20.3依赖，本机已有Cargo下载缓存，因此offline build成功19.12秒；“缓存已有依赖”不表示需要C/C++或系统库。

每个专用Rayon worker一个TLS Bump，分配串行发生在该worker上，无共享Bump。posting以raw pointer+len/cap记录初始化前缀，arena整个train期间不回收或移动数据；删除entry仅丢弃handle。train_typed返回、所有posting owner及任务销毁后，再broadcast释放4个arena。输出只有owned vocab/merges/stats，释放时无输出指向arena。模型/操作循环未改。只有posting payload保留，哈希表、candidate堆、corpus和临时workspace沿用原生命周期。

配置限定u32/AtomicU32/flat32/init4/merge4，防止第二初始化pool或其它生命周期用法；并非可直接用于所有库配置的生产实现。16MiB smoke与512MiB完整模型摘要、N/E/pairs、batch/fused/posting visits门槛通过；终点posting容量与分桶逐项一致。此次没有新增全套库测试或更广配置支持。

实际每次heap申请累计局部count/capacity，终点broadcast读取backing/chunk，故无需把terminal容量当成累计容量猜测。growth counter为0验证当前flat路径一次预留。

## 资源与证据

标准J采样峰RSS 4,761,030,656B、Bump 4,762,210,304B；/proc VmHWM差额936KiB比0.5s采样更适合回答微小峰值变化。最低MemAvailable标准 3.087GiB、Bump 3.110GiB，进程VmSwap0B。全局paging保留在raw JSON，不等同本进程换页。

[signature](j-bump-retain.signature.json)、[标准J raw](j-bump-retain-baseline.jsonl)、[Bump raw](j-bump-retain-512.jsonl)、[arena原始计数](j-bump-retain-512.stderr)、[补丁](j-bump-retain.patch)、[smoke](j-bump-retain-smoke.jsonl)。所有raw相邻environment保存输入/源码/lock/runner/binary hashes。

复现：复制`.build/native-j-posting-inventory`为新的`native-j-bump-retain`，应用补丁，将runner包名和依赖绝对路径改为新副本，offline release构建；使用run_native_fair，512MiB/none/50k/min2/init4/merge4/atomic-corpus，输出新名字。实际命令和配置完整保存在environment中，不能覆盖旧binary或计时结果。

参考：[Bump容量API及生命周期](https://docs.rs/bumpalo/3.20.3/bumpalo/struct.Bump.html)。
