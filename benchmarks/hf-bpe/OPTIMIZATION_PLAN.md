# BPE 下一轮优化计划

## 基线与约束

基线为原接口u32初始化4/merge4版本 `bpe/parallel-count4`，训练源码固定在 `c07a6e39`，当前仅benchmark清理后的HEAD为 `b6a28768`。固定512MiB真实中文语料上训练54.760秒、merge27.457秒、峰值RSS3.390GiB；PR基准308.111秒/6.569GiB。原子对照训练52.240秒，单次差异不作为选型证据。

继续保持原 `BpeTrainer::do_train/train_vocab/Trainer::train`、公共字段和serde格式。语料与posting统一u32，目标词表50,000、min2、none、串行feed/4线程训练。仅系统MemAvailable≤1GiB停止，进程与系统swap记录；构建、测试、下载与正式计时错开。保存原始计时、RSS、最低余量、换页、源码/二进制/输入hash和模型签名。

每项改动先写本文件，再创建独立worktree/branch并提交。候选实现通过相称正确性检查后才测量；收益、代价和放弃原因同步到 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)。固定源文件与临时探针构建区分记录，历史基线不覆写。

## 候选A：并行语料构造

计划分支 `bpe/corpus-parallel`，worktree `/root/code/tokenizers-worktrees/corpus-parallel`，从非原子初始化4版本派生。

- [ ] 拆分alphabet、容量/区间规划、最终数组分配、语料填充阶段计时；13.068秒现有总计不能全部算作填充。
- [ ] 固定现有词遍历顺序，分成连续词块；并行统计每块保留字符数和边数。
- [ ] 前缀和确定独占区间，一次分配最终corpus，各worker并行解码并写入最终布局，归并词边界/权重。
- [ ] 初始lengths活跃标记用局部结果归并，避免多个worker写同一ID槽；保持special预留ID尚未活跃的0长度状态。
- [ ] 保留alphabet与canonical ID分配行为、有限limit_alphabet同频边界；本候选先沿用原alphabet实现。
- [ ] 覆盖空输入、全裁剪、重复权重、强制alphabet、预留special ID、原子/非原子slot和跨块边界；非空affix仍走原generic路径。
- [ ] 通过差分与原接口测试，固定commit，构建再进行一次512MiB关键比较，报告分阶段耗时与RSS。

初始化执行并发由现有私有initialization_workers控制，merge workers保持原值。尽量只保存词引用及每块规划元数据，不构造每worker完整ID语料再复制，不能用新增4N临时数组换取时间。

## 候选B：减少Plan与delta重复走访

在候选A测量完成后选择其胜出版本或原基线为新分叉起点，独立命名worktree/branch，记录确切parent commit。

- [ ] 逐段核对原型的有效位置检查、邻边delta、出生链和最终邻居判定，列出融合需要保持的协议。
- [ ] 细分delta中的权重查找、哈希/出生记录与提交工作，避免把14.668秒全部归给一个操作。
- [ ] 实现一个可证明的融合候选，优先减少过滤后再次走访和全局16字节Plan存储；保留AA与预留ID规则的特殊路径。
- [ ] 若共享语料融合需要AtomicU32，独立从对应基线分叉；Relaxed访问不增加槽位大小，但算法正确性仍须另证。
- [ ] 测试逐轮greedy顺序、相邻同批合并、有限长度门控、AA、阈值聚合与最终posting位置；复用独立审查。
- [ ] 通过门槛后仅做一次关键大语料比较；若原子/布局/其它改动同时发生，明确收益不能单独归因。

混合规则全局排序目前支持selected邻居识别、权重游标与独占写区；不能单独删除。候选失败或无实用收益时保留branch/commit和原因，胜出版本才进入推荐入口。

## 记录与本轮结束条件

- 每个候选有干净worktree和独立commit，公共接口保持一致。
- 原接口、完整模型签名及关键边界验证通过；独立审查结论持久化。
- 汇总训练/merge/初始化加速比、RSS与余量、Atomic内存事实、热点变化、下一步建议和仍未验证的方案。
- 完成本轮A及一个经过源码推导的B候选比较，选择有收益且可维护的实现；不展开完整benchmark矩阵。
