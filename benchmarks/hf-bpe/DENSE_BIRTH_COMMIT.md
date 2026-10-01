# BPE 出生频率与 posting 汇总

分支 `bpe/dense-birth-commit` 从 `adb219cda4600e52ea2809d2537ac551c168dd27` 派生，在现有 prepare、rewrite、commit 阶段中采用出生分桶与邻居 ID 直接汇总。旧任务切分、读取旧状态、有效位置表及 rewrite 屏障继续提供原有并发语义。

## 数据流

对于现有 AtomicU32/AtomicU16 flat prepare、token ID 数不超过 65,536 的邻居聚合路径：

1. 每个任务仍按规则 posting 总数切分，读取旧语料与选定规则映射来计算最终出生 pair。
2. 删除频率继续汇总到线程局部旧边表。出生分组按 `(owner, 规则, 左/右方向)` 保存邻居 ID、权重、数量和原 Node 链头。
3. prepare 全部完成后执行原 rewrite 阶段。
4. 每个 owner 的 commit 用一个词表大小的邻居目录汇总同桶的任务片段。目录在桶间复用，只重置触及项。
5. 全部任务的频率相加后再检查全局门槛。接受的 pair 一次分配最终 posting，逆向填充原 Node 链，随后插入 ledger 并进入候选 heap。

出生 pair 包含本批新身份，该身份确定规则。左出生的邻居未参与本批合并；两个选定区间相邻时，边界由左区间的右侧产生。因此一个最终 key 不会跨两个规则/方向桶。原有任务中的同规则 posting 顺序，以及同规则跨任务的顺序，保证逆向填充后 posting 全局递增。

## 成本与覆盖

省去出生 flush 的线程局部哈希入表和 commit 的临时出生哈希表。新增分桶向量、邻居目录、合计组与片段链；每 owner 的目录为 `4 × 当前身份数` 字节，四个 owner 在该路径上合计不超过 1 MiB。其余向量随实际分组数增长，完整进程峰值单独测量。

AA 原路径、generic 多 block、超过目录身份上限，以及现有非 shared corpus 路径使用原 commit。身份别名配置沿用原串行兼容入口。公共训练 schema 与参数不变。

## 核验

该代码全部 64 个库测试通过，覆盖 HF 逐轮随机差分、独立 greedy oracle、全局出生门槛、特殊 token 激活、大权重、长度限制、AA、跨 block 与不同存储宽度。完整语料对照还校验模型签名及 posting arena 的分配/退休闭合。

独立对照使用 `build_standalone_bpe.py` 的固定 feed 哈希种子和原 Trainer API；融合实验的同二进制三种组合及诊断记录在主实验目录 `results/fused-rewrite/`。最终选型和实际运行次数以 `FUSED_REWRITE_REPORT.md` 为准。
