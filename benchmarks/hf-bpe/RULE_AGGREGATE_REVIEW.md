# J规则邻居聚合静态复核

来源：候选 `bpe/prepare-rule-aggregate/376363d2`，对照H `00216d91`。独立只读审查未发现阻塞错误；此处不代替差分测试或性能对照。

## 身份与顺序

- left目录按prior聚合，flush生成旧 `(prior,left)` 删除与新 `(prior,replacement)` 出生。
- right删除按原next，出生按final_next分别入目录，保留选中邻居的新旧ID差异。
- 局部出生按原posting顺序前插；flush把局部tail接到该key旧head，保持较晚片段逆链在较早片段之前。unique producer与有序jobs保证最终posting顺序。
- 不同链共享owner Node Vec，但只沿各自索引读取，不要求节点连续。tail最初的next为NONE，拼接后不会串入其它key。

## 边界与所有权

成功flush后，touched index恢复NONE，groups清空并保留容量。零权重出生仍保留occurrence和Node；有限长度只限制birth，remove仍累计。目录以完整lengths域创建，包含replacement；ID域超过65,536时走旧直接路径，无截断。

局部group索引受目录域限制，Node索引转换、NONE碰撞和occurrence合并检查溢出；权重沿用原预检上界。新增代码无unsafe。prepare错误时scratch/output销毁，apply未执行；flush出错后不保证scratch可重用，当前调用链会直接退出。

## 成本解释

每job每次prepare初始化两份目录，payload为`8×lengths.len()`，上限512KiB/job；另有LocalGroup Vec、逐事件目录访问、逐Task flush/reset及tail链接。每birth的owner计算与Node追加仍存在，全部计入prepare。

聚合粒度为Task×侧×neighbor。H诊断的57.84M output delta groups只给去重下限，不等于J实际probe/flush次数；10.81%的birth/remove self样本也不是纯hash成本。`peak_prepare_aggregate_bytes`是各job scratch capacity之和的跨batch峰值，不能称为实际同时驻留峰值。
