# 早期混合 prototype 静态审查

Reviewer: `write_plan_review`，独立上下文、全程只读，2026-10-10 06:04 UTC。
固定对象为 HEAD `3a298346` 加补丁 SHA256
`796648066cc8efbcb0b360c4acbd7583f25c78344320c4efc9ba9187acbb8566`。
该快照同时包含 birth 与 Writes prototype；不能当作纯 Writes 补丁 fa86dced 的
固定对象审查，不能用本报告证明真实成本。

本轮静态审查结果：新增缺陷 0；值得启动下一轮实验的全局简化候选 0。
已通读六模块、完整测试、独立 reference 和相关历史反证；未执行目标、测试
或编译。CodeGraph 无此工作树索引，依据源码判断。

- Writes 等价路径成立。保存的是 fresh_matcher 当场生成的完整 Match；原
  Compact 应用时重构的 start/right/after 与它一致。准备、应用之间没有下一轮
  identity 更新，joined 阶段和端点不相交条件保持。删除 record 的 Result 没有
  丢掉检查：原 push_ordered 始终返回 Ok。
- Birth 统一未发现新增语义问题。fresh 每个 newborn key 仍由一个规则产生，
  indexed tasks 保持空间顺序；延后完整 birth 入 shard 不会改变旧边界 removal，
  因为新 replacement ID 尚未出现在快照中。Partial 汇总后 floor admission、
  完整 producer 提前剪枝、零权重 positions 均保持。
- reuse 与清理保持。removal-before-birth、逐 action checked signed ledger、
  左右顺序、bucket 和 cohort 发布条件不变。片段解码后仍排序并保留重复与
  完整 u64；owned Partial 可单独释放，Route drains 与 joined 错误退出仍释放
  未发布 payload。

继续搜索了初始化→队列→批选→准备→应用→提交→重启→输出。未找到新的
净收益完整切片：Fresh/reuse 的事件语义、AA 的贪心 starts、Plan/Corpus 的
生命周期差异仍有实际责任；直接合并会保留或新增例外。现有索引、权重缓存、
窄端点存储也没有形成需要消除的跨模块协调负担。没有把性能猜测当成否决依据。

结论限于静态增量：Writes 的 4→24 B 临时记录成本在当时仍待测量；既有等优先
reuse cohort 顺序证明缺口没有因此关闭。隔离副本旧文档按作者已说明的采用
阶段更新安排处理。
