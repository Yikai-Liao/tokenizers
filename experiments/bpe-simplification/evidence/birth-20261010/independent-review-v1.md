# 新上下文只读复核：压缩 birth

对象为 `3a298346` 加纯 Rust patch
`c3f7affc48e0ea0aa83d35c2b679369c710655c9ac77edf06871fafe5a8d249d`。
审查结束时再次核对哈希。审查者覆盖六模块、公共训练入口、输出与相关测试；
CodeGraph 未索引这个 worktree，结论由目标源码回证。没有修改、编译、执行测试
或运行 benchmark。未发现新增已证明缺陷。

完整 producer 保留提前 floor 剪枝；partial 使用 owned 压缩片段。Owner 保留
每个 action 的 removal-before-birth、signed 检查与 bucket 顺序；reuse 解码、
排序、发布历史 cohort。Fresh newborn 含本批 replacement，快照 removal 不会
命中该新 key；reserved inactive 身份单独执行，AA 仍为 partial。因此延后
完整 birth 的 count 插入未发现计数干扰。单片段移交、多片段编码及失败时
drain／owned payload 释放和 attempt 丢弃均有对应保护。

只读核对作者原始日志：default／no-default 各 17 native tests＋1 doctest；
Clippy 成功；直接导入实际 positions 模块的 Miri 两项测试通过。中文成对
测量在审查结束时尚未完成，本复核不作运行成本结论。前轮等优先 reuse cohort
次序的证明限制仍存在，不归为本次新增问题。

新的全局候选只有一项：统一 Writes 为 `Vec<Match>` 与 replacement。Fresh
已取得完整 Match，却只保存 start 与规则几何并在 apply 重构；reuse 保存
完整 Match。统一可能退出双表示、几何镜像、坐标往返和 record／apply 分派。
在 64 位平台通常 4 字节的 fresh 写记录会增至 24 字节，须对冻结中文 4／6
workers 与长 AB／AA 写缓冲压力实测 wall、CPU、HWM，不能静态拒绝。
如果需要补偿机制控制成本，或没有净收益，则还原。AA starts 统一压缩 Source
需要更多门槛与 codec／seek 工作，本轮净收益较弱，没有列第二项凑数。
