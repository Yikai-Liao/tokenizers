# 两份独立 fresh 审查：main 成本差异与代码预算

审查代理 `main_cost_gap_review_fresh`、`lean_budget_design_review_fresh` 均为新代理。
固定 HEAD `3ca2a00b` 加 complete-producer diff；main `e4f787dc`。
源码捕获后只读快照，未编辑、编译、测试或采样。未发现 complete 变更阻断问题。

成本审查优先顺序：

1. main 按 `ceil(total/workers)` 保留完整 ordinary candidate；当前较细 grain 和
   fragment block 数会过切。净增约 9–13 行，扩大 complete 快路覆盖，AA不改。
2. main 写计划保存未压缩坐标；当前 Compact 写计划重复编码、apply再解码。
   用 Vec<u64> 约4–6行替换、净不增行，但临时存储变8B/项，RSS必须实测。
3. fresh partial owner用bucket+neighbor目录代替 `(bucket,Pair)` hash group，
   约净增50–80行。收益待测，不把旧Perf直接当当前收益。

直接简化建议约省31–40行：initial直接汇总到Shard.states，去第二HashMap；
Cursor用block Range控制终止，去总remaining；scan_symbols去恒零offset，
initial_spans直接返回usize；Shard封装fresh发布，去重复states+queue逻辑。
原有variable fragment/U64/checked frequency/队列tie-break必须保持。

结构性剩余成本：main16B descriptor、raw pool、复用scratch、最终一次分配互相配合；
当前Positions72B和每key builder不能用旧Inline16实验宣称等价。
完整恢复估净增230–330行，删除上述冗余后仍可能超2000上限，不能承诺。
complete-only raw pool是约45–75行的窄提取，但16B/event节点和链遍历可能增加RSS/CPU。
目前先测前两项，不预承诺补齐30s与20s差距。
