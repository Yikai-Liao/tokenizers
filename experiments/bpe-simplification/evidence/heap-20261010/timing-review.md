# 统一队列阶段计时的只读复核

独立复核读取作者的 timing-instrument.py、timing-measure.py 与隔离 timing
工作树的 index.rs／mod.rs，没有执行代码、编译、测试或 benchmark。

原四段计时互不重叠：initial 在 owners join 后；best／take 在选择时由
coordinator 串行调用；birth push 在 commit owner jobs join 后。占比使用同
一 profile 进程的 `sum(ns) / 1e9 / train_seconds`，不能除 clean 中位数或 CPU。

initial 包含 Candidate 收集、队列分配、heapify；best 包含 count hash 校验、
过时 pop／修正 push 和 dead Positions drop/free；take 包含 pop、owner/hash/
map remove；birth push 包含统计 records、遍历并释放返回的 Candidate vectors、
push 与可能的队列扩容。它们是队列协调的 elapsed，不能称纯 heap CPU。

复核指出原版计时遗漏最终 drop(index)。作者随后补了 index_cleanup 独立计时，
包含 queue、剩余列表、count maps 和 routes 的销毁。该项是索引清理的上界，
不把 count-map 清理误称纯 heap 操作。并行 owner 产生 Candidate、计数更新、
列表编码与 Arena／corpus 最终清理不在这些区间内。

计时源码仅用于独立 profile 二进制，原生产代码与无插桩成对测量不含计时器。
Instant、OnceLock 和 atomic 统计有少量开销；每 case／worker 只一轮阶段
profile，结果是描述性归因。后补 index_cleanup 的具体源码由作者验证，独立
复核最终消息对应此前四段源码，不能宣称它执行或重新审过新增清理实现。
