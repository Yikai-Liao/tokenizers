# Perf热点与优先级

2026-10-09，固定Fork main e4f787dc 与1719行head-read候选。
同一中文ByteLevel word map，4核、50K；cycles 99Hz / dwarf8192调用栈。
完整模型相同、child swap=0、lost samples=0。Perf有明显开销，诊断CPU/wall不用于正式性能排名。
硬件计数约72–85% running，包含load/serialization/record-wrapper；不能当作精确训练阶段PMU。

## 主要差距

下表按最接近的可识别engine调用栈归属加权cycle samples，每个样本仅计一次。
包含malloc等子调用；engine_other和outside/unknown单列。不是阶段wall计时。

| 归属 | main样本周期B | 候选样本周期B | 候选可识别engine占比 |
|---|---:|---:|---:|
| merge prepare | 63.22 | 100.48 | 42.46% |
| owner commit | 12.76 | 55.14 | 23.30% |
| apply | 9.65 | 20.70 | 8.75% |
| initial pairs | 25.81 | 33.06 | 13.97% |
| materialize | 5.87 | 5.04 | 2.13% |

owner周期规模约4.3倍、prepare约1.6倍；两项约占归属周期差额的73%。
这是不同采样运行的方向性比较，不能据此断言恢复某机制就获得相同加速。
Positions::push自占9.86%全进程周期样本；FlattenCompat<read_blocks,Cursor>::next自占9.88%。
两类按调用栈约23%可识别engine样本，横跨initial/prepare/apply/owner。
main编码/解码常被inline，不能用独立codec符号的0与候选比较。

## 测试顺序与理由

1. 恢复真正的硬件prefetch16ahead，源码小且main已有验证。此前同步head-read不等价，
   不能据此评价_mm_prefetch；使用Slots内部安全范围检查封装架构intrinsic。
2. 位置流减少每个坐标的Box/FlatMap状态机和函数边界；其自样本很高，decoder annotate显示
   入口寄存器压栈/状态tag访问已有大量样本，值得试紧凑有类型cursor。
3. fresh owner避免再次解码/重编码完整births，提取main complete-producer原则。
   可变长度restart block需要显式entry offset；固定128步的当前格式不能直接拼bytes。
   只移动完整有序U64块，reuse仍排序。收益、额外元数据和预算均需实测。
4. 再检验prepare邻居/selected lookup及slot单次dispatch；初始dense/目录细节降优先级。

perf annotate对decode成功，其余5个热点符号在本机perf6.12.111上segfault，
关闭source/demangle后重试仍失败。保留失败记录，必要时用sample IP+objdump映射，
不会编造这些符号的source级热点。

证据见evidence/perf-*，raw perf.data/callers/stacks位于results/profiles。
profile.py重现诊断，profile_summary.py重现归属统计。最终选择仍需无采样相邻复测。
