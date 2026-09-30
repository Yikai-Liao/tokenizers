# 普通 Rust 循环的自动向量化检查

## 范围与已确认事实

当前工具链 rustc 1.98.1、LLVM 22.1.8；实际benchmark是默认release opt-level=3、默认x86-64 target。没有target-cpu=native、强制vector width、SIMD intrinsic、std::simd或汇编源码。

初始化当前用稳定radix，不代表全部循环容易向量化。histogram与scatter会多次更新同一个bucket，存在真实的跨迭代依赖；UTF-8解码、权重目录搜索和动态HashMap更新也不是单纯连续数组运算。LLVM可以自动vectorize间接访存，但会用成本模型决定是否划算，复杂控制流和调用也可能阻碍它。[LLVM向量化指南](https://llvm.org/docs/Vectorizers.html#diagnostics)

现有prepare把posting过滤、稀疏权重查询、selected邻居查询及HashMap/出生记录更新放在同一循环。固定循环使用Atomic的Relaxed读取，LLVM中为load atomic monotonic；其语义不会因为x86机器码是普通mov就变为普通load。[LLVM原子说明](https://llvm.org/docs/Atomics.html#monotonic)

## 实际编译证据

基线诊断从F `4a2f148a`的临时probe源码生成，**F与DE的fused_batch.rs逐字一致**。因此这份诊断用于查原prepare循环形状；不把F的posting安装diagnostic当成DE安装的证据（DE已替换为bulk）。

```bash
CARGO_TARGET_DIR=/root/code/tokenizers/benchmarks/hf-bpe/target \
/root/.cargo/bin/cargo rustc --offline --release \
  --manifest-path .build/native-singleton-birth/runner/Cargo.toml \
  -p tk-train --lib -- -C debuginfo=1 -C remark=loop-vectorize \
  -C llvm-args=-pass-remarks-missed=loop-vectorize \
  -C llvm-args=-pass-remarks-analysis=loop-vectorize --emit=llvm-ir,asm
```

这次有额外行号信息，仅作诊断；计时二进制未重新链接。四个prepare worker实例在对应优化IR函数中各有5处load atomic、没有vector.body循环；出现的局部向量IR不能据此当成posting循环向量化成功。radix histogram的remark明确为unsafe dependent memory operations；“unsafe”在这里描述依赖，未指出Rust内存安全错误。原始巨型IR/assembly/log保留在忽略的.build/target内，精简的实际二进制证据持久化到results。

G2 **实际参与计时的二进制** `87885ccdee0dd4950654b06f1cf86b8b89150ca5eba084643078b9612c119d15`：u32 prepare worker从0x3899e0开始，纯比较loop为0x389ce0–0x389d4f，含4条pcmpeqd、2条pand及packuswb；每迭代加8，有标量tail。它处理连续left/right数组，确实自动比较8个posting端点。source没有显式SIMD，编译器选择SSE2的4-lane向量与两份interleave。精简反汇编、命令、source/binary锁定见 [codegen.json](results/optimization-vector-screen.codegen.json)。

## 两个分开的候选

| 候选 | parent/commit | 修改模块 | 额外成本与限制 |
|---|---|---|---|
| G1 read-phase | DEc8702374 → cbb935b2 | 仅融合prepare只读阶段 | Rust>=1.98安全get_mut_slice；无完整数组copy |
| G2 prepare-blocks | G1 → 2f238505 | 融合prepare端点过滤结构 | 每运行job固定1152B栈数组，gather/mask/第二遍消费全部算prepare |

G1在prepare join期间从独占&mut corpus建立普通slice。标准库安全API表达没有并发原子访问，Prepared不返回引用，view在随后Atomic apply前结束；原有共享写以及join屏障保留。[Rust get_mut_slice](https://doc.rust-lang.org/core/sync/atomic/type.AtomicU32.html#method.get_mut_slice)

G2每128个posting先按原left-first规则读取端点，left失败仍跳过right随机读取；纯slice zip计算mask；再按原posting次序处理weight/selected/delta/birth。初始化、AA、rewrite、commit算法与G1相同；hash操作、间接查询仍为标量。纯kernel向量化不等于整个prepare已向量化，更不等于有端到端收益。

G1全47测试通过、G2全48通过，包括1500-case逐轮HF差分、Atomic各存储边界、相邻批次、跨worker有序出生与floor聚合；独立审查与关键计时并行，见 [AUTO_VECTORIZATION_REVIEW.md](AUTO_VECTORIZATION_REVIEW.md)。

## 当前性能证据

三项单次screen已完成，模型完整签名均一致，参数仍为512MiB中文none/50k/min2/u32/4线程。

| 秒 | DE | G1 | G2 |
|---|---:|---:|---:|
| prepare | 11.727 | 11.881 | 12.177 |
| delta（含prepare及AA，不重复相加） | 12.036 | 12.176 | 12.490 |
| commit | 7.177 | 6.112 | 6.259 |
| init | 9.311 | 8.888 | 7.916 |
| train | 34.354 | 32.568 | 32.044 |
| feed+train elapsed | 38.280 | 36.653 | 36.147 |

这个screen没有显示prepare改善；whole更快不能全部归因自动SIMD，未改init也变了。DE/G2三个交错pair已完成：prepare配对G2/DE比0.9783/1.0195/1.0409，中位1.0195；elapsed配对0.9572/1.0018/1.0528，方向混合、基本持平。未确认模块或完整组合稳定收益，不推荐G2替代DE。见 [稳定性报告](results/optimization-vector-stability.summary.md)。

### 测量的随机性

原Trainer feed使用默认AHashMap，corpus模块冻结wc.iter()顺序；本次没有固定哈希seed或跨进程物理词顺序。因此相同输入/model签名/N/E不代表字节完全相同的corpus排列。源码确认这是一个尚未控制的变量，但没有量化其影响，不能把所有波动归给它，也不能把三个样本当总体分布。保留原API默认行为，通过交错配对、阶段数据与sample CV报告实际表现；未偷偷改benchmark hash策略或强制编译器向量宽度。

## 优先级修正

用户要求只关注大块热点。当前实际SIMD只处理预收集端点的比较及mask写入，不覆盖随机corpus读、weight/selected查询或HashMap出生管理，代表的计算范围很小。已停止G1/G2继续推进，取消拟议纯mask微基准；无此微基准结果，不声称局部SIMD加速比。下一诊断定位DE prepare/初始分组的权重读取，避免只因一段可以SIMD就优先改它。
