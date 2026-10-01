# Wikipedia posting 规模、寿命与分配阈值：理论及独立静态分析

本报告研究 J `376363d2` 的 flat32 posting，而不是 tokenizer 输出 token 的长度。生产源码没有改动。初始静态分析及后续 43 次真实 J 生命周期诊断均已完成；[多语言实测](POSTING_THRESHOLD_EMPIRICAL.md) 给出拟合、留出误差及容量曲线，[阈值计时报告](POSTING_ARENA_THRESHOLD_REPORT.md) 给出同 binary 的实际速度与阶段峰值。下文第 5 节保留诊断设计，第 6 节保留首次 full-Bump 对照的证据范围。

## 1. 应该随规模变化的是什么

posting 的最大长度、中位数、分配次数和占用字节可能有不同增长规律。对于概率稳定的一个常见 pair，其物理出现次数可以近似与物理语料规模成正比。所有 pair types 的中位数同时受到新增稀有 pair 的影响，可以增长很慢、保持不变，甚至下降。平均值增加不能推出中位数增加。

适合分配决策的输入至少包括：

| 量 | 定义与用途 |
|---|---|
| 原始 bytes | 文件规模与 I/O 成本；不是 posting 的直接单位 |
| `U` | 预分词后不同片段数量 |
| `N_u = Σ L_i` | 去重后的物理字符数；`L_i` 是不同片段的字符数 |
| `E_u = Σ max(L_i−1,0)` | 去重后初始物理边数；初始 posting 总位置数的直接上界 |
| `N_w = Σ w_i L_i`、`E_w = Σ w_i max(L_i−1,0)` | 还原片段重复权重后的字符数、边数；参与 pair 排名和停止规则 |
| `M` | 实际学到的 merge 规则数，不是目标词表大小 |
| `D_L`、`D_w` | 片段长度与重复权重的分布；还应观察两者的联合关系 |
| `S(M)` | 截止第 M 条规则累计成功的物理合并次数；不乘权重 |
| birth/retire | posting 出生/退休的规则时刻、容量、退休原因 |

在第 t 步，令 `n_{i,k}(t)` 为片段 i 中当前有效 pair k 的物理出现数，则：

```text
有效物理出现数      e_k(t) = Σ_i n_{i,k}(t)
加权 pair frequency f_k(t) = Σ_i w_i n_{i,k}(t)
```

J 的 posting 还保存后来失效的位置，其长度 `h_k` 是出生时安装的历史记录数；频率下降时不会逐项压缩这个列表。只要该 entry 仍存活，就有 `e_k(t) ≤ h_k`；`f_k(t)` 的单位是加权出现次数，不能直接当成 `h_k`。例如把已经满足 floor 的全部片段权重同时乘十，会改变 weighted N/E，却不增加物理位置；固定实际 merge 数、顺序与 floor 资格不变时，posting 大小也不变。

`L_i` 必须数实际 feed 片段中的字符。这个 benchmark 自定义 `Lines<R>` 调用 `BufRead::read_line`，保留行尾 LF/CR；none 模式把行尾作为可训练字符，whitespace_split 模式才会通过空白拆分去掉它们。不能按标准 `BufRead::lines()` 的语义先删行尾。

片段 ESS `=(Σw_i)²/Σw_i²` 可概括重复权重集中程度，但不能替代 `E_u`。一个长片段中的位置共享权重，pair 与片段权重也可能相关；把 ESS 当成独立位置数量会引入新的错误。

## 2. 先推导可验证的资源上界

J flat32 的重要性质是：新 pair 含有刚产生的 token ID。旧 pair 身份不会重新出生，posting 在出生时统计数量、一次预留、一次安装。J radix 初始化和 flat births 均使用 `SmallPosting::with_capacity(count)`，之后没有扩容；这次 512 MiB Bump 诊断实际 growth counter 为 0。

初始 posting 记录数至多 `E_u`。每次成功的物理 merge 消灭一条边，最多产生左、右两条新邻居 pair 记录。把所有训练期间出生的记录加起来：

```text
P_birth(M) ≤ E_u + 2 S(M) ≤ 3 E_u
S(M) ≤ E_u
```

这里保守地允许每次 rewrite 两个邻居都产生出生记录；片段边界和批量邻接合并通常会使实际值更低。被 frequency floor 丢弃、没安装 posting 的记录只会进一步减少分配。该上界覆盖保留至训练结束的所有历史 posting，不依赖 word Zipf 的指数、语言和 raw bytes。

当前 `SmallPosting<u32,2>` 对长度 0–2 内联；堆缓冲区的请求 capacity 是 `max(h,4)`，因此对每个堆 posting `cap ≤ 4h/3`。在一次预留且无 growth 的当前路径上：

```text
累计 heap requested slots ≤ 4 E_u
累计 heap requested payload bytes ≤ 16 E_u
```

更紧的表达为 `4 × (P_heap_birth + A_len3)` 字节，`A_len3` 是长度恰为 3 的堆分配数；因为只有 h=3 被补成 capacity=4。它是保守的 payload 上界，不包括 owner 表、candidate 堆、corpus、临时 route/radix 数组、allocator 元数据和 Bump chunk 增长余量，也不等于 RSS 上界。若改用其他初始化路径、引入扩容或重新出生，必须重新核对前提。

**这提供一个真正随工作负载变化的公式：首先按 `E_u` 和已观测的 `S(M)` 判断全量 arena 的预算，而不是先猜一个 `C × raw_bytes^α` cutoff。** 它不能单独预测最终中位数，却能回答是否需要混合分配才能装入内存。

## 3. Zipf/Heaps 能给出的候选模型

### 3.1 一个固定 pair 的长度

若某个阶段的物理 pair 概率稳定，忽略相邻边的相关性，在物理边规模 x 上，固定 pair 的初始长度期望为 `x p_k`。有限支持的 occupancy 模型是：

```text
E[V(x)] = Σ_k [1 − (1−p_k)^x]
        ≈ Σ_k [1 − exp(−x p_k)]
```

它自然包含开始快速发现新 types、后期趋于饱和的阶段。对字符初始化，支持集有限；对于固定 alphabet A 和固定 merge 数 M，pair 身份空间最多 `(A+M)²`，所以无限规模下的无界 Heaps 增长不能作为普遍渐近律。不同规模可能学到不同 merge 顺序，这又使稳定 `p_k` 成为需要验证的假设。

### 3.2 中位数与均值为什么不能共用一个幂指数

假设在有限适用区间内按频率排序后的**物理**长度满足 `h(r) ≈ a x (r+b)^−α`，并且 types 数 `V(x) ≈ K x^β`。则平均长度尺度为 `x/V ≈ K⁻¹ x^(1−β)`；第 q 个按 type 数量取的长度分位数近似：

```text
Q_q(x) ≈ a x [ (1−q)V(x) + b ]^−α
       ∝ x^(1−αβ)     （忽略 shift 的局部区间）
```

如果 type 数增长刚好抵消固定 pair 的次数增长，`αβ≈1`，中位数便近似常数。这个理想模型里频数分布的尾部 `Pr(h>z) ∝ z^(−1/α)`，其分位数由低端截断和 α 决定；它不要求最大值或均值也保持不变。

因此理论没有规定“cutoff 必须按 raw N 的某个正幂增长”。可以检验 `Q_q = a E_u^γ`、带有限支持饱和的 occupancy 模型、以及 empirical CDF 自适应阈值；不能在测量前替这些模型决定 γ。用两个规模只能算局部 log 斜率，无法检验曲率或饱和，至少需要三个规模。

Heaps 的有限尺寸偏差本身已有研究；即便理想 Zipf 指数恒定，真实有限规模的 vocabulary 曲线也不一定是严格幂律。[吕、张、周，2010](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0014139)。字符单元还可能表现出有限支持效应，不能直接套用英文 word-level 规律。[吕、张、周，2013](https://arxiv.org/abs/1202.2903)。

### 3.3 merge 数和语言如何进入

终点存活 posting 是多个出生阶段的混合分布。若第 b 阶段按长度 h 出生的数量为 `B_b(h)`，存活到 M 的概率为 `s_b(h,M)`，则：

```text
终点存活长度分布 ∝ Σ_{b≤M} B_b(h) s_b(h,M)
全量 Bump 请求量 = Σ_{b≤M,h>2} B_b(h) · 4 max(h,4)
```

merge 目标同时改变出生分布和退休筛选，不能只在输入 bytes 上加一个系数。BPE 跨 47 种语言的研究发现，早期被合并的重复模式与形态类型有关；该结果支持跨语言验证，未提供 J 物理历史 posting 的通用指数或容量阈值。[Gutierrez-Vasques 等，2023](https://aclanthology.org/2023.cl-4.5/)。

需要测量 `size × lifetime` 的联合分布。小对象不必长寿，大对象也不必短寿；“小且长寿用 slab，大且短寿用 heap”是候选分配策略，不是从 Zipf 推出的事实。终点 survivor 的寿命只有下界，是右删失数据；不能把它当成已经观察完整的寿命。

## 4. 将理论变成可解释的 cutoff 公式

对容量 c 字节的小 posting 使用训练期间不回收的 arena、其余仍使用 heap。令：

```text
A_{≤T}(t) = 到时刻 t，容量≤T 的累计分配 payload 字节
L_{>T}(t) = 时刻 t，容量>T 的当前存活 payload 字节
P(T)      = max_t [ A_{≤T}(t) + L_{>T}(t) ]
```

当 growth=0、对象不在两个 allocator 间迁移时，这个 replay 精确计算 posting payload 的混合峰值；对应额外保留量为每个时刻已退休的 `≤T` payload。Bump backing chunks 的分配余量与 heap metadata 需要另外实测。每 batch 一个同步快照能得到在这些检查点上的峰值，只有同时覆盖 batch 内的重要生命周期事件，才能称为全阶段精确峰值。

### 4.1 先判断阶段余量，再决定是否释放

如果目标是在原来整次训练峰值内运行，预算不是终点 live payload，而是全程峰值。用概念上的阶段量表示：

```text
B(t) = 阶段 t 的其它驻留内存 + baseline live posting 驻留内存
P0   = max_t B(t)
H(t) = memory_budget − B(t)
```

arena 在某阶段多保留的**驻留**空间只要小于 H(t)，就不会突破预算；取 `memory_budget=P0` 时也不会提高整次 peak。初始化 route/radix 大临时缓冲区释放后，即使 merge 后半段累计退休了很多短 posting，仍可能有大量阶段余量。此时逐个释放它们带来的 live payload 减少未必有用户可见的内存收益，而 free 本身会增加时间成本。

对候选 T，payload replay 得到的增量是 `retired_{≤T}(t)`，它可先筛出容量明显过大的方案。但不能直接把这个数加到 baseline RSS：普通 allocator 已释放对象仍可能留在页内，arena backing 也可能只有部分页面被触及。阶段 RSS/HWM 与 allocation replay 联合使用；准确的 RSS 因果结论仍要实际候选运行。只有 initialize 后和每 batch 的采样时，既不能精确拆出 B(t)，也不能排除两个检查点之间的短暂峰值。

因此，激进 merge 造成许多短 posting 退休，并不自动意味着需要 pool 或大幅降低 cutoff。若这些退休空间始终落在初始化峰值释放后的余量里，全量 Bump 可能是合适选择；阈值敏感性应以完整训练时间和全程资源预算判断。

阈值可由预算决定：

```text
T* = argmax_T 估计减少的分配/释放时间(T)
     subject to B_other(t) + backing_arena_{≤T}(t)
                + heap_live_{>T}(t) ≤ memory_budget
```

在“每个小 allocation 的收益大致相当”这个需要核验的近似下，目标可以暂用覆盖的 allocation 数；随后只对少数能改变选择的 cutoff 做完整训练计时。`T*` 是 `E_u、M、预分词、分布、机器预算` 的函数，其参数可以是用户可理解的内存预算，无需先选择一个语言魔法系数。数据规模改变时，分布 replay 会自然改变 T*；若全部 posting 放入 arena 已满足预算，T* 可取无限大。

Pool/slab 的资源式不同：每个容量 class 的可复用槽位需求取同时存活对象的峰值，而非累计出生数，还加 pages 内碎片。Generational arena 能回收整代的前提是这一代全部对象退休；“多数短寿”不能保证整块 page 可以释放。三者不能共用 Bump 的 retained-payload 曲线来宣称收益。

初始 CDF 可以作为早期预测，但后续每批 birth 可能改变分布。廉价策略可在已确定的容量 classes 上统计出生、退休并在线受预算约束；它本身的维护成本也需要测量。固定 16/32/64 等 size classes 是实现机制，策略 cutoff 可以自适应，二者并不冲突。

## 5. 最小可拟合的真实 J 诊断设计

主线程计划的诊断预算是：4 语言 `en/zh/de/ja × 1/4/16 MiB × 实际 4k/16k rules` 共 24 次 none；再加 en/zh 同规模、同规则的 whitespace_split 共 12 次。它们用于模型和内存曲线，不是正式性能排名。

保持固定 Wikipedia revision、同一种段落抽样方式、min frequency、线程配置和代码版本。每组固定**实际 merge 数**；目标词表要加上该输入的 initial alphabet 大小。因 floor 提前耗尽、无法达到规则预算时应记录 actual M、把样本视为不完整比较，不伪装成同 M。规模样本若不是严格嵌套，应记录各自 hash 和选样差异。

必要输出：

1. 输入的 U、物理/加权 N/E、unique/weighted piece length 分布、weight 分布及 ESS。
2. 初始与终点 posting 的精确长度/容量 CDF、分位数、最大值；inline 和 heap 分开，也保留全体分母。
3. 每个容量 class 的累计 allocated/retired count、payload bytes、physical lengths；initial births 单列。
4. 每 batch 的累计 allocation 与当前 live capacity；检查点应覆盖 selected-list drop 与 births install，或者给出峰值包络。
5. birth/retire 的 rule age 分布，selected/floor/终点 survivor 分开；survivor 标记右删失。
6. 累计成功物理 rewrite S，校验 `P_birth ≤ E_u+2S`；floor 丢弃和 growth 次数。

只有两个规则预算时，可以比较 M 对曲线的影响，不能同时拟合多个 M 指数、饱和值和语言 interaction 参数。优先对每个 PT、每个固定 M 拟合最简单的局部 `log Q_q ~ log E_u`，同时查看 1→4 和 4→16 的斜率是否接近；分位数常数、线性及局部幂律作为候选。用多个语言的留一语言预测误差检验是否需要语言专属参数；只有同语言拟合好、跨语言明显失配才引入语言系数。

allocator 决策的最终验证使用预算 replay 和完整训练，而不是选 log-log R² 最大的模型。若一段宽阈值区间覆盖类似 allocation 数、退休保留量都在预算内，再对区间两端做小预算正式计时，才能说参数不敏感。当前尚没有跨语言训练的阈值敏感性结论。

## 6. 当前已知的真实全量 Bump 对照

现有 512 MiB 中文 none、目标 50k/min2、init4/merge4 的诊断：累计 posting heap requests 9,323,712 次，payload capacity 1,403,805,436 B；结束仍存活 809,100,424 B；退休空间保留 594,705,012 B。4 个 TLS Bump 共 51 chunks、backing capacity 1,996,817,216 B。同期标准 allocator 与 Bump 各一次，训练 24.351→19.126 秒；全程 HWM 4.4337→4.4346 GiB，差 936 KiB。来源：[实测报告](results/j-bump-retain.summary.md)。

这说明当前这一个工作负载的全量 arena 已可行，不支持在测量前断言必须切掉大 posting。退休请求字节增加 73.50% 与 HWM 几乎不变并不矛盾：阶段峰值取最大值，backing capacity 不等于驻留 RSS。该轮没有阶段 RSS 时间序列；后续同 binary 阈值实验已补齐阶段采样，确认中文 512 MiB 的峰值在初始化返回前建立，详见 [阈值计时报告](POSTING_ARENA_THRESHOLD_REPORT.md)。精确碎片贡献仍未分离。

## 7. 独立初始静态分析

数据、范围和结果见 [posting-threshold-static.json](posting-threshold-static.json)。解析 Wikipedia revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` 的 en/zh/de/ja 1/4/16 MiB 本地文本，按真实 runner 的 line/Unicode whitespace 语义统计去重片段和初始字符 pairs；没有执行 BPE merges，没有限制 alphabet，没有 normalize。floor2 用 weighted pair frequency 筛选。JSON 中每个输入记录 SHA256 和实际字节数；静态解析秒数仅诊断开销，不是 tokenizer 的运行时间。

初版静态分析耗时约 163 秒；按实际 runner 保留行尾后重算全部 12 个 none 组合的校正过程耗时 114.2 秒，串行单核、nice 10，没有训练，也没有与正式计时重叠。whitespace_split 的旧、新片段 Counter 在全部 12 个输入上精确相等，因此沿用其统计。4 种语言的 1→4→16 MiB 文本均已逐字节确认严格前缀嵌套；从固定 shard 中按 article/paragraph hash rank 选样，而非随意取第一个百科文章。下表中的 median/max 针对 weighted frequency≥2 的**初始 pair types**，包含内联，长度单位是 u32 位置数。

| 语言、预分词 | median：1→4→16 MiB | max：1→4→16 MiB | 16 MiB `E_u/E_w` | 16 MiB unique 片段长度 median |
|---|---|---|---:|---:|
| en none | 11→10→8 | 24,715→99,619→394,771 | 99.65% | 215 |
| en whitespace | 7→7→6 | 4,710→12,563→32,448 | 17.32% | 8 |
| zh none | 3→3→3 | 3,133→13,949→47,248 | 94.77% | 66 |
| zh whitespace | 3→3→3 | 2,312→8,462→31,539 | 88.02% | 11 |
| de none | 12→11→10 | 29,604→118,409→472,284 | 99.61% | 230 |
| de whitespace | 8→8→7 | 11,281→32,634→91,356 | 26.47% | 9 |
| ja none | 3→4→5 | 2,137→8,216→32,826 | 99.59% | 82 |
| ja whitespace | 3→4→4 | 1,989→8,128→31,555 | 94.22% | 12 |

最大值都明显增加，中位数分别下降、近似不变或小幅增加；用同一个正幂缩放这些分布缺少依据。尤其 whitespace_split 后的去重：en 的物理边仅为加权边的 17.32%，zh 为 88.02%。这不是单纯 UTF-8 字节差异，边数已统一用字符/位置计算。

从 1→16 MiB 算输入 `E_u ∝ bytes^β_input` 的两端局部指数，whitespace_split 分别为 en 0.735、de 0.785、zh 0.982、ja 0.988；none 各语言约 0.991–1.000。这只描述此嵌套输入的物理边增长，不是已拟合的终点 posting 指数，不能用于外推超大 Wikipedia。

同一个 ≤32 B cutoff 对 16 MiB **初始堆 posting** 的覆盖：

| 语言、预分词 | allocation 数量覆盖率 | payload capacity 字节覆盖率 |
|---|---:|---:|
| en none | 38.50% | 0.08% |
| en whitespace | 42.35% | 0.67% |
| zh none | 65.27% | 11.36% |
| zh whitespace | 66.07% | 13.40% |
| de none | 36.39% | 0.08% |
| de whitespace | 40.43% | 0.39% |
| ja none | 54.79% | 5.53% |
| ja whitespace | 55.64% | 6.20% |

同一 cutoff 对语言的 allocation 覆盖不同；大量小 allocation 还可能只占很少 payload。优化 free 成本主要关心次数，而释放大 buffer 对容量的影响更大，两个目标需要共同看。这里没有测寿命和 merge 后的出生分布，所以不能据此给出“小对象池胜出”、语言专属最优 cutoff 或阈值不敏感的结论。

复现统计步骤：真实 runner 的自定义 `Lines<R>` 调用 `BufRead::read_line`，**保留**行尾 LF/CR；none 以含行尾的整行为片段，whitespace 按 Unicode White_Space 拆分；先 Counter 去重得到 w_i，再只遍历每个 unique 片段一次统计 char pair，物理计数加 n、weighted 计数加 w_i·n，最后按 weighted floor2 筛选。Rust White_Space 字符集合与 Python 默认 `str.split` 不完全相同，本次使用显式 Unicode 字符集合，避免 Python 额外分割 U+001C–U+001F。片段 unique length 分位数按 type 计数，而非按重复次数加权；JSON 中存储这一口径。


### 行尾语义校正记录

初版未提交统计把自定义 `Lines<R>` 误当成标准 `BufRead::lines()`，在 none 模式删除了 LF/CR。本版已经替换全部 none 数据并更新上述表格；JSON 的 `correction` 保存原因、旧文件 SHA256 和重算耗时。校正后 en1m-none 精确匹配真实 J 初始统计：`N_unique=1,042,596`、`E_unique=1,039,317`、`U=3,279`、floor2 pairs `3,048`。旧版 N/E 相差恰好 3,279 个行尾字符，并遗漏行尾相关 pairs。whitespace_split 结果不受此次行尾校正影响。理论中的 `L_i` 始终取真实 feed 片段长度，包含 none 片段的行尾。
