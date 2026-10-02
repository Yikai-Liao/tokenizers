# 同字节大小中英文 BPE 训练差距：最终 BLOCK freeze 分析

本轮已完成：从干净 BLOCK worktree 冻结构建、两份 16MiB 输入各一次 census、正式 NONE 对照、优化 DWARF 验证、每语料一次 perf stat 与一次 cycles 采样，以及有限源码/指令解析。共完成 **8 次 native 调用**，其中一份与编译重叠的计时保留但排除正式比较；其余 7 份用于正式或诊断证据。没有追加 native、采样或 512MiB 语料扫描。

## 1. 结论：约两倍差距来自不同的训练工作量与成本分布

最终公共默认 **posting bits32 / flat32 / NONE / N4/init4 / V30000/min2** 下，英文训练 **2118.011 ms**、中文 **950.019 ms**，单次同版本对照为 **2.229×**。英文多用的 **1167.992 ms** 中，merge 阶段多用 **1080.523 ms，占差额 92.511%**。这是实际墙钟阶段边界的差额分解。

相同字节大小没有给训练器相同的工作。英文唯一行 Unicode 符号和初始边数为中文的 **2.683× / 2.690×**，实际 merges 为 **1.343×**，选中 posting 访问为 **6.651×**，成功改写为 **4.896×**。英文更大的合并扫描和改写工作，对应主要 prepare、rewrite 与 commit 成本。中文 alphabet 和初始字典条目更多，初始安装及按组处理成本抵消了一部分差距；批次数很接近，因此最终时间不会按 visits 的 6.651× 线性放大。

perf 定位了端点校验、邻居聚合、ledger 查询、birth 链遍历与 posting 组装。全命令英文用户 instructions 为中文的 **2.386×**，cycles 为 **2.211×**，英文 IPC 反而略高。更多执行工作是这组数据的主要解释；较大工作集与更多通用 cache misses 是相关成本证据，现有采样不能把墙钟差额精确拆成“算法、cache、allocator”百分比。

历史 AFFIX v4 的公平 NONE 对照为 2.030×，本次正式对照为 2.229×，两份 record 诊断计时为约 2.0×。这些单次结果支持“这对输入约两倍”，不支持稳定到小数点的语言常数，也不支持把版本间比例变化归因于 BLOCK。

## 2. 最终来源，以及与 bits16 BLOCK 测量的区别

源码是干净 worktree `/root/code/tokenizers-worktrees/block-fused-prepare`，完整提交 **`1959202f30673fc21e68ab2868ba40213062c409`**，包含前置 affix 最佳默认。构建从此 worktree 冻结复制，未从 ROOT HEAD 派生。ROOT 归档 HEAD 及父代理后续文档提交不是本轮源码版本标识。

| 构建 | label / manifest | 实际 binary SHA256 | ELF Build ID |
|---|---|---|---|
| 正式 release，公共 bits32 | `language-gap-formal32` / [manifest](.build/native-language-gap-formal32/build_manifest.json) | `3499c17aabf1fa09b2df5ceb0de019f0b15436116249e9461ddf286d18e80214` | `3f03047cac3afbe333740b0daeaac4b2e195b186` |
| 独立优化 DWARF，公共 bits32 | `language-gap-dwarf32` / [manifest](.build/native-language-gap-dwarf32/build_manifest.json) | `caa86aa235299af143fe324210f94625652d4b27bfea840fafdc91fd61cc0823` | `32b7e7e78c061158e1681e47bb3fe06655b5e2ae` |

正式 binary 的准确路径：

`/root/code/tokenizers/benchmarks/hf-bpe/.build/native-language-gap-formal32/target/release/hf-bpe-native-language-gap-formal32`

两个 16MiB 案例均为 `parallel_u32_flat32`、4 B corpus 槽、`initial_blocks=1`、merge/init workers=4/4、`fused_block_batches=0`。没有进入新 BLOCK 多块 fusion 分支。父代理 `native-block-fragments-pooled` 强制 bits16 字典路径及其百分比收益不能充当本报告的语言对照。本报告也不覆盖父代理后来新增的完整 ZH512 posting 宽度矩阵；上面的 release binary 可作为其生产 flat32 参考。

两组均为整行 feed，保留 LF，没有 affix、alphabet 限制、强制 alphabet 或 special token。最终完整模型 SHA 与既有模型匹配：

- EN：`2958b29ca7214b9119b281391055655df69db0c79c86b449fd153169197304cc`。
- ZH：`e19702e3a081378577aba1e140fb7b9ac91b8d97d2f5e679b9d9ec14603d1899`。

feed AHash seeds 固定为 11/13/17/19；owner 内部 AHash 表仍默认随机。pair→owner 分区是确定的 ID 混合函数，和 owner 哈希表的随机布局是两件事。固定输入和 feed seed 没有消除所有物理布局与计时波动。

## 3. 输入核验：UTF-8 密度、整行边界与重复权重

[结构化最终比较](results/language-training-gap/final-none32.json) 保存输入 SHA 与 census。每输入只做一次 census，不读取 512MiB 来源；census 的 bytes、unique_words、initial_symbols、initial_edges、alphabet 与 native 逐项相等。

| 输入指标 | EN16 | ZH16 |
|---|---:|---:|
| bytes | 16,776,969 | 16,776,351 |
| SHA256 | `41b20e33253a4db8c66b95536cbadc86c79072e81eb9e5cd59ef136490abdec9` | `ebd64699a917bb0982e9b64d302d94cc08057f43d16bcb5efc041594684c12a6` |
| 总行数 / 唯一整行数 | 53,285 / 52,453 | 34,628 / 34,588 |
| 首次以外的重复行次数 | 832，1.5614% | 40，0.1155% |
| 总行 Unicode 符号 | 16,709,038 | 6,207,574 |
| 唯一行 Unicode 符号 | 16,649,633 | 6,204,526 |
| 全输入 bytes / Unicode 符号 | 1.004066 | 2.702562 |
| alphabet | 2,124 | 9,241 |
| 唯一行平均 Unicode 长度 | 317.420 | 179.384 |
| 唯一行长度 P50 / P90 / P99 / max | 215 / 676 / 1510 / 7992 | 136 / 346 / 858 / 6119 |
| 权重为 1 的唯一行 | 52,061 / 52,453 | 34,555 / 34,588 |
| 最大整行权重 | 59 | 4 |
| 未加权初始边 | 16,597,180 | 6,169,938 |
| 加权初始边 | 16,655,753 | 6,172,946 |

两输入全部以 LF 结束，没有 CRLF 或未终止末行。runner 的 `read_line` 保留 LF，process 返回完整字符串 `vec![s.to_owned()]`；`unique_words` 实际为唯一整行数，不是自然语言分词后的词数。空格和 LF 都参与 symbol、pair 与 merge。

feed 聚合重复整行为权重；corpus builder 只展开唯一行，频率使用权重，不会为重复行复制槽。英文权重有 17 个取值，中文 4 个，完整 histogram 在 JSON。`weight_interval_count=17/4` 是物理区间统计；本次 census 才直接核验权重分布，不能由区间数推出查询命中率。

UTF-8 密度解释初始符号差异的大部分：中文每符号约 2.70 B、英文约 1.00 B，同字节输入的总 Unicode 数比约 2.69×。整行重复在这两份输入上只小幅改变展开数量。但训练 scans 与成功 rewrites 已增长到 6.651× / 4.896×，所以编码密度不是完整归因。

本对照同时改变语言、alphabet、行数/长度、重复和字符组合分布。结论适用于这些输入和参数，没有做只改变“语言”的受控语料实验。

## 4. 墙钟：主要增加在 merge，初始化有不同字典成本

正式 `train_ms` 不含 feed，也不含模型 SHA 与进程退出。initialize 包含 tokenize，initial count 包含 radix/group/install，fused prepare 包含在 delta/merge 内，以下嵌套行不能全部相加。

| 阶段 ms | EN | ZH | EN−ZH | EN/ZH |
|---|---:|---:|---:|---:|
| train | 2118.011 | 950.019 | 1167.992 | 2.229 |
| initialize | 291.990 | 210.307 | 81.683 | 1.388 |
| merge | 1789.186 | 708.663 | 1080.523 | 2.525 |
| train−initialize−merge | 36.836 | 31.050 | 5.786 | 1.186 |
| feed，train 之外 | 78.755 | 108.432 | −29.677 | 0.726 |
| tokenize，initialize 子阶段 | 66.661 | 45.415 | 21.246 | 1.468 |
| initial route，initialize 子阶段 | 82.155 | 31.846 | 50.309 | 2.580 |
| initial count，initialize 子阶段 | 142.974 | 123.400 | 19.573 | 1.159 |
| radix sort，initial count 子阶段 | 64.399 | 41.401 | 22.997 | 1.555 |
| group count，initial count 子阶段 | 21.166 | 14.918 | 6.248 | 1.419 |
| posting install，initial count 子阶段 | 57.407 | 67.079 | −9.673 | 0.856 |
| fused prepare，delta/merge 子阶段 | 1130.620 | 368.337 | 762.283 | 3.070 |
| rewrite，merge 子阶段 | 133.618 | 22.490 | 111.128 | 5.941 |
| commit，merge 子阶段 | 366.267 | 201.147 | 165.120 | 1.821 |
| select，merge 子阶段 | 81.243 | 58.728 | 22.515 | 1.383 |

initialize、merge、边界余项各占全训差额 **6.993% / 92.511% / 0.495%**。余项没有独立操作计时，不直接命名为析构或输出。

初始 route 先扫描有效边计数，再写出 owner stream，每个初始有效边在两遍各处理一次。这由代码结构和核验的 edges 推导，不是新增动态查询计数。英文 edge 和 route buffer 为中文约 2.69×，本次 route 时间为 2.58×；perf 也采到两遍循环。

中文 native `initial_pairs` 记录 340,092 个条目，英文 8,911；中文 initial owner pair table 为 17,301,568 B，英文 540,736 B，group buffer 为 6,291,456 / 196,608 B。中文边少但字典/分组的条目工作更多，posting install 反而多用 9.673 ms。`initial_pairs` 按训练器初始字典口径理解，不能当成 census 已枚举全输入所有 pair 类型。

初始化不按字符数量统一缩放，且只解释差额约 7%。中文 feed 多用约 30 ms，却位于训练边界之外，不能解释英文训练更慢。

## 5. 合并工作量：成功改写、无改写访问与 AA/批次

同 V30k 不等于同 merge 数。无 special/forced alphabet、`reused_ids=0`，每个 merge 激活新 ID；`30000−actual_merges` 与 census alphabet 完全一致。

| 合并指标 | EN | ZH | 解释 |
|---|---:|---:|---|
| actual_merges | 27,876 | 20,759 | 1.343× |
| posting_visits | 24,345,699 | 3,660,298 | 6.651× |
| visits / merge | 873.357 | 176.323 | 4.953× |
| 成功改写，独立诊断计数 | 13,447,190 | 2,746,772 | 4.896× |
| 成功改写 / merge | 482.393 | 132.317 | 3.646× |
| 访问后没有改写 | 10,898,509 | 913,526 | visits 的 44.766% / 24.958% |
| 推导剩余 active symbols | 3,202,443 | 3,457,754 | initial_symbols−成功改写 |
| 初始 symbols 被合并减少的比例 | 80.766% | 44.270% | 不等于文件压缩率 |
| batch_rounds | 1017 | 971 | 只差 4.74% |
| 平均 merges / batch | 27.410 | 21.379 | 不代替分布 |
| 单规则批次 | 70 | 86 | 其中 AA 为 53 / 76 |
| 非 AA 单规则批次 | 17 | 10 | 没有逐批停止原因 |
| fused_batches | 964 | 895 | 此配置其余 AA 走 plan/delta |
| 实际 max_batch_rules | 256 | 155 | 实际观测最大值 |
| AA posting_visits | 251,259 | 42,778 | 总 visits 约 1.032% / 1.169% |
| AA 成功改写 | 21,194 | 18,245 | 总成功改写的 0.158% / 0.664% |

诊断计数在批次边界记录 visits、AA 和 batch histogram，成功改写来自 fused Prepared 的有效 position 列表或 parity 处理后的 plans，没有给每次 load/hash 加计时。histogram 的数量及加权规则数等于 batch_rounds/actual_merges；stat、record 和已有中文 control 的工作量计数一致。

英文不仅多做 merges，每规则也作用于更多位置。`1.343× × 4.953× = 6.651×` 分解扫描数量，成功改写为 4.896×。英文删除更多边界，处理更多邻居删除、出生记录和 posting 组装，与 prepare/rewrite/commit 时间落点一致。

“没有改写”包含端点失效和 AA 重叠筛选等原因，不能全部叫 stale。除去 AA，没有改写的选中 posting 仍为 EN 10,668,444 / ZH 888,993，说明差异主要不在 AA。历史 parallel `stale_posting_visits=0` 没有可靠计量这些事件，这里使用独立 visits 与改写差额。

AA 批次在中文反而更多，但只占很小的改写工作。正式 plan 子阶段为 EN 7.052 / ZH 5.944 ms，是 AA 路径的一部分，不等于 AA 全成本。批次、计数和阶段证据不支持把约 1.17 s 的总差额主要归于 AA 或批次数量。

## 6. perf stat：真实全命令分母与缓存证据

每输入一次 `perf stat --no-scale`，四个硬件事件 `cycles:u,instructions:u,cache-references:u,cache-misses:u`，加 task-clock、context-switches、cpu-migrations、page-faults。已有 corpus/posting/heap 字节差异支持选择 cache 事件，本轮未测 branch-misses。

所有事件报告 `time_running=100.00%`，未观测 multiplex，没有 unsupported 事件。CSV 保存 raw count、运行 ns、运行百分比，解析保留 scaled estimate。EN 运行 ns 为 6,420,085,954，ZH 为 3,203,506,239，是继承 workers 的事件 CPU 运行量，不能当墙钟。

| 全命令事件/指标 | EN | ZH | EN/ZH |
|---|---:|---:|---:|
| 用户 cycles | 14,180,409,192 | 6,412,576,569 | 2.211 |
| 用户 instructions | 12,417,986,280 | 5,205,581,876 | 2.386 |
| 用户 IPC | 0.8757 | 0.8118 | 1.079 |
| 通用 cache references | 164,767,878 | 66,701,860 | 2.470 |
| 通用 cache misses | 105,681,834 | 30,989,680 | 3.410 |
| 通用 miss / reference | 64.140% | 46.460% | — |
| 通用 cache MPKI | 8.510 | 5.953 | — |
| task-clock | 6420.09 ms | 3203.51 ms | 2.004 |
| context-switches | 16,021 | 9,404 | 1.704 |
| cpu-migrations | 26 | 2 | — |
| page-faults | 48,228 | 25,467 | 1.894 |

**cycles/instructions/IPC 均为完整 native 命令及 workers，包括 feed、训练、模型摘要和退出；不是只对 train 的计数。** 硬件事件为用户态，task-clock 包含 system CPU。raw CSV：[EN](results/language-training-gap/en-final-stat.perf-stat.csv) / [ZH](results/language-training-gap/zh-final-stat.perf-stat.csv)。

同范围的恒等分解为 `2.386× instructions × (ZH IPC / EN IPC) = 2.211× cycles`。英文执行指令更多，整体 IPC 未比中文差；这比“英文字符本身执行慢”有更直接的证据支持。不同操作组合也影响 IPC，不能把恒等式变成逐操作因果比例。

cache-miss 及 MPKI 更高，与英文较大 corpus/posting 工作集和更多扫描相符。使用的是通用 cache 事件，没有精确 load sampling，KVM PMU 报告 `max_precise=0`；不能指定为某个 cache 层或某条 load 的 miss 次数，也不能算“cache 占墙钟多少”。context switches 也不能单独证明锁竞争。

## 7. 采样到操作：self、inclusive、真实指令与源码行

每输入一次 `cycles:u` 199 Hz、DWARF 8192 B、`--clockid mono`，采目标命令及 inherited workers。EN 1394 samples，perf.data 11,959,222 B；ZH 732 samples，6,532,670 B。周期分母为样本 period 累加：EN **14,783,597,892**，ZH **6,577,710,410**。它们来自独立 record run，不要求等于另一轮 stat 计数。

下面是 **全命令 period-weighted 样本周期占比**，不是墙钟占比。self 为实际 leaf 符号，包含内联于其中的操作；inclusive 为同栈包含此符号，祖先与子孙不可相加。

| 路径 | EN self | ZH self | EN inclusive | ZH inclusive |
|---|---:|---:|---:|---:|
| fused prepare worker closure | 38.969% | 19.617% | 57.253% | 40.017% |
| flat_commit::dense 主体 | 8.026% | 15.503% | 16.878% | 23.774% |
| dense posting assembly closure | 5.160% | 2.883% | — | — |
| aggregate::Scratch::birth | 5.813% | 2.755% | — | — |
| Prepared::apply worker closure | 4.188% | 1.350% | — | — |
| initial radix sort | 3.793% | 4.381% | — | — |
| initial owner_route count / fill 两个 self 符号之和 | 3.642% | 4.006% | — | — |
| malloc / realloc / cfree 已解析 self 之和 | 3.631% | 4.646% | — | — |
| leaf `[unknown]` | 9.613% | 13.160% | 不归因 | 不归因 |

`WorkerThread::wait_until_cold` inclusive 达 66.845% / 63.705%，它是 worker 执行框架祖先，不能称为等待占了这些周期或墙钟。四个 worker 各占总 period 约 20.8%–29.4%，main 为 1.72% / 4.65%；工作分布于真实 workers，没有据此证明严格均衡。

每输入最多 200 个主要 native leaf IP，按已锁定 ELF 的原符号地址+实际 offset 批量解析该诊断源码，并检查主要指令区域。见 [perf JSON](results/language-training-gap/final-perf-analysis.json)、[批量 addr2line](results/language-training-gap/sampled-addresses.addr2line.txt)、[指令证据](results/language-training-gap/instruction-evidence.json)。

| ELF IP | EN / ZH period 占比 | 实际指令及操作 |
|---|---:|---|
| `0x44f434` | 26.365% / 11.321% | `cmp %rcx,%rbx`，posting 端点校验循环；前一条 `0x44f431` 为 `mov (%rax,%rbp,4),%eax` 读 corpus[p]，随后校验边界和 token |
| `0x2cd368` | 0.806% / 2.001% | `movdqa %xmm1,%xmm2`，邻近 `movdqu / pcmpeqb / pmovmskb`；ledger.get_mut 的 control-byte probe |
| `0x2cd3e0` | 2.778% / 4.137% | `mov -0x8(%r9),%rax` 读 ledger frequency，后续才 `sub` 和 underflow 检查 |
| `0x2d64df` | 2.560% / 1.801% | `mov %edx,(%rcx)` 更新链头，前后读取 Node.next/position；birth 链遍历和有序 posting 组装 |

最热 IP 的 DWARF 内联链含 `atomic_load / Slot::token / fused_batch.rs:190`，反汇编实际却是 load 后的 bounds compare。Relaxed `Atomic<u32>` 读取在这里生成普通 mov。这确定主要成本区域为按 posting 地址读 corpus 并校验端点，不证明 cmp 本身耗费全部周期，也不证明全是前一条 load 的 cache miss。

同理，frequency 更新源码行最热指令是载荷读取，不能把周期全叫整数减法；Node 位置对应读 next、更新链头、返回 position 的依赖访问，不能按源码行数估算动态成本。

英文 prepare self period 约 5.761 B，中文约 1.290 B，绝对比例约 4.46×。英文 visits/无改写访问更多、校验区域更热、prepare 墙钟更长，三项证据一致。Scratch::birth、实际 corpus apply 和链组装也体现更多改写。中文较大字典及按组固定成本使 commit 在其周期中占更高比例；阶段时间与 sampled inclusive 各自保持分母。

## 8. 内存与分配：请求字节、对象数量、峰值分开

| native 内存/分配指标 | EN | ZH | EN/ZH |
|---|---:|---:|---:|
| corpus bytes | 66,825,724 | 25,030,436 | 2.670 |
| initial posting bytes | 66,352,028 | 22,572,732 | 2.939 |
| initial route buffer bytes | 132,777,440 | 49,359,504 | 2.690 |
| posting arena buffer 次数 | 547,058 | 529,831 | 1.033 |
| posting heap buffer 次数 | 37,232 | 18,004 | 2.068 |
| posting arena requested bytes，累计 | 20,893,180 | 18,245,292 | 1.145 |
| posting heap requested bytes，累计 | 141,764,204 | 17,492,524 | 8.104 |
| 正式 native VmHWM KiB | 276,028 | 151,096 | 1.827 |

英文多的是大 posting 累计字节，小 arena buffer 次数只多约 3.3%。8.104× 累计 bytes 不是 malloc 次数、同时驻留峰值或 allocator 耗时。这些 counters 覆盖 posting allocation session，不覆盖全部 Vec、hash table 和 scratch 分配。

已解析 malloc/realloc/cfree self 约 3.6% / 4.6%，不能解释整段差额；尚有未知 leaf 和内联调用，也不能宣称 allocator 总成本只有这些。较大内存工作参与成本已有证据，allocator 竞争和全部分配/释放时间尚未独立量化。

8 次调用均完成并匹配模型。primary/speculative posting session 的 arena requested=retired、heap requested=freed、buffers=frees 检查通过。实际线程峰值 5、进程 VmSwap=0。正式 EN/ZH 最低 MemAvailable 为 7,425,179,648 / 7,575,846,912 B；stat wrapper rusage 未出现大缺页。主机 pswpin 的小增量不等同目标进程 swap。

## 9. 诊断质量、开销与预算审计

### 实际优化 ELF 与溯源

正式 release 有 .symtab，无 .debug_info/.debug_line。独立诊断为 release opt-level3、debug2、strip none，同默认目标 ISA、依赖和 bits32；附加计数与 CLOCK_MONOTONIC 标记只在隔离副本。源码 hashes、patch、runner/lock SHA、command/env 均在 manifest。本任务没有修改父 builder/runner 脚本或生产源。

[ELF 验证](results/language-training-gap/elf-validation.json) 保留真实 SHA、Build ID、debug 段、符号、工具版本及隔离源码 hashes。实际地址 0x120f40 解析到此 binary 的 train_vocab / mod.rs:483，0x120f60 含 compute_alphabet / mod.rs:335 内联链；采样地址又以同 ELF 批量核验。没有将旧 binary 地址映射新源码。

rustc 1.98.1 / LLVM 22.1.8、perf 6.12.111、GNU Binutils 2.44，主机为 KVM 6 vCPU Xeon Gold 6140。perf 的 demangle 仍留 Rust v0 编码，最终用一次批量 `c++filt -s rust -n`；已统计 leaf 名称无剩余 Rust 编码。

### 样本丢失、未解析与栈

| 质量指标 | EN | ZH |
|---|---:|---:|
| samples / raw SAMPLE records | 1394 / 1394 | 732 / 732 |
| 记录的 LOST records / events / samples | 0 / 0 / 0 | 0 / 0 / 0 |
| 未解析 leaf 样本 | 139，9.971% | 93，12.705% |
| 未解析 leaf period 占比 | 9.613% | 13.160% |
| 空栈 / 少于 3 frames 的栈 | 0 / 61 | 0 / 30 |
| 含可识别进程/线程入口的 period 比例 | 67.724% | 67.085% |
| 捕获动态栈达到 8192 B 的样本 | 1312 | 633 |
| 选择的 native leaf IP 数 | 200 | 200 |
| 这些 IP 覆盖全命令 period 比例 | 75.860% | 72.104% |
| 所选 IP period 成功解析源码比例 | 99.595% | 98.199% |

LOST=0 指文件的 PERF_RECORD_LOST/LOST_SAMPLES，不是零未报告硬件丢失的证明。raw 数和 period 累加与 script 完全一致。动态栈达到 8192 B 不能单独证明截断，也不能排除截断。已解栈能显示 fused/commit/worker 祖先，足够解释主要路径，inclusive 没有被当成完整无损的所有调用成本。

CLOCK_MONOTONIC 训练窗口与 record clockid 对齐：EN 窗口 1367 samples / 14.529 B period，ZH 692 / 6.272 B，约覆盖全命令 period 的 98.278% / 95.346%。这些仅是**训练窗口样本周期估计**，没有推成墙钟 gap 占比。merge 92.511% 的墙钟差额来自第 4 节实际计时。

首次默认 inline 解码反复启动 addr2line 并报错，已取消。最终关闭 perf inline，再以每语言最多 200 个主要 IP 合并批量解析内联源码。最终解析符合 40k samples、128MiB data、top20、200 IP/语言上限，没有追加采样补未知 leaf。

### 实际调用与开销边界

| 调用 | train ms | 用途 |
|---|---:|---|
| EN formal 初次 | 2091.371 | 与诊断编译重叠，排除正式比较 |
| EN formal clean | 2118.011 | 正式对照 |
| ZH formal | 950.019 | 正式对照 |
| ZH diagnostic 未采样 control | 886.107 | 同 binary 开销参照 |
| EN diagnostic stat | 2314.960 | 诊断，非正式排名 |
| ZH diagnostic stat | 1066.311 | 诊断，非正式排名 |
| EN diagnostic record | 2181.863 | 诊断，非正式排名 |
| ZH diagnostic record | 1092.915 | 诊断，非正式排名 |

初次 EN 并行编译是执行失误，完成时间戳确认重叠。为守 8 次预算，补跑独立 EN 正式计时并省去英文未采样诊断 control。**英文采样开销未由同 binary control 独立隔离。**

中文 stat/record 相对同 binary control 的单次 train 差值为 +20.337% / +23.339%，包含工具影响和波动，不是稳定纯采样开销测定。中文诊断 control 比正式 binary 快约 6.7%，不称为 debug 加速。历史小 no-op 曾有约 12% 波动；本报告按大工作量差异、路径和计数解释约两倍，不据这些小对照推演精确因果比例。

## 10. 复核入口与剩余边界

[主脚本](analyze_language_training_gap.py) 支持小元数据比较、显式 census 和复用已有 census/perf summary；复用不读语料或重解 perf：

```sh
python3 benchmarks/hf-bpe/analyze_language_training_gap.py \
  --en-row benchmarks/hf-bpe/results/language-training-gap/en-final-formal-clean.jsonl \
  --zh-row benchmarks/hf-bpe/results/language-training-gap/zh-final-formal.jsonl \
  --final \
  --reuse-census benchmarks/hf-bpe/results/language-training-gap/final-none32.json \
  --perf-summary benchmarks/hf-bpe/results/language-training-gap/final-perf-analysis.json \
  --output benchmarks/hf-bpe/results/language-training-gap/final-none32.json
```

[隔离 builder](build_language_training_gap.py)、[采样监控](run_language_training_gap.py)、[有限离线解析](analyze_language_training_gap_perf.py) 保存方法。command/env、输入/模型/binary/manifest SHA、资源和原始 stdout/stderr/CSV/perf.data 均在 [结果目录](results/language-training-gap/)。正式/control 使用的父 run_affix_analysis.py SHA 已存在 environment；父后续修正该脚本不改变旧结果，历史脚本可由父 0e899a41 提交恢复。

已能串联的关系是：同字节输入产生不同 Unicode 和字典工作；同目标词表产生不同规则数和作用位置；更多扫描、无改写访问、邻居更新与组装对应英文更大的 merge 成本，中文的字典及按组成本使比例收敛到约两倍。尚未独立分开的量为精确 cache stall、全部 allocator 成本、非 AA 单规则逐批停止原因、未知 leaf 及具体每次 hash/load 延迟。主要工作量、阶段和真实指令证据足够交付本题，没有为这些细分项扩矩阵或加 native。

方法边界依据已阅读的 [performance skill](/root/.codex/skills/performance-optimization-draft/SKILL.md) 和 [案例及证据边界](/root/.codex/skills/performance-optimization-draft/references/cases.md)：真实 debug/地址、self/inclusive、阶段包含、未填零字段、事件分母、随机 hash、开销和有限预算。历史案例只用于方法边界，本轮结论以本 freeze、输入和记录为据。
