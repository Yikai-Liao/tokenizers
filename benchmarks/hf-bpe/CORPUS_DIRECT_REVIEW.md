# 候选 C：alphabet 与直接语料写入独立审查

## 范围与结论

本轮只读审查候选 C 相对候选 A 的 Rust 差分，没有构建、执行测试或运行 benchmark。在既有 plain、计数及地址范围约束内，没有发现阻塞正确性问题，可进入计划中的关键大语料测量。重点复核了无限 alphabet 的 ID 等价性、字符表、`MaybeUninit` 转换的安全条件及线程池屏障。候选 A 的词序、跨块权重、活跃 ID 和容量推导见 [CORPUS_PARALLEL_REVIEW.md](CORPUS_PARALLEL_REVIEW.md)，后续 merge 协议见既有 [REVIEW.md](REVIEW.md)。

worktree 为 `/root/code/tokenizers-worktrees/corpus-direct`，分支 `bpe/corpus-direct`；HEAD 为 `fbdc0b2bf736ffdc12aed3d0c3e0aba4a0004caa`，父提交为 `98ca7fc1c258d0661177d3aae15cca71536c3df4`。核对时工作树干净。下列路径相对 worktree，内容与 HEAD 相符：

| 文件 | SHA256 |
|---|---|
| `tokenizers/tk-train/src/trainers/bpe/indexed/parallel/alphabet.rs` | `82bfdedfd294ced41f05f79c3086cfa88caecec305a8de0fba8832820c892c4f` |
| `tokenizers/tk-train/src/trainers/bpe/indexed/parallel/corpus.rs` | `3637ae3ca47e4ca92adc07d017fbdef9a651a9e017c0644198d936c28c2ab9b5` |
| `tokenizers/tk-train/src/trainers/bpe/indexed/parallel.rs` | `bb6453be94759f65df95dcbc48b3ff836c01b56c7f96ea20e076b2de0a8b914e` |
| `tokenizers/tk-train/src/trainers/bpe/indexed.rs` | `acf95ecbc3284fe1eefa00d35b29887819d1299659cecd0199df6729ceb3085c` |
| `tokenizers/tk-train/src/trainers/bpe/mod.rs`（未修改） | `8c5ad11abcdb766ff6df1fee14c2df4bf2795c477e64b41d200497567e96c5c4` |

差分仅修改 `BPE_VARIANT.md`、`indexed.rs`、`parallel.rs`、`parallel/corpus.rs` 并新增 `parallel/alphabet.rs`。owner 路由、heap、合并计划、delta 与提交算法未修改。

## 一、无限 alphabet 的字符集合与 canonical ID

`alphabet.rs:14` 对全部 `limit_alphabet=Some(...)` 直接调用原 `compute_alphabet`，包括限制大于实际字符数的情况。原频率裁剪、同频边界的 hash 遍历和 forced-alphabet 规则均沿用原实现，没有重新实现选择器。

None 路径只需字符存在集合：原实现会把每个 corpus 字符放入 alphabet，即使词权重为零；随后加入初始 alphabet；最终按 codepoint 排序，再跳过 `ids` 中已有的 token。新实现逐字符设置私有存在 bit，OR 归并后加入初始 alphabet，再按 bitmap 下标及最低 set bit 枚举，恰好产生同一递增 codepoint 序列。

special 插入仍发生在 alphabet 之前。两种实现都在已有 `ids` 时跳过该字符，因此单字符 special 保留原 ID，多字符 special 也保留其预留 ID。词权重不影响无限 alphabet 的最终集合，零权重词的字符不会丢失。空 map 仍可加入 forced alphabet；没有 corpus 与 forced 字符时不会插入任何字符。

`CODEPOINTS=0x110000` 覆盖整个 char 数值域，长度可被 64 整除。所有置位来源均为 Rust char，所以不会设置 surrogate 位或域外位；`char::from_u32(...).expect(...)` 对这些 set bits 成立。workers 为零在私有配置入口被拒绝，因此 chunk 的除法有效。

## 二、字符表与 count/fill 一致

`alphabet.rs:70` 建立全域 u32 表，默认 NONE，只把 `ids` 中恰好一个 char 的 key 写入对应 codepoint。原 corpus 查询使用 `c.encode_utf8(...)` 作为完整 key，这个 key 也恰好一个 char。因此表查询与原查询等价：单字符 special 被纳入，多字符 special 和空 token 不会意外成为字符 ID。映射不依赖 map 遍历顺序，因为不同单字符 key 的 codepoint 不同。

字符表在 corpus 规划开始时构造，之后不可变。有限 alphabet 的 measure 与 fill 均读取它，使用同一个 `id != NONE` 条件；None 路径的 measure 可以直接数 chars，因为上一节已证明全部 corpus 字符在 `ids` 中。char 索引始终小于表长。

表中 ID 来自既有 vocabulary 域，且 NONE 保留为 separator；窄 slot 的选择仍按完整 strings/vocab 大小判定，未修改。region 边数、加权溢出检查、词顺序、length 活跃归并和 block 权重重建的逻辑与 A 相同。字符表只替换逐字符 UTF-8 编码与 hash 查询。

## 三、最终未初始化分配的安全条件

`corpus.rs:179` 创建 `Vec<MaybeUninit<C>>`，长度为已经 checked 的精确 capacity。`resize_with(MaybeUninit::uninit)` 允许长度中的元素尚未持有有效 C；它不要求为最终 corpus 先写一遍默认 C。起始槽零单独写入 `C::encode(NONE)`。

`fill` 对 `[1..]` 递归执行 `split_at_mut`，每个 leaf 得到一个独占连续区间。规划 slots 为该 region 的保留字符数加每词一个 separator；count/fill 等价保证每次递增 position 都写入下一槽，没有跳槽或重复写。每个 leaf 的末尾普通 assert 检查 `position == slots.len()`，所有 Rayon joins 完成后才继续。

`corpus.rs:194` 的转换满足以下必要条件：

- 槽零和全部其余长度内元素均已由 `C::encode` 初始化；空 map 只有已写入的槽零。
- `MaybeUninit<C>` 与 C 具有相同大小、对齐与 allocation layout，cast 不改变分配地址。
- `Vec::from_raw_parts` 使用原 pointer、length 和 capacity；容量中长度以外的元素仍无需初始化。
- 原 Vec 放入 `ManuallyDrop`，所有权只转移一次；返回的 `Vec<C>` 负责释放原分配。
- conversion 前没有读取未初始化的 C。若填充、索引或 coverage assert panic，就无法到达 conversion；原 `Vec<MaybeUninit<C>>` 不会对未初始化元素执行 C 的 drop。当前四种 Slot 为整数或整数原子。

因此该私有 unsafe 不创建未初始化的 C 引用，不引入别名写入，也不重复释放。转换后才能供 pair 初始化和 merge 读取。普通与原子 Slot 都通过 `C::encode` 创建有效值，原子路径不依靠 Relaxed 操作发布未初始化数据；阶段完成由 joins 保证。

安全推导依赖完整覆盖，必须与 region 规划和 fill 一起维护。若改动分段、保留规则、提前返回或异步执行，需要重新证明转换前每个槽都已初始化。

## 四、线程池与公开接口

merge 池和可选初始化池现在在 alphabet 之前建立。alphabet、corpus 构造和 pair 初始化均同步在初始化池执行；None 或与 merge workers 相同的配置只创建 merge 池。`Some(n)` 且 n 不同才创建额外初始化池，owners 数与后续 merge workers 仍由原 `config.workers` 决定。

alphabet install 返回前完成 bitmaps 归并和 IDs 插入，再判定最终 Slot 类型。corpus install 返回前完成写入、转换和 boundary metadata；pair 初始化返回后才进入 merge。没有阶段之间的并行共享读写。池的生命周期覆盖整次训练，且后续 coordinator 仍由原 merge 池的 install 执行。

`bpe/mod.rs` 与父提交的差分为空。原公开 `BpeTrainer`、builder、serde、tuple 返回及非空 affix fallback 未改变。新增 scratch/table 字段只属于私有 stats。

## 五、临时内存与计时口径

最终 corpus 仍只有一个分配，worker 直接初始化自己的最终区间。转换复用同一个 allocation，没有完整语料复制，也没有额外的每 worker corpus。

字符表长度为 1,114,112 个 u32，报告按实际 capacity 计字节；当前请求大小为 **4,456,448 字节，即 4.25 MiB**。表、词引用、regions 和局部填充结果在 corpus 构造返回前释放，字符表不会留到 pair 初始化或 merge。它的容量单独记录为 `character_table_bytes`，未加入既有 initial core 布局合计。

无限 alphabet 每个并行 chunk 保存一个 139,264 字节（136 KiB）位图，chunk 数最多为初始化 workers；归并另有一个同尺寸 bitmap。`alphabet_scratch_bytes` 统计这些 bitmap capacities、词引用数组 capacity 和 bitmap Vec 目录 capacity。它描述这些命名临时数组同时存在时的容量，不包含 allocator 元数据、线程栈或 vocabulary 容器，不能替代 RSS。有限 alphabet 分支返回零表示未统计原选择器的临时容器，并不表示该路径没有临时内存。

`alphabet_ms` 包含本次同步 install、无限集合扫描/归并/ID 插入或有限原选择器；池建立在计时开始之前。`corpus_measure_ms` 现在包含字符表构造；allocate 不再包含默认 C 写入。`corpus_fill_ms` 继续包含最终 slice 写入、conversion、block/lengths 元数据归并；临时容器在返回时的释放不包含在字段赋值之前的 fill 截点内。累计 `tokenize_ms` 仍包含整个构造阶段和池建立，不与子计时再次相加。

## 六、测试证据

本轮核对了新增 alphabet 测试源码：None 与原实现对比完整 ids/strings，含零权重词、forced 字符、最高有效 codepoint、单字符及多字符 special，覆盖 1/4 worker。Some 路径检查保留 special 和字符表映射；裁剪的实际算法等价性由直接调用原选择器保证，测试没有跨独立 hash seeds 强行比较同频裁剪结果。

候选 A 的构造测试继续以原 hash 查询生成预期值，对比四种 Slot、1/4 worker、过滤/不过滤、Unicode、空输入、完全裁剪词、稀疏与预留 ID，并以小 blocks 逐地址核对权重。本轮另核对既有 1500 case 差分及专用初始化池测试仍在源码中。主任务报告共 42 项库测试通过；本次独立审查未重跑这些测试。

后续测量应继续核对原 API 模型签名与实际 RSS。性能收益及选择哪一版本作为候选 B 的 parent，由本轮锁定源码的测量结果决定。
