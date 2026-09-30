# 候选 A：并行语料构造独立审查

## 范围与结论

本轮只读审查候选 A 的 Rust 源码与父提交差分，没有构建、执行测试或运行 benchmark。在既有 plain 配置、计数范围和地址范围约束内，没有发现阻塞正确性问题，可以进入计划中的大语料测量。这个结论不证明性能收益；既有 batch、AA 和合并独占写入推导继续见 [REVIEW.md](REVIEW.md) 与 [WORKTREE_REVIEW.md](WORKTREE_REVIEW.md)。

审查 worktree 为 `/root/code/tokenizers-worktrees/corpus-parallel`，分支 `bpe/corpus-parallel`，HEAD 为 `98ca7fc1c258d0661177d3aae15cca71536c3df4`，父提交为 `b6a28768feb4af4181f76fc1fb3f78993644f5c9`。核对时工作树干净，以下文件内容与该 HEAD 一致。路径均相对 worktree：

| 文件 | SHA256 |
|---|---|
| `tokenizers/tk-train/src/trainers/bpe/indexed/parallel/corpus.rs` | `acc846589362e99948bbc3b56b6c4e3b95a5e3694a84ec43357b7fb511bf8292` |
| `tokenizers/tk-train/src/trainers/bpe/indexed/parallel.rs` | `49054c54cc9659d1cfb2f0cbedb6544bf34fccd7aa173c728bd7490afd852c0d` |
| `tokenizers/tk-train/src/trainers/bpe/indexed.rs` | `613609ff8ace8455617a2dd92654e09365b8ab646d60aff4ac450175dec68e70` |
| `tokenizers/tk-train/src/trainers/bpe/mod.rs`（未修改） | `8c5ad11abcdb766ff6df1fee14c2df4bf2795c477e64b41d200497567e96c5c4` |

差分仅修改 `BPE_VARIANT.md`、上述 `indexed.rs`、`parallel.rs`，并新增 `parallel/corpus.rs`。owner 路由、计数、heap、合并计划与提交代码未修改。

## 一、词序、容量与填充一致

`corpus.rs:100` 在不可变的 `wc` 上只捕获一次原 map 遍历顺序，保存词引用及权重。连续 `par_chunks` 的规划结果按 indexed iterator 顺序收集；递归填充使用连续前半、后半区间，返回时也按前后顺序拼接。线程调度不能改变词序、词内字符顺序或相邻边位置。

规划与填充始终读取同一个不可变 `ids`。当 `limit_alphabet` 存在，规划逐字符执行与填充相同的单字符键查询。不存在 limit 时，未修改的 `compute_alphabet` 会把全部 corpus 字符加入 `ids`，所以直接计算 `chars().count()` 与填充保留字符数相同。special 与初始 alphabet 的 ID 分配仍在该阶段之前完成。

每词占用保留字符数加一个 separator；总容量再加起始 sentinel。各 region 的累计大小先经过 checked addition，递归 `cut` 的部分和及 `base + cut` 因而不超过已验证的最终容量。`split_at_mut` 把唯一最终分配切成不相交的 slices。每个 leaf 只修改自己的 slice，最后检查写入数等于 slice 长度；所有 joins 完成后才进入后续 pair 初始化。普通 u32、u16 与对应原子 slot 均通过现有 `Slot::set` 写入，窄 slot 的 NONE 编码没有改变。

空 map 只生成起始 sentinel，直接跳过 `fill`，不会访问空 regions 的首项。空词和完全裁剪的词仍各自占一个 separator，且不产生边。词内字符全部被裁剪时也不会跨 separator 连成新 pair。

## 二、活跃 ID 与跨块权重

每个 leaf 用私有 bitset 记录实际写入的 ID；主线程在 join 后归并为 `lengths[id] = 1`。因此没有多个 worker 同时写 length 槽的竞争。未出现的强制 alphabet 字符和多字符 special 预留 ID 仍保持零长度；实际出现的单字符 special 依照原 `ids` 查询活跃，包括有限 alphabet 下仍保留在 `ids` 的 special。

词起点按原词序重建。词的结束位置为下一词起点减一，最后一词为 `capacity - 1`，均包含本词 separator。这个范围与旧串行循环逐字符、再 separator 创建 blocks 的范围一致。跨块长词创建的新块带该词的 `previous_weight`；下一词若在已有块内开始，局部 pivot 切换到新权重；若恰好在新块开始，新块直接带新权重。空词同样插入 pivot，并占用自己的 separator。

局部 pivot 保持递增，块内地址仍限定在原 bits=16/32 的范围；权重游标和 AA 空间顺序前提没有改变。uniform 权重由全部词权重判定，空输入为 None；与原 values 遍历的判定等价。

## 三、溢出、布局与线程池

每词权重先 checked 转为 i64，每词边数 checked 转为 i64，再 checked 相乘。region 内与 region 间的 weighted-edge 总和均 checked 相加。由于权重和边数非负，分组归并不会使原本溢出的总量变为有效值。`symbols` 和 `edges` 的普通加法也安全：各自每一步不大于已经 checked 的 slot 部分和，总归并不大于 checked 的最终容量。

block size 继续 checked shift，块目录长度继续限定为 u32。flat32 判断现在使用保留后的精确 `corpus.len()`，以前使用未裁剪字符容量上界。有限 alphabet 因此可能更早选择 flat，但实际最终地址仍满足原 flat 的 u32 范围要求；该选择不会改变语料顺序和权重语义。

`parallel.rs:452` 使用 `initialization_pool.unwrap_or(pool)` 同步运行构造。`initialization_workers` 不同于 merge workers 时，构造及后续初始化在同一个专用初始化池完成；相同或 None 时使用 merge 池。region 数的规划参数使用相同的初始化并发设置。该同步 install 返回、所有填充与元数据归并结束后，才进入 pair 初始化和后续 merge。owners 数和 merge 池仍由原 `config.workers` 决定。

公开入口所在 `bpe/mod.rs` 与父提交差分为空；`BpeTrainer` 字段、builder、serde、tuple 返回、非空 affix 的 generic 路由及公开训练接口均未修改。新增四个计时字段只属于既有私有 `IndexedTrainingStats`。

## 四、分配与计时的解释

最终 corpus 只分配一个 Vec，worker 直接写它的独占 slices，没有每 worker 完整 ID 数组及其复制步骤。不过本实现会串行执行一次 `resize_with(C::default)`，随后并行覆盖；单一最终分配不等于每个元素只写一次。

新增临时内存包括词引用/权重数组和词起点/权重数组，均随词数增长；每 region 的活跃 bitset 随 ID 数增长。region 数约为初始化 workers 的八倍。这些属于额外元数据，实际 RSS 仍需测量。

计时字段应按实际范围解读：

| 字段 | 覆盖范围 |
|---|---|
| `alphabet_ms` | 原 `compute_alphabet`；不包含此前的 special 插入 |
| `corpus_measure_ms` | 冻结词引用、并行保留数/边数统计、检查、容量归并和 uniform 判定 |
| `corpus_allocate_ms` | 最终 Vec 分配、串行默认初始化及起始 sentinel 写入 |
| `corpus_fill_ms` | 并行最终 slice 填充、词起点/活跃 bitset 生成、串行 block/lengths 归并 |

既有 `tokenize_ms` 仍是从训练 begin 到构造结束的累计时间，包含 special、alphabet、池建立、全部构造阶段与其它少量协调工作，不能与上述子计时再次相加。`corpus_fill_ms` 也不能直接当作仅字符解码的耗时。

## 五、已核对的测试证据及复核条件

新增测试源码对照原串行词循环生成期待 slots、lengths、symbols、edges 和每个地址的 weight。它使用小 bits=4 覆盖跨块长词、空词、全裁剪词、Unicode、稀疏 ID 和未活跃预留 ID，并覆盖四种 Slot、1/4 worker、filtered/unfiltered 与空 map。主任务报告原 40 项库测试及此新增测试通过；本轮未独立执行。原接口模型签名和大语料性能比较仍由本轮后续测量验证。

修改 alphabet 缩减规则、词序、IDs、region 范围、块元数据重建、同步 pool 屏障或 flat 地址判断后，需要重新审查相应推导并更新 commit/SHA。当前结论仅覆盖本报告锁定的源码。
