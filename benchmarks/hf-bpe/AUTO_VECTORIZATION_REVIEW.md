# G1 / G2：读取视图与分块过滤独立审查

## 范围与结论

本轮只读审查 G1 的独占读取阶段及 G2 的 128 项过滤分块，没有构建、运行测试或 benchmark，也没有修改 Rust。没有发现阻塞内存安全或结果等价问题。融合出生顺序和 DE 的批量 posting 安装继续沿用 [FUSED_DIRECT_REVIEW.md](FUSED_DIRECT_REVIEW.md) 与 [POSTING_BULK_REVIEW.md](POSTING_BULK_REVIEW.md)；本轮只证明新增差分。

两处 worktree 核对时均干净：

| 候选 / worktree（位于 `/root/code/tokenizers-worktrees/`） | HEAD | parent |
|---|---|---|
| G1 / `read-phase` | `cbb935b2948ed81659519784eb3816ec273be956` | `c8702374bd8a3812f8bca34cd53e21afe01632c3`（DE） |
| G2 / `prepare-blocks` | `2f238505b50c8555c92b4294fb7201247e35138a` | `cbb935b2948ed81659519784eb3816ec273be956`（G1） |

以下路径相对 worktree，内容与对应 HEAD 相符：

| 候选 | 文件 | SHA256 |
|---|---|---|
| G1、G2（完全相同） | `tokenizers/tk-train/src/trainers/bpe/indexed/parallel.rs` | `163a9d608c84009902cb814358ce18c0df14b961e422feb8c26c90ba48602b27` |
| G1 | `tokenizers/tk-train/src/trainers/bpe/indexed/parallel/fused_batch.rs` | `c9f91c177ee9fd3f84cdc0e39bda64ef4ebd50ec646dd66d419761a882cf3654` |
| G2 | `tokenizers/tk-train/src/trainers/bpe/indexed/parallel/fused_batch.rs` | `8b1d5d5efcbc6026d6c6e1cd17eb9dac5f3dd5d8c6b00b24b9df4883763e33ea` |

## 一、G1 的借用与读取表示

`Slot::read_phase` 接受整个 corpus 的独占 `&mut [Self]` 借用，返回与该借用绑定的共享读取 slice（parallel.rs:15–19）。u32/u16 返回原 slice；AtomicU32/AtomicU16 使用标准库安全 API `get_mut_slice`，再将其结果作为共享 u32/u16 slice 返回（:29–105）。没有新增 raw-pointer 转换、allocation 或 corpus 副本。

四个现有实现的 `Read` 都是普通整数 Slot，`SHARED=false`。u32 的 token 值保持原样；u16 和 AtomicU16 的原读取都经过同一 u16 `token` 逻辑，因此 `u16::MAX` 仍解码成 NONE，其它值仍转换为同一 u32 ID。这保持过滤、邻居访问及出生计算的全部 token 输入。

调用位置仍限于 flat、共享原子 corpus、非 AA 的融合分支（parallel.rs:979–996）。该分支在同步 `pool.install` 内取得独占读取视图，并将其传入 `prepare`。`prepare` 的 `par_iter().map(...).collect::<Result<Vec<_>>>()` 完成后才返回；它没有脱离作用域的任务，也没有保存 corpus 指针。返回的 `Prepared` 仅持有 Valid positions、Output 和计数，没有 corpus 引用或相应生命周期。

因此所有普通读取在 `pool.install` 返回前已完成，独占借用随后结束；再调用 `prepared.apply(&corpus, ...)` 使用原 Atomic Slot。apply 的并行写入也同步 join，才进入 commit 或下一轮读取。此前一轮写入完成、当前读取完成、后续写入开始的阶段顺序没有变化，普通读取与原子写入不会并发。若 prepare 返回错误，读取任务仍先完成，apply 不执行。

初始化扫描、AA/fallback、实际 apply 的原子 store、owner commit 和计时边界均未改变。新测试源码覆盖 wide/narrow 的原值与 NONE，以及一次共享写入后重新进入读取阶段；主任务报告 G1 的 47 项库测试通过及 release 构建完成，本审查没有独立运行。

## 二、G2 的过滤条件与尾部范围

G2 在每个执行中的 job 内建立两个 `[u32;128]` 与一个 `[u8;128]` 数组（fused_batch.rs:149–151）。每个 posting chunk 长度为 n，满足 `1<=n<=128`；gather、mask 计算和消费都只访问 `[..n]`（:156–173）。数组被完整初始化，活动范围每轮完整覆盖；上一块的尾部内容不会被使用。空 task 不进入 chunk 循环。

每个 p 首先读取左 token；只有左端匹配且 `right=p+left_len` 在 corpus 内，才读取右 token，否则写入 NONE（:158–170）。这保留旧循环的左端短路读取条件和越界检查。`right` 的计算、posting p 的范围以及完整 token span 的安全前提均与旧版本相同。

`endpoint_mask` 对每项写入两个等值比较的 0/1 结果的按位与（:12–20）。所选 rule 的两端都是真实 ID，均不等于 NONE，因此 guard 失败写入的 NONE 不会变成有效右端。对每个 posting，mask 非零当且仅当原条件「左端匹配、右端在界内、右端匹配」成立。u16 输入已由 Slot::token 映射成 u32，NONE 的判定同样成立。

gather 先读取一块全部端点，再处理该块有效项。prepare 从不修改 corpus，局部 Output 的更新也不会改变 rules、lengths、权重或任何后续端点，因此这个读取次序变化不改变结果。邻居 delta、出生位置、max_length 判定和权重计算的表达式保持原样。

## 三、posting、游标与 join 顺序

原 job 划分和有序并行 collect 未修改。对每个 Task，`chunks(128)` 按原 posting 顺序划分连续区间，块内的 zip 也保持顺序；被过滤后保留下来的 positions 序列与 G1 完全相同。

权重 cursor 在 Task 开始时初始化，跨 chunk 延续（:155、186），没有在块边界归零。有效位置上的 weight 查询、remove 和 birth 仍按原次序执行。因此局部出生链顺序、跨 job 的 Output 拼接顺序、后续倒序 posting 填充和全局 floor 聚合均沿用原证明。数组只存在于 job 的栈作用域，不进入 Prepared，apply 协议完全不变。

新增测试源码用长度 0、1、2、127、128 检查 mask、NONE/不匹配右端以及未使用尾部保持原值；G2 的测试和性能结果由主任务另行记录。本审查不将测试源码视为执行结果。

## 四、性能与内存口径

G1 使 prepare 的 corpus 输入成为普通整数 slice；G2 把两个端点的等值过滤写为连续 slice 循环。源码提供了 LLVM 可以优化的形式，实际是否生成 SIMD 指令及整版收益须以对应构建和测量结果确认，本审查不作指令生成或加速结论。

G2 新增固定数组的逻辑存储为每个执行中 job 1,152 字节（两个 512 字节数组与一个 128 字节数组），不随该 job 的 posting 数增长；没有新增 heap 缓冲区。实际栈布局和寄存器分配由编译器决定。既有 `peak_valid_start_bytes` / `peak_selected_lookup_bytes` 不包含这些栈数组。

`fused_prepare_ms` 和融合路径的 `delta_ms` 仍记录同一段耗时，不应相加；apply、commit 和初始化计时边界保持不变。G1 对 DE 可比较读取视图的影响，G2 对 G1 可比较分块过滤的影响。任意后续变更若让 corpus 视图逃出 prepare、引入未 join 的任务、在读取阶段写入 corpus，或改变 mask 的 NONE guard、活动 slice 长度与 posting 顺序，都需要重新审查。
