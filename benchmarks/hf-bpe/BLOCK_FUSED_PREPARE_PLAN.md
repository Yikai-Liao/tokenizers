# 多 block 融合 prepare 实现计划

用户指定的机制用于普通字符 BPE 的 block 字典路径：按源 block、规则和 posting 切片读取旧语料，立即输出邻边变化，读阶段全部结束后改写。目标是取消非 AA 批次的逐位置 Plan、位置排序、全局 Plan 拼接以及邻边的第二遍遍历。

## 当前实现与替换位置

最佳基线的 `parallel.rs` 在 block 内验证位置并保存 `Plan { position: usize, rank: usize }`，混合规则排序，随后拼接全部 Plan。邻边阶段再次读取语料，并依赖相邻 Plan 判断相邻合并。单 block flat 路径已经使用 `fused_batch::Selected` 查询旧邻居，不需要这些 Plan。

新增 block prepare 复用 Selected 的判断。它读取的是全局旧语料，block 只划分位置索引。因此一次合并跨 block，或者左 token 跨多个 block，都可以直接读取旧端点。

## 数据及所有权

任务保存源 block、规则 rank、连续切片起止，以及源 posting 的共享所有权。切片任务按源 block、规则 rank、起点建立；并行执行结果按该顺序收集。大 posting 可以拆给多个 worker，小 posting 共用一份 job 目录。源 posting 在最后一个切片任务结束时释放。

有效位置只保存 `u32` 局部偏移。出生节点保存 `u32 offset` 和 `u32 next`，每条 8 字节。片段描述保存目标 block、新 pair、逆序链头、位置数和权重和；new pair 中的新身份唯一标识生产规则与方向，另一身份是邻居。规则及 block 不进入逐位置节点。

每个规则及方向的邻居目录只记录已触及的邻居。对固定邻居，出生位置递增，目标 block 也只会向前变化；目标 block 变化时开启下一个片段。大 ID 域使用哈希目录，避免每个 job 分配无界 dense 数组。

## 正确性约束

- 权重查询使用合并起点所属源 block 的 pivots 和 previous_weight；每个有序切片重新建立游标。
- 右出生位置是合并起点，左出生位置是起点减去左邻居真实跨度。按出生位置计算目标 block，不能用源 block 减一。
- 相邻规则产生的新 pair 由左侧规则的右方向生产；右侧规则跳过左出生。普通 BPE 的批次证书及首次身份激活使每个新 pair 只有一个规则和方向生产。
- 全部 job 的 pair 权重先汇总，再应用频率阈值。局部零权重片段保留全部物理位置。
- 每个目标 block 和 pair 按逻辑生产顺序拼片段，反向填充每条链，得到严格递增 posting。
- 全部 prepare 读取 join 后才共享槽位改写；全部改写 join 后才提交及开始下一批。非 AA 区间不重叠的证书保持原样。
- AA 使用原奇偶传播和 Plan 路径。非共享槽位实现保留原路径作为结果对照。

## 提交及成本

全局 pair 频率继续由原 owner 账本汇总。接受 pair 后，一次按目标 block 建立片段索引，避免每个 block 扫描全部片段。各 block 独立计数并分配最终 posting，逆序链直接填入，然后更新 block 目录。最终目录由 owner 收集；多个线程不同时修改同一个 block 的 HashMap。

收益假设是减少 `16M` 字节 Plan 载荷、位置排序及邻边重复读取。有效起点载荷改成 `4M` 字节及任务描述。新增出生链、片段、邻居目录和片段路由索引均计入峰值；不能把 Plan 的节省直接当作 RSS 的净下降。

## 验证与首轮预算

复用原有完整 merge trace、vocab、merges 差分测试，覆盖词权重大于 u32、保留 ID、有限长度和 AA。在小 block fixture 中补跨多个 block 的长左 token、跨块相邻规则、零权重片段和 wide ID 目录检查。

基线为 affix 优化最终采用的源码。首轮仅 EN16 与 ZH512 的字典 16 位布局，初始化及 merge 四线程、词表目标分别 30000 和 50000、频率阈值 2。保留 32 位 offset 和 16/32 位 corpus 的语义检查；机器无法分配完整超过 4 GiB 槽位语料时，说明 wide dictionary 的全规模性能范围未覆盖。模型 SHA、真实路线、train、prepare、rewrite、commit、route 和 RSS 决定保留。正式计时与编译测试错开；只有具体未决异常才追加测试。
