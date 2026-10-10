# 独立复核：统一 owning heap 实验

复核上下文只读取源码、作者命令及其输出；未执行或修改目标代码。
目标为 `177302e2873649deb8c11d8be371f2710e44ed8f` 加 `index.rs` 补丁：

- index SHA256：`191de8af497430a99c10a43404933ec051d5e4c3231d318e0e538a0d8cca4cc8`
- patch SHA256：`dab1336d97454a1321f6d3c3d67b75c77bd7bec5cc33ea1ddedebaa00e4f54dc`

静态复核未发现新增的列表所有权、checked 数值更新或错误清理缺陷。
fresh count 删除、reuse signed ledger 保留、partial/complete birth 单次转移
保持原路径。reuse 初始零 count 列表原本不进入队列，新实现丢弃其列表不影响
选择。精确 Positions layout 字节、joined checkpoint、输入加载与计时边界
适合本次资源测量。竞争检测应使用实际 executable 路径，作者已修正。

原 default/no-default 各 17 native tests + 1 doctest，以及 Clippy 已通过。
但扩大参考对照后，语义结论有如下限制：

- 生成 case 84 的 baseline 与 candidate engine trace 逐字一致，两者均不等于
  reference。这是已有差异，不能归因为统一 heap。
- 加入 64 个独立高优先 pair 的 alias fixture 后，两边都出现 reference
  差异；两次调用方 map 的 ID 顺序不同，不能宣称跨实现精确等价。
- 初始 priority 唯一只保证当时的出队次序，不能证明 heapify 改变内部形状后，
  后续同 pair、同 count 的历史 cohort 次序保持。

因此实现保留为内存实验，不作为完整语义等价并可直接替换的结论。实测输入
上的完整模型相等可以支持那些输入的资源比较，不能关闭 reuse 次序问题。

原子 live-owned 包含临时编码片段；index-owned 是该 checkpoint 的索引列表。
map capacity 仅估算元素载荷，排除 bucket/control metadata。HWM 包含输入加载，
初始化峰值可能掩盖后续退化；应同时展示 commit 后的 stale-owned 和 RSS。
