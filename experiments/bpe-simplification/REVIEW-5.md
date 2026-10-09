# Typed cursor 与 Arena 方案独立审查

Fresh reviewer typed_arena_review_fresh，固定e1ddbc43，相对59bfe6f4 diff SHA256
`a517cf1f1e7d52925653060937981838af30bc36ba0c69703aa02c557afdd3ca`。
只读，未构建、测试或benchmark。

Typed cursor未发现阻断：合法block range、空流、from(len)、任意短restart、重复边界、
lower_bound及完整U64保持原语义；私有encoder保证最多十字节delta与重建有界。
建议更短等价Cursor：持blocks Range与block_remaining，块耗尽时blocks.next()；
可去掉总remaining、初始化的end-entry算术及每项额外减计数，仍是单typed cursor。

Arena建议Lease在每个实际closure内建立/销毁，不放进map_init状态，不持guard嵌套Rayon。
MutexGuard lifetime已携带Arena借用，额外arena字段可省；slot取模保持通用测试安全。
单处新片段ptr→slice unsafe可绑定整个Arena生命周期，无需unsafe Send/Sync。
Arena不得reset或早drop，分配必须初始化且不重叠，全部任务join后才释放。

8→256增长最多留下504字节/曾分配列表，转Heap及消费/过滤后仍保留退休空间；
256阈值不构成总RSS上限。Bytes enum可能增大每个Positions约8字节，blocks仍单独Vec。
收益及累计RSS须实测，本轮方案审查不证明Arena实现或其性能。
