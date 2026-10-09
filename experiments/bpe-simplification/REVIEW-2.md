# 只读简化审查 2

固定c26308538800a813ef350d26031fa9942447b5f2 + head-read ring；1719生产行。
merge SHA256 d74ce9e08e440330f119212352103e8c11d2f7f41fc235bb6cfaf697a36fcd8c。
审查者lean_engine_review_1；不编辑、不编译、不跑测试或benchmark。

- ring初始化16项、顺序补位、短尾结束正确；重复坐标和完整U64输入保留。
- head与matcher必须来自同一corpus、position及prepare snapshot；当前满足此契约，apply仍在join之后。
- 同步token读不等于硬件prefetch，不能保证16个outstanding cache misses；收益仍须测量。
- read闭包可改为iter.map后直接heads.next()，预计少2行、少一层闭包。
- fresh_matcher与matched复制部分几何判断；可统一task-local matcher预计少8–12行，但可能增加hot-loop occurrence分支，须测量。无可信收益时直接撤缓存更简单。
- Source有序位置与读取窗口、Corpus几何判断的边界没有明显变浅。
- 建议剔除无收益ring/cache、独立关闭dense初始计数，再做零offset/Vecusize纯精简；first-piece移入/small-inline另测，联合收益不拆归因。

本轮结束时用户要求改为Perf驱动优先级：先定位main与候选的主要差距，不继续盲目叠小优化。
