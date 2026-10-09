# Commit 快路径静态核对

新代理phase_driven_pool_design_fresh核对当前whole完整普通producer直接State+queue，
与main核心发布方式一致；AA/partial/reuse不进入，main还需要wholejob节点预算。
main在编码前floor裁剪，当前编码后裁剪，这个差异发生在prepare。
当前路由wholeChange、partial新hash/sort/append与main紧凑引用/容量复用/owner片段编码不同。
静态审查不能分配4.6s到单项；之后同口径细分定位路由+2.5s。
用户决定放弃Arena及其结束释放收益；最初提出的共享pool候选已取消、未实现。
无编辑/编译/测试/性能执行。
