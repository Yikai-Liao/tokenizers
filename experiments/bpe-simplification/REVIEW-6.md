# Arena 实现独立审查

Fresh reviewer arena_implementation_review_fresh，固定e1ddbc43加Arena diff
SHA256 `e801f9bd38547fb47ad87629861babb85669567b6814915d8f0fc0c49ca853bd`。
只读，未构建、测试或benchmark。当前目录CodeGraph无索引，使用源码。

未发现阻断：Bump片段已初始化、独占且不reset，slice lifetime绑定Arena，
lease结束后内存仍有效。Send/Sync按字段自动推导；worker>=1，closure内一次lease，
不持锁嵌套Rayon。正常/错误/restart均join再释放，模型输出前drop训练资源。
空payload不分配，空目标move；append先check排序/count，再扩展bytes，再修改目录，
allocation错误保留前缀。range/seek、公有行为保持。

局部建议：多字节push逐次extend可能在中途allocation Err后留下半条delta，
生产调用立刻?终止训练，当前无公有恢复问题；用10-byte栈缓存后一次extend可
获得可重试性并减少逐字节调用。unsafe注释加SAFETY前缀。
Arena保留淘汰/增长/临时Writes的小payload，Bytes状态变大，须按新采样裁决。
