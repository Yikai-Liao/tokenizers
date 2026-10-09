# Main Arena 调用链与局部提取独立审查

Fresh reviewer main_arena_alignment_review_fresh；main固定e4f787dc，current固定
ed47d291加diff SHA256 `19d479ed22d4cc7c281b5d983491ec611af5dc03852a3906c4ff46a5bd35df9e`。
只读，未编辑、构建、测试或benchmark。

main Arena仅处理最终小分配，按worker租用Bump；threshold针对整个layout，
包含prefix、directory、seed、stream；随physicalitems计算，不是固定payload256。
SortedPositions本体16B，一、二项通常inline。initial和fallback在group后复用
worker encoding scratch，再一次final allocation/copy；完整producer先floor裁剪，
measure/replay直接final alloc，owner移动发布。initial多wave append仍可能增长复制。

当前第三原型Inline16是payload，不是16B列表本体；各局部列表仍grow自己的
heap Vec，owner拼接再copy到bump，blocks Vec独立且按片段保留restart。
建议先测这个具体局部提取，配相同builder的NoArena控制；结果不能否定main
Arena/scratch/directproducer/packed directory组合。
Freeze/Inline未发现具体生命周期或解码问题：alloc失败保留Heap，Frozen后修改
先复制为Heap；lease释放不reset，不持锁nested pool，join后列表先于Arena释放。
零merge early return在训练资源存活时构造model，影响峰值而非语义；建议最终修正。
