# 新reviewer的Perf潜力审查

审查者perf_priority_review_fresh；固定1719 head-read源码、c2630853+diff。
只读，不编译/测试/benchmark。后续每轮都新开reviewer。

1. hardware prefetch：明确遗漏的main机制，同步head load不等价；收益需无采样验证。
2. by-value Change路由+fresh ownedstream append：消除owner和initial再codec，保留分组/计数/发布顺序。
3. typed Positions/Source cursor：去Box/FlatMap，但解码占比包括真正varint工作，不能全归动态分派。

owner23.30%识别engine，codec整体22.93%，owner中的codec仅7.03%约16.64Bcycles。
append不承诺消除整个owner成本，也没有复现main全规则producer/编码前pruning。

可变block增加entry序号，push按最后entry计算128边界，read/from按entry定位；lower_bound仍严格first<target。
append空目标move，否则排序检查、source首restart、offset调整、bytes复制；坐标仍完整U64，reuse仍排序。
路由可按值(Change,remove,birth)，不同owner removal复制scalar/空positions，birth独占原stream。
Positions隐藏所有编码知识，owner负责计数/所有权，边界合理；估计新增70–120行。
Block16→24bytes、短片段和removal route padding可能增RSS，必须测试。
需验证不整齐拼接、singleton、append后push、全from序号、跨边界重复lower_bound、逆序拒绝及完整模型。
