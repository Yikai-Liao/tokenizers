# BPE 原理证明与证据附录

2026 年 10 月 12 日

本文中 producer 是一条合并规则及其全部子任务，cohort 是原实现的一次兼容批次，family 是该规则所属 cohort 完成时按一侧与相邻 ID 归类的一组边界。drain 指补齐并发布窗口尚未发布的合法出生项，使队列重新完整。

本文说明哪些工作可以在保持精确贪心 BPE 语义的前提下省去，哪些工作只是改了位置，以及每项结论需要什么条件。对象是带词权重、raw AA 计数、固定 fresh 几何和确定性 pair-ID 平局顺序的训练器。

核心结论是：保留现有 cohort 内并行与阶段 join，可以延后构造 newborn Positions；最小 FRONTIER 版本甚至不需要第一次准备中的 newborn family 汇总。更强的 exact M 版本另行支付完整 family 归约，换取更强 admission 和整 producer 的提前拒绝。两者均是待实现、待原生测量的方案。

本文提供数学推导、固定源码核查和独立有限模型证据。没有运行、导入、编译、测试、修改或推送生产仓库代码；没有新的 Rust 原生性能、内存、调度或借用检查结果。模型中“所有检查成立”不等于原生实现通过。

## 1 语义和计量单位

### 1.1 选择的是加权 raw adjacency

设输入词行集合为 W，行 w 的权重为非负整数 ω(w)，当前 token 路径为 t₁…tₙ。pair p=(a,b) 的选择频率为：

    f(p) = Σ_w ω(w) · |{ i : (t_i,t_(i+1)) = (a,b) }|。

频率大的 pair 优先；同频率时，数值 ID 对按字典序较小者优先。本文把完整优先级写成：

    priority(p) = (f(p), Reverse(p))。

这不是字符字符串顺序，也不是“不重叠替换所节省的 token 数”。AA 按所有相邻 AA 计数，实际应用仍从左到右选择不重叠起点。

例如 AAAAAA:1 中，AA 的 raw 频率为 5，实际 contraction 数为 3，结果 XXX 中 XX 的频率为 2。三个数不能互换。

物理坐标数、实际 contraction 数、加权质量、编码字节数、叶片数和完整 key 数分别计量。一行权重极大也可以只有一个起点；零权重行仍可以有很多需要修改的物理起点。

若 ABc:2 和 ABc:0 一起训练，Xc 的质量为 2，但 live Positions 必须保留两个物理坐标。标量加法可以略过零；不能由此省略零权重 contraction 或 live family 的零权重坐标。

### 1.2 fresh 和 appended 是两个条件

“fresh”指输出 ID 从未激活过；现有低数值 reserved ID 的第一次激活也可以是 fresh。“appended”还要求该 ID 是本次新追加，并大于所有窗口开始时的旧 ID。

终端频率 floor 的生存证明只要求 first activation。本文 joined window 的紧凑字典序界还要求 appended，故遇到未激活 reserved 输出时应先 drain，再走原有 singleton 路径。

曾激活、后来没有当前 occurrence 的 ID 仍是 active。若下一条确实应选中的规则复用 active 输出，必须触发原有 whole-attempt reuse restart；不能只清理当前窗口。

### 1.3 长度和 floor 的范围

令 F=max(min_frequency,1)。窗口内 F 固定，只有在完整 drain 后才可采用新证明的更高 floor。

初始化 pair 的 admission 不应用 newborn 长度门槛；新出生 key 使用 retained-symbol span 之和严格小于 cap 的条件。不是 ≤cap，不是 UTF-8 字节数，也不能反过来用 birth gate 淘汰已存在的初始 key。

因此“旧 key 不在索引中”有两种相关原因：质量低于 floor，或其出生时被长度规则拒绝。后一种 key 的物理质量可能很高；但 descendant span 不减，长度不合法的见证不能变出长度合法的更长 descendant。

## 2 joined old-input window 的等价性

### 2.1 窗口条件

窗口开始于完整可选择索引。令 B 是已检查合法的第一个 appended ID，旧输入域 I₀ 中所有 ID 小于 B。窗口中的每条 producer 规则满足：

1. 两个输入都属于 I₀；窗口输出在 drain 前绝不作为输入。
2. 输出是唯一的新追加 first activation；不同规则不能共用输出 ID。
3. 继续使用现有的 compatible priority prefix；不跳过被阻挡的堆顶。
4. 普通 non-AA cohort 的 heads 与 tails 不交；AA 保持 singleton 及既有 canonical nonoverlap 语义。
5. 每个 cohort 都完成 prepare join、apply join、old-loss owner join 后，才选择下一 cohort。
6. 每个候选都必须击败此前 cohort 尚未发布 newborn 的有效完整优先级上界。
7. 同一 cohort 的所有候选比较同一份此前上界；不把该 cohort 刚选成员的 guard 混入它自己的 admission。
8. 整个窗口累计最多 256 条规则；任何容量限制若截断一个原本更长的 compatible cohort，就结束当前窗口并 drain，不能每个 cohort 重新获得 256 个名额。实际 vocabulary target、身份检查、checked arithmetic 和失败后丢弃 attempt 的规则保持成立。

本节结论不允许跨 cohort 的 prepare/write 重叠。第 9 节单独讨论更昂贵的异步推广。

### 2.2 原始切口唯一所有权

一条原始 exterior cut 的任一端在窗口内最多被替换一次。若两端先后改变，由较早 cohort 的 producer 拥有这条切口；若同 cohort 改变，由物理左侧的 contraction 拥有。

较晚 producer 若在某一侧遇到先前窗口输出，跳过该侧的旧损失与出生所有权。同 cohort 仍使用原有 physical-left 判定，避免共享切口重复扣减。

最后的边界若为 (X,Y)，其所有权由 X、Y 的固定 producer/cohort 唯一确定。因此不仅每个 occurrence 有唯一 owner，整个完整 key 也有同一个 owner。

在 producer X 的最终发射中：

    右邻居是窗口输出 Y：当 cohort(X) ≤ cohort(Y) 时由 X 发射。
    左邻居是窗口输出 Y：当 cohort(X) < cohort(Y) 时由 X 发射。

老邻居对应的边界由 X 发射。XX 只通过物理左侧 X 的右 port 发射一次；不能只检查 producer ID 相同后让左右两侧都发射。

### 2.3 为什么可以隐藏出生而保持旧索引精确

每次 contraction 新暴露的边都包含它的 fresh 输出，所以窗口内不可能出生 old-old key。旧 key 只有损失，已选或低于 floor 的旧 key 不会复活。

上述唯一 owner 给每条相关旧 exterior cut 恰好一次损失。cohort 的三次 join 关闭所有读、写和旧计数更新后，下一次选择看到精确的旧标量 frontier。

隐藏 key 由唯一 producer 的有效 guard 支配。旧堆顶若击败所有 hidden 上界，就同时击败完整参考队列中的所有旧 key 和隐藏 key，故是参考算法的同一下一赢家。

结合既有 cohort 内 compatible-prefix 结论，对 cohort 次序归纳，得到同一条规则序列、选择频率、输出身份次序和物理 contraction 结果。最终 drain 只重建每条 surviving owned cut 的最终 key，恢复完整索引。

### 2.4 no-K0 版本可保留普通删键

K₀ 表示窗口开始的旧 key 成员集合。它可作为可选的原始见证过滤器，但不是 joined window 定理必需的运行时结构。

若准备和最终发射都不查询 K₀，则可以保持普通 selected-key 删除、below-floor 删除和 absent-key lazy heap repair。不需要冻结旧 count-map 拓扑、零 tombstone、retired-key log 或专门清理阶段。

活的正计数仍做 checked subtraction；目标已不存在时忽略后来损失，和原有 fresh 语义一致。不能对所有计数 saturating subtract，也不能对已经删除的 selected key 强行制造 underflow。

删除标量 cell 不会取消已 detach candidate 的 Positions 所有权。其 read task 与 snapshot 生命周期仍需完整结束后才能释放。

如果保留可选 K₀ 过滤，就必须保留该过滤器需要的成员信息，尤其是被选中的 AA 见证；不能一边删成员，一边继续把“是否存在”当作原始见证。

## 3 不做出生汇总的 FRONTIER 证书

### 3.1 普通 cohort 的 pre-write frontier

选完普通 non-AA compatible cohort、移走其 selected keys 后，在任何本 cohort 写入或旧损失发生之前，读取剩余旧队列的精确最佳项 p，频率为 u。

每个可选择的 current-owned newborn 都有一个未选中的旧边界见证。若该见证存在，它的完整优先级不高于 p；若不存在，它只能是 subfloor，或长度不合法且 descendant 不会改善。

因此，这个 cohort 的任意可选择 descendant k 满足：

    f(k) ≤ u；若 f(k)=u，则 head(k) ≥ head(p)。

对 old-old 候选 q，可用下面的紧凑充分条件：

    f(q)>u，或者 f(q)=u 且 head(q)≤head(p)。

等 head 时 q 的 tail 仍小于 B，而旧左侧 newborn 的 tail 是 appended 输出，因此 q 赢得完整 pair 平局。

若普通 cohort 的 pre-write 旧 frontier 为空，则其所有 current-owned descendant 都不可选择。可以在实际应用和相关 reader 结束后释放这些 starts；这不表示不存在物理 newborn 边。

不能在本 cohort 的损失之后捕获 frontier：旧见证可能已经被消耗，而它留下的 newborn 仍很强。

### 3.2 AA 的 raw fallback

AA 的选中旧 key 本身可能是 XX 的见证，不能套用“所有见证均未选中”。AA singleton 保留：

    guard=(selected raw frequency f，selected input head a)。

每个可选择 owned child 的质量 ≤f；若达到 f，其 head ≥a。一个更小 head 的合法旧左见证本来就会先于该 AA 被选中。

该界不需要 accepted-mass 计数，也不需要 newborn family totals、minima、predicted key map 或 per-family eligibility 表。

### 3.3 实际省去的准备工作

第一次准备只保留 accepted starts、固定 producer/cohort/span 描述和唯一 owned 旧边损失。左侧同 cohort predecessor 检查仍必需。

右侧旧损失只需原来的 (selected tail, old next)。为预测 newborn 而做的右邻居替换查找、部分 AA membership 查询、birth span 检查和 newborn family tally 可从这一遍移走；最终 drain 读取真实 final neighbors 后再做出生工作。

旧邻居读取、旧损失分组、全部 contraction、三个 join、starts 保留以及最终完整 floor 聚合依然存在。

### 3.4 最多四个 cohort 是结构上限

设普通 seed 留下的 guard 为 (u,a)。旧计数之后只会下降，仍为 u 的 key 的完整字典序不会早于捕获的旧堆顶。

所以任何后续通过该 guard 的旧规则必须同时满足 f=u 和 head=a。后来的 guard 不能放松仍保留的这一份约束。

同一 count/head plateau 依次最多分成三段：tail<a 的普通兼容段、AA singleton、tail>a 的普通兼容段。在窗口累计 256-rule cap、且容量截断即结束窗口的条件下，每个普通段不会在同一窗口内因重置配额再分成多批。因此包含 seed 在内，FRONTIER 窗口最多四个 cohort。若每 cohort 独立重给 256 个名额并继续窗口，这个四批结论不成立。

若只用每个 producer 的 raw (f,head) guard，则 seed 已落在同一 plateau 内，最多三个 cohort。小型确定例子可以达到这两个界。

seed 自身可以包含多个频率下降；不能把结论误写成“整个窗口只有一个频率”。四-cohort 结论也不适用于另有独立证明、能够安全降低 guard 的不同协议。

## 4 exact coherent-family maximum 的更强证书

### 4.1 完整 family 与不合流定理

在 producer X 所属 cohort 的一致完成视图中，仅取由 X 唯一 owned、且满足现有 source birth-length gate 的边界，按方向及 coherent neighbor ID 分 family h。此处不以 task-local floor 过滤，subfloor family 仍完整归约。右 XX 规范化为右 family。所有 producer jobs 对同一 h 的贡献先完整求和：

    F_h=Σ_j v_j(h)，M_X=max_h F_h；无合法 owned family 时 M_X=0。

一个 surviving old neighbor c 以后可以成为严格较晚的输出 Y。X 右侧的 Y 唯一确定原来的 first input c；X 左侧的 Y 唯一确定原来的 last input c。

已存在的 earlier/current 输出不会再被消耗。因此两个不同 coherent families 不可能在后续窗口操作中合流成同一个最终完整 key；两种方向只有 XX 的重复端口问题，而它已由唯一右 owner 排除。

每个最终 descendant k 因而有唯一祖先 family h(k)：

    f(k)≤F_h(k)≤M_X。

必须先让同 cohort 的所有邻居替换按一致视图分类。半处理的 XC 与 XY 可能稍后同变为 XY，这种错误分类会直接破坏不合流条件。

### 4.2 完整优先级与字典序平局

当 M_X>0 时，令 T_X 是所有全局最大 families 中最小的完整 pair。由于所有替换 ID genuinely appended，descendant pair 不小于 coherent ancestor pair。

    f(k)=M_X ⇒ F_h(k)=M_X 且 k≥h(k)≥T_X。

因此 (M_X,Reverse(T_X)) 是完整优先级上界。不能从任意一个局部赢家或任意一个全局赢家取代最小 tie key。

针对 old-old q=(a,b)，可进一步压缩为：

    ell_X=min{c : old-left family (c,X) 的完整质量为 M_X}；无此 family 时 ell_X=B。
    q 击败证书 ⇔ f(q)>M_X，或 f(q)=M_X 且 a≤ell_X。

仅全局最大 families 影响等频比较；非最大 family 永远达不到 M_X。右 family 和 new-left family 在等频时都输给 old-old 候选。

全窗口只需取 M_win=max_X M_X，再在达到 M_win 的 producers 中取最小 ell_X。不是把 producer guards 相加。

### 4.3 整 producer 的早拒绝

若 M_X<F，X 的全部 owned descendants 都低于固定 floor。X 完成原本的 contraction、旧损失和必需 reader 后，可省去自己的最终 starts 遍历、出生 tally、leaf 编码及随后路由与丢弃。

X 的 descriptor 仍须保留：其他 producer 的 final neighbor 可以是 X。被拒 producer 中的零权重 starts 也必须先应用。

若 M_X≥F，最小版本会丢弃临时 family totals，并正常发射这个 producer 的所有其他合法侧。它不会顺便剔除其中每个 subfloor family；保留逐-family 决策需要另一个存储和 ancestry 查询合同。

### 4.4 no-K0 不削弱可选择 floor 以上的 exact M

去掉有效 K₀ gate 新增的 length-eligible coherent family，若其见证因 subfloor 缺失，则其完整质量仍小于 F；若因长度不合法缺失，其 descendant 已被长度门槛排除。

故去掉 K₀ 可以改变一个始终低于 F 的数值 M，却不会改变 M≥F 时的 (M,ell)，也不会改变整 producer 的 M<F 判决。其全局 maximizing tie set 的有效部分相同。若连 length-ineligible families 也计入 M，上界仍可保守安全，但更弱，不能再声称这里的 floor 以上等价。该等价也不能直接移给局部最大值之和：不同 jobs 的不同 subfloor families 可能被相加成跨过 floor 的弱上界。

### 4.5 归约的真实成本和关键接口陷阱

可复用两份直接 ID directory 和两个按 side 分开的 sparse vectors，逐条 checked-add。旧 directory slot 必须同时通过索引范围与当前 key 的 back-reference 验证；不必加 generation counter。

若用单一 sparse vector，验证必须包含 side 或完整 key。两份 directory 不会自动让 neighbor-only 验证安全。目录 ordinal 应指向 side-local family，不能无理由用任意数量的扁平记录 ordinal。

现有 `map_init` scratch 在迭代结束后不会自动留给后续 reducer。复用技巧不等于复用那块 allocation；冷启动 O(ID domain) 初始化、scratch 数量和高水位容量都要计费。

尤其要修改 count-only liveness：现有 `Neighbors::finish` 按 removed_weight 非零或 Positions 非空保留记录。禁用 payload 后仍用这一谓词，会丢掉 positive born-only family。

例如 ABCD:1、AB:3、CD:2 让 AB4 与 CD3 同 cohort，输出 X、Y。X 右侧先形成 old C loss group，再形成 removed_weight=0 的 Y birth group。XY1 必须进入 M_X；省去 Positions 后不能把它过滤掉。

单个完整 family 被 selected raw count 支配，但不同 family 之和不一定 fit u64。CABD:u64::MAX 可以产生 CX 和 XD 各 u64::MAX；合法的 M 是 MAX，两者的无关总和却溢出。

## 5 动态终端 floor 的生存证书

### 5.1 horizon 由实际目标和激活历史决定

在完整 drain 的 fresh frontier，令 V=vocabulary.len()，目标 T>V，R 为现有 ID 中从未激活者的数量，A=V−R 为 ever-activated ID 数。安全的剩余规则上界是：

    H=(T−V)+R=T−A。

每次 first activation 要么追加一个 ID，要么激活一个现有 inactive ID，A 都增加 1，所以 H 恰好减 1。因为 R≤V，H≤T；这里的 usize 加法也有范围证明。

必须数激活历史，而不是当前 occurrence support、正权重 support、special-token 声明数或固定 256 字符。零权重词观察到的字符也会激活；最后一个 occurrence 消失不会把历史清零。

只用剩余词表槽数会错：预留 AB，输入 AB:5、CD:1，初始 V=5、target=6。AB5 激活 reserved ID 不增加 V，之后 CD1 才填满目标。H=2，而 T−V=1 错把最终频率认证成 5。

以下证书在 H≥1 时证明接下来的 H 次选择都有频率 ≥L，因此也证明在实际更早达到目标前的全部选择。H=0 或无可选择规则时不存在需要宣称的“最终合并频率”。

### 5.2 不相交 token 见证 matching

选一组当前合法 key S，频率均 ≥L，且不同 key 的 endpoint ID 集合不交；AA 占用一个 endpoint。

一条规则只消耗两个输入身份，所以最多碰到 S 中两个 key。r=H−1 次选择后，若：

    |S|≥2H−1，

仍有一个未受影响见证保持原频率，下一条贪心规则就至少为 L。对每个前缀应用同一论证，得到整个 horizon 保证。

一般只找 H 个见证不够：词 ABC、BCD、AB、CD 各权重 L/2，可令 BC 在 AB、BC、CD 的频率 L 平局中先选。AB 和 CD 的 matching 有两个 key，但选完 BC 后二者都降到 L/2。

若快照有 V₀ 个身份，非 loop matching 至多 ⌊V₀/2⌋；这种简单证书通常只能覆盖约 V₀/4 的 horizon。它不是从 256 字符直接认证数万次未来规则的工具。

### 5.3 directed threshold hit 证书

冻结完整阈值集合 C={p:c_p≥L}。选 q=(a,b) 只会改变 C 中以 a 结尾、以 b 开头及 q 自身的旧 key。

令 in_C(a)、out_C(b) 分别为这些有向度，则一次旧规则可能命中的 key 数为：

    h_q=in_C(a)+out_C(b)−1[(b,a)∈C]+1[q∈C 且 a≠b]。
    h_new=max(max_a in_C(a), max_b out_C(b))。

含一个 fresh 身份的未来 key 最多影响一个旧角色；含两个 fresh 身份的 key 不影响任何 frozen key。每个快照旧 key 至多选一次，所以 r 次动作的安全 hit budget 是：

    T_r=r·h_new + top_r_sum(max(h_q−h_new,0))。

若 |C|>T_r，就有 untouched 见证存活，认证终端 floor L。无条件可让 q 遍历全部快照 selectable keys；first-failure 归纳允许只用 q∈C，因为在首个低于 L 的选择前，初始 sub-L 旧 key 不可能入选。

给每个 witness 非负整数 credit，并对各动作可能命中的总 credit 给出安全容量，还能得到加权 packing 版本。可行的任意 credit 分配已经是证明；浮点 LP 的名义最优值未经向安全方向修复不能直接当作证书。

### 5.4 frozen excess-mass depletion 证书

令 M=max_p c_p，冻结势能：

    D_L=Σ_p max(c_p−L+1,0)。

它是把所有 frozen keys 都降到 L 以下至少需要消耗的整数质量。一次物理替换最多破坏 selected edge 与两个邻边，其加权替换质量 ≤raw selected frequency≤M。

最简单条件 D_L>3rM 足够。计入旧 selected key 自身只需耗掉其 excess，可加强为：

    D_L>r·(3M−L+1)。

进一步对每个 frozen q 设 b_q=2c_q+max(c_q−L+1,0)，利用每个旧 q 只选一次，而 newborn 动作最多消耗 M：

    B_r=rM + top_r_sum(max(b_q−M,0))。
    D_L>B_r ⇒ 接下来 H 次选择的频率均 ≥L。

AA 的实际替换质量小于 raw count 时，以上只是更保守，不会失效。用 checked wide arithmetic 或向保守方向饱和的比较；证书计算溢出应使证书失败，不能拒绝原本合法的训练输入。

### 5.5 整个阈值集合兼容的特殊证书

若整个 C 没有 self-pair，且 Heads(C)∩Tails(C)=∅，它的规则在被选中前保持频率。期间可选择 newborn 的见证在 C 之外，频率严格小于 L。

故 |C|≥H 就足够。这里必须是整个 threshold set，不能只挑一个兼容子集而忽略外部高频冲突规则。

### 5.6 fully drained 是不可删除的假设

“可见旧计数准确”不等于“当前所有候选都已包含”。隐藏 newborn 会改变 M、threshold set 和每步 depletion budget。

独立定向模型采用六行 xyab…xyag，各权重 2，再加 uv:1，floor 1、严格长度 cap 4、target 14。先选 XY12 后，XA12 隐藏，可见 AB…AG 各 2、UV1。

若在未 drain 状态只看旧 key，L=2 时会错误得到 D_L=6>预算 5，从而错误退休 UV。正确完整 trace 频率为 [12,12,1]、最终 V=14；错误提前 floor trace 为 [12,12]、V=13。

所以应仅在 fully drained、vocabulary 与 activation/span domain 对齐、没有 hidden producer 时认证。窗口内维持此前有效 floor。

### 5.7 floor 只能在同一保证下保持或提高

对于相同 target、语义和 fresh attempt：

    F_run=max(配置 floor，至今全部有效终端证书)。

后一次启发式扫描给出较弱下界，并不会使先前整个 horizon 的保证失效，也不构成降低 floor 或重建索引的理由。

若 target/horizon 改变导致原保证不再适用，单改阈值无法恢复已删计数、丢弃 leaves 或被整 producer 拒绝的数据。继续前必须完整重建合适的 lower-floor frontier；最简单可靠路径是从原始输入 restart。

active-alias 触发 reuse restart 时，应重置提高过的 floor 和全部证书/退休状态。此前的 first-activation 生存保证仍确保第一个真正应被选中的 alias 不会藏在被错误剪掉的低频区。

## 6 完整 producer 出生发布的独立等价性

本节是另一个独立变换，单个普通 fresh cohort 就能用；不以 count-first 或多-cohort window 为前提。

唯一完整-key owner 已由第 2 节给出。按原 task/spatial 顺序读取 producer 的 nonempty leaf records，直接按 final side/neighbor 分组，等价于原 owner stable sort 后每个 key 的子序列。

完整求和后才测试 floor；live 组只调用一次既有 `Positions::concat`。零质量但 nonempty 的叶也进入 stable grouping。叶输入顺序与边界验证不变。

当前 concat 移动有序 leaf descriptors 和其 payload，不解码或重新编码坐标；多叶生成 Root，而且 Root 不能再作为 concat 的叶输入。不能写成任意二叉树式 `concat(concat(a,b),c)`。

因此 producer 归约得到的完整 Candidate 必须走专门的 owner insertion seam，不能伪装成普通 leaf Change 再经过现有 concat。所有 producer 结果完成后再结构发布，owners 完成后再入 heap/允许下一次选择。

令 H_f 为 nonempty birth fragments，K_live 为最终 admitted keys。birth hash/route 可从 H_f 降到 K_live；H_f 次分组求和并没有消失，而是搬到了 producer reducer。

若原 combined commit 有 L 个正 loss records，E 个同 shard 的 loss/birth 合并 action，则：

    原 pending actions=L+H_f−E。
    分开 loss 与完整 Candidate 后=L+K_live。
    action 数减少=H_f−E−K_live，可能为负。

birth-only window drain 的最终阶段 L=E=0，路由缩减比较干净；此前各 cohort 的旧损失必须另计，不能重复领功。

per-key SmallVec staging 可避免 singleton key 的外部分配，但大组仍分配。两遍 flat redistribution 可消掉逐-key staging 分配，却新增第二遍目录访问、prefix/range 规划和 H_f 个已初始化 Positions slots。

两者都有单 producer 的 Ω(H_f) 顺序 grouping tail。很多 key 原先可能分散到并行 owners；少做总工作不保证单巨大 producer 的关键路径更短。

## 7 工作与空间的可证界

### 7.1 当前 fresh baseline 的物理坐标账本

对一次成功 fresh attempt 的任意已提交前缀，令 R_c 为实际 contractions（含零权重行），D 为选中 candidate Positions 解码坐标数（AA 原始扫描计一次），R_AA 为 AA contractions。

    R_c≤D≤3R_c。
    当前 matcher 次数≤D+R_AA≤4R_c。

理由：每个接受坐标对应一个 contraction；每个 stale posting 或 live AA overlap 可唯一记到账于使其失效的 contraction 的某一 exterior cut。每个 contraction 最多两条这样的 cut，fresh cohort 又只消费一次。

若 E₀ 为初始物理 word-internal edges，P_b 为 newborn push 数，则：

    R_c≤E₀，P_b≤2R_c，D≤E₀+P_b≤3E₀。

这些是物理 work units 的摊还界，不含 weight-region 查找、AA membership 二分、directory 初始化、分配、sort、heap、cache miss 或 worker idle。一个晚期 batch 仍可含大量被早期 contractions 支付的 stale work。

局部替换与摊还清理是既有 Re-Pair 原理；这里的常数界依赖本源码的唯一切口所有权、sealed fresh cohort、AA 和 restart-aligned codec 合同，不作新颖性宣称。参见第 11 节文献 1 至 3。

### 7.2 延迟窗口的坐标版本账本

固定同一段已接受 old-input window。令 N₁ 为只变化一次的 surviving cut，N₂ 为两端在不同 eager cohorts 改变的 surviving cut，N=N₁+N₂。同 cohort 双端变化已经被原实现一次合成，不能重复计算其节省。

不计其他 gates 时：

    eager newborn 版本=N+N₂；最终 deferred 版本=N。

FRONTIER 不做第一次 family tally，因此直接少了 N₂ 次 birth tally/encode，但新增 final starts/context/weight traversal 与 retention/guard 成本。

exact M 再令 N_R 为整 producer 被 M<F 拒绝的 final eligible cut events，则：

    eager scalar/encode events=N+N₂。
    exact-family scalar tallies=N；最终 scalar/encode events=N−N_R。
    少编码=N₂+N_R；净增 scalar events=N−N_R−N₂。

假设仅为解释而令每版本编码成本为 b、标量 tally 成本为 s，surviving replay 为 R_s，跨 job exact reducer 为 E_g，其他已验证净收益为 H_net：

    净收益≈(b+s)(N₂+N_R)−sN−R_s−E_g+H_net。

令 ρ=N₂/N、q=N_R/N，可整理 break-even 条件；它只是等待实测系数的可证伪账本，不是速度预测。N=0 时不用比值。

当 ρ=q=0，count-first 没省出生版本却新增归约/重放；当 ρ=0,q=1，单 cohort 就可能省掉全部出生 payload；当 ρ=1,q=0，只是少一半 eager 出生版本，绝不是总 runtime 翻倍。

编码坐标数不决定编码字节：delta 宽度、restart、fragment 数与 capacity slack 会改变每坐标成本。

### 7.3 reducer 与关键路径

令 H_r 为 exact reducer 扫描的 records，H_r⁺ 为正 family records，K_r 为 distinct positive families。目录已定长时 work 为 Θ(H_r+H_r⁺+K_r)，该顺序 reducer 的 span 为 Θ(H_r)，临时 sparse storage 为 Θ(K_r)。

冷 directory 还要加 Θ(ID domain) 初始化。H_r 含 loss-only placeholders；不能只按 K_r 报成本。令 J_r 为 producer jobs，局部最大和 G_r 与完整 M_r 满足：

    M_r≤G_r≤min(selected raw f，J_r·M_r)，M_r>0。

精度差可随 job 数增长，但不证明加强证书值得支付 reducer。单 job 没有跨-job 精度收益。

joined 路径仍有各 cohort 的 selector、prepare/reducer closure、apply、old-loss owner 的串联，以及各窗口 final drain。外层窗口少不等于这些 join 按同样比例减少。

理想 fork-join 中 max(work/P,span) 是下界；实际机器还受带宽、缓存、allocator 和调度约束。若 C 个 cohorts 的 surviving fragments 总数为 N_f，把多个小 sort 合成一个大 sort 也可多出至多 N_f log C 量级的比较排序代理项，不能默认“最后一次 sort 更便宜”。

### 7.4 retained starts 的容量界

对 spill 到 heap 的 starts 计 `capacity×sizeof(usize)`；inline starts 已包含在 job 结构内，不能重复算。再加完整 retained job-vector capacity、输出描述、容器头和其他实际拥有的 allocation。

令 S_i 为 cohort i 在释放前的 starts 加 job-container charge，C_i≤S_i 为安全释放后留下的 charge，H_mem 为整个 attempt 至今最大的 S_i。

每次完成 cohort 后，若 retained R>H_mem 就 drain；继续时保持 R≤H_mem。下一 cohort 的 joined checkpoint 有：

    R_before+S_i≤H_mem(old)+H_mem(new)≤2H_mem(new)。

另加 descriptors 容量。该 2H_mem 是方案自身的 joined-checkpoint owned-storage 界，不是相对原实现的两倍总内存，也不是 RSS 界。构造中的双 allocation、AA canonical vector 和最终 payload 重叠须另计。

H_mem 不能悄悄每窗口重置，否则两份正 retained cohort 就通常触发 drain，退化为最多约两个 cohort 的策略。固定阈值 B₀ 的 post-cohort 检查只是软预算，下一 cohort 可以任意超出它。

256-rule cap 只限规则和描述数量，不限 occurrence 字节。预先用 candidate Positions.len() 可上界 accepted 个数，但 hard byte cap 还需证明 vector growth、job layout、inline 与容量；超大 singleton 仍需明确的超限策略。

一个反例足以说明“少编码”不等于“少内存”：500 行不同的 unit-weight ABBCD_i，在四叶布局中可省的 XB short-gap payload 只有 560 used bytes；两 cohort 同时保留 1,000 个 raw usize starts，在 64-bit 下为 8,000 used bytes，而 eager baseline 的单个当前 500-start cohort 是 4,000 used bytes。额外一个 raw cohort 的 4,000 bytes 就可大于省掉的 560 bytes；总 HWM 还不能由这两个快照相减得到。

## 8 集成时不能省略的语义边界

- 必须先取得并验证候选的 priority/support/cap/target admission，再做有副作用的 resolve、take、activate，或据 resolve 结果决定 alias restart。被 hidden guard 阻挡的旧头不是当前应选规则；提前 restart 也可能选错时机。
- ordinary old heap 空而有 pending births 时必须 drain。例如 ABAB:1 选 AB2 后旧队列空，hidden XX1 仍可成为下一规则。只有所有合法出生已发布或有证据拒绝后才可判 underfill。
- 检查实际 vocabulary.len()，不以已选规则数替代。reserved 激活可能不增加 len；实际 target 达到后，不应再检查下一条超出 horizon 的潜在 alias。
- 真正 terminal elision 要保留精确 ordered records，并确保调用方不会再选择；不能返回成功却丢掉最后已选规则。已经启动的 fallible work 仍必须传播错误，不能靠“最终不再需要 payload”吞掉错误。
- `AAA:u64::MAX` 的初始计数溢出仍须在原来的初始化阶段报告，即使 target 已经满足。ID 上限与 word-separator 保留值检查不能由比较用 sentinel 取代。
- count-only 正 birth records、zero-weight 物理坐标、同 cohort 左所有权、跨 cohort 早所有权、final leaf 顺序，以及完整 Candidate 直插且 Root 不得再次送入 leaf-only concat，都是正确性条件，不是可以用 timing 结果抵消的细节。

## 9 异步扩展的三个独立证明层

这一节说明减少 worker 空泡有实质数值机会，但不是给 joined 方案删几个 join。它需要 count backend 的并发所有权、immutable metadata、任务预登记、实际 happens-before 和 scope/error 生命周期的额外实现。

### 9.1 渐进 pending-birth 上界

令 producer 的 immutable selected raw frequency 为 f；已完成 job j 的 accepted weighted mass 为 m_j，同 job 每个完整 child family 聚合后的最大质量为 c_j（没有合法 owned 出生时取 0）。fresh canonical ownership 给出 0≤c_j≤m_j，AA 另有总 accepted mass≤f。

    U=f−Σ_completed(m_j−c_j)。

已完成 jobs 对任意 child 的贡献 ≤Σc_j，未完成 jobs 的剩余贡献 ≤f−Σm_j，故每个可选择 child 的频率≤U。U 单调不增；每个完整 job 只发布一次非负 credit m_j−c_j。

不能先发布减 m_j、再加 c_j；中间会出现假的低上界。m_j 来自已做的 accepted-weight 读取，c_j 来自完整 local grouping，不需要重扫 Positions 数频率。零质量 job 未完成仍可能缺物理坐标，不能由 U 精确就宣布 payload 完整。

一个 source-shaped 正例用 500 行 ABAC_iZ_iABABD_i、每行权重 2，再加 EF:1600。初始 AB3000、BA2000、EF1600；selector 在选 AB 后被冲突 BA 阻挡。1,500 个 AB 坐标在四线程、restart 128 下切成 384/384/384/348。

前三 job 各 m=768、c=256 完成并交付相关旧损失后，U=1464，BA upper=464。EF1600 的计数与 AB 角色无关，能在最后 348-position job 未完成时获得数值 admission 证书。这是符合实际 task-cut 约束的有限模型机会，不是测得的原生 overlap 或速度。

若 guard 始终保持 raw f 不收紧，后来旧频率不会超过较早所选 f，下面的 selective readiness 本身通常无法创造 prepare overlap。渐进 credit 是这个正例不可缺少的一部分。

### 9.2 selective old-loss closure 定理

对未选旧候选 q=(u,v)，旧 producer r=(a,b) 只可能在 u=b、v=a 或 q=r 时毁掉 q 的 occurrence。操作中 q=r 已被选走；共享 head 或共享 tail 本身并不构成影响。

未交付的旧损失非负，故当前旧标量是逻辑完整旧计数的上界。对低于 F 后语义置零的情形，应使用 Φ_F(c)=c 当 c≥F、否则 0；不是声称零 cell 上界了任意正残余物理质量。

若 q 的所有相关 producers 都 loss-closed，其 selectable 标量精确。q 若也是覆盖全部旧 upper priorities 的修复后堆顶，且严格击败所有 pending U，就等于包含隐藏 births 的完整逻辑队列赢家。其他旧 key 过于乐观只会阻挡进度，不会让已精确的 upper-heap head 错赢。

loss-closed 必须同时包含：未来任务发现已封口、全部有关 prepare 已产生完损失、全部损失恰好一次应用并 ACK、count 更新先于完成状态对协调者可见。P done、W done、loss ACK、guard closure、payload closure 与实际 scope join 是不同状态。

可扫描至多 256 个 producer descriptors：存在未 closure 的 r 且 r.tail=u 或 r.head=v 就等候。无需默认新增 O(vocabulary) role arrays。若使用计数数组，必须先登记、后分发、只减一次，计数也不能用会在 256 溢出的 u8。

这个数值证明不授权在异步 reader 下普通删除共享 map。joined no-K0 的结构删除安全性来自 join；异步版本须另外提供合法并发后端或不可变读视图。

### 9.3 物理 interval readiness 和 word clipping

对 job j，令 f_j、l_j 为首坐标和合法 inclusive 尾上界，L_j 为两输入 span 之和，N 为 corpus 槽数，S 为所有可能出现的 frozen old ID 的最大固定 span，包括 stale interior 的旧 ID。

    W_j=[f_j,min(N,l_j+L_j))。
    R_j=[max(0,f_j−2S−1),min(N,l_j+L_j+2S+1))。

R 覆盖 matcher、左右邻居及至多第二跳；W 覆盖应用时至多三个 stores。算术先 checked/saturating，再裁边界，不能先 wrap。记录 bounds 是新的私有 API seam；next restart seed 可作 inclusive 上界，不能减一，因为重复坐标可跨 restart。

令 P_j 是完整 reader phase。加入 P_j→W_j；较早 cohort 的 W_i 若交 R_j，则 W_i→P_j；同或较早 cohort 的 P_i 若交 W_j，则 P_i→W_j。rank=(cohort,phase) 单调增加，所以无环，并保证每个 reader 看到完整早期前缀和未写入的本 cohort。

该图证明仍不等于 Rust happens-before。AA 迟发现 jobs 时要保留 joined 边界，或从 admission 起登记覆盖全部未来 reader/writer 的 aggregate promises；零个已发现 job 不等于没有未来 job。

若 J 为 jobs、Q≤256 为 producers、J_max 为单 producer 最大 job 数、E 为实际报告的 interval intersections，E 可为 Θ(J²)。有序数组与完成时重查可用 O(J) metadata，约 O(QJ log J_max+E) bookkeeping；没有线性最坏时间保证。宽 halo、稀疏坐标的大 hull 和密集 overlap 可吞掉并行机会。

word clipping 可取首尾覆盖词的 body [s,e)，其中 e 是右 separator：R'_j=R_j∩[s−1,e+1)，W'_j=W_j∩[s,e)。它只对 genuine historical fresh postings 的 provenance 成立，任意 resident 坐标不足以证明不跨词。

现有 fresh materialize 会丢弃 starts 词目录；clipping 需保留已构建的每词一个 usize，即真实 O(words) resident memory，及每非空 job 两次 O(log words) 查找。对真实 same-pair seed/tail bounds，W clipping 实际冗余；主要收紧的是 R。以上均不构成 native speedup 结果。

## 10 有限模型证据及不可外推范围

以下数字来自分开的标准库数学模型和源码形状核查。它们没有导入生产仓库。各行可能重用 corpus 集合、参考语义或事件生成器，不能求和称作“不同测试总数”；同一 case 的多个 assertions 也不是独立试验。

| 证据对象 | 已记录规模 | 支持与边界 |
|---|---:|---|
| exact family 独立物理切口模型 `check_physical_cuts.py` | 11,050 cases；24,967 guards；42,456 descendant 检查；24,012 rejection 等价检查 | 验证 no-coalescence、完整平局与零坐标；非原生 geometry/borrow 测试 |
| FRONTIER loss-only `check_frontier_loss_only.py` | 13,271 cases；14,711 eager-rule、13,415 old-count 检查；2,852 live 零权重坐标 | 运行时模型无 birth-family maps，最终坐标全比；ordinary-delete 变体重跑同一集合，不另算新 cases |
| no-K0 独立退休模型 `check_unfrozen_counts.py` | 12,550 corpora；25,100 deletion/tombstone 等价比较；47,604 normalized exact-M/tie 比较 | 两种 guard 的 ordinary deletion 与 tombstone 等价；含长度拒绝的高物理质量缺失 key，K₀ 仅用于独立 oracle |
| cheap guard 完整 admission schedules | RAW 和 FRONTIER 每策略各 15,000 seeded corpora、各 118,075 rule 比较 | 最大 cohort 分别 3、4；两策略样本不能直接当 30,000 个不同 corpus；非真实负载机会比例 |
| cheap guard family 界 | 108,347 compatible cohorts；196,805 coherent families | 包括 raw/frontier 等频检查；不独自证明完整 schedule |
| producer-complete publication | 38,235 path/fragment traces；344,115 reducer 等价比较 | stable grouping、零 leaves、XX、stale directory 与 routing 算术；以 trace 为条件，非 greedy admission 证明 |
| 终端 floor 独立模型 | 3,000 corpora；524,352 certificate checks；80,182 成功证书 | directed/depletion/compatible 证书，AA、zeros、caps、inactive low IDs；另有 hidden-frontier 定向反例 |
| 渐进 pending guard | 1,956 nonempty、360 empty cases；14,278 job subsets；24,832 monotonic edges | 使用既有独立 boundary recorder；另有 240-corpus reachable-history supplement，非全部 arrival schedules 穷举 |
| selective loss readiness | 23,488 cases；32,150 admissions；6,668 次早期 loss seal 未闭、2,235 次早期 prepare 未完时 admission | 全部对齐 serial greedy；数值 singleton 模型，没有原生 lazy heap 或物理事件 |
| directed role lemma | 797,040 producer/candidate-role 比较；131,787 实际 destroyed-pair 比较 | 三符号词长 1 至 8；验证有向影响范围，非并发内存模型 |
| interval event 模型 | 1,500 cases；15,303 jobs；163,962 events；86,045 loads；42,313 stores | 共享先前语义生成器；顺序一致抽象事件；未执行 AA aggregate promises 或 Rust atomics |
| word-clipped geometry | 2,000 cases；534,424 probes（218,358 stale）；2,070,407 loads；1,420,686 queries | genuine postings 的 reference-access 覆盖；非并发 scheduler 或性能验证 |

全部所述有限检查在其明确覆盖域内成立。随机种子与 case 构造限制并不等价于任意输入、任意 fault、任意调度的机器证明。模型脚本名用于标识证据，不表示本附件包含可执行模型包。

尚需原生验证的是：完整 ordered-model parity、source task/codec geometry、borrow/atomic 内存序、errors/unwind、启动工作丢弃、reuse、zero/cap/target/alias 边界，以及时间对齐的 owned capacity 与 RSS。本文没有把源码中已有测试函数当作本次已运行测试。

## 11 文献关系与被排除的推论

以下为一手论文；它们提供局部更新、摊还、增量或协调理论的背景。本文的 exact priority、coherent family、终端 floor 与具体 scheduler 协议仍由前述推导承担，不能由论文标题直接推出。

1. [Bille 等 Space-Efficient Re-Pair Compression](https://arxiv.org/abs/1611.01479)：使用 fresh symbols、邻边更新及 lazy occurrence 清理的摊还思路。其无权字符串上的物理密度不能直接用本训练器的 weighted frequency 替代，尤其存在零权重词时。
2. [Bille 等 Practical and Effective Re-Pair Compression](https://arxiv.org/abs/1704.08558)：实用 grouping、计数表和批量维护是相关先例。非 stable clustering 可能打乱同 key 的空间 leaves；不能不付代价地替代 ordered concat 前的稳定分组。
3. [Zouhar 等 A Formal Perspective on Byte-Pair Encoding](https://aclanthology.org/2023.findings-acl.38/)：形式化压缩目标并讨论更快局部更新。其目标、AA 计分及队列假设需逐一匹配；总压缩近似不等于最后一条规则频率下界，也不等于本实现完整工作界。
4. [Kozma 与 Voderholzer Theoretical Analysis of Byte-Pair Encoding](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ESA.2026.80)：明确区分 overlapping raw occurrence 和 replaceable frequency。近似结论容忍某种 tie 变化，不意味着输出词表身份序列不变。
5. [Morgan Batching BPE Tokenization Merges](https://arxiv.org/abs/2408.04653)：冲突关系和 batching 提供对照，但其启发式及 delayed descendants 不能替代逐条相同 greedy prefix 的证明。
6. [Budiu 等 DBSP](https://www.vldb.org/pvldb/vol16/p1601-budiu.pdf)：线性筛选、投影和求和的增量代数适合描述标量 events。精确 argmax、有限整数 checked 错误、Positions 所有权与 payload 完整发布不自动线性化。
7. [Ameloot 等 Relational Transducers for Declarative Networking](https://arxiv.org/abs/1012.2858)：CALM 刻画其特定关系模型中的单调性与协调关系。它不能直接证明共享内存 trainer 必须有全局 barrier，也不能把递减计数上的 exact argmax 当成单调查询。
8. [Conway 等 Logic and Lattices for Distributed Programming](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2012/EECS-2012-167.html)：lattice merge 的结合、交换、幂等及完成信息值得借鉴。exactly-once sum 不具幂等性；知道任务 manifest 仍须证明全部任务和 delivery 已完成。
9. [Bailis 等 Coordination Avoidance in Database Systems](https://arxiv.org/abs/1402.2237)：是否可免协调取决于事务集合、merge 与 invariant 的共同性质。两个各自合法的 decrement 合并后仍可能 underflow；唯一物理 cut ownership 是本问题必须补上的约束。
10. [Tsai 等 A Revised and Verified Proof of the Scalable Commutativity Rule](https://arxiv.org/abs/1809.09550)：语义可交换性包含返回值和后续可观察行为。相同净和不等于 `fetch_sub` 返回轨迹相同；存在可扩展实现的定理不保证当前 atomic 热点没有 cache-line 争用。

据此排除几种看似简短但不成立的替代：用 top-H 当前频率作终端 floor；用 weighted count 当 posting occupancy；以 partial fragment 最大值当完整 family 最大值；无条件放过所有 old-old 同频；把 producer 完成当 guard 可退休；以完整标量代替完整零权重 payload；将 Root 当 leaf 做树归约。

同样不采纳“边界更新可交换，所以整个 trainer 无须 closure”的推论。fresh 标量非负损失可重新组织，并不消除 missing births 或指定 ID/priority 次序的信息要求；reuse 的 signed checked arithmetic 还可能因运算前缀不同而改变是否溢出。

## 12 固定源码依据和最终证据范围

源码引用固定在 [890ea1f58e65f8d8503f7cd3fc97ad1651eaf284](https://github.com/Yikai-Liao/tokenizers/tree/890ea1f58e65f8d8503f7cd3fc97ad1651eaf284)。所引 corpus、index、merge、mod、positions 文件已核对与数学/source audit 所用 a2b20f32 的相应 blobs 相同。

- [merge 的选择和 prepare apply commit 顺序](https://github.com/Yikai-Liao/tokenizers/blob/890ea1f58e65f8d8503f7cd3fc97ad1651eaf284/tokenizers/tk-train/src/trainers/bpe/merge.rs#L16-L123)
- [direct groups 和 payload-based finish](https://github.com/Yikai-Liao/tokenizers/blob/890ea1f58e65f8d8503f7cd3fc97ad1651eaf284/tokenizers/tk-train/src/trainers/bpe/merge.rs#L147-L280)
- [邻居读取与共享切口处理](https://github.com/Yikai-Liao/tokenizers/blob/890ea1f58e65f8d8503f7cd3fc97ad1651eaf284/tokenizers/tk-train/src/trainers/bpe/merge.rs#L360-L483)
- [route coalescing 和完整 birth floor reduction](https://github.com/Yikai-Liao/tokenizers/blob/890ea1f58e65f8d8503f7cd3fc97ad1651eaf284/tokenizers/tk-train/src/trainers/bpe/index.rs#L194-L430)
- [leaf-only concat 与 restart tasks](https://github.com/Yikai-Liao/tokenizers/blob/890ea1f58e65f8d8503f7cd3fc97ad1651eaf284/tokenizers/tk-train/src/trainers/bpe/positions.rs#L46-L131)
- [fresh word-directory 生命周期](https://github.com/Yikai-Liao/tokenizers/blob/890ea1f58e65f8d8503f7cd3fc97ad1651eaf284/tokenizers/tk-train/src/trainers/bpe/corpus.rs#L175-L201)；[激活历史和应用端点](https://github.com/Yikai-Liao/tokenizers/blob/890ea1f58e65f8d8503f7cd3fc97ad1651eaf284/tokenizers/tk-train/src/trainers/bpe/corpus.rs#L332-L408)
- [attempt restart 与实际 terminal caller](https://github.com/Yikai-Liao/tokenizers/blob/890ea1f58e65f8d8503f7cd3fc97ad1651eaf284/tokenizers/tk-train/src/trainers/bpe/mod.rs#L332-L417)

Vocabulary 的身份/resolve 证明对应 [已审计 a2b20f32 版本](https://github.com/Yikai-Liao/tokenizers/blob/a2b20f329d1b4a0d39b5721f4a7a678a80d0d9b2/tokenizers/tk-train/src/trainers/bpe/vocabulary.rs)。较新的 [a583a0 初始化实现](https://github.com/Yikai-Liao/tokenizers/blob/a583a0b4025495ef0006a6adea84c6139cb7b295/tokenizers/tk-train/src/trainers/bpe/vocabulary.rs#L181-L369) 已另行检查 InitialTokenIds 与原字符边界处理；此处不把旧行号冒认为新版行号，也不据该初始化变化宣称窗口方案已实现。

最终可成立的表述是：这些条件给出精确语义与实际可省工作类别的证明，并有有限模型与固定源码相互校验。它们尚未给出 native Rust 集成正确性或速度承诺；任何性能结论必须同时解释没省的工作、新增顺序尾部、完整内存峰值和可复现的整段训练时间。
