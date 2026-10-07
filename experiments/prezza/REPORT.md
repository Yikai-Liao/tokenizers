# Prezza 语料表示消融：初步实测报告

日期：2026-10-08。基线为 fork 最新 main `8faaff79d859bfd6b2417cfe8c93ea2851c3aaca`；
实验实现为 `07502ccf16efa328120a1a844448cec3692acd5c`。

在已完成的 **512 MiB 中文、非 Byte level、4 线程单次诊断**中，端点方案训练更快：
Prezza 在 50K／100K 下分别慢 **15.6%／16.1%**。50K 时 Prezza 峰值 RSS 高 **2.7%**，
100K 时低 **3.0%**。两边完整词表 ID 和合并顺序一致。

这些是一次配对观察，不能当成重复实验的稳定幅度，也不足以比较 `T1/T4` 并行加速比。
用户随后明确要求停止详细 Prezza 测试，转向 Owner 的两个候选方案；因此没有完成
1／4 线程重复矩阵，也没有补测英文、代码和 Byte level。取消记录一并保留。

## 改动与泛化

依据 Bille、Gørtz、Prezza 的
[Practical and Effective Re-Pair Compression，§3.1](https://arxiv.org/abs/1704.08558)，
以及作者固定版本的
[skippable_text.hpp](https://github.com/nicolaprezza/Re-Pair/blob/ffb411d8ce9c1232980ec67bee7e678c0b88fd5c/internal/skippable_text.hpp)，
实现原地空槽借用、存活位图和块跳距。Rust 代码独立编写，未复制 GPL C++ 实现。

按用户要求直接复用 fork 的 `slot_bits`、`U16Slots`、`PackedU24Slots`、`U32Slots`：
按最大保留初始 ID 加一选择 16／24／32 位槽，预留分隔符编码。16／24 位槽借用
合并后紧邻的空槽拼接完整 32 位 ID；32 位槽直接保存完整 ID。稀疏高位初始 ID、
特殊 token、affix、过滤和身份复用保持原有语义，不加半字初始字母表限制，也不重编号。

Prezza 的邻居和 occurrence span 从位图／跳距恢复，因此不保留端点布局的共享跨度表或
不同跨度复用时的逐位置 span 平面。仍保留原有批选择、初始计数、位置索引、权重区间、
合并事件及发布机制。本实验比较的是 **同一训练引擎中的两种语料表示**，不是整套
Re-Pair 压缩器与 Tokenizer 的比较。

位图用 `fetch_and` 清位，防止不同合并共享一个 bitmap word 时丢更新；槽和跳距写入仍属于
不相交的物理合并区间。并行写期间仅从原子位图／跳距恢复本区间内的两个 successor；
相邻合并保留其左端存活坐标。token ID 读取遵循既有 joined 阶段契约。

## 实验口径

语料来自固定版本中文维基 `20231101.zh`，数据集 revision
`b04c8d1ceb2f5cd4588862100d08de323dccfbaa`。512 MiB 是目标取样量，取完整行后实际为
**536,870,289 字节**；SHA-256 为
`a0d40d4102cba1e933f25e5ccd17552d2eaebaec4ef770bc39d8e46e740b4d4f`。
未重复文本扩充。使用仓库 `Whitespace` 预分词，含标点切分，不做 ByteLevel 映射。

预处理产生 12,971,649 个去重加权词、150,678,717 个物理槽（含分隔符／哨兵），
初始有效词表为 20,755。50K／100K 实际词表均达到目标，分别产生 29,245／79,245 条合并。
训练参数：`min_frequency=2`，无 prefix／suffix／max_token_length。

运行于 KVM 的 Xeon Gold 6140，6 个虚拟 CPU、约 16 GiB 内存；本次诊断绑定 CPU 0–3。
两臂使用**同一实验二进制**，仅 `BPE_CORPUS_LAYOUT` 不同，避免接入代码的二进制差异。
诊断打开 `BPE_CORPUS_STATS`；未改动 main 的独立二进制另外通过小输入精确模型对照。

构建使用同一锁文件、release、fat LTO、单 codegen unit，无 `target-cpu=native`。
runner 使用 `do_train` 的公开接口；词频 JSON 加载、调用者 map 重建在训练计时之前，
模型序列化在之后。RSS 为序列化之前的进程 HWM，包含启动／加载，不是增量训练分配。
完整 job、source／binary hashes、CPU 绑定、壁钟／CPU 时间与 RSS 采样见
[evidence](evidence/)。

## 实测结果

| 目标词表 | 端点训练秒 | Prezza 训练秒 | Prezza 时间变化 | 端点 HWM MiB | Prezza HWM MiB | Prezza HWM 变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50K | 13.604 | 15.724 | +15.6% | 2620.56 | 2691.41 | +2.7% |
| 100K | 17.455 | 20.258 | +16.1% | 2775.43 | 2692.83 | −3.0% |

训练 CPU 时间分别是：50K，端点 42.793 秒、Prezza 52.050 秒；100K，端点
53.654 秒、Prezza 63.904 秒。增加分别为 21.6%／19.1%。不把 CPU 时间当成壁钟时间，
也不把线程总 CPU 时间据此解释为某个局部 load 的成本。

另记录了实际槽数组容量及位图／跳距分配，不用输入文件字节推算 U：

| 目标词表 | 端点布局 | 端点语料分配字节 | Prezza 布局 | Prezza 语料分配字节 |
| --- | --- | ---: | --- | ---: |
| 50K | 16 位 | 301,357,436 | 16 位＋位图＋跳距 | 339,027,116 |
| 100K | 24 位 | 452,036,154 | 16 位＋位图＋跳距 | 339,027,116 |

此处包含槽 capacity／guard 和两个块数组；共享跨度表、权重、词表、posting、scratch
另计。端点在刚物化时的共享跨度表为 166,040 字节，随后随词表增长；Prezza 不保留它。
这些分配数与进程 HWM 是不同口径，不能把组件节省比例当作整个训练器节省比例。

## 正确性与边界

- 端点与 Prezza 的无默认 feature 构建各通过全部 104 项库测试；Prezza 默认 feature
  构建也通过全部 104 项。`clippy --all-targets -- -D warnings` 通过。
- 新测试用独立存活位置列表验证随机合并、跨 64 槽长空区、三个槽宽、分隔符、全 32 位 ID，
  包括低半字等于分隔符编码的 ID，并验证相邻并行清位。
- 既有语义/reference 测试覆盖 weighted counts、AA、reserved IDs、affix、身份复用、
  strict birth length、溢出、失败与多 worker trace。
- 不可变 main 与实验 Prezza 完成 32 次小输入精确模型检查，覆盖 core／pipeline、
  四种预分词和 1／4 线程；大中文的四次诊断全部精确匹配。
- 详细矩阵在首个 1 线程 warmup 中依用户指示中断，没有完成配对块。
  [取消报告](evidence/cancelled-matrix/report/REPORT.md) 明确标记不可作性能结论。

最终判断：当前证据支持继续保留端点方案作为速度优先的默认实现；Prezza 在 100K 下展示
了以时间换内存的可用取舍。更宽初始域会自动使用 24／32 位 Prezza 槽，其主项分别约为
3.25U／4.25U，不能把原半字的 2.25U 无条件推广。没有完成这些宽域的性能测量，也没有
完成 T1/T4，因此不宣称它全面更省内存或改善并行加速比。

复现入口和并发契约见 [README](README.md)，具体检查见
[VALIDATION.json](evidence/VALIDATION.json)，发布文件内容哈希见
[ARTIFACTS.json](evidence/ARTIFACTS.json)。每次运行的模型以协议的词表键排序形式去重保留，
ID 与合并顺序均不改变；映射关系见各目录 `DEDUPLICATED_MODELS.json`。
