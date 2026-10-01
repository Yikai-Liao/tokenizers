# J单线程与四线程加速比

## 配置与范围

J源码`376363d25b6b1de917b9b8f68c11c0394ca37bb2`，branch `bpe/prepare-rule-aggregate`。同一release binary SHA-256 `7ef9b90385e373d420c78e73ae73f371d9c2aae53d080b95cec65b6006849ea3`，使用原Trainer接口；初始化/merge分别1/1与4/4线程，feed均串行。输入536,870,289B中文、none、vocab50k/min_frequency2、u32/AtomicU32、flat32 posting、batch256。输入SHA-256 `a0d40d4102cba1e933f25e5ccd17552d2eaebaec4ef770bc39d8e46e740b4d4f`。

仅单线程→四线程各一次，不混用历史版本或J旧screen。通过benchmark变量`HF_BPE_BENCH_WORKERS`设置已有parallelism线程数，实际stats校验1/1和4/4；生产源码未修改。默认AHash种子及物理词顺序未固定。

## Feed口径

feed读取文件、逐行调用切分回调、以CompactString哈希表累计片段频次，保留重复行的权重。本次none时整行是一个片段；字符整数编码和初始pair计数属于train初始化。

生产feed已有maybe_par_bridge/map/reduce并行路径。benchmark runner在feed前显式关闭parallelism、train前开启，因此此处串行feed反映测量配置。未测试并行feed收益；端到端2.70×包含该串行限制。

## 总体结果

| 阶段 | 单线程秒 | 四线程秒 | 加速比 T1/T4 |
|---|---:|---:|---:|
| Feed（串行） | 3.768 | 3.916 | 0.96× |
| 初始化合计 | 19.027 | 5.547 | 3.43× |
| Merge合计 | 46.630 | 13.528 | 3.45× |
| 结果构造/清理及返回区间 | 4.882 | 4.551 | 1.07× |
| 完整训练 | 70.539 | 23.626 | 2.99× |
| 端到端：feed+train返回 | 74.307 | 27.541 | 2.70× |

四线程训练加速2.99×，效率（加速比/4）74.6%；端到端加速2.70×。初始化与merge均约3.4×，串行feed及结束后的构造/清理区间约1×，使完整结果低于两个核心阶段的加速比。

## 初始化细项

| 阶段 | 单线程秒 | 四线程秒 | 加速比 T1/T4 |
|---|---:|---:|---:|
| 字母表 | 1.382 | 0.411 | 3.37× |
| 语料构造合计（含字母表） | 3.961 | 1.088 | 3.64× |
| 长度/词区间测量 | 0.710 | 0.195 | 3.65× |
| 最终corpus填充 | 1.868 | 0.482 | 3.88× |
| 初始空间路由（含compact） | 5.615 | 1.644 | 3.41× |
| 其中：route compact | 2.203 | 0.756 | 2.91× |
| 初始pair计数合计 | 9.341 | 2.759 | 3.39× |
| 其中：radix排序 | 5.825 | 1.826 | 3.19× |
| 其中：分组频率累计 | 1.151 | 0.303 | 3.80× |
| 其中：posting安装 | 2.364 | 0.630 | 3.75× |
| 初始candidate堆构造 | 0.075 | 0.022 | 3.37× |
| 初始权重lookup构造 | 0.016 | 0.013 | 1.28× |

初始化包含语料构造、route、初始count及heap等；tokenize已包含alphabet/measure/fill，route包含compact，count包含sort/group/install及相关释放。父项与子项不相加。corpus分配计时只有约42/50微秒，重用weight lookup的merge入口计时接近零，完整原始值保留在JSON，不将其比值解读为扩展性。

## Merge细项

| 阶段 | 单线程秒 | 四线程秒 | 加速比 T1/T4 |
|---|---:|---:|---:|
| 候选选择 | 0.257 | 0.282 | 0.91× |
| Plan/AA fallback | 0.073 | 0.037 | 1.95× |
| Delta合计（含fused prepare） | 20.507 | 7.061 | 2.90× |
| 其中：fused prepare | 20.086 | 6.769 | 2.97× |
| Corpus rewrite | 1.703 | 0.622 | 2.74× |
| Owner commit | 24.025 | 5.449 | 4.41× |
| 收尾route处理 | 0.018 | 0.017 | 1.02× |

fused prepare包含在delta内。candidate选择、Plan/AA fallback和最后route处理较短，不以一次比例推断机制。owner commit的4.41×是此次观测，默认随机hash/物理顺序未固定，数据结构按owner分区和临时聚合工作也随线程数变化，未隔离超线性结果的原因。

结果构造/清理及返回区间按`train-initialize-merge`计算；这次没有给单项drop加新计时器。此前H独立诊断确定此区间主要为owner释放，J仍沿用串行drop，因此可作为整体扩展限制的线索，不能将本次区间全部标成精确owner-drop时间。

## 正确性、工作量与资源

完整模型SHA-256 `d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd`，两次均50,000 vocab / 29,243 merges / 1,429,915 unique words；N/E/pairs=204,660,029/203,230,114/2,697,517，initial corpus/posting=849,691,660/802,967,868B。两个运行均1550批/1435 fused批、125,409,599次posting访问；初始count均stable_radix16。校验见[signature](rule-aggregate-scaling.signature.json)。

| 资源 | 单线程 | 四线程 |
|---|---:|---:|
| 峰值RSS | 4.433 GiB | 4.433 GiB |
| 最低MemAvailable | 3.083 GiB | 3.137 GiB |
| 进程VmSwap采样峰 | 0 B | 0 B |
| Prepare aggregate scratch capacity汇总峰 | 690,416 B | 2,106,304 B |

内存停止门槛1GiB，两次均通过。scratch为Vec capacity组件统计，不等同RSS或实测同时驻留峰值。

## 复现

先完成构建，再串行运行，计时期间不运行其它构建或测试：

```bash
python3 build_native_fair.py /root/code/tokenizers-worktrees/prepare-rule-aggregate --label rule-aggregate-scaling
python3 run_native_fair.py --case j-scaling-1t-reproduction \
  --worktree /root/code/tokenizers-worktrees/prepare-rule-aggregate \
  --build-root .build/native-rule-aggregate-scaling \
  --binary target/release/hf-bpe-native-rule-aggregate-scaling \
  --corpus .build/gb-corpus/zh-512m.txt --output results/j-scaling-1t-reproduction.jsonl \
  --initialization-workers 1 --merge-workers 1 --atomic-corpus
# 四线程：改为 --initialization-workers 4 --merge-workers 4，使用独立case/output。
```

原始结果为`rule-aggregate-scaling.{1t,4t}.jsonl`及各自environment/stdout/stderr；所有阶段加速比数值保存在[ratios](rule-aggregate-scaling.ratios.json)。
