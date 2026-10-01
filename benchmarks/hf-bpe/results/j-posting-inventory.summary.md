# J结束时的posting分配与清理成本

## 回答

大量小posting堆分配是明确存在的机制：512MiB/none/50k/min2的J376363d2结束时有 **7,274,631个独立堆缓冲区**，其中4,447,125个（61.13%）容量不超过8个u32，即请求载荷最多32B。另有3,297,497个内联posting。请求载荷未包含allocator元数据、size-class取整或页碎片。

“结果构造/清理及返回”不是同一种工作。此前[H精确区间诊断](optimization-h-cost-probe.md)测得：post-merge 4.779766秒，owner drop 4.751972秒（99.48%），vocab构造11.619ms、merge字符串构造9.666ms。J保留同样的SmallPosting与串行owner销毁机制；本次J新增的是对象数量统计，没有把H的精确drop时间当成本次J时间。J线程对照中的4.551秒是train-init-merge残差，见[线程报告](rule-aggregate-scaling.summary.md)。

## 为什么会逐个释放

`SmallPosting = PackedPosting<u32, 2>`，对象16B，最多两个位置放在对象内；超过阈值则持有一个Vec分配。Drop在capacity非零时重建Vec并释放其缓冲区。Owner哈希表销毁遍历全部entry，对每个堆模式posting调用一次释放。u32元素本身没有逐元素析构。

结束时owner表还有10,572,128个entry，其中7,274,631个持有堆缓冲区；表里包括已失去候选资格而仍被保留的entry。存储位置总数207,224,101，是物理posting长度，包含历史记录，不等于当前有效边数。

Candidate是两个u64组成的Copy结构体。14,225,344个candidate分布在4个连续BinaryHeap backing Vec中，不代表同样数量的独立堆分配。

## 堆posting容量分布

容量单位为u32元素；字节列为capacity×4的请求载荷。

| 容量 | 分配数 | 占堆posting | 载荷容量 MiB |
|---|---:|---:|---:|
| ≤4 | 2,619,898 | 36.01% | 39.98 |
| 5–8 | 1,827,227 | 25.12% | 42.96 |
| 9–16 | 1,183,000 | 16.26% | 53.03 |
| 17–32 | 725,296 | 9.97% | 63.40 |
| 33–64 | 426,426 | 5.86% | 73.56 |
| 65–256 | 377,094 | 5.18% | 174.58 |
| 257–1024 | 100,724 | 1.38% | 173.42 |
| >1024 | 14,966 | 0.21% | 150.70 |

全部posting堆载荷容量771.62MiB；不超过8项的分配虽然占数量61.13%，只占载荷字节10.75%。Owner表capacity估算528.00MiB，4个candidate堆backing capacity合计329.29MiB；这些是结构容量统计，不是RSS，也没有计allocator开销。

## 证据范围

这确认了数百万个小分配及逐个释放的路径。尚未独立分离哈希表遍历、allocator free、缓存未命中各自的墙钟成本，因此不能把整个owner-drop都归因给allocator。此前并行owner释放K的两对清理区间改善，但完整训练未建立收益，已[停止采用](optimization-owner-parallel-drop.aggregate-n2.md)。

初始化还包含大数组的扫描、填充、排序和posting安装；构造成本不能整体用小对象解释。这里回答的主要是merge之后的结果构造与清理区间。

## 本次诊断与复现

独立副本`.build/native-j-posting-inventory`从J scaling构建副本派生；只在merge计时结束后、结果构造前扫描owners。补丁见[j-posting-inventory.patch](j-posting-inventory.patch)，完整输入、源码/runner/binary hashes、配置和资源采样见[environment](j-posting-inventory.environment.json)。JSON计数见[counts](j-posting-inventory.counts.json)，原始输出见[stderr](j-posting-inventory.stderr)。

扫描耗时191.301ms，新增扫描改变缓存状态，本次train 26.066秒仅作诊断，不参与版本排名。模型与J正式四线程一致，N/E/pairs及工作量门槛通过；RSS 4.44GiB、最低MemAvailable 3.04GiB、进程VmSwap 0B。

```bash
# 复制native-rule-aggregate-scaling到native-j-posting-inventory后应用补丁，
# runner包名与依赖路径改为新的副本，保持原profile及Cargo.lock。
CARGO_TARGET_DIR=target /root/.cargo/bin/cargo build --offline --release \
  --manifest-path .build/native-j-posting-inventory/runner/Cargo.toml
python3 run_native_fair.py --case j-posting-inventory-reproduction \
  --worktree /root/code/tokenizers-worktrees/prepare-rule-aggregate \
  --build-root .build/native-j-posting-inventory \
  --binary target/release/hf-bpe-native-j-posting-inventory \
  --corpus .build/gb-corpus/zh-512m.txt --output results/j-posting-inventory-reproduction.jsonl \
  --initialization-workers 4 --merge-workers 4 --atomic-corpus
```

## 用户建议：bumpalo或对象池是否值得尝试

值得作为候选。posting载荷只有u32，无元素析构，arena拥有其缓冲区可以把独立free变为整块释放；若Entry也没有逐项Drop，owner哈希表的销毁也可能省去逐项析构遍历。该收益仍需实际实现和完整train验证，现有4.551s收尾区间不是allocator可消除时间的精确测量。

本次进一步查过J flat32 birth路径：先汇总count，`SmallPosting::with_capacity(count)`，再`append_reversed_reserved`，当前主路径已经一次预留，动态Vec扩容并非主要障碍。真正的容量问题是训练中selected pair和frequency低于floor的entry会被remove并立即释放；Bump没有通用的单对象回收，改为整次训练arena会保留这些已退休posting。终点809MB容量不能预测累计arena峰值，需要累计新分配/退休容量才能判断。

bumpalo的Bump是Send但不是Sync，不能直接作为所有Rayon worker共享的分配器。可研究每owner独立arena并保持明确排他访问、最后统一释放；allocator归属与posting raw pointer生命周期需同步调整。当前SmallPosting用std Vec::from_raw_parts重建并释放，arena内存不可交给该旧路径，也不能仅把Vec对象放入Bump而保留其系统堆缓冲区。

PMR提供多种资源；Bump更接近monotonic_buffer_resource。针对中途频繁remove的小posting，按owner、容量4/8/16等分级且带freelist的pool/slab更适合复用退休空间。可以先限定容量≤8的对象，覆盖61.13%的终点堆分配数量，大posting保留原路径。它会新增free-list管理及容量统计成本，尚未实现或测量。

最小验证顺序：先补累计posting分配/退休容量以确定arena内存代价，再从J派生一种有清楚生命周期的候选；沿用模型/资源gate，直接初始化/commit/清理与完整train共同决定去留。这里仅记录用户建议的判断，没有新增候选或训练。

参考：[bumpalo官方说明](https://docs.rs/bumpalo/latest/bumpalo/)、[Bump的线程与reset约束](https://docs.rs/bumpalo/latest/bumpalo/struct.Bump.html)、[C++ pool资源规范](https://eel.is/c++draft/mem.res.pool.overview)、[C++ monotonic资源规范](https://eel.is/c++draft/mem.res.monotonic.buffer)。
