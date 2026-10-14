# 同一二进制的 mimalloc 对照

Chinese ByteLevel 256MiB, vocab 100000, min_frequency 2, workers 4 pinned CPU0-3, two reversed blocks; no observer

Debian mimalloc 3.0.1+ds-2.1 只解包到独立诊断目录，未替换系统库。按 [mimalloc 官方说明](https://github.com/microsoft/mimalloc#dynamic-override) 使用 LD_PRELOAD，仅对原生 benchmark 进程生效。每个实现的两种 allocator 使用相同 ELF SHA256。实际 Rust ELF 的五个分配 API 绑定与运行时映射均已验证，见 manifest；绑定探针不在计时运行内。

| 实现 | System wall/s | mimalloc wall/s | 配对 wall 变化 | System CPU/s | mimalloc CPU/s |
|---|---:|---:|---:|---:|---:|
| main | 16.685 | 15.720 | -5.9% | 53.774 | 51.723 |
| history-buffer | 23.137 | 20.152 | -12.8% | 71.041 | 62.960 |
| current | 25.102 | 23.813 | -5.1% | 78.627 | 74.551 |

时间/CPU 列为两轮中位数；百分比先在每个对应块内计算，再取中位数。全部 12 次完整模型一致、swap 为 0。未使用 phase、allocation 或 perf observer。

当前版与 main 的绝对 wall 中位数差距为 8.417 → 8.093 s，基本没有缩小。旧 buffer 版差距为 6.452 → 4.432 s，有明显收窄。

这项直接替换表明：分配器后端对旧 buffer 版的影响更大；仅更换分配器不能解释或恢复当前版相对 main 的大幅退化。它也没有消除分配、容量管理与复制操作本身，因此不能用此反推这些操作都无成本。前面的 native-CPU 采样仍是估计，此处使用实际替换后的完整训练结果。

当前版进程峰值 RSS（含输入驻留、模型校验前）约 2.87 → 3.97 GiB；此表不等于训练独占内存。

## 复现

使用已有 frozen runner 二进制、同一 job.json 和完整模型 oracle。先用 apt-get download 下载 libmimalloc3=3.0.1+ds-2.1，再用 dpkg-deb -x 解包。run_allocator.py 中库路径按本机解包位置调整；为每个原实现创建一个指向同一 ELF 的 `-mimalloc` 符号链接。只有后缀变体设置 LD_PRELOAD，System 对照清理继承环境。

```sh
python3 run_allocator.py --arms main main-mimalloc history-buffer history-buffer-mimalloc current current-mimalloc --cases zh-bytelevel --workers 4 --rounds 2 --modes core --label allocator-mimalloc-zh4
python3 summarize_mimalloc.py
```

若在其他环境复现，保留现有 runner/source 构建协议，并记录实际 allocator 版本、库/二进制哈希、输入哈希及 CPU affinity。
