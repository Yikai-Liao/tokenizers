# B 热点诊断采样

这是对已测B的32MiB单次perf诊断，非512MiB性能对比或矩阵。用于缩小prepare热点；不按它推算512MiB加速比。

命令：`perf record -F 199 -e cycles:u -o .build/fused-b-hot.perf -- target/release/hf-bpe-native-fused-direct /root/code/tokenizers-bpe-benchmark/data/text/zh-32m.txt none reference 50000 2`。

源提交c1ee2019；binary SHA256：c23c02edeb11c14687c4f30695cc2d19475646c0e8fd75ee6755b9b440dcb330。输入SHA256：6624193cbcc72657f766bf68aee0fe78be129578137ad7d6b4870d03147f4e0d。约2K cycles samples，无lost samples。

| samples比例 | 位置 |
|---:|---|
| 29.57% | fused prepare worker (inlined queries) |
| 14.84% | owner closure (initial count or commit; mangled paths retained in raw report) |
| 9.16% | owner closure (initial count or commit; mangled paths retained in raw report) |
| 3.37% | Output::birth |
| 3.33% | Output::remove |

prepare worker为最大单一采样位置，weight/selected操作被内联，不能从该报告单独量化两者。perf annotate在当前工具上崩溃，本轮不继续工具排查。B2查询改动是基于访问模式提出的候选，收益待512MiB测量。原perf文件和原始符号报告保留在ignored `.build/fused-b-hot.*`。
