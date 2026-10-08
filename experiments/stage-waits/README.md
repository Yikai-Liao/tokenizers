# BPE 阶段等待实验

每个初始索引分区连续完成排序、计数、编码，全部分区完成并检查错误后再发布。

从 `8faaff79` 独立分支开始。完整对照、冻结 binary、测试记录和证据归档见[实验报告](https://github.com/Yikai-Liao/tokenizers/blob/experiment/stage-waits/experiments/stage-waits/REPORT.md)。
