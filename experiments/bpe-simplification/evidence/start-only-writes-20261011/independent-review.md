# 独立审查：BPE 起点写入与连带清理

建议保留核心候选。从生产差异及直接调用关系看，没有发现 blocking defect。它确实减少跨阶段协议和几何状态的维护，而非只减少字段或行数。性能是否支持采用，仍以主 agent 的配对测量为准，本次审查没有运行性能或全量测试。

审查基准是 `567f20969c2c98fb467c6061d5ef84a39faacac8`；检查了 `/tmp/bpe-start-only-writes-evidence-20261011/production.patch`、纯生产工作树和 candidate 的测试 oracle，并完整阅读用户的 Writes 连带清理补充。下文生产行号以 `/tmp/bpe-start-only-writes-production-20261011/tokenizers/tk-train/src/trainers/bpe/` 为根；测试行号以 candidate 对应目录为根。CodeGraph 未索引隔离工作树，随后用源码与 `rg` 核验。

## 实际维护收益

旧实现让 preparation 决定 `Compact`/`Occurrences`，让 `Writes::fresh` 再解释一次 ID 跨度，让 `Writes::apply` 重新构造完整 `Match` 或转交 occurrence 快照，最后由 `Corpus::apply` 写入。日后调整几何来源或载荷，需要共同理解两文件的这套协议。

新实现中，fresh 与 cohort 两个 producer 都只记录已经接受的起点，Job 保存 `starts/replacement/changes`；`Corpus::apply_merges` 按自己拥有的 `occurrence_spans` 决定一次读取方式。规则接受仍由 matcher 负责，Corpus 写入入口不再重新匹配。这删除了 mode 转运、跨阶段 `left/total`、逐 occurrence 三坐标载荷，以及 `fresh → record → apply → Corpus::apply` 调用层。生产证据是 `merge.rs:186`、`:504`、`:601`、`:682`，以及 `corpus.rs:377`。

减少的知识是“准备结果该用哪种写入载荷和如何恢复它”，不是全部跨度规则。matcher 仍需计算范围来验证接受，apply 仍需恢复范围来写入，两处计算有不同职责，不能宣称跨度公式只剩一份。Corpus 原有 `span`、identity preparation 和 activation/occurrence 状态继续保留，符合它们仍有实际消费者的事实。

## 语义与边界核验

- `merge.rs:116` 的 preparation 返回完整 Job 集合后才进入 application；并行 application `collect` 完成后才在 `:124` commit，成功后才返回 records。纯生产 patch 没有改变 owner commit、publication、selection、Neighbors、activation history 或 Parity。
- `corpus.rs:378` 先处理空 starts，不会读取第一项。None 在第一条 store 之前从第一个实际接受的起点读取旧左 token 和旧右 token 的 ID span，随后缓存复用；不会从 replacement 或已改写的首个 occurrence 推断几何。
- Some 在每个 occurrence 的 token/span store 之前读取两个自身起点的跨度，保留旧 kernel 的 token 首点、必要右点清空、尾点、必要右 span 清零、首尾 span 更新顺序。相邻区间 `[start, after)` 的下一个 match 从 `after` 开始，不会触及前一个 match 的端点；任意应用次序因此不应改变当前 occurrence 的读取。
- 同一 Job 属于同一规则是 None 共用几何的前提；全轮所有 Job 的接受区间互不重叠是恢复正确的前提。现有 producer 每个 Job 绑定一个 rule，AA 过滤和 cohort 前向扫描仍在，改变只发生于输出表示。
- 空写入任务仍交付 `job.changes`，没有因 starts 为空而删除 removal-only 事件。删除 `Match.right` 后，生产 `Match` 仍表达 preparation 范围，应用的局部 `right` 不需要重新泄露到 Job。

## 非阻塞发现与代价

1. **就地契约：已处理。** 初审 `corpus.rs:373` 写了 one rule 和 nonoverlapping，但容易被理解为只限制单次调用中的 starts。维护者若未来合并或并行划分 Job，需要直接知道非重叠是整轮跨 Job 的要求，以及 None 必须在任何 store 前缓存同 rule 几何。主 agent 已在 `:373` 明确全轮所有 Job 的区间不重叠，在 `:383` 明确 None 缓存时机，在 `:401` 明确 Some 读取自身旧几何。我已只读核验最终源码；无剩余建议。

2. **测试规则快照：已处理。** 初审 candidate `merge.rs:195` 的 `rule: (Pair, u32)` 仅由 `assert_write_orders` 消费 replacement，Pair 没有支持其“实际接受结果属于该 rule”的证据作用。主 agent 已在 candidate `merge.rs:776` 以冻结的 start/right 读取原 token 并核对 rule pair，再开始预期或实际写入。我已只读核验；这是现有 oracle 的局部断言，不增加测试矩阵。

3. **两 arm 的 token kernel 重复可以接受。** 生产 `corpus.rs:391` 与 `:406` 分别保存相同的五行 token store 规则，意味着未来修改这些 store 时需更新两处；旧实现只有一个 `Corpus::apply` kernel。收益计算必须承认这一项增加。两份相邻、位于同一短方法，分别紧接其几何读取，并让 Some 的 token/span 顺序完整可见。我不建议为这点重复强制引入本地 closure，也不把它列为 blocker；若将来 kernel 扩展，本地共享才有更强收益。这个局部代价没有抵消删除跨模块载荷协议的收益。

4. **candidate 多余 Match derive：已处理。** candidate `corpus.rs:30` 曾额外派生 `Debug/PartialEq/Eq`，纯生产文件仍为 `Clone/Copy`。我核对了 `tk-train/src` 全范围的 Match 引用：其用途只有定义、matcher 返回、邻接参数与 freeze 输入，新增调试/相等实现没有消费者；几何断言比较的是有自己 derive 的 FrozenMatch。主 agent 已将 candidate 恢复为 `Clone/Copy`，我已只读核验，生产提取与候选该处一致。

## 测试 oracle 的必要性与限度

candidate `corpus.rs:45` 的 `FrozenMatch`、`:224` 的独立 plane 副本、`:522` 的旧 frozen kernel，以及 `merge.rs:771` 的顺序重放只在测试构建出现。冻结发生在实际 preparation 接受处（fresh `:527`，cohort `:706`），在 endpoint 写入前保存 right/after；预期写入只使用冻结坐标，未调用生产 `apply_merges` 重建期望。生产重建的全部几何在每条写入前与快照比较，正序、逆序 jobs/starts 和 parallel jobs 后比较全部 token 与 occurrence-span 槽位。它能够发现缓存首点错误、alias 共用跨度和恢复时机错误，验证的是此次删除快照后新增的契约，具有实际作用。

oracle 不是独立的 trainer 或 matcher 正确性证明：冻结仍使用真实 matcher 的接受结果和准备时 `span`，所以错误接受本身仍需既有 trace/model 测试发现。保留旧 frozen kernel 是刻意独立的预期实现；不要为了源代码去重让它调用生产 kernel，否则两边会共享同一写入错误。

测试代价也应单列：测试构建重新持有完整三坐标列表和 rule 快照，并在每轮复制/扫描完整 planes 多次。它增加测试代码及运行成本，不能拿这部分膨胀或删除计入生产简化数字。当前自动挂入 unit-test `merge_next`，对既有 fresh/reuse 路径提供广泛断言；是否未来收窄到定向用例，应由实际测试运行成本决定，本审查没有测量该成本，不据此要求另一次实现。

## 验证范围

本审查只读代码，未修改源文件，未启动性能或全量测试。主 agent 提供的前一轮验证结果是 baseline 与 candidate 默认/no-default 库测试均为 47 passed、1 个同一已知 roundtrip 失败；candidate Clippy `-D warnings` 和格式检查通过。该结果为主 agent 的验证记录，不是本审查另行重复运行的结果。上述局部处理后的最终验证由主 agent 继续执行，应以其最终记录为准。采用建议限于可维护性与源码语义核验，不把它等同于性能结论或完整测试全绿。
