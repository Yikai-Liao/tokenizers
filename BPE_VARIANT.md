# BPE trainer variant: corpus-direct

Entry points: original `BpeTrainer::train_vocab()`, `do_train()` and `Trainer::train()`.

- Algorithm: parallel.
- Corpus token ID and posting width: u32.
- Atomic corpus: False.
- Initialization workers: same as merge workers.
- Merge workers: existing tk_encode parallelism controls; disabled parallelism selects 1.
- Empty affixes use the guarded endpoint algorithm; nonempty affixes retain the compatible cohort path.
- Historical comparison helpers are module-private.
- Base snapshot: 8c968e10; fixed PR comparator: 6ac0de5359d9e0e1ed0608422575a360ef91b908.

Experiments and provenance: central `/root/code/tokenizers/benchmarks/hf-bpe/EXPERIMENT_LOG.md`.

The local benchmark supports only the original `reference` API route; legacy indexed features remain in the historical snapshot.

Optimization parent: `bpe/parallel-count4` at `b6a28768`.

Corpus construction now measures ordered word regions in parallel, writes each
region directly into its final exclusive slice, and combines boundary metadata
after the join. Alphabet selection and canonical ID assignment retain the original
behavior; nonempty affixes retain the generic path. Initialization worker controls
apply to construction and pair counting. No additional full corpus is retained.

Private statistics split alphabet, region measurement, final allocation and fill
times. The current round plan and measurements are maintained centrally in
`/root/code/tokenizers/benchmarks/hf-bpe/OPTIMIZATION_PLAN.md`.

Candidate C parent: `98ca7fc1` (corpus-parallel).

Unlimited alphabet collection uses worker Unicode presence bitmaps, followed by
canonical character ordering; limited alphabets keep the original selector.
A read-only Unicode-to-ID table removes per-position string hashing.
MaybeUninit final storage is initialized by exclusive word regions and converted
only after complete coverage and joins; its private unsafe conversion is documented
and tested. Alphabet temporary arrays and ID table capacity are measured separately.

## 匹配控制项

本分支从 corpus-direct `fbdc0b2b` 派生，仅把原接口 plain 路径的 `atomic_corpus` 改为 true。初始化仍采用 C 的并行 alphabet、直接查询和独占直接填充；merge 保留排序 Plan 与 delta 两阶段算法。用于与融合候选隔离算法收益。
