# BPE trainer variant: corpus-parallel

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
