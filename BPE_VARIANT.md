# BPE trainer variant: parallel-count4

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
