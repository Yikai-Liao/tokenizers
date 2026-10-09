# BPE training engine

This engine returns the complete vocabulary with token IDs, ranked merge rules,
and special tokens. It preserves weighted pair selection and compatible batch
execution while using one endpoint/event protocol.

The public entry points remain in [the trainer](../mod.rs). Read [mod.rs](mod.rs)
for the complete workflow: initialize identities and a borrowed corpus plan,
build the compressed occurrence index, materialize endpoints, select a compatible
batch, prepare against a stable snapshot, apply its
writes, commit counts and births, then publish the model. Each parallel phase
joins before the next stage begins. An active identity collision restarts from
original input and the retained alphabet; speculative merges are discarded.

The implementation is divided by the knowledge each module owns:

- [vocabulary.rs](vocabulary.rs) owns strings, alphabet interpretation, decorated
  IDs, active identity detection, and canonical output.
- [corpus.rs](corpus.rs) owns fixed coordinates, token endpoints, weights, and
  occurrence spans. Only this module converts full-u64 occurrence coordinates
  to checked resident indices.
- [index.rs](index.rs) owns exact priority correction, count shards, signed reuse
  ledgers, cohort queues, and complete birth publication.
- [merge.rs](merge.rs) owns compatible selection, read-only preparation, local
  event aggregation, and joined application. The coordinator sees `Batch` and
  `Prepared`, without handling write geometry or neighbor directories.
- [positions.rs](positions.rs) owns sorted full-u64 streams and restart blocks.
  It preserves duplicate positions and supports coordinates through `u64::MAX`.

Compatible batches retain the original priority prefix and permit shared heads
or shared tails. Crossed endpoints end the batch without skipping a candidate.
AA and reserved-ID rules run alone. Complete ordinary producers publish directly;
partial, AA and reuse births aggregate before publication. Commit routes compact
metadata references and moves each position stream only once. Active reuse uses
one cohort at a time; its
word scans follow the original alias/length-gate conditions.

Run library checks from the repository root:

```sh
cargo test --manifest-path tokenizers/tk-train/Cargo.toml --lib
cargo test --manifest-path tokenizers/tk-train/Cargo.toml --no-default-features --lib
cargo clippy --manifest-path tokenizers/tk-train/Cargo.toml --all-targets
cargo fmt --manifest-path tokenizers/tk-train/Cargo.toml --check
python3 experiments/bpe-simplification/count_lines.py
```

The tests compare full vocabulary IDs, ordered merges and per-rule traces with
an independent small-input reference. Fixed-seed combinations cover thread
counts, affixes, aliases, zero weights, ties, overlap and strict birth gates.
Literal expectations cover wide counts and public errors. Codec tests cover the
entire u64 domain, repeated values and restart boundaries. Public tests retain
feed, model reload, progress and ambient-versus-training pool behavior.

The source-size budget includes all engine modules and test-only code, plus the
reference and shared oracle helpers. Removing comments or moving implementation
outside this directory does not reduce the budget. Performance evidence and
build/input hashes are recorded by the [experiment](../../../../../../experiments/bpe-simplification/STATUS.md).
