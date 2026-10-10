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
  IDs, initial activation metadata, and canonical output.
- [corpus.rs](corpus.rs) owns fixed coordinates, token endpoints, weights, and
  ID activation and occurrence spans. Initial ID metadata moves here once;
  merge selection and geometry use this same activation record.
  Only this module converts full-u64 occurrence coordinates
  to checked resident indices.
- [index.rs](index.rs) owns exact priority correction, count shards, signed reuse
  ledgers, cohort queues, and complete birth publication.
- [merge.rs](merge.rs) owns compatible selection, read-only preparation, local
  event aggregation, and joined application. The coordinator sees `Batch` and
  `Prepared`, without handling write geometry or neighbor directories.
- [positions.rs](positions.rs) owns sorted full-u64 streams and restart blocks.
  Temporary buffers narrow to u32 with automatic full-u64 promotion. Published
  lists have a two-word descriptor and borrow small allocations from a scoped
  Arena, selected by a dynamic byte threshold. Duplicate positions and
  coordinates through `u64::MAX` are preserved.

Initial collection splits the borrowed plan at whole-word boundaries, with a
2²⁴-resident-slot block cap. An oversized single word is processed alone. Each
producer compresses its temporary positions before returning, so raw lists retire
without waiting for every producer. Owned compressed fragments retain complete
pair counts; global frequency admission precedes final list publication. There is
no extra wave barrier.

Preparation indexes unique rule heads directly by token ID and checks tail
presence before left-neighbor lookup. Shared heads fall back to exact pair keys.
Only monotone snapshot write/birth appenders skip repeated ordering checks;
general appends and final encoding retain validation.

Compatible batches retain the original priority prefix and permit shared heads
or shared tails. Crossed endpoints end the batch without skipping a candidate.
AA and reserved-ID rules run alone. Complete ordinary producers prune before encoding in preparation and publish
directly. Partial, AA and reuse births aggregate before encoding and publication. Commit routes compact
metadata references and moves each position stream only once. Active reuse uses one cohort at a time; its word scans follow the original alias/length-gate conditions.

Run library checks from the repository root:

```sh
cargo test --manifest-path tokenizers/tk-train/Cargo.toml --lib
cargo test --manifest-path tokenizers/tk-train/Cargo.toml --no-default-features --lib
cargo clippy --manifest-path tokenizers/tk-train/Cargo.toml --all-targets
cargo fmt --manifest-path tokenizers/tk-train/Cargo.toml --check
python3 experiments/bpe-simplification/count_lines.py
cargo +nightly miri test --manifest-path experiments/bpe-simplification/miri-codec/Cargo.toml
```

The tests compare full vocabulary IDs, ordered merges and per-rule traces with
an independent small-input reference. Fixed-seed combinations cover thread
counts, affixes, aliases, zero weights, ties, overlap and strict birth gates.
Literal expectations cover wide counts and public errors. Codec tests cover the
entire u64 domain, repeated values and restart boundaries. The standalone Miri
harness imports the actual position module and keeps borrow/leak checks enabled.
Miri samples seek starts and targets around restart boundaries; native tests
exhaust every start. Native and Miri allocation tests use scoped threads and both allocation cursors.
Public tests cover feed, model reload, progress and ambient-versus-training pool
behavior. [The coverage map](tests/COVERAGE.md) records the combined boundaries.

The 2200-line production budget covers all engine implementation, including any
logic moved outside this directory. All default BPE test-only code, the reference oracle, shared helpers and Miri
harness have a separate 800-line limit; redundant checks are combined before
using that allowance.
Removing comments does not reduce either count. Performance evidence and
build/input hashes are recorded by the [experiment](../../../../../../experiments/bpe-simplification/STATUS.md).
