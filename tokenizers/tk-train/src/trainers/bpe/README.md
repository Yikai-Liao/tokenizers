# BPE training

Training returns the complete vocabulary with token IDs, ranked merge rules,
and special tokens. It preserves weighted pair selection and compatible batch
execution while using one endpoint/event protocol.

The public entry points and private training coordinator share [mod.rs](mod.rs).
`BpeTrainer::train_counts` selects the requested worker count and calls `train`;
tests exercise the same training flow with explicit worker counts and observers.
Read it for the complete workflow: initialize identities and a borrowed corpus plan,
build the compressed occurrence index, materialize endpoints, select a compatible
batch, record its rules, complete the batch with `Batch::commit`, then publish
the model. Merge joins snapshot preparation before endpoint writes and joins
those writes before the index updates count owners and publishes births. A collision with a previously activated
identity restarts from original input and the retained alphabet; speculative
merges are discarded.

The implementation is divided by the knowledge each module owns:

- [vocabulary.rs](vocabulary.rs) owns strings, alphabet interpretation, decorated
  IDs, initial activation metadata, alphabet selection, and canonical output.
- [corpus.rs](corpus.rs) owns fixed coordinates, token endpoints, weights, and
  ID activation and occurrence spans. Initial ID metadata moves here once;
  merge selection and geometry use this same activation record.
  Only this module converts full-u64 occurrence coordinates
  to checked resident indices.
- [index.rs](index.rs) owns exact priority correction, count shards, signed reuse
  ledgers, one owning candidate queue, and complete birth publication.
- [merge.rs](merge.rs) owns compatible selection, read-only preparation, local
  event aggregation, and a complete joined round. The coordinator selects a
  `Batch`, records its rules and commits it; Merge owns the execution order.
- [positions.rs](positions.rs) owns sorted full-u64 streams and restart blocks.
  Producers use `SmallVec<[u64; 2]>` and construct immutable lists directly.
  Empty, one-value and two-value lists stay inline; longer lists own compressed
  byte slices. Construction accepts a sorted slice or consumes ordered fragments
  without an encoder service. Duplicates and coordinates through `u64::MAX` are preserved.
  Readers use iteration, `iter_from_value(coordinate)`, or borrowed `chunks(target_items)`;
  each chunk exposes only its length and iteration. Restart ranges stay private.
- [feed.rs](feed.rs) owns streaming batches and bounded local word caches;
  `LocalCounts` controls cache flushing and successful completion.
- [word_counts.rs](word_counts.rs) owns borrowed and collected count views,
  content equality and the flat serialization contract.

One owning priority queue holds fresh lists and reuse cohorts. Its historical
resource measurements and equal-priority reuse ordering limitation are recorded
in the [heap experiment](../../../../../experiments/bpe-simplification/HEAP_EXPERIMENT.md).
Larger occurrence lists use owned allocations; dropping a list releases its bytes,
including short compressed lists that previously remained in an attempt arena.

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
metadata references and moves each position stream only once. Reuse mode selects
one cohort at a time; its word scans follow the original alias/length-gate conditions.

Run library checks from the repository root:

```sh
cargo test --manifest-path tokenizers/tk-train/Cargo.toml --lib
cargo test --manifest-path tokenizers/tk-train/Cargo.toml --no-default-features --lib
cargo clippy --manifest-path tokenizers/tk-train/Cargo.toml --all-targets
cargo +nightly miri test --manifest-path experiments/bpe-simplification/miri-codec/Cargo.toml
```

The tests compare full vocabulary IDs, ordered merges and per-rule traces with
an independent small-input reference. Fixed-seed combinations cover thread
counts, affixes, aliases, zero weights, ties, overlap and strict birth gates.
Literal expectations cover wide counts and public errors. Position storage tests cover the
entire u64 domain, repeated values and restart boundaries. The standalone Miri
harness imports the actual position module and keeps borrow/leak checks enabled.
Miri samples seek starts and targets around restart boundaries; native tests
exhaust every start. Native and Miri storage tests use scoped threads and verify that frozen lists
remain readable after their construction inputs are dropped.
Public tests cover feed, model reload, progress and ambient-versus-training pool
behavior. [The coverage map](COVERAGE.md) records the combined boundaries.

Progress uses the upstream `ProgressBar`, `ProgressStyle` and `ProgressFormat`.
The trainer's original helpers retain the Tokenize words, Count pairs and Compute
merges stages. Parallel collectors increment the bar at work-block boundaries;
the coordinator reports merges after each joined batch. There is no additional
progress thread, channel or timer.

The counting command also checks formatting of ordinary BPE and its tests;
sibling trainers and unchanged legacy parity files retain upstream formatting.
Source counts and diffs are descriptive, with no hard line limits. The report
includes the ordinary BPE implementation, public API, feed and word-count storage;
tests, the reference oracle, shared helpers and Miri harness are counted separately.
Pass `--change-base <revision>` to `count_lines.py` to report this change alongside
the default comparison with HF `origin/main`. Git diffs detect renames and include
comments and documentation, while source counts exclude comments and blank lines.
Performance evidence and build/input hashes are recorded by the
[experiment](../../../../../experiments/bpe-simplification/STATUS.md).
