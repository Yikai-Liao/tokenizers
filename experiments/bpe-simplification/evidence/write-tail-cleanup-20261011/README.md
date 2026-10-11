# Write coordinate and neighbor parameter cleanup

This follow-up is relative to `30f06c04fc77d66e8f1b5b9c996db3330270a8b5`.

`Job` and `CohortPreparation` store accepted starts as `SmallVec<[usize; 2]>`,
and `Corpus::apply_merges` accepts `&[usize]`. Preparation already produces
resident indices, so application no longer converts them to `u64` and back
through `resident`. Positions retain their `u64` coordinate representation and
convert when entering the corpus. The existing alias test uses resident starts
directly too.

`FreshSnapshot::record_neighbors` obtains its position from `matched.start`.
Its caller no longer supplies a second copy of that coordinate. The two
application branches remain separate: one caches shared rule geometry and the
other loads each occurrence's geometry before writing. Their store order and
preparation/application join boundary are unchanged.

The two Rust files have 11 insertions and 17 deletions, including test adaptation.
The maintenance benefit is removing a redundant conversion and an implicit
parameter equality, with no additional helper or abstraction. The independent
read-only review in `review.md` found no required changes.

Validation logs and command metadata are recorded alongside this report. Checks
use the same frozen tk-train lockfile as the previous validation (SHA-256
`7ffd9e72dd8d77a9909b2dcce454bc99f602e57178abbdaf9acc9dd60ce2cc9b`).
The BPE test selection covers the existing deferred-write oracle, fresh/reuse
training, alias geometry, empty jobs, inline/spilled jobs and write-order tests.
The existing `feed_training_tokenizer_json_and_encoding_roundtrip` failure was
already present in the baseline; its prior logs remain in the start-only writes
results. Unrelated library tests are filtered out for this follow-up.

Both default and no-default-features BPE runs finished with 40 passed, 1 known
baseline failure and 7 unrelated library tests filtered out. All deferred-write
oracle tests passed. Clippy with `--all-targets -- -D warnings`, the changed-file
formatter check and `git diff --check` passed.

No performance tests were repeated for this small cleanup. The earlier four-worker
256 MiB measurements in `start-only-writes-20261011` describe the earlier measured
source revision and are retained as historical evidence.
