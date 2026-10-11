Restores the element-range interface introduced by `bc944215`, after its reversal
in `fb0864a0` was retained in merge `31355de5`. The shard refactor does not require
the borrowed `Chunk` interface, and the earlier evidence did not isolate a
performance benefit from restoring Chunk.

`Positions::partition_ranges` returns standard `Range<usize>` element indices.
`iter_range` reads any valid half-open range, replays at most one restart prefix
for an unaligned start, and rejects inverted or out-of-bounds input. Merge tasks
retain their rule rank and range and read that rule's immutable candidate list.
Complete publication requires exactly `0..candidate_len`; filtered AA slices
remain partial. The custom Chunk type and block-range read interface are removed.

The shard implementation is unchanged. The shared `Compressed` directory/data
view, `partition_point` restart search, `split_first_chunk` seed reader and both
subsequent decoder regressions are preserved. Merge preparation retains the
`VecDeque` lookahead and cohort `find` scan. This is a selective interface
restoration, with no changes to the encoding format or list ownership.

The restored tests cover partition coverage, repeated coordinates, arbitrary
empty/short/unaligned/cross-restart/suffix reads, range validation on all storage
variants, concurrent range readers, and Complete/Partial publication. Results
are checked against independent slice reads and the existing sequential trainer
oracle at 1/4/8 workers.

Default and no-default-features library runs each pass 38 tests, excluding the
previously reproduced baseline encoder roundtrip assertion. The native codec
harness imports the production position module and passes all seven tests.
Clippy with warnings denied, the changed-file formatter and diff whitespace
checks pass. Commands and source preservation checks are in `validation.json`;
source snapshots and logs are retained here. These checks establish correctness
and preservation of the later implementations; no new performance claim is made.

The separate `initial_ranges` linear-scan experiment is paused at the user's
request. Its existing uncommitted change remains in the working tree and was
present before and after this interface restoration. Its frozen binaries and
measurements do not describe the restored Range implementation.

Before pushing, default and no-default library tests (38 each) and Clippy were
rerun against the staged Range-only source snapshot, which retains the baseline
binary `initial_ranges`. The linear-scan production change and its evidence are
excluded from this commit. `push-*.log` and `push_validation` record these checks.
