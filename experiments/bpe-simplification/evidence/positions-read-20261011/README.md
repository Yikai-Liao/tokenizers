# Borrowed position reads: Chinese 256 MiB comparison

The baseline is `26aa5926`, before the read-interface change. It already uses
`SmallVec<[u64; 2]>`, directly encoded owned positions, and the inline Enum cases.
The candidate contains the borrowed chunks and coordinate-based iteration from
`98524a4d`, plus the explanatory comments and test cleanup recorded here.
This comparison therefore measures the read-interface change, rather than the
older removal of encoder scratch or mutable width promotion.

`merge.rs` stores a borrowed Chunk whose only operations are length and iteration.
Block ranges, restart alignment and list-index conversion stay in Positions.
An ordinary fragment covering the candidate's actual length is complete even
when alignment makes it larger than the target; filtered AA slices remain partial.
Chunks allocate no per-fragment storage and decode only during iteration.

## Protocol

The two inputs are Chinese ByteLevel-regex and Whitespace, each derived from
256 MiB of original text. Frozen prepared word maps and their original text
hashes are listed in [inputs.json](inputs.json). Vocabulary target is 50,000,
minimum frequency 2, with no affixes or maximum token length.

Each case runs one excluded warmup pair followed by four formal pairs, reversing
arm order on alternating blocks. Four workers are pinned to CPUs 0–3. No builds,
tests or other benchmarks run during timing. Complete parsed models are compared
with the first baseline reference, and each sample checks swap and competing
build/benchmark processes.

Training CPU and elapsed time cover public `do_train`, excluding input loading
and model serialization. Peak memory is process high-water RSS before validation
and includes the caller's loaded word map and earlier allocations. Percentage
deltas are medians of the four within-block ratios; absolute columns are medians
of the four samples per arm. Shared-VM observations do not establish a universal
speedup or a precise attribution of small time differences.

The cached baseline binary and every BPE source hash were verified against
`26aa5926`. Both arms use the same Rust compiler, release profile, Cargo manifests,
runner source and lockfile. Build commands, binary/source hashes and compiler
information are in [builds.json](builds.json); complete snapshots are in `source/`.

## Comments and tests

The source now states the count/directory/data layout, data-relative offsets and
single-block directory omission. `compressed_bytes` requires the Compressed
variant and returns only its data area. Comments explain stepping back after
restart-seed binary search, array padding versus valid zero coordinates,
monotonic validation across restart boundaries, and the internal encoder
invariants relied upon by Cursor. PairIndex also records why priority repair
must preserve pop/update/push for equal-priority occurrence cohorts.

Positions retains five storage tests. Removing SmallVec scaffolding, repeated
empty-fragment setup and redundant chunk assertions reduces the test section
from 219 to 187 formatted lines; the file changes from 507 to 487 lines including
the added comments. Exact encoded bytes, the sole-fragment pointer reuse check,
cross-boundary duplicates, full-u64 values, concurrent reads and Miri seek
sampling remain. Descending input at a restart is checked with invalid input.

The 17 scoped no-default BPE checks and all-target Clippy with warnings denied
pass. The pre-existing tokenizer encoding assertion remains excluded as in the
preceding changes. Formatting uses the changed-file-only temporary entrypoint.

## Results and archived artifacts

Four formal pairs per case give the following median within-pair deltas.
Positive times mean more time; negative RSS means lower process high-water memory.

| Chinese 256 MiB input | CPU | Elapsed | Peak RSS |
| --- | ---: | ---: | ---: |
| ByteLevel-regex | +0.81% | +0.87% | -0.43% |
| Whitespace | -0.15% | +0.89% | -0.56% |

All 20 complete-model comparisons match, including four excluded warmup runs
and sixteen formal runs. No sample records swap or a concurrent build/benchmark.
The initial two ByteLevel pairs had CPU +2.93% and elapsed +3.98%; completion of
all four reduced those deltas to +0.81% and +0.87%. This illustrates why early
pairs do not establish a stable cost of the chunk interface.

All final paired medians differ by less than one percent. The user chose to
retain the measured reader without further optimization at this difference
level. The retained production source hashes match the candidate snapshot.
These data do not identify a chunk-specific performance regression or prove
that any individual private helper has zero cost.

[All runs](screen/core-runs.json), [summary](screen/core-summary.json),
[protocol and commands](screen/core-manifest.json),
[jobs and runner output](screen/jobs-and-output.json), and two compressed
complete reference models in `screen/reference-models/` preserve the evidence.
CPU/elapsed/RSS absolute medians remain in the summary; percentage deltas above
are paired medians, not ratios of those absolute medians. Compiler/Cargo logs
and the controller output are also archived. Trailing blank lines in logs are
removed for repository whitespace checks.
