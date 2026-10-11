# BPE position element ranges

Baseline: `c8e7348e890d45dd06aadc8464e085df8bcb1202` on
`simplify/bpe-flat-20261010`.

The position module exposes only the custom container `Positions`; `Chunk` is
removed. `partition_ranges(target_items)` chooses restart-aligned ranges of
**element indices**, and `iter_range(range)` reads exact bounds through an
anonymous iterator. `iter()` reads the full list. The private cursor and byte
format stay inside the module. Arbitrary valid starts replay at most 127 preceding
values; empty ranges decode nothing, and invalid bounds panic before byte access.
`iter_from_value` retains coordinate-based seeking for reuse tasks.

Ordinary merge tasks retain the rule rank and a standard `Range<usize>`, then
read that rule's immutable candidate in preparation. No fragment copy, full
fragment decompression, extra Box or generic task enum is introduced. Complete
publication requires exactly `0..candidate_len`; filtered AA slices remain
partial. The caller must keep each range paired with its originating list.

## Validation

Default and no-default-feature library suites each pass 30 tests, excluding the
existing `feed_training_tokenizer_json_and_encoding_roundtrip` assertion already
reproduced on the baseline in [the previous report](../trainer-interface-20261011/README.md).
The unfiltered run reproduces that failure (encoding 测 yields ID 3, expected 4).
Clippy with warnings denied, changed-file formatting and whitespace checks pass.
The independent codec harness passes all five tests natively. Miri was not run:
`rustup` and `cargo-miri` are unavailable in the current PATH.

Storage tests cover partition continuity, duplicates, targets through usize::MAX,
all native range starts, exact unaligned/empty ends, u64::MAX, malformed bounds,
coordinate seeks and concurrent readers. Merge regressions retain aligned
complete/partial birth checks and AA partial semantics. The sequential oracle
checks full models and every rule at 1/4/8 workers. Commands and limits are in
[validation.json](validation.json); test/compiler logs are retained beside it.

## Performance

The paired public `do_train` comparison uses the two existing large prepared
inputs, locked release builds (opt-level 3, fat LTO, one codegen unit), four workers
on CPUs 0–3, one excluded warmup pair and four alternating formal pairs per input.
Timing includes output construction but excludes input loading, serialization
and validation. Process HWM includes loaded counts. All 20 full models match;
all samples have zero swap, with no concurrent build or benchmark.

| Input | Median wall seconds, baseline / candidate | Paired wall change | Paired CPU change | Paired HWM change |
| --- | ---: | ---: | ---: | ---: |
| bytelevel_regex | 9.7254 / 9.7480 | -0.52% | -0.69% | -0.16% |
| whitespace | 8.4887 / 8.4178 | -0.97% | -0.33% | +0.35% |

Individual wall changes vary in direction. These four-pair observations show no
clear regression and do not establish a stable speedup. These inputs exercise
fresh training; they do not measure every reuse workload or arbitrary range
shape. Generated native oracle tests separately cover reuse correctness.

[builds.json](builds.json) records binary/source hashes and baseline provenance.
The frozen baseline is the previous report's candidate; its archived binary and
production source hashes match the current baseline (test-only source differs).
[performance-manifest.json](performance-manifest.json) records input hashes,
profile and machine. Raw samples, full-model hashes and paired medians are in
[performance-runs.json](performance-runs.json), [performance.log](performance.log)
and [performance-summary.json](performance-summary.json). Large model files and
runner binaries remain outside the repository.

To reproduce, build and freeze runners from the baseline and this change using
the profile/command in builds.json, then run
`../trainer-interface-20261011/measure.py --baseline BIN --candidate BIN --bytelevel WORDS --whitespace WORDS --output DIR`
with the prepared inputs from the manifest. Performance scripts and runner
sources are unchanged.
