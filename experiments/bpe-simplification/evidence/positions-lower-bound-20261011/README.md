# Shared compressed-position reads and standard-library lower bound

Baseline: `bc9442153f27c615ec66f6cb5b5cb437999ea421` on
`simplify/bpe-flat-20261010`.

`Positions::compressed()` now supplies one private borrowed view of the restart
directory and data area. The directory is a slice of fixed-size byte arrays, so
it requires neither allocation nor integer alignment. The view's `bytes_at()`
decodes directory offsets for both range reads and value seeks; `block_bytes()`
also handles the first block, whose offset is always zero. Single-block lists
retain their existing format with an omitted directory.

`iter_range()` reads through this view. `lower_bound()` uses the same view with
the standard-library `partition_point`, comparing decoded restart seeds. It
still scans the predecessor block to retain the first duplicate when a repeated
coordinate spans a restart. Lists of at most 128 elements use the existing
iterator directly. The public interface continues to expose only `Positions`;
the view and cursor remain private. Encoding, ownership and stored bytes are
unchanged. `as_chunks` requires Rust 1.88; the tested compiler is 1.99.

## Validation

Default and no-default-feature library suites each pass 31 tests, excluding the
existing `feed_training_tokenizer_json_and_encoding_roundtrip` failure already
reproduced on the baseline in
[the earlier report](../trainer-interface-20261011/README.md). The independent
native codec harness passes all six tests. Clippy with warnings denied,
changed-file formatting and whitespace checks pass.

The new regression checks duplicate runs beginning immediately before, at and
after restart boundaries, and extending over multiple restarts. Existing tests
cover empty/inline/single-block lists, arbitrary range starts, u64::MAX,
malformed ranges, concurrent readers and full training against the sequential
oracle, including reuse at multiple worker counts. A separate differential
check compares baseline and candidate lower bounds with uncompressed slice
`partition_point` for 2,192,020 queries, including constant lists, long duplicate
runs, large gaps and values near u64::MAX. All results match.

Commands and results are in [validation.json](validation.json) and
[checks.json](checks.json); the archived logs and
[microbenchmark correctness log](microbench-correctness.log) retain the output.

## Query microbenchmark

Both implementations are compiled into one executable with Cargo release
defaults (opt-level 3). Each configuration runs 150,000 queries per sample for
seven repetitions, alternating which implementation runs first. Results below
are median nanoseconds per query on CPU 0; encoding is outside the measurement.

| Items | Distribution | Baseline ns | Candidate ns | Change |
| --- | --- | ---: | ---: | ---: |
| 4,096 | distinct | 266.30 | 185.99 | -30.2% |
| 4,096 | duplicates | 116.12 | 77.33 | -33.4% |
| 65,536 | distinct | 303.89 | 206.67 | -32.0% |
| 65,536 | duplicates | 303.53 | 206.64 | -31.9% |
| 1,048,576 | distinct | 423.47 | 315.04 | -25.6% |
| 1,048,576 | duplicates | 419.46 | 312.48 | -25.5% |

All sizes, including 2, 128 and 129, are in [microbench.csv](microbench.csv).
[The manifest](microbench-manifest.json) records compiler, source hashes,
binary hash and measurement settings. The speedup includes directly reading
restart seeds and bypassing binary search for short lists, along with the
standard-library search; it does not isolate `partition_point` alone.

Reproduce against the current checkout with:

```sh
python experiments/bpe-simplification/evidence/positions-lower-bound-20261011/measure_positions.py \
  --baseline bc944215 --output /tmp/positions-microbench
```

## Public training comparison

The existing two large prepared inputs use locked release runners (opt-level 3,
fat LTO, one codegen unit), four workers on CPUs 0–3, one excluded warmup pair and
four alternating formal pairs per input. Timing covers public `do_train` and
output construction, excluding input loading, serialization and validation.
Process HWM includes loaded counts. No build or other benchmark ran concurrently.

| Input | Median wall seconds, baseline / candidate | Paired wall change | Paired CPU change | Paired HWM change |
| --- | ---: | ---: | ---: | ---: |
| bytelevel_regex | 9.6947 / 9.7417 | +0.02% | -0.04% | +0.12% |
| whitespace | 8.4370 / 8.5002 | +0.85% | +0.65% | +0.84% |

All 20 complete models match, and every sample has zero swap. These fresh-training
measurements show no clear overall acceleration; they do not establish a stable
regression from the small observed differences. They do not measure large reuse
workloads, where value-based seeking occurs. The query microbenchmark measures
that operation separately, and native training oracle tests cover reuse
correctness.

[builds.json](builds.json) records baseline commit, runner hashes, candidate
source hash and build profile. The baseline runner was reused only after its
binary hash and all archived source hashes were verified against the current
baseline. Build command:

```sh
CARGO_TARGET_DIR=/tmp/bpe-interface-release cargo build --offline --release --locked \
  --manifest-path experiments/bpe-simplification/runner/Cargo.toml
```

Raw samples, paired medians, machine/input settings and runner output are in
[performance-runs.json](performance-runs.json),
[performance-summary.json](performance-summary.json),
[performance-manifest.json](performance-manifest.json) and
[performance.log](performance.log). Large model files and frozen binaries remain
outside the repository. Reproduce with the earlier report's `measure.py`, frozen
baseline/candidate runners, and prepared inputs from the manifest:

```sh
python experiments/bpe-simplification/evidence/trainer-interface-20261011/measure.py \
  --baseline BASELINE_BIN --candidate CANDIDATE_BIN \
  --bytelevel BYTELEVEL_WORDS --whitespace WHITESPACE_WORDS --output RESULTS_DIR
```
