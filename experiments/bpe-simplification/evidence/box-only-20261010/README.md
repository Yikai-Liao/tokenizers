# Box-only BPE positions

The retained change replaces `Empty`, `One`, `Two`, and `Compressed` with one
`Box<[u8]>` wrapper. Empty lists use an empty box; all other lists use the existing
restart/delta representation. Full-u64 coordinates, checked layout arithmetic,
owned lifetime, and concurrent immutable readers remain supported.

Baseline: `0416ae7a97bbc55f47f7969f4bdb04e8cfed1eff`. Retained production commit:
`9f5b553dba61dba45a75021f4292997c16359dfb` (the experimental counterpart is
`27027bfb`). The snapshots and `builds.json` identify the measured source and
binaries independently of later commits.

## Code and layout

`positions.rs` falls from 334 to 286 nonblank, noncomment production lines:
19 added, 67 deleted, net 48 removed. `count-code.py` defines the counting scope.
On this x86-64 build, `Positions` shrinks from 24 to 16 bytes and a queued
`Candidate` from 40 to 32 bytes. This saves metadata for every resident list but
introduces allocation and codec work for lists of one or two positions.

The runner's `.text` shrinks by 2,320 bytes, `.rodata` is unchanged, and the ELF
file grows by 6,712 bytes. Total ELF size is not used as a runtime memory measure.
See `positions-layout.rs`, `binary-size.json`, and the section dumps.

## Real inputs

Four frozen cases each represent approximately 256 MiB of original text:
English and Chinese with ByteLevel-regex and Whitespace preprocessing. Every
run receives the same prepared word map, fixed hash seeds, four workers pinned
to CPUs 0–3, a 50,000-token vocabulary target, minimum frequency 2, and no affixes
or maximum token length. Rust release builds use optimization level 3, fat LTO,
one codegen unit, and disabled incremental compilation.

Each case has one excluded warmup pair and four formal alternating BA/AB pairs,
40 runs in total. All 40 complete models match their case reference; sampled
swap is zero and no concurrent build or benchmark is recorded. Positive deltas
mean a cost increase for Box. Absolute columns are per-arm medians; delta
columns are medians of within-block ratios, so they need not equal ratios of
the displayed absolute medians.

| Case | Wall seconds, enum → box | CPU seconds, enum → box | Peak MiB, enum → box | Paired wall Δ | Paired CPU Δ | Paired peak Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| en-256MiB | 1.807 → 1.749 | 4.143 → 3.966 | 167.0 → 167.4 | -1.98% | -2.99% | +0.14% |
| en-whitespace-256MiB | 1.555 → 1.602 | 3.648 → 3.650 | 135.7 → 149.4 | +4.57% | +1.15% | +10.41% |
| zh-256MiB | 21.720 → 20.923 | 68.398 → 65.456 | 2289.1 → 2251.7 | -3.67% | -3.48% | -1.65% |
| zh-whitespace-256MiB | 21.309 → 22.646 | 57.351 → 61.034 | 1768.0 → 1729.4 | +6.23% | +5.26% | -2.16% |

Timing covers public `do_train` and excludes input loading and model
serialization. CPU is the sum of process/thread CPU time consumed during that
boundary. Peak memory is process high-water RSS before validation, so it includes
the loaded caller map and earlier allocations; it is not training-only memory.
English preprocessing deduplicates the original 256 MiB to a much smaller word
map, explaining the approximately 136 MiB baseline Whitespace peak. Its increase
is approximately 13.7 MiB despite the 10.4% relative change; the user explicitly
accepted this absolute increase.

## Chinese Whitespace confirmation and frequency 3

A second run freezes the same Chinese Whitespace input and changes only the
minimum frequency. Each floor has one excluded warmup pair and four formal
alternating pairs. All 20 models match their per-floor reference with zero swap.
The original 9.8% CPU observation was one pair, not a multi-run summary.

| Case | Wall seconds, enum → box | CPU seconds, enum → box | Peak MiB, enum → box | Paired wall Δ | Paired CPU Δ | Paired peak Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| zh-whitespace-min2 | 22.015 → 23.609 | 59.200 → 63.189 | 1767.6 → 1747.9 | +6.31% | +4.94% | -1.09% |
| zh-whitespace-min3 | 21.336 → 22.277 | 58.480 → 60.276 | 1653.4 → 1631.4 | +2.33% | +2.64% | -1.71% |

Combining the eight formal minimum-frequency-2 pairs gives a CPU paired median
of +5.26% and wall median +6.23%. The second batch independently gives +4.94% CPU.
The measurements support a repeatable cost around 5% on this workload, with VM
noise affecting individual samples. Frequency 3 reduces the CPU median gap to
+2.64%; it does not establish equality.

`perf stat` adds whole-process counter observations in the second batch. At
frequency 2, paired instruction and cycle medians increase by 1.27% and 4.13%;
at frequency 3 they increase by 0.34% and 1.67%. Hardware events are supplied by
the VPS virtual PMU and report full running time. They include loading,
serialization, and all threads, unlike training-only CPU timing. They corroborate
that there is extra work but cannot attribute it exclusively to allocation or
decoding. These are descriptive shared-VM measurements, not a general speed claim.

## Position-length diagnostic

A disposable instrumented Box worktree records cumulative `from_sorted` calls,
position counts, requested encoded payload bytes, and initial local weighted
counts. Its binaries are never used for timing. Both complete models match the
timed reference for their floor. `make-position-probe.py`, `position-probe.patch.gz`,
and `position-probe.json` preserve instrumentation and provenance.

At floor 2, 6,945,769 of 13,162,270 nonempty freeze calls (52.77%) contain one or
two positions. Initialization alone produces 3,470,161 one-position and 1,048,004
two-position fragments. Among initial one-position fragments, 3,343,442 (96.35%)
have local weighted count 1: the user's expectation about weights is correct.
Local weight and global pair frequency differ; equal pairs across ranges can add
up to the admission floor. Initial fragments are encoded before global admission.

The complete-birth phase adds 199,140 one-position and 1,848,243 two-position
calls at floor 2. Raising the floor to 3 cuts these to 47,631 and 107,722. Total
short calls fall to 4,702,382 of 10,918,883 nonempty calls (43.07%), while initial
short calls remain unchanged. This explains why frequency filtering can reduce
part of the opportunity for extra short-list allocation without eliminating it.
The histogram measures cumulative calls, not distinct pairs or simultaneously
resident lists. It identifies plausible allocation/codec costs, not their exact
share of training CPU. A subsequent experiment will admit initial pairs before
encoding and measure the CPU/memory tradeoff across input sizes.

## Validation and artifacts

The positions roundtrip, seeking, invalid-input, concurrent-reader, and ordinary
BPE oracle/cohort tests passed. Twelve scoped BPE tests passed after excluding
the pre-existing tokenizer-encoding roundtrip failure at the user's request.
All-target no-default Clippy with warnings denied, changed-file rustfmt, and diff
checks passed. No unrelated encoding fix is included.

`real/` and `frequency-sweep/` contain manifests, every per-run metric,
job/output records, and one compressed full reference model per case. Per-run
model SHA-256 values are retained rather than duplicating every large model.
The latter directory additionally includes all raw perf CSV records, their
summaries, and the eight-pair combined calculation. `builds.json` records source,
lockfile, binary hashes, compiler, and build flags; `validation/` holds build,
Clippy, and probe build output.

Run `measure.py` with `BPE_MEASURE_ROUNDS=5`; use `BPE_MEASURE_INPUTS` to select
`frequency-sweep-inputs.json` and `BPE_MEASURE_PERF=1` to collect the counter pass.
Input paths refer to the frozen local corpus store; their sizes and hashes are
in each manifest. `archive-measurements.py` validates models and swap status
before archiving results. No directed synthetic benchmark is claimed here.
