# Archived lean engine: first measurements

Fixed Fork main baseline: `e4f787dc189d9be7192107490d652096cde7480e`.
The independent archive records the first complete simplified implementation before
performance tuning. Production is 1456 rustfmt nonblank/noncomment lines versus
5714 in main (74.5% fewer); tests including the independent oracle/shared helpers
are 1384 lines versus 6192. Implementation modules decrease from 26 to 6.

## Current measurements

English/Chinese ByteLevel, 256 MiB raw text, 4 workers, 50K vocabulary, minimum
frequency 2. Release opt-level 3, fat LTO, one codegen unit; no default features.
This host is a 6-vCPU Xeon KVM guest. Full vocabulary IDs and all ordered merges
match main in every completed run; child swap is zero. One candidate sample per
case is exploratory. Ratios below use historical A/A baseline medians except
Chinese core, which has an immediately preceding main sample.

| Case | Main train s | Lean train s | Main pipeline s | Lean pipeline s | Main RSS GiB | Lean RSS GiB |
|---|---:|---:|---:|---:|---:|---:|
| en-core-w4-v50000 | 1.09 | 3.16 | — | — | 0.18 | 0.27 |
| en-pipeline-w4-v50000 | 1.09 | 3.07 | 5.13 | 7.50 | 0.18 | 0.27 |
| zh-core-w4-v50000 | 18.80 | 70.92 | — | — | 2.41 | 3.29 |
| zh-pipeline-w4-v50000 | 19.35 | 71.36 | 24.24 | 76.47 | 2.39 | 3.37 |

Chinese pipeline was sampled with `perf record -F49`; it is diagnostic, not an
uninstrumented ranking sample. The first Chinese core pair is 3.77× slower and
uses about 0.88 GiB more peak RSS. English core is about 2.9× slower. This archive
is a correctness/complexity checkpoint, not the accepted final performance choice.

## What disappeared and what remains

Deleted: custom radix sort, bounded/keyed record layout dispatch, U24 slots,
linked birth chains and arena lifetimes, adaptive birth feedback, complete versus
buffered producer protocols, parallel AA parity exchange, priority prefix cache,
selected-rule specialization, and unsafe corpus pointers. No production logic
was relocated outside the counted engine. The counter checks formatted sources.

Retained: exact priority/ties, compatible ordered-prefix batches, local occurrence
updates, contiguous births, parallel preparation/application/count-owner commit,
U16/U32 token slots, full-u64 compressed positions, serial left-to-right AA choice,
reserved/active ID handling, restart with retained alphabet, and signed reuse
cohort semantics. Resident allocations still obey usize/isize bounds.

The replacement owns a single endpoint geometry, event form and position stream.
It adds temporary match vectors, hash directories and owner birth sorting. Corpus
materialization occurs before initial pair construction; the old deferred plan
and overlap are removed. These choices simplify maintenance but cost performance
and peak memory. Reuse keeps its original conditional whole-word scan domain.

## Validation and reproduction

Default library tests: 35 passed. No-default-features library tests: 35 passed.
Clippy for all targets with `-D warnings`: passed. rustfmt and source-budget check:
passed. Tests cover full models/merge traces, affixes/reuse, numeric errors, full
U64 restart codec boundaries, actual high token IDs, batch selection, public
feed/reload/progress and isolated worker-policy processes. No Miri/sanitizer run;
new engine production has no unsafe block.

See `evidence/` for input/build/source hashes, formatted counts, A/A noise and
candidate samples. Full raw observations and independent baseline/candidate
binaries remain in `/root/code/tokenizers-simplification-results/`. The local
`runner/` contains the benchmark source and lockfile. Build it with identical
features/profile separately against each worktree, copy the binaries as immutable
`bin/baseline` and `bin/<label>`, and point `run_pairs.py --root` at a manifest and
results directory. `summarize.py` recomputes paired ratios when paired samples
exist; this first-candidate table is not a six-pair acceptance analysis.

A/A used 56 processes (eight warmups plus six pairs in four cases). Candidate
repetition was stopped at the user's request after the first complete pair;
three remaining candidate-only cases reuse the existing model oracles. An
unfinished next main invocation is retained in the raw directory and excluded.
Future iterations use a small smoke comparison before deciding whether repeats
are justified. The PR #2501 comparison against HF main, Fork YTTM and HF PR #2348
has not been rerun for this archive; earlier host/timing results cannot be mixed
into a direct speedup claim.
