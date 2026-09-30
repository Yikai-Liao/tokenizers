# C and B2 combination stability (512 MiB)

This comparison uses three paired runs in alternating order: C/B2, B2/C, C/B2. It evaluates the two complete trainer combinations end to end. C uses direct corpus construction with a non-atomic corpus; B2 combines fused batch preparation, weight lookup, and atomic corpus storage. The comparison cannot isolate the weight-query implementation or the atomic-corpus setting by itself.

## Fixed setup and provenance

- Input: `zh-512m.txt`, 536,870,289 bytes, SHA-256 `a0d40d4102cba1e933f25e5ccd17552d2eaebaec4ef770bc39d8e46e740b4d4f`.
- Shared parameters: original reference trainer API, split `none`, vocab 50,000, min frequency 2, feed parallelism disabled, initialization workers 4, merge workers 4, u32 token IDs and postings.
- C: worktree `corpus-direct`, commit `fbdc0b2bf736ffdc12aed3d0c3e0aba4a0004caa`, non-atomic corpus.
- B2: worktree `fused-lookup`, commit `a0832c488a7ece429630c7d1493da5dd772c87aa`, atomic corpus.
- Each raw record captures its source, instrumented copy, runner, binary, input, environment, CPU and memory provenance. Runs were serialized; no build, test, download, or additional benchmark ran during the six calls.
- Memory policy stopped only at `MemAvailable <= 1 GiB`. Per-run process VmSwap and host paging counters are preserved.

## Per-call results

`Common prep` is C `plan_ms + delta_ms`; for B2 it is `plan_ms + delta_ms + weight_lookup_build_ms`. B2's `fused_prepare_ms` is already inside the delta path and is reported separately, not added again.

| Pair/order | Variant | Train | Elapsed incl. feed | Merge | Common prep | Peak RSS | Minimum available |
|---|---|---:|---:|---:|---:|---:|---:|
| 1, C → B2 | C | 42.887 s | 47.701 s | 28.049 s | 18.928 s | 3.43 GiB | 4.24 GiB |
| 1, C → B2 | B2 | 34.942 s | 38.965 s | 19.602 s | 12.291 s | 3.42 GiB | 4.33 GiB |
| 2, B2 → C | C | 44.451 s | 48.700 s | 29.605 s | 19.908 s | 3.39 GiB | 4.34 GiB |
| 2, B2 → C | B2 | 34.520 s | 38.768 s | 19.415 s | 12.124 s | 3.39 GiB | 4.28 GiB |
| 3, C → B2 | C | 40.664 s | 44.879 s | 26.485 s | 18.153 s | 3.42 GiB | 4.33 GiB |
| 3, C → B2 | B2 | 35.088 s | 39.137 s | 19.579 s | 12.029 s | 3.42 GiB | 4.28 GiB |

Sample CV is sample standard deviation divided by mean.

| Metric | C median | C min–max | C CV | B2 median | B2 min–max | B2 CV |
|---|---:|---:|---:|---:|---:|---:|
| Train | 42.887 s | 40.664–44.451 s | 4.46% | 34.942 s | 34.520–35.088 s | 0.85% |
| Elapsed incl. feed | 47.701 s | 44.879–48.700 s | 4.21% | 38.965 s | 38.768–39.137 s | 0.47% |
| Merge | 28.049 s | 26.485–29.605 s | 5.56% | 19.579 s | 19.415–19.602 s | 0.52% |
| Common prep | 18.928 s | 18.153–19.908 s | 4.63% | 12.124 s | 12.029–12.291 s | 1.09% |

## Paired deltas and ratios

Ratios are B2 divided by C; negative deltas mean B2 used less time.

| Pair | Train delta / ratio | Elapsed delta / ratio | Merge delta / ratio | Common prep delta / ratio |
|---|---:|---:|---:|---:|
| 1 | -7.945 s / 0.815 | -8.736 s / 0.817 | -8.447 s / 0.699 | -6.637 s / 0.649 |
| 2 | -9.931 s / 0.777 | -9.933 s / 0.796 | -10.190 s / 0.656 | -7.784 s / 0.609 |
| 3 | -5.576 s / 0.863 | -5.742 s / 0.872 | -6.905 s / 0.739 | -6.124 s / 0.663 |
| Median paired | -7.945 s / 0.815 | -8.736 s / 0.817 | -8.447 s / 0.699 | -6.637 s / 0.649 |

B2 was faster on train, elapsed, merge, and common prep in each pair. The paired median ratios correspond to 18.5% lower train time, 18.3% lower elapsed time, 30.1% lower merge time, and 35.1% lower common-prep time for the complete B2 combination. These paired measurements support choosing B2 over C when selecting between these two complete combinations. They do not measure an isolated weight-query or atomic-corpus effect. The prior unpaired C and B2 timings were not used to select the result.

## Lookup and phase details

B2's weight lookup build times were 13.217, 12.713, and 11.357 ms (median 12.713 ms, range 11.357–13.217 ms). The lookup used 3,220,160 bytes per run; peak selected lookup storage was 800,128 bytes per run. B2 recorded 1,435 fused batches and 17,301,504 bytes peak valid-start storage in each call. These storage/build values are already covered by B2's common-prep definition where applicable; the standalone lookup build time is exposed for inspection.

Other worktree phases (plan, delta, rewrite, commit, corpus preparation, fused prepare) and their complete measurements are preserved in the raw JSONL files. Fused prepare lies inside B2's delta path; do not add it to delta. Whole-train differences describe C versus B2 as complete variants; they do not identify the isolated query effect or atomic effect.

## Memory, CPU, and correctness

Sampled process VmSwap peak was zero in all six runs. Minimum `MemAvailable` ranged from 4.24 to 4.34 GiB. Every run records child user/system CPU, faults, context switches, child wall duration, host CPU/load snapshots, and host paging counters in its result record.

The six-run full-signature gate passed: model SHA-256 `d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd`, actual vocabulary 50,000, 29,243 merges, 1,429,915 unique words. Gate record: `optimization-combination-stability.signature.json`.

## Raw artifacts

Each call has JSONL, environment, stdout, and stderr files: `optimization-combination-stability.p1.c.*`, `.p1.b2.*`, `.p2.b2.*`, `.p2.c.*`, `.p3.c.*`, and `.p3.b2.*`.
