# Optimization B2: fused weight lookup (512 MiB)

One focused 512 MiB run of the `fused-lookup` worktree through the original trainer API. This candidate keeps the fused direct-corpus atomic path and adds a 256-bucket weight lookup directory plus a selected-ID lookup table. The run had four initialization workers, four merge workers, u32 token IDs/postings, and `atomic_corpus=true`.

## Locked setup

- Worktree: `/root/code/tokenizers-worktrees/fused-lookup`, commit `a0832c488a7ece429630c7d1493da5dd772c87aa`.
- Binary: `hf-bpe-native-fused-lookup`, SHA-256 `596200773ff413504d82aa4038b3c5e5902e835dff2bbdfb362e5d0818d208a5`.
- Input: `zh-512m.txt`, 536,870,289 bytes, SHA-256 `a0d40d4102cba1e933f25e5ccd17552d2eaebaec4ef770bc39d8e46e740b4d4f`.
- Parameters: reference backend, split `none`, vocab size 50,000, minimum frequency 2; feed parallelism off; four initialization and merge workers; `parallel_u32_flat32` layout.
- Worktree source, instrumented build copy, runner, build script, binary, input, and environment hashes are recorded in `optimization-b2-512.environment.json`.
- A separate read-only proof review was active concurrently. No test, build, or other benchmark process ran during this timed call.

## Result

| Measurement | B2 fused lookup |
|---|---:|
| Feed | 4.265 s |
| Train | 40.265 s |
| Initialization | 14.399 s |
| Merge | 21.386 s |
| Fused prepare | 11.889 s |
| Fused batches | 1,435 |
| Peak valid-start storage | 17,301,504 bytes (16.50 MiB) |
| Weight lookup build | 11.784 ms |
| Weight lookup size | 3,220,160 bytes (3.07 MiB) |
| Peak selected lookup size | 800,128 bytes (0.76 MiB) |
| Alphabet / scratch / character table | 0.345 s / 12,135,736 bytes / 4,456,448 bytes |
| Corpus measure / allocate / fill | 0.189 s / 0.000035 s / 0.681 s |
| Initial corpus / postings | 849,691,660 bytes / 1,147,872,496 bytes |
| Peak RSS | 3.39 GiB (3,550,348 KiB runner HWM; 3,627,929,600 bytes sampled) |
| Minimum `MemAvailable` | 4.41 GiB (4,740,362,240 bytes) |
| Sampled process VmSwap peak | 0 bytes |
| Host `pswpin` / `pswpout` deltas | 88 / 845 pages |

The trainer-reported merge subphases were plan 0.039 s, delta 12.196 s, rewrite 0.765 s, select 0.320 s, and commit 7.952 s. `fused_prepare_ms` is reported separately from `delta_ms`; do not add these two counters as disjoint wall-time components.

## Comparison with the prior fused run and Optimization C

| Variant | Atomic corpus | Init workers | Initialize | Merge | Train | Fused prepare | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Optimization C, direct non-fused | no | 4 | 9.767 s | 26.092 s | 39.805 s | — | 13.888 s |
| Optimization B, fused direct | yes | 4 | 13.251 s | 31.887 s | 49.697 s | 22.858 s | 23.142 s |
| Optimization B2, fused lookup | yes | 4 | 14.399 s | 21.386 s | 40.265 s | 11.889 s | 12.196 s |

Against the prior fused run, B2 recorded 10.501 s less merge time and 9.432 s less total train time; its initialization was 1.148 s longer. The fused-prepare and delta values are shown independently and must not be summed. Against C, B2's merge was 4.706 s faster (18.04%), while total training was 0.460 s longer (1.16%). The whole-train result therefore does not establish an improvement.

Initialization source is unchanged from C. In particular, pair-count time was 12.162 s for B2 versus 8.015 s for C; the weight directory is built later, during merge setup, so its runtime cannot explain this initialization difference. These are single-run observations, and the reason for the initialization timing gap is unknown.

The fused implementation combines weight-lookup changes with atomic corpus storage. No matched fused non-atomic control was run, as requested, so these results do not isolate an atomic-corpus effect. Each variant has one timing; the differences are observations, not a variance estimate.

## Process and host diagnostics

Child `RUSAGE_CHILDREN` delta: 116.496 user CPU seconds, 7.016 system CPU seconds, 788,299 minor faults, 0 major faults, 25,428 voluntary context switches, and 32,391 involuntary context switches. Child wall duration was 45.625 s.

Host aggregate CPU busy fraction was 53.68%, steal fraction 0.037%. Load average (1/5/15 minute) changed from `[1.28, 1.58, 1.48]` to `[1.89, 1.74, 1.55]`. These host counters are interval-wide and do not attribute activity to the benchmark process.

## Model signature gate

B2 matches the PR and all prior native runs on model SHA-256 `d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd`, actual vocabulary 50,000, 29,243 merges, and 1,429,915 unique words. The ten-run gate result is `optimization-b2-512.signature.json`.

Raw result, phase stats, memory and process/host diagnostics are in `optimization-b2-512.jsonl`; stderr and full provenance are in the matching `.stderr` and `.environment.json` files.
