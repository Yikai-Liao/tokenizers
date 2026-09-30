# Optimization A merge-time diagnostic (512 MiB)

To investigate the earlier single-run observation that Optimization A's merge phase was about 6.5 s slower than `count4-parallel`, I ran one additional baseline call followed immediately by one candidate call. Both used the same 512 MiB input and trainer configuration. The count4 baseline and corpus-parallel candidate have identical merge-loop source; their worktree commits differ in corpus preparation and related bookkeeping.

These two runs did **not** reproduce the merge slowdown. They do not identify the cause of the earlier result. Host load changed during both calls, so the recorded system activity is useful context, but it does not establish that contention caused the earlier gap.

## Locked setup

- Corpus: 536,870,289 bytes, SHA-256 `a0d40d4102cba1e933f25e5ccd17552d2eaebaec4ef770bc39d8e46e740b4d4f`.
- Both runs: split `none`, reference backend, vocab 50,000, minimum frequency 2, four initialization workers, four merge workers, u32 token IDs and postings, non-atomic corpus.
- Baseline: `parallel-count4`, commit `b6a28768feb4af4181f76fc1fb3f78993644f5c9`, binary SHA-256 `889a8e7b2bda9bacb00f1a8ea6bfe9a5c3a231b0195e5ee26d9e20cf67023eec`.
- Candidate: `corpus-parallel`, commit `98ca7fc1c258d0661177d3aae15cca71536c3df4`, binary SHA-256 `2f3f2d438689c33a2cdae0b106236882e24e2ab2df99c7a2e6707ec56cf8cc43`.
- Actual copied source hashes, instrumented runner hashes, corpus manifest, and build metadata are in the two `.environment.json` records. The baseline measurement reuses its existing release binary/build copy; the candidate reuses the already built Optimization A binary/build copy.
- Process RSS and swap were sampled every 0.5 s. Runs stop only if `MemAvailable <= 1 GiB`.

## Side-by-side measurements

| Measurement | Baseline count4 | Optimization A | A minus baseline |
|---|---:|---:|---:|
| Train | 57.982 s | 49.737 s | -8.245 s |
| Initialization | 23.760 s | 15.421 s | -8.339 s |
| Merge | 29.810 s | 29.756 s | -0.054 s |
| Feed | 4.595 s | 4.386 s | -0.208 s |
| Peak RSS | 3.39 GiB | 3.42 GiB | +0.02 GiB |
| Minimum `MemAvailable` | 4.39 GiB | 4.34 GiB | -0.05 GiB |
| Sampled process VmSwap peak | 0 B | 0 B | — |

Initialization phases (ms):

| Phase | Baseline | Optimization A |
|---|---:|---:|
| Alphabet | included in existing timing | 2,562.940 |
| Tokenize words | 13,893.478 | 5,436.163 |
| Corpus region measurement | not separately recorded | 199.344 |
| Corpus allocation | not separately recorded | 265.864 |
| Corpus fill | not separately recorded | 2,407.319 |
| Initial route | 723.562 | 460.444 |
| Initial pair count | 9,049.197 | 9,445.345 |

Merge subphases (ms):

| Phase | Baseline | Optimization A |
|---|---:|---:|
| Plan | 4,275.032 | 4,261.383 |
| Delta | 15,856.170 | 15,978.918 |
| Rewrite | 732.285 | 726.667 |
| Select | 326.092 | 289.353 |
| Commit | 8,554.290 | 8,430.523 |
| Trainer recorded merge total | 29,810.105 | 29,756.126 |

The measured merge totals and each major merge subphase are nearly equal. The previous 27.457 s baseline versus 33.964 s candidate result was a single observation and was not reproduced by this immediate two-call comparison.

## Process and host diagnostics

`RUSAGE_CHILDREN` deltas for baseline → candidate:

| Counter | Baseline | Candidate |
|---|---:|---:|
| User CPU seconds | 149.061 | 149.369 |
| System CPU seconds | 6.792 | 5.218 |
| Minor faults | 543,902 | 458,511 |
| Major faults | 0 | 1 |
| Voluntary context switches | 35,212 | 35,515 |
| Involuntary context switches | 38,378 | 37,572 |
| Child wall duration | 64.664 s | 55.639 s |

Host aggregate `/proc/stat` busy fraction was 48.2% during the baseline interval and 54.1% during the candidate interval; steal fraction was 0.046% and 0.048%. Load average (1/5/15 minute) changed from `[0.77, 0.68, 0.82]` to `[2.53, 1.21, 0.99]` during the baseline run and from `[2.19, 1.20, 0.99]` to `[3.39, 1.68, 1.17]` during the candidate run. These are host-wide observations, not per-process attribution.

The current evidence shows that the earlier merge-time gap is not repeatable in this pair of runs. The exact cause remains unknown; the host counters do not establish a specific explanation.

## Model signature gate and raw artifacts

Baseline and candidate match the PR and previous native baseline on model SHA-256 `d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd`, actual vocabulary 50,000, merges 29,243, and unique words 1,429,915. Gate output: `optimization-a-diagnostic.signature.json`.

Raw records: `optimization-a-diagnostic.baseline.jsonl` with its environment, stdout, and stderr; `optimization-a-diagnostic.candidate.jsonl` with its environment, stdout, and stderr.
