# D candidate: stable radix initial counting, 512 MiB corpus

## Provenance and configuration

- Worktree: `/root/code/tokenizers-worktrees/initial-radix`, commit `d15c18cc07047479ddc2eacd1da1844cb9ac9358`.
- Binary: `benchmarks/hf-bpe/target/release/hf-bpe-native-initial-radix`; SHA-256 `4c7d1268470da0eaa02f6f48b242defd0ace0614f4461527ff26ddbf1430ac1b`.
- Input: `benchmarks/hf-bpe/.build/gb-corpus/zh-512m.txt`, 536,870,289 bytes; SHA-256 `a0d40d4102cba1e933f25e5ccd17552d2eaebaec4ef770bc39d8e46e740b4d4f`.
- One screening run using `run_native_fair.py`; result and environment/diagnostic sidecars are `optimization-d-512.jsonl`, `optimization-d-512.environment.json`, and corresponding stdout/stderr files.
- Configuration: reference backend, split `none`, vocabulary target 50,000, minimum frequency 2, initial/merge thread counts 4/4, atomic corpus enabled, u32 symbols. Configuration assertions passed; recorded `initial_count_backend` is `stable_radix16`.

## Timings and model

| Metric | Result |
|---|---:|
| Train time | 33.264576 s |
| End-to-end elapsed | 37.360854 s |
| Feed time | 4.096278 s |
| Initialization | 9.616983 s |
| Merge | 19.267330 s |
| Merge planning / delta / rewrite / commit | 0.039351 / 11.711063 / 0.764616 / 6.353040 s |
| Actual vocabulary / merges / unique words | 50,000 / 29,243 / 1,429,915 |
| Model SHA-256 | `d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd` |

The full signature check against PR, C, and B2 passed: model hash, vocabulary size, merge count, and unique-word count all match.

## Initial stable-radix counting

| Phase or allocation | Result |
|---|---:|
| Symbols / edges / initial pairs | 204,660,029 / 203,230,114 / 2,697,517 |
| Weight lookup | 9.545 ms; 3,220,160 B |
| Route compaction | 1,327.454 ms |
| Stable radix sort | 2,044.553 ms |
| Group counting | 2,682.716 ms |
| Posting installation | 1,036.878 ms |
| Route buffer / peak route buffer | 1,625,840,912 / 3,251,681,824 B |
| Radix scratch / group buffer | 1,625,840,912 / 100,663,296 B |
| Initial corpus / posting storage | 849,691,660 / 802,967,868 B |

Other initialization detail: alphabet setup 368.161 ms (12,135,736 B scratch; 4,456,448 B character table); corpus measure/allocate/fill 188.843 / 0.040 / 668.899 ms; fused preparation 11.426 s over 1,435 batches with 17,301,504 B peak valid-start storage.

The initial corpus bytes are unchanged from C/B2. Final initial posting allocation fell from 1,147,872,496 to 802,967,868 bytes; the initial owner table estimate fell from 276,824,128 to 138,412,096 bytes. The new path requests counted posting capacity and prunes before building the owner table; count 3 still follows SmallPosting's minimum 4-slot heap allocation. The logical signature and N/E/pair counts match. This screening run is promising but does not establish a stable whole-train improvement over B2; the combined candidate will be compared with the current winner.

## Resource diagnostics

- Process maximum RSS: 4,648,388 KiB (4.43 GiB); sampled peak RSS: 4,760,743,936 B (4.43 GiB).
- Minimum host `MemAvailable`: 3,640,709,120 B (3.39 GiB); process sampled swap remained zero. Host swap used changed by 5,218,304 B; host `pswpin`/`pswpout` deltas were 69/1,515 pages.
- Child usage deltas: 90.663 s user CPU, 9.441 s system CPU, 1,112,305 minor and 0 major faults, 24,026 voluntary and 26,038 involuntary context switches; measured wall time 38.605 s.
- Host CPU busy fraction 52.2%, steal 0.0302%; load average changed from 0.74/1.32/1.48 to 2.35/1.68/1.60.
- Available memory at start was 8,317,784,064 B (7.75 GiB). No strict RSS gate was applied.

Raw record, environment, and signature-check output retain the full diagnostics and provenance.
