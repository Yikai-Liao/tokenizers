# Optimization B: fused direct corpus training (512 MiB)

One focused 512 MiB run of the `fused-direct-atomic` worktree through the original trainer API. This candidate combines direct corpus construction with fused batch preparation and atomic corpus storage. It is a combined variant: the user canceled the proposed isolated corpus-C atomic control, so this run cannot identify or support a standalone atomic-corpus effect.

## Locked setup

- Worktree: `/root/code/tokenizers-worktrees/fused-direct-atomic`, commit `c1ee201978aadd853d032621add6e056d1949a76`.
- Binary: `hf-bpe-native-fused-direct`, SHA-256 `c23c02edeb11c14687c4f30695cc2d19475646c0e8fd75ee6755b9b440dcb330`.
- Input: `zh-512m.txt`, 536,870,289 bytes, SHA-256 `a0d40d4102cba1e933f25e5ccd17552d2eaebaec4ef770bc39d8e46e740b4d4f`.
- Parameters: reference backend, split `none`, vocab size 50,000, minimum frequency 2; feed parallelism off, four initialization workers and four merge workers; u32 token IDs/postings, `parallel_u32_flat32`, `atomic_corpus=true`.
- Provenance hashes for the worktree, instrumented source copy, runner, build script, binary, corpus, and environment are in `optimization-b-512.environment.json`.

## Result

| Measurement | Fused direct atomic |
|---|---:|
| Feed | 4.020 s |
| Train | 49.697 s |
| Initialization | 13.251 s |
| Merge | 31.887 s |
| Alphabet collection | 0.413 s |
| Alphabet scratch | 12,135,736 bytes (11.57 MiB) |
| Character table | 4,456,448 bytes (4.25 MiB) |
| Corpus measure / allocate / fill | 0.184 s / 0.000025 s / 0.512 s |
| Fused prepare | 22.858 s |
| Fused batches | 1,435 |
| Peak valid-start storage | 17,301,504 bytes (16.50 MiB) |
| Peak RSS | 3.42 GiB (3,584,200 KiB runner HWM; 3,668,107,264 bytes sampled) |
| Minimum `MemAvailable` | 4.40 GiB (4,722,102,272 bytes) |
| Sampled process VmSwap peak | 0 bytes |
| Host `pswpin` / `pswpout` deltas | 133 / 3,599 pages |

The initial corpus allocation was 849,691,660 bytes and initial postings were 1,147,872,496 bytes. The runner checked all nine required numeric phase/storage fields and verified workers=4, initialization_workers=4, the u32 flat32 layout, and atomic corpus mode.

## Process and host diagnostics

Child `RUSAGE_CHILDREN` delta: 147.814 user CPU seconds, 6.149 system CPU seconds, 566,284 minor faults, 0 major faults, 26,087 voluntary context switches, and 41,551 involuntary context switches. Child wall duration was 55.148 s.

Host aggregate CPU busy fraction was 54.86%, steal fraction was 0.036%. Load average (1/5/15 minute) changed from `[2.34, 2.15, 1.61]` to `[3.23, 2.42, 1.74]`. These are host-wide interval measurements, not per-process attribution.

## Model signature gate

This run matches the PR and all prior native runs on model SHA-256 `d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd`, actual vocabulary 50,000, 29,243 merges, and 1,429,915 unique words. The nine-run gate result is `optimization-b-512.signature.json`.

## Comparison limits and artifacts

The combined fused-direct-atomic result cannot isolate the atomic-corpus contribution because no matched fused-direct non-atomic control was timed. Comparisons with non-fused Optimization C combine changes to corpus construction, batch preparation, and atomic storage and should be read as whole-variant observations.

Raw result, phase stats, RSS and diagnostic counters are in `optimization-b-512.jsonl`; stderr and full provenance are in the matching `.stderr` and `.environment.json` files.
