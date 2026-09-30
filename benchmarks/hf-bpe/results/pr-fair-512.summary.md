# PR#2348 fair-run record (512 MiB)

This is the pinned PR implementation run on the 512 MiB corpus used by the existing three-run comparison. The native `count1-parallel-worktree` and `count4-parallel-worktree` runs requested for the revised comparison are pending; this PR-only result does not close their digest gate.

## Locked inputs and build

- PR checkout: `benchmarks/hf-bpe/.build/pr-head`, commit `6ac0de5359d9e0e1ed0608422575a360ef91b908` (PR #2348).
- Input: 536,870,289 bytes, SHA-256 `a0d40d4102cba1e933f25e5ccd17552d2eaebaec4ef770bc39d8e46e740b4d4f`.
- Corpus: pinned `wikimedia/wikipedia` revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, first two parquet shards in order; normalized source paragraphs, no repetition or synthetic text. The complete manifest is captured in `pr-fair-512.environment.json`.
- Preflight: `preflight-512.json`, SHA-256 `ff38d0a4501109598d789a9391e6a0e494e797eb14277a7a029984711bfc7404`; 1,445,902 upper-bound lines, 203,974,788 upper-bound edges, 4,302,752 upper-bound pairs, alphabet 20,757.
- Binary: `target/release/hf-bpe-profiled-pr4`, SHA-256 `65280aba2ba180ab03a9d2e6c8fc898dff37daed19624d87e68da5f8e9e00c8c`.
- Instrumented source files, runner, Cargo files, and build scripts are individually SHA-256 locked in the environment JSON. Instrumentation adds phase timers to a disposable checkout; the pinned PR checkout itself remained unchanged.
- Parameters: vocab size 50,000; minimum frequency 2; split `none`; reference backend; one corpus-feed thread; four Rayon merge workers; parallel word-scan threshold 1,000.
- Environment overrides: `TOKENIZERS_PARALLELISM=false`, `RAYON_NUM_THREADS=4`, `TOKENIZERS_TRAIN_PARALLEL_MIN=1000`. Runner sets four threads and disables parallelism during feed, then enables parallelism for training.

## Algorithm and representation

PR#2348 uses a `WordArena` with `Symbol { u32 ID, u32 length }`, historical word cohorts, one merge rule per round, an `OctonaryHeap`, and conditional parallel scans over words. Its merge-time word processing becomes parallel once the cohort reaches the configured 1,000-word threshold. The `u32` symbol ID and length are part of the PR representation and remain present during training.

The revised native comparison is being built with `u32` token IDs and `u32` postings in both variants. That representation alignment does not remove algorithm-specific state such as the PR's per-symbol length field.

## Result

| Measurement | PR#2348 |
|---|---:|
| Train time | 308.111 s |
| Total elapsed | 312.599 s |
| Feed time | 4.488 s |
| Peak RSS | 6.57 GiB (6,888,164 KiB from runner; 6,829,707,264 bytes sampled) |
| Minimum system MemAvailable | 1.52 GiB (1,633,619,968 bytes) |
| Peak process VmSwap sampled every 0.5 s | 0 bytes |
| Global swap used, before → after | 5.56 → 5.87 GiB |
| Global `pswpin` / `pswpout` deltas | 23,376 / 103,130 pages |
| Unique words | 1,429,915 |
| Merges | 29,243 |
| Actual vocabulary | 50,000 |
| Model SHA-256 | `d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd` |

Phase timers: alphabet 2.467 s; tokenize words 16.933 s; count pairs 91.579 s; merges 188.533 s; finalize model 0.026 s. Special-token setup rounded to 0.000 s.

The process stayed above the 1 GiB `MemAvailable` stop threshold, so the run completed. Process swap remained zero; global swap counters increased during the run and are recorded as host-wide activity, not attributed solely to this process.

The pre-run sizing formula predicted 4.57 GiB, while measured peak RSS was 6.57 GiB. The estimate omits growth in historical cohorts and merge-time heap state and is not an abort gate. Future scheduling should use this observed peak as the more useful planning reference.

## Comparison status and artifacts

The complete model signature is recorded in `pr-fair-512.jsonl`. Its digest matches the older `parallel-key-512-final.jsonl` result (`d50fb836…`), but that historical native runner used the prior token-width implementation and is retained only as background; it is not the revised fair comparison. The equality gate for the new `u32` count1/count4 native worktrees remains pending.

The previous `pr-key-512.jsonl` attempt was aborted by its now-superseded policy that treated any process swap as a stop condition. This successful rerun used the corrected policy: stop only when `MemAvailable <= 1 GiB`, while recording process and host swap.

Raw output, phase logs, environment/provenance, and the monitored runner are `pr-fair-512.jsonl`, `pr-fair-512.stderr`, `pr-fair-512.environment.json`, and `run_pr_fair.py`.
