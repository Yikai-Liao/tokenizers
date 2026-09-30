# Fair native comparison (512 MiB)

Three native worktrees were built through the original `BpeTrainer::train` API and timed once each on the exact PR input. Pair-count initialization uses one worker for `count1-parallel`; the main comparison and atomic control use four. All three use four merge workers. The PR run is included as a reference point; the older `parallel-key-512-final` u16 results are legacy data and are not used in this comparison.

## Fixed inputs and conditions

- Corpus: 536,870,289 bytes, SHA-256 `a0d40d4102cba1e933f25e5ccd17552d2eaebaec4ef770bc39d8e46e740b4d4f`.
- Source dataset: `wikimedia/wikipedia`, revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, first two shards in order, normalized paragraphs; no repetition or synthetic additions. Full manifest and preflight details are in each `.environment.json`.
- Training parameters: split `none`, reference backend, vocab size 50,000, minimum frequency 2.
- Input feed: one thread; training merge pool: four threads. `TOKENIZERS_PARALLELISM=false`, `RAYON_NUM_THREADS=4`; the instrumented native runner toggles parallelism off for feed and on afterward.
- Stop policy: sample every 0.5 s; stop only if system `MemAvailable <= 1 GiB`. Record process `VmSwap` and host swap/page counters; process swap alone is not a stop condition.
- Each worktree and binary was run once. This is a focused critical comparison, not a performance distribution across repetitions.

## Results

| Implementation | Worktree commit | Pair-count init workers | Merge workers | Atomic corpus | Train | Initialize | Merge | Peak RSS | Min. available | Process VmSwap peak |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PR #2348 | `6ac0de5359d9e0e1ed0608422575a360ef91b908` | 1 feed; PR count is serial | 4 | n/a | 308.111 s | 110.979 s* | 188.533 s* | 6.57 GiB | 1.52 GiB | 0 B |
| `count1-parallel` | `63e384b81786e6d3e0e865a7194acaf8e08e008f` | 1 | 4 | no | 96.906 s | 61.722 s | 30.859 s | 3.63 GiB | 4.28 GiB | 0 B |
| `count4-parallel` | `c07a6e398b85b875f309adedd853995542f0d024` | 4 | 4 | no | 54.760 s | 22.975 s | 27.457 s | 3.39 GiB | 4.42 GiB | 0 B |
| `atomic-parallel` | `f2c5415fa21d7c75a09a594d22cca1143e34636b` | 4 | 4 | yes | 52.240 s | 22.365 s | 26.342 s | 3.39 GiB | 4.45 GiB | 0 B |

\* For PR, this is the sum of the instrumented alphabet, tokenization, and pair-count phases, not a complete initialization measure. The remaining time is outside those inserted clocks. Native `initialize_ms` is the trainer's recorded initialization phase. Feed time is separate and excluded from `train_ms`.

The native corpus and postings use `u32` slots/offsets and the recorded layout is `parallel_u32_flat32`. The atomic control has the same recorded layout and `atomic_corpus=true`. Exact stats, including allocated initial corpus/posting bytes, are in each result JSONL. The u32 corpus allocation was 849,691,660 bytes; initial posting allocation was 1,147,872,496 bytes in all three cases.

## Complete model-signature gate

The PR and all three native runs match on the full signature:

- Model SHA-256: `d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd`
- Actual vocabulary: 50,000
- Merges: 29,243
- Unique words: 1,429,915

The checked result is `native-fair-512.signature.json`. A mismatch would have stopped the signature checker.

## Memory and host activity

All native runs remained well above the 1 GiB `MemAvailable` stop threshold. Sampled process `VmSwap` was zero throughout. Host swap and page counters are recorded per run in the environment/result records; they include unrelated system activity and are not attributed to an individual benchmark process.

The PR's measured RSS (6.57 GiB) was notably above the preflight estimate (4.57 GiB), reflecting merge-time historical cohorts and heap growth that the estimate did not capture. The native runs peaked between 3.39 and 3.63 GiB under the same input. These are single observations; no variance estimate is available.

## Provenance and artifacts

Each native environment record captures the worktree commit, all tracked Rust/Cargo source hashes, actual instrumented build-copy and runner hashes, binary hash, corpus hash/manifest, configuration, and starting host memory counters. Binary SHA-256 values:

- `count1-parallel`: `d4fb43dec1f7b0158ff13c75fc9c7641a08785c3446ea481b34c15fd842cd4d2`
- `count4-parallel`: `889a8e7b2bda9bacb00f1a8ea6bfe9a5c3a231b0195e5ee26d9e20cf67023eec`
- `atomic-parallel`: `13ccd7d2b3849380bd85aee88b09f77af85718bc9b1cbe2cafbe33c723ede7b9`

Raw results and environment locks are `native-fair-512.count1.jsonl`, `native-fair-512.count1.environment.json`, `native-fair-512.count4.jsonl`, `native-fair-512.count4.environment.json`, `native-fair-512.atomic.jsonl`, and `native-fair-512.atomic.environment.json`. The PR record is `pr-fair-512.jsonl` with its own environment lock and phase log.

The measurement commits above identify the exact trainer versions. Any later cleanup of legacy benchmark runners or report documents is separate from these measured worktree commits and does not alter this data.
