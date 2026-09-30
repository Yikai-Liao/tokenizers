# Fused lookup stability comparison (512 MiB)

This focused check compares only B and B2, three immediate pairs in the prescribed order `B, B2 / B2, B / B, B2`. It uses the same corpus, feed path, initialization and merge worker counts, u32 layout, and atomic corpus mode for both variants. No atomic control was run.

## Locked setup

- Corpus: 536,870,289 bytes, SHA-256 `a0d40d4102cba1e933f25e5ccd17552d2eaebaec4ef770bc39d8e46e740b4d4f`.
- B: `fused-direct-atomic`, commit `c1ee201978aadd853d032621add6e056d1949a76`, binary SHA-256 `c23c02edeb11c14687c4f30695cc2d19475646c0e8fd75ee6755b9b440dcb330`.
- B2: `fused-lookup`, commit `a0832c488a7ece429630c7d1493da5dd772c87aa`, binary SHA-256 `596200773ff413504d82aa4038b3c5e5902e835dff2bbdfb362e5d0818d208a5`.
- All six calls: backend `reference`, split `none`, vocab 50,000, min frequency 2, feed parallelism disabled, initialization workers 4, merge workers 4, `parallel_u32_flat32`, atomic corpus enabled.
- Environment, worktree and instrumented build-copy source hashes, binary/input hashes, RSS/swap, child usage and host counters were captured separately for every run in the paired `.environment.json` and `.jsonl` files.
- No build, test, download, or other benchmark ran during the six measurements.

## Fused prepare time

Sample CV is sample standard deviation divided by the mean. Lower B2/B ratios indicate less prepare time for B2.

| Variant | Runs (s) | Median | Min–max | Sample CV |
|---|---|---:|---:|---:|
| B | 22.301, 20.249, 22.410 | 22.301 s | 20.249–22.410 s | 5.62% |
| B2 | 11.704, 11.173, 12.080 | 11.704 s | 11.173–12.080 s | 3.91% |

| Pair | Order | B prepare | B2 prepare | B2 − B | B2/B |
|---|---|---:|---:|---:|---:|
| 1 | B → B2 | 22.301 s | 11.704 s | -10.597 s | 0.525 |
| 2 | B2 → B | 20.249 s | 11.173 s | -9.076 s | 0.552 |
| 3 | B → B2 | 22.410 s | 12.080 s | -10.331 s | 0.539 |
| Median paired result | — | — | — | -10.331 s | 0.539 |

B2's fused prepare counter was 44.8%–47.5% lower in each pair; the median paired ratio was 0.539, a 46.1% reduction. `fused_prepare_ms` is an instrumented counter within the merge flow. It overlaps with `delta_ms`; these counters are reported separately and must not be added together.

## Lookup build and storage cost

The B2 weight lookup is built during merge setup, after initialization. This cost is reported separately from fused prepare:

| Measurement | Runs | Median | Min–max | Sample CV |
|---|---|---:|---:|---:|
| Weight lookup build | 12.368, 15.146, 11.275 ms | 12.368 ms | 11.275–15.146 ms | 15.43% |
| Weight lookup allocated bytes | 3,220,160 each run | 3,220,160 B | 3,220,160–3,220,160 B | 0% |
| Peak selected lookup bytes | 800,128 each run | 800,128 B | 800,128–800,128 B | 0% |

Each call completed 1,435 fused batches. Peak valid-start storage was 17,301,504 bytes in each run.

## Delta counters and whole-run context

| Pair | B delta | B2 delta | B2 − B | B2/B |
|---|---:|---:|---:|---:|
| 1 | 22.578 s | 12.027 s | -10.552 s | 0.533 |
| 2 | 20.507 s | 11.472 s | -9.035 s | 0.559 |
| 3 | 22.701 s | 12.397 s | -10.304 s | 0.546 |
| Median paired result | — | — | -10.304 s | 0.546 |

For context only, initialization and total train time by call were: B `11.963 / 46.033 s`, B2 `9.599 / 33.117 s`; B2 `11.163 / 33.804 s`, B `10.107 / 41.531 s`; B `10.674 / 45.164 s`, B2 `11.287 / 35.492 s`. Do not attribute these initialization differences to the lookup: the directory is built after initialization. These whole-run timings are secondary to this module-focused comparison.

## Memory, CPU, and correctness

Peak RSS was 3.39–3.42 GiB across the six runs; minimum available memory was 4.29–4.40 GiB; sampled process VmSwap was zero in all runs. Per-run child CPU time, minor/major faults, context switches, host CPU/load snapshots, and paging counters are in the raw records.

The full model signature gate matched the PR and prior native results across seven records (PR plus these six calls): SHA-256 `d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd`, vocab 50,000, 29,243 merges, and 1,429,915 unique words. Gate output: `optimization-lookup-stability.signature.json`.

## Artifacts

The six call records are `optimization-lookup-stability.p1.b.jsonl`, `.p1.b2.jsonl`, `.p2.b2.jsonl`, `.p2.b.jsonl`, `.p3.b.jsonl`, and `.p3.b2.jsonl`, each with its own environment, stdout, and stderr file. This is the complete three-pair stability check; no additional calls are included.
