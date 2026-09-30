# H cost and route-count diagnostic

This was one diagnostic call, not a performance ranking. It measures the H training path to locate the post-merge timer gap and count the flat-route work. The instrumented source lives in an isolated copy under `.build/native-h-cost-probe`; the formal H source and binary were not changed.

## Provenance and gates

- Base source: clean H worktree commit `00216d914186e42458d45e72276b13c700749c6a` (`weight-one-buckets`).
- Probe source: copy of H's native runner source with temporary counter/timer instrumentation. `parallel.rs` SHA-256 `ca7f2d2be9ff91ce30fff62a3da770adc1745a52ce8de18286a8f0275ff499be`; runner `main.rs` SHA-256 `5589c1daf1abbb58cbbe944f23f2654710fdf8da6d6821b958fb2db6d0beeefd`; runner manifest SHA-256 `cf74c491c398866927e781d2688beb4e2988586fe0f9a1130faab844f85af801`.
- Build: offline `cargo build --release`, opt-3 release defaults, no debug or strip override. Isolated target `.build/h-cost-probe-target`. Probe binary SHA-256 `3770296ea578243792c7b6c24e37352fb9611b96ae4726a6ea248f58c0c344f8`.
- Input `zh-512m.txt`, 536,870,289 bytes, SHA-256 `a0d40d4102cba1e933f25e5ccd17552d2eaebaec4ef770bc39d8e46e740b4d4f`; reference backend, split `none`, vocab 50,000, minimum frequency 2, four initialization and four merge workers, atomic corpus, `parallel_u32_flat32` layout.
- The original `bench_indexed_stats` record appeared exactly once. Full output signature matched H: SHA-256 `d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd`, vocab 50,000, 29,243 merges, 1,429,915 unique words. Initial symbols/edges/pairs were 204,660,029 / 203,230,114 / 2,697,517; initial corpus/posting bytes were 849,691,660 / 802,967,868.
- Peak RSS was 4,649,184 KiB; minimum available memory was 3,580,387,328 bytes; sampled process swap was zero. Host paging deltas were 82 pages in and 3,042 pages out. The 1 GiB available-memory stop gate did not trigger.

## Timer accounting

All values below are for this single instrumented run. `initialize_ms` includes tokenization (`tokenize_ms=1,224.012`); tokenization was not subtracted a second time.

| Timer | Value |
|---|---:|
| Train | 30,434.470 ms |
| Initialize | 7,636.320 ms |
| Merge | 18,018.384 ms |
| Train − initialize − merge | **4,779.766 ms** |
| Instrumented post-merge conversion/drop/payload interval | **4,777.022 ms** |

The probe measures conversion, inventory, and each explicit drop after the unchanged `stats.merge_ms` endpoint. Their combined time accounts for the post-merge interval to within about 2.744 ms, which includes report emission and return overhead. `owners` destruction took 4,751.972 ms, or **99.48%** of the measured post-merge interval. This is a direct time measurement on one diagnostic call, not a speedup estimate.

| Post-merge operation | Time | Detail |
|---|---:|---|
| Vocabulary conversion | 11.619 ms | `ids` to owned output strings |
| Merge conversion | 9.666 ms | merge IDs to owned output strings |
| Special-token clone | 0.004 ms | |
| Owner inventory | 0.000 ms | 10,572,128 ledger entries and 14,225,344 heap items before drop |
| Drop weight lookup | 0.001 ms | |
| Drop owners | **4,751.972 ms** | owner ledgers and their postings/heaps |
| Drop blocks | 0.011 ms | |
| Drop corpus | 3.491 ms | |
| Drop lengths | 0.003 ms | |
| Drop strings | 0.254 ms | |
| Return-payload construction | 0.001 ms | |

The explicit diagnostic drop order is weight lookup, owners, blocks, corpus, lengths, then strings. The `owners` count scan happens before those drops. The counters/timers do not alter the merge algorithm or posting/span updates.

## Flat-route counts

| Counter | Count |
|---|---:|
| Flat `Output::remove` calls | 173,155,595 |
| Physical births (`Σ route.nodes.len()`) | 173,155,595 |
| Local groups with `occurrences > 0` | 29,000,154 |
| Local groups with `occurrences == 0` | 28,835,989 |
| Total local delta groups | 57,836,143 |
| Fused batches / AA fallback batches / other flat fallbacks | 1,435 / 115 / 0 |

The local group counts are deduplicated route-map keys across output chunks and batches. They are a theoretical deduplication lower bound for work represented by the routes, **not** counts of J's actual task/rule/side flushes. Birth keys are typically single-producer; remove keys can arrive from multiple rules or sides. The probe does not measure J flushes.

The diagnostic group scan took 653.054 ms and is reported separately as `route_scan_ms`. It ran outside the existing subphase timers but inside the broader merge timer, so the 18.018 s merge value includes this measurement overhead and must not be compared to formal H timing.

## Artifacts

- Raw result and runner provenance: `optimization-h-cost-probe.jsonl` and `optimization-h-cost-probe.environment.json`
- Program output: `optimization-h-cost-probe.stdout` and `.stderr` (`bench_cost_stats` is in stderr)
- Isolated probe source and runner: `../.build/native-h-cost-probe/`
- Binary: `../.build/h-cost-probe-target/release/hf-bpe-native-h-cost-probe`
- DWARF source-attribution report: [optimization-h-debug-profile.md](optimization-h-debug-profile.md)
