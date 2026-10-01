# Current candidate PERF audit

Date: 2026-10-01. This is an independent diagnostic of clean candidate `bpe/initial-owner-waves` at `029ab45bd0b0f446035f71acbd247c804bea60b8`. It does not change candidate or production source. The captured profile and summaries are under [results/current-perf-audit](results/current-perf-audit); the 296,943,378-byte (about 283 MiB) `perf.data` stays in the ignored `.build` directory and is identified by SHA-256 `1b8ec30aa074de3b238929477faedffaec7c679ed2ab0b7a7c936441857beaa3`.

## Result

The formal benchmarks already show a substantial end-to-end improvement. In the two paired DE→H comparisons, median train time fell from 32.397 s to 25.577 s (21.1%), and feed+train elapsed fell from 36.766 s to 29.838 s (18.8%). The subsequent H→J screening also improved both prepare and the full call. These results are documented in [the optimization report](OPTIMIZATION_REPORT.md).

This PERF audit examines what remains after those improvements. Its diagnostic timings do not replace the formal benchmark results. The remaining large costs are posting validation, neighbor-delta construction, and owner updates; this audit found no clear large redundant operation to remove, so this round of optimization stops here.

## Workload and measurement

One complete original-API training call used the fixed real 512 MiB Chinese Wikipedia input, 536,870,289 bytes, SHA-256 `a0d40d4102cba1e933f25e5ccd17552d2eaebaec4ef770bc39d8e46e740b4d4f`; `none`, reference frontend, vocabulary 50,000, minimum frequency 2, atomic u32 corpus, and four initialization/four merge workers. The full model digest matched the established result: `d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd`; vocabulary 50,000, 29,243 merges, 1,429,915 unique words.

The candidate was copied from the clean worktree into a unique diagnostic build root and unique Cargo target directory. The optimized release binary retained DWARF level 2 and symbols; ELF contains `.debug_info`, `.debug_line`, `.debug_abbrev`, and `.debug_str`. Binary SHA-256 is `45d0147acc733cf9d66e593916495ff164aec115f5de68bdbcf9d35107059b52`, Build ID `7faa1d5a8f37cd203ba61fb94200d5ba80ebfa42`. The runner adds only the existing private indexed-stat output probe and worker setup, calls the original trainer API, and uses Rust's default system allocator. Build details and source hashes are in [build.json](results/current-perf-audit/build.json); run parameters and monitoring are in [current.provenance.json](results/current-perf-audit/current.provenance.json).

The full workload gate passed against the prior H run: unique words `1,429,915`, initial symbols `204,660,029`, slots `206,089,945`, edges `203,230,114`, pair keys `2,697,517`, posting visits `125,409,599`, pruned pairs `21,518,771`, and model SHA-256 all match. The separate `stored_positions=207,224,101` inventory count has a different scope and is not part of this parity gate.

The one diagnostic call recorded `cycles:u` and generic `cache-misses:u` at 99 Hz with `--call-graph dwarf,16384`. It captured 8,719 cycles samples and 8,447 cache-miss samples, 17,166 total, zero lost. Weighted event periods were 221,481,049,753 cycles and 1,376,613,550 cache misses. The unresolved top-IP period was 9.66% and 12.96%, respectively. These are sampled top-IP event shares, not wall-time shares. The generic cache event and sampled IP skid do not identify a cache level or prove a particular load caused the misses.

System resources stayed above the stop threshold: minimum `MemAvailable` was 4.28 GiB; target-process sampled `VmRSS` peak was 3.24 GiB; `VmHWM` was 3.38 GiB; sampled process `VmSwap` was 0; system `pswpin`/`pswpout` deltas were zero. The profile run measured 31.294 s inside train and 35.923 s from feed start through train return. These profiled timings include a DWARF diagnostic binary and recording overhead and are not formal speed results.

## Actual branch and stage cost

This call selected `parallel_u32_flat32` with one initial block and `stable_radix16`. The generic full-key adaptive branch did not run: `initial_bounded_tiles=0`, `initial_bounded_groups=0`. This profile therefore represents the current 512 MiB flat workload; it cannot establish performance for positions above `2^32` or generic block fallback workloads.

| Measured scope | Time | Notes |
|---|---:|---|
| Feed | 4.630 s | Includes reading and vocabulary-frequency preparation |
| Initialize | 7.028 s | `initial_count_ms` 4.271 s; radix sort 1.437 s; posting install 2.048 s; route 1.357 s are nested/overlapping probes, not additive |
| Merge | 19.442 s | Fused prepare/delta 8.945 s; owner commit 8.879 s; rewrite 0.867 s; route 0.021 s; select 0.269 s. `delta_ms` includes the fused prepare interval, so do not add it again |
| Train | 31.294 s | Initialize and merge do not cover all trainer setup/finalization; the remaining 4.824 s is not uniquely attributed by this probe |

The current profiled train duration is 31.294 s versus 28.569 s in the prior H diagnostic, an observed increase of 2.725 s (9.5%). This is a diagnostic comparison, not a formal speed verdict: profile overhead, code version, and source instrumentation prevent attributing the difference to candidate performance.

The merge probes show where this call spent measured stage time. The main PERF self-IP shares were `train_in_pool` owner closure 23.35% cycles / 19.20% cache-miss period; `fused_batch::prepare_with_mode` closure 22.13% / 17.20%; stable radix sort 5.24% / 9.85%; `HashMap<u64, Entry>::insert` 4.75% / 2.77%; `cfree` 3.37% / 1.59%; and `Prepared::apply` 2.65% / 7.03%. Full top-IP rows are in [event-summary.json](results/current-perf-audit/event-summary.json).

A bounded batch of addresses from this exact binary resolved the owner closure's hot IPs through DWARF. Samples land in output-group iteration (`parallel.rs:1215`), `ledger.entries.get_mut` and hashbrown's `find_inner` probe (`parallel.rs:1223`), and the frequency-floor check (`parallel.rs:1228`). The prepare closure scans candidate postings, validates token neighbors, reads weights, and builds exact remove/birth deltas (`fused_batch.rs:178–258`). `Prepared::apply` performs the separate corpus writes, including the replacement store at `fused_batch.rs:300`. This distinguishes the hash lookup, delta construction, and corpus rewrite paths without assigning all samples in a broad closure to one operation. The source/address checks are saved in [top-ip-addresses.txt](results/current-perf-audit/top-ip-addresses.txt).

The complete operation still visits 125,409,599 merge postings; this run recorded zero stale posting visits. The observed prepare and owner-commit paths carry exact boundary checks, neighbor-frequency updates, and owner ledger updates for those positions. The profile does not identify a large repeated scan or redundant update that can be removed while preserving the BPE result.

## Comparison with the prior complete PERF

The latest saved full 512 MiB DWARF PERF before this audit is the H diagnostic at source `00216d914186e42458d45e72276b13c700749c6a`, not a J or current-candidate profile. It used the same input and trainer parameters, optimized release with DWARF 2, the same `cycles:u` and `cache-misses:u` events at 99 Hz, and DWARF call stacks. Its binary SHA-256 was `b712704589be5228a7ba1295d39f989993ac1d987b8f96197f9504c30b701d34` (Build ID `e80ac816f0da8a7b8019205f97b665ba08d13045`). See [the H profile](results/optimization-h-debug-profile.md) and its original run metadata.

| PERF sample category | Prior H | Current candidate |
|---|---:|---:|
| Cycles / cache-miss samples | 8,084 / 7,813 | 8,719 / 8,447 |
| Top-IP unresolved period | 9.55% / 9.63% | 9.66% / 12.96% |
| `Output::birth` self-IP | 5.35% cycles / 1.69% misses | No longer a top symbol; current task-local aggregate birth is 2.47% / 2.12% |
| `Output::remove` self-IP | 5.46% / 3.25% | 1.53% / 0.80% |
| Current owner commit closure | Prior closure's direct IPs totaled 18.94% cycles | 23.35% / 19.20% cycles/cache-miss period |
| Current prepare closure | H inclusive prepare callchain was 34.08% / 24.30% | 22.13% / 17.20% direct self-IP; 32.36% inclusive cycle share |
| Hash-table growth | H `RawTable` top-IP aggregate 3.40% / 7.41%, mostly `reserve_rehash` | Current resolved `reserve_rehash` IPs together are 2.55% cycles / 7.49% cache-miss period across the sampled events |

These rows do not represent interchangeable source ranges: H's `Output::birth/remove` were per-occurrence paths; the current source groups neighbor changes in task-local `Scratch` and commits route maps by owner. H's inclusive prepare figure also cannot be compared numerically with the current direct self-IP figure. The current sample confirms that the old standalone birth/remove hotspot was structurally changed and redistributed across prepare, aggregation, and owner commit. It does not let us subtract the percentages as a precise optimization gain. The current `reserve_rehash` aggregate covers all resolved symbols in the full PERF sample stream, not just the top-25 table symbols; its main entries are `BirthGroup` 1.54% / 4.62%, `(u64, u32)` 0.70% / 2.33%, `Entry` 0.24% / 0.45%, and `CompactString` 0.07% / 0.09% cycles / cache-miss period. These sampled event shares are not a count of growths or a standalone estimate of time recoverable by reserving capacity. The exact full-symbol aggregation is saved in [event-summary.json](results/current-perf-audit/event-summary.json).

There is separate end-to-end evidence for the implemented J grouping change: the original-API H→J 512 MiB screening call reported prepare lower by 8.1%, train lower by 11.4%, and elapsed lower by 9.6%; the unmodified initialization also happened to be 2.011 s faster in that single comparison. That result supports retaining the already selected J change, while leaving its exact full-train attribution limited. The new profile does not establish an additional speed gain beyond those already measured in formal benchmarks. Historical and current weighted periods are not elapsed time and should not be compared as absolute work totals.

## Remaining work and stop decision

The largest current costs are the exact prepare loop and owner-side aggregate/ledger update for real posting occurrences. Both have already had substantial structural work: grouped task-local neighbor deltas, fewer owner route lookups, direct flat initialization into final streams, and lower-scratch stable sorting. The current profile shows those operations still consume time; it does not expose a single high-cost redundant operation with a bounded, low-complexity removal. The observable table rehash IPs are spread across structures; only the `Entry`-table growth symbol is below half a percent of cycles, while the `BirthGroup` event share is not proof of a precise cache-latency cause. Reserving more capacity would add memory and lacks evidence of a large full-call opportunity. `cfree` alone does not identify which allocations are avoidable or quantify wall time.

For the selected 512 MiB flat workload, I recommend stopping further in-memory optimization work for now. This means no clear high-gain candidate is supported by the current evidence, not that the implementation is globally optimal. The generic `>2^32` full-key adaptive path remains outside this audit's workload and needs its own representative measurement before making a performance claim.

## Artifacts and reproduction

- [Build script](results/current-perf-audit/build_current_perf.py) creates the optimized DWARF diagnostic binary in a unique build root and target directory; it records hashes for the modified source and runner inputs and saves the [instrumentation patch](results/current-perf-audit/instrumentation.patch).
- [Run script](results/current-perf-audit/run_profiled_call.py) performs exactly one guarded run, writes the provenance and complete result, and stops at `MemAvailable <= 1 GiB`.
- [Summary script](results/current-perf-audit/summarize_perf.py) rebuilds the weighted top-IP summary from the ignored raw perf file without expanding every callchain.
- [Raw perf header](results/current-perf-audit/perf-header.txt), [record output](results/current-perf-audit/perf-record.stderr), [training result](results/current-perf-audit/current.jsonl), and [DWARF top-IP mapping](results/current-perf-audit/top-ip-addresses.txt) preserve the diagnostic details.
- Raw perf data: `.build/native-j-current-perf.perf`; SHA-256 is recorded in `current.provenance.json`. Its large binary data remains outside the results directory and Git.

To reproduce in a fresh audit checkout, place the same hashed corpus at the recorded path, run `build_current_perf.py`, run `run_profiled_call.py`, then run `summarize_perf.py`. The scripts refuse to overwrite previous outputs.
