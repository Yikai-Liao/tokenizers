# Ordinary BPE feed unification

This report archives the initial A/B comparison. The integrated A subsequently removes the collection-phase SCC reservation; current scaling evidence and revised-source validation are in [the follow-up report](../feed-cost-20261011/README.md).

Baseline: `52ca295163936be3cf81cb91a3a8d654c78ac6d4`. The baseline feed file is byte-identical to plan reference `a2f1c0f7`; the user confirmed that the requested comparison is against this pre-feed-change implementation. Three independent worktrees start at this commit; only ordinary BPE feed changes between production arms. Common boundary tests and measurement diagnostics are applied identically. Parity Trainer is outside scope.

## Decision

**Retain A (local aggregation)**. `count` now owns named `flush` and `accumulate` closures, startup-single/32-input fused batching, and ordinary error reduction. It removes `LocalCounts`, the generic accumulation adapter, the batching helper, and the two intermediate result collections. B (direct SCC updates) is archived as an experiment. Its two English four-worker pipeline comparisons regress by about 17% and 21%, with about 23% and 29% more feed CPU; directed repeats expose the per-occurrence shared-update cost. A accepts small observed pipeline time increases and phase-specific time/memory costs in exchange for simpler organization, under the user’s stated real-workload criterion; it is not presented as a universal speed or memory improvement.

Callbacks execute before checking an earlier ordinary error. Ordinary fold/reduce finishes input callbacks; errors do not gain an input-order guarantee. Successful folds can publish their private tails before another fold fails. The SCC table belongs only to this call, and the caller replaces trainer state only after success. Serial input retains its AHashMap and never creates an SCC table; the vacant-key length check remains, and a serial flush is a no-op. The 2048-key bound is only a parallel local distinct-key bound, not a byte or global-memory bound.

## Real-corpus measurement

Chinese and English are the existing approximately 256 MiB Wikipedia corpora: 268435162 and 268434441 bytes. `source-corpora.json` records dataset configuration/revision, shard hashes, ordered nested-prefix selection, whitespace normalization, 32–8192-byte paragraph filtering, and line protocol. `inputs.json` records hashes and actual sizes. Both files pass SHA256 and streaming strict UTF-8 validation. No fresh truncation or generated replacement is used.

Each sample streams UTF-8 lines through the public ByteLevel-regex or Whitespace processor, feeds counts, then calls `train_vocab` with vocabulary 50000, minimum frequency two and no affixes. Timing excludes post-timing count export and model serialization, and includes stream reading, preprocessing and training output construction. `subprocess_wall` also includes export/serialization and is not the reported pipeline timer.

Rust/compiler and CPU details are in `measurements/manifest.json`. Release uses opt-level 3, fat LTO and one codegen unit. One/four workers are pinned to CPU 0/0–3. `RAYON_NUM_THREADS` controls the ambient feed pool as well as recording requested training workers; every sample checks actual feed worker count. SHA256/UTF-8 validation warms the ordinary OS file cache before the matrix; there is no forced eviction, and runners do not preload the corpus. Builds/tests and other benchmark processes are excluded.

Two formal rotating-order paired blocks per scenario are the initial matrix; no fixed five-repeat requirement or excluded warmup. Paired percentages are descriptive observations on a shared laptop. Positive values mean more time or memory. Full raw samples, commands and logs are retained.

The table uses all available A pairs (initial plus targeted extra checks) and the two initial B pairs.

| Input / workers | A pairs | A feed wall | A pipeline wall | B feed wall | B pipeline wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| bytelevel_regex-w1 | 2 | -2.72% | -1.29% | -3.85% | -1.49% |
| bytelevel_regex-w4 | 5 | +8.76% | +0.32% | +16.30% | +3.57% |
| whitespace-w1 | 2 | -4.49% | -0.87% | -6.31% | -1.52% |
| whitespace-w4 | 2 | -0.26% | -0.42% | +14.16% | +1.53% |
| en-bytelevel_regex-w1 | 5 | +3.63% | +2.95% | +0.14% | +0.17% |
| en-bytelevel_regex-w4 | 2 | +1.48% | +1.64% | +23.16% | +16.68% |
| en-whitespace-w1 | 5 | +3.46% | +2.97% | -0.52% | -0.37% |
| en-whitespace-w4 | 2 | +2.43% | +1.78% | +28.76% | +20.66% |

The following CPU/memory medians describe the initial two-pair matrix (additional raw snapshots remain in `extra`): feed / pipeline CPU seconds; process HWM through feed / before validation in MiB.

| Input / workers | Baseline CPU | A CPU | B CPU | Baseline HWM | A HWM | B HWM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bytelevel_regex-w1 | 6.903 / 34.781 | 6.715 / 34.338 | 6.638 / 34.269 | 703.8 / 2283.8 | 703.7 / 2279.3 | 703.7 / 2288.8 |
| bytelevel_regex-w4 | 8.312 / 40.192 | 8.769 / 40.504 | 8.606 / 40.435 | 1094.3 / 2382.8 | 1260.3 / 2381.7 | 1201.5 / 2368.0 |
| whitespace-w1 | 5.304 / 24.516 | 5.065 / 24.293 | 4.971 / 24.144 | 507.4 / 1709.1 | 507.3 / 1708.3 | 507.3 / 1708.2 |
| whitespace-w4 | 6.161 / 30.821 | 6.193 / 30.710 | 7.037 / 31.772 | 695.6 / 1789.6 | 695.7 / 1781.5 | 863.3 / 1774.1 |
| en-bytelevel_regex-w1 | 6.391 / 7.547 | 6.608 / 7.762 | 6.398 / 7.559 | 53.3 / 157.2 | 53.3 / 166.4 | 53.3 / 158.1 |
| en-bytelevel_regex-w4 | 7.925 / 9.882 | 8.054 / 10.024 | 9.783 / 11.762 | 67.4 / 184.0 | 67.6 / 190.7 | 88.4 / 189.4 |
| en-whitespace-w1 | 5.752 / 6.747 | 5.857 / 6.844 | 5.721 / 6.720 | 53.2 / 142.1 | 53.2 / 146.3 | 53.2 / 143.8 |
| en-whitespace-w4 | 7.076 / 8.839 | 7.244 / 9.039 | 9.132 / 10.921 | 63.7 / 159.7 | 63.8 / 153.0 | 63.6 / 147.8 |

These are cumulative process high-water marks at the named boundaries, not separately instrumented exclusive stage peaks. `feed_rss_kib` and `train_rss_kib` are current RSS snapshots; sampled process RSS can include post-timing validation. Feed/pipeline MiB/s and all CPU, RSS, minor/major fault metrics are in the summary and raw records. Throughput uses source bytes, not callback-output bytes.

## Additional paired checks

Additional runs quantify the initial time differences and memory variability. They use the same frozen binaries, timing boundaries and validation. Initial and later samples are retained separately; percentages below combine same-block pairs across both sets.

| Input / workers | Pairs | A feed wall | A pipeline wall | A feed CPU | A pipeline CPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| bytelevel_regex-w4 | 5 | +8.76% | +0.32% | +10.66% | +1.77% |
| en-bytelevel_regex-w1 | 5 | +3.63% | +2.95% | +3.72% | +3.03% |
| en-whitespace-w1 | 5 | +3.46% | +2.97% | +3.46% | +2.94% |

## Time and memory tradeoff

A simplifies organization and has measurable phase costs. Across the eight scenarios, the median paired full-pipeline wall changes span -1.29% to +2.97%; four-worker pipeline changes span -0.42% to +1.78%. Small differences around one percent are not established speed improvements on this shared laptop. The English single-worker feed increase is reproducible at roughly 3.5–3.6%, and its pipeline increase is about 3%.

Chinese ByteLevel/four workers has five pairs. Feed wall changes are -0.42%, +8.76%, -0.76%, +11.74%, +11.33% (median +8.76%), while full-pipeline wall median is +0.32% and CPU median +1.77%. Process HWM through feed alternates between roughly 929 MiB and 1258–1264 MiB in both arms; A has more high readings in this sample set, giving median paired +35.35% (about +330 MiB in the three high/low pairs). Full-pipeline process HWM median changes +0.50%, because training reaches a larger peak. The cause of the high/low memory readings was not isolated, so the data does not establish a specific allocator or SCC mechanism. This stage cost is retained under the user’s stated allowance for small complete-training regressions in exchange for simpler code, and it is not hidden by the overall timer.

English ByteLevel/single-worker initial pipeline HWM was about 9–10 MiB higher for A. The additional pairs reverse direction, giving all-five-pair median -5.26%; this does not establish either memory savings or a stable increase. Feed HWM is virtually identical in those runs. Full raw peaks and paired deltas are preserved, and no universal memory improvement is claimed.

## Directed checks

The standalone micro harness imports verbatim baseline/A/B feed modules and the unchanged WordCounts module; only its main generates inputs and callbacks. It compares complete count contents after each run. Seven alternating-order repetitions use the first as warmup and report six formal pairs. These probes explain costs and do not substitute for real corpora. CPU snapshot overhead makes microsecond-scale short-input ratios noisy.

| Workload / workers | Baseline wall | A wall change | B wall change |
| --- | ---: | ---: | ---: |
| cheap-repeat / 1 | 29.2195 ms | +5.68% | +0.53% |
| cheap-repeat / 4 | 8.9499 ms | +0.55% | +353.67% |
| cheap-distinct-zh / 1 | 24.3159 ms | -7.97% | +0.64% |
| cheap-distinct-zh / 4 | 14.1345 ms | -1.82% | -19.15% |
| expensive-short / 1 | 82.1831 ms | -0.02% | -0.00% |
| expensive-short / 4 | 20.8370 ms | +0.05% | -0.19% |
| cheap-short / 1 | 0.0040 ms | +3.44% | -5.17% |
| cheap-short / 4 | 0.0105 ms | -26.30% | -23.71% |

Repeated-word input demonstrates the shared-update cost removed by local aggregation; distinct Chinese keys demonstrate the extra local-map/drain/second-hash cost B can avoid. The expensive four-input callback remains parallel thanks to startup singletons. No AtomicU64 variant was added: the ordinary direct-update experiment is sufficient to decide between these two strategies, and atomics would require a separate contention/overflow analysis.

## Correctness and validation

All 66 real-corpus processes exited successfully, produced matching canonical full-count hashes and identical complete parsed vocabulary/ordered-merges models across arms and worker counts, and had zero sampled child swap. All recorded feed/pipeline major-fault counters are zero. Large count exports were removed only after validation; their canonical hashes and record counts remain archived.

Count export runs after the timed interval without creating a second full dictionary: the streaming serde formatter reconstructs a single key, writes JSON-key/tab/count records, and external `LC_ALL=C sort -S128M` produces a canonical SHA256. The formatter test covers UTF-8, escaped quotes/backslashes/newlines/NUL and exact frequencies. Model equality parses full objects, retaining merge-array order rather than comparing JSON field order.

Common tests cover disabled parallelism, enabled/one worker, four workers; empty input/output; every input read and processed exactly once; non-fused exhaustion; startup W and W+31/32/33/64 boundaries; 2047/2048/2049 keys and multiple flushes; repeated words around flush boundaries; first/middle/last/multiple errors with all callbacks; preexisting trainer-state preservation; controlled two-worker publication/error interleaving; and small public training/model equality. Baseline/A/B each pass the same 36 no-default library tests with the known encoder-roundtrip assertion filtered.

Final default and no-default library checks each pass 36 tests with the known roundtrip assertion filtered. Unfiltered default records 36 passes and one failure (`测`: actual ID3, expected6); the same selected assertion is reproduced on the frozen baseline. All-target Clippy with warnings denied, changed-file formatting and Git whitespace checks pass. Exact commands, exits and final-source equality with the measured A arm are recorded in `validation.json`. No Parity implementation, dependencies or toolchain versions are changed. Benchmark diagnostics are archived as a common runner patch, not added to the production runner.

## Source size and reproduction

Feed physical lines: 146 → 93 (A), or 75 (B). Nonblank, noncomment production lines: 113 → 81 (A), or 67 (B). Formatting/merged comments also affect physical lines; these are descriptive counts, not a complexity target.

`final-production.patch` reproduces the integrated production/tests/docs change without measurement diagnostics. Apply one complete archived arm patch to its own worktree at the baseline commit; patches already include common tests and diagnostics. `source/<arm>` and `builds.json` retain exact changed source/dependency snapshots and hashes. Build the runner with `CARGO_TARGET_DIR=<separate target> cargo build --offline --release --locked --manifest-path experiments/bpe-simplification/runner/Cargo.toml`; freeze each binary before building the next arm. Adjust absolute binary/source paths in `builds.json` and `inputs.json`, then run `python measure.py`. Defaults run baseline/local/direct serially for two blocks and all eight scenarios. Additional comparisons use `BPE_FEED_ARMS`, `BPE_FEED_BLOCKS`, `BPE_FEED_CASE` (comma-separated scenarios) and a fresh `BPE_FEED_OUTPUT`.

The micro Cargo manifest references the baseline tk-encode checkout by absolute path; change it to the restored baseline path before locked building. Its three feed files and WordCounts are actual source snapshots. Large corpora, binaries and model/count payloads stay outside Git. Raw jobs/stdout/stderr are consolidated in sample-artifacts JSON files, with exit states and raw metrics in runs JSON.
