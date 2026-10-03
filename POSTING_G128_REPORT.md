# G128 posting implementation and final selection

Retain the D1 posting implementation at measured source commit `1c5ed8e47b569743cbc765df9a9875556b868211`, with restart length **128**. The final two local optimization attempts did not produce a clear complete-training improvement, so they are excluded from the delivered implementation. The user authorized stopping when further direct changes stopped yielding useful gains.

The current target is complete training time within 10% of Flat, judged using means/medians and observed variation on the shared VPS, with each paired process-memory peak within 5%. **A stable 10% time limit has not been demonstrated.** The retained implementation meets the memory limit in the latest measured comparisons described below.

## What the implementation does

`BlockPosting` stores periodic full-U64 seeds and unsigned LEB128 gaps. Restart offsets, lengths, and allocation capacities use machine-sized words; gaps are not narrowed to fit a measured corpus. The handle occupies 16 bytes and small postings can stay inline. Larger streams use restart directories so consumers can resume from a group boundary.

The selected constructor reads its producer once and writes the encoded stream backward into reusable worker scratch storage, then copies the initialized suffix into the final allocation. It avoids per-record and whole-stream byte reversals. Overflow or failed scratch reservation falls back to the exact two-pass constructor. Appends retain the previously validated implementation.

The fused consumer decodes interleaved batches and prefetches positions. Restart length, consumer batch length, and prefetch distance remain separate. Normal builds use restart length 128, fused decode, and prefetch. `TK_POSTING_RESTART=32|64|128|256|512`, `TK_POSTING_FUSED_DECODE`, and `TK_POSTING_PREFETCH` are compile-time ablation controls.

Production scratch coordinates use 32 low bits plus high parts. `TK_POSTING_SCRATCH_BITS=16` is an experimental setting that brings high-part transitions into small-memory tests; it does not replace the production address domain. The measured address/scratch geometry below is explicitly the 16-bit experiment.

## Latest complete-training comparisons

Both experiments reused the same immutable Flat and G128 binaries. Each case ran Flat, retained G128, candidate, then the reverse order. Settings were four workers, `none/reference/50000/2`, addr16/scratch16, restart128, fused1, prefetch1, and trace off. The pressure corpus contains 536,870,289 bytes from the legacy shard-order sampling family; the 33,554,395-byte Chinese case uses a different ranked-prefix sampling family.

The following pressure results use **each experiment's own Flat control**. Times are seconds; sample variance is in seconds squared. With two observations per arm, median equals mean and the variance estimate has only one degree of freedom.

| Experiment and arm | Run A / B | Mean = median | Sample variance | Sample SD | CV | Mean overhead / Flat |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v25 Flat | 18.909 / 19.625 | 19.267 | 0.256 | 0.506 | 2.63% | — |
| v25 retained G128 | 21.619 / 20.037 | 20.828 | 1.252 | 1.119 | 5.37% | +8.10% |
| v25 append candidate | 21.097 / 20.626 | 20.861 | 0.111 | 0.333 | 1.60% | +8.27% |
| v26 Flat | 18.762 / 18.309 | 18.536 | 0.103 | 0.320 | 1.73% | — |
| v26 retained G128 | 21.244 / 20.410 | 20.827 | 0.347 | 0.589 | 2.83% | +12.36% |
| v26 store candidate | 21.355 / 25.235 | 23.295 | 7.527 | 2.743 | 11.78% | +25.68% |

The retained G128 pressure mean was almost unchanged between experiments, while the Flat mean changed. Its paired time overheads were +14.33%/+2.10% in v25 and +13.22%/+11.48% in v26. These results describe observed variation; they neither establish a stable worst-case bound nor identify the cause of every difference. They are kept as separate controlled experiments rather than combined into a favorable ranking.

The retained G128's paired VmHWM overheads were +4.09%/+3.89% on v25 pressure, +3.81%/+3.42% on v26 pressure, and +4.35%/+3.18% on v25 Chinese 32MiB. All are below 5%. VmHWM is the process high-water mark, separate from sampled RSS and exact encoded allocation. The largest candidate overhead was +4.58%.

### Why the last two candidates were discarded

The v25 candidate encoded an appended suffix in one producer pass using already initialized scratch space. Its mean pressure time was +0.16% versus retained G128; the paired directions were opposite. On Chinese 32MiB it averaged −2.21%, again with opposite paired directions. These observations do not establish a useful net gain for the additional implementation complexity. It passed 74 related tests and completed one release build and 12 native runs.

The v26 candidate reused known varint lengths and wrote exact 2-/4-byte chunks without padding. It averaged +11.85% versus retained G128 on pressure. The initial-install stage, directly affected by the writer, became slower in both rounds: 2.386→2.819 seconds and 2.342→2.930 seconds. Its longer second run coincided with higher host activity, but the sampled data cannot attribute the entire regression to contention. It passed 72 related tests and completed one release build and six native runs. Both candidates preserved the exact G128 stream/directory/restart counters and model outputs.

The initial v25 gate accidentally compared training acceptance with old G128 instead of Flat. That report was corrected from the existing 12 measurements, without rerunning native training. Old G128 is the control for the change's effect; Flat is the reference for the user's overhead target. The corrected values appear here and in the accompanying data.

## Earlier parameter search and correctness

The same retained source was previously tested at restart lengths 32, 64, 128, 256, and 512: five release builds and 108 native runs across Chinese, English, Japanese, and German at 32/128MiB, plus the separate legacy pressure corpus. All output models matched Flat. G128's equal-case geometric mean time overhead over the eight multilingual cases was +8.59%, but Chinese 32MiB and 128MiB were +21.28% and +13.31%. No fixed restart length met the 10% mean target on every case. The final choice of 128 follows the user's request after the space comparisons, not a claim of universal speed optimality.

For each restart setting, all 60 related parallel tests passed. Coverage includes independent forward encoding/restart-offset oracles, full-U64 gaps, address translation, partial appends, group boundaries, allocation growth, interrupted producers, and parallel cursors. The preceding default-128 constructor passed all 112 library tests; the final ablation commit ran the related 60-test subset for each setting. See [test evidence](benchmarks/hf-bpe/results/posting-g128/TESTS.md).

On pressure, the retained G128's exact initial stream was 738,442,780 bytes, its directory was 30,384,232 bytes, and its logical restart count was 3,934,053. These repeated exactly in the latest comparisons. Their sum describes the measured encoded allocation and is not a process-memory estimate. On Chinese 32MiB the corresponding values were 38,769,894 bytes, 3,458,304 bytes, and 556,280 groups.

All 18 latest native runs completed with matching corpus hashes, model SHA, vocabulary, and merge counts. Sampled child swap was zero; minimum MemAvailable was 3.66GiB in v25 and 3.97GiB in v26. Sampled process CPU ticks omit startup and exit-tail time and are not exact rusage or train-only CPU time.

## Reviewable evidence

The Flat binary came from `8d9c2351ff57c8b0e2dfa2fd2da6c912ca2056d5`, SHA-256 `6988dac0bce39659007819cab53355b3ecb9c48a5cd196603bdff0a58bd24644`. The retained G128 binary came from `1c5ed8e47b569743cbc765df9a9875556b868211`, SHA-256 `8e0f1d5b7c04677c941c8f7cd9d614af23a317ea35b129b8e289d49a133bf9e7`. Delivery adds this report and evidence to that measured source; it makes no further executable-code change.

- [v20 raw samples](benchmarks/hf-bpe/results/posting-g128/v20-samples.csv): the full restart-length matrix.
- [v25 raw samples](benchmarks/hf-bpe/results/posting-g128/v25-samples.csv) and [v26 raw samples](benchmarks/hf-bpe/results/posting-g128/v26-samples.csv): every latest arm, stage times, memory, and sampled CPU/host observations.
- [Computed comparisons](benchmarks/hf-bpe/results/posting-g128/comparison.json): original times, means, medians, sample variance/SD/CV, and paired ratios with explicit units.
- [Run provenance](benchmarks/hf-bpe/results/posting-g128/run-provenance.json): source/binary identities, corpus and model hashes, and successful exit/model checks for all 18 latest runs.
