# B1: Reuse corpus measurements as word starts

Retain the isolated B1 change: the existing parallel symbol measurement fills one
usize vector; the existing serial checked prefix loop reads each count and then
replaces that slot with the word start. There is no second per-word directory
allocation. The user accepts a small training-time regression for this reduction
in temporary storage. No general speedup or whole-process peak reduction is claimed.

## Isolated comparison on 2f4803a9

Both arms contain the identical latest-at-start Feed implementation, exponential
initial_ranges search, Shard ownership and standard Positions ranges. Production
changes are confined to CorpusPlan::build. Empty/filtered-word separators, checked
length arithmetic, weighted edge mass, allocation limits and zero-merge checks
keep their previous order. No parallel stage becomes serial.

Four pinned 256 MiB Chinese/English Wikipedia cases use ByteLevel-regex and
Whitespace, four workers bound to CPUs 0–3, vocabulary size 50,000 and minimum
frequency 2. Each case excludes one warmup pair and retains two alternating
formal pairs. Input loading and output serialization are outside the training
timer; cumulative process HWM includes loading. Binaries are uninstrumented,
opt-level 3, fat LTO and one codegen unit. No builds or other experiments run
during measurements. Source, dependency, input and binary hashes are preserved.

| Case | Training wall | Training CPU | Process HWM | Layout bytes saved (MiB) |
| --- | ---: | ---: | ---: | ---: |
| zh-whitespace-256MiB | +2.52% | +2.29% | +0.00% | 52.09 |
| zh-bytelevel_regex-256MiB | +1.06% | +0.65% | +1.60% | 52.34 |
| en-whitespace-256MiB | -0.50% | -2.16% | +2.44% | 5.53 |
| en-bytelevel_regex-256MiB | +1.56% | +3.36% | -9.74% | 6.38 |

Changes are medians of paired percentages, not ratios of arm medians. All 24
complete models (including warmups) agree on full vocabulary IDs and ordered
merges; sampled child swap is zero. Chinese Whitespace wall is +1.85%/+3.19% in
the two formal pairs. Other cases have mixed directions; two pairs do not
establish stable speed gains or a cause for the whole-training time changes.
The change removes an allocation but is retained for temporary storage, not speed.

## Layout checkpoints

Separate diagnostic copies sample actual vector capacities and /proc RSS/HWM
immediately before and after the real prefix loop. They run zero-merge jobs;
normal layout validation and initial pair counting still execute. These binaries
are never used for production timing conclusions. For W words the measured
capacity reduction is exactly 8W bytes on this 64-bit host. Both Chinese cases
save about 52 MiB, English Whitespace 5.53 MiB and English ByteLevel 6.38 MiB.
The baseline's RSS grows by the touched directory during the prefix loop; the
candidate has no such growth. Earlier input-loading peaks mask this difference
in the layout HWM. Whole-training HWM varies separately and has no universal
reduction. All 24 diagnostic zero-merge models agree and have zero sampled swap.

## Correctness and Feed integration

Default and no-default library suites each pass 39 tests with the known
feed_training_tokenizer_json_and_encoding_roundtrip assertion filtered.
All-target no-default Clippy with denied warnings, changed-file formatting and
git diff whitespace checks pass. The uninstrumented latest-at-start baseline's
unfiltered suite has 39 passes and that same one failure.

The suite exercises exact input counts, callback/error contracts, first
exhaustion, disabled/one/four-worker Feed, actual cache flush followed by error
without replacing previous counts, and fed versus fixed-map training equality.
Large fixed-cache model comparisons above do not constitute separate public-Feed
pipeline comparisons. Feed, Shard, Positions and merge implementation sources
are unchanged by B1. The earlier Feed experiment's resident-coordinate panic
remains unattributed; these successes do not establish its cause or repair it.

One attempted baseline test run accidentally shared a diagnostic source checkout
with probe construction. Probe stderr broke strict JSON progress parsing, so that
run has an extra progress-test failure. It is retained in
[diagnostic-contaminated-tests.log](diagnostic-contaminated-tests.log), excluded
from validation, and followed by the restored, hash-checked uninstrumented
[baseline-unfiltered-tests.log](baseline-unfiltered-tests.log). It is not a new
baseline correctness failure.

## Latest integration

A1–A3 and cutoff-tie validation arrived upstream as b4d2baba during these runs.
B1 was applied on that latest commit without changing the new initialization,
merge_next, Shard, Feed, Positions or dependency sources. Both final arms contain
those same upstream updates; their sole production difference is B1. The original
isolated results above retain their original baseline.

The final uninstrumented integration comparison repeats the four cases with the
same inputs, profile, workers/affinity, one excluded warmup and two formal pairs.

| Case | Training wall | Training CPU | Process HWM |
| --- | ---: | ---: | ---: |
| zh-whitespace-256MiB | -3.40% | -3.73% | -0.44% |
| zh-bytelevel_regex-256MiB | +0.51% | +0.52% | -0.48% |
| en-whitespace-256MiB | -0.77% | -0.42% | -0.13% |
| en-bytelevel_regex-256MiB | -1.20% | -2.40% | -10.12% |

All 24 final models agree, and their sampled child swap is zero. Four complete
model comparisons also confirm equality between the original and latest upstream
baselines. Final default/no-default suites each pass 42 tests with the same one
known encoder assertion filtered. The b4d2baba unfiltered baseline has 42 passes
and that same failure. Clippy, scoped formatting and whitespace checks pass.

The timing directions differ between baseline contexts, and some paired results
within a context also differ. These limited samples do not establish a universal
speedup or isolate a cause for whole-training time differences. The user accepts
small timing differences and authorized pushing B1; no further diagnostics or
repeat matrices were added after the final integrated comparison.

[integrated/builds.json](integrated/builds.json) freezes both final sources and
binaries; [integrated/training](integrated/training) retains the manifest, raw
samples and pair summary. Reproduce it by applying the same nine-line Corpus
patch over b4d2baba and using the final frozen binaries with the same
measure_training.py arguments. Those frozen benchmark source and binary hashes are verified after
measurements and before committing. Diagnostic code is absent from both final
arms. [validation.json](validation.json) distinguishes both baselines and the
known failure from the excluded diagnostic-contamination attempt.

## Push-time upstream sync

After the final performance matrix, upstream added 25a24372's evidence and
303597a2's direct merge-identity resolution / final-result return. B1 was applied
on 303597a2 without conflicts, preserving both changes. Corpus remains byte-for-byte
identical to the measured B1 source; every other BPE Rust source matches that
upstream commit. Default/no-default suites again pass 42 tests with the same known
encoder failure filtered; all-target Clippy and scoped formatting pass. See
[push-validation/validation.json](push-validation/validation.json) and its logs.

No new performance matrix was run for this last upstream sync after the user
approved pushing. The measured whole-training binaries remain tied to 2f4803a9
and b4d2baba; their numbers are not relabeled as timings of 303597a2. The fixed
B1 layout delta remains one fewer W-element usize allocation. This is the sole
production B1 change in the pushed commit; the independently updated upstream
merge/vocabulary implementation is not part of B1's performance delta.

## Reproduction and records

- [plan.md](plan.md) preserves the supplied scope and validation plan.
- [builds.json](builds.json), source snapshots and build logs freeze the isolated
  production binaries; [inputs.json](inputs.json) and [cases.json](cases.json)
  identify the four source corpora and prepared fixed-seed inputs.
- [training](training) contains the manifest, raw runs, pair summary and per-job
  stdout/stderr. Full models remain outside git; their hashes and equality checks
  are recorded. measure_training.py reproduces this alternating comparison.
- [layout-summary.json](layout-summary.json), layout-probe-runs.json and
  layout-samples preserve diagnostic checkpoints. build_probes.py builds copies
  using source snapshots; measure_probes.py runs them; probe-builds.json records
  diagnostic hashes. report.py archives raw records and computes layout summaries.

Build both frozen source arms over 2f4803a9 with the same locked release runner,
freeze their binaries before further builds, run prepare_inputs.py as needed,
then measure_training.py with --cases cases.json --rounds 3 --workers 4.
Diagnostic probe source and production source must use separate checkouts during
correctness tests. No corpora, full models or compiled binaries are committed.
