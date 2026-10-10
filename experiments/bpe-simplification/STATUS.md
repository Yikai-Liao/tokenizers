# Adopted online compression and prepare optimizations

The selected combination is now applied on the simplification branch: whole-word
blocks with producer-local compression, exact selected-rule indexing and monotone
snapshot appends. The extra wave barrier is omitted. Production is **2185/2200**
and complete default BPE tests/helpers are **797/800**, counted after rustfmt,
excluding blanks and comments. The default line checker uses these approved limits.

The adopted Rust files are byte-identical to the previously validated combined
candidate: default/no-default native suites, Clippy and strict Miri passed before
selection. The transfer validation and source hashes are recorded in
`evidence/online-adoption.json`. [ONLINE_COMPRESSION.md](ONLINE_COMPRESSION.md)
contains the individual and conditional gains, all 24 pretokenizer contrasts and
the remaining Chinese Whitespace initial-index gap. These single observations do
not establish a stable whole-training speedup or universal parity with main.

## Previous 2100-line baseline review and performance follow-up

Reviewed Rust source: `4d181c51`; production **2100/2100**, complete default BPE
test/helper budget **780/800**, counted after rustfmt, excluding blanks and comments.
Six engine modules contain the full implementation. No production implementation
was moved outside the counting scope. The optional parity trainer remains unchanged.

Fresh structural reviews found no material design issue. Independent test reviews
identified missing public/codec boundaries and two budget-scope omissions; all were
fixed. The final fresh whole-crate/source/coverage audit of `4d181c51` found no new
material, actionable issue and independently reproduced 2100/780.

Default and no-default native library suites: 17 passed each; no-default doctest:
1 passed. All-target Clippy with denied warnings, fmt, budget, whitespace and strict
Miri (2 tests, default borrowing/leak checks) passed. Rust sources are unchanged
since the tested `e26115c2` revision. See REVIEW-12.md and the engine coverage map.

Uninstrumented final four-case baseline/candidate comparison completed: all eight
processes have identical complete models and child swap0. Chinese core: **21.248s**,
CPU68.621s, process HWM3086264KiB. Paired main: 16.004s, CPU52.794s, HWM2555272KiB.
This is one pair, without a warmup or statistical precision claim. A shared-host
attempt was explicitly excluded after a concurrent unrelated benchmark was detected.
See evidence/review-final-runs.json and evidence/manifest-review-final.json.

Completed follow-up: joined-stage wall/CPU/RSS attribution and consistent-prefix
384/512MiB Chinese memory scaling; see PERFORMANCE_REVIEW.md. Uninstrumented whole
HWM is 20–22% higher; diagnostic initial-stage peaks are 30.8% / 64.3% higher at
256/512MiB, with observed main bounded waves 1 / 2. All eight follow-up processes
have exact complete models and swap0. OPTIMIZATION_ROI.md records historical
net-LOC screening and the rejected owner directory. The cutoff ablation supports
keeping resident slots; see CUTOFF_ABLATION.md. Earlier delivery numbers and
exploration remain historical evidence, not the final measurement.

The final fresh evidence/completion review after correcting the historical owner
directory control found no new material, actionable finding. Source, input,
binary, diagnostic patch and measurement-script hashes were independently checked.
