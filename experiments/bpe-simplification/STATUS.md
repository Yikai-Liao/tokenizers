# Current BPE organization

The implementation is flattened into `tokenizers/tk-train/src/trainers/bpe`.
The public trainer and private `train` coordinator share `mod.rs`; alphabet
selection belongs to `vocabulary.rs`. Components retain their storage and phase
contracts. See [the BPE guide](../../tokenizers/tk-train/src/trainers/bpe/README.md)
and [the flattening report](FLATTEN.md) for validation and scoped diffs against
HF main and the starting revision. Counts are descriptive; hard line limits
have been retired. Earlier counts and limits below describe historical snapshots.

## Ownership and library simplification, 2026-10-10

The selected implementation combines owned position lists, ThreadLocal encoder
scratch, the existing restart/delta encoding, FixedBitSet alphabet membership and
`branches` cache hints. `LocalCounts`, `InitialPairCounts`, `CohortPreparation` and
explicit Empty/Unique/Shared head states encapsulate their existing state contracts.
The published model and public count-map serialization retain their existing format.

Frozen positions use Empty/One/Two/Compressed(Box<[u8]>) variants. This removes raw
allocation/deallocation, pointer tagging, manual Send/Sync and the storage lifetime
propagated through candidates, batches and births. The 64-bit descriptor grows from
16 to 24 bytes. Compressed bytes are released when their owner retires; Codec retains
only temporary encoding buffers. ThreadLocal RefCell workers replace worker-index
routing, mutex guards and poison recovery. Leases remain non-reentrant and no nested
parallel work runs while a scratch lease is held.

The ordinary eight-module production scope, including public API, feed and word
counts, contains **2691** nonblank noncomment lines after rustfmt, versus **2746** at
`7de4e068` (net **−55**). Test-only items are excluded from production; the complete
test/helper scope is 883 lines. Encapsulating local state adds some code while the
ownership change removes substantially more. These counts describe source size and
do not measure architecture quality. The line report and source snapshot are in
[evidence/libraries-20261010](evidence/libraries-20261010/validation.json).

### Candidate decisions

The following are screening observations from separate processes on the same shared
VM. They identify clear costs; one pair does not establish small performance gains.
Each screen checks the entire model and child swap. Changes are compared against
owned restart/delta storage unless a different control is named.

| Candidate | Chinese ByteLevel observation | Chinese Whitespace observation | Decision |
| --- | --- | --- | --- |
| SUx EF, inline 0–2, raw Box 3–32, EF above 32 | wall +13.69%, CPU +11.46%, peak +0.12% | wall +7.09%, CPU +8.97%, peak +4.79% | Reject significant time cost |
| Vers EF with the same hybrid cutoff | wall +51.03%, CPU +53.74%, peak +8.08% | wall +21.65%, CPU +26.50%, peak +5.65% | Reject significant time cost |
| unsigned-varint / vint64 / vu128 | wall +13.35% / +24.23% / +11.04% | mixed; the control run was unusually slow | Retain the existing codec |
| unsigned-varint with trusted unwrap_unchecked | wall +13.54%, CPU +15.30% | Not repeated after clear ByteLevel loss | Reject |
| vu128 with eight initialized padding bytes | wall +17.06%, CPU +18.75% | wall +10.23%, CPU +13.13% | Reject; removing tail copies did not recover cost |
| Neighbor IndexMap | wall +7.64%, CPU +10.47% | Control timing varied | Keep direct directories |
| WordCounts IndexMap, public feed + train | feed RSS +18.64%, pipeline wall +13.20%, peak +5.54% | feed RSS +28.47%, pipeline wall +12.45%, peak +7.15% | Keep the Map/Entries representation |
| FixedVec Builder, earlier SUx control | CPU roughly +15%; no net source reduction across the scope | Not repeated | Keep SmallVec promotion |
| ThreadLocal scratch | Later screen wall +0.48%, CPU +0.89%, peak −0.07% | wall +1.86%, CPU +4.70%, peak +1.14% | Include in the final combination check |

The API and generated-code observations concern the exact crate versions listed
above, rather than all varint interfaces. Full native and Miri checks passed for
the trusted decoder and both EF hybrids before their performance screens.

The early broad matrix stopped after 25 complete valid ByteLevel processes. No
incomplete round supplies a result. Later experiments screened changes on the owned
storage control and only investigated a plausible mechanism after a clear loss;
the final comparison runs just the baseline and the selected combination.

### Why library compression did not win

The allocation probe imports the actual three storage implementations. With a
three-position dense list, owned restart/delta storage requests 18 payload bytes in
one allocation; SUx requests 240 bytes in four allocations and Vers requests 298
bytes in seven. Each descriptor is separately 24 bytes. The probe reports requested
and glibc usable sizes, excluding prewarmed scratch. Large dense lists can favor EF:
for 65536 positions, live requested bytes are 73224 / 17696 / 33706 respectively.
This explains why short-list fixed costs merit a hybrid; the measured hybrid still
has unacceptable training time. The probe does not measure the real workload's
list-length distribution.

Portable-build perf sampling includes input loading. Vers select-next accounts for
22.42% of whole-process cycle samples, and its EF iterator another 3.44%. Source and
annotated assembly show software select/PDEP work in this build. The host supports
BMI2, but the library selects this path through compile-time target features. A
CPU-specific build might change the result; no such performance claim is made here.
SUx construction also appears in profiles. Allocation counts and sample shares are
diagnostics, not an additive explanation of all elapsed-time differences.

unsigned-varint 0.8.0 and vint64 1.0.1 expose checked decoding, without a public raw
unchecked decoder. A valid internal stream permits Result::unwrap_unchecked, but
the real unsigned-varint experiment retained its decoding loop and was still slow.
vint64 decoding uses a temporary buffer/copy in the generated code. vu128 1.1.0
already offers the low-level fixed-nine-byte array interface used by the experiment.
Its short-tail wrapper requires a copy unless nine initialized readable bytes are
available; padding eliminated that copy and still lost. There is no reason from
these results to add unsafe decoding to the selected implementation.

### Validation and final measurement

Default and no-default library suites each pass 17 tests; the no-default doctest
passes. All-target Clippy with denied warnings, rustdoc with denied warnings,
rustfmt and whitespace checks pass. Strict-provenance Miri passes both actual
position-module tests, including concurrent readers, restart boundaries, full-u64
positions, and frozen lists read after Codec is dropped. Optional parity and word
implementations retain their original source.

Final uninstrumented comparison uses two Chinese 256 MiB inputs, four workers
pinned to CPUs 0–3, 50000 vocabulary entries and minimum frequency two, without
affixes. Each case has one excluded warmup pair and two measured pairs against
`7de4e068`. All 12 processes produce identical complete models and child swap zero.
The public do_train timer excludes loading and serialization; process peak includes
input and retained counts. Absolute values below are medians; changes are medians
of the within-pair ratios.

| Chinese input | Baseline / selected wall (s) | Paired wall change | Paired CPU change | Baseline / selected peak (MiB) | Paired peak change |
| --- | --- | --- | --- | --- | --- |
| ByteLevel | 21.406 / 21.687 | +1.39% | +0.05% | 2418.9 / 2285.4 | -5.52% |
| Whitespace | 20.853 / 20.735 | -0.55% | -3.59% | 1933.5 / 1763.3 | -8.80% |

The two-arm loop ran baseline before selected in each pair. The generic rotating
round wording in the generated manifest is corrected by [final-method.json](evidence/libraries-20261010/final-method.json), which records
actual order. The shared VM, fixed order and two measured pairs limit precision.
ByteLevel's individual wall changes are +6.66% and −3.88%, so this evidence does
not support a stable speedup claim. It supports adopting the simpler ownership
representation with lower observed peak memory and no consistent large time cost.
Raw runs, build/input hashes and source/lock snapshots accompany the summary in
[evidence/libraries-20261010](evidence/libraries-20261010/final-core-summary.json).

The cache-hint dependency replaces local x86 intrinsics with the same L1 hint and
adds the library AArch64 implementation. Cross-target code generation was checked;
AArch64 training performance was not measured.

## Prior adoption of online compression and prepare optimizations

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
