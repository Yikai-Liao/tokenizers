# Current BPE organization

The implementation is flattened into `tokenizers/tk-train/src/trainers/bpe`.
The public trainer and private `train` coordinator share `mod.rs`; alphabet
selection belongs to `vocabulary.rs`. Components retain their storage and phase
contracts. See [the BPE guide](../../tokenizers/tk-train/src/trainers/bpe/README.md)
and [the flattening report](FLATTEN.md) for validation and scoped diffs against
HF main and the starting revision. Counts are descriptive; hard line limits
have been retired. Earlier counts and limits below describe historical snapshots.

## Sparse neighbor libraries, 2026-10-10

This experiment replaces only the left/right neighbor grouping in `merge.rs`,
using the accepted owned-position/ThreadLocal implementation at `1396e736` as
control. All candidates retain Rayon `map_init` scratch reuse, partial/complete
birth contracts, and first-seen left-before-right event order. The earlier
IndexMap candidate allocated a fresh map per task; its time result combines hashing
and lost reuse. A reused IndexMap is therefore included as a diagnostic control.

### API and maintenance cost

| Library | Exact version | Net production lines | Extra normal/build packages | Adaptation and maturity |
| --- | --- | --- | --- | --- |
| sparsley | 0.1.0 | −8 | 2 | `entry` and allocation-retaining `drain` fit directly; first release 2026-10-04 |
| xsparseset | 0.2.5 | +2 | 1 | Vec storage requires usize keys; remove-last and reverse restore order; latest release/update 2022-10-10 |
| bevy_ecs | 0.20.0 | −1 | 35 | `get_or_insert_with`; remove-last and reverse; actively maintained Bevy project, defaults disabled and only std enabled |
| cranelift-entity | 0.136.2 | +23 | 4 | Key type and value-key adapter; pop and reverse; actively maintained Wasmtime/Cranelift project |
| IndexMap diagnostic | existing 2.14.2 | -8 | 0 | Reused hash maps and ordered drain; previous fresh-map result is not a pure hashing comparison |

Line changes use the same ordinary production scope and exclude blank/comment/test
lines after rustfmt (control 2691). Extra packages are the exact added normal/build
nodes on the Linux runner's selected feature graph, including the library itself;
optional and inactive target-specific lock entries are excluded. Bevy adds 56 lock
entries but only 35 normal/build nodes under these features. Cranelift uses its
entity container crate, without the code-generation backend. Current declared
MSRVs are 1.92 / unspecified / 1.95 / 1.96 respectively. These observations do not
make an existing project-wide MSRV promise.

All four remove the handwritten sparse-directory/touched-slot invariant. The BPE
side/weight/publication rules remain application code. Some API adaptations restore
complexity: Cranelift values must carry their key, and three libraries lack an
owning ordered drain. Our remove-last/pop adapters reverse each published side
independently; left still precedes right. No individual removal occurs while groups
are being accumulated. Reuse reset also clears any payload left by a failed task.
`sparsley` has the closest API fit, but its short release history provides little
production evidence. Project history and maintenance activity are evidence for
maturity, rather than a proof of correctness. Versions and registry/repository
snapshots are recorded in the evidence directory. Primary sources:
[sparsley](https://github.com/LilDojd/sparsley),
[xsparseset](https://github.com/xstater/xsparseset),
[Bevy](https://github.com/bevyengine/bevy), and
[Cranelift](https://github.com/bytecodealliance/wasmtime/tree/main/cranelift/entity).

### Memory mechanism

The allocation probe imports the actual adapter headers and actual Builder/Codec
source. It counts requested live allocator bytes, after warming ahash's global seed,
with empty position payloads and two sides. At a key domain of 1,000,000 and 24
live groups per side, current / sparsley / xsparseset / Bevy / Cranelift retained
storage after publication is **8,001,088 / 8,006,882 / 16,005,184 / 16,004,928 /
8,005,184 bytes**. The latter two 16 MB maps use machine-word sparse slots on
this 64-bit host, versus u32 slots in the other three. This measures a controlled
map shape, not the real workload's token-domain or group-size distribution.

Reusing dense values has a second cost: their allocations coexist with the returned
change vector. With domain 50,000 and 10,000 groups per side, live requested bytes
at publication are **3,283,648 / 5,249,826 / 5,780,800 / 5,649,728 / 5,380,800**
in the same order. After returned changes are dropped, reusable maps retain more
capacity than the current directories. All probe allocations release when the
owning scratch and Codec are dropped. IndexMap avoids a domain-sized directory,
but pays hashing costs. Full-training HWM below establishes whether these structural
costs materially affect the real process.

### Training measurements and selection

| Candidate | ByteLevel wall / CPU / HWM | Whitespace wall / CPU / HWM |
| --- | --- | --- |
| sparsley | -0.15% / -1.64% / -0.04% | +8.14% / +6.03% / -1.34% |
| xsparseset | +4.86% / +2.87% / +0.25% | +8.36% / +10.68% / -1.44% |
| bevy | +6.28% / +4.37% / +0.06% | +12.12% / +13.71% / -2.56% |
| cranelift | -0.32% / -1.00% / -0.05% | +6.53% / +6.12% / -1.41% |
| indexmap | +3.69% / +2.38% / -0.21% | +3.21% / +1.20% / -1.20% |

Screening uses one process per arm per Chinese 256 MiB input, four workers pinned
to CPUs 0–3, vocabulary 50,000, minimum frequency two and no affixes. The public
`do_train` timer excludes input loading and serialization. HWM includes loaded
input and retained counts. Every process checks the entire model against the
control and records child swap and concurrent-build/benchmark detection. These
shared-VM single observations screen candidates; small differences need repetition.

The final confirmation compares the control, a reduced Cranelift adapter and a
reused IndexMap adapter in two measured rounds per input, ABC then CBA. No round
is excluded as warmup. Earlier screening had already exercised the inputs, but
there is no dedicated warmup guarantee. The final adapters contain +12 / −17
production lines respectively, relative to the control.

| Final adapter | ByteLevel wall / CPU / HWM | Whitespace wall / CPU / HWM |
| --- | --- | --- |
| Cranelift | +8.05% / +7.27% / −0.04% | +7.98% / +9.19% / +0.60% |
| Reused IndexMap | +2.69% / +3.10% / −0.04% | +5.55% / +4.90% / −1.28% |

All 24 timed processes across screening and confirmation publish equal full
models and report zero child swap, with no concurrent build or benchmark.
Cranelift ByteLevel CPU differs substantially between its two final observations
(77.78 and 68.28 seconds), so the median is descriptive rather than a stable
estimate of an 8% penalty. Its Whitespace observations both cost about 59.6 CPU
seconds against 54.0–55.2 for the control. Neither finalist establishes a useful
whole-training improvement.

**Decision: retain the existing direct directories.** Among the four requested
libraries, Cranelift provides the strongest combination of active maintenance,
u32 sparse storage and measured screening performance. Its key and ordered
publication adapters nevertheless increase application code, and confirmation
shows a material cost without a memory gain. Sparsley's ordered drain is a better
API fit and removes eight production lines, but its first-release maturity and
Whitespace cost do not justify adoption. xsparseset and Bevy double sparse-slot
width on this host, add little code reduction, and are slower in screening.
Reused IndexMap removes 17 lines with no added dependency; its 3–5% observed CPU
cost still outweighs that small reduction for this performance-sensitive path.
The earlier fresh-map IndexMap result should not be used to attribute its entire
10.47% CPU penalty to hashing.

Each candidate's default library suite passes 17 tests, including every generated
merge against the sequential oracle. A harness extracts byte-identical actual
neighbor adapter headers, stubs the unused training coordinator types and imports
the real position module. Native and strict-provenance Miri each pass eight adapter
cases: baseline, four libraries, reused IndexMap and the two reduced final adapters. Cases cover event order,
side buckets, partial output, complete floor/encoding, failed-task cleanup and
key reuse. This validates the exercised integration paths, not every public API
in each third-party crate. Source/lock/input/binary hashes, full-model comparisons,
raw runs, allocation CSV and primary-source metadata accompany the report.

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
| Earlier fresh-map neighbor IndexMap (see reused-map experiment above) | wall +7.64%, CPU +10.47% | Control timing varied | Keep direct directories |
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
