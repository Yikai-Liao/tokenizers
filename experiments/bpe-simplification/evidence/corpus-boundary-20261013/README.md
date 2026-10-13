# Corpus merge boundary experiment

Baseline: `826479034460c18530ebe2522aaa1585133f97fe`.
Final implementation: `309f4f92` on `refactor/bpe-corpus-boundary-20261013`.
[final-manifest.json](final-manifest.json) freezes the baseline and initial candidate;
[aligned-manifest.json](aligned-manifest.json) freezes the final local-iterator borrow.

## Implementation and correctness

`merge_next` selects and resolves the ordered priority prefix, moves its Positions
payloads into `Corpus::merge`, then commits index changes and vocabulary history.
Private preparation joins all readers before disjoint endpoint writes, and joins
all writers before returning events. Fresh/Reuse mode and the optional birth-span
limit are bound by the plan for the attempt.

AA uses the original historical length for its task budget, releases that payload
after acceptance, and shares global accepted-start membership across tasks. AB
uses one pair lookup and restart-aligned chunks. Reuse retains historical word
domains, candidate seeks and virtual neighbors. Matches use standard ranges;
word-count conversions use `From`; identity and aggregation records name their roles.
Task splitting and producer/commit order are preserved.

Both default and no-default-feature ordinary-BPE test selections finish with
**35 passed, 1 known baseline failure**. `tokenizer_roundtrip` fails for Chinese
`测`, encoding ID 3 instead of the expected ID; the unchanged baseline reproduces
this failure, with expected IDs varying in baseline repeats. Current logs are
[codegen-tests-default.log](codegen-tests-default.log) and
[codegen-tests-no-default.log](codegen-tests-no-default.log); baseline logs and
[baseline-roundtrip-repeat.log](baseline-roundtrip-repeat.log) retain the control.
[aligned-clippy.log](aligned-clippy.log) finishes successfully with two existing
redundant-field warnings.

`rejected_start_and_aa_tasks` checks original-length budgeting and global AA
membership at 1/6 workers. Preparation/post-write failure tests first commit AB,
then verify retained history and trace while withholding the failing round's births.
The preparation failure has independent valid CD and overflowing EF tasks; endpoints
stay unchanged. Index failure occurs after writes, with no rollback promise.

[verify_aligned.py](verify_aligned.py) validates **30 complete model and selection/
commit trace comparisons**: 14 original cases at 1/6 workers plus the larger AB
case at 1/6. Coverage proves AA, compatible multi-rule AB, reserved fresh ID,
alias restart and configured length reuse paths. Twelve stage models also match.
[aligned-validation.json](aligned-validation.json) retains hashes and coverage.

## Measurement boundary and inputs

Both arms share Rust 1.98.1, dependency locks, allocator, features and the standalone
runner's release profile: opt-level 3, fat LTO, one codegen unit. Affinity and cgroup
limits are in [baseline-manifest.json](baseline-manifest.json). Heavy runs execute
serially with 1/6 workers; paired order alternates.

Every primary sample is one **complete public `do_train` call in a fresh process**,
including initialization, merging and retirement. Input loading and model export
are timed separately. Process HWM is sampled before export, but includes loading.
Full token IDs and ordered merges are serialized and compared by canonical hash.
Diagnostic binaries and repeated-call runners are excluded from primary timing.

[inputs.json](inputs.json) freezes data hashes, preprocessing and configuration.
`en256` and `zh256` use full prepared sets from their 256 MiB sources, respectively
`whitespace_split` and `whitespace`, with vocabulary target 50,000. The two ByteLevel
cases use complete UTF-8 lines capped at **8 MiB** from those sources, preprocessed
with `bytelevel_regex`, vocabulary target 30,000. They are not full 256 MiB ByteLevel
measurements. Caller map seeds are `[11, 13, 17, 19]`; internal default AHashMap/Set
seeds remain randomized in fresh processes.

AA-large contains four million symbols; AB-large and reserved-large contain about
two million. Alias/length fixtures use a high-weight `baaba` restart trigger plus
two million `xy` symbols; length admission is `Some(9)`. AB-steady scales AB-large
geometry and short-word weights by ten, giving a separate 20-million-symbol case.
It is not pooled with AB-large. Initial screening inputs that missed reuse are
archived under `trace-screen` and excluded from final branch admission.

## Primary results

[primary-summary.json](primary-summary.json) is authoritative for the final binary.
It keeps the pre-borrow candidate, final candidate, acceptance series, and identical-
baseline controls separate and asserts binary/input/model hashes. Percentages below
are medians of paired changes; negative means less time. HWM and every raw percentage
remain in the JSON. Samples of the unchanged baseline/final binary from subsequent
three-arm isolations are included; experimental layout/span arms are excluded.

| Case | Pairs 1/6 | Wall change 1/6 | CPU change 1/6 |
| --- | ---: | ---: | ---: |
| en256 | 2/2 | -1.86%/-2.31% | -1.76%/-1.58% |
| zh256 | 2/2 | -0.13%/-6.45% | -0.01%/+1.11% |
| en-bytelevel | 10/10 | -3.23%/+3.73% | -1.30%/+3.64% |
| zh-bytelevel | 34/14 | +1.60%/-1.95% | +1.58%/-0.64% |
| aa-large | 6/6 | -3.80%/-2.98% | -3.77%/+4.15% |
| ab-large | 6/6 | +8.55%/-5.66% | +8.57%/-11.13% |
| ab-steady | 4/4 | -0.53%/-3.28% | -0.79%/+3.41% |
| reserved-large | 6/6 | -3.01%/-2.66% | -3.17%/+5.25% |
| alias-large | 6/6 | -4.07%/-3.41% | -4.38%/-3.95% |
| length-reuse-large | 6/6 | -3.07%/-0.91% | -3.00%/+2.32% |

The earlier 14-pair Chinese ByteLevel single-worker acceptance series had CPU
**+2.30%** (wall +2.35%). Twenty later paired samples of the same frozen binaries
bring the combined median to **+1.58%**. The user accepted the approximately 2%
difference and requested stopping further root-cause investigation. Its remaining
cause is **undetermined**; this report does not claim zero overhead or universal speedup.

English ByteLevel six-worker CPU is +3.64%, while eight identical-baseline control
pairs yield +4.13%, with broad positive/negative variation. Other positive CPU
medians remain visible: AA six-worker +4.15%, reserved six-worker +5.25%, AB-steady
six-worker +3.41%, length reuse six-worker +2.32%. Corresponding baseline controls
are recorded separately, not subtracted as a correction. AB-large single-worker
is +8.57% at about 0.3 seconds; the separately amplified three-second AB-steady
case is −0.79%. These results limit any claim of a consistent AB slowdown or speedup.
Small branch fixtures take about 3–15 ms and have large relative variation; their
raw results are retained for correctness and are not the performance decision alone.

## Removed copy and remaining diagnostic limits

The initial candidate introduces an unconditional 144-byte decoder-state copy per
AB task. Borrowing a local iterator removes it: the AB helper's stack allocation
changes from `0x4c8` to `0x3e8`; baseline is `0x408`. The source has a PERF comment.
[assembly/README.md](assembly/README.md) distinguishes this executed copy from
cold deque-growth memcpy branches. This establishes a concrete removed cost; it
does not establish the cause of every timing difference.

Chinese single-worker investigation retained both arms' perf samples and absolute
cycle differences in [zh-profile-difference.json](zh-profile-difference.json).
Those two profile pairs did not reproduce the whole positive difference. Birth
sorting increased by about 0.5% of old whole-process sampled cycles. Temporary
`repr(C)` restoration of old BirthFragment layout and one-time valid-range span
calculation did not consistently reduce complete training time. Neither is landed;
see [zh-birth-layout.md](assembly/zh-birth-layout.md) and
[zh-match-span.md](assembly/zh-match-span.md).

Aggregate phase probes in `zh-phase-profile` print once after the attempt, with
matched clock/atomic instrumentation in both arms. Their initial-count and merge
phase medians favor the candidate, but wall-clock probes can disturb code generation
and cannot replace frozen production CPU measurements. Hardware counters in
`zh-aligned-profile` show fewer instructions/branches and variable cache misses;
the zero-valued stat cycle counter is unusable. Default internal hash seeds were
identified as an uncontrolled variable; no deterministic-seed experiment was run.

## Resource and historical costs

Actual production-grain diagnostic types retain Positions 32 bytes, Candidate 48,
Priority 16, pending action 16, and resolved identity 8. AA selected records shrink
56→48 bytes and tasks 112→32; the two-million accepted-start cache keeps its capacity,
while the selected historical Positions is already dropped. AB tasks remain 112
bytes, with one lookup per batch. Prepared payload capacities and lifetimes are
recorded in stage telemetry; RSS alone cannot prove release because allocators
retain pages. Process HWM in the primary samples remains comparable.

Separate scratch probes find a common six-worker English ByteLevel cost of roughly
24,000 directory initializations and 0.65–0.67 billion filled u32 entries (about
2.6 GB of writes) in both versions. This is historical work, not an introduced
refactor cost. No Rayon grain/fold-budget optimization is included. The fold-budget
experiment was stopped on the user's request; its partial fixture directory is
not a completed acceptance group.

Formatted descriptive counts are 2,572 production / 1,691 test lines versus
baseline 2,456 / 1,574, with parity excluded and the unchanged Miri harness included.
[aligned-source-counts.json](aligned-source-counts.json) records the scope. Counts
are descriptive rather than a performance or design gate.

## Reproduction and archive boundaries

Use the frozen locks/profile to build each source arm. `prepare_evidence.py` creates
inputs and separate production-grain trace/stage copies (`cfg(test)=false`, initial
count grain `1<<24`). `run_evidence.py --arms baseline=BINARY candidate=BINARY
--cases CASES --workers 1 6 --rounds 2 --group GROUP` records each job, stdout,
stderr, hashes, metrics and compressed full model. `summarize_primary.py` checks
formal binary provenance; `verify_aligned.py` checks final trace/model equality.
Raw gzip model bytes may have normalized headers; canonical model contents and
hashes remain unchanged. Compressed perf archives can be unpacked before replaying
`perf report`/`perf script`.

`summary.json` describes the initial pre-borrow measurements. `repeated-*` uses a
different repeated-call runner and is auxiliary; its source is archived separately.
`range-*`, `fold-*`, `scratch-*`, `perf-stat`, stage/phase probes and the experimental
arms of `zh-birth-layout`/`zh-match-span` diagnose costs and are not final production
arms. Experimental source manifests preserve their differences. Current production
source and the formal single-call caller are frozen in `aligned-manifest.json`.
