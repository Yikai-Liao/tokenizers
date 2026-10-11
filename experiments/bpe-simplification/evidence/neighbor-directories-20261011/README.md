# Neighbor directories and vocabulary insertion, 2026-10-11

Retain two independent interface/state simplifications against the actual baseline
`303597a24e013c644209df71b6d82fd9a649c8a2`. The preceding MergeToken and
AttemptResult removal is already in that baseline. This round leaves Corpus,
Feed, Index and Positions byte-identical in the measured candidates; Corpus B1
was excluded at the user's request. Parity Trainer remains outside scope. After
measurement, remote 8f078a8c already contained the independently completed Corpus
B1 and its evidence. That existing work is preserved in the final integration.
The 32 timings below precede integration and do not mix B1 into either candidate.

Directories keeps its two direct-ID Vec<u32> arrays and existing domain resizing.
Each cached index is checked against the current task's group array and its
existing left/right neighbor key. An index outside that array or naming another
neighbor is overwritten when the new group is created. Matching keys reuse the
current group. No touched Vec, creation ledger or reset traversal remains; failed
preparation needs no cleanup. Group order, left/right separation, event buckets,
weights and publication are preserved. The hot path adds a current-group key
check, so fewer writes/cleanup do not by themselves imply lower runtime.

Vocabulary's shared insertion helper uses IndexSet::insert_full and returns the
checked ID plus whether the identity already existed. Constructed merge text
therefore uses one container lookup/insertion operation. Initialization retains
the borrowed get_index_of before constructing CompactString. indexmap in the
locked build is 2.14.2; its local implementation returns the existing index and
does not change existing-key insertion order, matching
[the referenced 2.14.0 API](https://docs.rs/indexmap/2.14.0/indexmap/set/struct.IndexSet.html#method.insert_full).
No dependency changes. u32 and separator bounds are checked on the returned index
after insertion. A failing new insertion can briefly change the local vocabulary;
the whole attempt is discarded and no invalid ID or result escapes. Pre-insertion
rollback semantics are not promised.

The production portions before the embedded test modules fall by seven physical
lines, or six nonblank/noncomment lines. The benefit is removal of cleanup state
and duplicate container lookup, rather than a large line reduction. A directed
regression adds test code.

## Native correctness

Final default checks record **43 passes and the known encoder-roundtrip failure**;
the actual baseline had 42 passes and that same assertion for 测. Expected IDs vary
with the existing input-map traversal, so logs retain the literal failure. The
assertion is preserved. No-default records 43 passes with that one test filtered.
All-target Clippy with warnings denied, changed-file formatting and whitespace
checks pass. This is not a claim that the unfiltered suite is entirely green.

The new fixture abandons neighbor aggregation after a weighted overflow, then
reuses both directories without reset. It checks an old index outside the empty
current groups, old indices naming different keys, valid hits, and the complete
ordered event pairs/weights/buckets/position lists. Existing native comparisons
still verify every certified pair/count/replacement, full vocabulary IDs and
ordered merges at 1/4/8 workers, including AA, alias reuse, reserved identities,
actual restart and preparation/owner errors. Numeric error cases and transactional
public model replacement remain covered.

## Independent real-corpus comparison

Four isolated arms freeze exact baseline, directories-only, insert-only and
combined sources. Sources outside the selected file(s), runner code, dependency
manifests and locks are SHA256-identical. Release builds use opt-level 3, fat LTO,
one codegen unit and the same compiler/flags. Each binary is copied and hashed
before building the next arm. Production hashers and phase code are unchanged;
the runner's caller map uses its existing fixed [11,13,17,19] seeds.

The four prepared word maps come from the existing Chinese/English 256 MiB
Wikipedia inputs with ByteLevel regex or Whitespace. Their exact hashes, bytes,
source hashes and dataset revision are in inputs.json. Each fresh process loads
the same map before the public do_train wall/process-CPU timer; model serialization
and verification follow the timing/peak boundary. Four workers are pinned to CPUs
0–3. The first block uses baseline/directories/insert/combined, the second reverses
that order. No warmup is silently discarded; input verification warms ordinary OS
cache. No builds, tests or other benchmark arms overlap measurements.

All **32 samples** match their complete parsed vocabulary IDs and ordered merges,
with zero sampled child swap and zero training major faults. The table gives the
median of two paired percentage changes relative to baseline. Process HWM includes
input loading and is not an isolated directory allocation measurement.

| Input | Directories-only train wall | Insert-only train wall | Combined train wall | Combined train CPU | HWM baseline → combined (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chinese ByteLevel | +0.35% | -0.75% | -0.06% | +0.03% | 2364.8 → 2376.8 |
| Chinese Whitespace | +0.93% | +1.16% | +0.70% | +0.24% | 1804.8 → 1803.9 |
| English ByteLevel | +0.02% | -0.54% | -0.70% | -0.66% | 185.0 → 185.2 |
| English Whitespace | +0.75% | +0.15% | -1.26% | -0.93% | 158.5 → 161.5 |

These small paired observations screen for substantial regression; they do not
establish stable speedup or uniform process-memory reduction. The removed touched
allocation is a structural saving, while process peaks include much larger maps,
corpus/index allocations and default-hasher variability. These no-affix real
workloads primarily exercise fresh preparation; native tests cover reuse behavior,
but this matrix does not establish real-corpus reuse-path performance. Feed is not
timed in this do_train comparison.

## Reproduction and evidence

prepare.py freezes the four arms from the baseline and the candidate files; the
independent patches and baseline/combined source trees reconstruct every arm.
Restore the ignored tk-train and runner Cargo.lock files from the snapshots.
Adjust local paths, run build.py to build/freeze each arm sequentially, then run
measure-core.py. Native check-final.py preserves the known baseline failure as an
explicit expectation. checks.json records commands/statuses; checks/ retains logs
for native checks and every release build, including the runner's unused diagnostic
RSS-helper warning (the production Clippy check passes).

builds.json and core/manifest.json retain source/binary hashes, flags, compiler,
CPU affinity, ordering and boundaries. core/runs.json and sample-artifacts.json
retain every command/job/stdout/stderr and complete-model hashes; summary.json
contains the comparison. measured-source-hashes.json records the measured
combined source. integration/ records the merge and repeated native validation;
its hash comparison confirms that the only Rust difference from the measured
combined arm is the exact remote Corpus source. Directory and Vocabulary code
match their frozen files. No additional timing claim is made for the integration. Corpora, executables and complete output
models are not committed. The previously reported isolated Feed coordinate panic
and existing encoder failure remain separate; this work claims to fix neither.
