# Serde, heap certification and state simplification

The final source revision is `a31898fa0320bbd2f1023143c8e453c8c471bac3`,
built from the existing release runner with opt-level 3, fat LTO and one codegen
unit. It retains Serde delegation for both word-count representations, restores
one pop/push heap-certification protocol, and applies the five requested state
and interface simplifications. No dependencies or frozen-position storage types
were changed.

## Heap decision and directed regression

The original `d4e3fa45` / `f4ed2324` comparison uses two frozen Chinese 256 MiB
inputs, ByteLevel and Whitespace, vocab 50,000, minimum frequency 2, four workers
pinned to CPUs 0–3, and no affixes. Each case has an excluded AB warmup followed
by formal BA and AB blocks. All 12 complete models agree, and measured child swap
is zero. Public do_train timing excludes input loading, validation and
serialization. Therefore this comparison measures the heap change rather than
the WordCounts Serialize change.

| Input | Paired wall change | Paired CPU change | Paired HWM change |
| --- | ---: | ---: | ---: |
| ByteLevel | −2.47% | −2.21% | +0.04% |
| Whitespace | +0.03% | +2.09% | −1.93% |

These two formal pairs per input do not establish a general performance gain.
Source/binary hashes, jobs, stdout, stderr, raw metrics and complete-model
validation are in [initial-heap](initial-heap).

The directed regression injects two historical cohorts for the same pair into
an otherwise complete initial queue: snapshot counts 10 and 5, current count 5.
Their occurrence payloads differ. With active-ID reuse, an alias raises a left
neighbor count when the heavier cohort is processed. PeekMut count repair and
pop/push select different equal-priority cohorts and produce different complete
merge traces. The retained regression asserts every expected rule and the entire
resulting vocabulary/merge model. The archived pre-fix failure establishes this
ordering difference in the injected state; it does not establish that natural
training reaches that state or supply a naturally failing corpus. The failure
was at the trace assertion, so it does not prove the two final models differed.

A conditional intermediate implementation retained pop/push for reuse and used
PeekMut for fresh training; its passing fixture log is archived for provenance.
The user preferred one implementation given the inconsistent timing benefit.
Final PairIndex::best uses the previous pop/push algorithm for both modes.
The final 20-test suites include the cohort regression.

## Serialization decision

These standalone probes compile the exact old/new WordCounts sources with the
same optimization settings and cached dependencies. They serialize 262,144
synthetic UTF-8 keys with weights 1–17, using fixed AHashMap seeds and equal
traversal order. Both representations produce identical 4,317,665-byte JSON.
The preallocated Vec probe excludes input construction, buffer allocation and
output validation, performs ten warmup calls per implementation, then six
alternating AB/BA blocks of 50 calls per implementation. One CPU is pinned;
no cargo/rustc or other task benchmark runs concurrently.

| Representation | Paired CPU change | Paired wall change | Added allocations |
| --- | ---: | ---: | ---: |
| Entries | +4.65% | +4.60% | 0 |
| Map | +0.98% | +1.01% | 0 |

Map delegates to its existing Serialize implementation; Entries delegates to
Serializer::collect_map over borrowed pairs. Global allocator counters report
zero allocations, deallocations or reallocations during either serializer with
the output buffer reserved. The earlier byte-counter-writer probe also showed
no gain (Entries about +10% CPU, Map about +3%); its shorter samples and different
writer make it a supporting diagnostic rather than the primary result.
The user explicitly chose to retain collect_map and accept the measured local
serialization cost. No serialization speedup is claimed. Raw samples, harnesses,
source copies and compiler arguments are in [serialization](serialization).

## Five retained simplifications

1. push_ordered and Writes::record return unit. Generic push/append ordering
   checks and arithmetic overflow errors remain fallible.
2. scan_symbols returns ControlFlow carrying the break reason. initial_edges
   propagates the error without a separate failure variable. Tests cover plain
   and decorated scans, multibyte UTF-8 boundaries, filtered first/last characters,
   and stopping callbacks after the first failure.
3. Builder owns an unfed Trainer; Trainer defines defaults once. Public builder
   methods and serialized fields stay unchanged. Tests check defaults, every
   builder option, JSON, and the existing nonserialized progress format.
4. Reuse collects adjacent-deduplicated word IDs with existing itertools. Original
   occurrence order stays intact; no HashSet replaces the ordered sequence.
5. Rule reads its pair from Candidate.priority through a small accessor instead
   of storing a duplicate pair.

The initial dense-count staging boundary remains unchanged.

The collection probe uses the actual owned Positions codec and iterator plus
copies of Corpus::resident checks and its exact word partition_point lookup.
A token plane is not needed for this collection-only experiment. Each case has
262,144 positions, ten excluded warmup collections per implementation and six
alternating AB/BA blocks of 50 collections per implementation, on CPU 0. Setup
and ordered-output equality checks are excluded. This measures collection,
not complete reuse training or process RSS.

| Occurrences | Paired CPU change | Old/new Vec capacity | Old/new reallocations |
| --- | ---: | ---: | ---: |
| no duplicates | +7.01% | 2,097,152 / 2,097,152 bytes | 16 / 16 |
| one duplicate per 33 positions | +8.83% | 2,097,152 / 2,097,152 bytes | 16 / 16 |
| heavy duplicates | -6.63% | 2,097,152 / 32,768 bytes | 16 / 10 |

Low-duplicate cases retain the same geometric Vec capacity and incur a local
CPU increase; both old and new paths grow the Vec through 16 reallocations.
With 64 occurrences per word, new capacity is 64 times smaller and collection
CPU decreases. Requested allocation-byte totals are sums including reallocations,
not simultaneous resident bytes. This tradeoff is retained and is not presented
as a universal speedup. See [dedup](dedup).

## Final training comparison and validation

The final comparison uses the same pre-change `d4e3fa45` baseline binary and
final `a31898fa` binary. Thus both arms use pop/push heap certification; remaining
training-source differences are the five simplifications. Serialization is
outside this training timer. As in the initial comparison, each input has an
excluded AB warmup, followed by formal BA and AB blocks (12 runs total, 8 formal).
All 12 full models agree, including with the initial comparison's reference
models. No measured child swap or concurrent task builds/benchmarks were observed.

| Input | Baseline / final wall (s) | Baseline / final CPU (s) | Paired wall change | Paired CPU change | Paired HWM change |
| --- | ---: | ---: | ---: | ---: | ---: |
| ByteLevel | 21.740 / 21.104 | 68.391 / 67.088 | -2.77% | -1.83% | +0.09% |
| Whitespace | 19.435 / 19.683 | 53.059 / 54.018 | +1.36% | +1.88% | -0.59% |

Absolute values are medians of two formal samples per arm; deltas are medians
of within-block percentage changes. A ratio of the displayed absolute medians
can differ from the paired delta. Individual CPU changes reverse direction:
zh-256MiB: +1.83% and -5.48%; zh-whitespace-256MiB: +6.92% and -3.16%; these observations do not establish a stable speedup or a consistent
training slowdown. Source/build metadata, raw runs and model validation are in
[final-training](final-training).


Default and no-default-feature library suites each pass all 20 tests, including
64 generated cases that compare every rule and complete model with the sequential
oracle at 1, 4 and 8 workers. All-target Clippy passes with warnings denied.
Changed Rust files pass rustfmt and the diff passes whitespace checks. Existing
whole-crate import-format differences in five untouched files remain unrelated.
No production unsafe code changed. See [validation](validation).

[source](source) contains exact BPE Rust sources for the original baseline,
initial PeekMut change and final simplified source. Per-run model validation
includes canonical hashes; one complete compressed reference model is retained
for each input/comparison. Absolute paths in archived compiler commands and jobs
record the actual execution environment; input hashes and build metadata identify
required local assets. The shared VM and small number of formal training pairs
limit these observations to descriptive comparisons.
