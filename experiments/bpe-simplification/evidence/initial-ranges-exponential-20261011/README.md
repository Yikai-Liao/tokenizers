The dependency is retained. The current implementation in
[`explicit-one-word/`](explicit-one-word/README.md) admits one word first, then
asks the library how many additional words fit. This removes the initial trial's
`fitted.max(1)` fallback and separates mandatory progress from budget admission.
Exact-budget ends remain included, and the final corpus boundary remains handled
locally because `starts` stores word starts without a terminal sentinel.

The library owns doubling probes and the bounded binary search; this module owns
whole-word segmentation, including oversized words. Adopting the dependency does
not by itself make segmentation simpler than the standard-library binary
baseline. The improvement is that the chosen exponential algorithm stays outside
this function and its remaining business rules are explicit. It adds no grouping
object, boundary allocation or new public abstraction.

Other BPE uses were inspected through CodeGraph and the ordinary trainer sources:

| Site | Applicability and current decision |
| --- | --- |
| `Positions::lower_bound`, restart directory | A random-access sorted directory is compatible with exponential search. Any replacement must find the first seed at or above the target, retain the predecessor scan and preserve duplicate positions. Current seeks can target anywhere in the directory, so a gain from searching near its beginning is unproven. Keep `partition_point`. |
| `Corpus::word` / `weight_region` | Both use the partition index directly. Searches start at the directory beginning and query varying corpus positions; replacing `partition_point` adds result adaptation without removing a business rule. Keep the standard library. |
| Fresh self-pair neighbor membership | The two `binary_search(...).is_ok()` calls can technically use exponential search, but exact membership is already directly expressed. No known small-distance distribution justifies changing them. |
| Reuse cohort scanning | `iter_from_value` performs the initial seek; subsequent `find` advances a decoded iterator. The library requires a random-access slice. Materializing decoded positions just to use the API would add storage and work. Keep the iterator. |
| `Shard::commit` grouping | Birth fragments are grouped incrementally by `(bucket, pair)` in a `BTreeMap`, after ordered ledger updates. Slice grouping handles adjacent elements of a slice, so it cannot replace this aggregation directly. Sorting actions would also disturb per-action overflow checks. |

`exponential_group_by` is not used for whole-word budget segmentation: it would
require accounting for the absent final end boundary and its reverse traversal
would have different greedy-budget semantics. The search primitive fits the
existing representation without introducing those obligations.

The evidence below belongs to the **initial trial**, before the explicit
one-word formulation. Its frozen source, binary hashes and measurements remain
unchanged. The current source's validation and measurements are recorded
separately in `explicit-one-word/`.

The local implementation uses `slice-group-by` 0.3.1 with default features
disabled. The released crate has no runtime dependencies. Its published search
implementation doubles the probe distance, then searches the bounded interval.
The comparator subtracts the segment start before comparing with the budget,
avoiding coordinate addition. Word starts are strictly increasing, so an exact
match identifies one fitting boundary and contributes `exact + 1`; the insertion
index otherwise counts fitting boundaries. A fitting final tail is admitted
directly, and at least one word is always retained. `builds.json` records the
released crate checksum, source/binary/lockfile hashes and build profile.

The microbenchmark extracts the frozen binary, linear and exponential production
methods. Only the plan fields and independent word count are stubbed. All arms
share the same input layout, use black-box inputs and outputs, and include
allocation and drop. Each arm calibrates its own iteration count to approximately
80 ms to keep fast calls measurable. One warmup block is excluded; twelve formal
blocks cover all six arm orders twice, pinned to CPU 0. All three methods agree
on the correctness matrix and direct 3/4/2/6 expectations. The following values
are per-arm medians; raw runs and paired changes are in the JSON/CSV files.

| Layout | Binary | Linear | Exponential |
| --- | ---: | ---: | ---: |
| 100k words, all fit | 0.0567 µs | 84.61 µs | 0.0563 µs |
| 2m words, 3 large segments | 0.167 µs | 3120.65 µs | 0.247 µs |
| 100k words, one per segment | 6455.38 µs | 378.76 µs | 1054.09 µs |
| 100k words, about 3 per segment | 2262.03 µs | 158.20 µs | 486.39 µs |
| 100k words, 417 medium segments | 41.79 µs | 97.69 µs | 22.08 µs |

Exponentially searching is a performance compromise: it avoids the linear
variant's long scans and the binary variant's repeated whole-remainder searches
for short segments. Linear remains fastest for very short segments; binary
remains faster for a few large segments, although that difference is only tens
of nanoseconds here. These timings do not establish a simpler implementation.

Full training compares the pushed `ac32723d` binary-search version against the
exponential candidate. Both retain the restored Range interface and Shard; no
position/index/merge code changes are included. Frozen runners use locked
dependencies, opt-level 3, fat LTO and one codegen unit. Prepared Whitespace word
counts come from 256 MiB Chinese and English text, with a 50000 vocabulary target,
minimum frequency 2 and four workers pinned to CPUs 0–3. Each corpus excludes one
warmup pair and includes four alternating formal pairs. Loading and serialization
are outside the timer; output construction is included. HWM includes input
loading. No builds, tests or microbenchmarks run during formal training.

All twenty complete parsed-model comparisons pass, with zero observed swap.
Positive paired changes below mean an increase. Training manifests, individual
runs and summaries are in `training-zh/` and `training-en/`.

| Corpus | Paired wall-time change | Paired CPU-time change | Paired HWM change |
| --- | ---: | ---: | ---: |
| Chinese | +0.17% | +1.13% | -0.04% |
| English | -0.42% | +0.70% | +2.78% |

Wall-time changes have two positive and two negative pairs for each corpus.
This small sample does not demonstrate a stable end-to-end speedup or establish
equivalence. Under the current production slot budget, these results do not
establish an overall training-speed benefit from the dependency.

Default and no-default-features library checks each pass 38 tests, excluding the
previously reproduced baseline encoder roundtrip assertion. Clippy with warnings
denied, changed-file formatting and whitespace checks pass. Existing independent
procedural-linear comparisons and range invariants remain; direct expected
ranges for widths 3/4/2/6 at budgets 7/9/15/2 were added to verify exact matches,
final tails and oversized items. `validation.json` records commands and results.
The initial trial is superseded by the explicit-one-word formulation recorded in the subdirectory.
