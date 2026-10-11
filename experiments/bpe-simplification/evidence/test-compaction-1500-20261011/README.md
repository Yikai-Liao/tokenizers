# Ordinary BPE tests within 1500 lines

Code commit: `0506b39c466f2c751755bdf499edb8fe9fd0083a`. The [source metadata](source.json) records checksums and preserved blocks.

The second pass reduces ordinary BPE test code from 2691 lines at `e67e4bc1` to **1500 lines**, including the four-line Miri entry. Relative to the original `b106736b` tree, the reduction is 1306 lines. This implements the newer hard-budget plan; it deliberately removes checks and does not claim equivalent coverage.

The complete independent sequential reference and the existing encoding roundtrip test are byte-for-byte unchanged. `positions.rs` is also byte-for-byte unchanged. The roundtrip still fails for the real input `测`; it is neither ignored nor excluded from the BPE test command. No production algorithm, interface, storage, or instrumentation changes occur in this pass. The [syntax audit](counts.json) compares all production tokens after removing only `cfg(test)` items, fields and statements; all comparisons match both `e67e4bc1` and `b106736b`.

The count removes Rust comments with tree-sitter, then counts nonblank lines. It includes all of `tests.rs`, every ordinary BPE `cfg(test)` item/field/statement, all helpers and fixtures, the independent reference, and the Miri entry. Parity Trainer and its unchanged `word.rs` remain outside this ordinary BPE scope. Individual module allowances are redistributed within the global 1500-line cap; nothing is moved into external fixtures or another test category.

| Module | Original `b106736b` | First pass `e67e4bc1` | Final | Reduction from first pass |
| --- | ---: | ---: | ---: | ---: |
| `tests.rs` | 1226 | 1141 | 736 | 405 |
| `index.rs` | 370 | 370 | 135 | 235 |
| `merge.rs` | 457 | 529 | 157 | 372 |
| `corpus.rs` | 264 | 194 | 84 | 110 |
| `vocabulary.rs` | 181 | 149 | 104 | 45 |
| `word_counts.rs` | 54 | 54 | 30 | 24 |
| `positions.rs` | 238 | 238 | 238 | 0 |
| `mod.rs` test support | 12 | 12 | 12 | 0 |
| Miri entry | 4 | 4 | 4 | 0 |
| **Total** | **2806** | **2691** | **1500** | **1191** |

See [counts.json](counts.json), [original comparison](counts-vs-original.json), and [count-and-audit.py](count-and-audit.py). Formatting uses only the repository's changed-file formatter, preserving blank lines between adjacent functions. Test names are shortened and the subprocess `--exact` name is updated.

## Remaining responsibilities and deliberate exits

- The model/reference differential still checks complete vocabulary IDs, merge order and trace with 1, 4 and 8 workers. It retains generated cases, real IDs above 65535, signed/u64 limits, zero-merge validation, AA overlap, aliases, and a large AB/AA fixture crossing task boundaries and producing below-floor partial births. The separate 127/129/257 fragment matrix and dense 128-word reserved-alias matrix are removed. Internal full/partial publication shapes and the exact number of independent rules committed in one batch are no longer separately asserted.
- Restart has two distinct fixtures. A real fresh attempt exceeds the alphabet cutoff with a forced `a/b` set and a `q/z` frequency tie, then restarts into reuse. The selected tie winner is forced in a separate oracle trainer so repeated alphabet selection cannot make the differential depend on a different valid tie winner. The other fixture commits XY, allocates a speculative UV ID, then restarts when AB reuses an activated identity; final model/trace checks exclude the failed attempt. Repeated internal worker/entry-point matrices and the third restart harness are removed.
- Feed and public pool/progress observations share one isolated subprocess entry. Feed retains first exhaustion of a non-fused iterator, one callback per input, empty output, duplicate/Unicode counts, and 2047/2048/2049/4097-key flush boundaries. Callbacks observe the ambient two-worker pool directly; disabled parallelism permits at most one participant. A synchronized two-fold failure proves that 4097 published keys and a successful tail precede another callback's failure, while stored trainer state remains unchanged. The separate ordinary-error position matrix, repeated state/serde/successful-model comparisons, length matrix and long-string checks are removed.
- Public training still observes the actual requested pool inside materialization, fresh preparation and reuse preparation. Progress retains parseable JSON, final merge current/total, zero merges, real restart and Silent output. Exact field counts, per-stage initial zero events, intermediate restart completion, and separate empty/no-bar scenarios are removed.
- Index retains unsigned/signed intermediate overflow, preparation failure before any write, and owner commit failure after writes with no published births. A small history regression creates two real cohorts in separate commits: count 10 becomes 5 through a removal of 8 and birth of 3. Both cohorts are then repaired and taken in the expected order. The previous hand-driven vocabulary/full-trace harness, bucket/container publication matrices and repeated worker failures are removed.
- Corpus retains adjacent fixed-span writes in forward/reverse/parallel order, an unchanged empty write, and occurrence-specific spans with an existing replacement ID and a one-slot right endpoint. Literal initial ranges cover empty, oversized word, zero budget, exact boundaries and `usize::MAX`. Random layouts and the second linear range oracle are removed.
- Merge retains real preparation with a rejected first coordinate followed by valid AA starts across two tasks, four compatible rules with shared heads/tails and exactly-once adjacent replacement births, a real reuse cohort containing different occurrence spans, stale directory recovery after an error, and a reachable deletion-only birth. Setup/order/worker matrices and redundant write-plane comparisons are removed. These finite fixtures do not recover the all-round freeze/copy/replay oracle removed in the first pass.
- Vocabulary retains literal affix IDs and spans for single/empty words, empty affixes, original filtered boundaries, cached identity hits and special-token identity reuse. A separate multi-character UTF-8 fixture preserves decoration ID order; independent Unicode presence is compared with 1 and 4 workers and empty input. Zero-weight observed specials and forced unused characters still have explicit activation checks. Repeated vocabulary initialization and complete model matrices are removed.
- Builder retains default JSON/restore and the exclusion of `progress_format` from serialization; exhaustive setter forwarding is removed. Input reuse retains actual borrowed and stored training without consuming input, while repeated model-update/serde checks are removed. WordCounts retains equality across representations, changed value/key/length, wide serde, duplicate-key last value, and invalid signed input; clone self-comparisons, empty self-comparisons, double-direction assertions and exact Debug output are removed.

The first pass already removed the global write oracle and its production-adjacent test fields. This pass removes no additional instrumentation; worker observations and the independent trace remain. No throughput benchmark matrix was run for this test-only change, and the Miri entry is retained/count-audited rather than rerun.

## Verification

Final commands use the exact existing Cargo lock, `--offline --locked`, `CARGO_INCREMENTAL=0`, and the shared debug target. Both BPE commands filter only `trainers::bpe`; neither uses `--skip` nor `--ignored`.

| Check | Result |
| --- | --- |
| Default BPE | 32 passed, 1 existing `测` roundtrip failure, 7 unrelated tests filtered |
| No-default BPE | 32 passed, the same existing failure, 7 unrelated tests filtered |
| Clippy, both configurations, `-D warnings` | Existing two `redundant_field_names` errors at production `mod.rs:514` |
| Clippy, both configurations, `-D warnings -A clippy::redundant_field_names` | Pass; every other lint remains denied |
| Changed-file formatter and `git diff --check` | Pass |
| Production syntax, preserved reference/roundtrip/Positions | Match baseline |

The [first-pass baseline report](../test-compaction-20261011/README.md) records original 43-pass/1-known-failure and first-pass 44-pass/1-known-failure results under both feature configurations, with the same two Clippy errors. This pass has fewer tests because responsibilities and matrices were deliberately removed.

Raw final logs and representative temporary mutation results are included alongside this report. All temporary production edits are restored before final validation and commit. The mutations only demonstrate that the small fixtures detect the named errors; they do not establish equivalence to the retired broad oracle or correctness under every concurrent schedule.

The three temporary mutations were all detected: heap-head repair changed the historical-cohort order from expected `[21]` to `[20]`; an adjacent rule used the wrong replacement ID; and a reused identity read shared ID geometry instead of the occurrence span. See [mutations.json](mutations.json) and the corresponding raw logs. Both complete BPE feature checks were rerun after restoring all three production edits.
