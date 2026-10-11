# BPE fixed-length Vec → Box experiment

Retain the fixed-length ownership change. The user requested stopping the detailed benchmark and committing once correctness and the initial neutral performance observations were available. The eight-scenario matrix is intentionally incomplete; no stable performance or memory improvement is claimed. Code commit: `d6dbdaeb`.

A is actual clean `bcbdabf1a241685e461048add139c59ade5b9635`. B changes only four ordinary BPE trainer files, with the separate [boxed.diff](boxed.diff). Existing B1, WordCounts representation, Feed, codec, Index routing/inner buffers, Positions, directory scratch and dynamic activation spans are preserved. No rebase occurs. [builds.json](builds.json) records frozen source/lock/binary hashes and release commands; [audit.json](audit.json) verifies unchanged production boundaries. Applied production files exactly match the checked and measured B hashes.

Freeze boundaries: sorted borrowed Plan.words; checked starts and completed weight_regions, moving directly into Corpus as needed; constructed atomic Slots; lazily allocated occurrence_spans after its existing fill; fixed Unicode characters and conditional decorated table after a local resize; per-batch heads/tails; outer shard array after unzip. Elements remain mutable, with unchanged atomics, joins, error/unwind access protocol and drop timing. Dynamic spans retain Vec; fresh starts are empty Box. No unsafe, feature, dependency or public API changes.

Both arms, in both default and no-default library suites, have **43 passes and the existing encoder-roundtrip failure**. Tokenizer JSON roundtrip checks precede that encoding assertion. Both feature settings of all-target Clippy with denied warnings pass; the four changed files pass formatting/whitespace checks. All commands and raw logs are retained in [checks.json](checks.json) and compressed logs. Existing oracle tests cover empty/zero-target/zero-weight/no-merge inputs, Unicode and affixes, narrow/wide Slots, reuse restart, alias geometry and errors, comparing complete models and per-rule traces at 1/4/8 workers. No tests are deleted.

Independent diagnostic copies add 15 complete canonical model/special-token/trace comparisons across A/B at 1/4/8 workers, all equal. Directed wide-ID/alias oracle fixtures pass. Valgrind on the tiny alias fixture records zero memory errors and actual occurrence-span materialization; it observes eight candidate shrink realloc calls in B and zero at those boundaries in A. These are diagnostic allocator observations, not native performance results.

## Completed measurement

29 completed fresh-process pipelines are valid: 20 formal samples and 9 uncounted warmups. Full sorted counts, vocabulary IDs, ordered merges, serialized models and special-token fields match, with zero sampled swap and pipeline major faults. One in-progress English warmup is intentionally interrupted at the user request; its job/logs and [user-stop.json](user-stop.json) remain. English formal comparisons are not made. This interruption is not an algorithm failure. The historical resident-coordinate assertion remains separate from the known encoder test failure; it did not recur in these completed samples.

Each completed Chinese scenario uses one warmup per arm, then A→B/B→A formal pairs. Two scenarios also finish the pre-recorded targeted third pair before the user stops the matrix. CPU affinity is 0 for one worker or 0–3 for four. Same 256 MiB Wikipedia source and public streaming preprocessing enter Feed. Wall/CPU include train and complete model/special-token output; HWM is captured before count export, canonical external sort, validation or encode. Normal production hashers and system allocator are used. Diagnostic trainer hooks are excluded. Common runner changes, feature/profile, inputs and cache policy are frozen identically.

| Scenario | Formal pairs | Wall paired median | CPU paired median | Process HWM paired median |
| --- | ---: | ---: | ---: | ---: |
| bytelevel_regex-w1 | 2 | -0.56% | -0.59% | +0.17% |
| bytelevel_regex-w4 | 3 | -1.15% | -2.44% | +0.14% |
| whitespace-w1 | 2 | -0.19% | -0.19% | +0.11% |
| whitespace-w4 | 3 | +0.47% | -0.93% | +0.05% |

Percentages are medians of paired B/A deltas. [measurements/summary.json](measurements/summary.json) retains every A/B value, min/max and per-pair percentage; [runs.json](measurements/runs.json) retains execution order, commands, raw metrics and validation hashes. Every completed job/stdout/stderr is archived, with one full model reference per input and hashes for all copies. The pre-result screening/third-pair rules and initial-two-pair decisions are retained; further pairs and lifecycle splits are stopped by the user instruction.

## Ownership and capacity findings

On this target Vec descriptors are 24 B and Box slice descriptors 16 B. Measured struct sizes: InitialTokenIds 80→64, Corpus 160→128, Plan 200→160, Slots 32→24, SelectedRules 112→96, PairIndex 64→56 B. These descriptor savings do not halve arrays or establish RSS benefits.

Representative Chinese ByteLevel/four-worker diagnostics have 202,787,422 narrow slots with len==capacity, saving zero payload. Words/starts/characters/heads/tails likewise have no spare payload in the representative real runs. Chinese weight_regions discard 8,416 requested bytes; English Whitespace/one-worker discards 432 region bytes plus 336 outer-shard bytes. Box still drops/frees buffers normally. Diagnostics keep phase RSS/HWM, capacity/growth, conversion times and pointer changes separately; cumulative HWM differences are not treated as allocations. The native pointers stay unchanged in these representative conversions; that does not rule out in-place realloc. Valgrind replaces the allocator and its address moves cannot predict native movement.

See [diagnostic-summary.json](diagnostic-summary.json), diagnostic source/hashes, compressed raw fixture/phase/allocator logs and [canonical-fixtures.json](canonical-fixtures.json). Diagnostics do not establish that ordinary real corpora exercise alias materialization.

## Reproduction and delivery

Changed trainer and diagnostic sources, common runner sources/patches, Cargo locks, source/binary/input hashes, feature checks and completed raw measurements are retained. Unchanged trainer files are reconstructed from the frozen baseline SHA and verified with manifest hashes. Formal runner binaries and corpora remain at the local paths in [delivery.json](delivery.json). The common runner serializes all public AddedToken fields; an initial unsuccessful common-runner compile attempt is kept as a compressed log and occurs before any frozen binary or measurement.

Reproduce from the baseline SHA, applying boxed.diff only to B and the identical runner patch/locks to both. Build/freeze with build.py after updating local worktree/binary paths in the manifest. check.py runs the unfiltered feature/Clippy matrices. measure.py implements the original full matrix and third-pair rule; resuming it would run additional work deliberately stopped by the user. diagnostics.py/run-diagnostics.py reconstruct separate diagnostic builds. No further benchmark, automatic push or merge is performed. Retain this small change for ownership clarity; the completed data show no substantial regression and do not establish resource gains.
