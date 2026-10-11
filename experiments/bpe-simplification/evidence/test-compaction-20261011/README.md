The first test-compaction pass replaces per-round corpus freezing, copying and
write replay with finite endpoint and preparation fixtures. The complete
independent sequential model/trace reference remains. This deliberately gives up
checking write permutations in every round of every training input; the focused
fixtures do not provide equivalent coverage or prove all concurrent schedules.

Base: `b106736b`. Code: `e67e4bc1`. This is the first uploaded plan; the later
1500-line plan is a separate, unfinished task.

Counts exclude blank lines and Rust AST comments. They include all of tests.rs,
all test-only items/fields/statements in ordinary BPE modules, and the complete
four-line Miri entry. Parity code and its word.rs are outside this task.

| Module | Before | After | Change |
| --- | ---: | ---: | ---: |
| tests.rs | 1226 | 1141 | -85 |
| corpus.rs | 264 | 194 | -70 |
| merge.rs | 457 | 529 | +72 |
| vocabulary.rs | 181 | 149 | -32 |
| index.rs | 370 | 370 | 0 |
| positions.rs | 238 | 238 | 0 |
| mod.rs | 12 | 12 | 0 |
| word_counts.rs | 54 | 54 | 0 |
| Miri entry | 4 | 4 | 0 |
| **Total** | **2806** | **2691** | **-115** |

Removed responsibilities and retained observations:

- The symbols omnibus test duplicates initial_edges' plain/decorated UTF-8
  filtering, coordinates, weights and exact error prefixes. Literal vocabulary
  IDs retain decoration order. Iterator-only short-circuit checks are dropped.
- The older single Feed callback failure duplicates the unification error matrix
  in disabled/one/many subprocesses. Public pool, callback independence and
  first-exhaustion checks remain. Subprocess setup and repeated model expectations
  now have one shared implementation.
- Repeated initialization with one immutable alphabet is replaced by the existing
  real cutoff tie and fresh-to-reuse restart test. Field-move pointer identity and
  repeated cache reads are dropped; literal spans/IDs and decoration texts remain.
- FrozenMatch, Corpus's write_oracle/copy/replay/probes, Job's frozen/rule fields,
  CohortPreparation's frozen transport, and every-round assert_write_orders are
  removed. Corpus fixtures compare complete literal token/occurrence-span arrays
  for adjacent writes, unequal occurrence lengths, old/new replacement identities,
  one-slot right symbols and empty input, with forward/reverse/parallel writes.
- Actual prepare paths check rejected initial coordinates, empty/inline/spilled
  tasks, AA greedy disjoint starts, four compatible distinct replacements and
  reuse after AA creates unequal BA spans. Shared-rule weights are now distinct
  so complete expected coordinate arrays have a deterministic layout.
- An unreachable empty-starts/nonempty-changes Job is replaced by a real
  length-gated deletion-only Change, endpoint application and index commit.

No production syntax changes: count-and-audit.py removes test-only AST ranges
and compares the remaining Rust syntax. index.rs, positions.rs, mod.rs, Parity
and word.rs are byte-identical to the base. Existing fragment-publication and
neighbor-directory recovery tests remain; no Source/Birth/Positions contracts
were changed. Tests were not moved to another file or reclassified.

Validation, with the same locked dependencies and no test skips/ignores:

- Base default/no-default: 43 pass, 1 existing failure each.
- Final default/no-default: 44 pass, 1 same existing failure each. The preserved
  feed_training_tokenizer_json_and_encoding_roundtrip fails on `测`.
- Three temporary errors are detected by the focused assertions: reuse the first
  occurrence's geometry, mix cross-rule replacements, accept overlapping AA
  starts. All source bytes were restored before old instrumentation was removed.
- Changed-file formatter and git diff --check pass.
- Clippy -D warnings fails on both base and final in both configurations, solely
  for the two unchanged redundant_field_names errors at mod.rs:514. Both final
  configurations pass when only that existing lint is exempted on the command
  line; all other warnings remain denied. These are not reported as clean strict
  Clippy runs.

Raw logs, exact commands, mutation descriptions, counting code and audit results
are included here. No 256 MiB performance matrix was rerun for this test-only pass.
