# BPE module flattening

The public trainer and its existing private training coordinator now share
`trainers/bpe/mod.rs`. The five algorithm components are sibling modules instead
of children of `engine`. Alphabet selection moved to `vocabulary.rs`, alongside
the interpretation of alphabet and decorated token identities. This reduces the
navigation between the trainer entry and training workflow while retaining the
components' existing contracts. The former engine was a useful boundary; this
change has a modest organizational benefit, rather than eliminating an invalid
abstraction.

Starting revision: `245ac0ffaff8927c36b9d97ee0326f5e96edec41`.
HF main comparison: `fb49a29223724a96f26e061bec8ec7517aa14177` (`origin/main`).

## Preserved behavior

The coordinator remains one private function. Its body is byte-identical to the
old `engine::train`, with only `pub(super)` removed from the declaration. `train_counts` still
chooses the default worker policy, and tests call the same coordinator with
explicit workers and a test observer. Worker arguments do not themselves require
a separate function; the coordinator retains a complete training workflow.

Dedicated pool construction, phase joins, attempt restart with a retained
alphabet, Arena lifetime, error propagation and final publication are unchanged.
The position module is byte-identical; corpus, index and merge bodies are
byte-identical after their import sections. Alphabet selection changes its
receiver from `self` to `trainer` without changing its rules. The reference shares
only this selector and retains its independent merge implementation.

Public configuration, builder, trainer fields, serde declarations and the
`Trainer` implementation are unchanged. The rejected compact birth and combined
Writes experiments remain reverted. The equal-priority reuse cohort proof gap
from the heap experiment is still open.

Tests and their subprocess `--exact` names moved together. The public pool test
still requires child execution markers and progress output, rather than treating
an empty test selection as success. The Miri harness now imports the real
`bpe/positions.rs`. Current guide links were updated; historical instrumentation
and measurements keep their original paths.

## Descriptive differences

There are no hard line limits. The source count includes ordinary BPE components,
public API, feed and word-count storage. Test counts include embedded test code,
the reference, shared trainer wrapper and Miri harness exactly once. Optional
parity implementation is outside this ordinary-BPE count.

| Nonblank, noncomment Rust lines | Starting revision | Flattened | Difference |
| --- | ---: | ---: | ---: |
| Ordinary BPE production and API | 2571 | 2567 | -4 |
| Default BPE tests and helpers | 799 | 796 | -3 |

The earlier 2120 production count covered a narrower scope; it cannot be compared
with 2567. The baseline above was recomputed from exact `git show` inputs using
the same current counting functions. The test reduction is removal of a duplicate
test Pair alias plus formatting; no test case was removed.

| Git text diff, with rename detection | Added | Deleted | Net |
| --- | ---: | ---: | ---: |
| This change: BPE-related Rust | 159 | 169 | -10 |
| This change: BPE-related tree including guides | 182 | 185 | -3 |
| HF main to this branch: BPE-related Rust | 3429 | 685 | +2744 |
| HF main to this branch: BPE-related tree including guides | 3778 | 685 | +3093 |

The Git scope is the BPE directory, WordPiece, trainer wrapper and Miri source.
It includes comments and formatting. The tree rows also include BPE guides and
the coverage map. Repository-level reports, evidence and tooling are outside
this table. HF main rows describe the cumulative training rewrite, not the size
of this flattening. Raw per-file additions and deletions are available in
[diff-summary.json](evidence/flat-20261010/diff-summary.json); the counting report
also records its exact scope. These metrics describe the changes and do not
measure complexity or correctness.

## Validation and independent review

Default and no-default suites each passed 17 unit tests and 1 doctest. All-target
Clippy passed with warnings denied. Formatting and source migration checks
passed. Miri passed 2 tests with `MIRIFLAGS` and `RUSTFLAGS` unset, preserving
default borrowing and leak checks. No benchmark was rerun for this source
relocation, and no speedup is claimed.

The independent implementation review found zero new actionable implementation
defects. It independently checked complete source bodies, visibility, worker
selection, observer execution, phase ownership, restart behavior, test selection,
the test logs and the broadened count scope.

A separate review compared the flattened implementation with main and nearby
usable components. It found zero responsibility merges that need immediate
action. It also identified possible shared `feed` and `WordCounts` storage with
WordLevel, a separate whole-word vocabulary model. That proposal requires changes
outside `bpe` and has been withdrawn under the confirmed BPE-only code scope.
WordLevel was not modified. With that scope and the unavailable parity path,
this review found no further actionable responsibility-merging candidate.

The review initially identified duplicated alphabet selection in parity. A
subsequent actual `cargo check --features parity-aware-bpe` failed with E0432:
the legacy code imports missing `super::BPE`. That import also exists in both
the starting revision and main, whose ordinary model is now `PipelineBPE`.
The parity sharing proposal was withdrawn from current recommendations; the
duplicate is historical. This change does not repair that feature.

Training vocabulary and encoding tables retain distinct mutation lifecycles.
Vocabulary transfers initial spans to Corpus once; these are not two synchronized
authorities. Corpus, PairIndex, Batch/Prepared and Positions retain the geometry,
count/cohort, snapshot/application and storage-lifetime contracts they own.
Combining their files would not remove those responsibilities. WordPiece already
reuses BPE training; the model wrapper, progress lifecycle and independent
reference also retain separate responsibilities.

[Validation manifest](evidence/flat-20261010/validation.json) records commands,
results and source hashes. [Migration verifier](evidence/flat-20261010/verify_migration.py)
reproduces the body checks and baseline counts without a build.
[Independent review record](evidence/flat-20261010/independent-reviews.md)
records the reviewers' scope and final conclusions. Test and feature-check logs
are retained alongside these files.
