# BPE simplification execution

Baseline: Fork main `e4f787dc189d9be7192107490d652096cde7480e`. Branch: `simplify/bpe-maintenance-20261009`.

User constraints supersede the uploaded plan's provisional budgets:

- Engine production logic at most 2000 nonblank, noncomment lines after rustfmt; no compressed formatting or moving logic outside the counting scope.
- Test logic no larger than production, including the independent reference and shared test helpers.
- Preserve compatible batch selection and parallel preparation/owner aggregation.
- Exact complete vocabulary IDs and ordered merges; small-input full merge traces.
- Prioritize 4-core English and Chinese ByteLevel; choose measured speed/memory tradeoffs under the size cap.
- Preserve useful module boundaries and readable upper-level training stages.

The small ablations in PLAN.md alone cannot meet the hard size cap. The implementation therefore consolidates the engine around vocabulary, corpus geometry, compatible batches and count owners. The original baseline stays in a detached worktree and its release runner is immutable. The archived candidate uses safe U16/U32 atomic endpoints, full-u64 restart/delta streams, contiguous temporary births and joined merge phases. Initial pair counting streams directly into compressed lists. Performance tuning continues independently of the pinned archive.

Artifacts: `/root/code/tokenizers-simplification-results/`. Source and binary hashes, input hashes, machine and compiler settings are recorded in `manifest.json`. Removed rebuildable caches and sizes are in `cleanup.json`. Benchmarking starts only after builds stop. This machine is a 6-vCPU KVM guest, not the plan's fixed-frequency laptop.

Status: the pinned 1456-line archive remains separate. The delivery candidate
currently has 1858 production / 1439 test lines. Default and no-default library
tests pass (36 each), and all-target Clippy with warnings denied passes. Final
core/pipeline validation passed for English/Chinese, four workers and 50K
vocabulary: all complete models match main, child swap is zero. Arena is explicitly forgone at the user's request.

The current phase diagnostics identify compact event routing as about 2.5s of
the 4.7s commit gap, independently of Arena's end-release benefit. The delivery
candidate now restores compact references, route capacity reuse and actual-action
filtering. [PHASES.md](PHASES.md) gives the boundaries and limits of attribution.
[JOURNAL.md](JOURNAL.md) records measured variants and archive commits. [REPORT.md](REPORT.md) records final selection and measured tradeoffs; no global
optimum or comparison against PR #2501's other implementations is claimed.
