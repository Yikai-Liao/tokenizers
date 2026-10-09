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

Status: 1456 production / 1384 test logic lines; default and no-default library tests pass (35 each), Clippy with warnings denied passes. Four first-run complete models equal main. The archive has a large measured regression and is not the final recommended performance choice. See REPORT.md and evidence/.
