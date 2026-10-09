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

Status: the pinned 1456-line archive remains available separately. Current HEAD
is the 1797-line typed-cursor round, with complete core models matching main:
English 1.959s and Chinese 30.873s (single exploratory samples). The Arena
prototype has 1924 production / 1462 test lines; 37 library tests passed and
performance measurement is running. Hardware prefetch and owned stream append
precede this round. Each round now uses a fresh read-only reviewer.
[JOURNAL.md](JOURNAL.md) records all measured variants and their commits.
Overall combination selection, adjacent baseline/candidate measurements and
final validation remain pending. No global optimum or comparison against
PR #2501's other implementations is claimed.
