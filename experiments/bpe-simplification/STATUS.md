# Completed: <25s core within 2100 production lines

Final source commit: d19e5bc6, branch simplify/bpe-maintenance-20261009.
Pinned baseline: Fork main e4f787dc189d9be7192107490d652096cde7480e.

Uninstrumented Chinese ByteLevel core, 4 workers, 50K vocabulary, min_frequency2:
23.878528469s, CPU76.444302s, RSS2943096KiB, complete model equality and child swap0.
Production2074 formatted nonblank noncomment lines; tests1522 including oracle,
shared helpers and Miri harness. User-approved production limit2100.

Complete producer publication, compatible batching/parallel aggregation, full-u64
positions and main's dynamic Arena threshold remain. Four final cases/eight
processes all match complete baseline models with swap0. Default/no-default tests,
Clippy, fmt, budget, strict Miri and exact binary rebuild verification passed.
See REPORT.md and evidence/manifest-final.json for results and practical limits.
Earlier archived deliveries and experiments remain in git history.
