# Review iteration after the measured delivery

Current review snapshot: 2100 formatted production logic lines and 766 test logic
lines, including all default BPE tests, the independent reference, public hooks
and Miri harness. The
production limit is 2100; the accepted test ceiling is 800.
Selection states, event routing, corpus construction and trusted codec inputs
have been tightened. Tests are consolidated around full-model/per-rule parity,
public integration and shared immutable storage; see the engine coverage map.

Default and no-default native tests, doctests, Clippy with denied warnings, fmt,
budget and strict Miri passed for this review round. The first fresh structural review found no material issue. Its independent test
review found undercounted feed/word-count tests and omitted public/codec boundaries.
The next round fixes the scope and consolidates those contracts; another fresh
structural/test review follows. Performance of these new changes has not yet been measured;
the results below belong to the earlier measured source.

## Earlier measured delivery: <25s core

Final source commit: d19e5bc6, branch simplify/bpe-maintenance-20261009.
Pinned baseline: Fork main e4f787dc189d9be7192107490d652096cde7480e.

Uninstrumented Chinese ByteLevel core, 4 workers, 50K vocabulary, min_frequency2:
23.878528469s, CPU76.444302s, RSS2943096KiB, complete model equality and child swap0.
Production2074 formatted nonblank noncomment lines; tests1522 including oracle,
shared helpers and Miri harness. User-approved production limit2100.

Complete producer publication, compatible batching/parallel aggregation, full-u64
positions and the dynamic Arena threshold remain. The threshold formula matches
the baseline; its input is resident slots, whereas the baseline uses initial
physical edges. This difference is tracked in the cutoff ablation. Four final cases/eight
processes all match complete baseline models with swap0. Default/no-default tests,
Clippy, fmt, budget, strict Miri and exact binary rebuild verification passed.
See REPORT.md and evidence/manifest-final.json for results and practical limits.
Earlier archived deliveries and experiments remain in git history.
