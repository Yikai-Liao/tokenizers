# Final independent structural and test review

Reviewed source: `4d181c51`. Accepted limits: production 2100, tests 800; rustfmt
nonblank noncomment lines. Entire default BPE test scope includes engine tests,
independent reference, public API/feed/word-count helpers, shared trainer wrapper,
conditional test imports and the standalone Miri harness.

The first new round replaced the restart boolean with explicit Selection states,
separated CorpusPlan metadata from the materialized Corpus, centralized event-route
stream ownership and closed Positions inputs over trusted source types. Positions
now rejects reversed block ranges before pointer access. Local comments explain
AA buckets, left-owned neighbor boundaries, signed reuse ledger actions, Arena
lease non-reentrancy and borrowed-storage lifetime invariants.

Independent structural review found no material issue. Test review identified
missing tokenizer JSON/encode/special-token checks, real worker-pool observation,
feed flush/error/non-Fused iterator and strict progress-schema coverage, a lost
long alias cohort and a ten-byte delta query. Consolidated high-level fixtures now
cover those boundaries while retaining the independent semantic oracle. Original
fixtures were checked against the new oracle before consolidation.

A fresh second structural/test review found no semantic loss; its sole actionable
finding was an omitted two-line conditional test import. Another fresh budget audit
found the shared BPE wrapper test's 12 lines. Both omissions were fixed in the
counter and status, producing 2100/780. No extra test compression was needed.

The final fresh whole-crate audit independently traversed tk-train, all six engine
modules, the oracle, tests and DESIGN proof. It independently obtained 2100/780,
found no implementation outside scope or default BPE test helper omitted, and
reported no new material, actionable finding. That completed the structural/test
iteration criterion. All Rust is identical to the fully validated `e26115c2`.

Native default/no-default: 17 library tests each; no-default doctest: 1; strict
Miri: 2. Clippy all targets with -D warnings, rustfmt, budget and whitespace passed.
Raw logs are archived under evidence/*review-round2*.log. The compact count report
is evidence/lines-review-final.json. Performance follow-up uses this exact source.

Performance follow-up was independently audited after completion. One reviewer
caught a historical owner-directory baseline mismatch; the report now uses the
actual whole-candidate compressed-write control, and retains its raw record. A
new fresh review after the fix independently verified the final source scope,
metrics, hashes, geometry, phase boundaries, historical ROI and cutoff pairing,
and found no new material, actionable issue. No new benchmark or Rust change was
needed for that evidence correction.
