# Two affix slot candidates, neither adopted

The original isolated flags comparison locates a repeatable initialization slowdown in the named-bool decoration path: Chinese +5.96%/+12.68%, English +12.61%/+12.48%. Whole-train deltas change direction in both language repeats. This differs from the structural cost of affixes: their initial decorated identities must be allocated in original caller-map traversal order, so the current plain-alphabet shortcut cannot skip that scan for affix cases. The reusable decoration string and three-slot cache already exist; neither candidate changes allocation policy, token IDs, activation order, or the alphabet selector.

The first diagnostic compares old `38b283c0` with flags-only. Approximate externally controlled initialization counters show instructions +4.4–5.6%, branches +7.6–9.7%, and no rise in branch misses. This supports extra per-retained-character control/cache work. It does not establish a complete attribution of machine instructions to the tuple match or Option alone.

Both attempts below compare to the same current `0dd1ca48` production logic, with identical stage hooks and prepared-input protocol. They are independent from the accepted SelectedRules HashMap change. Neither attempted slot change is in the simplification branch.

A guards the no-affix case, then uses `if add_suffix { 1 + usize::from(add_prefix) } else { 0 }` to select slots 0/1/2. It keeps the named booleans, Option return, and cache. Chinese initialization slows 8.75%; English improves 19.14%, but its unchanged alphabet selector also improves 14.92%. A does not show a consistent improvement. No instruction profile was collected for A, so a compiler or branch explanation is not proven.

B uses `(usize::from(add_prefix) + 2 * usize::from(add_suffix)).checked_sub(1)`. Zero means no affix/None; prefix, suffix and both map to slots 0/1/2. Chinese initialization slows 1.58%; English slows 19.90%, with its unchanged alphabet selector also slower 36.78%. English external-counter repeats show about 2.3% fewer instructions and 1% fewer branches than current, yet about 16–23% more cycles and 4–6% more branch misses. Reducing arithmetic/control instructions alone did not reduce observed cycle cost. Those counter windows have controller latency and are diagnostic; they do not prove a specific cause for the cycle difference or predict whole-train throughput.

These results justify retaining the current implementation. Both failed experiments and every raw sample remain available; no result is discarded because of direction. A faster implementation would need a measured improvement in the cached per-character path, or a deliberate redesign of the initial affix scan that preserves first-observation ID allocation. That larger redesign has not been implemented or measured here.

All four case pairs compare complete vocabulary entries in ID order and every ordered merge byte-for-byte, and are equal. A has 42 passing BPE tests in both configurations; B has 42 passing no-default BPE tests. All retain exactly the existing encoding-roundtrip failure. Both Clippy runs with all targets/no defaults and `-D warnings` pass. Changed production-file formatting and whitespace checks pass. Commands and raw logs are captured.

Protocol: 4 workers pinned to CPUs 0–3, serial fresh processes, release opt-level 3/fat LTO/one codegen unit, exact dependency locks. Raw Chinese/English inputs are about 256 MiB, Whitespace-prepared with fixed map reconstruction/seeds [11,13,17,19]. Actual alphabet limits 2047/128, vocab 50000, min frequency 2. Both prefix `##` and suffix `</w>` are nonempty. The selector is unchanged within each candidate/current pair. CPU and memory sampling, model export, and stage hooks use the same driver as the previous report.

`train_seconds` is public do_train only, excluding prepared-input load and model export. Vocabulary initialization excludes alphabet selection and CorpusPlan::build; Combined sums selection and Vocabulary initialization. HWM is process peak before validation/export, including input load. Paging counters span the whole process, including load/export; host counts include other processes. The current source in this benchmark predates the separately accepted HashMap integration, so these are slot-isolation results rather than fresh measurements of the combined published branch.

The perf controller enables between receiving the selection log and initialization log. The acknowledgment can take tens of milliseconds, significant for English's short stage; counters are approximate diagnostic windows and are not included in the primary timing rows. Complete perf commands, counter content and controller timestamps are retained under profile-init. No cache-drop, swap-disable or host workload change was performed. No compile/test overlapped primary samples.

Durable artifacts: `/root/code/tokenizers-workspaces/bpe-affix-slot-results-20261011`. Exact A/B production source, diagnostic source/locks/binaries, patches, complete models, jobs, raw logs, CSV/JSON and hashes remain there. Git contains a compact report and raw logs; raw unified patches, original trailing-column TSV counters, binaries and full models are in the durable archive. TSV content is also embedded verbatim in profile-init.json. CSV uses LF.

| Variant | Language | Alphabet s | Vocab init s | Combined s | Whole train s | Train CPU s | HWM MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| A baseline | zh | 1.973388 | 1.866912 | 3.840372 | 26.660850 | 69.219095 | 1769.54 |
| A candidate | zh | 2.030691 | 2.030213 | 4.060989 | 26.939377 | 68.275660 | 1774.75 |
| A candidate | en | 0.053612 | 0.056005 | 0.109699 | 1.653643 | 3.439195 | 159.12 |
| A baseline | en | 0.063010 | 0.069265 | 0.132379 | 1.725241 | 3.369150 | 156.85 |
| B baseline | zh | 1.953081 | 1.871902 | 3.825065 | 26.224369 | 65.203476 | 1774.80 |
| B branchless | zh | 2.011210 | 1.901491 | 3.912794 | 26.895632 | 67.024466 | 1772.35 |
| B branchless | en | 0.067543 | 0.063577 | 0.131202 | 1.736662 | 3.701580 | 155.27 |
| B baseline | en | 0.049382 | 0.053027 | 0.102507 | 1.630597 | 3.407282 | 161.82 |

Deltas against current in the same case; negative time means faster.

| Variant | Language | Alphabet | Vocab init | Whole train | HWM |
|---|---|---:|---:|---:|---:|
| A | zh | +2.90% | +8.75% | +1.04% | +0.29% |
| A | en | -14.92% | -19.14% | -4.15% | +1.44% |
| B | zh | +2.98% | +1.58% | +2.56% | -0.14% |
| B | en | +36.78% | +19.90% | +6.50% | -4.05% |

Whole-process paging, not only the train timer:

| Case | Arm | Child major faults | Max child swap KiB | Host swap-in pages | Host swap-out pages |
|---|---|---:|---:|---:|---:|
| zh-affix | baseline | 0 | 0 | 15 | 19508 |
| zh-affix | candidate | 0 | 0 | 7 | 0 |
| en-affix | candidate | 0 | 0 | 0 | 0 |
| en-affix | baseline | 0 | 0 | 1 | 0 |
| zh-affix-branchless | baseline | 0 | 0 | 9 | 0 |
| zh-affix-branchless | branchless | 0 | 0 | 4 | 0 |
| en-affix-branchless | branchless | 0 | 0 | 1 | 0 |
| en-affix-branchless | baseline | 0 | 0 | 0 | 0 |
