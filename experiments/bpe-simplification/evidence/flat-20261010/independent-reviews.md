# Independent review record

Both reviewers started without the implementation conversation, read the exact
working tree based on `245ac0ff`, and performed read-only reviews. They did not
modify source, compile, run tests or benchmarks. Validation executions belonged
to the implementation task.

## Implementation review

Result: zero new actionable implementation defects.

Independently compared the complete old coordinator with the new private
function, the unchanged position source, and corpus/index/merge bodies. Confirmed
the selector's receiver-only change and its reference sharing boundary. Reviewed
private visibility, default worker selection, test observers, the scoped Arena,
whole-attempt restart, phase joins and publication. Reviewed default/no-default
17+1 results, Miri 2 results, subprocess execution guards and coverage paths.
Reproduced the baseline 2571/799 and candidate 2567/796 counts. Confirmed that
the new counting scope includes public entry, feed and word-count representation
and has no hard limit. Current documentation links and preserved historical
instrumentation paths were checked.

## Integration review with main

Result after the parity clarification: zero immediate responsibility-merging
issues. One valid follow-up candidate remains: private shared feed/count storage
for BPE and WordLevel. The initial alphabet sharing proposal with parity was
withdrawn after the actual feature check failed because `super::BPE` is missing;
the same stale reference exists in main and the starting revision.

WordLevel can borrow count entries and retain its own descending-frequency,
lexical-tie sorting without rebuilding a map. A complete follow-up must verify
flat serde, Unicode tie ordering, duplicates, preservation of old state on
failure, callback execution after errors and ambient-pool behavior. Small-input,
long-word and high-cardinality performance has not been measured. The proposal
does not extend to Unigram's u32 counters or parity's per-language maps.

The reviewer retained boundaries between training Vocabulary and encoding tables;
initial-ID interpretation and Corpus activation/geometry; Corpus, PairIndex,
Batch/Prepared and compressed Positions; fixed-coordinate ordinary training and
linked-word parity; root private coordinator and public policy entry; wrapper,
WordPiece and progress rendering; production merges and the independent oracle.
No existing encoding component can directly replace append-only training ID
assignment without acquiring new incremental-maintenance responsibilities.

The equal-priority reuse cohort proof gap remains unresolved. The review did not
claim that this relocation closed it.
