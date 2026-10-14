# Ordinary BPE test coverage

The differential helper compares every selected `(pair, count, merged ID)`,
complete vocabulary IDs and ordered merges at 1, 4 and 8 workers. Its sequential
oracle owns word edits, the count ledger and historical birth cohorts, and
shares alphabet selection with the trainer. Literal Unicode IDs, the weighted
cutoff expectation and character-presence tests independently check that selector.

| Contract | Retained coverage |
| --- | --- |
| Weighted ordering, equal counts, empty words, zero weights, Unicode, duplicate reserved strings and affixes | `parity` uses 128 fixed-seed combinations, including stale reuse cohorts in case 84; `overlap_and_alias` supplies equal-count and literal alias traces |
| Compatible rules and shared boundaries | `targeted_models` compares full models and traces for `abacdbdc` and weighted `ab/ac/db/dc` words |
| Multiple words and weight regions | `targeted_models` uses forty distinct words with repeated, unit and zero weights |
| Alias geometry and length admission | `targeted_models` compares suffix-`a` models for `baaaaaba/ba` with no limit, limit 3 and `usize::MAX`; `parity` includes limits 0, 1 and 2 |
| AA overlap, restart blocks and preparation across tasks | A 20001-symbol odd AA word and a long AB producer in `overlap_and_alias` compare complete models and traces without fixing the number or size of internal jobs |
| Alphabet filtering and decorations | `targeted_models` checks literal weighted-cutoff IDs and the complete filtered Unicode model; `affix_ids`, `unicode_id_order`, `alphabet_activation` and `unicode_presence` retain independent Unicode/decoration expectations; `limited_alphabet_wide_counts` checks full-u64 weights, repeated-character frequencies and the initial-alphabet cutoff |
| Full token IDs and counts | `wide_ids` crosses 16-bit IDs; `count_boundaries` covers full-u64 counts, per-key overflow, signed reuse limits and validation at vocabulary targets 0, 2 and 64 |
| Vocabulary target and allocation bounds | `vocabulary_bounds` checks small and empty input at the largest supported u32 target, and rejects larger targets on wider usize platforms |
| Initial-pair exception to the length gate | `initial_pairs_bypass_length_gate` retains the literal two-symbol merge with limits 0, 1 and 2 |
| Restart identity and retained alphabet | `restart_discards_ids` checks discarded speculative IDs and successful history; `restart_keeps_alphabet` exercises a real fresh-to-reuse restart with a constrained alphabet |
| Stale historical starts before alias activation | `reuse_stale_starts` covers a consumed head that formerly caused predecessor underflow or separator-span indexing; `reuse_stale_models` checks the nonpanicking wrong-model case. Both compare complete models and traces at 1, 4 and 8 workers and retain literal merge expectations |
| Adjacent concurrent writes | `adjacent_writes` checks both write orders, parallel adjacent writes and reused occurrence spans |
| Preparation and commit failures | `failed_rounds` first commits AB, then injects preparation overflow before writes and index overflow after writes at 1 and 6 workers; prior history/trace survives and failed owners publish no births |
| Intermediate arithmetic overflow | `intermediate_overflow` retains fresh-u64 and reuse-i64 failures while draining pending streams |
| Reusable public input and model failure | `reusable_input` preserves borrowed/stored counts over repeated training; `failed_train` preserves the old model after count overflow and a trained affix rejected by the model reader |
| Empty public training | `empty_training` replaces an existing model with an empty vocabulary at targets 0 and 64, and retains configured special tokens and initial characters without merges |
| Public serialization and encoding | `builder_serialization` checks trainer defaults/options; `tokenizer_roundtrip` checks feed, trainer serde, special-token return, model options, tokenizer JSON reload and literal affixed token IDs before and after reload |
| Feed callback execution and transactional failure | `public_behavior` isolates enabled/disabled/environment/one-worker settings; callbacks check the ambient pool, serial caller thread, exact reads/counts, nonfused exhaustion and 2047/2048/2049/4097 unique-key boundaries; controlled two-worker partial failure preserves prior trainer state; `feed_replaces_input` checks successful replacement, clearing with empty input, and continued callbacks after errors while preserving prior counts |
| Public progress and parallel-use flags | `public_behavior` covers normal, serial, environment-disabled, one-worker, zero-merge, Silent and restart settings; JSON parses strictly and finishes at the actual merge count; long-word models match the sequential oracle |
| Numeric position width, duplicates, restart blocks and seek | `gap_width` covers seven-bit boundaries through ten-byte u64 gaps; `leaf_roundtrip` covers empty/small/full-u64 sequences and 2049 repeated values with independent slice lower bounds |
| Chunk and leaf/root boundaries | `leaf_roundtrip` and `concat_boundaries` flatten chunks to the exact original sequence and retain duplicates across restart/leaf boundaries |
| Append failure and immutable roots | `append_errors` checks descending append rollback at restart boundaries, full-u64 append, length overflow and truncated cursor failure; `concat_boundaries` retains invalid ordering and root rejection |
| Sorted stream merging | `merge_streams` retains mixed roots/leaves/empty streams and duplicate coordinates through `u64::MAX` |
| Word-count representation and serde | `equality_and_serde` retains equality/order and full-u64 serialization limits |

Related [encoder tests](../../../../tk-encode/src/models/bpe/tests.rs) cover empty
character vocabularies, affixed ASCII and Unicode single-character atoms, and
byte-level inputs whose Unicode character ID does not replace their required byte
merges. These run through direct and batched span tokenization with caching enabled
and disabled for character models, before and after config reconstruction.
Empty byte-level vocabularies retain their missing-atom error.

The test-slimming change retained complete observable training comparisons.
It did not reproduce every removed private assertion. The suite deliberately
no longer checks precise initial range formulas, fresh-reader prefetch/cache
state, event-payload ordering, equal-priority cohort identity, prepared-job and
raw-slot layout, exact job counts or first-job length, injected invalid starts,
private occurrence-span arrays, directory reuse after an injected scratch
failure, artificially truncated scan domains, literal leaf bytes, repeated
cursor exhaustion, decoration-slot getters/setters, private edge coordinates,
standard iterator error recovery, `sizeof`, SmallVec inline/spill shape, or a
phase bitmap proving each internal stage's worker identity. That test-slimming
change left production queue, cohort, priority and count-repair semantics unchanged.

The formatted count excludes blank lines/comments and includes all ordinary
BPE test helpers, active test hooks, the complete oracle, the shared trainer
wrapper and the four-line Miri harness once. Parity Trainer and its exclusive
`word.rs` remain outside this scope. No tests moved to escape the count. The
Miri harness imports the same position module and its retained tests.

Historical slimming results, the before/after line audit and deletion mapping are
recorded in
[the slimming validation report](../../../../../experiments/bpe-simplification/evidence/test-slimming-20261014/VALIDATION.zh.md).
