# Combined BPE test coverage

The suite groups related contracts around complete training results, public
integration and immutable storage. The differential helper compares every
selected `(pair, count, replacement ID)`, full vocabulary IDs, ordered merges and
special tokens at 1, 4 and 8 workers. The sequential oracle owns its own word
edits, ledger and historical birth cohorts; it shares only alphabet selection.
Literal alphabet expectations independently check that shared selector.

| Contract | Combined coverage |
| --- | --- |
| Weighted ordering, equal counts, empty words, zero weights, Unicode and duplicate reserved strings | Fixed-seed model combinations plus an explicit equal-count fixture |
| Affixes, activated identity reuse, unactivated reserved IDs and restart behavior | Generated prefix/suffix combinations, duplicate specials, literal `baaba` traces, and 128 long words with interleaved zero/positive weights across three affix/gate settings; a reserved long merge activates after occurrence geometry is established |
| AA overlap and position restart boundaries | A 4097-symbol AA run and generated repeated words |
| Complete and partial birth pruning | Long AB producer split across workers, a competing complete XY producer, and complete-model/trace comparison |
| Strict newborn length admission, including limits 0, 1 and 2 | Generated length gates and independent sequential neighbor admission |
| Filtered symbols, forced alphabet and decorations | Generated filtered/decorated models, a filtered training fixture, and literal alphabet IDs |
| Full token IDs beyond 16 bits and separator distinction | 65536 reserved IDs followed by ordinary training and literal ID expectation |
| Joined compatible rules | Explicit three-rule batch preparation, application and commit, then finished selection |
| Full-u64 counts, per-key overflow, signed reuse limits and validation before zero merges | Literal success/error cases repeated with vocabulary targets 0, 2 and 64 |
| Public feed, trainer serialization, special-token return, model options and tokenizer JSON reload | Feed-to-training integration and complete serialized model equality and three literal input-to-ID expectations before/after tokenizer JSON reload |
| Feed flushing, first `None`, duplicate callback words and transactional process errors | Isolated public test at 0/31/32/33/127/128/129/257 items, resumed nonfused input, empty/Unicode/long callback words, bulk 2047/2048/2049 unique words, exact flat counts, full callback count after error and unchanged prior state |
| Requested training pool versus ambient pool and serial settings | Child processes install an ambient two-thread pool; materialization and both preparation paths check requested pool size and record that each executed; feed callbacks verify ambient worker size and concurrent/serial execution |
| JSON progress schema, starts and completion | Child-process matrix covers normal/no-bar/zero-merge/empty/Silent; strict JSON parsing, exact three-field schema, initial zero and final actual merge count |
| Numeric position width, duplicate values, restart blocks, seek, append, promotion and owned byte storage | One matrix of empty and small lists and lengths around block boundaries, values through `u64::MAX`, explicit ten-byte deltas, all native seek starts and independent lower-bound expectations for present and gap values |
| Published list lifetime, shared readers and position encoder scratch reuse | Two scoped threads build lists while reading earlier owned lists, then read them again after Codec is dropped |
| Invalid decoder ranges and unsorted construction | Explicit rejection before byte access; Miri imports this same implementation and tests |

The former fixture files and legacy trainer tests are consolidated into these
contracts. Their repeated helper infrastructure and overlapping private feed tests are
replaced by public flat-count/callback contracts. This map describes
asserted behavior; it does not claim exhaustive coverage of all combinations.
The storage constructor accepts only a slice, the engine's own builder or owned-list fragments, so
arbitrary safe iterators can no longer supply a false allocation cardinality.

The formatted line count includes test hooks, the complete reference oracle,
all default BPE public/helper test declarations and the Miri harness. Dedicated
word-count tests retain representation/order equality and full-u64 serde limits. Comments and blank lines are
excluded. The separately feature-gated parity trainer is unchanged and is not
used as a test helper for this engine.
