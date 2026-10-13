# Combined BPE test coverage

The suite groups related contracts around complete training results, public
integration and immutable storage. The differential helper compares every
selected `(pair, count, merged ID)`, full vocabulary IDs and ordered merges
at 1, 4 and 8 workers. Public integration checks special-token return.
The sequential oracle owns its own word
edits, ledger and historical birth cohorts; it shares only alphabet selection.
Literal alphabet expectations independently check that shared selector.

| Contract | Combined coverage |
| --- | --- |
| Weighted ordering, equal counts, empty words, zero weights, Unicode and duplicate reserved strings | Fixed-seed model combinations plus an explicit equal-count fixture |
| Affixes, activated identity reuse, unactivated reserved IDs and restart behavior | Generated prefix/suffix combinations, duplicate specials, literal `baaba` traces, and 128 long words with interleaved zero/positive weights across three affix/gate settings; a reserved long merge activates after occurrence geometry is established; a fresh XY rule precedes an activated-ID restart, the limited alphabet survives, and only the successful trace is returned |
| AA overlap and position restart boundaries | An 8195-symbol AA run spans restart blocks and two preparation jobs even at one worker; full model/rule agreement and generated repeated words |
| Complete and partial birth pruning | Long AB producer split across workers, a competing complete XY producer, and complete-model/trace comparison; 127/129/257 repeated ABC groups check rounded whole fragments versus partial publication and full model/rule agreement |
| Strict newborn length admission, including limits 0, 1 and 2 | Generated length gates and independent sequential neighbor admission |
| Completed initialization and retained cutoff | Ordered frozen input has four characters with limit two; literal multibyte prefix/suffix IDs precede weight sorting, original UTF-8 boundaries survive filtering, spans retain allocation identity on transfer, a true four-way cutoff tie reuses its two selected characters, and a separate four-character/three-slot cutoff forces an actual fresh-to-reuse restart and matches complete models/traces at 1/4/8 workers |
| Filtered symbols, forced alphabet and decorations | Generated filtered/decorated models, a filtered training fixture, and literal alphabet IDs |
| Full token IDs beyond 16 bits and separator distinction | 65536 reserved IDs followed by ordinary training and literal ID expectation |
| Joined compatible rules | Explicit four-rule preparation checks shared heads/tails and exact boundary births; full differential models and traces check ordered successful history at 1/4/8 workers |
| Preparation failure before writes | A committed AB round precedes a batch with a valid CD task and an overflowing EF task; `merge_next` fails with every endpoint unchanged at 1 and 6 workers |
| Reusable neighbor directories without reset | A task stops on weighted overflow; the next task reuses both directories with old indices outside its groups, indices naming different neighbors, and valid current-group hits; exact event order, weights, birth groups and position lists are checked |
| Shard-local commit and historical birth grouping | Unordered and zero-weight reuse fragments retain duplicates and publish in birth-group order below the floor; fresh removals retire counts before complete/partial publication; fresh grouped and intermediate signed overflow drain pending streams while retaining capacity |
| Full-u64 counts, per-key overflow, signed reuse limits and validation before zero merges | Literal success/error cases repeated with vocabulary targets 0, 2 and 64 |
| Borrowed map, stored entries, feed and repeat training | Complete model equality through the public entries, repeated successful calls with unchanged counts, and training after flat-map serde; decorated representations preserve the same borrowed traversal order |
| Failed training and failed model construction | Pair-count overflow and a successfully trained oversized affix rejected by the reader both leave the previous model vocabulary intact |
| Public feed, trainer serialization, special-token return, model options and tokenizer JSON reload | Feed-to-training integration and complete serialized model equality and three literal input-to-ID expectations before/after tokenizer JSON reload |
| Feed flushing, first `None`, duplicate callback words and transactional process errors | Isolated public test at 0/31/32/33/127/128/129/257 items, resumed nonfused input, empty/Unicode/long callback words, bulk 2047/2048/2049 unique words, exact flat counts, full callback count after error and unchanged prior state |
| Feed unification boundaries and failure interleaving | Disabled parallelism, enabled/one worker and four workers; startup W and W+31/32/33/64 lengths; per-ID callback/read counts; first exhaustion; 2047/2048/2049/4097 keys with repeated words across flushes; first/middle/last/multiple ordinary errors; exact counts and public model equality; controlled two-worker shared writes plus a successful tail and failing partition leave prior trainer state intact |
| Requested training pool versus ambient pool and serial settings | Child processes install an ambient two-thread pool; materialization and both preparation paths check requested pool size and record that each executed; feed callbacks verify ambient worker size and concurrent/serial execution |
| JSON progress schema, starts and completion | Child-process matrix covers normal/no-bar/zero-merge/empty/Silent; strict JSON parsing, exact three-field schema, initial zero and final actual merge count; the zero-merge case executes no materialization or preparation tasks |
| Numeric position width, duplicate values, restart blocks and seeking | Empty/small lists and a 2049-value repeated sequence; values through `u64::MAX`, gap-width boundaries through ten bytes, and value seeking checked against independent slice lower bounds |
| Independently consumable chunks and leaf/root boundaries | Nested chunks flatten to the exact original sequence at targets 0/1/127/128/129/511/usize::MAX; concatenated leaves of widths 1/2/3/127/128/129/700/1300 retain duplicated coordinates and seeking behavior |
| Direct encoding, append failures and immutable roots | Literal tail/seed/gap bytes, actual inline/spilled byte storage, empty leaf filtering, sole-leaf collapse, root rejection by concat and push, descending append rejection preserving the old sequence at restart boundaries, and exhausted/truncated decoder behavior |
| Interleaved sorted stream merging | Empty input, a sole root retained as a root, mixed roots/leaves/empty streams, duplicate coordinates and duplicate `u64::MAX` values |
| Fresh matches, stale coordinates and weight regions | Forty words cross more than one prefetch window; consumed starts, tails and separators are filtered while exact match intervals and repeated/unit/zero word weights remain correct |
| Reuse word scans and virtual neighbors | Literal reuse_spans checks endpoints and occurrence spans; reuse_scan_domain_and_bound_configuration checks historical-only scans versus `Some(usize::MAX)` whole-word scans and separator exclusion; complete reference comparisons cover affixes, aliases and length gates at 1/4/8 workers |
| Lazy initial-edge consumption and consumer errors | Filtered multibyte symbols yield literal pair/coordinate/weight triples; a consumer error leaves the exact consumed prefix and unconsumed suffix intact |

The former fixture files and legacy trainer tests are consolidated into these
contracts. Their repeated helper infrastructure and overlapping private feed tests are
replaced by public flat-count/callback contracts. This map describes
asserted behavior.
Position appends validate ordering; concat accepts owned leaves and sorted-stream
merging consumes owned Positions. Appended coordinates determine the stored
length and encoded payload. The standalone Miri harness imports the same position module and
its tests; historical storage experiments retain their separate native/Miri evidence.

The formatted line count includes test hooks, the complete reference oracle,
all default BPE public/helper test declarations and the Miri harness. Dedicated
word-count tests retain representation/order equality and full-u64 serde limits. Comments and blank lines are
excluded. The separately feature-gated parity trainer is unchanged and is not
used as a test helper for this engine.

The corpus-boundary refactor also exercises the actual owning preparation path:
`rejected_start_and_aa_tasks` verifies the original-position chunk budget and
global AA membership at 1/6 workers. `failures_before_and_after_writes` first
commits a round, then checks that both preparation and post-write index failures
preserve that history and trace while withholding new births. Production-grain
trace and resource evidence is recorded in the
[corpus-boundary experiment](../../../../../experiments/bpe-simplification/evidence/corpus-boundary-20261013/README.md).
