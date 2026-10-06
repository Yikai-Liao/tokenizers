# BPE training design

## Terms and identity modes

| Term | Meaning in this engine |
| --- | --- |
| worker | A thread executing a task in the training pool. Work stealing can run a logical owner's task on a different worker. |
| pair owner | A logical shard that owns pair state and count updates. The owner index is separate from the worker index. |
| pair count | The sum of word weights for the pair's occurrences. Engine pair, birth and removal operations check their numeric domains. |
| reuse ledger | The shared signed count for a pair in active-ID reuse mode. Selection stores the ledger's bits as `u64`. |
| priority snapshot | A cached queue value. Selection validates or repairs the snapshot before consuming its candidate. |
| candidate / cohort | A candidate owns a position list and a priority snapshot. Reuse can keep several independent cohorts for one pair key. |
| first activation | A new ID, or an existing reserved ID, becomes active for the first time in this attempt. |
| active-ID reuse | An already active ID becomes a merge result again. The engine restarts with a ledger and independent cohorts. |
| corpus slot | One fixed coordinate for a retained input symbol, or a word separator. Token spans count retained-symbol slots. |
| arena / allocation lease | The arena owns small allocated buffers. A lease grants temporary exclusive access to one worker's allocation cursor. |
| prepare / apply / commit | Prepare reads the corpus and builds writes/events. Apply writes the corpus. Commit updates and publishes index state. Each parallel phase joins before the next phase. |

`FirstActivationOnly` is the first-activation policy, also called fresh mode.
`AllowActiveReuse` is the active-ID reuse policy, also called reuse mode.
Existing fresh keys only lose count; reuse keys can revive through later births.

## Sequential semantics and token identity

A word contributes its weight to each adjacent token pair. In first-activation
mode, selection takes the largest positive pair count, with smaller token-ID pairs
winning equal counts. Applying a rule selects nonoverlapping matches from left
to right. Reuse mode preserves the reference cohort queue's frontier repair and
unsigned priority representation of its signed ledger, described below. Training
stops at the target vocabulary or when the mode's selection queue has no eligible
winner.

`Vocabulary` owns both text lookup and ID order in one insertion-ordered set.
Vocabulary initialization allocates retained alphabet IDs in codepoint order.
The unlimited-alphabet path marks observed characters active from a bitmap.
For limited alphabets or nonempty affixes, corpus planning marks observed symbols
active in the borrowed input's traversal order. The same traversal creates new
affix-decorated IDs. Character filtering uses original UTF-8 positions: dropping a
character does not redefine whether another character had a prefix or suffix.
One shared scanner implements that rule for initial edges and slot filling.
Within an attempt, inserting tokens never renumbers assigned IDs; an already
present merged string retains its ID. Training initializes a new vocabulary, and
public `train` replaces its model rather than preserving an old model's IDs.

Limited-alphabet truncation resolves frequency ties with the existing selector,
without a complete codepoint tie break. Retained characters are then inserted in
codepoint order. Decorated symbols receive IDs in the borrowed input's traversal
order: the caller's map for `do_train`, or the stored map/entries after `feed`.
Pair tie breaks and worker parity for one input view do not establish
identical initialization across arbitrary hash seeds.

A merged string can already exist. Activating a reserved ID for the first time
supports the ordinary first-activation algorithm, but reusing an active ID can
revive old keys and give one identity different occurrence spans.
`FirstActivationOnly` detects that case before accepting the rule, discards the
attempt, and rebuilds from unchanged input under `AllowActiveReuse`. The restart
retains the chosen alphabet. Rules accepted earlier in the batch may already have
changed vocabulary and span metadata; the entire attempt is abandoned before its
output is published. A pruned fresh index cannot switch in place to a reuse ledger.

The reuse path selects one birth cohort at a time, preserves intermediate births,
and maintains checked signed ledger updates. A candidate's stored count can be
stale; the index repairs it when it reaches the selection frontier. Cohorts that
have positive mass remain observable even below the current selection floor.
Priority snapshots store the signed ledger as `u64` bits, including negative
values; lazy frontier repair preserves this ordering. It does not provide the
decreasing-count upper bound used by fresh selection. Literal fixtures and
reference traces protect these details.

`PairIndex` owns shard state, lazy priority queues and the active identity policy.
First-activation keys cannot revive, so complete low-count keys can be retired.
Reuse needs a signed ledger and separately owned position cohorts. These states
stay private to the index. Routing changes ownership, never rule priority; the
precomputed reciprocal router produces the same owner as integer remainder.

## Order-preserving batches and complete producers

The coordinator takes an ordered prefix without skipping an incompatible
candidate. Rules may share a left token or share a right token. A crossed overlap
ends the prefix: a later rule's left token cannot equal an earlier rule's right
token, or vice versa. AA and reserved-ID rules run alone; active reuse selects one
rule at a time. Preparation checks selected neighbors so adjacent batch rewrites
emit the final newborn boundaries and checked removal mass in sequential order.

### Why the prefix keeps sequential priority

This proof sketch follows the local first-activation rules. Crossed-overlap checks
prevent an earlier rule from consuming
endpoints needed by a later accepted rule. Shared heads or shared tails alone
cannot overlap endpoints, so accepted old pair counts stay unchanged.

Each newborn boundary descends from an old boundary `(L, R)`. Its occurrences are
a subset of that boundary's occurrences and inherit their word weights, so a new
key's count cannot exceed the old boundary count. If that old key was selected
for a merge in the batch, the relevant head/tail combination would have failed
the crossed-overlap check. Otherwise its certified priority was below each later
accepted rule. When counts tie, replacing L or R by a newly appended ID increases
the lexicographic pair key, so the newborn boundary cannot win that tie ahead of
the accepted prefix. First activation gives each new identity one producer rule,
preventing births for unrelated identities from accumulating into a revived key.

For adjacent rules `(A, B) -> X` and `(C, D) -> Y`, the new boundary `(X, Y)`
descends from the old boundary `(B, C)`. The same bound applies when one or both
sides of that boundary are replaced.

Reserved IDs may be smaller than the replaced ID, so the tie-break step does not
apply to them; they end the batch and run alone. AA overlapping matches and active
ID reuse have their own single-rule semantics. The [batch tests](batch.rs) check
shared endpoints and the first crossed candidate, while [reference traces](tests/semantic_parity.rs),
[parallel ordering](tests/routing_and_publication.rs) and [identity fixtures](tests/identity_reuse.rs)
check the resulting rule order across ordinary, adjacent and reserved-ID cases.

### Producer coverage and AA overlaps

Whole-pair packing keeps ordinary candidates in one task and gives large lists
ordered spatial ranges. A complete rule task can aggregate every newborn key in
its neighbor directories, prune the complete count and encode its final position
list immediately. This shortcut also requires the job's node budget to fit.
Partial tasks emit chains for later owner reduction; they cannot independently
prune an incomplete birth. AA and identity reuse keep their dedicated paths.

AA positions overlap. Each chunk summarizes the parity of its trailing run;
ordered chunk summaries determine whether the next chunk skips its first match.
Workers then select their own starts. The coordinator processes summaries rather
than scanning every occurrence, while selection stays exactly left to right.

## Phase order and failure contract

The coordinator directly shows the round's lifetime boundaries:

```text
select → prepare → release candidates → apply → commit → release events
```

Selection consumes candidates, resolves vocabulary identities and prepares corpus
span metadata. At the end of selection, cached fresh prefixes return to their
heaps so commit can invalidate counts before refilling. Preparation reads the
stable corpus; all preparation tasks join before writes begin. Candidate lists
then have no readers and are released before
commit allocates the next generation of positions.

`PreparedMerges` owns write plans and neighbor events. Its consuming `apply` holds
a mutable corpus borrow until parallel writes join, preventing a second use of
that plan value and safe concurrent corpus access. Ordinary jobs own disjoint
endpoint spans; occurrence-span jobs own complete whole-word regions. The type
does not bind a plan to a corpus instance or version: the coordinator must preserve
the preparation snapshot through apply.

Commit borrows position chains owned by the returned events. Events stay alive
until every owner finishes; selection resumes only after commit joins. Preparation
errors return no plan to apply. Commit errors may follow corpus writes and partial
shard updates. Errors abandon the attempt and release its state; the failed index
cannot be resumed and a round has no transactional rollback.

At successful completion, position owners are dropped before their allocation
arena. Corpus and worker scratch are freed before public model strings are built.
An active-ID restart also drops all attempt state and reuses only the original
input and chosen alphabet.

## Counts and publication

### Counts, positions and queue snapshots

| Value | First activation | Active reuse |
| --- | --- | --- |
| Position-list length | Number of recorded addresses, including stale ones | Number of addresses owned by one cohort, including stale ones |
| Count | Weighted pair mass, which only decreases for an existing key | Shared signed pair ledger, updated in original action order |
| Cached priority | A decreasing-count upper bound until `best_first_activation` certifies it | The shared ledger's `u64` bits when the cohort snapshot was made; repaired only at the global head by `best_active_reuse` |

The reuse queue's selection test accepts nonzero unsigned ledger bits at or above
`min_frequency`; a negative signed value has a large unsigned representation.
New cohort publication instead requires a positive signed ledger, and retains
positive values below the selection floor. The local negative-ledger fixture
protects this distinction; it is an index-level compatibility case, not evidence
that a public training input reproduced an error.

Checked arithmetic covers engine pair counts, births and removals. `feed` and
limited-alphabet frequency accumulation retain ordinary additions. A nonempty
prefix/suffix, or a reuse attempt, bounds both the maximum word weight and initial
weighted edge mass by `i64::MAX`. Empty affixes use the plain path. Plain
first-activation input can have total mass above `u64::MAX` when every individual
pair count still fits in `u64`.

### Complete birth handoff and owner work

The complete-producer path returns `CompletedBirth { key, weight, positions }`.
`weight` includes every contributing occurrence; the compressed list borrows the
training arena rather than the encoding lease. The same neighbor-directory drain
filters and encodes these results, keeping the original removal events. It emits
no routed birth chains for them. Preparation knows neither index state layout nor
ledger bits. Only commit constructs `PairState` and candidate priorities.

Each owner task performs these operations in order:

1. Stably group birth metadata without altering routed count actions.
2. Apply original-order actions, with removal before birth for `Both`. Fresh
   subtraction checks `u64` and permanently retires low counts; reuse checks each
   signed `i64` update. Summing removals and births separately would change
   intermediate overflow and error behavior.
3. Publish already encoded complete births by moving their lists into fresh state.
4. Reduce the remaining birth fragments, then encode and publish each complete key.
   Partial producers must be summed before applying the fresh floor. Reuse uses
   the updated ledger and retains positive cohorts below the selection floor.
5. Refill the fresh owner's candidate prefix before return, including when the
   owner has count changes or complete births but no routed births.

One fragment vector is reused across keys and rule/direction buckets. Fresh
fragments follow disjoint spatial runs and use direct reverse traversal. Reuse
left/right chains can interleave, so its encoder merges actual overlaps.
Zero-weight births with nonempty position chains remain routed: weight alone
does not determine occurrence ownership. Completely untouched owners skip this
work and refill lazily at selection. All operations share the existing joined
owner phase. The fragment vector avoids a separate intermediate vector per key.
Each retained key still allocates backing storage when its encoded list needs it.

## Fixed coordinates and delayed construction

`CorpusPlan` sorts words by descending weight, measures retained symbols, assigns
physical coordinates and records checkpoints for long UTF-8 words. It remains
immutable while initial grouping reads the original strings. Initial waves own
left endpoints; their final edge reads one symbol of lookahead. Both checkpoint
seeking and ordinary scans use the same symbol interpretation. Checkpoints map
original UTF-8 byte offsets to retained-symbol slot coordinates; filtered ranges
can produce repeated slot coordinates. This delayed allocation trades extra
UTF-8 decoding and ID lookup for reduced overlap between raw records and mutable
slots.

Only after raw initial records retire does the plan allocate a mutable corpus.
Each word has a separator; each live token stores its ID at its first and last
slots. Merges replace endpoints without moving the suffix. Token spans locate the
next live token, and the preceding endpoint locates the previous one. Old pair
positions can remain in compressed lists; a matcher checks the current endpoints
before interpreting one as a live occurrence.

A statically selected slot type uses 16, packed 24 or 32 bits according to the ID
domain, reserving a separator code. Packed 24-bit reads have an initialized guard
slot and require the joined read/write protocol. First-activation spans normally
come from the token-ID table. Active reuse materializes occurrence spans before
rewriting unequal occurrences, then schedules whole-word write regions.

The internal birth span gate is strict `< max_token_length`, measured in corpus
slots. Decorations affect token strings and IDs, not those physical spans. Initial
pair counting bypasses this gate, and applying a selected pair has no separate
length check. A limit below two can therefore still admit an initial two-symbol
merge. The setting is a birth admission gate, not a uniform cap on output token
strings; spans exclude affix text, filtered characters and UTF-8 byte widths.

## Initial grouping and pair ownership

Bounded waves emit twelve-byte records with a complete `u64` pair key and a local
coordinate offset. Stable radix grouping preserves physical order within each
key; the wave base restores full-width positions. Counting uses interval weight
runs and a unit-weight shortcut. Initial owners can build compressed lists in
parallel, borrowing independent allocation and encoding resources from whichever
pool worker executes the task.

## Private storage and worker resources

BPE is the sole consumer of these facilities, so `storage` is private to the
engine. Coordinate buffers and linked nodes share low halves and promote to a
separate high plane only when required. A buffer is a specialization of that
storage, with no independent forwarding wrapper. Linked chains add bounded local
node references and reverse traversal.

`SortedPositions` uses inline small cases and delta encoding with unsigned
LEB128 gaps and periodic restart points. Each compressed group holds at most 128
positions: an eight-byte absolute seed followed by gaps from preceding positions.
A restart-offset directory lets range cursors replay at most 127 gaps. The codec's
base 128 and the 128-position restart interval are independent choices. This is a
private in-memory layout, not a model serialization format. Append measures and
checks the suffix before publishing a new count, preserving the previous readable
prefix on failure. The allocator uses worker-exclusive bump cursors below a fixed
cutoff; large allocations remain individually owned. Published lists borrow the
arena, not the temporary lease. Arena and heap ownership tags determine cleanup.

`Execution` owns the pool and reusable directories and codec scratch. Task-level
closures acquire and return directories, resetting touched IDs on success and
errors. Unwinding drops accumulator values and leaves empty reusable state.
Encoding resources follow executing worker IDs, which differ from logical pair
owners. Arena-backed containers use `'arena` for the same storage lifetime;
`PositionCursor` separately borrows a list as `'list`, event fragments borrow
chunks as `'events`, and preparation shares short borrows as `'prep`. These are
compile-time relationships, with no runtime reference count or stored reference
in `PhantomData`. A temporary `AllocationLease` grants exclusive cursor access;
published positions borrow the arena and can survive release of that guard.
No task may nest pool work while holding worker resources. This avoids
reentry into its own lock. The final joined phase releases scratch, corpus and
position allocations before constructing public model strings.

## Algorithm sources and local costs

The integer gap encoding is unsigned LEB128 (base-128 varints), as described by
[Protocol Buffers' integer encoding documentation](https://protobuf.dev/programming-guides/encoding/#base-128-varints).
That reference defines the integer encoding; it is neither a source-code port nor
the source of the complete restart-list design.

`storage/radix.rs` is a Rust port of Robert Clausecker's BSD-2-Clause
[`radixsort_permuted.c`](https://github.com/clausecker/radsort/blob/f69e816c3cd79d312cd67aea5b9cf1c338c1b371/radixsort_permuted.c),
at revision `f69e816c3cd79d312cd67aea5b9cf1c338c1b371`. Its complete copyright and
license remain in the source; the reference
[COPYING](https://github.com/clausecker/radsort/blob/f69e816c3cd79d312cd67aea5b9cf1c338c1b371/COPYING)
and [Clausecker and Schintke paper](https://arxiv.org/abs/2607.05302)
provide the original attribution. Local adaptation sorts full 64-bit keys in
12-byte records and retains stable incoming payload order. The fixed 512-record
block uses 3 MiB scatter scratch plus nine bytes of metadata per input block and
fixed directory overhead. Metadata therefore grows with n/512; the paper's
square-root space bound does not apply to this fixed-block parameterization.

Packed slots and compression add representation logic in exchange for memory
savings. Deferred construction reduces peak overlap with initial records. Arena
reuse avoids repeated small allocations. Complete producers avoid routing birth
chains and owner-side re-encoding, while split rules still require aggregation
within the same phase joins. The radix sorter keeps bounded scatter scratch and
its stable full-key contract. Rayon handles scheduling; the engine has no
experimental environment switches or phase-diagnostic subsystem.

The [engine guide](README.md#correctness-checks) gives correctness test commands
and the measurement boundaries. Repeated measurements describe fixed inputs,
compiler settings, worker counts and affinity. `feed`, public `do_train` and the
end-to-end boundary identify different costs; changes to input collection require
separate attribution. Source-specific results, historical snapshots and unused
experiments belong with benchmark records rather than this implementation contract.
