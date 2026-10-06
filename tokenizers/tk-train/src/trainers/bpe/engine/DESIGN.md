# BPE training design

## Sequential semantics and token identity

A word contributes its weight to each adjacent token pair. The next rule has the
largest count, with smaller token-ID pairs winning equal counts. Applying a rule
selects nonoverlapping matches from left to right. Training stops when the target
vocabulary is reached, no positive eligible candidate remains, or the best count
is below `min_frequency`.

`Vocabulary` owns both text lookup and ID order in one insertion-ordered set.
Initial characters and affix-decorated symbols activate IDs in original word-map
traversal order. Character filtering uses original UTF-8 positions: dropping a
character does not redefine whether another character had a prefix or suffix.
One shared scanner implements that rule for initial edges and slot filling.

A merged string can already exist. Activating a reserved ID for the first time
supports the ordinary first-activation algorithm, but reusing an active ID can
revive old keys and give one identity different occurrence spans.
`FirstActivationOnly` detects that case before accepting the rule, discards the
attempt, and rebuilds from unchanged input under `AllowActiveReuse`. The restart
retains the chosen alphabet. It never publishes a speculative batch or switches
an already pruned index into a reuse ledger.

The reuse path selects one birth cohort at a time, preserves intermediate births,
and maintains checked signed ledger updates. A candidate's stored count can be
stale; the index repairs it when it reaches the selection frontier. Cohorts that
have positive mass remain observable even below the current selection floor.
Literal fixtures and reference traces protect these details.

## Fixed coordinates and delayed construction

`CorpusPlan` sorts words by descending weight, measures retained symbols, assigns
physical coordinates and records checkpoints for long UTF-8 words. It remains
immutable while initial grouping reads the original strings. Initial waves own
left endpoints; their final edge reads one symbol of lookahead. Both checkpoint
seeking and ordinary scans use the same symbol interpretation.

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
pair counting and later birth admission keep their distinct existing semantics.

## Initial grouping and pair ownership

Bounded waves emit twelve-byte records with a complete `u64` pair key and a local
coordinate offset. Stable radix grouping preserves physical order within each
key; the wave base restores full-width positions. Counting uses interval weight
runs and a unit-weight shortcut. Initial owners can build compressed lists in
parallel, borrowing independent allocation and encoding resources from whichever
pool worker executes the task.

`PairIndex` owns shard state, lazy priority queues and the active identity policy.
First-activation keys cannot revive, so complete low-count keys can be retired.
Reuse needs a signed ledger and separately owned position cohorts. These states
stay private to the index. Routing changes ownership, never rule priority; the
precomputed reciprocal router produces the same owner as integer remainder.

## Certified batches and complete producers

The coordinator takes an ordered prefix of rules whose heads and tails do not
conflict. AA and reserved-ID rules end the batch. The accepted prefix preserves
sequential rule order. Preparation checks selected neighbors so adjacent batch
rewrites emit the same final newborn boundaries and checked removal mass.

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

`PreparedMerges` owns a read-only preparation result. Its consuming `apply` method
begins endpoint writes after all preparation readers join. Writes belong to
separate selected occurrences or whole-word regions. After applying, event records
route removals in original order and group births stably on each count owner.
Owners publish complete counts, position lists and priorities before the next
selection phase. The joins are part of correctness, not optional scheduling.

## Private storage and worker resources

BPE is the sole consumer of these facilities, so `storage` is private to the
engine. Coordinate buffers and linked nodes share low halves and promote to a
separate high plane only when required. A buffer is a specialization of that
storage, with no independent forwarding wrapper. Linked chains add bounded local
node references and reverse traversal.

`SortedPositions` uses inline small cases and G128 gap encoding with restart
indexes. Range cursors replay at most 127 preceding deltas. Append measures and
checks the suffix before publishing a new count, preserving the previous readable
prefix on failure. The allocator uses worker-exclusive bump cursors below a fixed
cutoff; large allocations remain individually owned. Published lists borrow the
arena, not the temporary lease. Arena and heap ownership tags determine cleanup.

`Execution` owns the pool and reusable directories and codec scratch. Task-level
closures acquire and return directories, resetting touched IDs on success and
errors. Unwinding drops accumulator values and leaves empty reusable state.
Encoding resources follow executing worker IDs, which differ from logical pair
owners. No task may nest pool work while holding worker resources. This avoids
reentry into its own lock. The final joined phase releases scratch, corpus and
position allocations before constructing public model strings.

## Costs and evidence

Packed slots and compression add representation logic in exchange for memory
savings. Deferred construction reduces peak overlap with initial records. Arena
reuse avoids repeated small allocations. Complete producers reduce redistribution
and encoding copies, while large split rules still need aggregation and barriers.
The radix sorter retains its bounded block scratch and stable full-key contract.
Released Rayon handles scheduling; the engine has no experimental environment
switches or phase-diagnostic subsystem.

The external benchmark repository compares the unchanged Best Multicore snapshot
with this implementation using identical inputs, compiler settings and affinity.
It keeps historical idle-policy experiments separate from the production library.
Time and HWM comparisons include the public `do_train` boundary; canonical output
validation follows timing. Local results do not establish performance on every
machine or corpus. Tests and measured evidence are linked from the engine guide.
