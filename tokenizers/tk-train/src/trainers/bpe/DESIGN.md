# BPE training contracts and simplification boundaries

## Joined rounds and ownership

The private `train` coordinator in `mod.rs` owns the vocabulary, corpus, pair index
for one attempt. Public inputs and worker policy enter through
`BpeTrainer::train_counts`; tests exercise the same coordinator with explicit workers.
A selected `Batch`
owns the occurrence lists of its rules. `Batch::prepare` consumes those lists,
returns owned writes and neighbor changes, and joins all readers before return.
`Prepared::apply` joins endpoint writers before handing changes to `PairIndex`.
The index directly publishes complete ordinary births, reduces partial births,
and updates count owners in parallel.

Errors discard the attempt. Reader and owner jobs finish before their borrowed
state can be dropped. Commit can fail after writes and partial count updates;
rounds do not promise rollback. Tokens and occurrence spans use ordinary relaxed
atomic accesses. Corpus cache hints use `branches` with addresses from checked
accesses to live slots. Stable x86_64 emits L1 prefetch and stable AArch64 emits
`prfm pldl1keep`; unsupported targets retain the library fallback. Position
storage uses inline values or owned Box allocations and checked slices, with automatic Send/Sync.
Atomics do not replace the disjoint-match and joined-phase semantic requirements.

## Token identity and initialization

Token strings are stored once in an insertion-ordered vocabulary. Insertion
indices are u32 IDs; `u32::MAX` is reserved for separators. Special tokens are
inserted first. Plain alphabets are ordered by codepoint. Limited alphabets keep
the existing frequency selector in `vocabulary.rs`, including its unspecified ties.
The independent reference shares only this selector. Decorations
are resolved in the input view's traversal before weighted words are reordered.
Filtering and first/last decoration flags use original UTF-8 word coordinates.

Vocabulary constructs initial ID spans (zero for unactivated, one for activated) and
moves the table into CorpusPlan. After materialization Corpus alone records ID
activation. Resolving a merge string returns its ID; the corpus determines
whether that ID has been activated and prepares its geometry. Vocabulary keeps
no activation mirror that must be updated during merges.

Each attempt starts from original weighted words. The selected alphabet is
retained through a reuse restart. A reserved but unactivated result keeps its ID
and runs alone. Selecting a previously activated result during a fresh attempt causes
an immediate restart; traces and merges from that attempt are abandoned.

The input view is borrowed and unchanged. Feed aggregation, serialization and
public thread/parallelism settings are handled by the trainer entry. Training creates
one requested-size Rayon pool and installs all its stages there, independently
of any ambient pool used by feed.

## Construction and task-local aggregation

A borrowed CorpusPlan resolves original symbols and weighted word intervals.
Initial counting gathers sorted temporary positions, reduces counts and freezes
retained lists before allocating token slots. Materialization consumes the plan
and releases thin input references.
Fresh mode releases its per-word start directory; weights retain only adjacent
equal-weight regions. Reuse keeps word starts for cohort scan domains.
`InitialPairCounts` owns the dense-versus-sparse counting choice and its domain.
`CohortPreparation` owns one rule's writes and neighbor aggregation through scanning
and publication. Explicit Empty/Unique/Shared head states replace the shared-head
separator marker without changing exact pair fallback.

Preparation uses Rayon `map_init` to reuse private directories within each
parallel task. Each side maps neighbor IDs to u32 change-entry indices; touched slots are reset before reuse.
These indices do not represent positions. The directories have no shared scratch lock or
assumption relating Rayon worker indices to an external directory array.

## Fixed-coordinate corpus

Words occupy disjoint original-symbol intervals separated by sentinel slots.
A logical token stores its ID at its first and last retained-symbol coordinates.
Its span skips interior holes. Initially spans are one for activated IDs. Fresh
identities have a single span, so one ID-to-span table describes the corpus.
When an activated alias acquires unequal spans, the corpus initializes a separate
occurrence-span plane from the current logical words. All subsequent geometry
reads the occurrence endpoints; it does not infer length from token text.
The ID table still grows with the vocabulary and marks every activated ID as
nonzero, including previously unactivated reserved IDs. Once the occurrence plane
exists, new ID metadata only needs an activation marker; it no longer sums
representative ID spans. Activation means that an ID has ever been used, even
after its last occurrence is consumed.

Lists store u64 coordinates. A physical allocation still has resident usize and
isize bounds, enforced before construction. The position codec does not narrow
coordinates to u32 and can round-trip `u64::MAX` independently of allocation size.

## Compatible batch proof

Pair priority is decreasing frequency, followed by ascending token-ID pair.
In fresh mode old counts only decrease and a new result ID cannot occur in old
keys. Selection accepts an ordered prefix, never skipping a conflicting winner.
Multiple rules may share a head or a tail, but a selected tail cannot be another
selected head. Thus their matches and endpoint writes are disjoint. Neighbor
preparation accounts for adjacent selected rules as their final replacements.

Each newborn boundary descends from an old boundary key. Its occurrences are
a subset of that key's occurrences and retain their word weights, so its total
mass cannot exceed the witness's count. The witness cannot be an accepted rule:
merging either side would cross a selected endpoint. Since selection takes a
prefix without skipping a conflicting winner, the witness cannot precede any
later accepted rule. A witness below the floor also keeps its newborn below it.

For equal counts, replacing either witness endpoint with a newly appended ID
increases the pair key. Each appended ID names one producer rule, so all tasks
for a newborn key share this same witness; their combined mass obeys the bound.
Together, the frequency and tie bounds prevent newborns from overtaking the
accepted prefix. Adjacent selected matches emit their shared boundary once,
from the left match, using both final replacements.

Reserved identities lack the appended-ID tie bound and run alone. A previously
activated ID collision restarts the whole attempt in reuse mode before the
conflicting batch is applied.
AA selection also runs alone and greedily accepts valid starts from left to right,
with each match excluding the next overlapping edge. Preparation and application
of accepted AA matches remain parallel; the selection itself is serial.

## Counts and occurrence publication

Pair ownership uses the existing `ahash` dependency to hash the ordered token-ID
pair, then reduces it modulo the owner count. Fixed zero seeds make initial
counting, queue lookup and commit use the same mapping within a build, without
per-call randomness. This replaces a local multiplier/rotation heuristic; hash
outputs and owner indices are internal and are not stored in the model format.

`OwnerCommit` encapsulates one owner's sequential commit work. It borrows the count
shard, fixes the reuse/frequency policy, and owns partial birth groups, pending
candidates. Its commit method drains
the route in order, removes old weight before recording each birth, and consumes
the work state when publishing groups. The coordinator receives only completed
candidates after owner jobs join. Errors release unpublished storage and discard
the attempt; the object does not roll back already-applied count changes.

Owners hold counts; one global Candidate priority queue owns fresh lists and
reuse cohorts. Taking a fresh candidate removes its count key and queue record.
Old-boundary removal events decrease existing counts and delete keys below the
floor. Their lists survive until the stale queue records reach the head or the
attempt ends. Selection repairs counts and discards missing fresh keys through
the same queue loop used for reuse ledgers. Each new pair belongs to one producer rule;
partial jobs aggregate before pruning and publication. An ordinary source that
covers the whole candidate marks its birth count complete. It first drops births below the floor while preserving removal
events, then encodes retained lists; the owner publishes the count and returns an
owning Candidate. Joined owner results are pushed into the global queue serially.
AA and reuse do not
enter this shortcut. Small ordinary candidates remain whole by item count; large
candidates use spatially ordered block ranges. Partial fresh lists remain raw until owner reduction; complete
fresh lists are encoded during preparation. Both move to each owner in indexed
task order. Each key has one producer, so this preserves spatial order without
another sort. Reuse births can interleave; their full-u64 coordinates are sorted before
encoding. There is no adaptive birth feedback or linked-node promotion.

Commit retains one immutable metadata array and owner routes with reusable
capacity. Each route action is a resident usize record reference and two flags.
Only nonzero removals and nonempty births route; zero-weight births with
positions still route. Each position stream moves to exactly one owner payload
vector, in the order of that owner's birth actions. Removal precedes birth for
each action; owner order is the original producer order. Draining both vectors
retains capacity. Errors drop active drains; the attempt is discarded after
owner jobs join. Successful commits drain every route before the next round.

Published lists use an Enum with Empty, One and Two inline cases and an owned
`Box<[u8]>` for longer restart/delta streams. The Enum is 24 bytes on the measured
64-bit target. Each list releases its bytes when its queue, fragment or birth
owner retires. Ownership and Send/Sync follow from its fields.

`Positions::from_sorted` accepts a nondecreasing u64 slice; `Positions::concat`
consumes ordered fragments, drops empty fragments, and returns a sole nonempty
fragment unchanged. Multiple fragments are encoded in order, preserving
duplicates and rejecting descending coordinates. Construction writes directly
into an owned Vec, fills restart offsets in its reserved prefix, and converts
it to Box. Larger gaps can grow Vec and boxing may copy or reallocate; one
encoding pass does not promise one allocation. Readers borrow immutable bytes
and carry private decoder state. No encoder service, lease or scratch lifetime
is carried through the index or preparation objects.

Reuse owners retain a signed i64 ledger and the queue owns independent occurrence
cohorts. Selection repairs the head cohort against the shared ledger, preserving
the reference's unsigned ordering of signed count bits. Positive ledger values
publish birth cohorts even below the selection floor. Removal precedes birth
for each aggregated neighbor change, and left/right drain ordering is retained.

A reuse cohort scans positions until alias reuse or a configured length gate
requires scanning complete represented words. Whole-word scans may find matches
outside the historical position list. Logical preceding tokens include earlier
matches in that word, retaining intermediate births and removals before writes.

## Numeric and length rules

Initial pair, removal and birth arithmetic is checked in u64. Nonempty affixes
and reuse also require maximum input word weight and initial weighted edge mass
to fit i64. Reuse ledger additions/subtractions are checked in i64. Plain fresh
training has no global u64 mass cap; distinct keys can each carry `u64::MAX`.
Initial counts are checked even when the initial vocabulary reaches the target.
Feed and limited-alphabet accumulation retain their existing ordinary additions.

`max_token_length` is a strict admission gate for newborn boundary spans in
retained-symbol coordinates. Initial pairs bypass it, and the selected match
itself has no additional gate. Decorated string byte length does not define this
span. Limits zero, one and two therefore retain their existing distinct behavior.

## Position encoding

Temporary lists use library `SmallVec<[u64; 2]>` directly. Every coordinate
retains its full width, with two values inline before heap storage is needed.
Fresh producers visit coordinates in order and can move the first partial
fragment to its owner; subsequent fragments append in producer order. Reuse
groups retain an unordered Vec and sort before construction. The immutable
constructor validates the resulting nondecreasing sequence.

The read interface exposes full iteration, `iter_from_value(coordinate)` and
borrowed `chunks(target_items)`. Chunks round their target up to restart boundaries
inside Positions and expose only `len()` and `iter()`. They neither allocate
per-fragment storage nor decode ahead of iteration. Merge preparation stores the
borrowed chunk instead of a range of compression blocks. An ordinary producer
is complete when its actual chunk length equals the candidate length, including
cases where alignment produces one chunk larger than the target. Filtered AA
starts remain partial even when a single slice contains them all.

Each block contains at most 128 positions. Its first position is a full-u64
restart; later positions store unsigned base-128 deltas. Checked monotonic input
makes reconstruction exact, including a gap of `u64::MAX`. Zero gaps retain
repeated coordinates. Block directories contain resident byte offsets, never
truncated corpus coordinates. A task reads independent complete blocks; AA and
cohort selection can stream the full list without a corpus-sized bitmap.

## Validation

Full-model comparisons cover vocabulary IDs and complete ordered merges. Small
fixtures also compare every `(pair, count, replacement ID)` with an independent
sequential reference. Literal expectations cover numeric limits and alias
cohorts; storage tests and Miri exercise immutable storage, concurrent readers,
owned concatenation, and exact restart/delta bytes when the output Vec grows.
See [the test coverage map](COVERAGE.md).

Performance evidence belongs to the experiment report: paired independent
processes record wall time, CPU time, RSS, swap, build/input hashes and model
equality. Timing results include the host and observed A/A variation.

## Attribution

The endpoint representation and compatible batching retain the algorithmic
lineage documented by the original engine: Yikai Liao's efficient BPE prototypes,
BatchBPE and YouTokenToMe's conditional rule pipeline. The custom BSD radix-sort
translation has been removed; this engine uses standard sorting and a small
delta stream with immutable owned byte storage.
The invariants above describe the safety and ordering requirements used here.
