# Engine contracts and simplification boundaries

## Joined rounds and ownership

The coordinator owns the vocabulary, corpus and pair index. A selected `Batch`
owns the occurrence lists of its rules. `Batch::prepare` consumes those lists,
returns owned writes and neighbor changes, and joins all readers before return.
`Prepared::apply` joins endpoint writers before handing changes to `PairIndex`.
The index directly publishes complete ordinary births, reduces partial births,
and updates count owners in parallel.

Errors discard the attempt. Reader and owner jobs finish before their borrowed
state can be dropped. Commit can fail after writes and partial count updates;
rounds do not promise rollback. Tokens and occurrence spans use ordinary relaxed
atomic accesses. The only unsafe block is an x86_64 cache-hint intrinsic whose
address comes from a checked access to the live borrowed slot allocation. Other
architectures use a no-op hint. There is no unsafe endpoint or allocation protocol.
Atomics do not replace the disjoint-match and joined-phase semantic requirements.

## Token identity and initialization

Token strings are stored once in an insertion-ordered vocabulary. Insertion
indices are u32 IDs; `u32::MAX` is reserved for separators. Special tokens are
inserted first. Plain alphabets are ordered by codepoint. Limited alphabets keep
the existing frequency selector, including its unspecified ties. Decorations
are resolved in the input view's traversal before weighted words are reordered.
Filtering and first/last decoration flags use original UTF-8 word coordinates.

Each attempt starts from original weighted words. The selected alphabet is
retained through a reuse restart. A reserved but inactive result keeps its ID
and runs alone. Selecting an already active result during a fresh attempt causes
an immediate restart; traces and merges from that attempt are abandoned.

The input view is borrowed and unchanged. Feed aggregation, serialization and
public thread/parallelism settings remain outside the engine. Training creates
one requested-size Rayon pool and installs all its stages there, independently
of any ambient pool used by feed.

## Construction and task-local aggregation

A borrowed CorpusPlan resolves original symbols and weighted word intervals.
Initial counting directly builds compressed positions before allocating token
slots. Materialization consumes the plan and releases thin input references.
Fresh mode releases its per-word start directory; weights retain only adjacent
equal-weight regions. Reuse keeps word starts for cohort scan domains.

Preparation uses Rayon map_init directories scoped to each job. Each side maps
neighbor IDs to u32 change-entry indices; touched slots are reset before reuse.
These indices do not represent positions. There is no shared scratch lock or
assumption relating Rayon worker indices to an external directory array.

## Fixed-coordinate corpus

Words occupy disjoint original-symbol intervals separated by sentinel slots.
A logical token stores its ID at its first and last retained-symbol coordinates.
Its span skips interior holes. Initially spans are one for active IDs. Fresh
identities have a single span, so one ID-to-span table describes the corpus.
When an active alias acquires unequal spans, the corpus initializes a separate
occurrence-span plane from the current logical words. All subsequent geometry
reads the occurrence endpoints; it does not infer length from token text.

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

Newborn mass cannot outrank its producer's selected count. New IDs follow old
IDs in ties. Reserved identities lack that tie bound and run alone. AA selection
runs alone and greedily accepts valid starts from left to right, with each match
excluding the next overlapping edge. Preparation and application of the accepted
AA matches remain parallel; the selection itself is serial.

## Counts and occurrence publication

Fresh owners hold one count and one list per retained pair. Taking a candidate
removes that state. Old-boundary removal events decrease existing states and
retire counts below the floor. Each new pair belongs to one producer rule;
partial jobs aggregate before pruning and publication. An ordinary source that
covers the whole candidate marks its birth count complete. After local encoding
it drops births below the floor while preserving removal events; the owner
moves retained lists directly into states and the queue. AA and reuse do not
enter this shortcut. Small ordinary candidates remain whole by item count; large
candidates use spatially ordered block ranges. Fresh local lists are
delta encoded during preparation and streamed to each owner in indexed task
order. Each key has one producer, so this preserves spatial order without another
sort. Reuse births can interleave; their full-u64 coordinates are sorted before
encoding. There is no adaptive birth feedback, linked-node promotion or Arena.

Commit retains one immutable metadata array and owner routes with reusable
capacity. Each route action is a resident usize record reference and two flags.
Only nonzero removals and nonempty births route; zero-weight births with
positions still route. Each position stream moves to exactly one owner payload
vector, in the order of that owner's birth actions. Removal precedes birth for
each action; owner order is the original producer order. Draining both vectors
retains capacity. Errors drop active drains; the attempt is discarded after
owner jobs join. The next commit clears any unprocessed route before reuse.

Arena's bulk retirement and end-of-attempt release benefits are deliberately
forgone at the user's request. Per-list Vec ownership keeps lifetimes explicit
but leaves allocation/retirement costs in preparation, commit and final release.

Reuse owners retain a signed i64 ledger and the queue owns independent occurrence
cohorts. Selection repairs the head cohort against the shared ledger, preserving
the reference's unsigned ordering of signed count bits. Positive ledger values
publish birth cohorts even below the selection floor. Removal precedes birth
for each aggregated neighbor change, and left/right drain ordering is retained.

A reuse cohort scans positions until alias reuse or a configured length gate
requires scanning complete represented words. Whole-word scans may find matches
outside the historical position list; therefore they are enabled only under the
same conditions as the baseline. Logical preceding tokens include earlier
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

Each block contains at most 128 positions. Its first position is a full-u64
restart; later positions store unsigned base-128 deltas. Checked monotonic input
makes reconstruction exact, including a gap of `u64::MAX`. Zero gaps retain
repeated coordinates. Block directories contain resident byte offsets, never
truncated corpus coordinates. A task reads independent complete blocks; AA and
cohort selection can stream the full list without a corpus-sized bitmap.

## Evidence and budget

The baseline is Fork main `e4f787dc189d9be7192107490d652096cde7480e`.
The user requires at most 2000 formatted production logic lines and no more test
logic than production, preserving compatible batch aggregation and good module
boundaries. Performance work prioritizes 4-core Chinese and English ByteLevel.
The earlier ablation plan supplies behavioral coverage, but its provisional
3%/RSS thresholds were replaced by the hard source-size constraint.

Full-model comparisons cover vocabulary IDs and the complete ordered merges;
small fixtures also compare every `(pair, count, replacement ID)`. Tests alone
are not performance evidence. The experiment records independent-process paired
wall time, CPU time, RSS, swap, build/input hashes and model equality. A/A measures
host noise before candidate comparisons. The current KVM host is not the original
plan's fixed-frequency laptop, and that limitation accompanies every result.

## Attribution

The endpoint representation and compatible batching retain the algorithmic
lineage documented by the original engine: Yikai Liao's efficient BPE prototypes,
BatchBPE and YouTokenToMe's conditional rule pipeline. The custom BSD radix-sort
translation has been removed; this engine uses standard sorting and a small
safe delta stream. Git history and the pinned baseline preserve the original
implementation and its detailed proofs.
