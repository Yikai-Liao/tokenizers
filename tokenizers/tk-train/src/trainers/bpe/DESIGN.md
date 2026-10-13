# BPE training contracts and simplification boundaries

## Joined rounds and ownership

`BpeTrainer::do_train_impl` in `mod.rs` owns the dedicated pool, progress and retry
flow. `do_train` and `train_vocab` borrow a `WordCountsView` and enter it directly;
the Trainer implementation adapts `train_vocab` results into a replacement model.
The private `train_attempt` owns the vocabulary, corpus and pair index for one attempt.
Tests exercise this same coordinator with explicit workers and inspect its successful
Vocabulary, whose trace field exists only under `cfg(test)`. No test callback enters
the production call chain.
`BpeTrainer::merge_next` in `merge.rs` selects a compatible priority prefix,
resolves and activates identities, then publishes the complete round. It moves
`SelectedMerge { pair, merged_id, positions }` into `Corpus::merge`, commits
returned changes to the index, and appends ordered history to the vocabulary.
Its result is Finished, RestartForReuse, or Committed with the rule count. The
trainer uses that count for progress. Selection scratch retires before preparation.

The private `corpus/merging.rs` owns preparation and endpoint application.
Consuming preparation releases selected streams, task iterators and shared lookups
before writes. All readers join before application; all writers join before index
publication. Pair records and test-only priority scalars are captured before moving
selected payloads, retaining the certified count and historical cohort order.
One record allocation holds at most 256 pairs in fresh mode and one in reuse mode.

Selection may mutate identities and activation before preparation. Errors discard
the attempt. Preparation errors leave endpoints unchanged; index errors can follow
endpoint writes and partial count updates. Successful history and trace publication
occurs only after index commit succeeds.

Reader and owner jobs finish before their borrowed state is dropped. Tokens and
occurrence spans use ordinary relaxed atomic accesses. Corpus cache hints use `branches` with addresses from checked
accesses to live slots. Stable x86_64 emits L1 prefetch and stable AArch64 emits
`prfm pldl1keep`; unsupported targets retain the library fallback. Position
storage uses inline or heap-owned SmallVec bytes and immutable boxed leaf arrays,
with checked slices and automatic Send/Sync.
Joined phases and disjoint matches keep endpoint reconstruction valid; relaxed
atomics provide safety for each individual access.

## Token identity and initialization

Token strings are stored once in an insertion-ordered vocabulary. Insertion
indices are u32 IDs; `u32::MAX` is reserved for separators. Special tokens are
inserted first. Plain alphabets are ordered by codepoint. Limited alphabets keep
the existing frequency selector in `vocabulary.rs`, including its unspecified ties.
The independent reference shares only this selector. Decorations
are resolved in the input view's traversal before weighted words are reordered.
Filtering and first/last decoration flags use original UTF-8 word coordinates.

Vocabulary initialization returns both the identity table and completed
InitialTokenIds, whose spans mark unactivated IDs as zero and activated IDs as one.
CorpusPlan takes ownership of these completed IDs.
After symbol writes join, `into_metadata` moves the activation/span Vec and releases both lookup
tables; Corpus alone then records ID activation. Resolving a merge string returns
its ID and whether it already existed; the corpus determines
whether that ID has been activated and updates its span metadata.

Merge resolution uses IndexSet::insert_full for lookup and insertion together.
Initialization retains its borrowed string lookup before constructing an owned
string. The returned insertion index is checked for u32 and separator bounds;
failure discards the attempt, including any token inserted before that check.

Each attempt starts from original weighted words. The selected alphabet is
retained through a reuse restart. A reserved but unactivated result keeps its ID
and runs alone. Selecting a previously activated result during a fresh attempt causes
an immediate restart; vocabulary, index, corpus, traces and merges from that attempt
are abandoned. The core permits one fresh-to-reuse switch. A restart request from
the reuse attempt is an internal invariant error. The successful attempt returns
the Vocabulary with committed ID rules and its test-only trace.
The public BPE exits translate rules into string pairs, publish vocabulary and
clone special tokens once after success and after large attempt storage retires.
The builder directly owns a BpeTrainer and returns it by move; trainer defaults
have one definition.

The input view is borrowed and unchanged. Feed aggregation, serialization and
public thread/parallelism settings are handled by the trainer entry. Training creates
one requested-size Rayon pool and installs all its stages there, independently
of any ambient pool used by feed.

## Construction and task-local aggregation

A borrowed CorpusPlan resolves original symbols and weighted word intervals.
Parallel measurement fills one per-word usize vector. The serial checked prefix
pass reads each symbol count before replacing its slot with the word start; the
same vector becomes the plan's directory. Separators, weighted edge mass and
resident allocation checks retain their original order, including zero merges.
Initial counting gathers sorted temporary positions, reduces counts and freezes
retained lists before allocating token slots. Zero-merge calls still perform all
plan and initial-index checks, then discard both without materialization.
Materialization consumes the plan and releases thin input references. On successful
merge attempts the index and corpus are explicitly released before output strings
and the final vocabulary map are built. The corpus binding is declared before the
index but initialized only for merge attempts; reverse declaration-order cleanup
also releases index before corpus on errors and restart. Both paths share progress
completion and model-part construction.
Fresh mode releases its per-word start directory; weights retain only adjacent
equal-weight regions. Reuse keeps word starts for cohort scan domains.
CorpusPlan exposes its token count to the index.
`InitialPairCounts` owns the dense-versus-sparse choice and its domain: at most
256 IDs use dense pair keys; larger domains use a sparse table.
`CorpusPlan::chunks` returns indexed parallel edge readers paired with
word counts. Word ranges and pair/coordinate/weight generation remain private.
Each count task consumes its reader directly and stops on its own count error.
Progress advances by the bound word count only after a successful count task.
The plan owns the attempt's fresh/reuse mode; PairIndex reads it from the plan,
so index ledgers and materialized word directories use the same configuration.
The plan also binds `Option<usize>` for the birth-span limit: every `Some` enables
whole-word reuse scans, including `Some(usize::MAX)`. Index derives its owner count
from the current training pool. Corpus collects unique word owners from historical
positions and retains their parallel partition. Each word scan maintains its
virtual left neighbor sequentially and produces deferred writes and neighbor events.

Fresh AA tasks carry the pair, merged ID and a copied slice iterator. Acceptance
runs once from left to right. The original historical length determines the chunk
budget; its Positions payload is released before task assembly. All chunks borrow
one global accepted-start set, so shared-edge ownership extends across boundaries.
AB tasks carry the pair, merged ID and a restart-aligned Positions iterator, with
one pair-to-ID lookup per batch. Both branches use the same fresh event loop with
statically dispatched adjacency closures.

Between adjacent selected matches, the left match owns the shared edge and emits
both final IDs. The right match skips that left event across rule and task boundaries.
Matches are `Range<usize>` values in original coordinates; word ranges exclude
separators. The fresh reader pairs each valid range with its cached word weight.

Preparation uses Rayon `map_init` to reuse private directories within each
parallel task. Each side maps neighbor IDs to u32 change-entry indices. A cached
index is valid only when the current task has a group at that index with the same
neighbor ID. Stale indices need no reset, including after failed preparation.
The neighbor accumulator sizes these borrowed directories during construction.
Their u32 entries identify event groups. Rayon `map_init` owns scratch for the
parallel task lifetime, which can differ from an OS worker lifetime.

## Fixed-coordinate corpus

Words occupy disjoint original-symbol intervals separated by sentinel slots.
A logical token stores its ID at its first and last retained-symbol coordinates.
Its span skips interior holes. Initially spans are one for activated IDs. Fresh
identities have a single span, so one ID-to-span table describes the corpus.
When an activated alias acquires unequal spans, the corpus initializes a separate
occurrence-span plane from the current logical words. All subsequent geometry
reads spans from occurrence endpoints.
The ID table still grows with the vocabulary and marks every activated ID as
nonzero, including previously unactivated reserved IDs. Once the occurrence plane
exists, new ID metadata records activation with a nonzero marker. Activation means that an ID has ever been used, even
after its last occurrence is consumed.

`Corpus::fresh_matches` accepts nondecreasing coordinates and returns valid
matches with their word weights. It converts coordinates to resident indices,
keeps at most 16 prefetched positions ahead, caches fixed pair geometry and
reuses weight regions. Stale starts do not query weights. Every refill follows
one pop, preserving the bounded queue. Each preparation task borrows its local
input iterator, avoiding a measured 144-byte decoder-state move for AB tasks.
The borrow ends before the task returns; all readers finish before endpoint writes. AA overlap selection
checks the overlap boundary before calling `fixed_matcher`, so rejected
overlapping starts incur no endpoint reads.

The private `scan_word` scans each selected historical word domain and Positions
stream. Candidate seeks advance over stale positions using occurrence spans;
alias activation or a configured length gate enables whole-word scans. The left
neighbor may be a merge accepted earlier in the word; the right neighbor comes
from the joined endpoint snapshot. A candidate jump reconstructs the left neighbor
from physical endpoints, while adjacent accepted matches retain the virtual ID.
The scan records ordered neighbor events and accepted starts for later application.

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

Each private `Shard` owns its count table, pending actions and birth streams.
`PairIndex` routes work directly into those shards and commits nonempty shards in
parallel. `Shard::commit` drains actions in producer order, removes old weight
before recording each birth. Compressed fragments are stably sorted by birth group/pair,
preserving producer order within each group. The owner checks aggregated birth
weights before connecting fresh fragments or merging interleaved reuse streams.
Groups and candidates remain local to the algorithm. The coordinator receives
completed candidates only after shard jobs join. Errors release unpublished
storage and discard the attempt; applied count changes are not rolled back.

Owners hold counts; one global Candidate priority queue owns fresh lists and
reuse cohorts. Taking a fresh candidate removes its count key and queue record.
Old-boundary removal events decrease existing counts and delete keys below the
floor. Their lists survive until the stale queue records reach the head or the
attempt ends. Selection repairs counts and discards missing fresh keys through
the same queue loop used for reuse ledgers. Each fresh newborn pair belongs to one
producer rule. Preparation appends births directly to compressed leaves, and
owner reduction aggregates their weights before admission. Joined owner results
are pushed into the global queue serially. Large ordinary candidates split into
spatially ordered iterator chunks; filtered AA starts split as ordinary slices.
Fragments move to each owner in indexed task order, and stable grouping preserves
their spatial order. Reuse births can interleave; Positions merges their sorted
coordinate streams before publication.

Commit retains one immutable metadata array and shard-local pending vectors with
reusable capacity. Each action is a resident usize record reference and two flags.
Only nonzero removals and nonempty births route; zero-weight births with
positions still route. Each position stream moves to exactly one owner payload
vector, in the order of that owner's birth actions. Removal precedes birth for
each action; owner order is the original producer order. Draining both vectors
retains capacity. Errors drop active drains; the attempt is discarded after
owner jobs join. Successful commits drain every shard's pending vectors before
the next round.

Published lists are either appendable compressed leaves or immutable,
nonrecursive roots owning boxed leaf arrays. A leaf uses `SmallVec<[u8; 16]>`,
so actual encoded byte width determines inline storage. Positions is 32 bytes
on the tested 64-bit target. Retiring a queue, fragment or birth owner releases
its bytes; ownership and Send/Sync follow from the fields.

`Positions::push` checks nondecreasing full-u64 coordinates and reserves storage
before mutation. `concat` consumes ordered leaves without re-encoding heap
payloads, filters empty leaves and returns a sole nonempty leaf unchanged. Roots
are rejected as inputs, including a sole root; multiple leaves form an immutable
root. `merge` accepts interleaved sorted streams, including roots, retains
duplicates and moves a sole stream unchanged. Multiple streams are read through
a buffered iterator frontier and appended directly into one compressed leaf.
Readers borrow immutable storage and carry private decoder state.

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

Initial counting and preparation append directly to compressed Positions leaves.
All coordinates retain their full width and every append checks order. Fresh
fragments connect in producer order; reuse streams merge their sorted coordinates
while preserving duplicates.

Positions is the only custom container exposed by the position module. Readers
use `iter()`, `iter_from(coordinate)` and `chunks(target_len)`. Chunks are
opaque, independently consumable iterators implementing Send; merge carries the
pair, merged ID and iterator directly. Positions groups short leaves and splits
long leaves at restart boundaries, retaining the already located starting leaf.
Chunk lengths may differ from the requested target; zero is treated as one.
Positions chunks decode incrementally from compressed storage. AA separately
retains the global accepted-start vector for adjacency checks.
Coordinate seeking retains equal values across leaf and restart boundaries.

Each block contains at most 128 positions. Its first position is a full-u64
restart; later positions store unsigned base-128 deltas. Checked monotonic input
makes reconstruction exact, including a gap of `u64::MAX`. Zero gaps retain
repeated coordinates. Block directories contain resident byte offsets, never
truncated corpus coordinates. Nonempty leaves cache their tail before a reserved,
geometrically grown directory. Directory offsets are relative to the data area
and survive relocation when capacity grows. The private codec handles absolute
restart seeds and uses leb128fmt for gap encoding and decoding. A task reads
independent complete blocks; AA and
cohort selection can stream the full list without a corpus-sized bitmap.

## Validation

Full-model comparisons cover vocabulary IDs and complete ordered merges. Small
fixtures also compare every `(pair, count, merged ID)` with an independent
sequential reference. Literal expectations cover numeric limits and alias
cohorts; storage tests and Miri exercise immutable storage, concurrent readers,
owned concatenation, and exact restart/delta bytes when storage grows. Corpus
tests check stale starts and repeated, unit and zero-weight regions across more
than one prefetch window; stream tests include interleaved roots and duplicates.
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
