# BPE training engine

The public entry is `BpeTrainer::do_train`. It consumes weighted words and returns
a vocabulary, merges in rank order, and special tokens. `feed` invokes the caller's
preprocessing callback and collects weighted words; `train` builds a model through
the same private training entry as `do_train`.

Feed uses the ambient Rayon pool, including a pool installed by the caller.
Training owns one pool per call and uses `tk_encode::parallelism::num_threads()`.
Both respect the parallelism switch. Ordinary callback errors do not stop later
callbacks; a failed feed leaves the trainer's previous counts intact.

The implementation preserves weighted pair counts, ascending pair-ID tie breaks,
left-to-right overlap handling, affixes, reserved IDs, and active-ID reuse. Training
counts use checked `u64` arithmetic; configurations that need a signed reuse ledger also
check its `i64` domain. Existing vocabulary IDs remain stable. Alphabet frequency
ties retain the public trainer's existing behavior. Decorated IDs are assigned
in the count view's traversal order before weighted words are sorted; neither
hash-map traversal nor parallel feed entries promise a stable order.

## Reading order

The trainer's private `../feed.rs` counts words in a single map for sequential
input. Parallel input keeps the upstream iterator bridge and batches local counts
into a shared table. Feed consumes that table into globally unique, unordered
entries, without rebuilding a dictionary. Both paths share callback and error
handling. `../word_counts.rs` exposes a borrowed view of either owned result or
the caller-owned map passed to `do_train`. Training orders weighted words; feed
does not sort. Trainer serialization remains a flat word-count map, and equality
compares counts independently of their internal representation.

| Module | Responsibility |
| --- | --- |
| `mod.rs` | Attempts, ID-reuse restart and phase ordering |
| `batch.rs` | Reusable batch workspace, certified selection and restart boundaries |
| `vocabulary.rs` | Token strings, IDs, activation and original-byte symbol interpretation |
| `corpus/prepare.rs` | Weighted word ordering, checkpoints and deferred slot construction |
| `corpus/slots.rs` | 16-, 24- and 32-bit planes and their local safety contracts |
| `corpus/mod.rs` | Live endpoints, token spans, stale-position matching and disjoint writes |
| `initial_pairs.rs` | Bounded initial waves, stable full-key grouping and weighted counts |
| `pair_index.rs` | Candidate priorities, selection and lazy count repair |
| `pair_index/commit.rs` | Publish births, update ledgers and retire old candidates |
| `merge/mod.rs` | Merge contracts and consuming application after readers join |
| `merge/prepare/mod.rs` | Path dispatch and shared neighbor-accounting rules |
| `merge/prepare/{ordinary,aa,cohort}.rs` | Complete preparation algorithms for each path |
| `merge/events.rs` | Compact event references and stable owner-local birth grouping |
| `aa_parity.rs` | Exact overlap selection across AA chunks |
| `execution.rs` | Pool and reusable worker resources, including error-path cleanup |
| `storage/` | Private coordinate storage, G128 encoding, allocation and ID directories |

`train_attempt` initializes the vocabulary and corpus plan, then selects the slot
width. `train_with_slots` builds the initial index and runs the visible phase loop:
`batch.select` → `prepare_merges_with_births` → release candidate readers →
`prepared.apply` → `index.commit_merges_with_prepared`. An attempt releases its
training allocations before `complete_model` constructs the public output.

[DESIGN.md](DESIGN.md) explains the semantics and the cost of the retained
optimizations. Safety arguments remain next to each unsafe operation.

## Correctness checks

From the repository root:

```sh
cargo test --manifest-path tokenizers/tk-train/Cargo.toml --lib
cargo test --manifest-path tokenizers/tk-train/Cargo.toml --no-default-features --lib
```

Tests compare vocabulary IDs, ordered merges and per-rule traces across worker
counts. Named cases cover ties, AA, affixes, filtering, ID reuse, strict birth
limits, overflow, zero-merge returns and segmented long words. Storage tests
exercise full-width positions, restart cursors, append failures, arena lifetime
and directory recovery. The small-input reference preserves upstream queue and
cohort behavior independently of engine storage; explicit expectations cover
wide-count boundaries beyond that reference's count domain.

## Public interface compatibility

`BpeTrainer` builder, feed, train, `do_train` and `train_vocab` signatures match
Hugging Face upstream. Model serialization and Python/Node bindings are unchanged.
The optional parity trainer is unchanged: upstream currently fails to compile
that feature because it references the removed legacy `BPE` type. This change
preserves that existing behavior rather than introducing a separate interface
repair.

## Performance evidence

Batching, G128 positions, bounded stable radix grouping, deferred corpus creation,
allocation reuse and weight cursors remain part of the implementation. Complete
producer rules encode their births directly; split rules retain the aggregation
path. The radix implementation keeps its original BSD license and full-key
stability contract. Sorting uses the existing implementation.

Reproduction tools and experiment records live in
[tokenizers-bpe-benchmarks](https://github.com/Yikai-Liao/tokenizers-bpe-benchmarks).
They record source and binary hashes, pinned corpus inputs, compiler settings,
affinity, public training time, CPU time, process HWM and exact output comparisons.
Measurements apply to their recorded machine and workloads.
