# BPE training engine

The public entry is `BpeTrainer::do_train`. It reads a borrowed weighted-word map
and returns a vocabulary, merges in rank order, and special tokens. `feed` applies
the caller's preprocessing function and collects weighted words. `train_vocab`
trains those words, and `train` builds the public BPE model. Both use the same
private training entry as `do_train`, borrowing the stored counts through
`WordCountsView`.

Feed uses the ambient Rayon pool, including a pool installed by the caller.
Training owns one pool per call and uses `tk_encode::parallelism::num_threads()`.
Both respect the parallelism switch. Ordinary callback errors do not stop later
callbacks; a failed feed leaves the trainer's previous counts intact.

Within a training attempt, adding tokens never renumbers assigned IDs, and a
merged string already in that vocabulary retains its ID. Each training call
initializes a new vocabulary from configuration and weighted input; `train`
replaces the model. The engine preserves weighted pair counts, pair-ID tie breaks,
left-to-right overlap handling, affixes, reserved IDs, and active-ID reuse.

Engine pair, birth and removal arithmetic is checked in `u64`; signed reuse
updates are checked in `i64`. Nonempty affixes also enforce signed input bounds
before reuse is detected. `feed` and limited-alphabet accumulation retain their
existing arithmetic. Pair tie breaks are fixed, but equal-frequency alphabet
truncation and decorated ID allocation depend on input traversal. Neither map
traversal nor parallel feed entries promise a stable order. Tests check worker
parity for the same input view, not determinism across arbitrary hash seeds.

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
| `batch.rs` | Reusable batch workspace, priority-preserving selection and restart boundaries |
| `vocabulary.rs` | Token strings, IDs, activation and original-byte symbol interpretation |
| `corpus/prepare.rs` | Weighted word ordering, checkpoints and deferred slot construction |
| `corpus/slots.rs` | 16-, 24- and 32-bit planes and their local safety contracts |
| `corpus/mod.rs` | Live endpoints, token spans, stale-position matching and disjoint writes |
| `initial_pairs.rs` | Bounded initial waves, stable full-key grouping and weighted counts |
| `pair_index.rs` | Candidate priorities, selection and lazy count repair |
| `pair_index/commit.rs` | Ordered counts, completed-birth publication, routed birth reduction and refill |
| `merge/mod.rs` | `CompletedBirth`, write-plan contracts and consuming application after readers join |
| `merge/prepare/mod.rs` | Path dispatch and shared neighbor-accounting rules |
| `merge/prepare/{ordinary,aa,cohort}.rs` | Complete preparation algorithms for each path |
| `merge/events.rs` | Compact event references and stable owner-local birth grouping |
| `aa_parity.rs` | Exact overlap selection across AA chunks |
| `execution.rs` | Pool and reusable worker resources, including error-path cleanup |
| `storage/` | Private coordinates, delta-encoded positions, allocation and ID directories |

`train_attempt` initializes the vocabulary and corpus plan, then selects the slot
width. `train_with_slots` builds the initial index and runs the visible phase loop:
`batch.select` → `prepare_merges_with_births` → release candidate readers →
`prepared.apply` → `index.commit_merges_with_prepared` → release events. An attempt
releases its training allocations before `complete_model` constructs the public
output.

[DESIGN.md](DESIGN.md) explains the semantics and the cost of the retained
optimizations. Safety arguments remain next to each unsafe operation.

## Review routes

Each route follows a contract, its implementation, and the tests that check it:

| Review question | Contract → implementation → tests |
| --- | --- |
| Semantic compatibility | [Identity modes](DESIGN.md#sequential-semantics-and-token-identity) → [vocabulary](vocabulary.rs), [batch/restart](batch.rs), [mode-specific selection](pair_index.rs) → [semantic traces](tests/semantic_parity.rs), [identity fixtures](tests/identity_reuse.rs) |
| Concurrent writes | [Batch priority proof](DESIGN.md#why-the-prefix-keeps-sequential-priority) → [ordinary](merge/prepare/ordinary.rs), [AA](merge/prepare/aa.rs), [cohort](merge/prepare/cohort.rs), [apply](merge/mod.rs), [slot safety](corpus/slots.rs) → [parallel ordering](tests/routing_and_publication.rs), [AA parity tests](aa_parity.rs) |
| Counts and publication | [Owner sequence](DESIGN.md#complete-birth-handoff-and-owner-work) → [neighbor accounting](merge/prepare/mod.rs), [actions](merge/events.rs), [commit](pair_index/commit.rs) → [producer/fallback tests](tests/producer_fast_path.rs), [index publication tests](pair_index.rs) |
| Position representation | [Storage contract](DESIGN.md#private-storage-and-worker-resources) → [full-width coordinates](storage/position_storage.rs), [codec/cursor/append](storage/sorted_positions.rs), [arena](storage/arena.rs) → local duplicate, restart, append-failure and arena-lifetime tests in those files |
| Optimization costs | [Retained costs and sources](DESIGN.md#algorithm-sources-and-local-costs) → [initial grouping](initial_pairs.rs), [direct producers](merge/prepare/ordinary.rs) → [corpus equivalence](tests/initial_corpus.rs), [producer equivalence](tests/producer_fast_path.rs), then [measurement requirements](#retained-optimizations-and-measurement) |

[tests/mod.rs](tests/mod.rs) owns the shared fixtures and reference comparison
helpers. [public_contract](tests/public_contract.rs) checks feed/reload, thread
policy, progress and failure boundaries. Local codec, arena and index tests stay
with the implementation whose invariant they exercise.

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

`completed_and_partial_births_publish_once_after_ordered_removals` covers direct
and routed births in one commit, partial fragments that reach the floor only
after reduction, retirement, and list ownership after events are dropped.
`reuse_both_checks_removal_before_birth_without_netting` checks the intermediate
signed overflow boundary, where a zero net change still fails in original order.

## Complete births and owner commit

`CompletedBirth` transfers a fresh sole producer's complete count and encoded
positions to commit. Preparation keeps removal events and omits the birth chains
for those records. Commit moves the lists into private `PairState` values.

The [owner sequence](DESIGN.md#complete-birth-handoff-and-owner-work) explains
ordered counts, direct-result publication, remaining fragment reduction and fresh
prefix refill. The [phase contract](DESIGN.md#phase-order-and-failure-contract)
explains snapshot continuity, joined tasks and attempt disposal after errors.

## Retained optimizations and measurement

Batching, delta-encoded positions, bounded stable radix grouping, deferred corpus
creation, allocation reuse and weight cursors control repeated work and memory
use. Complete producers encode births directly; split rules retain owner
aggregation. [DESIGN.md](DESIGN.md#algorithm-sources-and-local-costs) records codec
and radix sources and the cost of their local representations.

Performance measurements must identify source and binary hashes, input hashes,
compiler/features, worker count and CPU affinity, and retain raw repeated results.
Report wall time, CPU time, process HWM and exact output validation for the measured
workloads. Measure `feed`, public `do_train` and end-to-end time separately so input
collection and engine changes can be attributed. Historical experiments belong in
benchmark records; their numbers describe their recorded machines and workloads.
