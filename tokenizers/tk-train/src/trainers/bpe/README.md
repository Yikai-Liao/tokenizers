# BPE training

Training returns the complete vocabulary with token IDs, ranked merge rules,
and special tokens. It preserves weighted pair selection and compatible batch
execution while using one endpoint/event protocol.

The public entry points and private training methods share [mod.rs](mod.rs).
`do_train` borrows the caller's map; `train_vocab` borrows stored counts. Both enter `do_train_impl`, which selects workers, installs a
dedicated pool and progress context, and runs one fresh attempt with at most one
identity-reuse retry. The Trainer implementation builds and replaces the model
from `train_vocab` results. Tests enter the real core with explicit workers and
inspect the successful Vocabulary's test-only trace. The builder directly owns
the trainer it returns; default option values have one definition.

Read `train_attempt` for the complete workflow: initialize identities and a
borrowed corpus plan, build the compressed occurrence index, materialize endpoints
when merges are needed, call `merge_next` for each complete compatible round,
advance progress from its committed rule count, then publish the model parts.
Vocabulary owns committed ID history. `Corpus::merge` joins preparation before
endpoint writes and joins those writes before index updates.
A collision with a previously activated identity discards the fresh attempt and
restarts from original input and the retained alphabet. Errors and restart requests
return no trace or speculative model parts. Zero-merge calls retain numeric and
initial-index checks without allocating corpus slots; all successful paths release
large attempt structures before building output. Special tokens are cloned once
at the successful output exit, which translates ordered rules into string pairs.

The implementation is divided by the knowledge each module owns:

- [vocabulary.rs](vocabulary.rs) owns strings, alphabet interpretation, decorated
  IDs, initial activation metadata, alphabet selection, committed rule history,
  and canonical output.
- [corpus.rs](corpus.rs) owns fixed coordinates, token endpoints, weights, and
  ID activation and occurrence spans. Initial ID metadata moves here once;
  merge selection and geometry use this same activation record.
  It measures per-word symbol counts in parallel, then replaces those same
  slots with checked word starts in one serial prefix pass. No second
  per-word usize array is allocated during layout construction.
  Only this module converts full-u64 occurrence coordinates
  to checked resident indices. Its `fresh_matches` iterator filters stale starts
  and returns matches with weights, keeping prefetch and weight-region caching
  inside each reader. Matches use `Range<usize>` in original coordinates.
  Its private [corpus/merging.rs](corpus/merging.rs) consumes selected payloads,
  prepares neighbor events and historical word scans, then applies joined writes.
  Candidate seeks, virtual left neighbors, length gates and shared-edge ownership
  stay beside the geometry they interpret. Mode and the optional birth-span limit
  are bound by the plan for the entire attempt.
- [index.rs](index.rs) owns exact priority correction, count shards, signed reuse
  ledgers, one owning candidate queue, and complete birth publication.
  `InitialPairCounts` chooses dense or sparse storage from the plan's token count.
  Index derives its owner count from the current training pool.
  Initial chunks bind whole-word edge iterators and progress units; range
  assembly stays private. Index performs checked count accumulation.
- [merge.rs](merge.rs) owns compatible selection and round publication.
  `BpeTrainer::merge_next` resolves identities, activates IDs, moves ordered
  `SelectedMerge` payloads into `corpus.merge`, then commits returned changes to
  the index and ordered rules to the vocabulary. It returns finish, restart, or
  the committed rule count. The coordinator uses that count to advance progress.
- [positions.rs](positions.rs) owns sorted full-u64 streams and restart blocks.
  Producers append directly to compressed leaves backed by `SmallVec<[u8; 16]>`;
  byte width alone determines whether storage spills. `concat` consumes ordered
  leaves without re-encoding their heap payloads; `merge` merges interleaved
  streams while retaining duplicates. Coordinates through `u64::MAX` are preserved.
  Readers use `iter()`, `iter_from(coordinate)` and `chunks(target_len)`.
  Chunks are independently consumable iterators with private restart alignment
  and already located starting leaves. The nested codec owns sequential decoding.
- [feed.rs](feed.rs) owns streaming batches and word aggregation. Named
  `flush` and `accumulate` closures share callback/error handling. Serial input
  moves its completed map into owned entries; parallel folds flush at 2048
  distinct local keys and publish successful tails before ordinary error
  reduction. The shared table stays
  private to one call, and failed feed leaves the previous trainer state intact.
  After folds finish, consumption moves entries without enlarging the shared table;
  SCC checks shrinking after its consumption traversal, and a reservation can
  itself trigger another enlargement.
  Startup singletons expose short expensive streams to parallel callbacks; later
  batches contain at most 32 inputs and retain the first exhaustion state.
- [word_counts.rs](word_counts.rs) stores trainer-owned counts as entries and
  borrows caller-owned maps for public `do_train`. `From` conversions move owned
  maps into entries or borrow either input representation. `from_unique` accepts
  already aggregated entries. Content equality, map-style
  Debug and the flat serialization contract are independent of entry order.

One owning priority queue holds fresh lists and reuse cohorts. Its historical
resource measurements and equal-priority reuse ordering limitation are recorded
in the [heap experiment](../../../../../experiments/bpe-simplification/HEAP_EXPERIMENT.md).
Larger occurrence lists use owned allocations; dropping a list releases its bytes,
including short compressed lists that previously remained in an attempt arena.

Initial collection splits the borrowed plan at whole-word boundaries, with a
2²⁴-resident-slot block cap. An oversized single word is processed alone. Each
producer appends positions directly to compressed leaves. Owned fragments retain complete
pair counts; global frequency admission precedes final list publication.

Each range admits one word first, including an oversized word. A fitting final
tail is admitted directly; otherwise `slice-group-by`'s `exponential_search_by`
counts additional words whose end boundaries fit, including an exact-budget end.
Vocabulary initialization returns the completed identity table and `InitialTokenIds`
together. The plan owns initial IDs, including activation spans, and consumes them
after endpoint materialization: spans move to Corpus while character/decorated
lookups are released before merging. Initialization resolves decorations in input
order before words are sorted by weight; reuse retains the selected limited alphabet.
The initial IDs expose one filtered `symbols()` iterator to initial edge counting and
endpoint materialization. Decorations still use original UTF-8 byte boundaries;
edge coordinates enumerate retained symbols. Unlimited alphabet observation uses
Rayon bitmap reduction, retaining the full Unicode bit domain for empty input.

Fresh preparation specializes AA and AB task inputs. AA selects nonoverlapping
starts from left to right, releases its historical payload, then uses the global
accepted-start set to account for adjacent matches across chunks. Its chunk budget
uses the original payload length. AB keeps one exact-pair lookup and borrows
restart-aligned position chunks. Position appends retain ordering checks.

Corpus readers keep at most 16 decoded positions ahead in a `VecDeque`. Initial
filling takes at most 16 elements; every refill follows one pop, so the queue never
exceeds that length. The same reader validates fresh matches and caches weight
regions. Cohort scans use `find` to consume the next eligible occurrence.
Commit stably sorts compressed fragments by birth group/pair, preserving each group's
producer order before connecting fresh fragments or merging reuse streams.

Compatible batches retain the original priority prefix and permit shared heads
or shared tails. Crossed endpoints end the batch without skipping a candidate.
AA and reserved-ID rules run alone. Preparation records compressed birth streams;
commit aggregates their weights before admission and publication. Commit routes
compact metadata references and moves each position stream only once. Reuse mode selects
one cohort at a time; its word scans follow the original alias/length-gate conditions.

Run library checks from the repository root:

```sh
cargo test --manifest-path tokenizers/tk-train/Cargo.toml --lib
cargo test --manifest-path tokenizers/tk-train/Cargo.toml --no-default-features --lib
cargo clippy --manifest-path tokenizers/tk-train/Cargo.toml --all-targets
cargo +nightly miri test --manifest-path experiments/bpe-simplification/miri-codec/Cargo.toml
```

The tests compare full vocabulary IDs, ordered merges and per-rule traces with
an independent small-input reference. Fixed-seed combinations cover thread
counts, affixes, aliases, zero weights, ties, overlap and strict birth gates.
Literal expectations cover wide counts and public errors. Position storage tests
cover the full u64 domain, repeated values, restart boundaries, independent chunks
and interleaved stream merging. Corpus tests cover stale starts and weight-region
transitions across more than one prefetch window. Historical native and Miri
storage evidence is retained in the experiment reports.
Public tests cover feed, model reload, progress and ambient-versus-training pool
behavior. [The coverage map](COVERAGE.md) records the combined boundaries.

Progress uses the upstream `ProgressBar`, `ProgressStyle` and `ProgressFormat`.
The trainer's original helpers retain the Tokenize words, Count pairs and Compute
merges stages. Parallel collectors increment the bar at work-block boundaries;
the coordinator reports merges after each joined batch.

The counting command also checks formatting of ordinary BPE and its tests;
sibling trainers and unchanged legacy parity files retain upstream formatting.
Source counts and diffs describe the implementation scope. The report
includes the ordinary BPE implementation, public API, feed and word-count storage;
tests, the reference oracle, shared helpers and Miri harness are counted separately.
Pass `--change-base <revision>` to `count_lines.py` to report this change alongside
the default comparison with HF `origin/main`. Git diffs detect renames and include
comments and documentation, while source counts exclude comments and blank lines.
Performance evidence and build/input hashes are recorded by the
[experiment](../../../../../experiments/bpe-simplification/STATUS.md).
The Corpus boundary refactor's correctness, timing and assembly evidence is in the
[validation report](../../../../../experiments/bpe-simplification/evidence/corpus-boundary-20261013/README.md).
