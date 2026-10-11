# BPE successful commit records

Implemented on `simplify/bpe-flat-20261010` from its fetched head,
`c638658303c97d420973c79e584b6916de36e803`, following the uploaded
`bpe-trainer-interface-plan.md.txt`.

`BpeTrainer::train_attempt` still executes the initialization, merge loop, progress,
restart and result conversion. The production change adds only `MergeRecord`:
`Batch::commit` captures bounded scalars, consumes preparation, joins writes and
index owners, and returns records after successful publication. The trainer appends
merges/trace and advances progress after that return. The precommit `pairs` and
`trace` accessors are removed. Selection order, certified counts and repeated
historical cohort rules are preserved. No initialization, selection, counting,
position storage, heap or public interface algorithm was replaced.

The corpus binding is declared before index and initialized only when merges are
needed. Reverse declaration-order cleanup releases index before corpus on early
returns; successful paths retain explicit drops before output construction.
Zero-merge training still builds and validates plan/index without resident endpoints.
Errors discard the attempt; endpoint writes and partial count updates can remain
inside failed state. Returning no records does not imply rollback.

## Validation

Both default and no-default-features library runs pass 30 checks, including 23 BPE
checks, after excluding the existing encoder roundtrip assertion. Clippy with
warnings denied and the changed-file formatter pass. See `validation.json` and
the corresponding logs for commands and results.

Full vocabulary IDs, ordered merges, special tokens and per-rule traces are checked
against the sequential oracle at 1/4/8 workers. Existing tests cover reserved and
activated aliases, AA overlap, shared endpoints, zero weights, length gates, wide
IDs, overflow, zero-merge validation and repeated pair/count cohorts. Changed and
added regressions check ordered successful records, real owner overflow after
endpoint writes without candidate publication, and restart midway through selection
after an earlier successful round. The latter commits reserved XY, tentatively
resolves UV, then restarts on activated AB; speculative UV and the old attempt's
history are discarded. The public subprocess matrix also checks that the first
attempt's progress reports only its one committed XY rule.

Two failures are reproduced on both the unmodified baseline and candidate:

- `feed_training_tokenizer_json_and_encoding_roundtrip`: encoding `测` yields ID 3
  where its assertion expects ID 6. The initial candidate run had 22 passing BPE
  checks and this failure; subsequent added coverage brings passing BPE checks to 23.
- Optional `parity-aware-bpe` fails compilation because `parity_trainer.rs` imports
  unresolved `super::BPE`; `word.rs` also has an unused import. This feature is not
  counted as validated.

## Resource comparison

Frozen baseline/candidate release runners use Rust 1.99.0, opt-level 3, fat LTO,
one codegen unit and the same locked dependencies. Both prepared Chinese 256 MiB
input hashes match the earlier training-core measurements. Whitespace counts are
prepared once from the same source; ByteLevel uses the existing frozen counts.

Each input has one excluded AB warmup followed by four alternating BA/AB formal
pairs, in separate processes with four workers pinned to CPUs 0–3. No builds or
tests run during these measurements. CPU/wall time covers public `do_train`,
including final output construction, and excludes loading, preprocessing and model
serialization. HWM RSS includes loaded caller counts and earlier input allocations.
Swap is sampled every 25 ms. All 20 complete parsed models agree, and observed swap
is zero. `performance-manifest.json`, `performance-runs.json` and
`performance-summary.json` retain inputs, binary hashes, host details and samples;
`builds.json` records source hashes. Test-only coverage was extended after freezing
the release binaries; measured production code stayed unchanged.

The table reports the median of each formal pair's relative candidate delta.
Positive values mean an increase. These laptop measurements describe this input
and run configuration; they do not establish a universal absence of regression.

| Input | Wall time | CPU time | Peak RSS |
| --- | ---: | ---: | ---: |
| ByteLevel | +1.14% | +1.55% | −0.50% |
| Whitespace | −0.03% | −0.82% | +0.48% |

ByteLevel has a small observed time increase; Whitespace time is approximately
unchanged. RSS changes remain within about half a percent. No performance
improvement is claimed, and the allocation measurements below do not establish
the cause of these differences.

## Lifetime and allocation diagnostics

`probe.py` copies current BPE source into an expendable worktree and adds diagnostic
destructor markers, isolated select/prepare/owner error injection, and a System
allocator counter enabled only on the coordinator thread during record collection.
The instrumentation is archived in `diagnostics.patch` and `record-probe.patch`;
none of it enters production. The frozen release binaries above are uninstrumented.

`lifecycle.json` and seven logs cover target completion, Finished, zero merge,
restart, select error, prepare error and owner error. Normal/Finished use 4096 AB
starts, including compressed candidate storage. Every preparation retires its
selected candidates before the write phase. Index precedes corpus on success,
restart and error; both precede output construction. Zero merge drops its initial
index and queued candidates but never creates/drops resident corpus. Error paths
return neither records nor final output. The prepare injection runs before writes;
the owner injection runs after writes. The production regressions separately
exercise real arithmetic failures in preparation and owner commit.

`records.json` and compressed raw probe logs measure the actual record Vec
allocations on both large inputs and a fixed generated reuse workload. Every batch
allocates once with no reallocations, exact capacity, and 8 bytes per production
record. All three instrumented models match the uninstrumented baseline.

| Input | Batches / allocations | Single-rule batches | Total requested bytes | Total capture time |
| --- | ---: | ---: | ---: | ---: |
| ByteLevel | 1646 | 172 | 397952 | 0.300 ms |
| Whitespace | 1390 | 146 | 257928 | 0.220 ms |
| Generated reuse | 2164 | 2164 | 17312 | 0.062 ms |

Singleton capture medians are 73.5 ns, 83 ns and 27 ns respectively. These are
diagnostic timer/allocator observations of allocation plus scalar copying,
including probe overhead, from one run per input. They are not isolated allocator
latency estimates or substitutes for the repeated full-training comparison. Total
requested bytes sum allocations across rounds; they are not simultaneous live
memory. Fresh batches remain capped at 256 pairs (2 KiB payload); reuse holds one
pair (8 bytes), excluding Vec/allocator metadata.

To reproduce the paired comparison, build baseline/current runners using the
profile in `builds.json`, freeze both binaries, then invoke `measure.py` with their
paths and the two prepared inputs from `performance-manifest.json`. To reproduce
diagnostics, run `probe.py` on a separate baseline worktree, format its changed
files, and build its library test and release runner. Run the exact
`trainers::bpe::tests::interface_lifecycle_probe` test with `--nocapture` and
`BPE_INTERFACE_PROBE` set to each scenario in `lifecycle.json`. Use
`BPE_INTERFACE_PROBE=records` with the runner jobs in `records-*-job.json` for the
allocation probe; update temporary input/output paths as needed. The generated
reuse words are retained in `reuse-words.json`.
