# Completed initialization and merge rounds, 2026-10-11

A1, A2 and A3 are integrated on the existing branch, preserving the pushed Feed.
B1 and fixed-table conversions are deferred. No Index/Positions production changes
or Parity Trainer changes belong to this work. The [submitted plan](plan.md) records
the initial design; the user's later instruction limits this round to A1–A3.

The actual pre-change baseline is `2f4803a936619aa575ba4f6cb49fc660057ef5e4`;
the verified final code is `b4d2baba` (full SHA in `builds.json`). Each completed
slice was committed and pushed separately:

- `e0a38be8`: restore builder-owned Config and construction.
- `45e1aa14`: initialize Vocabulary and InitialTokenIds together.
- `534de0c1`: own complete rounds in merge_next.
- `d3df8f38`: complete A1 by making Default delegate to builder().build().
- `b4d2baba`: test actual reuse restart with a tie at an exceeded alphabet cutoff.

## Upstream diff review

`git fetch upstream main` resolves Hugging Face's `tokenizers` main to
`fb49a29223724a96f26e061bec8ec7517aa14177`. It is the upstream comparison,
while `2f4803a9` is the performance and behavior baseline for this round.
The cumulative upstream diff includes the earlier training rewrite. It must not
be interpreted as the size or scope of A1–A3. [The cumulative diff](upstream/full.diff)
and [this round's diff](upstream/a1-a3.diff) are retained separately.

| Boundary | Review result |
| --- | --- |
| Config | All ten fields and types match upstream. |
| Builder defaults and methods | Executable Rust matches upstream after substituting only the required WordCounts empty storage for AHashMap. Public method signatures and moves are retained. Documentation explains current alphabet/span semantics. |
| Trainer Default | `Self::builder().build()` matches upstream; defaults are maintained once in Config. |
| Progress helpers | All four perform the same Rust operations as upstream; update_progress has one preexisting blank-line difference. A2 preserves stage startup after alphabet initialization, symbol progress increments and outer completion. |
| Initialization | InitialTokenIds is completed in vocabulary.rs before weight sorting. CorpusPlan receives owned IDs and has no mutable Vocabulary/completion protocol. |
| Merge round | Selection, consuming preparation, joined writes and index publication reside in merge_next; outer coordination appends only committed records. |
| Index, Positions and Feed | Exact production source equality to this round's baseline; Index changes only adapt/adversarially exercise tests. |
| Parity Trainer and word | Exact source equality to upstream main. |

`review.py` reproduces thirteen explicit checks in `upstream/review.json`.
Config, default and builder comparisons ignore comments/whitespace, and normalize
only the required empty-count expression in the builder. They do not certify
algorithmic equivalence by textual comparison; complete-model/trace tests do that.

## Initialization and lifetime boundaries

Vocabulary retains token identities and the continuing prefix. Initialization
borrows the suffix from trainer and keeps plain-ID resolution as local state.
InitialTokenIds owns activation spans, character/decorated-ID lookups and flags;
its token_count is spans.len(). The input-order decorated identity pass and
unlimited plain-alphabet shortcut remain. The limited alphabet is retained
across the actual fresh-to-reuse retry, including a tie at the cutoff.

CorpusPlan owns the IDs during measurement and initial pair counting. After
parallel symbol writes finish, into_spans consumes the IDs, moving the existing
span Vec and releasing lookup tables before merge rounds. No span clone or
extra corpus traversal is introduced. The layout and initial count checks still
precede the target-vocabulary shortcut, which skips resident endpoints. A low
initial vocabulary with no admitted pair remains a separate case.

merge_next contains the selection loop and returns Finished, RestartForReuse or
Committed(Vec<MergeRecord>). Its private Batch is consumed by prepare(self):
candidates and Positions retire before endpoint application. Selection's head/tail
sets are explicitly dropped before preparation to preserve their former release
boundary. The existing bounded record Vec is moved through the outcome without
an additional collection or allocation. All successful preparation joins precede
writes; all writes join before count-owner updates and ordered birth publication.
Only a successful index commit returns records for outer merges/trace/progress.
Selection and later failures can mutate local attempt state; the caller discards
the attempt, and no per-round rollback guarantee is added.

## Correctness checks

The final default suite records **42 passes and the one existing encoder-roundtrip
failure**. The same unfiltered baseline suite records 39 passes and that failure
(`feed_training_tokenizer_json_and_encoding_roundtrip`, input 测, ID 3 versus 4).
The assertion is preserved. No-default records 42 passes with that test filtered.
All-target Clippy with warnings denied, changed-file formatting and whitespace
checks pass. Commands, statuses and complete logs remain in `checks.json` and
`checks/`; this is not a claim that the unfiltered suite is all green.

The existing differential tests compare full vocabulary IDs, ordered merges and
every certified `(pair, count, replacement ID)` at 1/4/8 workers, including AA,
reserved identities, historical cohorts, restart and numeric errors. Preparation
failure now corrupts a queued candidate before merge_next and checks every endpoint
remains unchanged at those worker counts. Owner failure is injected before the
complete round: writes can occur, but failed publication returns no records.
Neither fixture bypasses the new complete-round boundary.

Added initialization checks use ordered input with four observed characters and a
limit of two, literal multibyte prefix/suffix IDs, original UTF-8 boundaries after
first/last filtering, and the allocation identity of moved spans. A separate four-
character/three-slot cutoff has forced a/b and tied q/z, actually triggers the AA
identity-reuse restart, preserves the chosen alphabet, and compares complete models
and traces in the reuse attempt at 1/4/8 workers. Cutoff fixtures exceed the limit.

The prior Feed resident-coordinate panic remains separately recorded and
unattributed in [the Feed follow-up](../feed-cost-20261011/README.md). This work does
not claim to fix it or classify it as the encoder-roundtrip failure.

## Paired real-input sanity check

Four 256 MiB Chinese/English Wikipedia scenarios use ByteLevel regex or Whitespace,
four workers pinned to CPUs 0–3, vocabulary 50000 and minimum frequency two, with
no affixes. Both arms include the exact same Feed and shared runner/dependency
sources. Release opt-level 3, fat LTO, one codegen unit and the same compiler/flags
are recorded in `builds.json`. Binaries are separately frozen and SHA256-verified.
`inputs.json` records the actual byte lengths, source hashes and dataset revision;
prepared-word metadata is inherited context and is not used for these pipelines.

This is one baseline/final pair per scenario, baseline first, with no discarded
warmup or forced cache eviction. Source SHA256/strict UTF-8 verification warms
ordinary OS cache before serial measurements; each child streams the input.
No tests, builds or other benchmark arms run concurrently. Small changes are
sanity observations, not estimates of a stable performance improvement.

| Input | Train wall baseline → final (s) | Train wall change | Pipeline wall change | Train CPU change | Pipeline CPU change | Pipeline HWM baseline → final (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chinese ByteLevel | 9.857 → 9.849 | -0.08% | -2.00% | +0.46% | -1.78% | 2400.2 → 2405.8 |
| Chinese Whitespace | 8.513 → 8.457 | -0.66% | +1.59% | -1.18% | +1.33% | 1770.9 → 1769.0 |
| English ByteLevel | 0.841 → 0.841 | -0.00% | +0.21% | +0.52% | +0.33% | 183.8 → 187.0 |
| English Whitespace | 0.742 → 0.739 | -0.41% | +0.06% | -0.31% | +0.37% | 159.0 → 147.5 |

Each of the eight samples matches its scenario's complete count hash and parsed
vocabulary/ordered-merge reference, with zero sampled child swap and zero major faults. Exports and
external canonical-count sorting occur after the timed/peak measurement boundary.
Feed wall shifts −10.10% for Chinese ByteLevel and +12.92% for Chinese Whitespace,
despite identical Feed source; the corresponding feed HWM readings also change in
opposite directions. This resembles previously observed SCC/random-hasher peak
variation, but this sanity run has no phase instrumentation and establishes no new
capacity cause. Feed measurements are kept in the raw records rather than credited
to A1–A3. Training wall changes range from −0.66% to approximately zero; these
single pairs reveal no substantial regression and do not establish a stable gain.
No universal pipeline memory reduction is claimed.

## Reproduction and retained evidence

Native validation is reproduced with check-final.py, using the baseline's known
failure as an explicit expectation. Changed-file formatting uses --base 2f4803a9
and only the six affected Rust files. The baseline/final source trees include exact
ordinary BPE, runner and dependency/lock snapshots; Feed is uninstrumented.

Create worktrees at the baseline/final commits, apply the identical common runner
snapshots, and restore the ignored tk-train and runner Cargo.lock files. Adjust
paths in prepare-builds.py/builds.json to your checkout. build.py builds offline and
locked, then freezes each binary before the other build. Run:

```sh
BPE_FEED_ARMS=baseline,final BPE_FEED_BLOCKS=1 \
BPE_FEED_CASE=bytelevel_regex-w4,whitespace-w4,en-bytelevel_regex-w4,en-whitespace-w4 \
python measure.py
```

sanity/{manifest,runs,summary,sample-artifacts}.json retains commands, CPU affinity,
metrics, complete-count/model hashes and child output for every attempt. The
manifest's generic worker list is [1,4]; every selected scenario/job actually uses
four workers. Native fixed-input/trace comparisons and these feed-to-training
comparisons are separate protocols. integrated-source-hashes.json verifies current
source against the frozen final release arm. No corpus, full model, count export
or executable is committed.
