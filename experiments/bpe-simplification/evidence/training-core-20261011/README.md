# BPE training-core consolidation

The change gives `BpeTrainer` three training levels: public input/model adapters,
`do_train_impl` for one call's execution environment, and `train_attempt` for one
complete attempt. The public signatures, stored count representations and flat
serialization format remain unchanged.

Stage 1, commit `36bc3b18`, moves the existing coordinator and attempt into the
trainer implementation, deletes `train_counts`, and puts constructors/public
training methods before the private training flow. The worker override enters
this actual core; its default worker policy is defined once. The attempt body
and release points are preserved in this commit.

Stage 2, commit `aa35055f`, removes test callback parameters and replay. An attempt returns vocabulary
and merges, plus a trace under `cfg(test)`; only a completed attempt carries this
result. The core permits a fresh attempt followed by at most one identity-reuse
attempt. The first limited alphabet survives the switch, while speculative
vocabulary, merges, index, corpus and trace are discarded. A second restart
request returns an explicit internal invariant error. Special tokens are cloned
once, after successful training, when the core assembles the public result.

Both successful paths share progress completion and model-part construction.
Zero-merge calls still build and validate the borrowed corpus plan and initial
pair index; they allocate no resident corpus. Merge calls explicitly drop index
and corpus before allocating output strings and the final vocabulary map.

The separately requested `Batch::commit` boundary was already implemented by
`ef418ae7` and `9ecbd0cd`, the latter being this task's baseline. Its preparation,
write and index-update joins remain unchanged. This task keeps its whole-round
regressions and does not mix new Merge changes into the entry-point commits.

## Validation

Stage 1 passes 20 BPE library checks; stage 2 passes 21. The suite compares full
vocabulary IDs, ordered merges, special tokens and successful traces with an
independent sequential reference at 1, 4 and 8 workers. Affixes, activated and
reserved IDs, AA overlap across task boundaries, strict length gates, wide IDs,
full-u64 counts, per-key overflow and signed zero-merge limits remain covered.

The added public checks compare caller maps, frozen entries and fed counts,
repeat training without consuming counts, train after serialization, and verify
successful model replacement. Decorated Map/Entries comparison deliberately
preserves the same input traversal order, since initial decorated IDs depend on
that order. A training overflow and a model-reader affix rejection both retain
the previous model vocabulary. The public subprocess matrix verifies the requested
pool, progress schema, and no materialization/preparation work for zero merges.

The dedicated restart fixture commits XY before an activated AA collision. It
checks the retained alphabet, the successful reuse result, and the real core's
trace against the reference, so speculative XY cannot leak into the result.
All-target Clippy runs with warnings denied. The changed-file formatter checks
only `mod.rs` and `tests.rs`.

The existing `feed_training_tokenizer_json_and_encoding_roundtrip` assertion is
excluded, as in prior checks; this change does not claim that encoder assertion
passes. Commands and results are recorded in `validation.json` and the logs.

## Resource protocol

Each step compares successive frozen release binaries on the same Chinese
256 MiB source, using ByteLevel regex and Whitespace prepared count maps. The
source and prepared-input hashes are in `inputs.json`; the same lexical map
reconstruction and fixed AHash seeds are used by every binary. Builds use Rust
1.98.1, opt-level 3, fat LTO and one codegen unit, with a locked offline build.
`builds.json` records every BPE and runner source hash and binary hash; the full
sources, manifests and lockfile are frozen under `source/`.

Each case has an excluded AB warmup, then BA and AB formal pairs, in separate
processes. Training uses four workers pinned to CPUs 0–3. No other build, test or
benchmark runs during measurement. CPU and elapsed time cover public `do_train`,
including output construction, and exclude input loading and model serialization.
Peak RSS is the process high-water mark sampled before validation and therefore
includes loaded caller counts and earlier input allocations. Every run compares
its complete parsed model with a reference and checks for zero swap.

The tables report the median of each formal pair's relative delta. Positive time
or memory values mean an increase. Two formal pairs on a shared VM are descriptive
observations; they do not establish a universal speedup or slowdown. Explicit
release boundaries and measured RSS together check for accidentally prolonged
attempt storage, rather than attributing small RSS changes to one local object.

| Comparison | Input | CPU time | Elapsed time | Peak RSS |
| --- | --- | ---: | ---: | ---: |
| Stage 1 vs before | ByteLevel | +2.33% | +3.70% | +0.59% |
| Stage 1 vs before | Whitespace | +0.36% | -0.54% | +0.48% |
| Stage 2 vs stage 1 | ByteLevel | +3.46% | +3.77% | -0.64% |
| Stage 2 vs stage 1 | Whitespace | -1.68% | -2.49% | +0.29% |

All 24 runs are valid, including 16 formal runs. Complete models agree in both
steps and across their shared stage-1 references, with no observed swap. Peak
changes range from -0.64% to +0.59%; release boundaries remain explicit. ByteLevel
elapsed time increases by about 3.7% in each successive comparison, while
Whitespace decreases. The two comparisons are separate measurement series;
their percentages must not be compounded into a final-versus-baseline estimate.
No stable performance improvement is claimed for this structural change.

Raw manifests, runs, summaries, per-job output and one compressed reference model
per case are archived in `stage1/` and `stage2/`. The same stage-1 reference is
also checked across both measurement sets. `measure.py` and
`archive-measurements.py` retain the measurement and archive protocol.
