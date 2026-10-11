# Scope and protocol

Baseline: 303597a24e013c644209df71b6d82fd9a649c8a2. The preceding
MergeToken/AttemptResult removal is already in this baseline. Corpus B1 is excluded
at the user's request; Corpus, Feed, Index and Positions remain byte-identical.

1. Keep two Vec<u32> neighbor directories and their existing domain sizing.
   Remove touched, record-on-creation and reset traversal. A cached index must
   refer to a current group whose existing left/right neighbor key matches.
2. Use IndexSet::insert_full to resolve constructed merge text once. Keep the
   borrowed lookup for initialization. Validate the returned index against u32
   and the separator afterward; errors discard the entire attempt.

API reference: https://docs.rs/indexmap/2.14.0/indexmap/set/struct.IndexSet.html#method.insert_full
Actual ignored Cargo.lock resolves indexmap 2.14.2, whose local implementation
and unchanged-order contract were checked directly. No dependency update.

Native checks compare full models and every selected pair/count/replacement at
1/4/8 workers, including reserved IDs, alias reuse, restart and numeric errors.
A directed neighbor fixture abandons a task on weighted overflow, reuses both
directories without reset, and validates stale out-of-range indices, mismatched
keys and current hits through complete ordered events.

Freeze four arms: baseline, directories only, insert only, combined. Exact sources,
patches, locks, build commands and binary hashes are recorded before measurement.
Common runner reads process fault counters outside the public do_train timer;
production code has no profiling or seed override.

Measure four existing Chinese/English 256 MiB Wikipedia preparations using
ByteLevel regex and Whitespace. Each sample reconstructs the same fixed-seed
caller map; internal production hashers stay at their default. Load and output
serialization are outside training timing. Run four workers pinned to CPUs 0–3.
Block 0 uses baseline/directories/insert/combined; block 1 reverses that order.
No discarded warmup; input SHA256 validation warms ordinary OS cache. Run serially
without tests/builds/other benchmark arms. Compare every complete parsed model,
record sampled child swap/RSS and train major faults. HWM includes loading and
is not an isolated allocation measurement. Small paired changes are descriptive.

If a concern appears, retain all attempts and perform a targeted confirmation;
do not silently exclude failures or attribute unrelated Feed variation to these
trainer changes. No stable speed or total-memory gain is assumed.
