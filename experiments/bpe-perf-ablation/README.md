# BPE performance ablations

Base/control: `36c819c5`. This diagnostic branch publishes independent experiment trees in consecutive commits. Use the named commit for each experiment, rather than assuming later commits combine earlier changes.

The scratch-reuse commit changes only `NeighborScratch` lifetime: cache the two numeric directories on their executing thread between joined preparation rounds. Pool entry, initial indexing, position encoding, commit and selection keep their control behavior. The cache intentionally survives a training call; it contains numeric lookup entries, not corpus references. This is a diagnostic prototype, not a production lifetime design.

Measure the same frozen Chinese ByteLevel input, vocabulary 100000, min_frequency 2, workers 1/4 locally or 8 on the fixed-frequency host. Compare each variant to the base/control using three paired blocks and identical compiler, features, release flags and CPU affinity. Check complete vocabulary IDs and ordered merges; keep model serialization outside train timing. Record train wall/CPU and preparation+commit separately and combined. Compare real work counters in a separate diagnostic run to avoid perturbing formal timing.

Useful counters: preparation task/commit owner work distribution; cumulative initialized neighbor-directory cells; birth streams and positions; streams of length 1/2; below-floor births discarded after preparation; allocation calls/bytes by phase. The pool-only comparison already exists and need not be repeated.

Local standalone artifacts are in `/root/code/tokenizers-perf-regression-20261014`; these host paths are not portable. The branch's Rust changes apply directly to this repository and need no diagnostic artifact directory.

Validation: `cargo test --offline --manifest-path tokenizers/tk-train/Cargo.toml --lib --no-default-features trainers::bpe` passed all 32 selected tests, including the serial-reference comparisons and overflow/restart cases.
