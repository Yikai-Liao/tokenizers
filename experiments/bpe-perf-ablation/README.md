# BPE performance ablations

Base/control: `36c819c5`. This diagnostic branch publishes independent experiment trees in consecutive commits. Use the named commit for each experiment, rather than assuming later commits combine earlier changes.

Commit `20fdfc0a` (scratch reuse) changes only `NeighborScratch` lifetime: cache the two numeric directories on their executing thread between joined preparation rounds. Pool entry, initial indexing, position encoding, commit and selection keep their control behavior. The cache intentionally survives a training call; it contains numeric lookup entries, not corpus references. This is a diagnostic prototype, not a production lifetime design.

Measure the same frozen Chinese ByteLevel input, vocabulary 100000, min_frequency 2, workers 1/4 locally or 8 on the fixed-frequency host. Compare each variant to the base/control using three paired blocks and identical compiler, features, release flags and CPU affinity. Check complete vocabulary IDs and ordered merges; keep model serialization outside train timing. Record train wall/CPU and preparation+commit separately and combined. Compare real work counters in a separate diagnostic run to avoid perturbing formal timing.

Useful counters: preparation task/commit owner work distribution; cumulative initialized neighbor-directory cells; birth streams and positions; streams of length 1/2; below-floor births discarded after preparation; allocation calls/bytes by phase. The pool-only comparison already exists and need not be repeated.

Local standalone artifacts are in `/root/code/tokenizers-perf-regression-20261014`; these host paths are not portable. The branch's Rust changes apply directly to this repository and need no diagnostic artifact directory.

Validation: `cargo test --offline --manifest-path tokenizers/tk-train/Cargo.toml --lib --no-default-features trainers::bpe` passed all 32 selected tests, including the serial-reference comparisons and overflow/restart cases.

The next commit is a separate fresh-commit grouping experiment: it restores `merge.rs` to the control, and changes only fresh birth grouping in `index.rs`. Instead of comparison-sorting owning birth fragments, it buckets by replacement ID/direction and groups neighbors with a retained numeric directory. Ordered count actions and the reuse path keep their original code. This prototype is an attempted local reconstruction of one old optimization, not the old main implementation or its arena layout. The branch tip therefore contains grouping alone; checkout `20fdfc0a` for scratch alone.

Grouping did not recover the commit regression locally. In the English 4-worker diagnostic, two paired blocks had median commit 0.429 s (control) versus 0.469 s (grouping); total training was 2.401 versus 2.352 s and CPU 5.277 versus 5.505 s. In the Chinese 4-worker diagnostic, one pair had commit 5.021 versus 5.721 s and total training 24.960 versus 27.996 s. All complete models matched; this is diagnostic evidence, not a formal performance estimate.

The Chinese 8-worker external report establishes the main/current regression: 6.434 versus 10.836 s train wall medians over 36 paired blocks, +68.5% wall and +43.9% CPU. Its single stage diagnostic puts most extra time in preparation and commit. Compare preparation plus commit together because their position-encoding work is placed differently across versions. The main implementation's persistent directories, arena-backed birth streams and encoding/publication layout remain separate hypotheses; this grouping prototype does not isolate all of them.

Next useful evidence on that host: per-worker preparation/commit work and CPU distribution, phase allocation calls/bytes, birth-stream length distribution (especially length 1/2), and positions encoded for births later discarded below the frequency floor. Collect these in separate diagnostic runs, not in formal timing runs.

Pure scratch lifetime ablation, retaining the control's pool behavior: English 4-worker, two paired diagnostic blocks, train 2.386 -> 1.967 s and preparation 0.952 -> 0.617 s; cumulative directory initialization 5.436 billion -> 0.995 million u32 cells. Chinese 4-worker, one diagnostic pair, train 27.497 -> 25.068 s, preparation 11.498 -> 10.491 s, commit 5.774 -> 5.171 s, CPU 81.959 -> 77.934 s; directory initialization 7.040 billion -> 0.998 million cells. All complete models matched. This confirms repeated directory initialization is extra work, while this limited diagnostic does not establish an 8-worker speedup or resolve the remaining regression.

Individual measurements, phase times and model/binary hashes are in [local-diagnostics.json](local-diagnostics.json). These timings come from phase-instrumented copies of each experiment, using a matched Cargo.lock and identical release settings (opt-level 3, fat LTO, one codegen unit). The published Rust experiment trees contain no phase timers.
