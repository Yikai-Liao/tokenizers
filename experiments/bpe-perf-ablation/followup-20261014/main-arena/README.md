# main AllocationArena ablation

This isolates Arena's role in main commit `e4f787dc189d9be7192107490d652096cde7480e`. The `main-no-arena.patch` replaces only the Arena backend. It applies cleanly to the base commit above. `SortedPositions` keeps its 16-byte descriptor, inline cases, encoding, owned heap layout, and `Drop`; every non-inline allocation uses the existing `std::alloc` route. The arena lease becomes stateless and no longer locks worker cursors. Merge, indexing, scratch, thread scheduling, and the rest of the training path stay byte-identical. `main-detail.patch` adds the stage probes used by the observer binary; applying `main-no-arena-detail.patch` after it disables Arena for that observer. Neither patch is needed for the ordinary timings.

The no-Arena test snapshot also changes one test assertion: the existing concurrent-reader test now expects no Arena ownership tag. It retains all value, restart, and concurrent-reader checks. The ordinary BPE test command passed 94 selected tests (seven unrelated filtered). The release binary was built from the same production source; the test-only change is behind `cfg(test)`.

## Result

Chinese ByteLevel, 256 MiB, vocab 100,000, `min_frequency=2`, four workers pinned to CPUs 0–3. Each timing arm ran twice in reversed order. System allocator and mimalloc use the same executable within each source variant; mimalloc 3.0.1 was loaded with `LD_PRELOAD` and its mapping was verified. Every run produced the same complete model (`db1ddf...49670`), and observed swap was zero.

| main source | allocator | Train wall median (s) | Train CPU median (s) | Peak MiB median |
| --- | --- | ---: | ---: | ---: |
| Arena | system | 16.010 | 52.977 | 2723 |
| No Arena | system | 18.424 | 56.110 | 2466 |
| Arena | mimalloc | 14.444 | 47.358 | 3766 |
| No Arena | mimalloc | 14.429 | 47.205 | 3684 |

Removing Arena raised system-allocator wall time by 2.414 s (15.1%). Under mimalloc, the Arena and no-Arena medians differ by 0.015 s. On this workload the Arena mainly shields repeated position-buffer allocations and releases from the system allocator; mimalloc recovers the measured difference. The no-Arena system run also has a lower process peak, so the timing change is not caused by a larger peak working set.

The detailed pair has separate phase observers. Under the system allocator, `retire` rises from 0.074 s with Arena to 1.189 s without it. Under mimalloc, it rises from 0.070 s to 0.227 s. This retirement phase drops the index, corpus, and arena after merge rounds. The observer also records lower-level worker CPU spans, but those spans overlap and must not be added to wall time. `results.json` retains the complete phase and observer values. Timings with observers are separate from the ordinary pair above.

This proves that Arena materially helps main on the system allocator and that mimalloc removes that penalty in this four-worker case. It does not account for the whole older-buffer/main gap: under mimalloc, the older buffer implementation still takes 19.923 s versus 15.412 s for main in the separate paired diagnostic. Those remaining costs are in the older preparation, initial-count, materialization, and commit paths; the artifacts in the parent report contain their stage measurements.

## Reproduction

From the repository root at the main commit, apply `main-no-arena.patch`, build the locked release runner with the same flags and lock as main, then run the existing core benchmark harness for `zh-bytelevel`, four workers, vocab 100000, and two rounds. The patch includes the one Arena-specific test expectation described above. Add `LD_PRELOAD=<mimalloc.so>` for the mimalloc arm. The local frozen source snapshots, runner locks, and raw process records remain in `/root/code/tokenizers-perf-regression-20261014`; this host path and its input files are not part of the Git branch.
