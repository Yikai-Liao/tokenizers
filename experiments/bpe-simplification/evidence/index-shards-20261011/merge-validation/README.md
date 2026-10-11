Merged the shard ownership refactor `fb0864a0` with remote `52ca2951`, which
includes compressed-position read sharing and ordinary BPE library simplifications.

`Shard::commit` retains local groups and candidates and now uses the remote
`BTreeMap` ordering directly, without collecting and sorting group keys. The
borrowed `Chunk` interface restored by the local change remains; its block reader
uses the remote shared `Compressed` layout and `partition_point` lower bound.
The remote restart-seed reader and duplicate-boundary tests are retained. Merge
preparation combines `Source::Chunk` with the remote `VecDeque` lookahead and
`find` cohort scan. Other remote corpus/vocabulary simplifications are retained.

Default and no-default-features library checks each pass 38 tests, excluding the
already reproduced baseline encoder roundtrip assertion. Clippy with warnings
denied and the changed-file formatter pass. `validation.json` records commands
and results; no-default and Clippy logs are included.

The full-training comparison uses the unmodified remote version as baseline and
the resolved merge as candidate. Both frozen runners use locked dependencies,
opt-level 3, fat LTO and one codegen unit. The input is the same prepared
Whitespace count file from 256 MiB Chinese, with 50000 vocabulary target,
minimum frequency 2 and four workers pinned to CPUs 0–3. Two independent series
each exclude one warmup pair and include four alternating formal pairs. No
builds or tests run during formal measurements. Input loading and serialization
are outside the training timer; output construction is included. Process HWM
includes input loading. `builds.json` retains binary/source hashes and the build
command, and each series retains its manifest, runs, summary and log.

All 20 complete parsed models agree and observed swap is zero. The table shows
formal-arm medians and median relative changes within paired blocks. Positive
changes mean an increase.

| Metric | Remote | Merged | Paired median change |
| --- | ---: | ---: | ---: |
| Training wall time | 19.812 s | 20.372 s | +2.18% |
| Training CPU time | 54.974 s | 55.136 s | +1.30% |
| Process HWM RSS | 1793.896 MiB | 1792.113 MiB | -0.11% |

Seven of eight paired wall-time changes are positive; CPU changes have mixed
signs. Wall time is modestly slower on this input and host, so these results do
not establish performance equivalence with the remote version. The comparison
covers the combined merge, including restoration of the borrowed Chunk interface;
it does not isolate the index refactor. Pre-merge measurements against c8e7348
remain in the parent directory and have a different baseline.
