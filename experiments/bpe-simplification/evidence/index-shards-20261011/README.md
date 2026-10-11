The index now has one private `Shard` for each count owner. It owns the count
map, compact action vector and birth-stream vector. `PairIndex` commits those
shards in parallel, then pushes joined candidates into the existing owning heap.
`Route`, `OwnerCommit`, `Group` and `GroupPositions` are removed. Initial
aggregation and partial birth grouping share `State<P>`; commit groups use
`State<SmallVec<[u64; 2]>>`. `Shard::commit` keeps its groups and candidates local
and follows action consumption, count adjustment, position gathering and
publication in one algorithm.

The production position-range experiment in `bc944215` is reverted. Position
storage and merge preparation match `c8e7348`; its successful-batch record
interface is retained. Historical position-range evidence is preserved. Against
`c8e7348`, the changed BPE files are index.rs and its design/coverage documentation.

Removal precedes birth, signed arithmetic checks each action, grouped birth
weights retain checked addition, complete producers precede bucket/pair-sorted
partial groups, and reuse retains duplicate coordinates. Fresh fragments append
in order and transfer their first allocation. Reuse appends copied slices into
owner-local storage and sorts before freezing. `Priority`, owning `Candidate`,
dense/sparse initial counting, and separate action/birth storage remain.

Both default and no-default-features library runs pass 33 checks after excluding
the existing encoder roundtrip assertion. Clippy with warnings denied, the
changed-file formatter and diff whitespace checks pass. Added regressions cover
reuse bucket order, duplicates, zero-weight births and below-floor positive
cohorts; fresh retirement and complete/partial publication; and signed/grouped
unsigned overflow that drains pending streams while retaining capacity.
`validation.json` and logs preserve commands and results. The encoder assertion
for “测” (actual ID 3, expected ID 6) is reproduced on the unmodified c8e7348 baseline.

The 2048-ID generated reuse fixture exposed pre-existing output-order variability:
two runs of the same c8e7348 binary and input preserve vocabulary IDs but differ
in ordered merges. `baseline-reuse-variability/` records both outputs. A diagnostic
fixes only the initial aggregate map hasher, giving both arms the same initial
heap order. Within each worker count, the exact production commit implementations
then produce identical complete vocabulary IDs and ordered merges in all six
runs (12 total at 1/4 workers). Output order can still differ between worker
counts; this diagnostic checks preservation against the corresponding baseline.
`check_seeded_reuse.py`, diagnostic patches, build hashes and comparisons are in
`seeded-reuse/`. No production hasher or heap-order policy was changed.

On the measured x86_64 target with the project's locked SmallVec 1.16.2 and union
feature, the group value shrinks from 40 to 32 bytes and PairIndex from 88 to 64
bytes. Actions remain 16 bytes, separately stored Birth values 32 bytes. A Shard
is 112 bytes, equal to the former 64-byte map plus 48-byte Route. In a synthetic
8192-group, two-position reuse commit, measured allocator calls fall from 8234
to 42. These layout/allocation diagnostics describe storage costs, not full
training throughput. `layout.json` and `commit_probe.py` retain the observations
and source/dependency hashes. Initial exploratory micro timings under SmallVec
1.16.3 are archived in `initial-append/`; they are not the performance acceptance
criterion.

Performance acceptance uses complete public `do_train` on prepared Whitespace
counts from 256 MiB Chinese, 50000 vocabulary target, minimum frequency 2, four
workers pinned to CPUs 0–3. Frozen baseline c8e7348 and final candidate runners use
the same locked dependencies, opt-level 3, fat LTO and one codegen unit. Each
sample is a separate process. Two independent series each exclude one warmup
pair before four alternating formal pairs (eight formal pairs total). Loading, preprocessing and model serialization are outside the
training timer; output construction is included. Process HWM includes loaded
input counts. No builds or tests run during formal measurements. Peak swap is
sampled every 25 ms. Input and binary hashes, host information, individual runs
and paired summaries are retained alongside this file.

All 20 complete parsed models agree and observed swap is zero. The table shows
formal-arm medians and the median relative change within each paired block.
Positive changes mean an increase. This comparison checks regression on this
input and host; it does not establish universal performance equivalence.

| Metric | Baseline | Candidate | Paired median change |
| --- | ---: | ---: | ---: |
| Training wall time | 17.997 s | 17.991 s | +0.49% |
| Training CPU time | 48.848 s | 48.474 s | +0.60% |
| Process HWM RSS | 1793.900 MiB | 1791.012 MiB | -0.01% |

The individual paired wall changes range from -7.87% to +6.52%;
CPU changes range from -8.53% to +6.01%. Their signs vary
across pairs. The observed median changes are small compared with this variation.
This run does not show a clear training performance regression. The change
is retained for ownership and representation simplification.

To reproduce the end-to-end comparison, build and freeze both release runners
with the command/profile in builds.json, then run:

```sh
python measure_training.py \
  --baseline BASELINE_BIN --candidate CANDIDATE_BIN \
  --zh ZH256_WHITESPACE_WORDS \
  --only zh --workers 4 --zh-label zh-whitespace-256MiB --zh-vocab 50000 \
  --output RESULT_DIR
```

The final acceptance run selects only the 256 MiB Chinese case.
