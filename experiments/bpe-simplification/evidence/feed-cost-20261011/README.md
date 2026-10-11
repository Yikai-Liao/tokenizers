# BPE feed capacity and scaling follow-up

Baseline is the same pre-feed revision `52ca295163936be3cf81cb91a3a8d654c78ac6d4`. Initial A and its 66 samples remain in [the unification report](../feed-unification-20261011/README.md). This follow-up answers the observed Chinese four-worker feed time/peak differences and removes one unnecessary operation. Parity Trainer remains outside scope.

## Integrated change

Keep A's named accumulation/flush closures, local 2048-key aggregation, startup singles, 32-line batches and fold/reduce completion. Remove `shared.reserve(shared.capacity())` before moving SCC entries into the output vector. There are no concurrent writers after aggregation. In pinned SCC 3.8.8, `iter_mut_sync` consumes entries while traversing and `for_each_writer_sync` checks shrinking after that traversal finishes. The reservation is unnecessary for this owned, finished table, and `reserve_capacity` itself calls `try_resize`. The default random hasher and aggregation growth policy remain in use. No initial global capacity, deterministic production seed or new dependency is introduced.

The attempted restoration of a single result collection is an archived alternative. In the matched-seed diagnostic matrix it adds median +5.34% feed wall time versus baseline, while initial A changes +0.24%. It offers no measured solution to the reported cost and is not integrated. The serial path is unchanged in this follow-up.

## Why the earlier high/low peaks differ

The earlier +8.76% feed wall median is an observation from five independently randomized pairs, not proof of a fixed A overhead. Both baseline and initial A show low and high peaks. With three identical controlled seed arrays across the three arms, all nine complete runs end aggregation at capacity 8,388,608 and roughly 929 MiB cumulative process HWM through feed. Initial A's median paired feed wall is +0.24%, feed CPU +0.41%, and feed HWM −0.004%. These are diagnostic comparisons, including identical trace overhead, and are not substituted for production-hasher timings.

Normal-random-hasher feed-only probes confirm that both baseline and initial A can end aggregation at capacity 16,777,216 and about 1260 MiB HWM. One baseline probe and one A probe instead change capacity 8,388,608 → 16,777,216 between the aggregation and post-reservation markers, reaching approximately 1473 MiB by the end of consumption; both later return to capacity 8,388,608. No worker is inserting between those markers: the reservation invokes another resize check. These observations isolate a real avoidable collection-phase allocation, while not assigning every high reading to the same phase.

SCC uses sampled occupancy and multiple rechecks, not an exact unique-key count threshold on every insertion. Its enlargement predicate is `num_entries > (capacity / 16) * 13`. At capacity 8,388,608 that threshold is 6,815,744; this 256 MiB input has 6,859,987 keys. Hash distribution, sampling and insertion progress can therefore affect capacity choices near this boundary. The controlled seed arrays are [11,13,17,19], [2,3,5,7] and [101,103,107,109]. This is a mechanism supported by pinned source and phase observations, not a claim that A uniquely forces growth. [Source excerpts](scc-source-excerpts.txt) include reserve, traversal and sampling code.

The revised six 256 MiB feed-only probes all remain at capacity 8,388,608 before consumption and about 929 MiB HWM; their empty table reaches capacity zero afterward. Normal aggregation can still grow before consumption, including in full production measurements. Removing the reservation eliminates its resize opportunity, not all SCC growth or all random variation. Feed-only probes do not train or validate complete counts/models; the separate full pipeline matrix does.

## Production scaling measurement

Chinese Wikipedia at fixed revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, configuration `20231101.zh`, using verified shard SHA256 `853ddf4a138c792ad7386d6de831788f96326c03e1e3349f7a315498392e8b56`. All four corpora are ordered nested prefixes after Unicode whitespace collapse, strip/LF and 32–8192-byte paragraph filtering. `prepare-scales.py` reproduces and verifies the four expected source hashes from this verified shard using PyArrow. The 128 MiB file is a prefix of the existing 256 MiB file; that file is a prefix of 384 and 512 MiB. The latter two hashes match the earlier scaling inputs, but earlier server timings are not reused. Actual bytes, hashes and recipe manifests are archived; the larger files contain additional source paragraphs, not repetitions of 256 MiB.

Four workers pinned to CPUs 0–3, ByteLevel with regex, public streaming `feed` followed by `train_vocab`, vocabulary 50000, minimum frequency two, no affixes. Release opt-level 3, fat LTO and one codegen unit. Two rotating-order blocks per scale; all runners run serially without competing builds/tests/benchmarks, source SHA256 and strict UTF-8 checks warm ordinary OS cache before the matrix, and no corpus is preloaded. The binaries use production `RandomState::default` with no capacity traces. `production/builds.json` verifies identical common runner/test/dependency source hashes. Positive percentages mean more time or memory. Small paired changes on this shared laptop are descriptive and not a stable speedup claim.

| Source MiB | A feed vs baseline | Revised feed vs baseline | Revised pipeline vs baseline | Revised feed vs A | Revised pipeline vs A |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | +0.74% | +0.02% | +0.10% | -0.72% | -0.82% |
| 256 | -8.60% | -2.99% | +0.11% | +6.13% | +2.91% |
| 384 | +0.39% | +1.03% | -0.56% | +0.63% | -0.23% |
| 512 | +1.27% | +1.08% | -0.40% | -0.19% | +0.47% |

Medians below are cumulative process high-water marks through feed / before post-timing validation, in MiB. They are not exclusive phase allocations. Larger training peaks can hide changes in feed peaks.

| Source MiB | Baseline HWM | A HWM | Revised HWM | Revised feed CPU vs baseline | Revised pipeline CPU vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | 550.1 / 1415.8 | 549.4 / 1455.9 | 553.3 / 1442.8 | +0.30% | +0.08% |
| 256 | 1252.1 / 2363.2 | 929.6 / 2369.8 | 1094.2 / 2391.7 | -3.88% | -0.78% |
| 384 | 1499.9 / 3211.4 | 1499.9 / 3240.8 | 1499.9 / 3235.0 | +1.16% | +0.05% |
| 512 | 1798.7 / 4283.8 | 1798.6 / 4241.5 | 1798.6 / 4241.4 | +0.64% | -0.87% |

All wall/CPU times, current RSS, cumulative HWM, minor/major faults, source-byte throughput, commands and sample order remain in `production/scaling/{runs,summary,manifest,sample-artifacts}.json`. Each successful sample exports the complete counts outside the timer, externally sorts JSON-key/tab/frequency records and compares SHA256, then compares complete parsed vocabularies and ordered merges. Large exports are deleted only after equality validation.

## Correctness and limitations

The final scaling matrix has 24 attempts: 24 successful full-count/model matches and 0 failures. All successful formal samples have zero sampled child swap; max feed/pipeline major faults are 0. The controlled matrix adds nine matching complete-count/model runs; normal-hasher probes total 27 successful feed-only processes.

An earlier initial A attempt panicked during training at `corpus.rs:208`, “coordinate belongs to the resident corpus”. It is preserved in `initial-attempt`, excluded from performance statistics and not labeled as a known baseline failure. A manual repeat of the same frozen A binary/input succeeded with identical complete counts and model; it had no swap poller and is excluded from formal timing conclusions. Full backtraces are enabled in subsequent formal attempts. This isolated training failure has not been reproduced or causally attributed; deleting the reservation is not claimed to fix it. See `retry-artifacts.json`.

Revised default and no-default library suites each pass 36 tests with the previously reproduced encoder-roundtrip test filtered. Unfiltered revised default records 36 passes plus that existing test failure, separate from the new corpus-coordinate observation. All-target Clippy with warnings denied and changed-file formatting pass. Integrated feed source/test hashes equal the frozen revised production arm, which has no diagnostic seed/trace code.

## Reproduction

`production/source/{baseline,unified,drain}` retains exact changed source/dependency snapshots for uninstrumented binaries; apply these over baseline `52ca2951`. Rebuild each arm offline/locked in its own worktree and freeze its release binary before the next build. Adjust binary/source paths in `production/builds.json` and `production/inputs.json`; run `BPE_FEED_ARMS=baseline,unified,drain BPE_FEED_BLOCKS=2 BPE_FEED_CASE=bytelevel_regex-w4,zh128-bytelevel_regex-w4,zh384-bytelevel_regex-w4,zh512-bytelevel_regex-w4 python production/measure.py`.

`source/{baseline,unified,collected}` and their patches are matched-seed diagnostic snapshots. `source/probe-*`, their patches and `probe.py` are normal-hasher feed-only diagnostics. The probe runner adds a separate early return after feed and the phase traces; neither is integrated. Reproduction keeps these three protocols separate. Restore the exact input manifests and corpus recipe; no models, corpora or binaries are committed. `final-production.patch` includes production/test/documentation changes and excludes runner diagnostics.
