# Counterbalanced stability run: B2, D and DE

## Scope and provenance

Nine sequential, single-call 512 MiB screenings in the requested counterbalanced order: B2 → D → DE; DE → B2 → D; D → DE → B2. All used reference backend, no normalization, vocab 50,000, min frequency 2, initialization/merge workers 4/4, atomic corpus, u32, same original API. No E run was included.

Input: `benchmarks/hf-bpe/.build/gb-corpus/zh-512m.txt`, 536,870,289 bytes, SHA-256 `a0d40d4102cba1e933f25e5ccd17552d2eaebaec4ef770bc39d8e46e740b4d4f`. Full signature checker passed across all nine: SHA-256 `d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd`, vocab 50,000, 29,243 merges, 1,429,915 unique words. Every run also matched symbols/edges/pairs 204,660,029 / 203,230,114 / 2,697,517.

| Candidate | Worktree commit | Binary SHA-256 |
|---|---|---|
| B2 | `a0832c488a7ece429630c7d1493da5dd772c87aa` | `596200773ff413504d82aa4038b3c5e5902e835dff2bbdfb362e5d0818d208a5` |
| D | `d15c18cc07047479ddc2eacd1da1844cb9ac9358` | `4c7d1268470da0eaa02f6f48b242defd0ace0614f4461527ff26ddbf1430ac1b` |
| DE | `c8702374bd8a3812f8bca34cd53e21afe01632c3` | `8d260f87357da76d4ac0c90fddfcac5cc86632cb38ca94584cb9fd65df500bee` |

D and DE both passed `initial_count_backend == stable_radix16`. B2 retains its original count path. Each JSONL result has an environment/provenance sidecar with complete source hashes and an individual signature was checked against the B2 baseline.

## Per-call results

Times in seconds. RSS is process max RSS in KiB; available is the minimum host `MemAvailable` in bytes. CPU is child user/system seconds; faults are minor/major; context switches voluntary/involuntary.

| Block/config | Train | Elapsed | Init | Merge | Commit | Initial install | RSS KiB | Min available B | VmSwap B | pswpin/out pages | CPU user/sys | Wall | Faults | Context switches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1.b2 | 37.654 | 42.216 | 12.955 | 20.608 | 6.676 | — | 3609580 | 4,597,010,432 | 0 | 55/0 | 108.512/5.816 | 44.119 | 443001/0 | 26052/52815 |
| p1.d | 32.010 | 36.196 | 8.689 | 19.286 | 6.272 | 0.862 | 4651440 | 3,661,344,768 | 0 | 9/4768 | 91.249/7.112 | 38.094 | 702592/0 | 23078/23887 |
| p1.de | 34.701 | 38.710 | 8.643 | 21.990 | 6.834 | 0.643 | 4648608 | 3,647,287,296 | 0 | 56/2848 | 92.409/6.003 | 40.118 | 669876/0 | 22769/46699 |
| p2.de | 30.502 | 34.479 | 7.685 | 18.746 | 5.782 | 0.624 | 4649248 | 3,688,570,880 | 0 | 29/0 | 86.225/5.694 | 36.099 | 672074/0 | 23208/25618 |
| p2.b2 | 35.212 | 39.369 | 11.058 | 20.005 | 6.273 | — | 3554768 | 4,812,619,776 | 0 | 18/0 | 105.715/5.399 | 41.112 | 441263/0 | 24923/29576 |
| p2.d | 35.442 | 39.740 | 9.198 | 21.460 | 7.023 | 1.006 | 4648864 | 3,638,079,488 | 0 | 170/666 | 99.280/6.537 | 41.101 | 678122/0 | 23951/28800 |
| p3.d | 35.960 | 40.140 | 9.147 | 21.956 | 7.266 | 0.931 | 4648392 | 3,690,086,400 | 0 | 348/0 | 101.565/6.677 | 42.093 | 607224/1 | 24480/24676 |
| p3.de | 31.332 | 35.493 | 7.844 | 19.314 | 5.955 | 0.571 | 4648816 | 3,589,857,280 | 0 | 486/0 | 88.611/5.853 | 37.102 | 617703/0 | 23196/26536 |
| p3.b2 | 35.952 | 40.151 | 10.881 | 21.093 | 6.962 | — | 3621548 | 4,686,340,096 | 0 | 26/0 | 108.751/4.512 | 41.616 | 438689/0 | 23829/29100 |

## Stability and paired comparison

Medians, ranges and sample CV (standard deviation with `ddof=1`, divided by the mean) across n=3 calls per candidate:

| Candidate | Metric | Median | Range | CV |
|---|---|---:|---:|---:|
| B2 | train | 35.952 s | 35.212–37.654 s | 3.45% |
| B2 | elapsed | 40.151 s | 39.369–42.216 s | 3.63% |
| B2 | init | 11.058 s | 10.881–12.955 s | 9.89% |
| B2 | merge | 20.608 s | 20.005–21.093 s | 2.65% |
| B2 | commit | 6.676 s | 6.273–6.962 s | 5.21% |
| D | train | 35.442 s | 32.010–35.960 s | 6.23% |
| D | elapsed | 39.740 s | 36.196–40.140 s | 5.61% |
| D | init | 9.147 s | 8.689–9.198 s | 3.11% |
| D | merge | 21.460 s | 19.286–21.956 s | 6.79% |
| D | commit | 7.023 s | 6.272–7.266 s | 7.56% |
| DE | train | 31.332 s | 30.502–34.701 s | 6.91% |
| DE | elapsed | 35.493 s | 34.479–38.710 s | 6.10% |
| DE | init | 7.844 s | 7.685–8.643 s | 6.37% |
| DE | merge | 19.314 s | 18.746–21.990 s | 8.65% |
| DE | commit | 5.955 s | 5.782–6.834 s | 9.11% |

Paired ratios use target/reference within each block; values below 1 favor the target. These are reported in train and elapsed time as requested:

| Ratio | Train by block | Median | Elapsed by block | Median |
|---|---|---:|---|---:|
| D/B2 | 0.8501, 1.0065, 1.0002 | 1.0002 | 0.8574, 1.0094, 0.9997 | 0.9997 |
| DE/B2 | 0.9216, 0.8662, 0.8715 | 0.8715 | 0.9170, 0.8758, 0.8840 | 0.8840 |
| DE/D | 1.0841, 0.8606, 0.8713 | 0.8713 | 1.0695, 0.8676, 0.8842 | 0.8842 |

## Reading the counters

D changes initial counting to stable radix. Its whole initialization medians include route compaction, radix sorting, grouping, posting installation and the other initialization stages; the D/B2 paired ratios therefore compare the tested whole candidates, not an isolated radix sort.

DE adds bulk posting installation and owner-based bulk commit on top of D. Across these three counterbalanced blocks, DE is the fastest whole candidate measured so far: it has the lowest whole-train median (31.332 s versus D 35.442 s and B2 35.952 s) and elapsed median (35.493 s versus D 39.740 s and B2 40.151 s), and it beats B2 in every paired block (median train ratio 0.8715, elapsed ratio 0.8840). Keep the n=3 ranges and CV in view. These are measured whole-candidate results; they do not establish that bulk operations caused all of the difference. DE median initial posting install was 0.624 s (runs: 0.643, 0.624, 0.571) versus D 0.931 s (0.862, 1.006, 0.931), about 0.307 s lower. Median commit was 5.955 s for DE (6.834, 5.782, 5.955), 7.023 s for D (6.272, 7.023, 7.266), and 6.676 s for B2 (6.676, 6.273, 6.962). DE-to-B2 paired commit differences change sign across blocks (+0.158, -0.491, -1.007 s), so the commit effect is not stable here. The install and commit differences are much smaller than the whole-train median gap; changes and run-to-run variation in other phases are not attributed to E. The standalone E screening result remains in its own file; the present ranking is among the three full candidates tested in these counterbalanced blocks.

All nine sampled zero process swap. Minimum available memory ranged from 3,589,857,280 B (3.34 GiB) to 4,812,619,776 B (4.48 GiB); no call approached the 1 GiB stop threshold. Global paging counters may include unrelated host activity. Raw JSONL, environment, stdout/stderr, and signature sidecars are named `optimization-radix-combination-stability.p{1,2,3}.{b2,d,de}.*` in this directory.
