# Weight-one-bucket change: DE vs H screening

## Scope

One 512 MiB screening call each, sequentially DE then H; these are single-call results, not a stability estimate. Both used the same corpus, `none reference 50000 2`, init/merge workers 4/4, atomic u32 corpus and original API. The prior 32 MiB DE `cycles:u` profile motivated testing `WeightLookup::weight`; it was a hot-path diagnostic, not a performance comparison. H changes the lookup fast path only. AHash physical ordering is not controlled, so one-call phase differences should be treated as screening evidence.

- Input: `zh-512m.txt`, 536,870,289 bytes, SHA-256 `a0d40d4102cba1e933f25e5ccd17552d2eaebaec4ef770bc39d8e46e740b4d4f`.
- DE commit `c8702374bd8a3812f8bca34cd53e21afe01632c3`; binary SHA-256 `8d260f87357da76d4ac0c90fddfcac5cc86632cb38ca94584cb9fd65df500bee`.
- H commit `00216d914186e42458d45e72276b13c700749c6a`; binary SHA-256 `97d890912374da73ab5f70f4c14ab6a296cfd395053b21817494302cf04f0cfd`.
- Complete model signature passed for both: SHA-256 `d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd`, vocab 50,000, 29,243 merges, 1,429,915 unique words. Both recorded `initial_count_backend=stable_radix16` and identical N/E/pairs: 204,660,029 / 203,230,114 / 2,697,517.

## Results

| Metric | DE | H | H / DE |
|---|---:|---:|---:|
| Train | 32.860 s | 26.538 s | 0.808 |
| Elapsed | 36.785 s | 30.574 s | 0.831 |
| Full initialization | 9.783 s | 5.874 s | 0.600 |
| Fused prepare | 11.320 s | 8.548 s | 0.755 |
| Delta, including AA | 11.602 s | 8.869 s | 0.764 |
| Commit | 5.762 s | 6.165 s | 1.070 |

Initial counting phases (milliseconds):

| Phase | DE | H | H / DE |
|---|---:|---:|---:|
| Weight lookup | 13.451 | 12.986 | 0.965 |
| Route compaction | 1,030.110 | 987.839 | 0.959 |
| Radix sort | 2,473.196 | 1,945.984 | 0.787 |
| Group count | 2,639.983 | 328.252 | 0.124 |
| Posting install | 715.293 | 602.828 | 0.843 |

H reports 797,827 weight-one buckets out of 805,039 (99.104%), using a 100,632-byte bitmap. Weight lookup storage rises from 3,220,160 B in DE to 3,320,792 B in H. Initial corpus/posting storage stayed identical at 849,691,660 / 802,967,868 B.

The H call reported max RSS 4,649,436 KiB, minimum available memory 3,795,484,672 B (3.53 GiB), and zero sampled process swap. DE reported max RSS 4,648,632 KiB, minimum available 3,729,002,496 B (3.47 GiB), and zero sampled process swap. Both passed model and configuration gates.

This result supports the H fast path for another review and stability decision. The single call does not estimate repeatability; the unchanged radix-sort and posting-install code also varied between calls, and these differences should not be attributed to the lookup change. Full raw records, source hashes, diagnostics and signature results are in `optimization-weight-screen.{de,h}.*` and `.signature.json` sidecars.
