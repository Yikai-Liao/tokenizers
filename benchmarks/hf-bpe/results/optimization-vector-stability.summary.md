# DE versus G2 vector stability screening

## Scope and provenance

This report covers six additional counterbalanced calls, in order DE → G2, G2 → DE, DE → G2. It keeps the earlier one-call DE/G1/G2 vector screen separate; those single-call measurements are not included in the stability statistics below.

All runs used the same 536,870,289-byte corpus (SHA-256 `a0d40d4102cba1e933f25e5ccd17552d2eaebaec4ef770bc39d8e46e740b4d4f`), reference backend, no normalization, vocab 50,000, min frequency 2, init/merge workers 4/4, atomic u32 corpus and original API. Full signatures passed across the six runs: model SHA-256 `d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd`, vocab 50,000, 29,243 merges, 1,429,915 unique words. Every run asserted `initial_count_backend == stable_radix16`.

| Candidate | Worktree commit | Binary SHA-256 |
|---|---|---|
| DE | `c8702374bd8a3812f8bca34cd53e21afe01632c3` | `8d260f87357da76d4ac0c90fddfcac5cc86632cb38ca94584cb9fd65df500bee` |
| G2 | `2f238505b50c8555c92b4294fb7201247e35138a` | `87885ccdee0dd4950654b06f1cf86b8b89150ca5eba084643078b9612c119d15` |

## Stability calls

Times are seconds. Min available is the minimum host `MemAvailable`; CPU gives child user/system seconds; faults are minor/major counts; context switches are voluntary/involuntary counts.

| Block/config | Train | Elapsed | Init | Merge | Fused prepare | Delta incl. AA | Commit | RSS KiB | Min available B | VmSwap B | pswpin/out pages | CPU user/sys | Wall | Faults | Context switches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1.de | 31.571 | 35.914 | 8.553 | 18.774 | 11.330 | 11.620 | 5.998 | 4,649,040 | 3,694,477,312 | 0 | 979/424 | 87.652/7.865 | 37.593 | 651,837/0 | 22,669/22,755 |
| p1.g2 | 30.209 | 34.379 | 7.683 | 18.394 | 11.085 | 11.361 | 5.900 | 4,649,636 | 3,757,051,904 | 0 | 98/0 | 86.389/5.424 | 35.592 | 638,248/0 | 23,644/19,547 |
| p2.g2 | 32.073 | 36.349 | 8.011 | 19.726 | 11.948 | 12.255 | 6.271 | 4,647,816 | 3,723,874,304 | 0 | 365/0 | 90.810/5.682 | 38.097 | 641,024/0 | 23,892/30,829 |
| p2.de | 32.170 | 36.285 | 8.010 | 19.389 | 11.720 | 12.020 | 6.196 | 4,648,624 | 3,708,473,344 | 0 | 161/0 | 89.681/6.986 | 38.098 | 639,790/0 | 23,448/23,436 |
| p3.de | 29.514 | 33.556 | 7.339 | 17.937 | 10.906 | 11.188 | 5.631 | 4,648,788 | 3,714,994,176 | 0 | 73/0 | 84.439/5.619 | 35.089 | 643,586/0 | 23,766/20,131 |
| p3.g2 | 31.147 | 35.327 | 7.942 | 18.801 | 11.351 | 11.644 | 6.005 | 4,648,616 | 3,698,425,856 | 0 | 51/0 | 88.218/6.185 | 37.099 | 631,264/0 | 23,577/21,378 |

## Medians, ranges and sample CV

CV is sample standard deviation (`ddof=1`) divided by the mean, n=3 per candidate.

| Candidate | Metric | Median | Range | CV |
|---|---|---:|---:|---:|
| DE | Fused prepare | 11.330 s | 10.906–11.720 s | 3.60% |
| DE | Delta incl. AA | 11.620 s | 11.188–12.020 s | 3.58% |
| DE | Train | 31.571 s | 29.514–32.170 s | 4.48% |
| DE | Elapsed | 35.914 s | 33.556–36.285 s | 4.20% |
| DE | Init | 8.010 s | 7.339–8.553 s | 7.63% |
| DE | Merge | 18.774 s | 17.937–19.389 s | 3.90% |
| DE | Commit | 5.998 s | 5.631–6.196 s | 4.82% |
| G2 | Fused prepare | 11.351 s | 11.085–11.948 s | 3.86% |
| G2 | Delta incl. AA | 11.644 s | 11.361–12.255 s | 3.89% |
| G2 | Train | 31.147 s | 30.209–32.073 s | 2.99% |
| G2 | Elapsed | 35.327 s | 34.379–36.349 s | 2.79% |
| G2 | Init | 7.942 s | 7.683–8.011 s | 2.19% |
| G2 | Merge | 18.801 s | 18.394–19.726 s | 3.60% |
| G2 | Commit | 6.005 s | 5.900–6.271 s | 3.16% |

## Paired ratios (G2 / DE)

Values below 1 favor G2.

| Metric | p1 | p2 | p3 | Median ratio |
|---|---:|---:|---:|---:|
| Fused prepare | 0.9783 | 1.0195 | 1.0409 | 1.0195 |
| Delta incl. AA | 0.9777 | 1.0196 | 1.0408 | 1.0196 |
| Train | 0.9569 | 0.9970 | 1.0553 | 0.9970 |
| Elapsed | 0.9572 | 1.0018 | 1.0528 | 1.0018 |

G2's prepare and delta medians are each slightly higher than DE's in this stability set. G2 has a lower unpaired median train and elapsed time, while the paired median ratios are essentially a tie; the elapsed paired median slightly favors DE. Both candidates have limited n=3 samples, so keep their ranges and CV in view. The vector screen also measured G1 once but it is not part of this DE/G2 confirmation.

These are whole-candidate measurements. G2 uses the read-phase view and 128-position blocks; the measurements do not isolate SIMD as the cause of any whole-run difference. Initialization and commit also varied between calls, and should remain visible alongside the prepare and delta phases rather than being attributed to the vector change.

All six calls sampled zero process swap; minimum available memory was 3,694,477,312 B (3.44 GiB). Host paging deltas may include unrelated activity. Raw JSONL, environment/provenance, stdout/stderr, and signature files are named `optimization-vector-stability.p{1,2,3}.{de,g2}.*` in the results directory.
