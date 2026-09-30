# Weight-one-bucket stability results (two completed pairs)

## Scope and provenance

The requested DE→H, H→DE, DE→H sequence stopped after two completed pairs, following the instruction to stop once the initial gain was clear. This report covers only p1 and p2 (n=2 per candidate); there is no p3 result and no n=3 estimate.

All calls used the same 536,870,289-byte corpus, reference backend, no normalization, vocab 50,000, minimum frequency 2, init/merge workers 4/4, atomic u32 corpus, and the original API. Input SHA-256: `a0d40d4102cba1e933f25e5ccd17552d2eaebaec4ef770bc39d8e46e740b4d4f`.

| Candidate | Worktree commit | Binary SHA-256 |
|---|---|---|
| DE | `c8702374bd8a3812f8bca34cd53e21afe01632c3` | `8d260f87357da76d4ac0c90fddfcac5cc86632cb38ca94584cb9fd65df500bee` |
| H | `00216d914186e42458d45e72276b13c700749c6a` | `97d890912374da73ab5f70f4c14ab6a296cfd395053b21817494302cf04f0cfd` |

All four runs passed the stable radix gate and full model signature check: SHA-256 `d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd`, vocab 50,000, 29,243 merges, 1,429,915 unique words. Every call also matched N/E/pairs 204,660,029 / 203,230,114 / 2,697,517 and initial corpus/posting bytes 849,691,660 / 802,967,868.

## Per-call observations

Times are seconds except initial subphases, which are milliseconds. RSS is max RSS in KiB; available is minimum host `MemAvailable` in bytes. Weight buckets are emitted by H.

| Block/candidate | Train | Elapsed | Init | Merge | Weight lookup | Route compact | Radix sort | Group count | Posting install | Prepare | Delta incl. AA | Commit | H one buckets / total | H bitmap B | RSS KiB | Min available B | VmSwap B | CPU user/sys s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1.de | 32.776 | 37.345 | 8.211 | 20.103 | 9.929 | 811.204 | 1954.977 | 2801.997 | 588.206 | 12.399 | 12.708 | 6.173 | — | — | 4,648,392 | 3,808,235,520 | 0 | 93.538/6.224 |
| p1.h | 26.909 | 31.133 | 5.829 | 16.232 | 14.426 | 841.904 | 1915.189 | 335.949 | 654.798 | 8.516 | 8.850 | 6.189 | 797,920/805,039 | 100,632 | 4,648,036 | 3,748,347,904 | 0 | 72.250/6.608 |
| p2.h | 24.246 | 28.544 | 5.331 | 14.781 | 12.796 | 735.650 | 1768.201 | 318.070 | 594.210 | 7.824 | 8.129 | 5.577 | 797,808/805,039 | 100,632 | 4,649,212 | 3,753,746,432 | 0 | 66.532/5.543 |
| p2.de | 32.018 | 36.188 | 8.959 | 19.014 | 12.423 | 805.215 | 1892.054 | 3733.820 | 691.709 | 11.752 | 12.064 | 5.798 | — | — | 4,648,872 | 3,733,995,520 | 0 | 89.485/5.852 |

## Medians, ranges and sample CV (n=2)

CV is sample standard deviation (`ddof=1`) divided by the mean. With two observations these descriptive values have high uncertainty; retain the individual calls and paired ratios above.

| Phase | DE median (range; CV) | H median (range; CV) |
|---|---:|---:|
| initial weight lookup | 0.011 s (0.010–0.012; 15.8%) | 0.014 s (0.013–0.014; 8.5%) |
| route compaction | 0.808 s (0.805–0.811; 0.5%) | 0.789 s (0.736–0.842; 9.5%) |
| radix sort | 1.924 s (1.892–1.955; 2.3%) | 1.842 s (1.768–1.915; 5.6%) |
| group count | 3.268 s (2.802–3.734; 20.2%) | 0.327 s (0.318–0.336; 3.9%) |
| posting install | 0.640 s (0.588–0.692; 11.4%) | 0.625 s (0.594–0.655; 6.9%) |
| initialization | 8.585 s (8.211–8.959; 6.2%) | 5.580 s (5.331–5.829; 6.3%) |
| fused prepare | 12.075 s (11.752–12.399; 3.8%) | 8.170 s (7.824–8.516; 6.0%) |
| delta incl. AA | 12.386 s (12.064–12.708; 3.7%) | 8.490 s (8.129–8.850; 6.0%) |
| commit | 5.986 s (5.798–6.173; 4.4%) | 5.883 s (5.577–6.189; 7.4%) |
| merge | 19.558 s (19.014–20.103; 3.9%) | 15.507 s (14.781–16.232; 6.6%) |
| train | 32.397 s (32.018–32.776; 1.7%) | 25.577 s (24.246–26.909; 7.4%) |
| elapsed | 36.766 s (36.188–37.345; 2.2%) | 29.838 s (28.544–31.133; 6.1%) |

## Paired ratios (H / DE)

Values below 1 favor H. Ratios compare calls within each completed block; the final column is the median of the two ratios.

| Phase | p1 | p2 | Median ratio |
|---|---:|---:|---:|
| Initial weight lookup | 1.453 | 1.030 | 1.241 |
| Route compaction | 1.038 | 0.914 | 0.976 |
| Radix sort | 0.980 | 0.935 | 0.957 |
| Group count | 0.120 | 0.085 | 0.103 |
| Posting install | 1.113 | 0.859 | 0.986 |
| Full initialization | 0.710 | 0.595 | 0.652 |
| Fused prepare | 0.687 | 0.666 | 0.676 |
| Delta incl. AA | 0.696 | 0.674 | 0.685 |
| Commit | 1.003 | 0.962 | 0.982 |
| Merge | 0.807 | 0.777 | 0.792 |
| Train | 0.821 | 0.757 | 0.789 |
| Elapsed | 0.834 | 0.789 | 0.811 |

H bucket counts varied slightly with the default randomized hash state: p1 797,920/805,039 = 99.116%, p2 797,808/805,039 = 99.102%. The bitmap was 100,632 B in both calls; total weight lookup storage was 3,320,792 B. Both calls had greater than 99.10% one-weight buckets.

Both candidates recorded zero sampled process swap. Minimum available memory across all four calls was 3,733,995,520 B (3.48 GiB); max RSS was about 4.43 GiB. Sidecars retain per-call host paging, faults, context switches, source hashes, commands, and full provenance. The planned third pair was not run.
