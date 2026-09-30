# Reverse-order K → H owner-drop screen

This is the separately labeled reverse-order pair for the owner-ledger destruction change. It uses one call per version and remains a screening result, not a repeatability estimate. K executes `owners.into_par_iter().for_each(drop)` after constructing the trained payload and after the existing `merge_ms` endpoint; the merge algorithm and public stats fields are unchanged.

## Setup and gates

The pair reused the original native training API and fixed setup: `zh-512m.txt` (536,870,289 bytes, SHA-256 `a0d40d4102cba1e933f25e5ccd17552d2eaebaec4ef770bc39d8e46e740b4d4f`), reference backend, split `none`, vocab 50,000, min frequency 2, initialization/merge workers 4/4, atomic corpus, and `parallel_u32_flat32`.

H is worktree `weight-one-buckets`, commit `00216d914186e42458d45e72276b13c700749c6a`, binary SHA-256 `97d890912374da73ab5f70f4c14ab6a296cfd395053b21817494302cf04f0cfd`. K is worktree `owner-parallel-drop`, commit `288ad858d8993aabc5c9a6e1349e7413337bb4d4`, binary SHA-256 `7c8d91cc45dbbfcc285e5ceac7b808e216cc77a72922222a79fab1d0ab848344`.

The runner configuration/resource gates passed. Both reported `initial_count_backend=stable_radix16`. The four-run model signature checker passed: SHA-256 `d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd`, 50,000 vocab entries, 29,243 merges, 1,429,915 unique words.

## Reverse-pair results

`Direct residual` is `train_ms - initialize_ms - merge_ms`. It includes output string materialization and cleanup after the merge timer; initialization already includes tokenization.

| Metric | K2 | H2 | K2 − H2 |
|---|---:|---:|---:|
| Train | 24.576 s | 24.766 s | −0.191 s (−0.8%) |
| Elapsed including feed | 28.956 s | 28.819 s | +0.136 s (+0.5%) |
| Initialize | 5.845 s | 5.240 s | +0.605 s |
| Merge | 16.047 s | 15.391 s | +0.656 s (+4.3%) |
| **Direct residual** | **2.683 s** | **4.136 s** | **−1.453 s (−35.1%)** |

The direct residual improves in both observed pairs. Full train is slower for K in the H→K pair and faster in this K→H pair; elapsed is slower for K in both. Merge is slower for K in both. The changed destruction runs after `merge_ms`, so that timer difference is not the destruction work itself. This two-pair screen does not establish a stable end-to-end benefit or explain phase variation.

## Resources and artifacts

| Resource | K2 | H2 |
|---|---:|---:|
| Peak RSS | 4,761,686,016 B | 4,765,290,496 B |
| Minimum available memory | 3,736,477,696 B | 3,735,973,888 B |
| Process swap sampled | 0 B | 0 B |
| Host `pswpin` / `pswpout` delta | 1,557 / 0 pages | 635 / 0 pages |

Host paging counters are system-wide observations and are not attributed to these processes. Raw records and sidecars use the `optimization-owner-parallel-drop-reverse.{k2,h2}` prefix; the four-run gate is `optimization-owner-parallel-drop.all4.signature.json`.
