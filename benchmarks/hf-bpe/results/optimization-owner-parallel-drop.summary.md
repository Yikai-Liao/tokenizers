# H → K owner-drop screen (512 MiB)

This is one serialized H→K call pair under the original native training API. It is a screening result, not a repeatability estimate. K changes owner-ledger destruction to `owners.into_par_iter().for_each(drop)` after constructing the trained payload and after the existing `merge_ms` endpoint; the merge algorithm and public stats fields are unchanged.

## Fixed setup and provenance

- Input: `zh-512m.txt`, 536,870,289 bytes, SHA-256 `a0d40d4102cba1e933f25e5ccd17552d2eaebaec4ef770bc39d8e46e740b4d4f`.
- Parameters: reference backend, split `none`, vocab 50,000, min frequency 2, init/merge workers 4/4, atomic corpus, `parallel_u32_flat32`.
- H: clean worktree `weight-one-buckets`, commit `00216d914186e42458d45e72276b13c700749c6a`, binary SHA-256 `97d890912374da73ab5f70f4c14ab6a296cfd395053b21817494302cf04f0cfd`.
- K: clean worktree `owner-parallel-drop`, commit `288ad858d8993aabc5c9a6e1349e7413337bb4d4`, binary SHA-256 `7c8d91cc45dbbfcc285e5ceac7b808e216cc77a72922222a79fab1d0ab848344`.
- The 47-test suite passed before this pair with `cargo test --offline --no-default-features --lib` (debug profile); the native benchmark runner was built in release mode. Both calls passed the runner's configuration and resource gates. A complete model-signature check passed for both: SHA-256 `d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd`, 50,000 vocab entries, 29,243 merges, 1,429,915 unique words. `initial_count_backend` was `stable_radix16` for each.

## Results

`Direct residual` is `train_ms - initialize_ms - merge_ms`; `initialize_ms` already includes tokenization, so tokenization is not subtracted again. This residual includes output string materialization and owner/resource cleanup after the merge timer.

| Metric | H | K | K − H |
|---|---:|---:|---:|
| Train | 26.063 s | 26.862 s | +0.799 s (+3.1%) |
| Elapsed including feed | 30.310 s | 30.804 s | +0.494 s (+1.6%) |
| Initialize | 6.790 s | 6.902 s | +0.113 s |
| Merge | 15.439 s | 17.232 s | +1.793 s (+11.6%) |
| **Direct residual** | **3.835 s** | **2.727 s** | **−1.108 s (−28.9%)** |

The direct residual moved in the intended direction, while merge and full train moved in the opposite direction. Since the K code executes after the merge timer, the measured merge increase is not owner-drop time. With one call per version, this mixed phase movement is not enough to identify a cause or establish stable end-to-end benefit; no timing repeat was made.

## Resources and artifacts

| Resource | H | K |
|---|---:|---:|
| Peak RSS | 4,760,813,568 B | 4,760,719,360 B |
| Minimum available memory | 3,571,191,808 B | 3,806,171,136 B |
| Process swap sampled | 0 B | 0 B |
| Host `pswpin` / `pswpout` delta | 71 / 1,478 pages | 2,759 / 58,116 pages |

The host-wide paging counters are recorded as observed and are not attributed to these processes. The available-memory cutoff was 1 GiB; neither call approached it.

Raw JSONL results, environment/provenance, stdout, and stderr are in `optimization-owner-parallel-drop.{h,k}.{jsonl,environment.json,stdout,stderr}`. The signature gate result is `optimization-owner-parallel-drop.signature.json`. The separate cost-boundary diagnosis that motivated K is documented in [optimization-h-cost-probe.md](optimization-h-cost-probe.md); its instrumented timing is diagnostic and is not used in this H→K ranking.
