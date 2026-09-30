# H → J rule aggregate screen (512 MiB)

One serialized H→J pair using the original native API. This is a screening result with one call per version. J adds per-rule task-local neighbor aggregation during fused prepare. No allocator or extra profiling run was performed.

## Setup and provenance

- Input: `zh-512m.txt`, 536,870,289 bytes, SHA-256 `a0d40d4102cba1e933f25e5ccd17552d2eaebaec4ef770bc39d8e46e740b4d4f`.
- Parameters: reference backend, split `none`, vocab 50,000, min frequency 2, initialization/merge workers 4/4, atomic corpus, `parallel_u32_flat32`.
- H: worktree `weight-one-buckets`, commit `00216d914186e42458d45e72276b13c700749c6a`, binary SHA-256 `97d890912374da73ab5f70f4c14ab6a296cfd395053b21817494302cf04f0cfd`.
- J: worktree `prepare-rule-aggregate`, commit `376363d25b6b1de917b9b8f68c11c0394ca37bb2`, binary SHA-256 `f4975f9e040ffd10f24fcc386fc7996c8aadfe2897f46faec1b2f729689ae6e2`.
- The 48-test suite passed in debug profile with `cargo test --offline --no-default-features --lib`; the native runners were release builds. The runner configuration/resource gates passed for both calls.

## Correctness and workload gates

The full model signature matches: SHA-256 `d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd`, 50,000 vocab entries, 29,243 merges, and 1,429,915 unique words. Both used `stable_radix16` for initial counting.

The workload and initial-storage counts also match exactly:

| Count / bytes | H | J |
|---|---:|---:|
| Initial symbols | 204,660,029 | 204,660,029 |
| Initial edges | 203,230,114 | 203,230,114 |
| Initial pairs | 2,697,517 | 2,697,517 |
| Initial corpus bytes | 849,691,660 | 849,691,660 |
| Initial posting bytes | 802,967,868 | 802,967,868 |
| Initial weight lookup bytes | 3,320,792 | 3,320,792 |

## Timings

| Metric | H | J | J − H |
|---|---:|---:|---:|
| Train | 27.790 s | 24.613 s | −3.177 s (−11.4%) |
| Elapsed including input feed | 31.762 s | 28.701 s | −3.061 s (−9.6%) |
| Initialize | 7.467 s | 5.456 s | −2.011 s (−26.9%) |
| Merge | 15.726 s | 15.044 s | −0.683 s (−4.3%) |
| **Fused prepare** | **8.400 s** | **7.723 s** | **−0.677 s (−8.1%)** |

There is one measurement per version, in H→J order. The apparent improvements are useful screening evidence, but this pair cannot establish repeatability; initialization moved by 2.011 seconds, so the full-train difference should not be assigned to prepare alone.

## Prepare scratch capacity and resources

J's peak job-local aggregation scratch capacity was **2,106,304 bytes** (about 2.01 MiB); H has no corresponding counter. This reports the sum, for a prepared batch, of the `Vec` capacities for each worker job's left and right neighbor index directories (`u32` entries) and `LocalGroup` arrays, taking the maximum across batches. The caches are reused between rule tasks within that job. It does not include the `Vec` headers, allocator bookkeeping, valid-position arrays, route outputs, or other process memory, so it is a bounded component estimate rather than total prepare RSS. J's peak valid-start bytes remained 17,301,504 and peak selected-lookup bytes 800,128, both matching H.

| Resource | H | J |
|---|---:|---:|
| Peak RSS | 4,763,422,720 B | 4,761,026,560 B |
| Minimum available memory | 3,474,558,976 B | 3,513,487,360 B |
| Process swap sampled | 0 B | 0 B |
| Host `pswpin` / `pswpout` delta | 2,146 / 6,295 pages | 1,969 / 0 pages |

Host paging counters are system-wide observations and are not attributed to these processes. The raw records and sidecars are `prepare-rule-aggregate-screen.{h,j}.*`; the signature check is `prepare-rule-aggregate-screen.signature.json`.
