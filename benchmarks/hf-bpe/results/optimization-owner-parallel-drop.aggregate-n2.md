# Owner parallel drop screen: aggregate of two serialized pairs

This combines the separately reported H→K pair and reverse-order K→H pair. Each version has two calls, so these arithmetic means are descriptive only; they are not a confidence interval or a repeatability estimate. See `optimization-owner-parallel-drop.summary.md` and `optimization-owner-parallel-drop-reverse.summary.md` for pair-level results and setup.

| Metric (mean of 2 calls) | H | K | K − H |
|---|---:|---:|---:|
| Train | 25.415 s | 25.719 s | +0.304 s (+1.2%) |
| Elapsed including feed | 29.565 s | 29.880 s | +0.315 s (+1.1%) |
| Initialize | 6.015 s | 6.374 s | +0.359 s (+6.0%) |
| Merge | 15.415 s | 16.640 s | +1.225 s (+7.9%) |
| **Direct residual** (`train - initialize - merge`) | **3.985 s** | **2.705 s** | **−1.280 s (−32.1%)** |

The direct residual moved in K's favor in each pair, while mean train, elapsed, initialization, and merge were higher. The end-to-end result remains mixed; this bounded screen does not support promoting K as an established full-operation improvement. The model signature and backend gates matched across all four calls. All four passed the resource gate with zero sampled process swap. Host-wide paging differed between calls and is reported at pair level without attribution.

Per-call records are `optimization-owner-parallel-drop.{h,k}.jsonl` and `optimization-owner-parallel-drop-reverse.{k2,h2}.jsonl`. The combined signature check is `optimization-owner-parallel-drop.all4.signature.json`.
