# One cursor for search and reading

Production commit `c1b6557a` implements the two requested read-path simplifications on top of `d9170827`. The only changed production file is positions.rs. It retains 32-byte Positions and the existing task grouping of small leaves.

The leaf binary search now returns only the aligned restart position. iter_from_value creates the reading iterator once and skip_while advances that same cursor to the first qualifying coordinate; equal values crossing leaf/restart boundaries remain visible. Cursor no longer decodes to an arbitrary element index. Its bytes slice ends exactly at the leaf data end, so bytes.is_empty() replaces Cursor.end. ReadTask still limits its own item count with take; tasks still group small leaves and split large leaves at restarts. Append, allocation growth and encoding are unchanged.

Default and no-default targeted runs each execute 29 passing BPE tests, excluding only the known JSON roundtrip failure reproduced in the previous report. Existing append/search/task tests cover empty and short leaves, duplicates across restart and leaf boundaries, u64 extremes, grouped tiny leaves and large split leaves. Clippy and rustfmt pass. Production AST count is 333→323 (+10/−20), and the full test budget is 1407→1406; the removed test line constructs the deleted Cursor.end field.

Normal paired comparisons below use the previous frozen 32-byte child versus this read refinement, serial fresh processes, 4 workers on CPUs 0–3, identical 256 MiB-derived prepared Chinese/English inputs, Whitespace/ByteLevel, vocab 50000 and min frequency 2. Every exported full model is byte-identical. AA and reuse controls also match. No child process swapped. Train wall excludes load/export and includes all training phases. This small comparison verifies the change without claiming a repeatable overall speedup; it does not re-evaluate the previously accepted English single-worker overhead.

| Case / suite | Base train s | Candidate train s | Change | Base CPU s | Candidate CPU s | Base merge s | Candidate merge s | Base HWM MiB | Candidate HWM MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en-bytelevel_regex-w4 / read-primary | 1.717 | 1.728 | +0.64% | 4.119 | 4.132 | 1.153 | 1.139 | 159.0 | 152.5 |
| en-whitespace-w4 / read-primary | 1.512 | 1.580 | +4.51% | 3.584 | 3.681 | 0.963 | 0.972 | 132.2 | 131.6 |
| zh-bytelevel_regex-w4 / read-primary | 24.812 | 23.744 | -4.31% | 78.299 | 76.697 | 17.065 | 16.183 | 2511.7 | 2527.4 |
| zh-whitespace-w4 / read-primary | 22.000 | 21.714 | -1.30% | 62.908 | 62.149 | 6.649 | 6.455 | 1735.4 | 1750.9 |
| en-whitespace-w4 / read-en-check | 1.468 | 1.519 | +3.46% | 3.544 | 3.569 | 0.938 | 0.965 | 131.9 | 130.5 |

English Whitespace was near the original 5% range, so the reverse order was checked (+3.46%, about 51 ms). All sampled whole-train regressions remain below 5%. This is not evidence that the new seek strategy improves every workload. Real inputs here use the ordinary fresh path; reuse control and existing tests verify query correctness but do not establish realistic reuse throughput.

Raw jobs/stdout/stderr, stage metrics, CPU/HWM/swap values, binary hashes and full compressed models are retained in this directory. Frozen sources and runner locks are under source/. The exact prepared input hashes and portable replay workflow are in ../positions-child-20261012/environment.json and replay.py; use the new frozen child-read source overlay in a disposable checkout of this report base SHA. measure-read.py preserves the original machine paths and schedule. Models are stored once by SHA-256 with per-run model-reference.json; decompression reproduces the exact exported bytes. `uv run count-source.py` from the repository root reproduces production counts.
