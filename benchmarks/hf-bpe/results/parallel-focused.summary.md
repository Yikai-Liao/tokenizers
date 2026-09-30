# Parallel BPE bounded comparison

One process per cell; timing is preliminary. Initial layout is the source-layout estimate immediately after count/filter/heap, excluding HF input/vocabulary, allocator overhead and thread stacks. PR has no comparable initialization counter.

| Language | Split | Engine | Train s | Init s | Merge s | Initial layout MiB |
|---|---|---|---:|---:|---:|---:|
| en | whitespace_split | dict16 | 0.566 | 0.093 | 0.459 | 8.90 |
| en | whitespace_split | indexed | 0.264 | 0.071 | 0.184 | 9.26 |
| en | whitespace_split | narrow16 | 0.587 | 0.102 | 0.473 | 7.11 |
| en | whitespace_split | parallel1 | 0.417 | 0.086 | 0.322 | 9.54 |
| en | whitespace_split | parallel4 | 0.395 | 0.079 | 0.305 | 9.54 |
| en | whitespace_split | pr_head | 0.356 | 0.099 | 0.236 | — |
| zh | none | dict16 | 0.845 | 0.535 | 0.274 | 54.95 |
| zh | none | indexed | 0.409 | 0.289 | 0.098 | 31.93 |
| zh | none | narrow16 | 0.909 | 0.568 | 0.301 | 48.76 |
| zh | none | parallel1 | 0.717 | 0.490 | 0.206 | 31.97 |
| zh | none | parallel4 | 0.658 | 0.458 | 0.178 | 31.97 |
| zh | none | pr_head | 0.857 | 0.496 | 0.269 | — |
