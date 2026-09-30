# HF BPE sequential prototype benchmark

Times are medians of the recorded separate processes in one shuffled serial run. Train includes initialization, indexing, merges and output. Total is feed + train.

| Language | MiB | Split | Engine | n | Train s | Merge s | Total s | RSS MiB | Model SHA-256 |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| en | 4 | whitespace_split | fused | 2 | 0.285 | 0.192 | 0.534 | 29.0 | `c13c2f955d1342c5073389c215be6a4a3c1ceb19d079293504740733a1f65bc1` |
| en | 4 | whitespace_split | indexed | 2 | 0.233 | 0.156 | 0.471 | 25.9 | `c13c2f955d1342c5073389c215be6a4a3c1ceb19d079293504740733a1f65bc1` |
| en | 4 | whitespace_split | pr_head | 2 | 0.364 | 0.244 | 0.614 | 35.6 | `c13c2f955d1342c5073389c215be6a4a3c1ceb19d079293504740733a1f65bc1` |
| zh | 4 | none | fused | 2 | 0.417 | 0.108 | 0.454 | 45.8 | `8526618fb878b97b5581a5d149e4ee2aad7399eea4db29af7c7ef507251b7120` |
| zh | 4 | none | indexed | 2 | 0.446 | 0.113 | 0.481 | 45.8 | `8526618fb878b97b5581a5d149e4ee2aad7399eea4db29af7c7ef507251b7120` |
| zh | 4 | none | pr_head | 2 | 0.829 | 0.277 | 0.872 | 79.8 | `8526618fb878b97b5581a5d149e4ee2aad7399eea4db29af7c7ef507251b7120` |

| Language | MiB | Split | PR/indexed train | PR/fused train | PR/fused total |
|---|---:|---|---:|---:|---:|
| en | 4 | whitespace_split | 1.56× | 1.28× | 1.15× |
| zh | 4 | none | 1.86× | 1.99× | 1.92× |
