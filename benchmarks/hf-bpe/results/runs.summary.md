# HF BPE sequential prototype benchmark

Times are medians of three separate processes in one shuffled serial run. Train includes initialization, indexing, merges and output. Total is feed + train.

| Language | MiB | Split | Engine | n | Train s | Merge s | Total s | RSS MiB | Model SHA-256 |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| en | 1 | bytelevel | indexed | 3 | 0.181 | 0.078 | 0.751 | 24.1 | `83a5b64ac7f1b6a92c22f329826f99d8aa8fbca2a4cbed92ae93fb79efde6618` |
| en | 1 | bytelevel | pr_head | 3 | 0.108 | 0.071 | 0.694 | 17.2 | `83a5b64ac7f1b6a92c22f329826f99d8aa8fbca2a4cbed92ae93fb79efde6618` |
| en | 1 | bytelevel | reference | 3 | 0.390 | 0.186 | 0.953 | 32.9 | `83a5b64ac7f1b6a92c22f329826f99d8aa8fbca2a4cbed92ae93fb79efde6618` |
| en | 1 | none | indexed | 3 | 0.827 | 0.414 | 0.833 | 80.4 | `25f3ebd11084ae7501303598e0f5ac98c2ff45137a4c290888eb5921b2015ab1` |
| en | 1 | none | pr_head | 3 | 0.819 | 0.665 | 0.826 | 44.9 | `25f3ebd11084ae7501303598e0f5ac98c2ff45137a4c290888eb5921b2015ab1` |
| en | 1 | none | reference | 3 | 2.393 | 1.835 | 2.397 | 100.6 | `25f3ebd11084ae7501303598e0f5ac98c2ff45137a4c290888eb5921b2015ab1` |
| en | 1 | whitespace_split | indexed | 3 | 0.214 | 0.081 | 0.272 | 25.8 | `c276702bf01ca73ede3f6150872d6a844d6df60b1d9dd4d32d3f390db23c1ccf` |
| en | 1 | whitespace_split | pr_head | 3 | 0.162 | 0.107 | 0.225 | 17.9 | `c276702bf01ca73ede3f6150872d6a844d6df60b1d9dd4d32d3f390db23c1ccf` |
| en | 1 | whitespace_split | reference | 3 | 0.483 | 0.241 | 0.549 | 36.1 | `c276702bf01ca73ede3f6150872d6a844d6df60b1d9dd4d32d3f390db23c1ccf` |
| en | 4 | bytelevel | indexed | 3 | 0.414 | 0.156 | 2.711 | 51.4 | `4c10b8979324133457078a30b15a45649b26c995384d388213dacc97ce1d7801` |
| en | 4 | bytelevel | pr_head | 3 | 0.266 | 0.180 | 2.561 | 32.2 | `4c10b8979324133457078a30b15a45649b26c995384d388213dacc97ce1d7801` |
| en | 4 | bytelevel | reference | 3 | 0.995 | 0.469 | 3.327 | 69.0 | `4c10b8979324133457078a30b15a45649b26c995384d388213dacc97ce1d7801` |
| en | 4 | none | indexed | 3 | 2.896 | 1.391 | 2.918 | 261.5 | `ffdac6960d748b27641806bf262f0707e600dd6377816d48737c030407d4289b` |
| en | 4 | none | pr_head | 3 | 3.283 | 2.798 | 3.301 | 124.1 | `ffdac6960d748b27641806bf262f0707e600dd6377816d48737c030407d4289b` |
| en | 4 | none | reference | 3 | 10.379 | 8.104 | 10.410 | 314.3 | `ffdac6960d748b27641806bf262f0707e600dd6377816d48737c030407d4289b` |
| en | 4 | whitespace_split | indexed | 3 | 0.542 | 0.205 | 0.814 | 63.4 | `c13c2f955d1342c5073389c215be6a4a3c1ceb19d079293504740733a1f65bc1` |
| en | 4 | whitespace_split | pr_head | 3 | 0.383 | 0.264 | 0.621 | 36.7 | `c13c2f955d1342c5073389c215be6a4a3c1ceb19d079293504740733a1f65bc1` |
| en | 4 | whitespace_split | reference | 3 | 1.315 | 0.644 | 1.587 | 84.9 | `c13c2f955d1342c5073389c215be6a4a3c1ceb19d079293504740733a1f65bc1` |
| en | 16 | whitespace_split | indexed | 3 | 1.583 | 0.637 | 2.556 | 171.9 | `71ed7e31bef234667e203f11f41010930d8ef9d753eb9d5f92bca2e6b9d87dcc` |
| en | 16 | whitespace_split | pr_head | 3 | 1.106 | 0.795 | 2.062 | 85.0 | `71ed7e31bef234667e203f11f41010930d8ef9d753eb9d5f92bca2e6b9d87dcc` |
| en | 16 | whitespace_split | reference | 3 | 3.834 | 1.900 | 4.778 | 222.0 | `71ed7e31bef234667e203f11f41010930d8ef9d753eb9d5f92bca2e6b9d87dcc` |
| zh | 1 | bytelevel | indexed | 3 | 0.570 | 0.265 | 0.958 | 72.8 | `3ca1ec14ae0017629d76afb506886a912ca14ea8d6f9c57ae677557ca7ff2b0d` |
| zh | 1 | bytelevel | pr_head | 3 | 0.407 | 0.278 | 0.813 | 46.6 | `3ca1ec14ae0017629d76afb506886a912ca14ea8d6f9c57ae677557ca7ff2b0d` |
| zh | 1 | bytelevel | reference | 3 | 1.320 | 0.695 | 1.752 | 95.0 | `3ca1ec14ae0017629d76afb506886a912ca14ea8d6f9c57ae677557ca7ff2b0d` |
| zh | 1 | none | indexed | 3 | 0.326 | 0.084 | 0.334 | 43.0 | `72b92b8f04c38f78fa6cda0cbe3679b07c71d8b8463c85bf0d6ec5cc96c0c9e9` |
| zh | 1 | none | pr_head | 3 | 0.245 | 0.121 | 0.253 | 33.0 | `72b92b8f04c38f78fa6cda0cbe3679b07c71d8b8463c85bf0d6ec5cc96c0c9e9` |
| zh | 1 | none | reference | 3 | 0.713 | 0.317 | 0.721 | 66.0 | `72b92b8f04c38f78fa6cda0cbe3679b07c71d8b8463c85bf0d6ec5cc96c0c9e9` |
| zh | 1 | whitespace_split | indexed | 3 | 0.334 | 0.081 | 0.353 | 40.9 | `ea16fd7cdc85de7401a87a7a3877e1175a20913b3f64655cafea8ab360796259` |
| zh | 1 | whitespace_split | pr_head | 3 | 0.249 | 0.120 | 0.266 | 32.4 | `ea16fd7cdc85de7401a87a7a3877e1175a20913b3f64655cafea8ab360796259` |
| zh | 1 | whitespace_split | reference | 3 | 0.646 | 0.225 | 0.666 | 64.2 | `ea16fd7cdc85de7401a87a7a3877e1175a20913b3f64655cafea8ab360796259` |
| zh | 4 | bytelevel | indexed | 3 | 1.999 | 0.904 | 3.566 | 215.2 | `42bbe48045502934e60a50f054c4893393881bded7169f3fcc72a4343a31ef0e` |
| zh | 4 | bytelevel | pr_head | 3 | 1.385 | 0.965 | 2.932 | 115.4 | `42bbe48045502934e60a50f054c4893393881bded7169f3fcc72a4343a31ef0e` |
| zh | 4 | bytelevel | reference | 3 | 4.777 | 2.506 | 6.283 | 281.2 | `42bbe48045502934e60a50f054c4893393881bded7169f3fcc72a4343a31ef0e` |
| zh | 4 | none | indexed | 3 | 1.113 | 0.154 | 1.155 | 112.0 | `8526618fb878b97b5581a5d149e4ee2aad7399eea4db29af7c7ef507251b7120` |
| zh | 4 | none | pr_head | 3 | 0.827 | 0.277 | 0.862 | 79.5 | `8526618fb878b97b5581a5d149e4ee2aad7399eea4db29af7c7ef507251b7120` |
| zh | 4 | none | reference | 3 | 2.528 | 0.875 | 2.565 | 188.3 | `8526618fb878b97b5581a5d149e4ee2aad7399eea4db29af7c7ef507251b7120` |
| zh | 4 | whitespace_split | indexed | 3 | 1.050 | 0.113 | 1.123 | 135.1 | `ff60a7bf159f5b6a2b974930b37493855f1a79c54b1eb610ff17fd58d5178a0a` |
| zh | 4 | whitespace_split | pr_head | 3 | 0.681 | 0.198 | 0.743 | 78.1 | `ff60a7bf159f5b6a2b974930b37493855f1a79c54b1eb610ff17fd58d5178a0a` |
| zh | 4 | whitespace_split | reference | 3 | 2.204 | 0.494 | 2.281 | 183.8 | `ff60a7bf159f5b6a2b974930b37493855f1a79c54b1eb610ff17fd58d5178a0a` |
| zh | 16 | whitespace_split | indexed | 3 | 4.887 | 1.128 | 5.178 | 479.3 | `2c658fb425cb692214045c62705e213b6535a452a55df0cf369161439a289e5e` |
| zh | 16 | whitespace_split | pr_head | 3 | 5.318 | 2.820 | 5.603 | 268.4 | `2c658fb425cb692214045c62705e213b6535a452a55df0cf369161439a289e5e` |
| zh | 16 | whitespace_split | reference | 3 | 11.104 | 4.675 | 11.381 | 657.6 | `2c658fb425cb692214045c62705e213b6535a452a55df0cf369161439a289e5e` |

| Language | MiB | Split | Reference/indexed train | PR/indexed train | Reference/indexed total | PR/indexed total |
|---|---:|---|---:|---:|---:|---:|
| en | 1 | bytelevel | 2.15× | 0.60× | 1.27× | 0.92× |
| en | 1 | none | 2.89× | 0.99× | 2.88× | 0.99× |
| en | 1 | whitespace_split | 2.26× | 0.76× | 2.02× | 0.83× |
| en | 4 | bytelevel | 2.41× | 0.64× | 1.23× | 0.94× |
| en | 4 | none | 3.58× | 1.13× | 3.57× | 1.13× |
| en | 4 | whitespace_split | 2.42× | 0.71× | 1.95× | 0.76× |
| en | 16 | whitespace_split | 2.42× | 0.70× | 1.87× | 0.81× |
| zh | 1 | bytelevel | 2.32× | 0.71× | 1.83× | 0.85× |
| zh | 1 | none | 2.19× | 0.75× | 2.16× | 0.75× |
| zh | 1 | whitespace_split | 1.93× | 0.75× | 1.89× | 0.75× |
| zh | 4 | bytelevel | 2.39× | 0.69× | 1.76× | 0.82× |
| zh | 4 | none | 2.27× | 0.74× | 2.22× | 0.75× |
| zh | 4 | whitespace_split | 2.10× | 0.65× | 2.03× | 0.66× |
| zh | 16 | whitespace_split | 2.27× | 1.09× | 2.20× | 1.08× |
