# E and DE screening results: 512 MiB corpus

Both calls used the same 536,870,289-byte Wikipedia corpus (SHA-256 `a0d40d4102cba1e933f25e5ccd17552d2eaebaec4ef770bc39d8e46e740b4d4f`), reference backend, no normalization, vocab 50,000, min frequency 2, init/merge workers 4/4, atomic u32 corpus, and the original API. Each was run once, sequentially. Full signature gates passed against B2, D and each other: model SHA `d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd`, vocab 50,000, merges 29,243, unique words 1,429,915. Both retained symbols/edges/pairs 204,660,029 / 203,230,114 / 2,697,517.

| Metric | E: posting bulk commit | DE: radix count + bulk install/commit |
|---|---:|---:|
| Worktree commit | `35eaf03cb59a421662d41a0ba50929fae7e2e2ba` | `c8702374bd8a3812f8bca34cd53e21afe01632c3` |
| Binary SHA-256 | `0d9e879f1af39ded11c88b20f6d6976ca51cbbc8737b70fc75f5698f570150b9` | `8d260f87357da76d4ac0c90fddfcac5cc86632cb38ca94584cb9fd65df500bee` |
| Train / elapsed | 37.082 / 41.612 s | 33.053 / 37.024 s |
| Initialization / merge | 13.173 / 19.842 s | 8.589 / 19.984 s |
| Initial posting installation / merge commit | base backend field not emitted / 6.687 s | 0.734 / 6.385 s |
| Max RSS (KiB) / minimum available | 3,594,340 / 4,741,312,512 B | 4,648,144 / 3,707,158,528 B |
| Sampled process swap | 0 B | 0 B |
| Host `pswpin` / `pswpout` deltas | 142 / 8,237 pages | 14 / 767 pages |
| Child user / system CPU | 112.326 / 6.601 s | 91.426 / 7.002 s |
| Child wall time | 43.107 s | 38.596 s |
| Minor / major faults | 570,586 / 0 | 1,035,033 / 0 |
| Voluntary / involuntary context switches | 24,353 / 26,745 | 23,398 / 26,063 |
| Host busy / steal | 53.57% / 0.0348% | 50.64% / 0.0260% |
| Initial corpus / posting storage | 849,691,660 / 1,147,872,496 B | 849,691,660 / 802,967,868 B |

E retains B2 initial counting and changes owner-based posting commit bulk fill. It reports normal fused preparation and commit stats; the original B2 initial count backend has no backend-name field. DE combines stable radix initial counting with bulk posting installation and owner-based bulk commit. Its backend field was asserted to equal `stable_radix16`. DE initial route compaction/radix sort/group count/posting install measured 1,088.242 / 1,927.273 / 2,636.396 / 733.554 ms; route and radix scratch buffers were 1,625,840,912 B each, peak route buffer 3,251,681,824 B, group buffer 100,663,296 B.

These are single screening calls. Host paging deltas include activity outside each child; process sampled swap remained zero. Raw results, environment/provenance, stdout/stderr, and signature checker outputs are in `optimization-e-512.*` and `optimization-de-512.*` sidecars.
