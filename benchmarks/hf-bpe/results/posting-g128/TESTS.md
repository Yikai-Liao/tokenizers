# Restart-length ablation verification

Frozen clean source: `1c5ed8e47b569743cbc765df9a9875556b868211`.

For every restart length, `cargo test --offline --manifest-path tokenizers/tk-train/Cargo.toml --lib --no-default-features trainers::bpe::indexed::parallel` ran with `TK_POSTING_RESTART` set to that length, scratch bits 16, fused decode 1, and prefetch 1. These are test builds, separate from the later optimized native benchmark builds.

| Restart length | Result | Test runtime |
| --- | --- | ---: |
| 32 | 60 passed, 52 filtered | 18.69 s |
| 64 | 60 passed, 52 filtered | 18.74 s |
| 128 | 60 passed, 52 filtered | 18.37 s |
| 256 | 60 passed, 52 filtered | 18.35 s |
| 512 | 60 passed, 52 filtered | 18.89 s |

The tests cover forward encoding and restart-offset oracles, full U64 gaps, translation-invariant allocation sizes, group-boundary and partial appends, amortized expansion, producer interruption/replay, and independent parallel cursors. Group-dependent boundary cases now use the actual restart length; varint radix boundaries remain fixed at 128.

The first 32-length run passed 59/60 tests: an existing interruption test assumed four reserved groups after literal counts 130/258/270, which represents ten groups at length 32. The test was changed to a construction with equivalent relative group counts, reserved capacity and a second-pass interruption. The failed attempt is preserved. All five configurations were rerun after this test change and passed. This failure did not require a storage-algorithm change.

The default 128-length constructor's preceding frozen version passed all 112 library tests (v18 verification); the ablation commit itself ran the 60-test parallel subset for each configuration, not all 112 tests for every length. Format and `git diff --check` passed before freezing.
