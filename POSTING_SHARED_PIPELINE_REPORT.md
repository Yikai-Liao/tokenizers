# Shared owner posting experiment

This branch tests research candidates 1 (shared high bits) and 14 (direct birth construction). The resident position stream is always `u32`, including the forced-small-block experiment. Each owner keeps one continuous local-offset array and a contiguous `(block: u32, end: u32)` run directory. There are no per-block pair dictionaries on the active training path.

The production geometry uses 32 offset bits: `(block as u64 << 32) | offset` represents every u64 address. Tests cover 0, both sides of 2^32, high-bit transitions, u64::MAX, randomized full-domain lists, and splitting/concatenating source waves. The scaled geometry uses 16 offset bits while retaining u32 storage, to isolate segmentation rather than integer width. This is a simulation of the mechanism, not an estimate of the real large-corpus segment-density distribution.

Initialization uses bounded radix waves, with a read halo preserving cross-wave edges. Pair frequencies are reduced across waves before the global minimum-frequency filter. The segmented and single-block cases share the grouping, posting construction, fused prepare, birth commit, and retirement implementation. A one-segment list can omit a directory heap allocation through the existing inline storage; it still has a real directory entry. Single-block construction skips a redundant count pass when all source addresses provably share a block.

Execution jobs are split by occurrence count. A job can consume slices spanning many address blocks. No Vec or map is allocated for every segment. Births are filled directly into their final owner posting. Temporary plan and birth-node coordinates use usize so cross-block neighbors remain exact; these are counted as extra memory cost, not hidden in the resident-posting savings.

Payload model: `4P + 8S` bytes before capacity slack, plus 32 bytes per posting object (40-byte frequency+posting Entry versus the original 24-byte Entry). One run is inline. When S is close to P the directory can erase savings versus u64 positions; no compression is guaranteed for arbitrary 64-bit sets. The current PackedPosting container retains the existing u32 per-list cardinality limit and errors if it is exceeded. Full u64 address coverage is distinct from supporting more than 2^32-1 entries in one list; multi-extent lists are a separate extension.

Required measurements: original U32 Flat; new U32 single block (bits32); new U32 forced multi-block (bits16). Same 512 MiB Chinese corpus, no affixes, split none, vocab 50,000, min_frequency 2, 4 workers/init workers. Compare native train time and peak RSS; validate identical model SHA. Results will be added after completed native calls. Previous U16-owner measurements are not this candidate's results.

Source and test status: implementation in progress. Artifact directory: `/root/code/tokenizers-workspaces/posting-experiment-results/shared-pipeline/`.
