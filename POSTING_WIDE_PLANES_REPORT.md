# Wide address planes experiment

Status: isolated prototype; not merged and not yet accepted on performance.

Goal: support 64-bit corpus coordinates and find a speed/memory balance close to flat address arrays. U32 is not a datatype requirement. Local experiments retain the U32 Flat baseline and force 16-bit address-block geometry while storing low components as U32, because this machine cannot establish a giant U64 Flat baseline.

## Representation and bounds

Each sorted list stores a raw low-U32 plane and 0–4 bytes of upper address per element. The first item obtained from its reverse producer is its maximum, so allocation needs no second chain traversal. Upper widths 1, 2 and 4 use primitive arrays; width 3 uses a U16 plane plus a U8 plane. Decode and reverse-fill dispatch are outside their loops. Width depends only on the address range, with no corpus-language thresholds or global single-block branch. The upper width can grow during initialization-wave append; a width-only change keeps spare capacity instead of doubling it.

The posting header is 24 bytes (usize length, usize capacity, inline value or tagged pointer); Entry is 32 bytes. One or two addresses are inline across the complete U64 domain. For inline lists, the otherwise unused capacity word holds the first address. Heap payload is capacity × (4 + upper bytes), with natural 4-byte alignment. The pointer only tags allocator origin. Upper width occupies the top three capacity bits: Layout already requires byte sizes ≤ isize::MAX and each item needs ≥4 bytes, so these bits are provably unused without imposing any additional allocation limit. All per-list headers, allocator padding, unused capacity and retained arena backing must be included in full-train RSS comparisons. Worst-case payload reaches 8 bytes per posting; this is not a promise of universal compression.

Append uses geometric capacity growth and at most four width promotions, with linear total copying over repeated waves. No physical-block or count-block directory is stored. The legacy metric `initial_directory_bytes` now counts upper-plane capacity bytes; `initial_block_pairs` counts nonempty lists. Those labels are preserved only for harness compatibility.

## Integer-range audit

- Posting length/capacity, global birth totals and global fragment links are usize. Layout multiplication is checked. Tests validate layouts above U32 cardinality without allocating that many elements.
- Local birth counts and links remain U32. Both fused and AA job inputs are capped at 2^26 occurrences/plans; each produces at most two nodes, staying below the U32 sentinel. This is a job bound, not a corpus bound.
- Initialization uses bounded spatial waves and local U32 offsets plus machine-size global bases. Cross-wave posting append is amortized.
- Normal 32-bit block geometry composes the complete U64 coordinate domain. Forced 16-bit geometry is a scaled simulation; it is not advertised as independently representing arbitrary U64 values.
- Weight-block indexing is usize, with U32 offsets inside normal 2^32-slot blocks. Long words may cross blocks and inherit weights; no whole-word U32 length requirement was introduced. The obsolete U32 block-count gate was removed.
- Out-of-scope legacy compatibility constraints: decorated `PreparedCorpus` and the serial fallback retain U32 pivots/token lengths; corpus preprocessing checks weighted edge mass against I64. Therefore complete address tests do NOT establish full-library U64-length support. The parallel undecorated path is the current performance experiment. The user explicitly deferred legacy paths; they are not a blocker and will not be widened during this optimization.

## Verification

The first revision passed all 93 library tests plus five targeted codec tests (two additions). The current second revision passed all 99 library tests with `TK_POSTING_SCRATCH_BITS=16` (`/tmp/posting-wide-v2-all-tests.log`). Tests cover all upper widths, odd capacities, full U64 addresses, allocator accounting, width growth, reverse-producer panic safety, and scaled temporary-plane materialization. `git diff --check` passed. Compare complete training time, model digest, peak RSS and phase timings. Do not infer real U64 Flat performance from the U32 control, or add capacity counters from different phases to explain RSS.

## First native triad (1f9e2911, one invocation each)

All models match SHA `d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd`, 29,243 merges. These first observations are not repeated medians.

| Variant | Train ms | Peak RSS B | Init ms | Delta ms | Commit ms |
|---|---:|---:|---:|---:|---:|
| Fresh U32 Flat | 19,528.058 | 3,564,494,848 | 5,222.541 | 7,787.965 | 4,816.236 |
| Planes32 | 21,277.810 | 3,718,340,608 | 5,864.037 | 8,198.366 | 5,461.870 |
| Planes16 | 22,280.717 | 4,378,509,312 | 6,428.990 | 8,367.816 | 5,635.278 |

The first revision is not accepted. Initial posting install was 1,513/2,265/2,557 ms, motivating the constructor dispatch change. Planes16 made 14,000,979 arena buffer allocations versus 9,168,324 in Planes32; two-element high-address lists could not be inline in v1. Arena requested/backing were 519,303,336/805,299,136 B for Planes32 and 747,117,046/1,610,605,248 B for Planes16. The v2 full-address pair inline and natural alignment target these costs. These counters come from different lifetime/phase measurements and must not be added to explain RSS.

Artifacts: `/root/code/tokenizers-workspaces/posting-experiment-results/countposting-wide-planes/`. Child peak VmSwap remained zero; host pswpin/pswpout counters were recorded separately and were not universally zero.

## More complete scaling in v2

`TK_POSTING_SCRATCH_BITS=16` is a compile-time simulation control. The production default remains 32, preserving the complete U64 address domain. The scaled mode makes temporary birth and valid-position high planes vary across small physical intervals as well. Const-generic unit tests independently cover true 32-bit halves and scaled 16-bit halves. The next screen pairs fresh Flat with planes32/scratch32, planes16/scratch32 and planes16/scratch16 to separate persistent and temporary representation costs. No giant U64 Flat run is planned.

## Second native screen (8080f281, one invocation each)

| Variant | Train ms | Peak RSS B | Initial install ms | Delta ms | Commit ms |
|---|---:|---:|---:|---:|---:|
| Fresh U32 Flat | 16,728.487 | 3,538,821,120 | 1,292.436 | 6,853.302 | 3,973.412 |
| Planes32 / scratch32 | 18,109.875 | 3,664,441,344 | 1,215.910 | 7,384.647 | 4,112.749 |
| Planes16 / scratch32 | 20,917.040 | 4,185,767,936 | 1,379.819 | 8,506.633 | 5,517.015 |
| Planes16 / scratch16 | 20,480.189 | 4,211,580,928 | 1,466.752 | 8,668.853 | 4,718.213 |

Models and merge counts match. The two forced variants have identical persistent allocation counts and 400,800,218 B of upper-plane capacity. Forced scratch increases peak valid-position buffers from 17,301,504 to 33,234,944 B and birth buffers from 46,137,344 to 68,444,160 B, proving that the extra temporary planes are actually exercised. The lower total time for scratch16 does not establish an optimization; the one-shot commit timings differ substantially, so repeat before attributing small differences.

Initial construction overhead is removed at this checkpoint, while initial group counting is 680.782 ms versus Flat's 334.739 ms and merge delta also remains slower. Natural alignment and full-width pair inline reduce Planes16 arena allocations to 9,084,676, requested bytes to 688,220,022 and backing to 939,516,800 B. Full-run RSS remains the comparison metric.

Artifacts: `/root/code/tokenizers-workspaces/posting-experiment-results/countposting-wide-planes-v2/`.

## Weight-interval cache revision

The current revision caches `(next actual weight boundary, weight)` during sorted posting scans, across any number of physical address blocks. Each block links to the next actual boundary, so an empty block does not force a lookup. This replaces per-position block resolution in the default weight path. Initial grouping uses the same cursor and restores checked multiplication for uniform weights. There is no corpus-single-block branch or language-tuned cutoff. Auxiliary block metadata increases by one usize per block; posting representation is unchanged.

All 101 library tests pass with scratch16 (`/tmp/posting-wide-v3-all-tests.log`), including sparse jumps, empty blocks, repeated pivots, zero weights, uniform count overflow and existing randomized training traces. Its later measurements and active-path limitation are recorded below.

## Third screen and active-path correction

V3 `d77c1e9a` first screen: Flat 16,193.757 ms / 3,574,423,552 B RSS; planes32/scratch32 17,391.908 ms / 3,638,378,496 B; planes16/scratch16 20,033.639 ms / 4,201,840,640 B. All models match; no repeats. Native artifacts: `posting-experiment-results/countposting-wide-planes-v3/`.

Source inspection explains why this was not a strong improvement: default `weight_lookup` is Some even when its allocated bytes are zero. Its weight-one interval lives inline. V3's new cursor was only selected when the existing index was disabled; default scans still resolved a physical block at each position. The one-shot V3 timing differences therefore cannot establish a benefit from that cursor in the default path.

V4 introduces a shared WeightLookups object: local fallback indexes plus a coalesced exact global weight-one interval. The existing range optimization now crosses arbitrary physical blocks without selecting a block first. Sorted weights outside that range use the exact cursor; unsorted weights retain the local bucket lookup. One shared metadata object is built per training call; no extra per-posting state, no global single-block branch, and no language-fitted parameter.

Validation: all 102 library tests passed with `TK_POSTING_SCRATCH_BITS=16` (`/tmp/posting-wide-v4-all-tests.log`), including a new test that explicitly exercises the default shared index across multiple empty blocks and both ordered/unordered weight modes. V4 native screening is recorded below.

## Fourth screen and interval counting follow-up

V4 `78c37db9`, one invocation each: Flat train 16,002.441 ms / RSS 3,526,742,016 B; planes32/scratch32 17,231.775 ms / 3,634,294,784 B; planes16/scratch16 19,371.728 ms / 4,209,328,128 B. All model hashes match; child VmSwap is zero. Artifacts: `posting-experiment-results/countposting-wide-planes-v4/`. These are screening observations, not repeated medians or an acceptance decision.

Initial group counting remains 837.627 / 836.719 ms for the candidates versus Flat 313.518 ms. The follow-up counts sorted records by actual weight intervals: obtain the next boundary once, locate the end of the occupied run with galloping search, then checked-multiply its cardinality by weight. The final remaining run needs only an endpoint check. Empty address blocks do not split runs. Runtime depends on occupied weight ranges, without a language-specific threshold, physical-block specialization or extra posting metadata. Unordered weight metadata retains its previous point lookup; overflow is now checked in that sum as well.

Implementation is isolated on `exp/posting-range-count`; all 104 library tests passed with scratch16 (104.03 s, `/tmp/posting-range-v5-all-tests.log`), including sparse/dense/repeated records, wave bases, duplicate boundaries and checked multiply/add overflow. `git diff --check` passed. Performance validation is pending. V4 multilingual screening uses the unchanged parent commit.

## Repeated V5 result and rejected inlining experiment

V5 `1ceade8d` Chinese n=3 balanced runs: Flat / normal / full-scaled train medians 16.000 / 17.721 / 19.274 s; OS VmHWM medians 3,586,707,456 / 3,673,223,168 / 4,237,602,816 B. Three-language one-shot models match. Full records live in `posting-experiment-results/countposting-wide-planes-v5/results/`.

V6 `33f754e5` forced the per-position consumer inline. Emitted-code validation confirmed removal of the per-position call, but grew the main task function from 3,072 B (+1,661 B outlined consumer) to 12,341 B. A fresh native screen and a separate adjacent V5/V6 comparison did not establish a win: normal V5 18,095.386 versus V6 18,003.324 ms (-0.51%); full V6 19,683.547 versus V5 19,454.011 ms (+1.18%). V6 is not selected; its results are preserved at `countposting-inline-consumer-v6/`.

## Bounded decoded batches and corpus prefetch experiment

The next candidate starts from V5. Each execution job reuses one 128-element usize stack buffer (1 KiB on this target). A list decodes at most that many coordinates, then one ordinary loop consumes them. This avoids a large consumer per codec width and avoids a consumer call per coordinate. It adds bounded L1 traffic; whether that tradeoff helps is an end-to-end question.

An independent compile-time `TK_POSTING_PREFETCH=1` switch issues x86_64 cache hints 16 coordinates ahead in the decoded batch, with an in-bounds pointer check; the default is off and other architectures retain plain batch consumption. It never reads a token value or changes the atomic rewrite order. Both variants preserve the same posting and temporary address representations. Instruction sampling near the first corpus token load motivates testing latency hiding; sampling skid prevents treating one instruction's sample share as exact stalled cycles.

Initial screening will use Flat, existing V5 full-scaled, plain V7 full-scaled, and prefetch V7 full-scaled. Both new variants use block16/scratch16. Normal-range and multilingual validation follow only if an end-to-end benefit is observed. Tests extend batch decode over all upper widths, inline/empty lists and partial batches. With scratch16, prefetch-on passed all 104 library tests (99.82 s, `/tmp/posting-batched-v7-prefetch-tests.log`); prefetch-off passed all 14 parallel-training tests (10.64 s, `/tmp/posting-batched-v7-plain-tests.log`). `git diff --check` passed. Native measurements are pending.
