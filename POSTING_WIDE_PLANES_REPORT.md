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
