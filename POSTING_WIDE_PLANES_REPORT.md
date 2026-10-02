# Wide address planes experiment

Status: isolated prototype; not merged and not yet accepted on performance.

Goal: support 64-bit corpus coordinates and find a speed/memory balance close to flat address arrays. U32 is not a datatype requirement. Local experiments retain the U32 Flat baseline and force 16-bit address-block geometry while storing low components as U32, because this machine cannot establish a giant U64 Flat baseline.

## Representation and bounds

Each sorted list stores a raw low-U32 plane and 0–4 bytes of upper address per element. The first item obtained from its reverse producer is its maximum, so allocation needs no second chain traversal. Upper widths 1, 2 and 4 use primitive arrays; width 3 uses a U16 plane plus a U8 plane. Decode dispatch is outside the loop. Width depends only on the address range, with no corpus-language thresholds or global single-block branch. The upper width can grow during initialization-wave append.

The posting header is 24 bytes (usize length, usize capacity, inline value or tagged pointer); Entry is 32 bytes. A singleton is inline at any address. Two U32-range values also fit inline. Heap payload is capacity × (4 + upper bytes), with 16-byte pointer alignment. All per-list headers, allocator padding, unused capacity and retained arena backing must be included in full-train RSS comparisons. Worst-case payload reaches 8 bytes per posting; this is not a promise of universal compression.

Append uses geometric capacity growth and at most four width promotions, with linear total copying over repeated waves. No physical-block or count-block directory is stored. The legacy metric `initial_directory_bytes` now counts upper-plane capacity bytes; `initial_block_pairs` counts nonempty lists. Those labels are preserved only for harness compatibility.

## Integer-range audit

- Posting length/capacity, global birth totals and global fragment links are usize. Layout multiplication is checked. Tests validate layouts above U32 cardinality without allocating that many elements.
- Local birth counts and links remain U32. Both fused and AA job inputs are capped at 2^26 occurrences/plans; each produces at most two nodes, staying below the U32 sentinel. This is a job bound, not a corpus bound.
- Initialization uses bounded spatial waves and local U32 offsets plus machine-size global bases. Cross-wave posting append is amortized.
- Normal 32-bit block geometry composes the complete U64 coordinate domain. Forced 16-bit geometry is a scaled simulation; it is not advertised as independently representing arbitrary U64 values.
- Weight-block indexing is usize, with U32 offsets inside normal 2^32-slot blocks. Long words may cross blocks and inherit weights; no whole-word U32 length requirement was introduced. The obsolete U32 block-count gate was removed.
- Remaining compatibility constraints: decorated `PreparedCorpus` and the serial fallback retain U32 pivots/token lengths; corpus preprocessing checks weighted edge mass against I64. Therefore complete address tests do NOT establish full-library U64-length support. The parallel undecorated path is the current performance experiment. Resolve these compatibility constraints before claiming complete production coverage.

## Verification

The complete library suite passed 93 tests (`/tmp/posting-wide-all-tests.log`); after adding width-by-width/odd-capacity and allocator-accounting coverage, all five codec tests passed (`/tmp/posting-wide-codec-tests.log`). Thus all 95 test cases are covered, but there was no single all-95 run. `git diff --check` passed. Immutable native benchmark artifacts will be recorded after completion. Compare complete training time, model digest, peak RSS and phase timings. Do not infer real U64 Flat performance from the U32 control, or add capacity counters from different phases to explain RSS.
