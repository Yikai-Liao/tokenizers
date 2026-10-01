# Independent mathematical review of the Rust Radsort translation

Scope: bounded source review of `prototype/src/radsort_u64.rs` as delivered, against the vendored author implementation at `f69e816c3cd79d312cd67aea5b9cf1c338c1b371`. No production-source edits or new CPU jobs were performed for this review. The root integration may additionally change pointer storage and add debug validators/tests.

Conclusion: no algorithmic discrepancy was found in `step()` or `compact()`. The following proof obligations explain the overwrite safety and stable order. Rust pointer provenance remains a separate obligation; capturing the input slice base once is preferable to repeatedly deriving pointers through mutable slice reborrows.

## Definitions and round-boundary invariant

Let radix `R = 256`, block length `B = 512`, input length `n`, full input block count `q = floor(n/B)`, and total physical block count `M = q + 2R`.

Physical IDs `[0,2R)` identify complete scratch blocks. IDs `[2R,M)` identify the `q` complete input blocks. The original partial tail is copied into scratch; it is not a physical input block. All physical blocks therefore have capacity B.

At a round boundary:

1. `perm` is a bijection on `[0,M)`.
2. Logical indices `[0,R)` and `[fill,M)` are free; `[R,fill)` hold the current sequence.
3. A logical block is full except at the sorted logical indices in `partials`; each partial length is in `[0,B)`. Unused initial sentinels have index M and length 0.
4. Concatenating the valid prefixes of `[R,fill)` yields exactly n records in the current stable digit order.
5. The final allocated logical block is partial, possibly empty. Empty buckets and exact multiples of B deliberately retain an empty terminal block.

Initialization assigns the disjoint physical ID ranges `[0,R)`, `[2R,M)`, `[R,2R)` in that order. Full input blocks are followed by the partial tail in scratch block R. Their valid lengths sum to `qB + n%B = n`.

## `step()`: recycling cannot overwrite unread input

Each digit bucket begins with one free output block. Every time a bucket fills, its next output block is reserved immediately, even if its eventual tail is empty.

Suppose a full-bucket event happens after J records have been read, with per-bucket counts `n_b` summing to J. The newly reserved logical block has index:

```text
output_index = R + sum_b floor(n_b/B) - 1
             <= R + floor(J/B) - 1
```

If the current input logical block is `R+t`, at most t earlier blocks plus this block have been read, so `J <= (t+1)B`. Hence:

```text
output_index <= R+t = current_input_index
```

If equality holds, then `J = (t+1)B`: every earlier input block and the current block must have length B, and the current block's last element has already been consumed. The block can be reserved then; the first write into it happens on a subsequent input record. If inequality is strict, that block is free or was consumed earlier.

This stronger argument is needed in addition to the code's `output <= input` assertion. The assertion alone does not state when equality is safe. All writes into the current output bucket occur before reservation of its successor, and the old source value is loaded before its destination write.

At end of input, bucket b has `floor(n_b/B)` full blocks plus exactly one partial block of length `n_b%B`. Thus its allocated block count is:

```text
counts[b] = 1 + floor(n_b/B)
O = sum_b counts[b] = R + sum_b floor(n_b/B) <= R+q
```

There are at least R free blocks because `O + R <= M`.

## `step()`: fixup restores a bijection and stable sequence

The first O entries of old `perm` are the blocks allocated to the output, recorded by allocation time. `usage` labels their buckets. Stable counting scatter of these block IDs places them into `[R,R+O)`, in bucket order and in allocation order within each bucket.

Allocation order within a bucket is record arrival order. The input is consumed in logical sequence order and each record is appended to its bucket, so the valid record sequence is a stable sort on the current digit.

The first R unallocated physical IDs, old logical `[O,O+R)`, become the next free head start. Old `[O+R,M)` fills the remaining free suffix. These three old index ranges are disjoint and exhaustive, so new `perm` remains a bijection.

Each bucket's final allocated physical block is identified by its end pointer. The pointer to its next free element gives its partial length, including zero. The partial blocks occur at the last logical index of each bucket interval; `counts[b] >= 1`, so their logical indices are strictly increasing in bucket order. Total valid length is:

```text
sum_b (B*floor(n_b/B) + n_b%B) = n
```

Four stable digit rounds at shifts 32, 40, 48, 56 therefore sort the entire high32 key, while retaining the incoming order of arbitrary low32 payloads. Increasing positions are one permitted payload distribution, not a prerequisite of the sorter.

## `compact()`: protect the next physical output block

The main compact loop processes logical input index `i = R+t` and physical destination block `d = 2R+t`, for `t in [0,q)`.

Its induction invariant is:

1. `perm2` is the inverse of `perm`.
2. Logical blocks before i have been consumed in order. Their valid concatenation is exactly `records[..start]`.
3. Every physical input block below d is assigned to an already consumed logical block. No unconsumed logical block maps there.
4. A tracked free physical block is assigned to a free logical index in the head or suffix.
5. All unconsumed logical blocks still contain their original valid record prefixes, possibly relocated to another physical block.

Before processing i, `start <= tB`; appending at most B elements ends at or before `(t+1)B`, the end of physical block d. Thus the destination interval can touch earlier physical input blocks and d, but no later input block. Earlier blocks contain only already consumed logical data by invariant 3.

If d holds an unconsumed later logical block (`i < inverse[d] < fill`), the implementation first copies that full physical block to the tracked free block, swaps those mappings, and updates their inverse entries. Its valid record prefix is preserved; copying unused suffix bytes has no semantic effect because all backing allocations were initialized.

If d maps to i itself, the current source is protected by memmove semantics. If it maps to an earlier logical index or a free index, its old contents can be overwritten directly.

The current block's valid prefix is then appended at `start` using `ptr::copy`, which permits overlap. The bookkeeping exchanges the mappings of i and the logical index formerly mapped to d, and updates both inverse entries. This is a swap, including the identity case, so the bijection remains valid.

Physical block d is now assigned to consumed logical i. The old source block becomes the next free block when source and destination differ. That source cannot be an earlier input block because such blocks already map to earlier consumed logical indices. Therefore treating it as free cannot expose the finalized output prefix to a later scratch relocation.

After q iterations, every physical input block maps to consumed logical `[R,R+q)`. All remaining unconsumed logical blocks must therefore be in scratch, which justifies `source < SCRATCH` in the second loop. Their valid prefixes can be appended without touching live scratch. Total valid lengths sum to n, so the final `start == n` assertion completes the proof.

The bookkeeping thereafter describes consumed blocks abstractly; their contents have been packed continuously. Validators during compact should check the bijection, inverse, free-index condition and live-block conditions, rather than assume every consumed logical block still has its former physical contents.

## Rust-specific obligations and integration advice

- Capture `records.as_mut_ptr()` and length once, and retain `PhantomData<&mut [u64]>` for exclusive lifetime ownership. No Rust slice reference to the backing records should be materialized while saved raw pointers remain in use.
- Keep scratch and input allocations fixed. The Vec scratch pointer can also be captured once for clarity. `perm`/`perm2` swaps do not move either record allocation.
- `block()` must exclude the original partial input tail: only physical IDs `[2R,M)` are valid full input blocks.
- `offset_from(base)` is evaluated only when the bucket end equals that physical block's one-past-end pointer. Both pointers then belong to the same backing allocation and valid block interval.
- Use `ptr::copy`, not `copy_nonoverlapping`, in finalize: partially packed records can overlap their source. The existing port uses the correct operation.
- Keep `M <= u32::MAX` before storing physical IDs as u32. The existing code has the guard.
- Round debug validators should check bijection, sorted/unique in-range partial indices, lengths below B, head/suffix free shape, and total valid length n. Empty terminal blocks are valid and must not be rejected.
- Add arbitrary/repeated/random low32 payloads to the existing stable-sort differential test, so stable order is checked independently of position values. Root is handling that coverage.

## Memory accounting

For `n >= 2`, the fixed-block heap scratch is:

```text
8 × (2R × B) + (4 + 4 + 1) × (floor(n/B) + 2R)
= 2 MiB + 9 × (floor(n/512) + 512) bytes
```

This is `O(n/512)` metadata at fixed B, not an asymptotic `O(√n)` implementation. Variable B yields the generalized `O(B+n/B)` tradeoff. Stack arrays, the Sorter structure, allocator metadata and debug-validator allocations are additional; the returned byte metric counts the explicit scratch Vec allocations only. `n < 2` returns zero scratch.

No full production Trainer result or Rust memory-model proof is implied by the algorithmic review. The captured-pointer change and root's debug validators make the intended ownership and overwrite conditions directly inspectable.
