# Birth-fragment layout in the Chinese single-worker investigation

This comparison uses the frozen `final-baseline-production` and
`aligned-candidate-production` x86-64 release binaries. Their SHA-256 values are
`5c927ecb9b30911dc6914c269b8b870e9cadd266c12fc6bbbbc1e4128548994f` and
`753d1c6040a6387d5e46240b373f67d498e1c9a9b1a0f5be5e89ef81722eeba4`.
The source and runner provenance are in `../final-manifest.json` and
`../aligned-manifest.json`.

Both versions stably sort birth fragments by `(birth_group, pair)` and move a
56-byte owning element. The named record changes the physical field order:

| Field | Baseline tuple offset | Aligned `BirthFragment` offset | Bytes |
| --- | ---: | ---: | ---: |
| `birth_group` | 0 | 32 | 8 |
| `pair.0` | 8 | 40 | 4 |
| `pair.1` | 12 | 44 | 4 |
| `weight` | 16 | 48 | 8 |
| `positions` | 24 | 0 | 32 |

The key remains contiguous. It is read at offsets `0/8/12` in the old comparator
and `32/40/44` in the aligned comparator. Both compare the group first and use
the same two unsigned pair comparisons when groups are equal. Stable equality
handling and full-element moves remain the same.

The complete layout is confirmed by the element-construction instructions,
alongside the comparator reads:

| Evidence | Baseline addresses | Aligned addresses |
| --- | --- | --- |
| Comparator group/pair reads | `0xc6ca0`, `0xc6ca9`, `0xc6cac` | `0xc6b50`, `0xc6b5a`, `0xc6b5d` |
| Median comparator group/pair reads | `0xc6f0f`, `0xc6f1a`, `0xc6f1d` | `0xc6ddf`, `0xc6dec`, `0xc6def` |
| Copy the 32-byte owning `Positions` into the temporary record | `0xe8ae5`–`0xe8af8`: temporary `rsp+0x38` | `0xea9ef`–`0xea9fe`: temporary `rsp+0x20` |
| Fill group, pair and weight in the temporary record | `0xe8afc`–`0xe8b12`: `rsp+0x20/0x28/0x30` | `0xeaa04`–`0xeaa1a`: `rsp+0x40/0x48/0x50` |
| Append the complete record, using a `0x38` stride | `0xe8728`–`0xe8756`, temporary base `rsp+0x20` | `0xea648`–`0xea676`, temporary base `rsp+0x20` |

Sorting helper symbols obtained from `nm -S --size-sort -C`:

| Helper | Baseline address | Baseline bytes | Aligned address | Aligned bytes |
| --- | --- | ---: | --- | ---: |
| `stable::driftsort_main` | `0xc56c0` | 276 | `0xc55c0` | 276 |
| `stable::drift::sort` | `0xc57e0` | 1888 | `0xc56e0` | 1904 |
| `stable::quicksort::quicksort` | `0xc6020` | 3192 | `0xc5f30` | 3098 |
| `shared::smallsort::sort4_stable` | `0xc6ca0` | 509 | `0xc6b50` | 530 |
| `shared::smallsort::insertion_sort_shift_left` | `0xed580` | 363 | `0xefd40` | 362 |
| `shared::pivot::median3_rec` | `0xc6ea0` | 283 | `0xc6d70` | 287 |

The comparator instruction sequence is the same apart from field displacements
and relocated branch addresses. Some helpers grow and others shrink. These static
sizes establish changed code generation; they do not establish a CPU-time cause.
Likewise, a 16-byte contiguous key at a different offset within a 56-byte stride
does not by itself imply a higher cache-line crossing frequency.

The two existing `zh-aligned-profile/record-*` pairs give the quicksort helper
sample shares of `3.58%/3.81%` for the tuple and `4.14%/4.27%` for the named
record. These whole-process samples identify a candidate for isolation; they
do not assign the remaining public-training CPU difference to sorting.

A strict layout isolation can keep the existing named fields and their current
declaration order (`birth_group`, `pair`, `weight`, `positions`) and add
`#[repr(C)]` to this private record. In the measured build that should restore
the baseline offsets and 56-byte stride. Verify those offsets in the emitted
comparator and constructor, then compare complete public-training calls. Task
budgets, task order, stable sort keys, stream ownership, Corpus boundaries and
the local iterator borrow should remain identical in that experiment. Its result
determines whether this introduced layout change explains any part of the
remaining Chinese single-worker difference.

The strict isolation has now been built as `birth-layout-production`, SHA-256
`0c0279158b0f3fb676cedb9b825bdbb9896c2522219ac6adc8d77c469fbd0fbc`.
The source diff adds only `#[repr(C)]` above `BirthFragment`; field declarations,
construction and consumption are unchanged. `../birth-layout-source.json` records
the source hashes. Only `index.rs` differs from the aligned source manifest,
and the shared runner-source hashes match.

The emitted comparator reads group/pair at `0/8/12` again: `sort4_stable` starts
at `0xc7500` and `median3_rec` reads them at `0xc776f`–`0xc777d`. Construction
at `0xeb2a2`–`0xeb2cf` puts `Positions` at temporary `rsp+0x38`, group/pair at
`rsp+0x20/0x28`, and weight at `rsp+0x30`, confirming restored element offsets
`Positions=24` and `weight=16` relative to the temporary record base `rsp+0x20`.

| Isolation helper | Address | Bytes | Baseline bytes |
| --- | --- | ---: | ---: |
| `stable::driftsort_main` | `0xc5f20` | 276 | 276 |
| `stable::drift::sort` | `0xc6040` | 1888 | 1888 |
| `stable::quicksort::quicksort` | `0xc6880` | 3192 | 3192 |
| `shared::smallsort::sort4_stable` | `0xc7500` | 509 | 509 |
| `shared::smallsort::insertion_sort_shift_left` | `0xefdb0` | 363 | 363 |
| `shared::pivot::median3_rec` | `0xc7700` | 283 | 283 |

All six sorting helper sizes return to the baseline sizes. This verifies the
layout and its sorting code generation; the paired public-training runs will
determine the CPU-time effect.
