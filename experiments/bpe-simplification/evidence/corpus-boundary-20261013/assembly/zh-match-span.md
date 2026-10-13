# Matched-span code-generation isolation

The frozen aligned production binary is
`753d1c6040a6387d5e46240b373f67d498e1c9a9b1a0f5be5e89ef81722eeba4`.
The isolated `match-span-production` binary is
`c395fb9f584cec1d4743fac99c2e0b6fab3b81debad2267cf622fb3ba00e5527`.

The temporary source diff changes only `FreshPreparation::record`: it computes
`matched_span = matched.end - matched.start` at entry and uses that value in the
two neighbor birth-length checks. The reader still returns standard `Range`
values. Tasks, adjacency callbacks, event order, Corpus boundaries and the local
iterator borrow are unchanged. The fixed matcher supplies valid ranges with
`end > start`, which is the internal precondition for this subtraction.

The AB helper changes from `0xe4950`, 8283 bytes and 1694 static instructions,
to `0xe5120`, 8315 bytes and 1699 static instructions. The whole function includes
cold paths and register-allocation changes; these totals do not determine the
dynamic effect of the isolated operations.

| Operation | Frozen aligned binary | Matched-span isolation |
| --- | --- | --- |
| Left birth-length check | `0xe5986`–`0xe5997`: subtract start from end, zero a register, conditionally select the nonnegative length, then add the prior span | `0xe6166`: directly add the shared value at `rsp+0xd8` to the prior span |
| Right birth-length check | `0xe5cf9`–`0xe5d0e`: compute and conditionally select that length again, then add the born span | `0xe64ce`–`0xe64d6`: load the same `rsp+0xd8` value and add the born span |

Both `Range::len` conditional selections disappear from these checks. The
compiler folds the subtraction back to the reader's fixed matched span: it loads
that reader field from `rsp+0x68` at `0xe58a4` and caches it at `rsp+0xd8` at
`0xe58bc`, before the main match loop. That slot remains unchanged across the
left neighbor-record call and is reused by the right check. The isolated code
therefore reuses the cached span rather than executing one subtraction for every
match. The other unsigned conditional moves in the helper belong to lookahead
and search operations.

This establishes that the isolation removes the introduced length guards and
reuses the value. Paired complete public-training calls determine whether it
explains any of the remaining Chinese single-worker CPU difference; the static
comparison alone does not assign a cause.
