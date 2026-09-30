# H owner commit: bounded direct-IP attribution

This reuses the existing H DWARF profile; no new training or perf collection was run. The authoritative full event records are `../.build/optimization-h-debug-profile.ip-only.txt` (15,897 samples: 8,084 cycles and 7,813 cache misses). The owner worker is identified directly by its raw IP symbol, demangled with `c++filt -s rust` as `train_in_pool::<Atomic<u32>, u32, 2>::{closure#21}::{closure#0}`. `nm -anC` locates that function at `0x22bfe0`; one bounded `addr2line -afiC` batch resolved the top offsets (15 unique executable addresses from the top 20 cycle and top 10 cache-miss offsets).

The worker's direct top-IP samples sum to 37,866,992,348 cycles (**18.94%** of the full 199,974,980,103-cycle period) and 222,438,828 cache misses (**15.76%** of 1,411,570,990). This is exclusive direct-IP attribution to the owner commit closure, not an inclusive call-chain share or wall-time estimate.

| Closure offset | Cycles | Cache misses | Source location / operation |
|---|---:|---:|---|
| `+0x221` | 4.4022% | 3.0586% | `parallel.rs:1200`; load the surviving ledger entry's frequency after `get_mut`, followed by checked subtraction/store |
| `+0x1c8` | 3.0116% | 1.6765% | hashbrown `get_mut` probe for `ledger.entries`, `parallel.rs:1199` |
| `+0x12ff` | 2.3461% | 1.6911% | traversing the route node chain (`node.next` / position) inside posting append, `parallel.rs:1240–1244`, `small_posting.rs:167` |
| `+0x1248` | 2.0473% | 1.7675% | hashbrown `get_mut` probe for the ledger birth entry, `parallel.rs:1233` |
| `+0x12b0` | 1.3474% | 0.7599% | load the existing posting length before the checked append range, `small_posting.rs:147` |
| `+0x1309` | 0.5331% | 0.5318% | loop decrement in the heap bulk-fill path, `small_posting.rs:166` |

These selected addresses show material ledger lookup/update and posting-chain/bulk-fill work in the commit closure. They do not isolate all of the closure's work. The temporary `born` hash aggregation at `parallel.rs:1189–1198` is in the same closure, but this bounded direct-symbol selection did not establish a separate top-IP share for it. No larger cost attribution is inferred.

Bounded disassembly from the actual function boundary confirms the operations: `+0x221` is `mov -0x8(%r8),%rax`; `+0x12ff` is the Node `next` load `mov 0x4(%r10,%rdi,8),%edi`; `+0x12b0` loads the posting length; `+0x1309` decrements the reverse-loop index. Probe offsets sit immediately after vector control-byte loads. This does not assign every sampled period to the exact load or arithmetic instruction: both events can skid, and the generic cache event does not identify a precise cache level or establish a memory-latency cause. In particular, 4.4022% at the frequency site is not a measurement of integer-subtraction cost alone.
