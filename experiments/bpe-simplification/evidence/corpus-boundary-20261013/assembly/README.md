# Fresh preparation code generation

The frozen `final-baseline-production` and `final-candidate-production` binaries
use the same single-call runner, locks and x86-64 release profile. The local-borrow
isolation binary changes only the input iterator's ownership within one fresh
preparation task. The final `aligned-candidate-production` also includes the
source PERF comment. Binary and source hashes are retained in the parent manifests.

| AB Rayon helper | Function bytes | Static instructions | Stack allocation |
| --- | ---: | ---: | ---: |
| Baseline | 8251 | 1702 | `0x408` |
| Initial candidate | 8715 | 1758 | `0x4c8` |
| Local borrow isolation | 8283 | 1694 | `0x3e8` |
| Final aligned candidate | 8283 | 1694 | `0x3e8` |

The initial candidate unconditionally copies `0x90` bytes of reader state after
the lookahead deque has been initialized, once per AB task. Borrowing the local
iterator removes this copy. Its decoder remains local while the adapters carry
a pointer; it is consumed once and all reader borrows end before task return.
The task split, coordinate order, 16-entry lookahead and neighbor event loop remain
the same. The final reader and decoder retire before preparation joins and before
corpus endpoint writes.

The initial-fill `memcpy` and `memmove` sites are generic deque growth branches.
They are not reached by filling a capacity-16 deque from `take(16)`, and their
presence does not establish a runtime copy. The coordinate loop retains deque
operations when later rotating the lookahead window.

The `.asm` files come from `objdump -d -C --no-show-raw-insn` restricted to each
helper's address range. `*-symbols.txt` comes from `nm -S --size-sort -C`.
Instruction counts include the full function's cold/error paths. Static size and
stack allocation describe code generation; the complete public-training timer
provides the performance evidence.

Separate scratch probes show nearly equal directory initialization counts and
newly filled entries between the old and initial candidate. Those common costs
are retained for future investigation. This refactor preserves the existing
Rayon splitting policy and scratch initialization.
