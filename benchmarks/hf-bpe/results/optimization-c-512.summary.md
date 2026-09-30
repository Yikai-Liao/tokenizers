# Optimization C: alphabet and direct corpus fill (512 MiB)

One focused 512 MiB run of the `corpus-direct` worktree through the original BPE trainer API. It follows Optimization A's ordered parallel word regions and adds a parallel presence-bitmap path for unlimited alphabet collection, a read-only Unicode-to-ID table for corpus fill, and exclusive initialization of final corpus slices without serial zeroing. The candidate keeps four initialization and four merge workers, u32 token IDs and postings, and a non-atomic corpus.

## Locked setup

- Worktree: `/root/code/tokenizers-worktrees/corpus-direct`, commit `fbdc0b2bf736ffdc12aed3d0c3e0aba4a0004caa`.
- Binary: `hf-bpe-native-corpus-c`, SHA-256 `8f98548f9b1272367fe074b246f5bd9fd998bbc5b5e37ae378a50097dae84d86`.
- Input: `zh-512m.txt`, 536,870,289 bytes, SHA-256 `a0d40d4102cba1e933f25e5ccd17552d2eaebaec4ef770bc39d8e46e740b4d4f`.
- Parameters: backend `reference`, split `none`, vocab size 50,000, minimum frequency 2; feed parallelism disabled, four initialization workers, four merge workers; layout `parallel_u32_flat32`, atomic corpus disabled.
- Provenance locks for the worktree, instrumented build copy, runner, build script, binary, corpus, and environment are in `optimization-c-512.environment.json`.

## Result and detailed phases

| Measurement | Optimization C |
|---|---:|
| Feed | 4.323 s |
| Train | 39.805 s |
| Initialization | 9.767 s |
| Merge | 26.092 s |
| Alphabet collection | 0.347 s |
| Alphabet scratch allocation | 12,135,736 bytes (11.57 MiB) |
| Character lookup table | 4,456,448 bytes (4.25 MiB) |
| Corpus region measurement | 0.187 s |
| Final corpus allocation | 0.000046 s |
| Corpus fill | 0.555 s |
| Tokenization | 1.090 s |
| Initial routing | 0.573 s |
| Initial pair counting | 8.015 s |
| Initial corpus allocation | 849,691,660 bytes |
| Initial posting allocation | 1,147,872,496 bytes |
| Peak RSS | 3.43 GiB (3,591,700 KiB runner HWM; 3,656,011,776 bytes sampled) |
| Minimum `MemAvailable` | 4.38 GiB (4,706,009,088 bytes) |
| Sampled process VmSwap peak | 0 bytes |
| Host `pswpin` / `pswpout` deltas | 219 / 11 pages |

The run confirmed the expected settings: `workers=4`, `initialization_workers=4`, `atomic_corpus=false`, and `layout=parallel_u32_flat32`. The runner required all six C-specific numeric fields and would fail if any were absent.

## Process and host diagnostics

Child `RUSAGE_CHILDREN` delta: 124.409 user CPU seconds, 6.569 system CPU seconds, 517,534 minor faults, 0 major faults, 32,280 voluntary context switches, and 27,039 involuntary context switches. Child wall duration was 45.641 s.

Host aggregate CPU busy fraction during the interval was 55.99%; CPU steal fraction was 0.026%. Load average (1/5/15 minute) changed from `[1.08, 1.23, 1.16]` to `[2.42, 1.55, 1.27]`. These counters describe the host interval and do not isolate activity caused by the benchmark.

## Model signature gate

Optimization C matches the PR and all previous native runs on model SHA-256 `d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd`, actual vocabulary 50,000, 29,243 merges, and 1,429,915 unique words. The eight-run gate result is `optimization-c-512.signature.json`.

## Artifacts

Raw result, phase data, and process/host counters are in `optimization-c-512.jsonl`; stderr and full source/binary/input provenance are in the corresponding `.stderr` and `.environment.json` files.
