# DE hot-path profile: 32 MiB diagnostic

## Scope and provenance

One profiling run was recorded; this is a diagnostic, not a performance comparison. Command parameters were `none reference 50000 2`, with `TOKENIZERS_PARALLELISM=false`, `RAYON_NUM_THREADS=4`; DE's default build uses init/merge workers 4/4 and atomic u32 corpus.

- Worktree: `/root/code/tokenizers-worktrees/radix-posting-bulk`, clean at commit `c8702374bd8a3812f8bca34cd53e21afe01632c3`.
- Binary: `benchmarks/hf-bpe/target/release/hf-bpe-native-radix-posting-bulk`, SHA-256 `8d260f87357da76d4ac0c90fddfcac5cc86632cb38ca94584cb9fd65df500bee`.
- Input: `/root/code/tokenizers-bpe-benchmark/data/text/zh-32m.txt`, 33,554,395 bytes, SHA-256 `6624193cbcc72657f766bf68aee0fe78be129578137ad7d6b4870d03147f4e0d`.
- Tracked Rust/Cargo source hash manifest SHA-256: `60069dc7672f85c70dfc1594ee4f1e46642b63571a351038a5cf6f1b51846e73`. Selected source SHA-256: `parallel.rs` `36b40c9ef701052c442c00d5ac54b9be6fedfbd50284000b9c0a16e5d0795284`; `parallel/weight_lookup.rs` `43c233b72278e35c83f006b958462262d5512d72dba2813b5d0626d19fa1be10`; `parallel/fused_batch.rs` `4a8d122f0e4a9252dba226731334424c5cc7fbdb3b4d64415f4c19cd05c8205f`; `parallel/radix_count.rs` `a2f53ae830dc4edad3a148ee142bccd70fb1ea1b2f45824b56e6a2f27fb2242e`.
- Model result: train 2.953 s, elapsed 3.271 s, peak RSS 305,660 KiB; vocab 50,000, merges 38,903, unique words 125,156, model SHA-256 `c8a8f56b51336799160204e20a377830a6ff951231f8f3e81e1ac63d4e671093`.

Profile command: `perf record -F 199 -e cycles:u -g --call-graph dwarf,16384`. The capture has 1,776 samples, zero lost samples, and approximately 18,329,077,861 event cycles. `perf report --stdio --no-children --sort symbol` was used; Rust v0 names were demangled with `c++filt -s rust`. The report's overhead is sampled cycle weight. It is not wall time or a phase timer.

## Relevant sampled symbols

Shares below use each sample's top instruction symbol; `perf report` overhead is included as a cross-check. Counts are out of 1,776 recorded samples.

| Symbol / interpreted stage | Samples | Sample share | `perf report` overhead |
|---|---:|---:|---:|
| `train_in_pool` worker closure receiving `(usize, &mut Owner)`; owner commit map at `parallel.rs:1179` | 278 | 15.65% | 15.32% |
| `fused_batch::prepare` worker closure | 240 | 13.51% | 13.59% |
| `WeightLookup::weight::<u32, 2>` | 145 | 8.16% | 7.96% |
| `radix_count::sort` worker | 92 | 5.18% | 5.18% |
| `radix_count::initialize` closure #4, grouped owner stream | 65 | 3.66% | 4.86% |
| `Output<u32, 2>::birth` | 68 | 3.83% | 3.82% |
| `Output<u32, 2>::remove` | 67 | 3.77% | 3.79% |
| Unresolved top symbols | 170 | 9.57% | — |

The `WeightLookup::weight` function is a distinct symbol at binary address `0x2e5e40`, size `0xf9` bytes. The implementation is reached from both initial radix grouping (`radix_count.rs:162`) and fused preparation (`fused_batch.rs:152`), so it is a direct profile target for the proposed conservative all-256-buckets-equal fast path. `Output::birth` and `Output::remove` are sampled independently at about 7.6% combined. The radix sort and grouped worker also appear in the sampled profile. The owner-commit closure is attributed from its demangled `(usize, &mut Owner)` input and the corresponding `owners.par_iter_mut().enumerate().map(...)` block; this is a symbol/source mapping, not a separate phase timer.

This single 32 MiB capture supports that `WeightLookup::weight` is worth testing in the planned H change. It does not quantify the change's wall-time benefit: samples mix training phases, and the lookup participates in both initial grouping and fused preparation.

## Artifacts

- Raw capture: `benchmarks/hf-bpe/.build/de-hot.perf` (SHA-256 `4df52075391142694fea6b9f9b70d18ec4615be91448acf60771503e05bdada3`).
- Raw `perf report --stdio --no-children --sort symbol` output: `benchmarks/hf-bpe/.build/de-hot.report.txt`.
- Captured command stdout/stderr: `.build/de-hot.stdout` and `.build/de-hot.stderr`.
- Complete source/input/binary hashes and sample metadata: `.build/de-hot.provenance.json`.
