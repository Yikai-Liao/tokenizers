# Optimization A: parallel corpus construction (512 MiB)

One focused 512 MiB run of the `corpus-parallel` worktree through its original `BpeTrainer::train` API. This candidate measures parallel ordered-region preparation, final corpus allocation, and fill separately. The run used the same corpus, tokenizer settings, and four-thread initialization/merge setup as the `count4-parallel` baseline.

## Configuration and locked inputs

- Worktree: `/root/code/tokenizers-worktrees/corpus-parallel`, branch `bpe/corpus-parallel`, commit `98ca7fc1c258d0661177d3aae15cca71536c3df4`.
- Binary: `hf-bpe-native-corpus-a`, SHA-256 `2f3f2d438689c33a2cdae0b106236882e24e2ab2df99c7a2e6707ec56cf8cc43`.
- Input: `zh-512m.txt`, 536,870,289 bytes, SHA-256 `a0d40d4102cba1e933f25e5ccd17552d2eaebaec4ef770bc39d8e46e740b4d4f`. It comes from the locked Wikipedia manifest in the environment file.
- Parameters: backend `reference`, split `none`, vocab size 50,000, minimum frequency 2; feed parallelism off, four initialization workers and four merge workers; u32 token IDs and u32 postings, `parallel_u32_flat32`, non-atomic corpus.
- The trainer measures ordered word regions in parallel, assigns each region a disjoint slice in the final corpus allocation, and combines boundary metadata after the parallel fill. It does not retain an additional complete corpus copy.
- The monitored run stops only if `MemAvailable <= 1 GiB`; process VmSwap and host paging are sampled and recorded.
- Worktree source, instrumented build-copy source, runner, build script, binary, corpus, and environment hashes are recorded in `optimization-a-512.environment.json`.

## Result

| Measurement | Optimization A |
|---|---:|
| Train time | 55.207 s |
| Feed time | 4.636 s |
| Initialization | 16.090 s |
| Merge | 33.964 s |
| Alphabet selection | 2.322 s |
| Corpus region measurement | 0.228 s |
| Corpus allocation | 0.876 s |
| Corpus fill | 2.371 s |
| Peak RSS | 3.40 GiB (3,562,060 KiB runner HWM; 3,640,385,536 bytes sampled) |
| Minimum system MemAvailable | 4.26 GiB (4,571,107,328 bytes) |
| Sampled process VmSwap peak | 0 bytes |
| Host `pswpin` / `pswpout` deltas | 147 / 10,164 pages |
| Initial corpus allocation | 849,691,660 bytes |
| Initial posting allocation | 1,147,872,496 bytes |

The measured trainer layout was `parallel_u32_flat32`, `workers=4`, `initialization_workers=4`, and `atomic_corpus=false`; the runner checked all four new timing fields were numeric.

## Model signature gate

Optimization A matches the PR and the three prior native runs on all four fields: model SHA-256 `d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd`, actual vocabulary 50,000, 29,243 merges, and 1,429,915 unique words. The five-run pass is recorded in `optimization-a-512.signature.json`.

## Comparison with the count4 baseline

Against the single `count4-parallel` baseline run, initialization decreased from 22.975 s to 16.090 s (6.885 s faster). Merge time increased from 27.457 s to 33.964 s; total training time changed from 54.760 s to 55.207 s. The corpus construction phase measurements account for 5.797 s of Optimization A's initialization. Each case has one timing, so these figures show this run's phase distribution and do not estimate run-to-run variance.

Raw output, phase statistics, memory samples, and provenance are in `optimization-a-512.jsonl`, `optimization-a-512.stderr`, and `optimization-a-512.environment.json`.
