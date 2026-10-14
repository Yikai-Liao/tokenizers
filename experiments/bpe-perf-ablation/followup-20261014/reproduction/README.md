# Reproducing a control

The runner sources and lock are frozen from the common frontend used for all arms. The `runner/Cargo.toml` paths point to the containing repository's tokenizers tree as a template. For a measured arm, export its named commit into a separate directory, copy the runner, and change the two dependency paths to that exported tree. Do not measure the diagnostic branch tip as the `36c819c5` control: its Rust tree contains an independent experiment; choose the named commit for the desired arm.

Example for the main observer (run from the tokenizers Git checkout):

```sh
mkdir -p /tmp/bpe-main-control/source
git archive e4f787dc189d9be7192107490d652096cde7480e tokenizers | tar -x -C /tmp/bpe-main-control/source
cp -R experiments/bpe-perf-ablation/followup-20261014/reproduction/runner /tmp/bpe-main-control/runner
```

Set `tk-train` and `tk-encode` paths in that copied Cargo.toml to `/tmp/bpe-main-control/source/tokenizers/tk-train` and `/tmp/bpe-main-control/source/tokenizers/tk-encode`. For phase diagnostics, apply `../patches/main-to-main-detail.patch` to the exported source by running `git apply` from that source directory with the patch's absolute path. For ordinary timing, leave the exported production source unmodified. Build with the supplied profile and no default features; keep dependencies identical between controls. Older commits do not use every current dependency, so a necessary lock pruning must preserve every common package/version pair.

The runner takes one protocol-v1 job JSON path. `mode: core` loads a prepared-words JSON with schema_version 1, pretokenizer bytelevel_regex or whitespace, strictly lexical ordered_words, and hash_seeds [11,13,17,19]. `mode: prepare` can construct this input from text outside the training interval. See `runner/src/protocol.rs` and `inputs.json` for the schema and the original host's paths. Those absolute input paths must be changed on another host. Input hashes are in `../input-manifest.json`.

Pin each child to CPU 0–3 for four workers and set RAYON_NUM_THREADS=4 and TOKENIZERS_PARALLELISM=true. Use workers=4 in the job. One worker uses CPU 0 and workers=1. Reuse a frozen prepared input; reverse arm order in alternate blocks. Compare full parsed models after the measured training interval, not just vocab/merge lengths or raw JSON byte order. The existing `run.py` provides these checks and swap/thread monitoring after its inputs/arms are laid out next to it.

For perf, replace the copied runner's `src/core.rs` with `core-perf.rs` and add `debug = 1` to its release profile. The gate is inactive unless BPE_PERF_CONTROL/BPE_PERF_ACK are supplied. The controller in `profile_pair.py` creates both FIFOs and holds O_RDWR descriptors so opening them cannot deadlock, starts `perf record -D -1 --control=fifo:control,ack`, and enables/disables sampling exactly around `do_train`. Loading and model serialization are excluded. Its paths and arm names are host templates to adapt. Keep perf diagnostics separate from formal timing.

The allocator observer builder is also included as an isolated diagnostic template. It counts calls and cumulative requested capacities by joined phase and thread. Run it separately; allocator instrumentation overhead and requested bytes do not equal production CPU time or resident memory.

The complete-producer control has no observer: apply `patches/current-to-current-complete-bypass.patch` directly to an exported `36c819c5` tree and use the ordinary runner. It is independent of the scratch, grouping, modulo and old encoder controls. The corresponding fine observer arms are for stage attribution only. Coverage follows the actual yielded Position chunks, not the requested chunk length; AA, reuse and (id,id) births retain the generic owner path.

`allocation-cpu-v2-results.json` records a sampled timing estimate with live per-sample empty-clock calibration. Do not interpret its corrected seconds as an ablation speedup. `allocation-sites-summary.json` locates allocation work in a separately labeled observer; no-shrink and exact-encode runs provide the independent timing controls.
