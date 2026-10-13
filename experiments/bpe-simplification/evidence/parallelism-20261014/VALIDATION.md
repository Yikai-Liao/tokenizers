# BPE upstream parallelism alignment

Baseline: `d5181300a85ce59031cbbe2d618845aecaf90d97` on
`refactor/bpe-corpus-boundary-20261013`. Upstream inspected:
`huggingface/tokenizers` main `fb49a29223724a96f26e061bec8ec7517aa14177`.

Ordinary BPE now follows upstream's conditional iterators from
`tk_encode::parallelism` and the ambient Rayon pool. It no longer builds a
training pool. Alphabet scanning, initial counting and owners, corpus layout
and materialization, merge preparation and writes, index commit, sorting and
retirement respect the repository parallelism setting. Retirement retains its
existing parallel grain behind an explicit setting check. Parity BPE is unchanged.

The filesystem remains flat. `BpeTrainer::merge_next` is in `mod.rs`;
`corpus.rs` loads the sole `merge.rs` as a private child module. The private
codec is inline in `positions.rs`.

## Training checks

Both commands finish with **35 passed and one existing failure**:

```sh
cargo test --manifest-path tokenizers/tk-train/Cargo.toml trainers::bpe -- --test-threads=4
cargo test --manifest-path tokenizers/tk-train/Cargo.toml --no-default-features trainers::bpe -- --test-threads=4
```

The failing `tokenizer_roundtrip` encodes Chinese `测` as ID 3 rather than
the expected ID. The unmodified baseline reproduces it with the same dependency
lock. It is outside this parallelism change.

The subprocess regression installs caller pools with one/two workers while
the encoder setting requests four. It checks the actual pool in alphabet,
initial count/owner, materialization, fresh/reuse preparation, commit and
retirement. Disabled overrides and `TOKENIZERS_PARALLELISM=false` keep callbacks
on the calling worker and leave the parallel-use flag unset. A long-word model
matches the independent sequential oracle in each setting.

Clippy with no default features and all targets succeeds, retaining only two
pre-existing redundant-field warnings. The changed-file formatter check and
`git diff --check` pass. A Luna review and its follow-up found no missed
production parallelism entry points.

## Benchmark checks

The separate `Yikai-Liao/tokenizers-bpe-benchmarks` repository was checked at
local commit `10ee2e4` (including remote main `0fa8d08`). Its **70 tests pass**.
Its supervisor already exports `RAYON_NUM_THREADS=W`, enables repository
parallelism and binds each child to W CPUs; no framework source change is needed.
The archived supervisor also exports that Rayon setting. Prior dedicated BPE
pools were sized by the runner's `set_num_threads(W)`, so their training worker
count was controlled as well.

The framework builds both baseline and candidate using the same runner lock
and release profile. Its two-revision integration smoke completes **112 native
process runs**, with exact vocabulary IDs and ordered merges, core/pipeline
equivalence, immutable resume and generated reports. It covers one/two workers,
four preprocessing definitions, affixes, empty input and zero vocabulary target.
Every attempt records the matching Rayon environment and CPU affinity.

- Baseline build: `684686d2798653b21bc753c2b2c9caa6d807b04ccf6cbc48665309385bd39ec1`
- Candidate build: `849c2f7d1820f4e92fbf3447c0db9d2137ce6c2e392ee0e4bcf191bc1b1afe51`

All candidate BPE Rust file hashes match the framework's frozen source snapshot.
This smoke verifies operation and correctness, not performance.

The repository-local evidence runner now checks and records effective Rayon
workers. `run_evidence.py` supplies the worker environment, and diagnostic
generation accepts the nested, flat and inline-coordinator layouts. Trace and
stage runners both build and complete five fixtures at one/two workers;
all 20 model outputs agree, with fresh/reuse/restart coverage preserved.

Local logs, per-attempt worker records, trace/stage summaries and generated
framework reports are retained under
`/root/code/tokenizers-bpe-benchmarks/.bench/parallelism-20261014`.
