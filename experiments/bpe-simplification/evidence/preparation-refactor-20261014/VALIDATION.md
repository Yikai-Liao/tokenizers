# Ordinary BPE preparation refactor

Baseline: `35d73e24c3ff07d47d7a442edacbcfb7073f40f3` on
`refactor/bpe-corpus-boundary-20261013`. Parity training is outside this change.

The scan used three principles: retain a reference or entry after selecting a
container element; place state-dependent private operations with their owner;
share complete operations while retaining mode-specific correctness rules.

`NeighborChanges::group` borrows its selected changes vector, and decoration
lookup retains the selected ID row. `Corpus::scan_word` is a private method;
it reads the word weight once and reuses a reconstructed neighbor span.
Alphabet selection and observed-special selection are trainer methods.
`Corpus::prepare_chunk` owns task initialization, scan error propagation, and
result assembly. AA/AB/reuse retain their task grains, input lifetimes, ordering,
and different scan rules. Shared endpoint publication, candidate construction,
and public training-result assembly now each have one implementation.

A fresh independent agent reviewed all eight ordinary production modules and
the Rust diff. It found one remaining repeated map lookup in fresh count removal.
The final `Occupied` entry implementation keeps absent keys absent, returns the
same subtraction error without changing the count, and retires only below the
frequency floor. The reviewer checked that final patch and found no further
confirmed issue. Repeated indexing across alias-span materialization remains:
that operation must read the old ID metadata before it can be updated.

## Verification

Final-source BPE tests pass with default features and without default features:
**35 passed in each configuration**, excluding the known `tokenizer_roundtrip`
failure. A full default run again reproduces that failure: Chinese `测` encodes
as ID 3 instead of its vocabulary ID. The unmodified baseline reproduction is
recorded in [the prior validation](../parallelism-20261014/VALIDATION.md).

The final trace runner builds with the locked release profile. Five fixtures
(AA, compatible AB batch, reserved identity, alias restart, and length-gated
reuse) at one/two workers produce **10 exact model and merge-trace matches**
against the prior aligned trainer. `validation.json` records input coverage,
model/trace hashes, and hashes of the final ordinary Rust sources. These are
correctness checks; no performance conclusion is claimed.

The initial refactor also passed the external framework's 112-process public
core/pipeline smoke, including four preprocessors, affixes, empty input,
zero targets, exact model comparison, and resume. The later entry refinement
was verified by the final-source unit tests and the 10 trace comparisons above.

The ByteLevel preparation launcher and Chinese profiling launcher now set the
child Rayon environment from their requested workers. English and Chinese
ByteLevel preparation completed with the current worker-checking runner under
parent `RAYON_NUM_THREADS=4` and `TOKENIZERS_PARALLELISM=false`; both children
reported requested/effective workers of 1. All local Python runner-launch sites
were inspected; `run_evidence.py` already provided the matching environment.
Python parsing and both diagnostic generators pass.

Clippy without default features and with all targets succeeds, retaining only
the two existing redundant-field warnings. Changed-file formatting and
`git diff --check` pass. Raw local outputs are under
`/dev/shm/bpe-preparation-final-trace`, `/dev/shm/bpe-preparation-bytelevel`, and
`/dev/shm/bpe-preparation-integration`.
