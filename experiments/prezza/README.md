# Prezza skippable-text ablation

This branch replaces only the mutable corpus representation of the BPE training
engine. Pair selection, batching, weighted counts, event publication, postings,
Feed and public model IDs retain their existing contracts.

Set `BPE_CORPUS_LAYOUT=prezza` before calling a public BPE trainer to select the
alternative. Unset it to use the existing endpoint representation. Selection
happens once per training attempt and dispatches to monomorphized storage.

## Representation and generalization

The alternative implements §3.1 of Bille, Gørtz and Prezza's
[Practical and Effective Re-Pair Compression](https://arxiv.org/abs/1704.08558),
using the author's
[skippable text](https://github.com/nicolaprezza/Re-Pair/blob/ffb411d8ce9c1232980ec67bee7e678c0b88fd5c/internal/skippable_text.hpp)
as an algorithm reference. This Rust implementation was written independently;
it does not copy the GPL C++ implementation.

It reuses this fork's `U16Slots`, `PackedU24Slots`, `U32Slots` and `slot_bits`.
The smallest plane that admits every retained **initial** ID and the separator
is selected automatically. Merged IDs can occupy the first cell and its following
blank cell when using 16 or 24 bits. A 32-bit cell stores the full ID directly.
No initial-alphabet restriction, ID remapping or narrowing of public IDs is added.
Reserved IDs, affixes and filtered symbols use the existing initial resolver.

| Retained initial ID count bound | Cell width | Text + bitmap + skips, logical bytes |
| --- | --- | --- |
| <=65,535 | 16 | `2U + 16 ceil(U/64)` |
| <=16,777,215 | 24 | `3U + 16 ceil(U/64)` + initialized guard |
| Larger | 32 | `4U + 16 ceil(U/64)` |

The bound is `maximum retained initial ID + 1`, not the number of distinct
symbols and not the final vocabulary target. For example, a 100K vocabulary
starting with 20K low IDs still uses a 16-bit Prezza plane. An initial ID of
65,535 selects 24 bits, even if only two symbols occur. This preserves sparse
reserved-ID configurations without a mapping table.

Live-token starts are marked in an atomic bitmap. Successor/predecessor queries
check the current and adjacent 64-slot blocks; a longer empty run uses the jump
distance stored in its first/last empty blocks. The new token remains at its
left start and the right start becomes blank. The endpoint ID-span table and
occurrence-span plane are unnecessary for this representation. Existing
whole-word/cohort selection remains available for reuse and length admission.

## Parallel safety

The existing preparation phase selects disjoint physical token spans. Bitmap
clears use atomic `fetch_and`, because disjoint spans can share a bitmap word.
Cell stores and long-gap skip cells remain inside each writer's span. During
application, geometry reads atomic bitmap/skip cells for the writer's two
successors; adjacent writes preserve the following live start. Readers of token
IDs run only after the write phase joins. Packed24 scalar reads retain their
existing guard and joined-phase contract.

Tests compare random merges against an independent list of live positions,
check full-width IDs (including cell values equal to separator codes), long gaps,
word separators, and adjacent concurrent bitmap clears. The existing engine
trace/reference tests run with both corpus selections, including affixes,
reserved IDs above 65K, identity reuse, weighted counts and overflow.

## Reproduce

Use `/root/code/tokenizers-bpe-benchmarks` or its published repository, with its
Python environment and locked runner builder. Build the latest-main baseline
and this branch with the same runner lock and flags. The benchmark runner uses
release optimization, fat LTO and one codegen unit. `BPE_CORPUS_STATS=1` emits
allocation diagnostics; leave it unset for timing.

From the benchmark repository:

```sh
export PATH=/root/.cargo/bin:$PATH
uv run python -m bench lock --source /root/code/tokenizers \
  --revision 8faaff79d859bfd6b2417cfe8c93ea2851c3aaca \
  --lockfile .bench/prezza/runner.lock
uv run python -m bench build --source /root/code/tokenizers \
  --revision 8faaff79d859bfd6b2417cfe8c93ea2851c3aaca \
  --lockfile .bench/prezza/runner.lock
uv run python -m bench build --source /root/code/tokenizers-prezza \
  --lockfile .bench/prezza/runner.lock
uv run --extra corpus python -m bench corpus \
  --dataset datasets/wikipedia-zh.json --size-mib 512 --out .bench/prezza/zh512
uv run python -m bench prepare-input --input .bench/prezza/zh512/text.txt \
  --pretokenizer whitespace --build MAIN_BUILD_JSON --out .bench/prezza/zh512-words
PYTHONPATH=. uv run python /root/code/tokenizers-prezza/experiments/prezza/run.py \
  --baseline MAIN_BUILD_JSON --candidate EXPERIMENT_BUILD_JSON \
  --words .bench/prezza/zh512-words/manifest.json --out .bench/prezza/zh512-matrix
```

The main matrix includes unmodified main, the experiment binary's endpoint
control, and the same binary's Prezza arm; both 50K and 100K vocabulary targets;
1/2/4/6 workers; a warmup and three balanced paired blocks. Configs, CPU affinity,
source and binary hashes, input identities, raw wall/CPU/RSS measurements and
exact-model comparisons are retained by the benchmark protocol. Core load and
model serialization are excluded from `do_train` timing; process HWM includes
loading. Diagnostic and correctness-smoke runs use separate output directories.
