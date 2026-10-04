# Preliminary BPE comparison and ablations

This suite runs the current trainer and eight variants derived from the same
source. It writes to a separate experiment directory and labels its results
`preliminary`. Review and the later formal experiment remain separate steps.

Copy `preliminary.example.json`, replace the three corpus paths, and run:

```sh
python3 benchmarks/bpe-suite/suite.py all \
  --config /absolute/path/preliminary.local.json \
  --out /absolute/path/preliminary-results \
  --release-cache /absolute/path/shared-release-cache
```

The repository must contain the recorded Git revisions. The entry also supports
`prepare`, `build`, `smoke`, `run`, and `report` separately. Builds use one shared
Cargo cache. Measurement starts after builds and correctness checks finish.
`run` resumes completed paired blocks and reruns an interrupted block in full.
Use a new output directory after changing source revisions or parameters.

The primary comparison uses original HF BPE at `bbccb051`, PR #2348 BPE at
`6ac0de53`, and Full at `f9ff8b97`. The measured trainers share the Full frontend,
dependency skeleton, dependency lock, release profile, streaming feed, and word
map insertion order. The word map uses fixed seeds `[11, 13, 17, 19]`; internal
trainer maps keep their native random seeds. Separate native HF and peer binaries
check that the ported trainers preserve exact outputs on smoke inputs. Native
control timings are excluded.

Training time measures the public `do_train(&word_counts)` call, including the
returned vocabulary and ordered merge strings. End-to-end time includes common
file reading, pretokenization, word aggregation, and training. It ends before
validation and does not build an inference model. Whitespace is `WhitespaceSplit`.
ByteLevel applies the GPT-2 regex and byte-to-character encoding, without adding
a prefix space, and supplies all 256 alphabet symbols. Lines retain newlines.

| Variant | Change from Full |
| --- | --- |
| `no_radix` | Stable comparison sorting, retaining the cached keys and grouping |
| `one_rule` | At most one certified rule per round |
| `flat64` | Full U64 position values instead of G128, with the same inline cases and allocator |
| `u32_corpus` | U32 corpus slots, retaining deferred construction |
| `eager_corpus` | Materialize the corpus before initial grouping |
| `no_position_arena` | Zero pooling cutoff, retaining leases, locks, inline lists and G128 scratch reuse |
| `scalar_weights` | Scalar weight accumulation and interval searches |
| `dual_strings` | Independently owned vocabulary text in the lookup map and ID vector |

The flat-position variant necessarily changes the encoding workspace. All deltas
are conditional on Full and must not be added together. The supplementary peer
control restores its pre-WordArena BPE source. That arena stores corpus symbols;
Full's arena pools position buffers, so the two controls measure different objects.

Correctness directly compares vocabulary entries in token-ID order and merges in
rank order. Each run writes its canonical output after timing. One reference model
per case is retained; matching temporary copies are removed. A mismatch stops the
experiment and retains the differing model for inspection.

Each run records wall time, CPU time, training HWM, sampled RSS and swap, word and
symbol counts, initial edges, host load, parameters, and raw stdout/stderr. HWM is
read before validation; sampled RSS includes validation and is a separate metric.
Allocation counters are disabled. Memory, RSS, and timeout guards retain failures.
Only complete paired blocks enter statistics. Warm-ups are recorded separately.
The initial profile uses five repetitions with rotated and reversed arm order.

`prepared.json` records revisions, settings, dataset sizes, and machine information.
`summary.json` retains valid raw samples and failures; `summary.csv` reports medians
and ranges. Old experiments and binary archives are unnecessary for this suite.
Docker packaging, dataset publication, reviewed formal runs, physical-core
scaling, and capacity sweeps belong to later phases.

Run supervisor checks with `python3 -m unittest discover -s benchmarks/bpe-suite`.
