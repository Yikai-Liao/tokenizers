# BPE retry and mutually exclusive working storage

All five changes are retained on `simplify/bpe-flat-20261010`, starting from
`499fd7abeb9e7a60ae8fffcf62b1bc694cd02761`. The retained source commit is
`1e3f87bf3b1928f8ddfca8556806c13d9639e6db`. The user accepted differences around
1–2% as an acceptable cost for simpler maintenance and authorized pushing the
retained change. The parity implementation and feature were outside this work.

## Representation and maintenance changes

1. `AttemptOutcome::Complete / RestartForReuse` replaces the retry's local boolean
   and `Option<ModelParts>` translation. A restart finishes merge progress and
   returns directly. The selected alphabet survives the retry; partial models
   and test traces stay private. Successful explicit drops and retry cleanup
   order remain intact.
2. `InitialTokenIds::decoration_flags` owns the prefix/suffix bit rule used by
   vocabulary activation and symbol lookup. Both callers preserve raw UTF-8
   first/last boundaries, filtering order and decorated-string caching.
3. An empty `Builder::append` destination takes ownership and returns. Nonempty
   destinations retain ordering checks and width promotion.
4. `GroupPositions::Ordered / Unordered` replaces two concurrently present
   buffers. The owner chooses the representation at construction; recording and
   publication match the valid payload. Fresh order, reuse sorting with
   duplicates, count overflow checks and signed-ledger publication are unchanged.
   The layout probe using the actual Builder definition and release SmallVec
   dependency reports `Group` **64 → 40 bytes** on this target.
5. `FreshMatches::Ordinary(SelectedRules) / SelfPair(starts)` replaces a mode
   boolean and parallel lookup/start fields. Ordinary endpoint lookup has one
   owner; AA preparation constructs only its nonoverlapping starts, omitting
   the selected-pair map and head/tail arrays. Task partitioning, lazy boundary
   lookups and shared neighbor-event logic remain unchanged.

The production diff adds 57 lines. Its maintenance benefit is fewer implicit
valid-buffer rules and state translations, with no new module, trait framework
or duplicated event algorithm. The extra enum constructors and matches make
the existing mode distinctions explicit. `Birth`, `Writes`, dense initial-count
staging, frozen positions, arena thresholds and heap certification are retained.

## Builds and comparison protocol

The four cumulative arms are `baseline`, `local` (changes 1–3), `group` (1–4),
and `snapshot` (1–5, the final source). Exact ordinary BPE sources are under
[source](source); unchanged parity source is available at the baseline commit.
[builds.json](builds.json) records source and binary hashes, compiler version,
build command and release settings. The runner uses opt-level 3, fat LTO and one
codegen unit with no default training features. The final working sources match
the archived `snapshot` source hashes.

The local changes did not produce identical release `.text` or `.rodata`
sections; [local-sections.json](local-sections.json) records that check. Training
measurements, rather than a byte-identical-code claim, support retention.

The real-input comparison uses the existing frozen Chinese 256 MiB ByteLevel
and Whitespace inputs, vocabulary 50,000, minimum frequency 2, no affixes, and
four workers pinned to CPUs 0–3. Each input has one excluded four-arm warmup and
two formal blocks with rotated arm order. Public `do_train` timing excludes
loading, model validation and serialization. HWM is captured before validation.
No task build or other benchmark runs concurrently; measured child swap is zero.

The table reports medians of within-block percentage changes against the
baseline. The ratio of absolute medians can differ from these paired values.

| Input | Arm | Wall | CPU | Process HWM |
| --- | --- | ---: | ---: | ---: |
| ByteLevel | local | −1.64% | +0.21% | −0.35% |
| ByteLevel | group | −4.69% | −1.88% | −0.29% |
| ByteLevel | snapshot (retained) | −1.70% | −0.13% | −0.14% |
| Whitespace | local | −2.83% | −2.26% | −0.05% |
| Whitespace | group | −1.35% | +0.16% | +2.30% |
| Whitespace | snapshot (retained) | −1.42% | −0.64% | −0.10% |

All 24 complete models agree for their respective inputs. The final cumulative
version has lower paired median HWM on both real inputs. The intermediate group
arm's Whitespace HWM increase is preserved in the table; a smaller object layout
does not establish a lower whole-process peak. Small sample count, shared-VM
noise and worker scheduling limit these results to descriptive comparisons.
They do not establish a general speedup.

## Directed AA and reuse costs

[generate-directed-inputs.py](generate-directed-inputs.py) generates fixed-seed
core-runner inputs with lexical reconstruction order:

- AA: 8,192 words consisting of 4,097 `a` characters and a five-digit index,
  exercising long greedy self-pair runs and restart-block boundaries.
- Reuse: 32,768 indexed words with repeated `aaaaaaaaabcd` and a `baaba` ending,
  with suffix `a`. Initial active `aa` makes the `(a, a)` merge request a retry
  for identity reuse. This measures the affix and historical-cohort path.

Both use weights 1–17, vocabulary 4,096, minimum frequency 2 and four workers.
Each has an excluded AB warmup followed by formal BA and AB blocks comparing
baseline with the retained `snapshot` arm. All 12 complete models agree and
child swap is zero.

| Directed input | Paired wall | Paired CPU | Paired process HWM |
| --- | ---: | ---: | ---: |
| AA | +1.17% | +0.53% | +2.04% |
| Reuse | −2.56% | −3.27% | −0.62% |

AA's formal absolute median HWM is 489,182 → 499,162 KiB (about 9.7 MiB higher).
This observation was reported to the user before push; retention follows the
user's acceptance of differences at approximately this scale. No universal
memory non-regression or smaller process peak is claimed. Allocation profiling
was not performed; the removed ordinary AA allocations and Group layout savings
are structural facts, not an attribution for the measured RSS differences.

## Validation and evidence

Default and no-default-feature library suites each pass all 20 tests. These
include 64 generated cases comparing every rule and complete model against the
sequential oracle at 1, 4 and 8 workers, directed historical-cohort ordering,
affix/Unicode filtering, storage width and order checks, public progress and
feed contracts. All-target Clippy passes with warnings denied. Changed Rust
files pass rustfmt and the diff passes whitespace checks. No unsafe code or
public API changed. Logs and commands are in [validation](validation).

[screen](screen) and [directed](directed) retain raw run records, manifests,
summaries, every job/stdout/stderr, and a compressed complete reference model
for each input. Records retain validation results and complete-model hashes for
every run. Absolute paths identify the actual local inputs and binaries; input
hashes and build metadata identify the assets required for reproduction.

To reproduce, restore one archived source arm into the baseline checkout, build
with the recorded runner command and environment, and copy the runner to
`/tmp/bpe-readability-<arm>-bin`. Then run:

```sh
BPE_MEASURE_ARMS=baseline,local,group,snapshot BPE_MEASURE_ROUNDS=3 \
  BPE_MEASURE_ROOT=/tmp/bpe-readability-screen-new python3 measure.py
python3 generate-directed-inputs.py
BPE_MEASURE_INPUTS=/tmp/bpe-readability-directed-inputs/inputs.json \
  BPE_MEASURE_ARMS=baseline,snapshot BPE_MEASURE_ROUNDS=3 \
  BPE_MEASURE_ORDER=reverse BPE_MEASURE_ROOT=/tmp/bpe-readability-directed-new \
  python3 measure.py
```

The scripts run from this evidence directory; the existing real-input manifest
must be present at the path used by `measure.py`. Build metadata must be copied
to `/tmp/bpe-readability-builds.json` before measuring.
