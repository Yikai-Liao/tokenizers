from pathlib import Path
import json
root=Path(__file__).resolve().parent
def comparisons(label):
    return json.loads((root/label/'comparisons.json').read_text())
def fmt(value): return f'{value:+.2f}%'
labels={'bytelevel_regex':'Chinese ByteLevel','whitespace':'Chinese Whitespace','en-bytelevel_regex':'English ByteLevel','en-whitespace':'English Whitespace','aa-neighbors':'AA neighbors','reuse-affix':'Reuse / activated alias','reuse-find':'Reuse / cohort find'}
text='''# Ordinary BPE: standard and existing-library logic replacements

Based on `a2f1c0f7fda871ad321579a8df690a5f6464043f`. Only the ordinary BPE
trainer is in scope. The user excluded Parity Trainer and prioritized real
Chinese/English training, permitting small time increases (within about 5%) when
code understanding and maintenance improve. Synthetic regressions are evidence
of a tradeoff, rather than an automatic rejection.

The retained changes use existing dependencies and the standard library:

| Direction | Handwritten responsibility removed | Decision |
| --- | --- | --- |
| `symbols()` + `tuple_windows` / `enumerate` | Callback scan protocol, previous-symbol state, coordinate increments and `ControlFlow` error unpacking | Retain shared filtered iterator |
| `Iterator::find` | Peek/skip/consume occurrence loop | Retain |
| `VecDeque` | Array ring head and modulo bookkeeping | Retain standard-library queue |
| `BTreeMap` partial birth groups | Hash-map collection into a Vec and explicit bucket/pair sort | Retain |
| Rayon `reduce_with` | Explicit bitmap-result Vec and serial union loop | Retain |
| `split_first_chunk::<8>()` | Separate seed/rest slice boundaries and slice-to-array conversion | Retain as readability cleanup |
| `partition_point` initial ranges | Per-word linear search inside each segment | Retain, with documented synthetic tradeoff |

Decorations retain original UTF-8 boundaries while resident coordinates count
retained symbols. The complete-alphabet symbol-count fast path stays in place.
Bitmap results retain 0x110000 bits even for empty input; unobserved initial
alphabet characters retain zero initial activation spans. Restart seeds remain
little-endian and truncated internal seeds panic. Segment boundaries include word
separators; the final boundary is handled separately, zero budgets progress, and
oversized words occupy one segment. Searches use subtraction rather than adding
slots to the starting coordinate.

## Real-corpus comparison

Both source corpora are approximately 256 MiB, from the locally frozen Chinese
and English Wikipedia inputs. Each uses ByteLevel regex and Whitespace counts.
`input-details.json` records source/prepared sizes, hashes, dataset revision and
paths. English Whitespace was prepared once using the frozen baseline runner;
its preparation job and result are archived. All variants train from identical
prepared counts with vocabulary 50,000, minimum frequency two and no affixes.

Rust 1.99.0; release opt-level 3, fat LTO, one codegen unit; four workers pinned to
CPUs 0–3. Timing covers public `do_train`, including output construction and
attempt cleanup, and excludes input loading, preprocessing and serialization.
Process high-water RSS includes loaded counts and earlier input allocations.
Swap is sampled every 25 ms. Builds/tests and competing benchmark processes are
excluded during training measurements. Raw jobs, metrics, full-model hashes,
commands and host details are archived; large inputs, binaries and model payloads
are kept outside Git.

The seven-arm screen has one excluded warmup and two formal blocks per input,
rotating arm order. Core contains `symbols`, edge/materialization adapters and
`find`; deque/tree independently extend core. Bitmap/seed/ranges independently
extend the same baseline, in separate worktrees. Do not add their measured
percentages to predict the combination.

The table reports median paired wall / process-CPU changes; positive is higher.

| Candidate / reference | Chinese ByteLevel | Chinese Whitespace | English ByteLevel | English Whitespace |
| --- | ---: | ---: | ---: | ---: |
'''
screen=comparisons('all-measurements')
for pair in ['core/baseline','deque/core','tree/core','bitmap/baseline','seed/baseline','ranges/baseline']:
    values=[]
    for case in screen:
        d=case['comparisons'][pair]['paired_median_delta_percent']
        values.append(fmt(d['train_seconds'])+' / '+fmt(d['train_cpu_seconds']))
    text+='| '+pair+' | '+' | '.join(values)+' |\n'
text+='''
These descriptive laptop results establish no stable general speedup. The final
combination is separately compared with the original baseline using one excluded
warmup and four alternating formal pairs per input. The complete parsed models
agree and every measured process reports zero sampled swap.

| Input | Baseline / combined wall (s) | Paired wall | Paired CPU | Baseline / combined HWM (MiB) | Paired HWM |
| --- | ---: | ---: | ---: | ---: | ---: |
'''
for case in comparisons('final-measurements'):
    b=case['arms']['baseline'];c=case['arms']['final'];d=case['comparisons']['final/baseline']['paired_median_delta_percent']
    text+=f"| {labels[case['case']]} | {b['train_seconds']:.3f} / {c['train_seconds']:.3f} | {fmt(d['train_seconds'])} | {fmt(d['train_cpu_seconds'])} | {b['process_hwm_kib_before_validation']/1024:.1f} / {c['process_hwm_kib_before_validation']/1024:.1f} | {fmt(d['process_hwm_kib_before_validation'])} |\n"
text+='''
Absolute table cells are per-arm medians; paired changes are medians of
same-block ratios. English ByteLevel has about 12 MiB higher HWM (+6.92% paired),
which is a retained memory tradeoff; the other real-input paired HWM changes are
within 0.4%. The isolated bitmap screen also shows higher English ByteLevel HWM,
but these runs do not isolate a mechanism.

Memory changes are observations of whole-process high-water marks, rather than
isolated container allocation costs. Neither the bitmap reduction nor the queue
replacement is claimed to reduce process RSS.

## Fixed-capacity queue follow-up

At the user's request, ArrayDeque 0.5.1 was tested in a separate worktree against
the standalone VecDeque variant, with the same core source. It stores a fixed
16-element buffer inline, and uses its default Saturating behavior: a full push
returns an error. The refill asserts that the preceding pop freed a slot. Its
additional dependency and lock change remain only in the archived candidate.

VecDeque is a growable ring buffer. At this call site, initial fill takes at most
16 positions and every refill follows one pop, so length never exceeds 16.
The former array ring likewise consumed a slot before replacing it and advanced
its head modulo 16; it never appended beyond its array or overwrote a pending
position. VecDeque changes allocation policy to a heap buffer, while preserving
the lookahead, order and queue-length invariant.

The user explicitly preferred VecDeque when real performance was similar. One
excluded warmup and two alternating formal pairs on each real input show:

| Input | ArrayDeque vs VecDeque wall | CPU | HWM |
| --- | ---: | ---: | ---: |
'''
for case in comparisons('queue-measurements'):
    d=case['comparisons']['arraydeque/deque']['paired_median_delta_percent']
    text+=f"| {labels[case['case']]} | {fmt(d['train_seconds'])} | {fmt(d['train_cpu_seconds'])} | {fmt(d['process_hwm_kib_before_validation'])} |\n"
text+='''
There is no consistent advantage sufficient to override that preference, so
VecDeque is retained. See [VecDeque's standard-library contract](https://doc.rust-lang.org/std/collections/struct.VecDeque.html)
and [ArrayDeque's fixed-capacity contract](https://docs.rs/arraydeque/0.5.1/arraydeque/struct.ArrayDeque.html).

## Directed and micro measurements

The initial four-arm screen additionally exercised AA partial-neighbor groups
and suffix-induced identity reuse. Its raw runs remain in `measurements`; it is
not combined with the later real-corpus screen. Final baseline/combination checks
add a reuse fixture with high-frequency XY merges before an activated AA alias,
so cohort-based occurrence scanning is exercised before full-word alias scans.
Generated word counts are reproducible with `generate-directed.py`; hashes match
`directed-inputs.json` exactly.

| Input | Combined vs baseline wall | CPU | HWM |
| --- | ---: | ---: | ---: |
'''
for case in comparisons('directed-measurements'):
    d=case['comparisons']['final/baseline']['paired_median_delta_percent']
    text+=f"| {labels[case['case']]} | {fmt(d['train_seconds'])} | {fmt(d['train_cpu_seconds'])} | {fmt(d['process_hwm_kib_before_validation'])} |\n"
text+='''
The micro harness imports both complete actual Positions modules. Bitmap and
range functions are extracted from the frozen production sources, with thin
input wrappers (`WordCountsView::keys`, and a plan whose word count is starts.len).
Input construction and position encoding are outside these timers. Both arms are
in one optimized binary, warmed before seven alternating measurements. Bitmap
runs use CPUs 0–3 and 1/4 workers; scalar runs use CPU 0. Raw CSV, source/binary
hashes, the pinned dependency lock and source are archived.

| Diagnostic | Original / standard median | Paired wall change |
| --- | ---: | ---: |
'''
for case in json.loads((root/'micro-summary.json').read_text()):
    name=case['case'];workers=case['workers']
    if name.startswith('positions/') and name not in ('positions/129/distinct','positions/1048576/distinct','positions/1048576/duplicates'): continue
    a=case['medians_ns']['original'];b=case['medians_ns']['standard']
    unit='ms' if a>=1e6 else 'µs';divisor=1e6 if unit=='ms' else 1e3
    text+=f"| {name}, {workers} worker(s) | {a/divisor:.4f} / {b/divisor:.4f} {unit} | {fmt(case['paired_median_delta_percent'])} |\n"
text+='''
The old segment algorithm is O(n) overall. The candidate is about O(k log n)
for k segments, so tiny budgets and many oversized words have a worse synthetic
case. No adaptive threshold or additional fallback protocol was introduced.
The production budget is 2²⁴ resident slots; the real-corpus results and simpler
use of an ordered boundary directory support retaining the standard search under
the user's stated criterion. Large-segment micro gains are not claimed as overall
training gains. Seed slicing is a clarity change without an established speedup.
Bitmap reduction removes the explicit result collection and nonempty final empty
bitmap; chunk bitmaps and the input-key Vec still allocate.

## Validation and source size

Default and no-default-feature library checks each pass 35 tests after filtering
the known encoder roundtrip assertion. The unfiltered default run retains 35
passes and the one existing failure (`测`: actual ID 3, expected 6), previously
reproduced on the baseline in the trainer-interface evidence. It is not counted
as a pass or changed in this task. Parity Trainer is outside scope.

A/B/C standalone candidates each pass 32 no-default library tests; core/deque/tree
and ArrayDeque each pass 31. Sequential integration with the added edge regression
passes 32, 33, 34 and 35 tests. The actual-source micro harness passes 15 tests.
Clippy for all targets with warnings denied, the changed-file formatter and
`git diff --check` pass; exact commands and logs are in `validation.json`.

Existing generated checks compare complete vocabulary IDs, ordered merges and
per-rule traces with the independent sequential oracle at 1/4/8 workers. Added
coverage checks full `(pair, coordinate, weight)` edge streams, empty/single-symbol
words, filtering at UTF-8 ends, affixes and immediate errors at every designated
edge. Bitmap checks cover empty/ASCII/CJK/non-BMP input and inactive initial
alphabet characters. Seed checks retain panic for 0–7-byte truncated seeds and
avoid reading bytes once complete. Generated segment layouts compare every range
with the old linear algorithm, including zero/MAX budgets, exact boundaries,
oversized words and near-usize::MAX coordinates, and assert complete nonoverlapping
coverage with the single-word oversize exception.

'''
counts=json.loads((root/'source-counts.json').read_text());b=counts['baseline'];f=counts['final']
text+=f"Descriptive ordinary-production counts are {b['production_total']} → {f['production_total']} nonblank, noncomment lines ({f['production_total']-b['production_total']:+d}). The same BPE-source test scope is {b['test_total']} → {f['test_total']} lines, including embedded tests and tests.rs; reference/wrapper/Miri files are outside this local delta and unchanged. This is not a line-count gate or a total-maintenance-cost metric. `source-counts.json` records each arm separately.\n"
text+='''
## Reproduction

Apply each archived candidate patch to its own worktree at the baseline commit.
Core/deque/tree patches already include their shared changes; bitmap/seed/ranges
are baseline-only. `final.patch` is the complete retained BPE change, including
its tests/docs. `arraydeque.patch` also contains its dependency and runner lock.
Frozen source snapshots and hashes in `builds.json` identify each measured arm.
Build using the command/profile there and freeze binaries before measuring.
Large prepared counts and source corpora are identified by `input-details.json`;
paths may need adjustment. Restore an arm's archived runner lock for locked builds.

`measure.py` defaults to the seven-arm screen. Select a frozen baseline/final pair
with `BPE_MEASURE_ARMS=baseline,final`, four workers, `BPE_MEASURE_ROUNDS=5`,
`BPE_MEASURE_INPUTS` pointing at the four real inputs, and a fresh
`BPE_MEASURE_ROOT`. Use three blocks and `deque,arraydeque` for the queue comparison,
and `directed-inputs.json` for directed checks. Run serially with no builds/tests.
`summarize.py <measurement-directory>` creates paired comparisons from raw runs.
The micro harness can be built with `cargo build --offline --release --locked`
and run with modes `bitmap`, `ranges`, and `positions` on the CPUs above.
'''
(root/'README.md').write_text(text)
